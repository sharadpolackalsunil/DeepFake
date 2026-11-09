"""GenD baseline training orchestrator on our standardized FF++ c23 split.

Trains GenDModel (CLIP ViT-L/14 with LayerNorm-only tuning + Linear Probe)
using BCE + Label-Aware Alignment + Uniformity loss + Slerp latent interpolation.
"""
import argparse
import copy
import os
import sys
import yaml
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional

from smsdt.data.video_dataset import PairedClipDataset, EvalClipDataset, paired_collate, Augment
from smsdt.eval import evaluate as evaluate_metrics
from smsdt.models.gend import GenDModel, slerp
from smsdt.train import cosine_with_warmup
from smsdt.train_v2 import save_checkpoint_atomic, get_clip_normalize


def train_gend(cfg_path: str, resume: bool = True, seed: Optional[int] = None, output_dir_override: Optional[str] = None):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    train_cfg = cfg["train"]
    data_cfg = cfg["data"]
    model_cfg = cfg.get("model", {})
    
    # Random seed for reproducibility
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    base_output_dir = cfg.get("output_dir", "outputs/checkpoints_gend_ours")
    if output_dir_override:
        output_dir = output_dir_override
    elif seed is not None:
        output_dir = os.path.join(base_output_dir, f"seed{seed}")
    else:
        output_dir = base_output_dir
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_str = f" [Seed {seed}]" if seed is not None else ""
    print(f"=== Starting GenD Baseline Training{seed_str} on {device} ===")
    print(f"Checkpoints will be saved to: {output_dir}")

    # 1. Build Model
    model = GenDModel(
        name=model_cfg.get("backbone", "openai/clip-vit-large-patch14"),
        tune=model_cfg.get("tune", "layernorm"),
        normalize_inputs=model_cfg.get("normalize_inputs", True),
    ).to(device)

    # Param groups: separate LayerNorm params and linear probe head
    lr_ln = float(train_cfg.get("lr_backbone_ln", 1.0e-4))
    lr_head = float(train_cfg.get("lr_head", 5.0e-4))
    wd = float(train_cfg.get("weight_decay", 0.05))

    ln_params = [p for n, p in model.encoder.named_parameters() if p.requires_grad]
    head_params = [p for p in model.head.parameters() if p.requires_grad]

    param_groups = [
        {"params": ln_params, "lr": lr_ln, "weight_decay": 0.0},
        {"params": head_params, "lr": lr_head, "weight_decay": wd},
    ]

    epochs = int(train_cfg.get("epochs", 30))
    batch_pairs = int(train_cfg.get("batch_pairs", 32))
    grad_accum = int(train_cfg.get("grad_accum", 1))

    # 2. Dataloaders
    norm_fn = get_clip_normalize()
    aug = Augment(
        size=int(data_cfg.get("crop_size", 224)),
        jpeg=tuple(data_cfg.get("jpeg_quality", [50, 100])),
        p_jpeg=float(data_cfg.get("p_jpeg", 0.3)),
        p_blur=float(data_cfg.get("p_blur", 0.1)),
        p_flip=float(data_cfg.get("p_flip", 0.5)),
    )
    train_ds = PairedClipDataset(
        index_csv=data_cfg.get("index", "data/splits/index.csv"),
        store_root=data_cfg.get("store", "data/store"),
        T=1,
        stride=1,
        p_sbi=0.0,
        augment=aug,
        normalize=norm_fn,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_pairs,
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 4)),
        pin_memory=train_cfg.get("pin_memory", True) and torch.cuda.is_available(),
        persistent_workers=train_cfg.get("persistent_workers", True),
        prefetch_factor=int(train_cfg.get("prefetch_factor", 2)),
        collate_fn=paired_collate,
        drop_last=True,
    )

    val_ds = EvalClipDataset(
        index_csv=data_cfg.get("index", "data/splits/index.csv"),
        store_root=data_cfg.get("store", "data/store"),
        split="val",
        K=int(data_cfg.get("val_k", 8)),
        T=1,
        stride=1,
        normalize=norm_fn,
        methods=data_cfg.get("val_methods", ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "FaceShifter"]),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=int(train_cfg.get("val_batch_size", 16)),
        shuffle=False,
        num_workers=int(train_cfg.get("val_workers", 0)),
        pin_memory=False,
    )

    total_steps = (len(train_loader) // grad_accum) * epochs

    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)
    scheduler = cosine_with_warmup(
        optimizer,
        total_steps=total_steps,
        warmup_frac=float(train_cfg.get("warmup_frac", 0.03)),
        min_ratio=float(train_cfg.get("min_lr_ratio", 0.01)),
    )

    ema_decay = train_cfg.get("ema_decay", 0.999)
    model_ema = copy.deepcopy(model)
    for p in model_ema.parameters():
        p.requires_grad = False
    model_ema.eval()

    val_start_epoch = int(train_cfg.get("val_start_epoch", 5))
    best_val_auc = 0.0
    start_epoch = 1

    last_ckpt_path = os.path.join(output_dir, "last.pt")
    best_ema_path = os.path.join(output_dir, "best_ema.pt")

    if resume and os.path.exists(last_ckpt_path):
        try:
            ckpt = torch.load(last_ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model"])
            model_ema.load_state_dict(ckpt["model_ema"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            start_epoch = ckpt.get("epoch", 0) + 1
            best_val_auc = ckpt.get("val_auc", 0.0)
            print(f"Resumed from {last_ckpt_path} at Epoch {start_epoch:02d} (Best Val AUC: {best_val_auc:.4f})")
        except Exception as e:
            print(f"Failed to resume from {last_ckpt_path}: {e}")

    precision = train_cfg.get("precision", "bfloat16")
    amp_dtype = torch.bfloat16 if precision == "bfloat16" and torch.cuda.is_available() else torch.float32

    lambda_align = float(model_cfg.get("lambda_align", 0.5))
    lambda_unif = float(model_cfg.get("lambda_unif", 0.5))
    slerp_prob = float(model_cfg.get("slerp_prob", 0.5))

    global_step = 0

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        train_ds.set_epoch(epoch)
        total_loss, total_bce, total_align, total_unif = 0.0, 0.0, 0.0, 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{epochs:02d} [GenD]", total=len(train_loader))

        for step, batch in enumerate(pbar):
            frames = batch["frames"].to(device)
            if frames.ndim == 5 and frames.shape[1] == 1:
                frames = frames.squeeze(1)  # (2B, 3, 224, 224)
            labels = batch["label"].to(device)             # (2B,)

            # Slerp latent interpolation between paired items
            do_slerp = (np.random.rand() < slerp_prob) and (frames.size(0) % 2 == 0)
            slerp_t = float(np.random.beta(0.5, 0.5)) if do_slerp else None
            pair_frames = None
            target_labels = labels.float()
            if do_slerp:
                idx = torch.arange(frames.size(0), device=device)
                paired_idx = idx ^ 1
                pair_frames = frames[paired_idx]
                target_labels = (1.0 - slerp_t) * labels.float() + slerp_t * labels[paired_idx].float()

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=torch.cuda.is_available()):
                logits, embeddings = model(frames, slerp_t=slerp_t, pair_x=pair_frames)
                loss_dict = model.compute_loss(
                    logits=logits,
                    labels=target_labels,
                    embeddings=embeddings,
                    lambda_align=lambda_align,
                    lambda_unif=lambda_unif,
                )
                loss = loss_dict["loss"] / grad_accum

            loss.backward()

            if (step + 1) % grad_accum == 0 or (step + 1) == len(train_loader):
                if train_cfg.get("grad_clip", 1.0) > 0:
                    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], train_cfg.get("grad_clip", 1.0))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

                # Step EMA
                step_ema = min(float(ema_decay), (1.0 + global_step) / (10.0 + global_step))
                with torch.no_grad():
                    for p_ema, p_model in zip(model_ema.parameters(), model.parameters()):
                        p_ema.data.mul_(step_ema).add_(p_model.data, alpha=1.0 - step_ema)
                global_step += 1

            total_loss += float(loss.detach()) * grad_accum
            total_bce += float(loss_dict["loss_bce"].detach())
            total_align += float(loss_dict["loss_align"].detach())
            total_unif += float(loss_dict["loss_unif"].detach())
            pbar.set_postfix({"loss": f"{total_loss/(step+1):.4f}", "bce": f"{total_bce/(step+1):.4f}"})

        # Validation
        if epoch >= val_start_epoch:
            metrics = evaluate_metrics(model_ema, val_loader, device=device, top_k_ratio=1.0)
            val_auc = float(metrics.get("auc", 0.0))
            if "video_level" in metrics:
                val_auc = float(metrics["video_level"].get("auc", val_auc))

            print(f"Epoch {epoch:02d}/{epochs:02d} | Val AUC: {val_auc:.4f} (Best: {best_val_auc:.4f})")
            
            ckpt_data = {
                "epoch": epoch,
                "val_auc": val_auc,
                "model": model.state_dict(),
                "model_ema": model_ema.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "cfg": cfg,
            }
            save_checkpoint_atomic(ckpt_data, os.path.join(output_dir, "last.pt"))

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                save_checkpoint_atomic(ckpt_data, os.path.join(output_dir, "best_ema.pt"))
                print(f"  * Saved new best GenD checkpoint with Val AUC: {best_val_auc:.4f}")
        else:
            print(f"Epoch {epoch:02d}/{epochs:02d} [Validation skipped until Epoch {val_start_epoch:02d}]")
            ckpt_data = {
                "epoch": epoch,
                "val_auc": 0.0,
                "model": model.state_dict(),
                "model_ema": model_ema.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "cfg": cfg,
            }
            save_checkpoint_atomic(ckpt_data, os.path.join(output_dir, "last.pt"))

    print(f"=== GenD Training Complete! Best Val AUC: {best_val_auc:.4f} ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_gend.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    train_gend(args.config, resume=not args.no_resume, seed=args.seed, output_dir_override=args.output_dir)

