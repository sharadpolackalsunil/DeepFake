"""Two-stage training orchestrator for S-MSDTv2 with CLIP ViT-L/14 PEFT.

Stage A: Frame-level pre-tuning of CLIP LayerNorms and CLS linear head (T=1, B=32 pairs).
Stage B: Spatio-temporal video training of NMSSSBlocks + EvidenceReadout (T=8, B=8 pairs).
"""
import argparse
import copy
import math
import os
import sys
import yaml
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torchvision import transforms

from tqdm import tqdm

from smsdt.data.sbi_video import sbi_clip
from smsdt.data.video_dataset import PairedClipDataset, EvalClipDataset, paired_collate
from smsdt.eval import evaluate as evaluate_metrics
from smsdt.losses import detection_loss
from smsdt.models.clip_encoder import CLIPFrameEncoder, CLIP_MEAN, CLIP_STD
from smsdt.models.smsdt_v2 import SMSDTv2
from smsdt.train import cosine_with_warmup


class ClipNormalize:
    """Torch normalization transform for CLIP inputs (picklable for Windows DataLoader spawn)."""

    def __init__(self, mean=CLIP_MEAN, std=CLIP_STD):
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x is (T, 3, H, W) in [0, 1]
        return (x - self.mean) / self.std


def get_clip_normalize():
    """Return picklable CLIP normalization transform."""
    return ClipNormalize()


def save_checkpoint_atomic(data: dict, filepath: str):
    """Save checkpoint atomically via temporary file to prevent corruption on interrupt."""
    tmp_path = filepath + ".tmp"
    torch.save(data, tmp_path)
    os.replace(tmp_path, filepath)


def evaluate(model, loader, device, max_videos=None):
    """Evaluate model on EvalClipDataset, computing video-level and clip-level AUC."""
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if max_videos is not None and i >= max_videos:
                break
            # frames: (B, K, T, 3, H, W)
            frames = batch["frames"].to(device)
            labels = batch["label"].to(device)
            B, K, T, C, H, W = frames.shape

            # Flatten K clips per video into batch
            frames_flat = frames.view(B * K, T, C, H, W)
            logits, _ = model(frames_flat)
            probs = torch.sigmoid(logits).view(B, K)

            # Mean clip probability as primary video score
            vid_score = probs.mean(dim=1)
            all_preds.extend(vid_score.float().cpu().numpy().tolist())
            all_labels.extend(labels.float().cpu().numpy().tolist())

    if len(set(all_labels)) < 2:
        return 0.5

    return float(roc_auc_score(all_labels, all_preds))


def train_v2(cfg_path: str, resume: bool = True):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    stage = cfg.get("stage", "stage_b")
    train_cfg = cfg["train"]
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    output_dir = cfg.get("output_dir", f"outputs/checkpoints_v2_{stage}")
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Starting S-MSDTv2 Training [{stage.upper()}] on {device} ===")

    # 1. Build Model
    encoder = CLIPFrameEncoder(
        name=model_cfg.get("backbone", "openai/clip-vit-large-patch14"),
        tune=model_cfg.get("tune", "layernorm"),
        grad_ckpt=model_cfg.get("grad_ckpt", True),
    )

    model = SMSDTv2(
        encoder=encoder,
        dim=model_cfg.get("dim", 384),
        depth=model_cfg.get("depth", 2),
        heads=model_cfg.get("heads", 6),
        max_frames=model_cfg.get("max_frames", 16),
        grid=model_cfg.get("grid", 16),
        topk_frac=model_cfg.get("topk_frac", 0.05),
    )

    # Optional checkpoint initialization (e.g. Stage B initializing from Stage A)
    init_ckpt = cfg.get("init_ckpt")
    if init_ckpt and os.path.exists(init_ckpt):
        print(f"Loading weights from {init_ckpt}...")
        ckpt = torch.load(init_ckpt, map_location="cpu")
        state_dict = ckpt.get("model_ema", ckpt.get("model", ckpt))
        model.load_state_dict(state_dict, strict=False)

    model.to(device)

    # Audit trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {trainable_params:,} trainable / {total_params:,} total ({100 * trainable_params / total_params:.2f}%)")

    # 2. Build Datasets
    norm_fn = get_clip_normalize()
    frames_per_clip = data_cfg.get("frames_per_clip", 1 if stage == "stage_a" else 8)

    train_ds = PairedClipDataset(
        index_csv=data_cfg.get("index", "data/splits/index.csv"),
        store_root=data_cfg.get("store", "data/store"),
        T=frames_per_clip,
        stride=data_cfg.get("frame_stride", 1),
        p_sbi=data_cfg.get("p_sbi", 0.5),
        sbi_fn=sbi_clip,
        normalize=norm_fn,
        mask_grid=model_cfg.get("grid", 16),
    )

    val_ds = EvalClipDataset(
        index_csv=data_cfg.get("index", "data/splits/index.csv"),
        store_root=data_cfg.get("store", "data/store"),
        split="val",
        K=int(data_cfg.get("val_k", 8)),
        T=frames_per_clip,
        stride=data_cfg.get("frame_stride", 1),
        normalize=norm_fn,
    )

    workers = int(train_cfg.get("num_workers", 4))
    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg.get("batch_pairs", 8),
        shuffle=True,
        collate_fn=paired_collate,
        num_workers=workers,
        pin_memory=bool(train_cfg.get("pin_memory", False)),
        persistent_workers=bool(train_cfg.get("persistent_workers", False)) if workers > 0 else False,
        prefetch_factor=int(train_cfg.get("prefetch_factor", 2)) if workers > 0 else None,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=int(train_cfg.get("val_batch_size", 4)),
        shuffle=False,
        num_workers=int(train_cfg.get("val_workers", 0)),
        pin_memory=False,
    )

    # 3. Optimizer & Schedulers
    param_groups = model.param_groups(
        lr_backbone_ln=float(train_cfg.get("lr_backbone_ln", 1e-4)),
        lr_head=float(train_cfg.get("lr_head", 5e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.05)),
    )
    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999))

    epochs = train_cfg.get("epochs", 30)
    grad_accum = train_cfg.get("grad_accum", 1)
    steps_per_epoch = max(1, math.ceil(len(train_loader) / grad_accum))
    total_steps = epochs * steps_per_epoch

    scheduler = cosine_with_warmup(
        optimizer,
        total_steps=total_steps,
        warmup_frac=train_cfg.get("warmup_frac", 0.03),
        min_ratio=train_cfg.get("min_lr_ratio", 0.01),
    )

    # Model EMA
    ema_decay = train_cfg.get("ema_decay", 0.999)
    model_ema = copy.deepcopy(model)
    for p in model_ema.parameters():
        p.requires_grad = False

    best_val_auc = 0.0
    start_epoch = 1

    last_ckpt_path = os.path.join(output_dir, "last.pt")
    best_ema_path = os.path.join(output_dir, "best_ema.pt")
    if resume:
        ckpt_to_load = None
        if os.path.exists(last_ckpt_path):
            try:
                ckpt_to_load = torch.load(last_ckpt_path, map_location=device)
                print(f"Resuming training from: {last_ckpt_path}")
            except Exception as e:
                print(f"Warning: Failed to load {last_ckpt_path} ({e}). Falling back to best_ema.pt...")
        if ckpt_to_load is None and os.path.exists(best_ema_path):
            try:
                ckpt_to_load = torch.load(best_ema_path, map_location=device)
                print(f"Resuming training from: {best_ema_path}")
            except Exception as e:
                print(f"Warning: Failed to load {best_ema_path} ({e}).")

        if ckpt_to_load is not None:
            model.load_state_dict(ckpt_to_load["model"])
            if "model_ema" in ckpt_to_load:
                model_ema.load_state_dict(ckpt_to_load["model_ema"])
            if "optimizer" in ckpt_to_load:
                try:
                    optimizer.load_state_dict(ckpt_to_load["optimizer"])
                except Exception:
                    pass
            if "scheduler" in ckpt_to_load:
                try:
                    scheduler.load_state_dict(ckpt_to_load["scheduler"])
                except Exception:
                    pass
            start_epoch = ckpt_to_load.get("epoch", 0) + 1
            best_val_auc = ckpt_to_load.get("val_auc", 0.0)
            print(f"  * Resumed at Epoch {start_epoch:02d}/{epochs:02d} | Best Val AUC: {best_val_auc:.4f}")

    precision = train_cfg.get("precision", "bfloat16")
    amp_dtype = torch.bfloat16 if precision == "bfloat16" and torch.cuda.is_available() else torch.float32

    lambda_mask = float(train_cfg.get("lambda_mask", 0.5 if stage == "stage_b" else 0.0))
    label_smoothing = float(train_cfg.get("label_smoothing", 0.05))

    # 4. Training Loop
    global_step = 0
    methods = getattr(train_loader.dataset, "methods", ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "FaceShifter"])

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        if hasattr(train_loader.dataset, "set_epoch"):
            train_loader.dataset.set_epoch(epoch - 1)
        total_loss, total_cls, total_mask = 0.0, 0.0, 0.0
        train_preds, train_targets = [], []
        optimizer.zero_grad(set_to_none=True)

        active_method = methods[(epoch - 1) % len(methods)] if methods else "Mixed"
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{epochs:02d} [{active_method}]", total=len(train_loader))

        for step, batch in enumerate(pbar):
            frames = batch["frames"].to(device)  # (2B, T, 3, 224, 224)
            labels = batch["label"].to(device)    # (2B,)
            mask = batch["mask"].to(device)      # (2B, T, g, g)
            has_mask = batch["has_mask"].to(device)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=torch.cuda.is_available()):
                logits, patch_logits = model(frames)
                loss, logs = detection_loss(
                    logit=logits,
                    label=labels,
                    patch_logits=patch_logits if lambda_mask > 0 else None,
                    mask=mask if lambda_mask > 0 else None,
                    has_mask=has_mask if lambda_mask > 0 else None,
                    smoothing=label_smoothing,
                    lam_mask=lambda_mask,
                )
                loss = loss / grad_accum

            loss.backward()

            if (step + 1) % grad_accum == 0 or (step + 1) == len(train_loader):
                if train_cfg.get("grad_clip", 1.0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        train_cfg.get("grad_clip", 1.0),
                    )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

                # Update EMA weights
                with torch.no_grad():
                    for p_ema, p_model in zip(model_ema.parameters(), model.parameters()):
                        p_ema.data.mul_(ema_decay).add_(p_model.data, alpha=1.0 - ema_decay)

                global_step += 1

            total_loss += float(loss.detach()) * grad_accum
            total_cls += logs.get("cls", 0.0)
            total_mask += logs.get("mask", 0.0)

            probs = torch.sigmoid(logits).detach().float().cpu().numpy().flatten()
            train_preds.extend(probs.tolist())
            train_targets.extend(labels.detach().float().cpu().numpy().flatten().tolist())

            cur_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else 0.0
            pbar.set_postfix({"Loss": f"{logs.get('cls', float(loss.detach())):.4f}", "LR": f"{cur_lr:.2e}"})

        avg_loss = total_loss / len(train_loader)
        avg_cls = total_cls / len(train_loader)
        avg_mask = total_mask / len(train_loader)
        train_auc = float(roc_auc_score(train_targets, train_preds)) if len(set(train_targets)) > 1 else 0.5

        # Periodic Validation
        print(f"\n--- Epoch {epoch:02d}/{epochs:02d} Validation ---")
        metrics = evaluate_metrics(model_ema, val_loader, device)
        v_metrics = metrics.get("video_level", metrics)
        val_auc = float(v_metrics["auc"])
        val_eer = float(v_metrics["eer"])
        val_f1 = float(v_metrics["f1_05"])

        print(f"Epoch {epoch:02d} | Train Loss: {avg_loss:.4f} | Train AUC: {train_auc:.4f} | Val AUC: {val_auc:.4f} | Val EER: {val_eer:.4f} | Val F1@0.5: {val_f1:.4f}")
        if metrics.get("per_manipulation"):
            breakdown_str = " | ".join(f"{m}: {d['auc']:.4f}" for m, d in metrics["per_manipulation"].items())
            print(f"  * Val AUC Breakdown: {breakdown_str}")

        # Save checkpoints
        ckpt_data = {
            "epoch": epoch,
            "val_auc": val_auc,
            "val_eer": val_eer,
            "val_f1_05": val_f1,
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
            print(f"  * New best Val AUC: {best_val_auc:.4f} saved to {output_dir}/best_ema.pt")

    print(f"=== Completed S-MSDTv2 [{stage.upper()}] Training! Best Val AUC: {best_val_auc:.4f} ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to Stage A or Stage B yaml config")
    parser.add_argument("--from-scratch", action="store_true", help="Start training from scratch without resuming")
    args = parser.parse_args()
    train_v2(args.config, resume=not args.from_scratch)
