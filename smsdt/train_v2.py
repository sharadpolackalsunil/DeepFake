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

from smsdt.data.sbi_video import sbi_clip
from smsdt.data.video_dataset import PairedClipDataset, EvalClipDataset, paired_collate
from smsdt.losses import detection_loss
from smsdt.models.clip_encoder import CLIPFrameEncoder, CLIP_MEAN, CLIP_STD
from smsdt.models.smsdt_v2 import SMSDTv2
from smsdt.train import cosine_with_warmup


def get_clip_normalize():
    """Torch normalization transform for CLIP inputs."""
    mean = torch.tensor(CLIP_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(CLIP_STD).view(1, 3, 1, 1)

    def _norm(x):
        # x is (T, 3, H, W) in [0, 1]
        return (x - mean) / std

    return _norm


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
            all_preds.extend(vid_score.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    if len(set(all_labels)) < 2:
        return 0.5

    return float(roc_auc_score(all_labels, all_preds))


def train_v2(cfg_path: str):
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
        K=4,
        T=frames_per_clip,
        stride=data_cfg.get("frame_stride", 1),
        normalize=norm_fn,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg.get("batch_pairs", 8),
        shuffle=True,
        collate_fn=paired_collate,
        num_workers=train_cfg.get("num_workers", 0 if sys.platform == "win32" and not torch.cuda.is_available() else 4),
        pin_memory=train_cfg.get("pin_memory", torch.cuda.is_available()),
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=4,
        shuffle=False,
        num_workers=0,
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
    precision = train_cfg.get("precision", "bfloat16")
    amp_dtype = torch.bfloat16 if precision == "bfloat16" and torch.cuda.is_available() else torch.float32

    lambda_mask = float(train_cfg.get("lambda_mask", 0.5 if stage == "stage_b" else 0.0))
    label_smoothing = float(train_cfg.get("label_smoothing", 0.05))

    # 4. Training Loop
    global_step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total_cls, total_mask = 0.0, 0.0, 0.0
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader):
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

            total_loss += float(loss) * grad_accum
            total_cls += logs.get("cls", 0.0)
            total_mask += logs.get("mask", 0.0)

        avg_loss = total_loss / len(train_loader)
        avg_cls = total_cls / len(train_loader)
        avg_mask = total_mask / len(train_loader)

        # Periodic Validation
        val_auc = evaluate(model_ema, val_loader, device)
        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {avg_loss:.4f} (cls: {avg_cls:.4f}, mask: {avg_mask:.4f}) | Val AUC: {val_auc:.4f}")

        # Save checkpoints
        ckpt_data = {
            "epoch": epoch,
            "val_auc": val_auc,
            "model": model.state_dict(),
            "model_ema": model_ema.state_dict(),
            "cfg": cfg,
        }
        torch.save(ckpt_data, os.path.join(output_dir, "last.pt"))

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(ckpt_data, os.path.join(output_dir, "best_ema.pt"))
            print(f"  * New best Val AUC: {best_val_auc:.4f} saved to {output_dir}/best_ema.pt")

    print(f"=== Completed S-MSDTv2 [{stage.upper()}] Training! Best Val AUC: {best_val_auc:.4f} ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to Stage A or Stage B yaml config")
    args = parser.parse_args()
    train_v2(args.config)
