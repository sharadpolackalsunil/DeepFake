"""Stage C Spatio-Temporal Video Training Orchestrator for S-MSDTv3.

Objective:
Fine-tunes foundation CLIP ViT-L/14 with high-frequency ResidualStream (SRM + Haar DWT)
and EvidenceReadoutV3 (LSE pooling + Perioral Prior) using two-phase curriculum optimization.

Phase C1 (Epochs 1-3): Freeze CLIP LayerNorms and existing head; train only ResidualStream
and EvidenceReadoutV3 to allow frequency learning without disturbing FS/DF discriminability.
Phase C2 (Epochs 4-28): Unfreeze all groups with cosine decay and warmup.
"""
import argparse
import copy
import math
import os
import sys
import yaml
import collections
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from smsdt.data.video_dataset import PairedClipDataset, EvalClipDataset, paired_collate, METHOD_ID
from smsdt.losses_v3 import detection_loss_v3
from smsdt.models.clip_encoder import CLIPFrameEncoder, CLIP_MEAN, CLIP_STD
from smsdt.models.smsdt_v3 import SMSDTv3
from smsdt.train import cosine_with_warmup

ID_TO_METHOD = {v: k for k, v in METHOD_ID.items()}


class ClipNormalize:
    """Torch normalization transform for CLIP inputs (picklable for Windows DataLoader spawn)."""

    def __init__(self, mean=CLIP_MEAN, std=CLIP_STD):
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


def get_clip_normalize():
    return ClipNormalize()


def save_checkpoint_atomic(data: dict, filepath: str):
    """Save checkpoint atomically via temporary file to prevent corruption on interrupt."""
    tmp_path = filepath + ".tmp"
    torch.save(data, tmp_path)
    os.replace(tmp_path, filepath)


def evaluate(model, loader, device, max_videos=None):
    """Evaluate model on EvalClipDataset, computing video-level AUC."""
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if max_videos is not None and i >= max_videos:
                break
            frames = batch["frames"].to(device)
            labels = batch["label"].to(device)
            B, K, T, C, H, W = frames.shape

            frames_flat = frames.view(B * K, T, C, H, W)
            logits, _, _ = model(frames_flat)
            probs = torch.sigmoid(logits).view(B, K)

            vid_score = probs.mean(dim=1)
            all_preds.extend(vid_score.float().cpu().numpy().tolist())
            all_labels.extend(labels.float().cpu().numpy().tolist())

    if len(set(all_labels)) < 2:
        return 0.5

    return float(roc_auc_score(all_labels, all_preds))


def train_v3(cfg_path: str, resume: bool = True):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    stage = cfg.get("stage", "stage_c")
    train_cfg = cfg["train"]
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    output_dir = cfg.get("output_dir", f"outputs/checkpoints_v3_{stage}")
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Starting S-MSDTv3 Training [{stage.upper()}] on {device} ===")

    # 1. Build Model
    encoder = CLIPFrameEncoder(
        name=model_cfg.get("backbone", "openai/clip-vit-large-patch14"),
        tune=model_cfg.get("tune", "layernorm"),
        grad_ckpt=model_cfg.get("grad_ckpt", True),
    )

    model = SMSDTv3(
        encoder=encoder,
        dim=model_cfg.get("dim", 384),
        depth=model_cfg.get("depth", 2),
        heads=model_cfg.get("heads", 6),
        max_frames=model_cfg.get("max_frames", 16),
        grid=model_cfg.get("grid", 16),
        n_queries=model_cfg.get("n_queries", 4),
        r0=model_cfg.get("r0", 3.0),
        topk_frac=model_cfg.get("topk_frac", 0.05),
        init_bias=model_cfg.get("init_bias", -4.0),
    )

    # 2. Warm-start from Stage B Checkpoint
    init_ckpt = cfg.get("init_ckpt")
    if init_ckpt and os.path.exists(init_ckpt):
        print(f"Loading Stage B baseline checkpoint: {init_ckpt}")
        ckpt_data = torch.load(init_ckpt, map_location="cpu")
        missing, unexpected = model.load_v2_weights(ckpt_data)
        print(f"  * Foundation & transformer weights loaded successfully.")
        print(f"  * New v3 module keys initialized: {len(missing)} keys (res.*, readout.*)")
    elif init_ckpt:
        print(f"WARNING: Checkpoint {init_ckpt} not found. Starting from scratch.")

    model.to(device)
    model_ema = copy.deepcopy(model)
    model_ema.eval()

    # 3. Two-Phase Parameter Groups
    lr_backbone_ln = float(train_cfg.get("lr_backbone_ln", 5.0e-6))
    lr_head = float(train_cfg.get("lr_head", 1.0e-4))
    lr_new = float(train_cfg.get("lr_new", 3.0e-4))
    weight_decay = float(train_cfg.get("weight_decay", 0.05))

    param_groups = model.param_groups(
        lr_backbone_ln=lr_backbone_ln,
        lr_head=lr_head,
        lr_new=lr_new,
        weight_decay=weight_decay,
    )

    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)

    # 4. Datasets & Loaders
    train_dataset = PairedClipDataset(
        index_csv=data_cfg.get("index", "data/splits/index.csv"),
        store_root=data_cfg.get("store", "data/store"),
        T=data_cfg.get("frames_per_clip", 8),
        stride=data_cfg.get("frame_stride", 1),
        p_sbi=data_cfg.get("p_sbi", 0.0),
        crop_size=data_cfg.get("crop_size", 224),
        jpeg_quality=tuple(data_cfg.get("jpeg_quality", [50, 100])),
        p_jpeg=data_cfg.get("p_jpeg", 0.3),
        p_blur=data_cfg.get("p_blur", 0.1),
        p_flip=data_cfg.get("p_flip", 0.5),
        transform=get_clip_normalize(),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.get("batch_pairs", 8),
        shuffle=True,
        num_workers=train_cfg.get("num_workers", 4),
        collate_fn=paired_collate,
        pin_memory=train_cfg.get("pin_memory", True),
        persistent_workers=train_cfg.get("persistent_workers", True),
        prefetch_factor=train_cfg.get("prefetch_factor", 2),
        drop_last=True,
    )

    # In-domain validation loader (pure FF++ 5 manipulation methods)
    val_dataset = EvalClipDataset(
        index_csv=data_cfg.get("index", "data/splits/index.csv"),
        store_root=data_cfg.get("store", "data/store"),
        split="val",
        K=data_cfg.get("val_k", 8),
        T=data_cfg.get("frames_per_clip", 8),
        stride=data_cfg.get("frame_stride", 1),
        crop_size=data_cfg.get("crop_size", 224),
        transform=get_clip_normalize(),
        methods=data_cfg.get("val_methods", ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "FaceShifter"]),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg.get("val_batch_size", 8),
        shuffle=False,
        num_workers=train_cfg.get("val_workers", 4),
        pin_memory=True,
    )

    # 5. Training Parameters & Curriculum
    epochs = train_cfg.get("epochs", 28)
    freeze_existing_epochs = train_cfg.get("freeze_existing_epochs", 3)
    steps_per_epoch = len(train_loader)
    total_steps = epochs * steps_per_epoch
    warmup_steps = int(total_steps * train_cfg.get("warmup_frac", 0.05))

    precision = train_cfg.get("precision", "bfloat16")
    use_autocast = precision in ("bfloat16", "float16") and torch.cuda.is_available()
    autocast_dtype = torch.bfloat16 if precision == "bfloat16" else torch.float16

    lam_rank = float(train_cfg.get("lam_rank", 0.5))
    lam_mask = float(train_cfg.get("lam_mask", 0.5))
    lam_freq = float(train_cfg.get("lam_freq", 0.3))
    margin = float(train_cfg.get("margin", 1.0))
    ohem_frac = float(train_cfg.get("ohem_frac", 0.5))
    pos_weight = float(train_cfg.get("pos_weight", 3.0))

    start_epoch = 0
    global_step = 0
    best_val_auc = 0.0

    print(f"Dataset: {len(train_dataset)} training pairs, {len(val_dataset)} validation videos.")
    print(f"Schedule: {epochs} epochs ({total_steps} steps). Phase C1 (freeze existing): Epochs 1-{freeze_existing_epochs}.")

    for epoch in range(start_epoch, epochs):
        model.train()

        # Phase C1 vs Phase C2 parameter freeze check
        is_phase_c1 = epoch < freeze_existing_epochs
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            is_new = name.startswith("res.") or name.startswith("readout.")
            if is_phase_c1:
                # Freeze existing foundation & head weights in Phase C1
                param.requires_grad = is_new
            else:
                # Unfreeze all trainable weights in Phase C2
                param.requires_grad = True

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/{epochs:02d} [{'Phase C1' if is_phase_c1 else 'Phase C2'}]")
        epoch_losses = collections.defaultdict(float)
        batch_count = 0

        for batch in pbar:
            global_step += 1
            frames = batch["frames"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)

            # Cosine learning rate schedule with warmup
            lr_scale = cosine_with_warmup(global_step, total_steps, warmup_steps, min_lr_ratio=train_cfg.get("min_lr_ratio", 0.01))
            for g in optimizer.param_groups:
                g["lr"] = g.get("initial_lr", g["lr"]) * lr_scale

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=use_autocast):
                logits, patch_logits, aux_logits = model(frames)
                loss, loss_dict = detection_loss_v3(
                    logit=logits,
                    label=labels,
                    patch_logits=patch_logits,
                    soft_mask=masks,
                    aux_logit=aux_logits,
                    lam_rank=lam_rank,
                    lam_mask=lam_mask,
                    lam_freq=lam_freq,
                    margin=margin,
                    ohem_frac=ohem_frac,
                    pos_weight=pos_weight,
                )

            loss.backward()

            if train_cfg.get("grad_clip", 0.0) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["grad_clip"])

            optimizer.step()

            # Dynamic EMA decay warmup schedule
            ema_decay = min(0.999, (1.0 + global_step) / (10.0 + global_step))
            with torch.no_grad():
                for p_ema, p in zip(model_ema.parameters(), model.parameters()):
                    p_ema.lerp_(p, 1.0 - ema_decay)

            batch_count += 1
            for k, v in loss_dict.items():
                epoch_losses[k] += v.item()

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "cls": f"{loss_dict['cls'].item():.3f}",
                "rank": f"{loss_dict['rank'].item():.3f}",
                "mask": f"{loss_dict['mask'].item():.3f}",
            })

        # Save last checkpoint atomically every epoch
        save_checkpoint_atomic({
            "epoch": epoch,
            "global_step": global_step,
            "model": model.state_dict(),
            "model_ema": model_ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "cfg": cfg,
        }, os.path.join(output_dir, "last.pt"))

        # Validation evaluation
        val_start_epoch = train_cfg.get("val_start_epoch", 3)
        if (epoch + 1) >= val_start_epoch:
            val_auc = evaluate(model_ema, val_loader, device)
            print(f"Epoch {epoch+1:02d} Validation AUC (EMA): {val_auc:.4f}")

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                save_checkpoint_atomic({
                    "epoch": epoch,
                    "global_step": global_step,
                    "model": model.state_dict(),
                    "model_ema": model_ema.state_dict(),
                    "val_auc": best_val_auc,
                    "cfg": cfg,
                }, os.path.join(output_dir, "best_ema.pt"))
                print(f"  * Saved new best checkpoint with Val AUC: {best_val_auc:.4f}")

    print(f"=== S-MSDTv3 Stage C Training Complete. Best Val AUC: {best_val_auc:.4f} ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_v3_stage_c.yaml")
    args = parser.parse_args()
    train_v3(args.config)
