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

from smsdt.data.sbi_video import sbi_clip
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


from smsdt.eval import evaluate


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
        try:
            ckpt_data = torch.load(init_ckpt, map_location="cpu", weights_only=False)
        except TypeError:
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

    # Audit trainable parameters
    trainable_names = {n for n, p in model.named_parameters() if p.requires_grad}
    trainable_params = sum(p.numel() for n, p in model.named_parameters() if n in trainable_names)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {trainable_params:,} trainable / {total_params:,} total ({100 * trainable_params / total_params:.2f}%)")

    # 4. Datasets & Loaders
    norm_fn = get_clip_normalize()
    frames_per_clip = data_cfg.get("frames_per_clip", 8)

    train_dataset = PairedClipDataset(
        index_csv=data_cfg.get("index", "data/splits/index.csv"),
        store_root=data_cfg.get("store", "data/store"),
        T=frames_per_clip,
        stride=data_cfg.get("frame_stride", 1),
        p_sbi=data_cfg.get("p_sbi", 0.0),
        sbi_fn=sbi_clip,
        normalize=norm_fn,
        mask_grid=model_cfg.get("grid", 16),
    )

    workers = int(train_cfg.get("num_workers", 8))
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.get("batch_pairs", 8),
        shuffle=True,
        num_workers=workers,
        collate_fn=paired_collate,
        pin_memory=train_cfg.get("pin_memory", True),
        persistent_workers=train_cfg.get("persistent_workers", True) if workers > 0 else False,
        prefetch_factor=train_cfg.get("prefetch_factor", 2) if workers > 0 else None,
        drop_last=True,
    )

    # In-domain validation loader (pure FF++ 5 manipulation methods)
    val_methods = data_cfg.get("val_methods", ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "FaceShifter"])
    val_dataset = EvalClipDataset(
        index_csv=data_cfg.get("index", "data/splits/index.csv"),
        store_root=data_cfg.get("store", "data/store"),
        split="val",
        K=int(data_cfg.get("val_k", 8)),
        T=frames_per_clip,
        stride=data_cfg.get("frame_stride", 1),
        size=int(data_cfg.get("crop_size", 224)),
        normalize=norm_fn,
        methods=val_methods,
    )

    val_workers = int(train_cfg.get("val_workers", 8))
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg.get("val_batch_size", 8),
        shuffle=False,
        num_workers=val_workers,
        pin_memory=False,
        persistent_workers=bool(train_cfg.get("persistent_workers", False)) if val_workers > 0 else False,
        prefetch_factor=int(train_cfg.get("prefetch_factor", 2)) if val_workers > 0 else None,
    )

    # 5. Training Parameters & Curriculum
    epochs = train_cfg.get("epochs", 28)
    freeze_existing_epochs = train_cfg.get("freeze_existing_epochs", 3)
    steps_per_epoch = len(train_loader)
    total_steps = epochs * steps_per_epoch

    scheduler = cosine_with_warmup(
        optimizer,
        total_steps=total_steps,
        warmup_frac=train_cfg.get("warmup_frac", 0.05),
        min_ratio=train_cfg.get("min_lr_ratio", 0.01),
    )

    precision = train_cfg.get("precision", "bfloat16")
    use_autocast = precision in ("bfloat16", "float16") and torch.cuda.is_available()
    autocast_dtype = torch.bfloat16 if precision == "bfloat16" else torch.float16

    lam_rank = float(train_cfg.get("lam_rank", 0.5))
    lam_mask = float(train_cfg.get("lam_mask", 0.5))
    lam_freq = float(train_cfg.get("lam_freq", 0.3))
    margin = float(train_cfg.get("margin", 1.0))
    ohem_frac = float(train_cfg.get("ohem_frac", 0.5))
    pos_weight = float(train_cfg.get("pos_weight", 3.0))
    base_ema_decay = float(train_cfg.get("ema_decay", 0.999))

    start_epoch = 0
    global_step = 0
    best_val_auc = 0.0

    last_ckpt_path = os.path.join(output_dir, "last.pt")
    best_ema_path = os.path.join(output_dir, "best_ema.pt")
    if resume:
        ckpt_to_load = None
        if os.path.exists(last_ckpt_path):
            try:
                ckpt_to_load = torch.load(last_ckpt_path, map_location=device, weights_only=False)
                print(f"Resuming Stage C training from: {last_ckpt_path}")
            except Exception as e:
                print(f"Warning: Failed to load {last_ckpt_path} ({e}). Falling back to best_ema.pt...")
        if ckpt_to_load is None and os.path.exists(best_ema_path):
            try:
                ckpt_to_load = torch.load(best_ema_path, map_location=device, weights_only=False)
                print(f"Resuming Stage C training from: {best_ema_path}")
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
            global_step = ckpt_to_load.get("global_step", start_epoch * steps_per_epoch)
            best_val_auc = ckpt_to_load.get("macro_auc", ckpt_to_load.get("val_auc", 0.0))
            print(f"  * Resumed at Epoch {start_epoch+1:02d}/{epochs:02d} (Global Step: {global_step}) | Best Macro Val AUC: {best_val_auc:.4f}")

    print(f"Dataset: {len(train_dataset)} training pairs, {len(val_dataset)} validation videos.")
    print(f"Schedule: {epochs} epochs ({total_steps} steps). Phase C1 (freeze existing): Epochs 1-{freeze_existing_epochs}.")

    for epoch in range(start_epoch, epochs):
        model.train()

        # Phase C1 vs Phase C2 parameter freeze check
        is_phase_c1 = epoch < freeze_existing_epochs
        for name, param in model.named_parameters():
            if name not in trainable_names:
                param.requires_grad = False
            else:
                is_new = name.startswith("res.") or name.startswith("readout.")
                param.requires_grad = (is_new if is_phase_c1 else True)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/{epochs:02d} [{'Phase C1' if is_phase_c1 else 'Phase C2'}]")
        epoch_losses = collections.defaultdict(float)
        method_counter = collections.Counter()

        for batch in pbar:
            global_step += 1
            frames = batch["frames"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            priors = batch["prior"].to(device, non_blocking=True) if "prior" in batch else None

            if "method" in batch:
                fake_mids = batch["method"][1::2].cpu().tolist()
                for mid in fake_mids:
                    method_counter[ID_TO_METHOD.get(mid, str(mid))] += 1

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=use_autocast):
                logits, patch_logits, aux_logits = model(frames, prior=priors)
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
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    train_cfg["grad_clip"]
                )

            optimizer.step()
            scheduler.step()

            # Dynamic EMA decay warmup schedule per Strategic Plan
            step_ema_decay = min(base_ema_decay, (1.0 + global_step) / (10.0 + global_step))
            with torch.no_grad():
                for p_ema, p_model in zip(model_ema.parameters(), model.parameters()):
                    p_ema.data.mul_(step_ema_decay).add_(p_model.data, alpha=1.0 - step_ema_decay)

            for k, v in loss_dict.items():
                epoch_losses[k] += v.item()

            cur_lr = scheduler.get_last_lr()[-1] if hasattr(scheduler, "get_last_lr") else 0.0
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "cls": f"{loss_dict['cls'].item():.3f}",
                "rank": f"{loss_dict['rank'].item():.3f}",
                "mask": f"{loss_dict['mask'].item():.3f}",
                "lr": f"{cur_lr:.2e}",
            })

        print(f"  * Methods trained in Epoch {epoch+1:02d}: {dict(method_counter)}")

        # Telemetry logging per §3.4
        with torch.no_grad():
            prior_gain_val = float(model.readout.prior_gain.item()) if hasattr(model.readout, "prior_gain") else 0.0
            r_val = float(model.readout.log_r.exp().item()) if hasattr(model.readout, "log_r") else 3.0
            gate_bias_val = float(model.res.gate.bias.mean().item()) if hasattr(model.res, "gate") else 0.0
            w_mix_val = float(torch.sigmoid(model.readout.mix).item()) if hasattr(model.readout, "mix") else 0.5
            print(f"  * V3 Telemetry: gate_bias={gate_bias_val:.2f}, prior_gain={prior_gain_val:.4f}, LSE r={r_val:.2f}, readout_mix={w_mix_val:.3f}")

        # Save last checkpoint atomically every epoch
        save_checkpoint_atomic({
            "epoch": epoch,
            "global_step": global_step,
            "model": model.state_dict(),
            "model_ema": model_ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "cfg": cfg,
        }, os.path.join(output_dir, "last.pt"))

        # Validation evaluation
        val_start_epoch = train_cfg.get("val_start_epoch", 3)
        if (epoch + 1) >= val_start_epoch:
            print(f"\n--- Epoch {epoch+1:02d}/{epochs:02d} In-Domain FF++ Validation ---")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            metrics = evaluate(model_ema, val_loader, device)
            v_metrics = metrics.get("video_level", metrics)
            val_auc = float(v_metrics["auc"])
            val_eer = float(v_metrics.get("eer", 0.0))
            val_f1 = float(v_metrics.get("f1_05", 0.0))

            print(f"Epoch {epoch+1:02d} | Val AUC (EMA): {val_auc:.4f} | EER: {val_eer:.4f} | F1@0.5: {val_f1:.4f}")

            # Per-manipulation breakdown & FF++ Macro AUC (DF, F2F, FS, NT, FSh)
            ffpp_methods = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "FaceShifter"]
            per_m = metrics.get("per_manipulation", {})
            if per_m:
                breakdown_str = " | ".join(f"{m}: {per_m[m]['auc']:.4f}" for m in ffpp_methods if m in per_m)
                print(f"  * Per-Method AUC: {breakdown_str}")

            ffpp_aucs = [per_m[m]["auc"] for m in ffpp_methods if m in per_m and "auc" in per_m[m]]
            macro_auc = float(np.mean(ffpp_aucs)) if ffpp_aucs else val_auc
            fs_auc = per_m.get("FaceSwap", {}).get("auc", 1.0)
            df_auc = per_m.get("Deepfakes", {}).get("auc", 1.0)
            floors_passed = (fs_auc >= 0.985) and (df_auc >= 0.980)
            print(f"  * FF++ Macro AUC: {macro_auc:.4f} (Floors: FS={fs_auc:.4f} [>=0.985], DF={df_auc:.4f} [>=0.980])")

            # Checkpoint selection strictly by Macro AUC subject to regression floors (§3.5)
            if macro_auc > best_val_auc and floors_passed:
                best_val_auc = macro_auc
                save_checkpoint_atomic({
                    "epoch": epoch,
                    "global_step": global_step,
                    "model": model.state_dict(),
                    "model_ema": model_ema.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "val_auc": val_auc,
                    "macro_auc": macro_auc,
                    "metrics": metrics,
                    "cfg": cfg,
                }, os.path.join(output_dir, "best_ema.pt"))
                print(f"  * Saved new best checkpoint with FF++ Macro Val AUC: {best_val_auc:.4f}")
        else:
            print(f"Epoch {epoch+1:02d}/{epochs:02d} [Validation skipped until Epoch {val_start_epoch:02d}]")

    print(f"=== S-MSDTv3 Stage C Training Complete. Best Macro Val AUC: {best_val_auc:.4f} ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_v3_stage_c.yaml")
    parser.add_argument("--no-resume", action="store_true", help="Start from epoch 0 ignoring existing last.pt")
    args = parser.parse_args()
    train_v3(args.config, resume=not args.no_resume)

