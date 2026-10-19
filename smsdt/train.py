"""
S-MSDT Training Script
=====================
Trains the S-MSDT model on preprocessed webdataset shards.
Supports mixed-precision (BF16), gradient accumulation, gradient checkpointing,
and multi-term loss (focal BCE + contrastive + alignment).
"""
import os
import argparse
import numpy as np
import torch
import torch.nn as nn

from omegaconf import OmegaConf
from smsdt.models.smsdt import SMSDT
from smsdt.data.dataset import build_dataloader
from smsdt.losses import total_loss
from smsdt.eval import evaluate

import torch.nn.functional as F

def main(cfg, resume=True):
    # --- Device setup ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        cap = torch.cuda.get_device_capability()
        print(f"GPU compute capability: {cap}")

    # --- Model ---
    model_type = cfg.model.get("type", "smsdt")
    if model_type == "baseline_b4":
        from smsdt.models.baseline_b4 import BaselineB4
        model = BaselineB4(
            pretrained=cfg.model.get("pretrained", True),
            drop_rate=cfg.model.get("drop_rate", 0.2),
        ).to(device)
        print("Initialized stripped baseline: BaselineB4 (EfficientNet-B4 + frame mean)")
    else:
        model = SMSDT(
            embed_dim=cfg.model.embed_dim,
            depth=cfg.model.depth,
            heads=cfg.model.get("heads", 8),
            num_classes=cfg.model.get("num_classes", 1),
            grid=cfg.model.get("grid", 7),
            stem_type=cfg.model.get("stem_type", "dw_separable_se"),
            pool_energy=cfg.model.get("pool_energy", True),
        ).to(device)
        print("Initialized full architecture: S-MSDT v3")

    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model parameters: {param_count:.1f}M")

    # Optional: torch.compile for sm_121 (verify support first)
    if device == "cuda" and cfg.train.get("compile", False):
        try:
            model = torch.compile(model, mode="max-autotune")
            print("Model compiled with torch.compile (max-autotune)")
        except Exception as e:
            print(f"torch.compile failed (may need nightly build for sm_121): {e}")

    # --- Gradient checkpointing ---
    if cfg.train.get("gradient_checkpointing", False):
        model.set_gradient_checkpointing(True)
        print("Gradient checkpointing enabled for ViT blocks")

    # --- Optimizer & scheduler ---
    if model_type == "baseline_b4":
        lr = float(cfg.train.get("lr", 3e-4))
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=float(cfg.train.get("weight_decay", 0.01)),
            betas=tuple(cfg.train.get("betas", [0.9, 0.999])),
        )
        max_lr = lr
    else:
        backbone_params = list(model.rgb_branch.parameters())
        backbone_param_ids = set(id(p) for p in backbone_params)
        head_params = [p for p in model.parameters() if id(p) not in backbone_param_ids]
        backbone_lr = float(cfg.train.get("backbone_lr", 3e-5))
        head_lr = float(cfg.train.get("lr", 3e-4))

        param_groups = [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": head_params, "lr": head_lr},
        ]
        opt = torch.optim.AdamW(
            param_groups,
            weight_decay=float(cfg.train.get("weight_decay", 0.05)),
            betas=tuple(cfg.train.get("betas", [0.9, 0.999])),
        )
        max_lr = [backbone_lr, head_lr]

    steps_per_epoch = cfg.train.get("steps_per_epoch", 1000)
    total_steps = cfg.train.get("total_steps") or (cfg.train.epochs * steps_per_epoch)
    warmup_steps = cfg.train.get("warmup_steps", int(total_steps * 0.05))
    pct_start = min(0.3, max(0.01, warmup_steps / max(1, total_steps)))

    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=max_lr,
        total_steps=total_steps,
        pct_start=pct_start,
    )

    # --- Data loaders ---
    train_loader = build_dataloader(
        cfg.data, split="train",
        num_workers=cfg.train.get("workers", 10),
    )
    val_loader = build_dataloader(
        cfg.data, split="val",
        num_workers=max(1, cfg.train.get("workers", 10) // 2),
    )

    # --- Logging ---
    use_wandb = cfg.train.get("use_wandb", False)
    if use_wandb:
        import wandb
        wandb.init(project="smsdt-dgx-spark", config=OmegaConf.to_container(cfg))
    else:
        print("WandB disabled — logging to stdout only")

    # --- Training state ---
    accum_steps = cfg.train.get("grad_accum", 8)
    grad_clip = cfg.train.get("grad_clip", 1.0)
    use_aux_losses = cfg.train.get("lambda_contrastive", 0.0) > 0 or cfg.train.get("lambda_align", 0.0) > 0

    loss_kwargs = {
        "loss_type": cfg.train.get("loss_type", "bce"),
        "pos_weight": float(cfg.train.get("pos_weight", 0.25)),
        "alpha": float(cfg.train.get("alpha", 0.2)),
        "gamma": float(cfg.train.get("gamma", 2.0)),
        "label_smoothing": float(cfg.train.get("label_smoothing", 0.0)),
        "lambda_contrastive": float(cfg.train.get("lambda_contrastive", 0.0)),
        "lambda_align": float(cfg.train.get("lambda_align", 0.0)),
        "align_mode": cfg.train.get("align_mode", "decorrelate"),
    }

    os.makedirs("outputs/checkpoints", exist_ok=True)
    checkpoint_path = "outputs/checkpoints/latest.pt"

    start_epoch = 0
    step = 0
    best_auc = 0.0
    save_interval_steps = cfg.train.get("save_interval_steps", 500)

    if resume and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint_data = torch.load(checkpoint_path, map_location=device, weights_only=False)
        try:
            model.load_state_dict(checkpoint_data["model_state_dict"])
            opt.load_state_dict(checkpoint_data["optimizer_state_dict"])
            sched.load_state_dict(checkpoint_data["scheduler_state_dict"])
            is_epoch_end = checkpoint_data.get("is_epoch_end", True)
            start_epoch = checkpoint_data.get("epoch", 0) + 1 if is_epoch_end else checkpoint_data.get("epoch", 0)
            step = checkpoint_data.get("step", 0)
            best_auc = checkpoint_data.get("best_auc", 0.0)
            print(f"Resuming training from epoch {start_epoch}, step {step} (Best AUC: {best_auc:.4f})")
        except RuntimeError as e:
            print(f"[Warning] Incompatible checkpoint architecture in {checkpoint_path}:")
            print(f"  {e}")
            print("  -> Ignoring incompatible checkpoint and starting from scratch.")
            start_epoch = 0
            step = 0
    elif not resume:
        print("Training from scratch (--from-scratch specified). Existing checkpoints ignored.")

    model.train()
    for epoch in range(start_epoch, cfg.train.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0
        epoch_opt_steps = 0
        last_batch_idx = 0
        train_epoch_logits, train_epoch_labels = [], []
        for i, batch in enumerate(train_loader):
            last_batch_idx = i
            frames = batch["frames"].to(device, non_blocking=True)   # (B,T,3,224,224)
            conf = batch["conf"].to(device, non_blocking=True)       # (B,T)
            labels = batch["label"].float().to(device, non_blocking=True)  # (B,)

            # Forward pass with mixed precision
            use_autocast = (device == "cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_autocast):
                if use_aux_losses:
                    features_out = model(frames, conf, return_features=True)
                    if len(features_out) == 4:
                        logits, z_rgb, z_dwt, z_post_vit = features_out
                    else:
                        logits, z_rgb, z_dwt = features_out
                        z_post_vit = None
                    manip_types = batch.get("manip_type")
                    if manip_types is not None:
                        manip_types = manip_types.to(device, non_blocking=True)
                    loss, loss_dict = total_loss(
                        logits, labels, z_rgb=z_rgb, z_dwt=z_dwt, z_post_vit=z_post_vit,
                        manipulation_types=manip_types, **loss_kwargs
                    )
                else:
                    logits = model(frames, conf, return_features=False)
                    loss, loss_dict = total_loss(logits, labels, **loss_kwargs)

                loss = loss / accum_steps

            # Backward
            loss.backward()

            train_epoch_logits.append(logits.detach().float().cpu())
            train_epoch_labels.append(labels.detach().cpu())

            if (i + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()
                if step < total_steps:
                    sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                epoch_opt_steps += 1

                # Periodic step-based checkpointing (so progress is saved mid-epoch)
                if save_interval_steps > 0 and step % save_interval_steps == 0:
                    periodic_data = {
                        "epoch": epoch,
                        "step": step,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": opt.state_dict(),
                        "scheduler_state_dict": sched.state_dict(),
                        "best_auc": best_auc,
                        "is_epoch_end": False,
                    }
                    torch.save(periodic_data, checkpoint_path)
                    print(f"  [Checkpoint] Periodic save at step {step} -> {checkpoint_path}")

                if steps_per_epoch is not None and epoch_opt_steps >= steps_per_epoch:
                    break

            epoch_loss += loss_dict["total"]
            epoch_steps += 1

            # Logging
            if step > 0 and step % 50 == 0:
                avg_loss = epoch_loss / max(1, epoch_steps)
                lr = sched.get_last_lr()[0]
                main_loss_key = "bce" if "bce" in loss_dict else "focal"
                log_msg = (f"Epoch {epoch} | Step {step} | "
                           f"Loss: {loss_dict['total']:.4f} ({main_loss_key}: {loss_dict.get(main_loss_key, 0):.4f}) | "
                           f"Avg: {avg_loss:.4f} | LR: {lr:.2e}")
                if "contrastive" in loss_dict:
                    log_msg += f" | Con: {loss_dict['contrastive']:.4f}"
                if "alignment" in loss_dict:
                    log_msg += f" | Align: {loss_dict['alignment']:.4f}"
                print(log_msg)

                if use_wandb:
                    wandb.log({
                        "train/loss_total": loss_dict["total"],
                        "train/loss_main": loss_dict.get(main_loss_key, 0),
                        "train/loss_contrastive": loss_dict.get("contrastive", 0),
                        "train/loss_alignment": loss_dict.get("alignment", 0),
                        "train/lr": lr,
                    }, step=step)

        # Flush any remaining unaccumulated gradients if loop ended on partial accumulation
        if (last_batch_idx + 1) % accum_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            if step < total_steps:
                sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1

        # --- Compute Epoch Training Diagnostics ---
        train_auc = 0.5
        avg_train_loss = epoch_loss / max(1, epoch_steps)
        if train_epoch_logits:
            tr_logits = torch.cat(train_epoch_logits).numpy()
            tr_labels = torch.cat(train_epoch_labels).numpy()
            tr_probs = 1.0 / (1.0 + np.exp(-tr_logits))
            from sklearn.metrics import roc_auc_score
            try:
                train_auc = float(roc_auc_score(tr_labels, tr_probs))
            except Exception:
                train_auc = 0.5
            print(f"\n--- Epoch {epoch} Training Summary ---")
            print(f"Train Loss: {avg_train_loss:.4f} | Train AUC: {train_auc:.4f}")

        # --- Validation ---
        print(f"Running validation...")
        metrics = evaluate(model, val_loader, device)
        auc = metrics["auc"]
        print(f"Epoch {epoch} | Train AUC: {train_auc:.4f} | Val AUC: {auc:.4f} | "
              f"EER: {metrics['eer']:.4f} | "
              f"F1@0.5: {metrics['f1_05']:.4f} | "
              f"FPR@0.5: {metrics['fpr_at_05']:.4f}")

        if use_wandb:
            wandb.log({
                "val/auc": auc,
                "val/eer": metrics["eer"],
                "val/f1_05": metrics["f1_05"],
                "val/precision_05": metrics["precision_05"],
                "val/recall_05": metrics["recall_05"],
                "val/fpr_at_05": metrics["fpr_at_05"],
                "epoch": epoch,
            })

        # Save epoch checkpoint and latest.pt
        epoch_save_data = {
            "epoch": epoch,
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "scheduler_state_dict": sched.state_dict(),
            "val_auc": auc,
            "best_auc": best_auc,
            "is_epoch_end": True,
        }
        torch.save(epoch_save_data, f"outputs/checkpoints/epoch{epoch}.pt")
        torch.save(epoch_save_data, checkpoint_path)
        print(f"  [Checkpoint] Saved epoch {epoch} -> {checkpoint_path}")


        # Save best
        if auc > best_auc:
            best_auc = auc
            epoch_save_data["best_auc"] = best_auc
            torch.save(epoch_save_data, checkpoint_path)
            torch.save(model.state_dict(), "outputs/checkpoints/best.pt")
            print(f"  >> New best model saved (AUC: {best_auc:.4f})")

    print(f"\nTraining complete. Best AUC: {best_auc:.4f}")
    if use_wandb:
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train S-MSDT")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to single unified YAML config file (e.g. configs/baseline.yaml)")
    parser.add_argument("--config-dir", type=str, default="configs",
                        help="Directory containing YAML config files (data.yaml, model.yaml, train.yaml)")
    parser.add_argument("--baseline", action="store_true",
                        help="Train stripped EfficientNet-B4 baseline with plain BCE")
    parser.add_argument("--override", nargs="*", default=[],
                        help="Override config values (e.g. train.lr=1e-4)")
    parser.add_argument("--from-scratch", action="store_true",
                        help="Start training from scratch, ignoring existing checkpoints")
    args = parser.parse_args()

    # Load configs
    if args.config and os.path.exists(args.config):
        cfg = OmegaConf.load(args.config)
    else:
        data_cfg = OmegaConf.load(os.path.join(args.config_dir, "data.yaml"))
        model_cfg = OmegaConf.load(os.path.join(args.config_dir, "model.yaml"))
        train_cfg = OmegaConf.load(os.path.join(args.config_dir, "train.yaml"))
        cfg = OmegaConf.create({
            "data": data_cfg,
            "model": model_cfg,
            "train": train_cfg,
        })

    # Apply --baseline flag overrides
    if args.baseline:
        cfg.model.type = "baseline_b4"
        cfg.train.loss_type = "bce"
        cfg.train.lambda_contrastive = 0.0
        cfg.train.lambda_align = 0.0

    # Apply CLI overrides
    if args.override:
        overrides = OmegaConf.from_dotlist(args.override)
        cfg = OmegaConf.merge(cfg, overrides)

    print("=" * 60)
    print("S-MSDT Training Configuration")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60)

    main(cfg, resume=(not args.from_scratch))
