"""
S-MSDT Training Script
=====================
Trains the S-MSDT model on preprocessed webdataset shards.
Supports mixed-precision (BF16), gradient accumulation, gradient checkpointing,
and multi-term loss (focal BCE + contrastive + alignment).
"""
import os
import argparse
import torch
import torch.nn as nn

from omegaconf import OmegaConf
from smsdt.models.smsdt import SMSDT
from smsdt.data.dataset import build_dataloader
from smsdt.losses import total_loss
from smsdt.eval import evaluate

def focal_bce(logits, targets, alpha=0.25, gamma=2.0):
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pt = p * targets + (1 - p) * (1 - targets)
    focal = ((1 - pt) ** gamma) * ce
    alpha_w = alpha * targets + (1 - alpha) * (1 - targets)
    return (alpha_w * focal).mean()

def video_level_loss(chunk_logits, video_labels, k=3, alpha=0.25, gamma=2.0):
    """
    Computes loss at the video level using Top-K aggregation of chunk logits.
    chunk_logits: (B, num_chunks)
    video_labels: (B,)
    """
    if chunk_logits.dim() == 1:
        chunk_logits = chunk_logits.unsqueeze(0)
    # Get top K highest scoring chunks (most likely to be fake)
    k = min(k, chunk_logits.size(1))
    topk_logits, _ = torch.topk(chunk_logits, k, dim=1)
    # Aggregate to a single video-level logit
    video_logits = topk_logits.mean(dim=1)
    return focal_bce(video_logits, video_labels, alpha=alpha, gamma=gamma)

def evaluate(model, loader, device):
    from sklearn.metrics import roc_auc_score
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for batch in loader:
            frames = batch["frames"].to(device)
            conf = batch["conf"].to(device)
            logits = model(frames, conf)
            all_logits.append(logits.float().cpu())
            all_labels.append(batch["label"].cpu())
    if len(all_logits) == 0:
        return 0.5
    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    try:
        return roc_auc_score(labels, logits)
    except ValueError:
        return 0.5 # Single class in batch

def main(cfg):
    # --- Device setup ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        cap = torch.cuda.get_device_capability()
        print(f"GPU compute capability: {cap}")

    # --- Model ---
    model = SMSDT(
        embed_dim=cfg.model.embed_dim,
        depth=cfg.model.depth,
        heads=cfg.model.get("heads", 8),
        num_classes=cfg.model.get("num_classes", 1),
        grid=cfg.model.get("grid", 7),
    ).to(device)

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
        for blk in model.blocks:
            blk.spatial.gradient_checkpointing = True
            blk.temporal.gradient_checkpointing = True
        print("Gradient checkpointing enabled for ViT blocks")

    # --- Optimizer & scheduler ---
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
        betas=tuple(cfg.train.get("betas", [0.9, 0.999])),
    )
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=cfg.train.lr,
        total_steps=cfg.train.total_steps,
        pct_start=cfg.train.get("warmup_pct", 0.05),
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
    use_aux_losses = cfg.train.get("lambda_contrastive", 0.1) > 0

    # Loss hyperparameters
    loss_kwargs = {
        "alpha": cfg.train.get("alpha", 0.8),
        "gamma": cfg.train.get("gamma", 2.0),
        "lambda_contrastive": cfg.train.get("lambda_contrastive", 0.1),
        "lambda_align": cfg.train.get("lambda_align", 0.05),
    }

    os.makedirs("outputs/checkpoints", exist_ok=True)
    checkpoint_path = "outputs/checkpoints/latest.pt"

    start_epoch = 0
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint_data = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint_data["model_state_dict"])
        opt.load_state_dict(checkpoint_data["optimizer_state_dict"])
        sched.load_state_dict(checkpoint_data["scheduler_state_dict"])
        start_epoch = checkpoint_data["epoch"] + 1
        print(f"Resuming training from epoch {start_epoch}")

    model.train()
    for epoch in range(start_epoch, cfg.train.epochs):
        for i, batch in enumerate(train_loader):
            frames = batch["frames"].to(device, non_blocking=True)   # (B,T,3,224,224)
            conf = batch["conf"].to(device, non_blocking=True)       # (B,T)
            labels = batch["label"].float().to(device, non_blocking=True)  # (B,)

            # Forward pass with mixed precision
            use_autocast = (device == "cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_autocast):
                if use_aux_losses:
                    logits, z_rgb, z_dwt = model(frames, conf, return_features=True)
                    manip_types = batch.get("manip_type")
                    if manip_types is not None:
                        manip_types = manip_types.to(device, non_blocking=True)
                    loss, loss_dict = total_loss(
                        logits, labels, z_rgb, z_dwt, manip_types, **loss_kwargs
                    )
                else:
                    logits = model(frames, conf, return_features=False)
                    loss, loss_dict = total_loss(logits, labels, **loss_kwargs)

                loss = loss / accum_steps

            # Backward
            loss.backward()

            if (i + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1

            epoch_loss += loss_dict["total"]
            epoch_steps += 1

            # Logging
            if step > 0 and step % 50 == 0:
                avg_loss = epoch_loss / epoch_steps
                lr = sched.get_last_lr()[0]
                log_msg = (f"Epoch {epoch} | Step {step} | "
                           f"Loss: {loss_dict['total']:.4f} | "
                           f"Avg: {avg_loss:.4f} | LR: {lr:.2e}")
                if "contrastive" in loss_dict:
                    log_msg += f" | Con: {loss_dict['contrastive']:.4f}"
                if "alignment" in loss_dict:
                    log_msg += f" | Align: {loss_dict['alignment']:.4f}"
                print(log_msg)

                if use_wandb:
                    wandb.log({
                        "train/loss_total": loss_dict["total"],
                        "train/loss_focal": loss_dict["focal"],
                        "train/loss_contrastive": loss_dict.get("contrastive", 0),
                        "train/loss_alignment": loss_dict.get("alignment", 0),
                        "train/lr": lr,
                    }, step=step)

        # --- Validation ---
        print(f"\nEpoch {epoch} finished. Running validation...")
        metrics = evaluate(model, val_loader, device)
        auc = metrics["auc"]
        print(f"Epoch {epoch} | Val AUC: {auc:.4f} | "
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

        # Save checkpoint
        torch.save({
            "epoch": epoch,
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "scheduler_state_dict": sched.state_dict(),
            "val_auc": auc,
        }, f"outputs/checkpoints/epoch{epoch}.pt")

        # Save best
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), "outputs/checkpoints/best.pt")
            print(f"  >> New best model saved (AUC: {best_auc:.4f})")

    print(f"\nTraining complete. Best AUC: {best_auc:.4f}")
    if use_wandb:
        wandb.finish()

        val_auc = evaluate(model, val_loader, device)
        wandb.log({"val/auc": val_auc, "epoch": epoch})
        print(f"Epoch {epoch} Val AUC: {val_auc}")

        save_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "scheduler_state_dict": sched.state_dict()
        }
        torch.save(save_data, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train S-MSDT")
    parser.add_argument("--config-dir", type=str, default="configs",
                        help="Directory containing YAML config files")
    parser.add_argument("--override", nargs="*", default=[],
                        help="Override config values (e.g. train.lr=1e-4)")
    args = parser.parse_args()

    # Load configs from YAML files
    data_cfg = OmegaConf.load(os.path.join(args.config_dir, "data.yaml"))
    model_cfg = OmegaConf.load(os.path.join(args.config_dir, "model.yaml"))
    train_cfg = OmegaConf.load(os.path.join(args.config_dir, "train.yaml"))

    cfg = OmegaConf.create({
        "data": data_cfg,
        "model": model_cfg,
        "train": train_cfg,
    })

    # Apply CLI overrides
    if args.override:
        overrides = OmegaConf.from_dotlist(args.override)
        cfg = OmegaConf.merge(cfg, overrides)

    print("=" * 60)
    print("S-MSDT Training Configuration")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60)

    main(cfg)
