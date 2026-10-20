"""S-MSDT Training Pipeline.

Supports:
- Phase 1: Paired real/fake map-style dataset (PairedClipDataset) with exact epochs,
  clip-consistent SBI augmentation, detection_loss (BCE + patch-mask BCE),
  cosine-with-warmup scheduler, and Model EMA.
- Legacy fallback: WebDataset shard streaming with OneCycleLR and Focal BCE.
- Mixed precision (BF16), decoupled weight decay, and non-blocking bulk GPU evaluation.
"""
import os
import math
import copy
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from omegaconf import OmegaConf
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from smsdt.models.smsdt import SMSDT
from smsdt.losses import detection_loss, total_loss
from smsdt.eval import evaluate


def cosine_with_warmup(optimizer, total_steps, warmup_frac=0.03, min_ratio=0.01):
    """Cosine learning rate scheduler with linear warmup."""
    warmup = max(1, int(total_steps * warmup_frac))

    def factor(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
        return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def build_param_groups(model, backbone_prefix="rgb_branch.backbone", lr_backbone=5e-5, lr_head=5e-4, wd=0.05):
    """Decoupled weight decay parameter groups: 0 decay on 1D/bias/norm/embed."""
    groups = {(b, d): [] for b in (True, False) for d in (True, False)}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_bb = n.startswith(backbone_prefix)
        no_decay = p.ndim == 1 or n.endswith(".bias") or "norm" in n.lower() or "embed" in n
        groups[(is_bb, not no_decay)].append(p)

    out = []
    for (is_bb, decay), ps in groups.items():
        if ps:
            out.append({
                "params": ps,
                "lr": lr_backbone if is_bb else lr_head,
                "weight_decay": wd if decay else 0.0,
            })
    return out


def main(cfg, resume=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        cap = torch.cuda.get_device_capability()
        print(f"GPU compute capability: {cap}")

    # --- Model initialization ---
    grid = cfg.model.get("grid", 14)
    use_dense_psit = cfg.model.get("use_dense_psit", True)
    pool_energy = cfg.model.get("pool_energy", False)

    model = SMSDT(
        embed_dim=cfg.model.embed_dim,
        depth=cfg.model.depth,
        heads=cfg.model.get("heads", 8),
        num_classes=cfg.model.get("num_classes", 1),
        grid=grid,
        stem_type=cfg.model.get("stem_type", "strided_patch"),
        pool_energy=pool_energy,
        use_dense_psit=use_dense_psit,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model parameters: {param_count:.1f}M (grid: {grid}x{grid})")

    if device == "cuda" and cfg.train.get("compile", False):
        try:
            model = torch.compile(model, mode="max-autotune")
            print("Model compiled with torch.compile (max-autotune)")
        except Exception as e:
            print(f"torch.compile failed: {e}")

    if cfg.train.get("gradient_checkpointing", False):
        model.set_gradient_checkpointing(True)
        print("Gradient checkpointing enabled for ViT blocks")

    # --- Optimizer setup ---
    backbone_lr = float(cfg.train.get("backbone_lr", 5e-5))
    head_lr = float(cfg.train.get("lr", 5e-4))
    weight_decay = float(cfg.train.get("weight_decay", 0.05))

    param_groups = build_param_groups(
        model,
        backbone_prefix="rgb_branch.backbone",
        lr_backbone=backbone_lr,
        lr_head=head_lr,
        wd=weight_decay,
    )
    opt = torch.optim.AdamW(param_groups, betas=tuple(cfg.train.get("betas", [0.9, 0.999])))

    # --- Data loader initialization ---
    store_root = cfg.data.get("store_root", "data/store")
    index_csv = cfg.data.get("index_csv", "data/splits/index.csv")
    is_store_pipeline = os.path.isdir(store_root) and os.path.exists(index_csv)

    if is_store_pipeline:
        print(f"Using Map-Style Paired Store Pipeline from {store_root}")
        from smsdt.data.video_dataset import PairedClipDataset, paired_collate, EvalClipDataset
        from smsdt.data.sbi_video import sbi_clip

        train_dataset = PairedClipDataset(
            index_csv=index_csv,
            store_root=store_root,
            T=cfg.data.get("chunk_length", 8),
            stride=cfg.data.get("chunk_stride", 1),
            p_sbi=float(cfg.train.get("p_sbi", 0.5)),
            sbi_fn=sbi_clip,
            mask_grid=grid,
        )
        batch_pairs = int(cfg.train.get("batch_pairs", 8))
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_pairs,
            shuffle=True,
            num_workers=int(cfg.train.get("workers", 8)),
            collate_fn=paired_collate,
            pin_memory=True,
            drop_last=True,
        )

        val_dataset = EvalClipDataset(
            index_csv=index_csv,
            store_root=store_root,
            split="val",
            K=8,
            T=cfg.data.get("chunk_length", 8),
            stride=cfg.data.get("chunk_stride", 1),
            size=cfg.data.get("img_size", 224),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=int(cfg.train.get("val_batch_size", 4)),
            shuffle=False,
            num_workers=int(cfg.train.get("val_workers", 2)),
            pin_memory=False,
        )
        steps_per_epoch = len(train_loader)
    else:
        print("Using WebDataset Shard Pipeline (Fallback)")
        from smsdt.data.dataset import build_dataloader
        train_loader = build_dataloader(cfg.data, split="train", num_workers=cfg.train.get("workers", 8))
        val_loader = build_dataloader(
            cfg.data, split="val",
            num_workers=cfg.train.get("val_workers", 8),
            batch_size=cfg.train.get("val_batch_size", 32),
        )
        steps_per_epoch = cfg.train.get("steps_per_epoch", 1000)

    # --- Scheduler setup ---
    epochs = int(cfg.train.get("epochs", 60))
    accum_steps = int(cfg.train.get("grad_accum", 1))
    total_opt_steps = math.ceil(steps_per_epoch / accum_steps) * epochs
    warmup_frac = float(cfg.train.get("warmup_frac", 0.03))
    min_lr_ratio = float(cfg.train.get("min_lr_ratio", 0.01))

    sched = cosine_with_warmup(opt, total_opt_steps, warmup_frac=warmup_frac, min_ratio=min_lr_ratio)
    print(f"Total Epochs: {epochs} | Steps/epoch: {steps_per_epoch} | Total Optimizer Steps: {total_opt_steps}")

    # --- Model EMA ---
    ema_decay = float(cfg.train.get("ema_decay", 0.999))
    ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(ema_decay)) if ema_decay > 0 else None

    # --- Logging & checkpointing ---
    use_wandb = cfg.train.get("use_wandb", False)
    if use_wandb:
        import wandb
        wandb.init(project="smsdt-sota", config=OmegaConf.to_container(cfg))

    os.makedirs("outputs/checkpoints", exist_ok=True)
    checkpoint_path = "outputs/checkpoints/latest.pt"
    best_path = "outputs/checkpoints/best.pt"
    best_ema_path = "outputs/checkpoints/best_ema.pt"

    start_epoch = 0
    step = 0
    best_auc = 0.0
    grad_clip = float(cfg.train.get("grad_clip", 1.0))
    label_smoothing = float(cfg.train.get("label_smoothing", 0.05))
    lam_mask = float(cfg.train.get("lambda_mask", 0.5))

    model.train()
    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0
        opt.zero_grad(set_to_none=True)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1:02d}/{epochs:02d} [Train]", total=len(train_loader))
        for i, batch in enumerate(pbar):
            frames = batch["frames"].to(device, non_blocking=True)
            labels = batch["label"].float().to(device, non_blocking=True)
            masks = batch.get("mask")
            has_masks = batch.get("has_mask")
            if masks is not None:
                masks = masks.to(device, non_blocking=True)
            if has_masks is not None:
                has_masks = has_masks.to(device, non_blocking=True)

            use_autocast = (device == "cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_autocast):
                logit, patch_logits = model(frames)
                loss, loss_logs = detection_loss(
                    logit, labels,
                    patch_logits=patch_logits,
                    mask=masks,
                    has_mask=has_masks,
                    smoothing=label_smoothing,
                    lam_mask=lam_mask,
                )
                loss = loss / accum_steps

            loss.backward()

            if (i + 1) % accum_steps == 0 or (i + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()
                if ema_model is not None:
                    ema_model.update_parameters(model)
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1

            epoch_loss += loss_logs["cls"]
            epoch_steps += 1

            cur_lr = sched.get_last_lr()[0]
            postfix = {"Loss": f"{loss_logs['cls']:.4f}", "LR": f"{cur_lr:.2e}"}
            if "mask" in loss_logs:
                postfix["Mask"] = f"{loss_logs['mask']:.4f}"
            pbar.set_postfix(postfix)

        avg_train_loss = epoch_loss / max(1, epoch_steps)

        # --- Validation loop at epoch end ---
        print(f"\n--- Epoch {epoch + 1:02d}/{epochs:02d} Validation ---")
        eval_model = ema_model.module if ema_model is not None else model
        metrics = evaluate(eval_model, val_loader, device)
        auc = metrics["auc"]
        eer = metrics["eer"]
        f1_05 = metrics["f1_05"]
        print(f"Epoch {epoch + 1:02d} Val AUC: {auc:.4f} | EER: {eer:.4f} | F1@0.5: {f1_05:.4f} | Train Loss: {avg_train_loss:.4f}")

        # Save latest checkpoint
        save_dict = {
            "epoch": epoch + 1,
            "step": step,
            "model_state_dict": model.state_dict(),
            "ema_state_dict": ema_model.state_dict() if ema_model is not None else None,
            "optimizer_state_dict": opt.state_dict(),
            "scheduler_state_dict": sched.state_dict(),
            "val_auc": auc,
            "best_auc": best_auc,
        }
        torch.save(save_dict, checkpoint_path)

        if auc > best_auc:
            best_auc = auc
            torch.save(save_dict, best_path)
            if ema_model is not None:
                torch.save(ema_model.module.state_dict(), best_ema_path)
            print(f"  * New Best Val AUC: {best_auc:.4f} (Saved to {best_path} and {best_ema_path})")

    print(f"\nTraining Complete. Best Validation AUC: {best_auc:.4f}")
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train S-MSDT")
    parser.add_argument("--config-dir", type=str, default="configs", help="Config directory")
    parser.add_argument("--override", nargs="*", default=[], help="Config overrides")
    parser.add_argument("--from-scratch", action="store_true", help="Start training from scratch")
    args = parser.parse_args()

    data_cfg = OmegaConf.load(os.path.join(args.config_dir, "data.yaml"))
    model_cfg = OmegaConf.load(os.path.join(args.config_dir, "model.yaml"))
    train_cfg = OmegaConf.load(os.path.join(args.config_dir, "train.yaml"))

    cfg = OmegaConf.create({"data": data_cfg, "model": model_cfg, "train": train_cfg})
    if args.override:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.override))

    main(cfg, resume=(not args.from_scratch))
