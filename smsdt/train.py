import torch
import wandb
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from smsdt.models.smsdt import SMSDT
from smsdt.data.dataset import build_dataloader
import os

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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = SMSDT(embed_dim=cfg.model.embed_dim, depth=cfg.model.depth).to(device)
    if device == "cuda" and hasattr(torch, "compile"):
        model = torch.compile(model, mode="max-autotune")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.train.lr, total_steps=cfg.train.total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=False)   # bf16 needs no loss scaling

    # Dummy loaders since we don't have real data yet
    train_loader = build_dataloader(cfg.data, split="train", num_workers=cfg.train.workers)
    val_loader = build_dataloader(cfg.data, split="val", num_workers=max(1, cfg.train.workers // 2))

    wandb.init(project="smsdt-dgx-spark", config=cfg, mode="disabled") # Disable wandb for local tests
    step = 0
    accum_steps = cfg.train.grad_accum

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
            labels = batch["label"].float().to(device, non_blocking=True)

            with torch.autocast(device_type=device, dtype=torch.bfloat16 if device == "cuda" else torch.float32):
                logits = model(frames, conf)
                loss = focal_bce(logits, labels) / accum_steps

            loss.backward()
            if (i + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1

            if step % 50 == 0:
                wandb.log({"train/loss": loss.item() * accum_steps, "lr": sched.get_last_lr()[0]}, step=step)
                print(f"Epoch {epoch} Step {step} Loss: {loss.item() * accum_steps}")

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
    from omegaconf import OmegaConf
    cfg = OmegaConf.create({
        "model": {"embed_dim": 256, "depth": 4},
        "train": {"lr": 3e-4, "weight_decay": 0.05, "total_steps": 1000, "grad_accum": 1, "grad_clip": 1.0, "epochs": 1, "workers": 0},
        "data": {"dataset": "ffpp", "batch_size": 2}
    })
    main(cfg)
