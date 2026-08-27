import torch
from sklearn.metrics import roc_auc_score
from smsdt.models.smsdt import SMSDT
from smsdt.data.dataset import build_dataloader

def evaluate(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad(), torch.autocast(device_type=device, dtype=torch.bfloat16 if device == "cuda" else torch.float32):
        for batch in loader:
            frames = batch["frames"].to(device)
            conf = batch["conf"].to(device)
            logits = model(frames, conf)
            all_logits.append(logits.float().cpu())
            all_labels.append(batch["label"].cpu())

    if not all_logits:
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
    if "checkpoint_path" in cfg and cfg.checkpoint_path:
        model.load_state_dict(torch.load(cfg.checkpoint_path, map_location=device))

    val_loader = build_dataloader(cfg.data, split="val", num_workers=cfg.eval.workers)
    auc = evaluate(model, val_loader, device)
    print(f"Evaluation AUC: {auc:.4f}")

if __name__ == "__main__":
    from omegaconf import OmegaConf
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="")
    args = parser.parse_args()

    cfg = OmegaConf.create({
        "model": {"embed_dim": 256, "depth": 4},
        "data": {"dataset": "ffpp", "batch_size": 2},
        "eval": {"workers": 0},
        "checkpoint_path": args.ckpt
    })
    main(cfg)
