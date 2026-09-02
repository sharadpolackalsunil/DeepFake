"""
S-MSDT Evaluation Script
========================
Computes comprehensive metrics: AUC, EER, Precision, Recall, F1, FPR, ROC curve.
Supports per-dataset and per-manipulation-type breakdown.
"""
import os
import argparse
import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_fscore_support,
    accuracy_score, confusion_matrix,
)
from omegaconf import OmegaConf

from smsdt.models.smsdt import SMSDT
from smsdt.data.dataset import build_dataloader


def compute_eer(fpr, tpr, thresholds):
    """Equal Error Rate: threshold where FPR == FNR."""
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    return float(fpr[idx]), float(thresholds[idx])


def compute_metrics(probs, labels):
    """Compute full metrics suite from probabilities and labels.

    Args:
        probs: (N,) numpy array of predicted probabilities (after sigmoid)
        labels: (N,) numpy array of binary labels

    Returns:
        dict with all metrics
    """
    if len(np.unique(labels)) < 2:
        print("Warning: Only one class present in labels, metrics may be unreliable")
        return {
            "auc": 0.5, "eer": 0.5, "eer_threshold": 0.5,
            "accuracy": float((labels == (probs > 0.5).astype(int)).mean()),
            "fpr_at_05": 0.0,
            "precision_eer": 0.0, "recall_eer": 0.0, "f1_eer": 0.0,
            "precision_05": 0.0, "recall_05": 0.0, "f1_05": 0.0,
        }

    # Core metrics
    auc = roc_auc_score(labels, probs)
    fpr_arr, tpr_arr, thresholds = roc_curve(labels, probs)
    eer_value, eer_thresh = compute_eer(fpr_arr, tpr_arr, thresholds)

    # Threshold-dependent metrics at EER threshold
    preds_eer = (probs >= eer_thresh).astype(int)
    prec_eer, rec_eer, f1_eer, _ = precision_recall_fscore_support(
        labels, preds_eer, average="binary", zero_division=0)

    # Threshold-dependent metrics at 0.5
    preds_05 = (probs >= 0.5).astype(int)
    prec_05, rec_05, f1_05, _ = precision_recall_fscore_support(
        labels, preds_05, average="binary", zero_division=0)

    tn, fp, fn, tp = confusion_matrix(labels, preds_05, labels=[0, 1]).ravel()
    fpr_at_05 = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    acc = accuracy_score(labels, preds_05)

    return {
        "auc": float(auc),
        "eer": float(eer_value),
        "eer_threshold": float(eer_thresh),
        "accuracy": float(acc),
        "fpr_at_05": float(fpr_at_05),
        "precision_eer": float(prec_eer),
        "recall_eer": float(rec_eer),
        "f1_eer": float(f1_eer),
        "precision_05": float(prec_05),
        "recall_05": float(rec_05),
        "f1_05": float(f1_05),
        "roc_fpr": fpr_arr,
        "roc_tpr": tpr_arr,
        "roc_thresholds": thresholds,
    }


def evaluate(model, loader, device):
    """Run evaluation and compute all metrics.

    Args:
        model: S-MSDT model
        loader: validation/test DataLoader
        device: "cuda" or "cpu"

    Returns:
        dict of metrics
    """
    model.eval()
    all_logits, all_labels, all_manip_types = [], [], []

    with torch.no_grad():
        autocast_enabled = (device == "cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
            for batch in loader:
                frames = batch["frames"].to(device)
                conf = batch["conf"].to(device)
                logits = model(frames, conf, return_features=False)
                all_logits.append(logits.float().cpu())
                all_labels.append(batch["label"].cpu())
                if "manip_type" in batch:
                    all_manip_types.append(batch["manip_type"])

    if not all_logits:
        print("Warning: No data in evaluation loader")
        return {"auc": 0.5, "eer": 0.5, "eer_threshold": 0.5,
                "accuracy": 0.5, "fpr_at_05": 0.0,
                "precision_eer": 0.0, "recall_eer": 0.0, "f1_eer": 0.0,
                "precision_05": 0.0, "recall_05": 0.0, "f1_05": 0.0}

    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()

    # Apply sigmoid to convert logits → probabilities
    probs = 1.0 / (1.0 + np.exp(-logits))

    # Overall metrics
    metrics = compute_metrics(probs, labels)

    # Per-manipulation-type breakdown
    if all_manip_types:
        manip_types = torch.cat(all_manip_types).numpy()
        from smsdt.data.dataset import MANIP_TYPE_MAP
        reverse_map = {v: k for k, v in MANIP_TYPE_MAP.items()}

        per_manip = {}
        for mt_id in np.unique(manip_types):
            mask = (manip_types == mt_id)
            if mask.sum() < 2:
                continue
            mt_name = reverse_map.get(int(mt_id), f"type_{mt_id}")
            mt_metrics = compute_metrics(probs[mask], labels[mask])
            per_manip[mt_name] = {
                "auc": mt_metrics["auc"],
                "f1_05": mt_metrics["f1_05"],
                "count": int(mask.sum()),
            }
        metrics["per_manipulation"] = per_manip

    return metrics


def print_metrics(metrics, title="Evaluation Results"):
    """Pretty-print evaluation metrics."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")
    print(f"  AUC:              {metrics['auc']:.4f}")
    print(f"  EER:              {metrics['eer']:.4f} (threshold={metrics['eer_threshold']:.4f})")
    print(f"  Accuracy @0.5:    {metrics['accuracy']:.4f}")
    print(f"  Precision @0.5:   {metrics['precision_05']:.4f}")
    print(f"  Recall @0.5:      {metrics['recall_05']:.4f}")
    print(f"  F1 @0.5:          {metrics['f1_05']:.4f}")
    print(f"  FPR @0.5:         {metrics['fpr_at_05']:.4f}")
    print(f"  Precision @EER:   {metrics['precision_eer']:.4f}")
    print(f"  Recall @EER:      {metrics['recall_eer']:.4f}")
    print(f"  F1 @EER:          {metrics['f1_eer']:.4f}")

    if "per_manipulation" in metrics:
        print(f"\n  Per-Manipulation Breakdown:")
        print(f"  {'Type':<20} {'AUC':>8} {'F1@0.5':>8} {'Count':>8}")
        print(f"  {'-'*48}")
        for mt_name, mt_data in metrics["per_manipulation"].items():
            print(f"  {mt_name:<20} {mt_data['auc']:>8.4f} "
                  f"{mt_data['f1_05']:>8.4f} {mt_data['count']:>8d}")

    print(f"{'=' * 60}\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate S-MSDT")
    parser.add_argument("--ckpt", type=str, default="outputs/checkpoints/best.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--split", type=str, default="val",
                        choices=["val", "test"], help="Evaluation split")
    parser.add_argument("--config-dir", type=str, default="configs",
                        help="Directory containing YAML config files")
    parser.add_argument("--workers", type=int, default=6,
                        help="Number of dataloader workers")
    args = parser.parse_args()

    # Load configs
    data_cfg = OmegaConf.load(os.path.join(args.config_dir, "data.yaml"))
    model_cfg = OmegaConf.load(os.path.join(args.config_dir, "model.yaml"))

    cfg = OmegaConf.create({"data": data_cfg, "model": model_cfg})

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Build model
    model = SMSDT(
        embed_dim=cfg.model.embed_dim,
        depth=cfg.model.depth,
        heads=cfg.model.get("heads", 8),
        num_classes=cfg.model.get("num_classes", 1),
        grid=cfg.model.get("grid", 7),
    ).to(device)

    # Load checkpoint
    if os.path.exists(args.ckpt):
        ckpt = torch.load(args.ckpt, map_location=device)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
            print(f"Loaded checkpoint: {args.ckpt} (epoch {ckpt.get('epoch', '?')})")
        else:
            model.load_state_dict(ckpt)
            print(f"Loaded checkpoint: {args.ckpt}")
    else:
        print(f"WARNING: Checkpoint {args.ckpt} not found. Evaluating random weights.")

    # Build loader and evaluate
    loader = build_dataloader(cfg.data, split=args.split, num_workers=args.workers)
    metrics = evaluate(model, loader, device)
    print_metrics(metrics, title=f"{args.split.upper()} Set Results")


if __name__ == "__main__":
    main()
