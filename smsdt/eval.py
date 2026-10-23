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


def evaluate(model, loader, device, top_k_ratio=0.5, max_batches=None):
    """Run evaluation and compute frame-level and video-level metrics.

    Args:
        model: S-MSDT model
        loader: validation/test DataLoader
        device: "cuda" or "cpu"
        top_k_ratio: fraction of highest-scoring chunks per video to average for video-level score
        max_batches: optional cap on number of batches to evaluate (None for full split)

    Returns:
        dict of metrics
    """
    model.eval()
    all_logits, all_labels, all_manip_types = [], [], []
    all_video_ids = []

    with torch.no_grad():
        is_cuda = (device == "cuda" or (isinstance(device, torch.device) and device.type == "cuda"))
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=is_cuda):
            for batch_idx, batch in enumerate(loader):
                if max_batches is not None and batch_idx >= max_batches:
                    break
                frames = batch["frames"].to(device, non_blocking=True)
                conf = batch.get("conf", None)
                if conf is not None:
                    conf = conf.to(device, non_blocking=True)

                prior = batch.get("prior", None)
                if frames.ndim == 6:
                    # (B, K, T, C, H, W) from EvalClipDataset
                    B, K, T, C, H, W = frames.shape
                    frames_flat = frames.view(B * K, T, C, H, W)
                    conf_flat = conf.view(B * K, T) if conf is not None else None
                    if prior is not None:
                        prior_flat = prior.to(device, non_blocking=True).view(B * K, T, -1)
                    else:
                        prior_flat = None

                    # Process in chunks of max 16 clips (128 frames, identical to training batch)
                    # to strictly eliminate GPU memory spikes and allocation fragmentation
                    chunk_size = 16
                    sub_logits = []
                    for start_idx in range(0, B * K, chunk_size):
                        end_idx = min(start_idx + chunk_size, B * K)
                        chunk_frames = frames_flat[start_idx:end_idx]
                        chunk_conf = conf_flat[start_idx:end_idx] if conf_flat is not None else None
                        chunk_prior = prior_flat[start_idx:end_idx] if prior_flat is not None else None
                        out = model(chunk_frames, chunk_conf, prior=chunk_prior)
                        logits_chunk = out[0] if isinstance(out, (tuple, list)) else out
                        sub_logits.append(logits_chunk)

                    logits_flat = torch.cat(sub_logits, dim=0)
                    probs_flat = torch.sigmoid(logits_flat).view(B, K)
                    probs_vid = probs_flat.mean(dim=1)
                    logits = torch.logit(probs_vid.clamp(1e-6, 1.0 - 1e-6))
                else:
                    prior_dev = prior.to(device, non_blocking=True) if prior is not None else None
                    out = model(frames, conf, prior=prior_dev)
                    logits = out[0] if isinstance(out, (tuple, list)) else out

                all_logits.append(logits.float().cpu())
                all_labels.append(batch["label"].float().cpu())
                if "manip_type" in batch:
                    all_manip_types.append(batch["manip_type"].cpu())
                elif "method" in batch:
                    meths = batch["method"]
                    if isinstance(meths, (list, tuple)):
                        from smsdt.data.video_dataset import METHOD_ID
                        all_manip_types.append(torch.tensor([METHOD_ID.get(m, 1) for m in meths]))
                if "video_id" in batch:
                    vids = batch["video_id"]
                    if isinstance(vids, torch.Tensor):
                        all_video_ids.extend(vids.tolist())
                    elif isinstance(vids, (list, tuple)):
                        all_video_ids.extend(list(vids))

    if not all_logits:
        print("Warning: No data in evaluation loader")
        return {"auc": 0.5, "eer": 0.5, "eer_threshold": 0.5,
                "accuracy": 0.5, "fpr_at_05": 0.0,
                "precision_eer": 0.0, "recall_eer": 0.0, "f1_eer": 0.0,
                "precision_05": 0.0, "recall_05": 0.0, "f1_05": 0.0}

    logits = torch.cat(all_logits).cpu().numpy()
    labels = torch.cat(all_labels).cpu().numpy()

    # Apply sigmoid to convert logits → probabilities
    probs = 1.0 / (1.0 + np.exp(-logits))

    # Overall frame/chunk-level metrics
    metrics = compute_metrics(probs, labels)

    # Video-level metrics via top-k chunk aggregation
    if len(all_video_ids) == len(probs):
        video_dict = {}
        for vid, p, l in zip(all_video_ids, probs, labels):
            video_dict.setdefault(vid, {"probs": [], "label": l})
            video_dict[vid]["probs"].append(float(p))

        v_probs, v_labels = [], []
        for vid, data in video_dict.items():
            ps = sorted(data["probs"], reverse=True)
            k = max(1, int(len(ps) * top_k_ratio))
            v_probs.append(float(np.mean(ps[:k])))
            v_labels.append(data["label"])

        v_probs = np.array(v_probs)
        v_labels = np.array(v_labels)
        metrics["video_level"] = compute_metrics(v_probs, v_labels)
        metrics["num_videos"] = len(video_dict)

    # Per-manipulation-type breakdown (each fake type evaluated against ALL real samples)
    if all_manip_types:
        from smsdt.data.video_dataset import METHOD_ID
        reverse_map = {v: k for k, v in METHOD_ID.items()}
        manip_types = torch.cat(all_manip_types).cpu().numpy()

        real_mask = (labels == 0)
        per_manip = {}
        for mt_id in np.unique(manip_types):
            mt_name = reverse_map.get(int(mt_id), f"type_{mt_id}")
            if mt_name == "real":
                continue
            fake_mask = (manip_types == mt_id)
            combined_mask = fake_mask | real_mask
            if fake_mask.sum() > 0 and real_mask.sum() > 0:
                mt_metrics = compute_metrics(probs[combined_mask], labels[combined_mask])
                per_manip[mt_name] = {
                    "auc": mt_metrics["auc"],
                    "eer": mt_metrics["eer"],
                    "f1_05": mt_metrics["f1_05"],
                    "fake_count": int(fake_mask.sum()),
                    "real_count": int(real_mask.sum()),
                    "count": int(combined_mask.sum()),
                }
        metrics["per_manipulation"] = per_manip

    return metrics


def print_metrics(metrics, title="Evaluation Results"):
    """Pretty-print evaluation metrics."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")
    print(f"  [Frame / Chunk-Level Metrics]")
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

    if "video_level" in metrics:
        v_m = metrics["video_level"]
        print(f"\n  [Video-Level Metrics (N={metrics.get('num_videos', '?')})]")
        print(f"  Video AUC:        {v_m['auc']:.4f}")
        print(f"  Video EER:        {v_m['eer']:.4f}")
        print(f"  Video Acc @0.5:   {v_m['accuracy']:.4f}")
        print(f"  Video F1 @0.5:    {v_m['f1_05']:.4f}")

    if "per_manipulation" in metrics:
        print(f"\n  Per-Manipulation Breakdown (each fake scored vs all real):")
        print(f"  {'Type':<20} {'AUC':>8} {'EER':>8} {'F1@0.5':>8} {'Fakes':>8}")
        print(f"  {'-'*56}")
        for mt_name, mt_data in metrics["per_manipulation"].items():
            print(f"  {mt_name:<20} {mt_data['auc']:>8.4f} {mt_data.get('eer', 0.5):>8.4f} "
                  f"{mt_data['f1_05']:>8.4f} {mt_data.get('fake_count', mt_data['count']):>8d}")

    print(f"{'=' * 60}\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate S-MSDT")
    parser.add_argument("--ckpt", type=str, default="outputs/checkpoints/best.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--split", type=str, default="val",
                        choices=["val", "test"], help="Evaluation split")
    parser.add_argument("--config-dir", type=str, default="configs",
                        help="Directory containing YAML config files")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for evaluation (default: 32)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of dataloader workers (default: 8)")
    parser.add_argument("--max-batches", type=int, default=None,
                        help="Optional maximum number of batches to evaluate")
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
        try:
            ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
        except TypeError:
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
    loader = build_dataloader(
        cfg.data, split=args.split,
        num_workers=args.workers,
        batch_size=args.batch_size,
    )
    metrics = evaluate(model, loader, device, max_batches=args.max_batches)
    print_metrics(metrics, title=f"{args.split.upper()} Set Results")


if __name__ == "__main__":
    main()
