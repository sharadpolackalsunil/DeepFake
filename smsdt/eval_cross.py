"""Cross-dataset evaluation runner for S-MSDT (v1 and v2 architectures).

Evaluates checkpoints across multiple benchmarks (FF++ test, Celeb-DF v2, DFD, DFDC, FFIW)
using standardized clip-level (K=8) deterministic center crops and fixed aggregation rules
(Mean clip probability and Top-50% mean), reporting Video AUC, Frame/Clip AUC, and EER.
"""
import argparse
import csv
import json
import os
import sys
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, log_loss
from torch.utils.data import DataLoader

from smsdt.data.video_dataset import EvalClipDataset
from smsdt.models.clip_encoder import CLIPFrameEncoder, CLIP_MEAN, CLIP_STD
from smsdt.models.smsdt import SMSDT
from smsdt.models.smsdt_v2 import SMSDTv2
from smsdt.train_v2 import get_clip_normalize


def compute_eer(y_true, y_scores):
    """Compute Equal Error Rate (EER) and optimal threshold."""
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1.0 - tpr
    idx = np.nanargmin(np.absolute(fnr - fpr))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    return float(eer), float(thresholds[idx])


def load_model(ckpt_path: str, device: torch.device):
    """Load S-MSDT v1 or v2 model from checkpoint with robust dictionary extraction."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt.get("cfg", {}) if isinstance(ckpt, dict) else {}

    # Extract state dict from various checkpoint formats
    if isinstance(ckpt, dict):
        if "ema_state_dict" in ckpt and ckpt["ema_state_dict"] is not None:
            state_dict = ckpt["ema_state_dict"]
            print("Loaded 'ema_state_dict' from checkpoint.")
        elif "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
            print("Loaded 'model_state_dict' from checkpoint.")
        elif "model_ema" in ckpt:
            state_dict = ckpt["model_ema"]
            print("Loaded 'model_ema' from checkpoint.")
        elif "model" in ckpt:
            state_dict = ckpt["model"]
            print("Loaded 'model' from checkpoint.")
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
            print("Loaded 'state_dict' from checkpoint.")
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    # Clean prefixes: 'module.' (DDP/AveragedModel) or '_orig_mod.' (torch.compile)
    cleaned = {}
    for k, v in state_dict.items():
        k_clean = k
        if k_clean.startswith("module."):
            k_clean = k_clean[7:]
        if k_clean.startswith("_orig_mod."):
            k_clean = k_clean[10:]
        cleaned[k_clean] = v
    state_dict = cleaned

    # Detect v1 vs v2 architecture
    is_v2 = any("encoder.vit" in k for k in state_dict.keys()) or cfg.get("stage") is not None

    if is_v2:
        model_cfg = cfg.get("model", {})
        enc = CLIPFrameEncoder(
            name=model_cfg.get("backbone", "openai/clip-vit-large-patch14"),
            tune="layernorm",
            grad_ckpt=False,
        )
        model = SMSDTv2(
            encoder=enc,
            dim=model_cfg.get("dim", 384),
            depth=model_cfg.get("depth", 2),
            heads=model_cfg.get("heads", 6),
            max_frames=model_cfg.get("max_frames", 16),
            grid=model_cfg.get("grid", 16),
            topk_frac=model_cfg.get("topk_frac", 0.05),
        )
        norm_fn = get_clip_normalize()
        model_type = "v2"
    else:
        # S-MSDT v1 (EfficientNet-B4 + DensePSIT)
        model = SMSDT(
            embed_dim=cfg.get("model", {}).get("embed_dim", 256),
            depth=cfg.get("model", {}).get("depth", 4),
            heads=cfg.get("model", {}).get("heads", 8),
            max_frames=16,
            grid=14,
        )
        # S-MSDT v1 was trained on [0, 1] tensors without ImageNet normalization (normalize=None)
        norm_fn = None
        model_type = "v1"

    incompat = model.load_state_dict(state_dict, strict=False)
    matched_keys = len(state_dict) - len(incompat.unexpected_keys)
    total_model_keys = len(model.state_dict())
    print(f"Weight loading report: {matched_keys}/{total_model_keys} parameters matched.")
    if len(incompat.missing_keys) > 0:
        print(f"  * Warning: {len(incompat.missing_keys)} missing keys in model!")
    if matched_keys == 0:
        raise RuntimeError("FATAL: 0 keys matched when loading checkpoint! Checkpoint dictionary did not contain model parameters.")

    model.to(device)
    model.eval()
    return model, model_type, norm_fn


def evaluate_dataset(model, ds, device, batch_size=4, eval_thresh=0.70):
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    video_ids, labels, methods = [], [], []
    video_scores_mean, video_scores_top50 = [], []
    clip_scores_all, clip_labels_all = [], []

    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            for batch in loader:
                frames = batch["frames"].to(device)  # (B, K, T, 3, H, W)
                y = batch["label"].cpu().numpy()
                vids = batch["video_id"]
                meths = batch["method"]

                B, K, T, C, H, W = frames.shape
                frames_flat = frames.view(B * K, T, C, H, W)

                logits, _ = model(frames_flat)
                probs = torch.sigmoid(logits).view(B, K).float().cpu().numpy()

                for b in range(B):
                    p_clips = probs[b]
                    p_mean = float(np.mean(p_clips))
                    # Top-50% clip score
                    k_top = max(1, K // 2)
                    p_top50 = float(np.mean(np.partition(p_clips, -k_top)[-k_top:]))

                    video_scores_mean.append(p_mean)
                    video_scores_top50.append(p_top50)
                    labels.append(float(y[b]))
                    video_ids.append(vids[b])
                    methods.append(meths[b])

                    clip_scores_all.extend(p_clips.tolist())
                    clip_labels_all.extend([float(y[b])] * K)

    labels = np.array(labels)
    scores_mean = np.array(video_scores_mean)
    scores_top50 = np.array(video_scores_top50)

    # Calculate overall metrics
    vid_auc_mean = float(roc_auc_score(labels, scores_mean))
    vid_auc_top50 = float(roc_auc_score(labels, scores_top50))
    clip_auc = float(roc_auc_score(clip_labels_all, clip_scores_all))
    eer, eer_thresh = compute_eer(labels, scores_mean)

    # Predictions at standard 0.5 threshold and at optimal EER threshold
    preds_05 = (scores_mean >= 0.5).astype(int)
    preds_eer = (scores_mean >= eer_thresh).astype(int)

    real_mask = (labels == 0.0)
    total_acc_05 = float(np.mean(preds_05 == labels))
    total_acc_eer = float(np.mean(preds_eer == labels))
    pristine_acc_05 = float(np.mean(preds_05[real_mask] == 0.0)) if real_mask.sum() > 0 else 0.0
    pristine_acc_eer = float(np.mean(preds_eer[real_mask] == 0.0)) if real_mask.sum() > 0 else 0.0

    # Breakdown per method (Accuracy & AUC)
    per_method_auc = {}
    per_method_acc_05 = {}
    per_method_acc_eer = {}

    methods_arr = np.array(methods)
    for m in sorted(set(methods)):
        if m == "real":
            continue
        m_mask = (methods_arr == m)
        eval_mask = real_mask | m_mask
        if eval_mask.sum() > 0 and len(set(labels[eval_mask])) > 1:
            per_method_auc[m] = float(roc_auc_score(labels[eval_mask], scores_mean[eval_mask]))
        if m_mask.sum() > 0:
            per_method_acc_05[m] = float(np.mean(preds_05[m_mask] == 1.0))
            per_method_acc_eer[m] = float(np.mean(preds_eer[m_mask] == 1.0))

    # Predictions at custom calibrated threshold
    if eval_thresh is not None:
        preds_custom = (scores_mean >= eval_thresh).astype(int)
        total_acc_custom = float(np.mean(preds_custom == labels))
        pristine_acc_custom = float(np.mean(preds_custom[real_mask] == 0.0)) if real_mask.sum() > 0 else 0.0
        per_method_acc_custom = {}
        for m in sorted(set(methods)):
            if m == "real":
                continue
            m_mask = (methods_arr == m)
            if m_mask.sum() > 0:
                per_method_acc_custom[m] = float(np.mean(preds_custom[m_mask] == 1.0))
    else:
        total_acc_custom = None
        pristine_acc_custom = None
        per_method_acc_custom = None

    return {
        "num_videos": len(labels),
        "num_reals": int((labels == 0).sum()),
        "num_fakes": int((labels == 1).sum()),
        "video_auc_mean": vid_auc_mean,
        "video_auc_top50": vid_auc_top50,
        "clip_auc": clip_auc,
        "eer": eer,
        "eer_thresh": eer_thresh,
        "eval_thresh": eval_thresh,
        "total_acc_05": total_acc_05,
        "total_acc_eer": total_acc_eer,
        "total_acc_custom": total_acc_custom,
        "pristine_acc_05": pristine_acc_05,
        "pristine_acc_eer": pristine_acc_eer,
        "pristine_acc_custom": pristine_acc_custom,
        "per_method_auc": per_method_auc,
        "per_method_acc_05": per_method_acc_05,
        "per_method_acc_eer": per_method_acc_eer,
        "per_method_acc_custom": per_method_acc_custom,
        "scores_mean": scores_mean,
        "labels": labels,
        "methods": methods_arr,
    }


def print_threshold_sweep(res, dataset_name, thresholds=None):
    """Evaluates an array of decision thresholds to find the optimal operating boundary."""
    if thresholds is None:
        thresholds = [0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]

    eer_th = res.get("eer_thresh")
    all_th = sorted(set([float(t) for t in thresholds] + ([round(float(eer_th), 3)] if eer_th is not None else [])))

    labels = res["labels"]
    scores = res["scores_mean"]
    methods = res["methods"]
    real_mask = (labels == 0.0)

    standard_order = [
        "Deepfakes",
        "Face2Face",
        "FaceSwap",
        "NeuralTextures",
        "FaceShifter",
        "DeepFakeDetection",
        "DFD",
    ]
    all_methods = res.get("per_method_acc_05", {})
    cols = [m for m in standard_order if m in all_methods]
    cols += [m for m in sorted(all_methods.keys()) if m not in cols]

    display_names = {
        "DeepFakeDetection": "DFD",
        "FaceShifter": "FaceShifter",
    }
    header_labels = [display_names.get(m, m) for m in cols]
    headers = ["Threshold"] + header_labels + ["Pristine", "Total Acc", "Balanced"]
    col_widths = [max(len(h) + 2, 12) for h in headers]
    col_widths[0] = 16

    def fmt_row(items):
        return " | ".join(f"{str(item):>{w}}" for item, w in zip(items, col_widths))

    line_sep = "-" * (sum(col_widths) + 3 * (len(headers) - 1))
    double_sep = "=" * len(line_sep)

    print("\n" + double_sep)
    print(f"THRESHOLD SENSITIVITY SWEEP: {dataset_name} (Finding Optimal Operating Cutoff)")
    print(double_sep)
    print(fmt_row(headers))
    print(line_sep)

    best_acc = 0.0
    best_acc_th = 0.5
    best_bal = 0.0
    best_bal_th = 0.5

    for th in all_th:
        preds = (scores >= th).astype(int)
        total_acc = float(np.mean(preds == labels))
        pristine_acc = float(np.mean(preds[real_mask] == 0.0)) if real_mask.sum() > 0 else 0.0
        fake_acc = float(np.mean(preds[~real_mask] == 1.0)) if (~real_mask).sum() > 0 else 0.0
        bal_acc = float(0.5 * (pristine_acc + fake_acc))

        if total_acc > best_acc:
            best_acc = total_acc
            best_acc_th = th
        if bal_acc > best_bal:
            best_bal = bal_acc
            best_bal_th = th

        m_vals = []
        for m in cols:
            m_mask = (methods == m)
            if m_mask.sum() > 0:
                m_vals.append(f"{np.mean(preds[m_mask] == 1.0):.3f}")
            else:
                m_vals.append("N/A")

        is_eer = (eer_th is not None and abs(th - eer_th) < 1e-3)
        th_str = f"{th:.3f} (EER)" if is_eer else f"{th:.3f}"
        print(fmt_row([th_str] + m_vals + [f"{pristine_acc:.3f}", f"{total_acc:.3f}", f"{bal_acc:.3f}"]))

    print(line_sep)
    print(f"  * Peak Total Accuracy:     {best_acc:.3f} at threshold = {best_acc_th:.3f}")
    print(f"  * Peak Balanced Accuracy:  {best_bal:.3f} at threshold = {best_bal_th:.3f}")
    if eer_th is not None:
        print(f"  * Optimal EER Threshold:   {eer_th:.3f}")
    print(double_sep + "\n")


def print_benchmark_table(res, dataset_name, split):
    """Prints the benchmark table in the exact FaceForensics++ format."""
    standard_order = [
        "Deepfakes",
        "Face2Face",
        "FaceSwap",
        "NeuralTextures",
        "FaceShifter",
        "DeepFakeDetection",
        "DFD",
    ]
    all_methods = res.get("per_method_acc_05", {})
    cols = [m for m in standard_order if m in all_methods]
    cols += [m for m in sorted(all_methods.keys()) if m not in cols]

    display_names = {
        "DeepFakeDetection": "DFD",
        "FaceShifter": "FaceShifter",
    }
    header_labels = [display_names.get(m, m) for m in cols]
    headers = ["Info"] + header_labels + ["Pristine", "Total"]
    col_widths = [max(len(h) + 2, 12) for h in headers]
    col_widths[0] = 16

    def fmt_row(items):
        return " | ".join(f"{str(item):>{w}}" for item, w in zip(items, col_widths))

    line_sep = "-" * (sum(col_widths) + 3 * (len(headers) - 1))
    double_sep = "=" * len(line_sep)

    print("\n" + double_sep)
    print(f"BENCHMARK RESULTS TABLE: {dataset_name} [{split.upper()}] (N={res['num_videos']} videos)")
    print(double_sep)
    print(fmt_row(headers))
    print(line_sep)

    # Row 1: Accuracy @ 0.5
    acc_05_vals = [f"{res['per_method_acc_05'].get(m, 0.0):.3f}" for m in cols]
    print(fmt_row(["Accuracy@0.5"] + acc_05_vals + [f"{res['pristine_acc_05']:.3f}", f"{res['total_acc_05']:.3f}"]))

    # Row 2: Accuracy @ Calibrated Threshold (if specified)
    if res.get("eval_thresh") is not None and res.get("per_method_acc_custom") is not None:
        acc_custom_vals = [f"{res['per_method_acc_custom'].get(m, 0.0):.3f}" for m in cols]
        custom_lbl = f"Accuracy@{res['eval_thresh']:.2f}"
        print(fmt_row([custom_lbl] + acc_custom_vals + [f"{res['pristine_acc_custom']:.3f}", f"{res['total_acc_custom']:.3f}"]))

    # Row 3: Accuracy @ EER
    acc_eer_vals = [f"{res['per_method_acc_eer'].get(m, 0.0):.3f}" for m in cols]
    eer_lbl = f"Acc@EER({res['eer_thresh']:.2f})" if "eer_thresh" in res else "Accuracy@EER"
    print(fmt_row([eer_lbl] + acc_eer_vals + [f"{res['pristine_acc_eer']:.3f}", f"{res['total_acc_eer']:.3f}"]))

    # Row 4: Video AUC
    auc_vals = [f"{res['per_method_auc'].get(m, 0.0):.3f}" for m in cols]
    print(fmt_row(["Video AUC"] + auc_vals + ["1.000", f"{res['video_auc_mean']:.3f}"]))

    print(double_sep + "\n")


def print_classification_report(res, thresh=None):
    """Prints a detailed confusion matrix and performance breakdown at a given threshold."""
    if thresh is None:
        thresh = res.get("eval_thresh", 0.70)
    scores = res["scores_mean"]
    labels = res["labels"]
    preds = (scores >= thresh).astype(int)

    tp = int(np.sum((preds == 1) & (labels == 1)))
    tn = int(np.sum((preds == 0) & (labels == 0)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))

    n_fakes = tp + fn
    n_reals = tn + fp
    total = len(labels)

    acc = (tp + tn) / total if total > 0 else 0.0
    sens = tp / n_fakes if n_fakes > 0 else 0.0
    spec = tn / n_reals if n_reals > 0 else 0.0
    bal_acc = 0.5 * (sens + spec)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0.0

    print("=" * 85)
    print(f"CALIBRATED CLASSIFICATION REPORT (Threshold = {thresh:.3f})")
    print("=" * 85)
    print(f"  * True Positives  (Fakes Correctly Caught):  {tp:4d} / {n_fakes:4d} ({sens*100:.1f}%)")
    print(f"  * True Negatives  (Reals Correctly Cleared): {tn:4d} / {n_reals:4d} ({spec*100:.1f}%)")
    print(f"  * False Positives (False Alarms on Reals):   {fp:4d} / {n_reals:4d} ({fp/n_reals*100:.1f}%)")
    print(f"  * False Negatives (Fakes Missed):            {fn:4d} / {n_fakes:4d} ({fn/n_fakes*100:.1f}%)")
    print("-" * 85)
    print(f"  * True Overall Accuracy [(TP+TN)/All]:      {acc*100:.2f}% ({tp+tn}/{total} videos)")
    print(f"  * Balanced Accuracy [Mean(Sens, Spec)]:      {bal_acc*100:.2f}%")
    print(f"  * Precision: {prec*100:.1f}% | Recall: {sens*100:.1f}% | F1-Score: {f1:.4f}")
    print("=" * 85 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate deepfake model across datasets")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint (.pt)")
    parser.add_argument("--index", type=str, default="data/splits/index.csv", help="Split index manifest")
    parser.add_argument("--store", type=str, default="data/store", help="Offline frame store directory")
    parser.add_argument("--split", type=str, default="test", help="Split name to evaluate (default: test)")
    parser.add_argument("--methods", nargs="*", default=None, help="Optional subset of methods to evaluate (e.g. Deepfakes Face2Face FaceSwap NeuralTextures DFD)")
    parser.add_argument("--eval-thresh", type=float, default=0.70, help="Calibrated threshold to evaluate (default: 0.70)")
    parser.add_argument("--thresholds", nargs="*", type=float, default=None, help="Custom threshold array for sensitivity sweep")
    parser.add_argument("--dataset-name", type=str, default="FF++ c23", help="Dataset display name")
    parser.add_argument("--out-csv", type=str, default="outputs/benchmark_results.csv", help="Output CSV path")
    parser.add_argument("--K", type=int, default=8, help="Clips per video")
    parser.add_argument("--T", type=int, default=8, help="Frames per clip")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint: {args.ckpt} on {device}")
    model, model_type, norm_fn = load_model(args.ckpt, device)
    print(f"Loaded model architecture: {model_type.upper()}")

    ds = EvalClipDataset(
        index_csv=args.index,
        store_root=args.store,
        split=args.split,
        K=args.K,
        T=args.T,
        stride=1,
        normalize=norm_fn,
        methods=args.methods,
    )
    print(f"Evaluating {args.dataset_name} ({args.split} split) with {len(ds)} videos...")
    res = evaluate_dataset(model, ds, device, eval_thresh=args.eval_thresh)

    # Render formatted benchmark results table
    print_benchmark_table(res, args.dataset_name, args.split)

    # Render classification report at calibrated threshold
    if args.eval_thresh is not None:
        print_classification_report(res, thresh=args.eval_thresh)

    # Render threshold sensitivity sweep table
    print_threshold_sweep(res, f"{args.dataset_name} [{args.split.upper()}]", thresholds=args.thresholds)

    # Append to output CSV
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    file_exists = os.path.exists(args.out_csv)
    with open(args.out_csv, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["checkpoint", "model_type", "dataset", "split", "videos", "reals", "fakes", "video_auc_mean", "video_auc_top50", "clip_auc", "eer", "acc_05", "acc_eer", "pristine_acc", "per_method_auc_json", "per_method_acc_json"])
        writer.writerow([
            os.path.basename(args.ckpt),
            model_type,
            args.dataset_name,
            args.split,
            res["num_videos"],
            res["num_reals"],
            res["num_fakes"],
            f"{res['video_auc_mean']:.4f}",
            f"{res['video_auc_top50']:.4f}",
            f"{res['clip_auc']:.4f}",
            f"{res['eer']:.4f}",
            f"{res['total_acc_05']:.4f}",
            f"{res['total_acc_eer']:.4f}",
            f"{res['pristine_acc_05']:.4f}",
            json.dumps(res["per_method_auc"]),
            json.dumps(res["per_method_acc_05"]),
        ])
    print(f"Appended results row to {args.out_csv}")


if __name__ == "__main__":
    main()
