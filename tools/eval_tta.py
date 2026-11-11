"""Test-Time Adaptation (TTA) Benchmark Runner for S-MSDT TMLR Study.

Evaluates:
  1. Static (Pre-TTA) Video AUC baseline
  2. Online Streaming LayerNorm Entropy Minimization (TENT-style)
  3. Batch-Episodic LayerNorm Entropy Minimization (resets after each batch)
  4. Collapse Bounds Audit (|ΔAUC| > 0.03 or degenerate positive fraction >= 95% / <= 5%)

Usage:
  python -m tools.eval_tta --ckpt outputs/checkpoints_smsdt_full/seed1/best_ema.pt \
      --index data/splits/celebdf_test.csv --store data/store_celebdf --split test \
      --dataset-name "Celeb-DF-v2" --mode both
"""
import argparse
import collections
import copy
import json
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from smsdt.data.video_dataset import EvalClipDataset
from smsdt.eval_cross import load_model, compute_eer
from smsdt.models.smsdt_v3 import SMSDTv3
from smsdt.tta import TTAAdapter, binary_entropy_loss


def evaluate_static(model, loader, device, is_v3: bool):
    """Run clean unadapted inference across test loader."""
    model.eval()
    video_scores, video_labels, video_ids = [], [], []

    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            for batch in loader:
                frames = batch["frames"].to(device)  # (B, K, T, C, H, W)
                B, K, T, C, H, W = frames.shape
                frames_flat = frames.view(B * K, T, C, H, W)
                prior = batch.get("prior", None)
                prior_flat = prior.to(device).view(B * K, T, -1) if prior is not None else None

                if is_v3 and prior_flat is not None:
                    out = model(frames_flat, prior=prior_flat)
                else:
                    out = model(frames_flat)

                logits = out[0] if isinstance(out, (tuple, list)) else out
                probs = torch.sigmoid(logits).view(B, K).float().cpu().numpy()
                p_mean = probs.mean(axis=1)

                video_scores.extend(p_mean.tolist())
                video_labels.extend(batch["label"].cpu().numpy().tolist())
                video_ids.extend(batch["video_id"])

    auc = float(roc_auc_score(video_labels, video_scores))
    return auc, np.array(video_scores), np.array(video_labels)


def run_tta_stream(
    adapter: TTAAdapter,
    loader: DataLoader,
    device: torch.device,
    is_v3: bool,
    mode: str = "online",
) -> dict:
    """Run TTA adaptation across the evaluation stream."""
    adapter.reset()
    adapter.mode = mode

    video_scores, video_labels = [], []
    batch_losses = []

    total_batches = len(loader)
    for batch_idx, batch in enumerate(loader):
        frames = batch["frames"].to(device)  # (B, K, T, C, H, W)
        B, K, T, C, H, W = frames.shape
        frames_flat = frames.view(B * K, T, C, H, W)
        prior = batch.get("prior", None)
        prior_flat = prior.to(device).view(B * K, T, -1) if prior is not None else None

        # Autocast during forward and backward with memory-safe chunking
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            logits, loss_val = adapter.adapt_batch(frames_flat, prior=prior_flat, max_chunk=4)

        probs = torch.sigmoid(logits).view(B, K).float().cpu().numpy()
        p_mean = probs.mean(axis=1)

        video_scores.extend(p_mean.tolist())
        video_labels.extend(batch["label"].cpu().numpy().tolist())
        batch_losses.append(loss_val)

        if (batch_idx + 1) % 25 == 0 or (batch_idx + 1) == total_batches:
            curr_entropy = float(np.mean(batch_losses[-25:]))
            print(f"  [{mode.upper()}] Step {batch_idx + 1}/{total_batches} ({((batch_idx + 1)/total_batches)*100:.1f}%) | Entropy Loss: {curr_entropy:.4f}")

        if mode == "batch_episodic":
            adapter.reset()

    # Final reset to restore source state
    adapter.reset()

    scores_arr = np.array(video_scores)
    labels_arr = np.array(video_labels)

    auc = float(roc_auc_score(labels_arr, scores_arr))
    mean_entropy = float(np.mean(batch_losses))
    pred_pos_fraction = float(np.mean(scores_arr >= 0.50))
    true_pos_fraction = float(np.mean(labels_arr == 1.0))

    return {
        "post_auc": auc,
        "mean_entropy": mean_entropy,
        "pred_pos_fraction": pred_pos_fraction,
        "true_pos_fraction": true_pos_fraction,
        "scores": scores_arr,
        "labels": labels_arr,
    }


def main():
    parser = argparse.ArgumentParser(description="Test-Time Adaptation (TTA) Runner for S-MSDT")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--index", type=str, required=True, help="Path to index CSV")
    parser.add_argument("--store", type=str, required=True, help="Path to frame store directory")
    parser.add_argument("--split", type=str, default="test", help="Dataset split")
    parser.add_argument("--dataset-name", type=str, default="Celeb-DF-v2", help="Dataset display name")
    parser.add_argument("--methods", nargs="*", default=None, help="Optional subset of methods")
    parser.add_argument("--K", type=int, default=4, help="Clips per video (default 4)")
    parser.add_argument("--T", type=int, default=None, help="Frames per clip (auto-detected if None)")
    parser.add_argument("--batch-size", type=int, default=2, help="Test batch size (videos per step, default 2)")
    parser.add_argument("--lr", type=float, default=1e-4, help="TTA LayerNorm learning rate (default 1e-4)")
    parser.add_argument("--mode", type=str, default="both", choices=["online", "batch_episodic", "both"],
                        help="TTA mode: online, batch_episodic, or both")
    parser.add_argument("--output-csv", type=str, default="results/tables/tta_summary.csv")
    parser.add_argument("--output-md", type=str, default="results/tables/tta_summary.md")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*80}")
    print(f"TEST-TIME ADAPTATION (TTA) BENCHMARK: {args.dataset_name}")
    print(f"Checkpoint: {args.ckpt}")
    print(f"Device: {device} | TTA LR: {args.lr} | Batch Size: {args.batch_size} videos")
    print(f"{'='*80}\n")

    model, model_type, norm_fn = load_model(args.ckpt, device)
    is_v3 = isinstance(model, SMSDTv3) or getattr(model, "has_prior", hasattr(model, "readout") and hasattr(model.readout, "prior_gain"))

    # Auto-detect T
    cfg = getattr(model, "cfg", {})
    if args.T is None:
        if cfg.get("stage") == "stage_a" or cfg.get("data", {}).get("frames_per_clip") == 1:
            actual_T = 1
        else:
            actual_T = int(cfg.get("data", {}).get("frames_per_clip", 8))
    else:
        actual_T = args.T

    ds = EvalClipDataset(
        index_csv=args.index,
        store_root=args.store,
        split=args.split,
        K=args.K,
        T=actual_T,
        stride=1,
        normalize=norm_fn,
        methods=args.methods,
    )
    print(f"Loaded {len(ds):,} evaluation videos (K={args.K}, T={actual_T}).")

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # 1. Baseline Static Evaluation
    print("\n--- 1. Evaluating Static Baseline (Unadapted Source Model) ---")
    pre_auc, static_scores, static_labels = evaluate_static(model, loader, device, is_v3)
    print(f"  * Static Pre-TTA Video AUC: {pre_auc:.4f}")

    adapter = TTAAdapter(model=model, lr=args.lr, mode="online")
    print(f"  * Active LayerNorm adaptation parameters: {len(adapter.ln_params)} tensors")

    modes_to_test = ["online", "batch_episodic"] if args.mode == "both" else [args.mode]
    results_list = []

    model_name = os.path.basename(os.path.dirname(args.ckpt))
    if model_name.startswith("seed"):
        parent_name = os.path.basename(os.path.dirname(os.path.dirname(args.ckpt))).replace("checkpoints_", "")
        run_identifier = f"{parent_name}_{model_name}"
    else:
        run_identifier = os.path.splitext(os.path.basename(args.ckpt))[0]

    for mode in modes_to_test:
        print(f"\n--- Running TTA ({mode.upper()}) ---")
        tta_res = run_tta_stream(adapter, loader, device, is_v3, mode=mode)
        post_auc = tta_res["post_auc"]
        delta_auc = post_auc - pre_auc

        # Collapse check criteria:
        # 1. AUC drops by more than 0.03
        auc_drop = (pre_auc - post_auc) > 0.03
        # 2. At least 95% of test videos predicted as one class while true share is <= 85%
        pred_pos = tta_res["pred_pos_fraction"]
        true_pos = tta_res["true_pos_fraction"]
        degenerate = (pred_pos >= 0.95 or pred_pos <= 0.05) and (0.15 <= true_pos <= 0.85)
        collapsed = bool(auc_drop or degenerate)

        print(f"  * Post-TTA AUC:        {post_auc:.4f} (ΔAUC: {delta_auc:+.4f})")
        print(f"  * Mean Binary Entropy: {tta_res['mean_entropy']:.4f}")
        print(f"  * Pred Positive Ratio: {pred_pos*100:.1f}% (Ground truth: {true_pos*100:.1f}%)")
        print(f"  * Collapse Detected:   {'YES (FAILED)' if collapsed else 'NO (STABLE)'}")

        results_list.append({
            "Model": run_identifier,
            "Dataset": args.dataset_name,
            "Mode": mode,
            "Pre_AUC": round(pre_auc, 4),
            "Post_AUC": round(post_auc, 4),
            "Delta_AUC": round(delta_auc, 4),
            "Mean_Entropy": round(tta_res["mean_entropy"], 4),
            "Pred_Pos_Pct": round(pred_pos * 100, 1),
            "True_Pos_Pct": round(true_pos * 100, 1),
            "Collapsed": collapsed,
        })

    # Save summary tables
    df = pd.DataFrame(results_list)
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    if os.path.exists(args.output_csv):
        existing_df = pd.read_csv(args.output_csv)
        combined_df = pd.concat([existing_df, df], ignore_index=True).drop_duplicates(
            subset=["Model", "Dataset", "Mode"], keep="last"
        )
    else:
        combined_df = df

    combined_df.to_csv(args.output_csv, index=False)

    # Output formatted markdown
    def df_to_markdown(dataframe: pd.DataFrame) -> str:
        headers = list(dataframe.columns)
        lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
        for _, row in dataframe.iterrows():
            lines.append("| " + " | ".join(str(val) for val in row) + " |")
        return "\n".join(lines)

    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write("# Test-Time Adaptation (TTA) Benchmark Summary\n\n")
        f.write("*Evaluated with pre-registered LayerNorm entropy minimization (lr=1e-4)*\n\n")
        f.write(df_to_markdown(combined_df) + "\n")

    print(f"\n{'='*80}")
    print("TTA BENCHMARK RESULTS TABLE")
    print(f"{'='*80}")
    print(df_to_markdown(df))
    print(f"\nSaved updated summary to {args.output_csv} and {args.output_md}\n")


if __name__ == "__main__":
    main()

