"""Diagnostic D4: Domain Shift and Pristine Real False Positive Rate (FPR) Auditor.

Evaluates pristine real faces in the test split separately across domains:
  1. YouTube Reals (official FaceForensics++)
  2. Actors Reals (DeepFakeDetection / Google-Jigsaw)

A large FPR disparity reveals domain shift, explaining high false alarm rates on pristine actors.
"""
import argparse
import sys
import numpy as np
import torch
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader

from smsdt.eval_cross import load_model
from smsdt.data.video_dataset import EvalClipDataset, load_index


def main():
    parser = argparse.ArgumentParser(description="Audit domain shift and FPR between YouTube and Actor reals")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--index", type=str, default="data/splits/index.csv")
    parser.add_argument("--store", type=str, default="data/store")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_type, norm_fn = load_model(args.ckpt, device)

    # Filter for pristine real videos in test split
    rows = [r for r in load_index(args.index, args.split) if r["method"] == "real"]
    print(f"Loaded {len(rows)} real videos from split '{args.split}'")

    ds = EvalClipDataset(args.index, args.store, split=args.split, K=4, T=8, normalize=norm_fn)
    # Map video_id to dataset index
    vid_to_idx = {r["video_id"]: i for i, r in enumerate(ds.rows)}

    scores_by_domain = {"youtube": [], "actors": []}
    model.eval()

    with torch.no_grad():
        for r in rows:
            vid = r["video_id"]
            if vid not in vid_to_idx:
                continue
            idx = vid_to_idx[vid]
            item = ds[idx]
            frames = item["frames"].unsqueeze(0).to(device)  # (1, K, T, 3, H, W)
            B, K, T, C, H, W = frames.shape
            logits, _ = model(frames.view(B * K, T, C, H, W))
            p_vid = float(torch.sigmoid(logits).mean().cpu().numpy())

            domain = "actors" if "actor" in r.get("path", "").lower() or "actors" in vid.lower() else "youtube"
            scores_by_domain[domain].append(p_vid)

    print("\n" + "=" * 60)
    print(f"DOMAIN SHIFT AUDIT RESULTS [{args.split.upper()} REALS]")
    print("=" * 60)
    for domain, scs in scores_by_domain.items():
        if not scs:
            continue
        arr = np.array(scs)
        fpr = float((arr >= args.threshold).mean())
        mean_score = float(arr.mean())
        median_score = float(np.median(arr))
        print(f"Domain [{domain.upper():8s}] (N={len(arr)}):")
        print(f"  * False Positive Rate @ {args.threshold:.2f}: {fpr * 100:.2f}% ({int((arr >= args.threshold).sum())}/{len(arr)})")
        print(f"  * Mean Fake Score:           {mean_score:.4f}")
        print(f"  * Median Fake Score:         {median_score:.4f}")

    if scores_by_domain["youtube"] and scores_by_domain["actors"]:
        gap = abs(np.mean(scores_by_domain["actors"]) - np.mean(scores_by_domain["youtube"]))
        print("-" * 60)
        print(f"Score Gap Between Domains: {gap:.4f}")
        if gap > 0.15:
            print("[FINDING] Significant domain shift detected! Actors distribution diverges from YouTube reals.")
        else:
            print("[FINDING] Domain shift is mild across pristine face sources.")
    print("=" * 60)


if __name__ == "__main__":
    main()
