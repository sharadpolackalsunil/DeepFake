"""Robustness perturbation benchmark for deepfake detection models.

Evaluates models under standard distribution shifts:
  1. JPEG Compression: Q in {90, 70, 50, 30}
  2. Gaussian Blur: sigma in {0.5, 1.0, 2.0}
  3. Downscaling/Rescaling: scale in {0.75, 0.50, 0.25}
"""
import argparse
import copy
import os
import cv2
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

from smsdt.eval_cross import load_model, compute_eer
from smsdt.data.video_dataset import VideoStore, contiguous_starts


class PerturbedEvalDataset(Dataset):
    """Wrapper that applies a specific visual perturbation to evaluation frames."""

    def __init__(self, base_ds, perturbation_fn=None):
        self.base_ds = base_ds
        self.perturbation_fn = perturbation_fn

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        item = self.base_ds[idx]
        if self.perturbation_fn is None:
            return item

        # frames is (K, T, 3, H, W)
        frames = item["frames"]
        K, T, C, H, W = frames.shape

        # Apply perturbation in numpy space
        perturbed_clips = []
        for k in range(K):
            clip = frames[k].permute(0, 2, 3, 1).cpu().numpy()  # (T, H, W, 3) in [0, 1]
            perturbed_frames = []
            for t in range(T):
                img_u8 = np.clip(clip[t] * 255.0, 0, 255).astype(np.uint8)
                p_img = self.perturbation_fn(img_u8)
                perturbed_frames.append(p_img.astype(np.float32) / 255.0)
            p_clip = torch.from_numpy(np.stack(perturbed_frames)).permute(0, 3, 1, 2)
            perturbed_clips.append(p_clip)

        item["frames"] = torch.stack(perturbed_clips)
        return item


def make_jpeg_filter(quality: int):
    def _jpeg(img_u8: np.ndarray) -> np.ndarray:
        ok, enc = cv2.imencode(".jpg", img_u8[..., ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        if ok:
            return cv2.imdecode(enc, cv2.IMREAD_COLOR)[..., ::-1]
        return img_u8
    return _jpeg


def make_blur_filter(sigma: float):
    ksize = int(round(sigma * 4)) | 1  # odd kernel size
    def _blur(img_u8: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(img_u8, (ksize, ksize), sigma)
    return _blur


def make_rescale_filter(scale: float):
    def _rescale(img_u8: np.ndarray) -> np.ndarray:
        h, w = img_u8.shape[:2]
        new_h, new_w = max(16, int(round(h * scale))), max(16, int(round(w * scale)))
        down = cv2.resize(img_u8, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(down, (w, h), interpolation=cv2.INTER_LINEAR)
    return _rescale


def evaluate_loader(model, loader, device):
    model.eval()
    scores, labels = [], []
    with torch.no_grad():
        for batch in loader:
            frames = batch["frames"].to(device)
            B, K, T, C, H, W = frames.shape
            logits, _ = model(frames.view(B * K, T, C, H, W))
            probs = torch.sigmoid(logits).view(B, K).mean(dim=1).cpu().numpy()
            scores.extend(probs.tolist())
            labels.extend(batch["label"].cpu().numpy().tolist())

    if len(set(labels)) < 2:
        return 0.5
    return float(roc_auc_score(labels, scores))


def main():
    parser = argparse.ArgumentParser(description="Evaluate model under robustness perturbations")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--index", type=str, default="data/splits/index.csv")
    parser.add_argument("--store", type=str, default="data/store")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max-videos", type=int, default=100, help="Subset for fast curve profiling")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_type, norm_fn = load_model(args.ckpt, device)

    from smsdt.data.video_dataset import EvalClipDataset
    base_ds = EvalClipDataset(args.index, args.store, split=args.split, K=4, T=8, normalize=norm_fn)

    # Clean unperturbed baseline
    loader = DataLoader(base_ds, batch_size=4, shuffle=False)
    clean_auc = evaluate_loader(model, loader, device)
    print(f"Clean Baseline AUC: {clean_auc:.4f}")

    perturbations = [
        ("JPEG Quality", [90, 70, 50, 30], make_jpeg_filter),
        ("Gaussian Blur (sigma)", [0.5, 1.0, 2.0], make_blur_filter),
        ("Downscale Factor", [0.75, 0.50, 0.25], make_rescale_filter),
    ]

    results = {"clean": clean_auc}
    for p_name, levels, fn_builder in perturbations:
        print(f"\n--- Testing {p_name} ---")
        for lvl in levels:
            p_fn = fn_builder(lvl)
            p_ds = PerturbedEvalDataset(base_ds, p_fn)
            p_loader = DataLoader(p_ds, batch_size=4, shuffle=False)
            auc = evaluate_loader(model, p_loader, device)
            print(f"  Level {lvl}: AUC = {auc:.4f} (drop: {clean_auc - auc:+.4f})")
            results[f"{p_name}_{lvl}"] = auc

    print("\nRobustness Summary:")
    print(results)


if __name__ == "__main__":
    main()
