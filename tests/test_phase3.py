"""Unit tests for Phase 3: Cross-dataset evaluation metrics, perturbation filters, and diagnostics."""
import os
import sys

# Ensure repository root is on sys.path for direct execution
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from smsdt.eval_cross import compute_eer
from tools.eval_robustness import make_jpeg_filter, make_blur_filter, make_rescale_filter


def test_compute_eer_exact():
    """Verify compute_eer on known synthetic distributions."""
    # Perfect separation: y_scores are 0 for reals, 1 for fakes
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_scores = np.array([0.1, 0.2, 0.15, 0.25, 0.8, 0.85, 0.9, 0.95])
    eer, thresh = compute_eer(y_true, y_scores)
    assert eer == 0.0, f"Expected 0.0 EER on separated data, got {eer}"
    assert 0.25 <= thresh <= 0.80

    # Symmetric overlap around 0.5
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_scores = np.array([0.2, 0.4, 0.6, 0.8, 0.2, 0.4, 0.6, 0.8])
    eer, thresh = compute_eer(y_true, y_scores)
    assert abs(eer - 0.5) < 0.15


def test_robustness_perturbation_filters():
    """Verify that JPEG, blur, and rescale filters run and return identical shape uint8 arrays."""
    img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)

    # 1. JPEG Filter
    jpeg_fn = make_jpeg_filter(50)
    p_jpeg = jpeg_fn(img)
    assert p_jpeg.shape == img.shape
    assert p_jpeg.dtype == np.uint8

    # 2. Gaussian Blur Filter
    blur_fn = make_blur_filter(1.5)
    p_blur = blur_fn(img)
    assert p_blur.shape == img.shape
    assert p_blur.dtype == np.uint8

    # 3. Rescale Filter
    rescale_fn = make_rescale_filter(0.5)
    p_rescale = rescale_fn(img)
    assert p_rescale.shape == img.shape
    assert p_rescale.dtype == np.uint8


def test_eval_clip_aggregation_shapes():
    """Verify clip aggregation arithmetic on synthetic predictions."""
    B, K = 4, 8
    probs = torch.tensor([
        [0.1, 0.2, 0.15, 0.3, 0.1, 0.2, 0.25, 0.1],  # real video
        [0.8, 0.9, 0.85, 0.95, 0.7, 0.8, 0.9, 0.85], # fake video
        [0.2, 0.3, 0.7, 0.8, 0.2, 0.3, 0.2, 0.1],   # local fake (top-50% should capture)
        [0.05, 0.05, 0.1, 0.05, 0.02, 0.03, 0.04, 0.01],
    ])

    mean_score = probs.mean(dim=1)
    assert mean_score.shape == (B,)
    assert mean_score[1] > mean_score[0]

    k_top = K // 2
    top50_score = probs.topk(k=k_top, dim=1).values.mean(dim=1)
    assert top50_score.shape == (B,)
    # Top-50% on sample 2 should be noticeably higher than simple mean
    assert top50_score[2] > mean_score[2]


def test_eval_clip_dataset_dfd_isolation():
    """Verify that EvalClipDataset with methods=['DeepFakeDetection'] selects ONLY DFD fakes."""
    from collections import Counter
    from smsdt.data.video_dataset import EvalClipDataset

    ds_dfd = EvalClipDataset(
        index_csv="data/splits/index.csv",
        store_root="data/store",
        split="test",
        methods=["DeepFakeDetection"],
    )

    # 1. Verify exactly 2,987 videos
    assert len(ds_dfd) == 2987, f"Expected 2987 DFD test videos, got {len(ds_dfd)}"

    counts = Counter(r["method"] for r in ds_dfd.rows)
    # 2. Verify ONLY DeepFakeDetection and real exist
    assert set(counts.keys()) == {"DeepFakeDetection", "real"}
    assert counts["DeepFakeDetection"] == 2604
    assert counts["real"] == 383

    # 3. Explicitly check that NO FF++ fakes leaked in
    for forbidden in ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "FaceShifter"]:
        assert forbidden not in counts, f"Found leaked method {forbidden} in DFD dataset!"

    # 4. In reverse, verify that pure FF++ evaluation contains ZERO DFD fakes or DFD reals
    ds_ffpp = EvalClipDataset(
        index_csv="data/splits/index.csv",
        store_root="data/store",
        split="test",
        methods=["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "FaceShifter"],
    )
    # Pure FF++ has 700 fakes (140 x 5 methods) + 140 pure YouTube reals = exactly 840 videos
    assert len(ds_ffpp) == 840
    ffpp_counts = Counter(r["method"] for r in ds_ffpp.rows)
    assert "DeepFakeDetection" not in ffpp_counts
    assert "DFD" not in ffpp_counts


if __name__ == "__main__":
    print("Running Phase 3 tests directly...")
    test_compute_eer_exact()
    print("  * test_compute_eer_exact: PASSED")
    test_robustness_perturbation_filters()
    print("  * test_robustness_perturbation_filters: PASSED")
    test_eval_clip_aggregation_shapes()
    print("  * test_eval_clip_aggregation_shapes: PASSED")
    test_eval_clip_dataset_dfd_isolation()
    print("  * test_eval_clip_dataset_dfd_isolation: PASSED")
    print("\nAll 4 tests in test_phase3.py passed successfully!")
