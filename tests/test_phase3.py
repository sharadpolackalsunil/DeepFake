"""Unit tests for Phase 3: Cross-dataset evaluation metrics, perturbation filters, and diagnostics."""
import numpy as np
import pytest
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
