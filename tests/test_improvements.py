"""
Unit tests for accuracy, generalization, speed, and evaluation improvements.
"""
import pytest
import numpy as np
import torch

from smsdt.losses import focal_bce, alignment_loss, supervised_contrastive_loss, total_loss
from smsdt.models.wavelet_branch import WaveletBranch
from smsdt.eval import compute_metrics, evaluate


def test_focal_loss_balanced_weighting_and_smoothing():
    """Verify focal BCE alpha=0.2 balances 4:1 fake:real ratio and supports label smoothing."""
    # 4 fake samples (1.0) and 1 real sample (0.0)
    logits = torch.zeros(5)
    targets = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0])

    # With alpha=0.2, effective fake weight is 4 * 0.2 = 0.8, real weight is 1 * (1 - 0.2) = 0.8 (balanced!)
    loss_balanced = focal_bce(logits, targets, alpha=0.2, gamma=2.0)
    assert not torch.isnan(loss_balanced)
    assert loss_balanced.item() > 0

    # With label smoothing
    loss_smoothed = focal_bce(logits, targets, alpha=0.2, gamma=2.0, label_smoothing=0.05)
    assert not torch.isnan(loss_smoothed)
    assert loss_smoothed.item() > 0


def test_alignment_loss_decorrelation():
    """Verify alignment loss in decorrelation mode penalizes correlation between RGB and wavelet."""
    # Highly correlated representations
    z_rgb = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    z_dwt_corr = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # identical
    z_dwt_orth = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])  # orthogonal

    loss_corr = alignment_loss(z_rgb, z_dwt_corr, mode="decorrelate")
    loss_orth = alignment_loss(z_rgb, z_dwt_orth, mode="decorrelate")

    # Decorrelation loss should be 1.0 for identical vectors and 0.0 for orthogonal vectors
    assert loss_corr.item() > loss_orth.item()
    assert torch.isclose(loss_orth, torch.tensor(0.0), atol=1e-5)


def test_contrastive_loss_small_batch_fallback():
    """Verify supervised contrastive loss handles small batches and falls back gracefully."""
    # Batch with only 2 samples
    features = torch.randn(2, 64)
    labels = torch.tensor([0.0, 1.0])
    loss = supervised_contrastive_loss(features, labels)
    assert not torch.isnan(loss)

    # Single sample batch -> returns 0.0 without crash
    single_feat = torch.randn(1, 64)
    single_label = torch.tensor([1.0])
    loss_single = supervised_contrastive_loss(single_feat, single_label)
    assert loss_single.item() == 0.0


def test_wavelet_branch_strided_patch_stem_and_energy():
    """Verify WaveletBranch with strided patch stem directly outputs (B, embed_dim, 7, 7)."""
    B = 2
    embed_dim = 128
    wv_patch = WaveletBranch(embed_dim=embed_dim, stem_type="strided_patch", pool_energy=True)
    x = torch.randn(B, 3, 224, 224)
    out = wv_patch(x)

    # 112 // 16 = 7, so output is directly (B, embed_dim, 7, 7)
    assert out.shape == (B, embed_dim, 7, 7)

    # Test backward pass
    loss = out.sum()
    loss.backward()
    assert wv_patch.proj.weight.grad is not None


def test_video_level_evaluation_scoring():
    """Verify evaluation aggregates multiple chunks per video into video-level AUC."""
    probs = np.array([0.9, 0.85, 0.1, 0.15])  # 2 chunks for fake video, 2 chunks for real video
    video_ids = ["fake_vid_1", "fake_vid_1", "real_vid_1", "real_vid_1"]
    labels = np.array([1.0, 1.0, 0.0, 0.0])

    class MockLoader:
        def __iter__(self):
            logits = np.log(probs / (1.0 - probs))
            yield {
                "frames": torch.zeros(4, 8, 3, 224, 224),
                "conf": torch.ones(4, 8),
                "label": torch.from_numpy(labels),
                "video_id": video_ids,
            }

    class MockModel(torch.nn.Module):
        def forward(self, f, c, return_features=False):
            logits = np.log(probs / (1.0 - probs))
            return torch.from_numpy(logits).float()

    metrics = evaluate(MockModel(), MockLoader(), device="cpu")
    assert "video_level" in metrics
    assert metrics["video_level"]["auc"] == 1.0
    assert metrics["num_videos"] == 2
