"""Unit tests for S-MSDT Phase 1 components.

Verifies:
1. Forward & backward tensor shapes and finite gradient flow.
2. Temporal order sensitivity (flip test).
3. TemporalDiff identity initialization.
4. SBI synthesizer output shapes and label validity.
"""
import pytest
import torch
from smsdt.models.smsdt import SMSDT
from smsdt.models.nmsss_vit import TemporalDiff
from smsdt.losses import detection_loss
from smsdt.data.sbi_video import sbi_clip
import numpy as np


def test_shapes_and_gradients():
    """Verify forward output shapes and backward gradient flow on all trainable parameters."""
    model = SMSDT(embed_dim=256, depth=4, heads=8, grid=14)
    x = torch.randn(4, 8, 3, 224, 224)
    conf = torch.ones(4, 8)
    logit, patch = model(x, conf)

    assert logit.shape == (4,), f"Expected logit shape (4,), got {logit.shape}"
    assert patch.shape == (4, 8, 196), f"Expected patch shape (4, 8, 196), got {patch.shape}"

    label = torch.tensor([0.0, 1.0, 0.0, 1.0])
    mask = torch.zeros(4, 8, 14, 14)
    mask[1, :, 5:9, 5:9] = 1.0
    has_mask = torch.tensor([1.0, 1.0, 1.0, 0.0])

    loss, logs = detection_loss(logit, label, patch, mask, has_mask)
    assert "cls" in logs
    assert "mask" in logs
    loss.backward()

    # Check for dead or non-finite gradients
    dead = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not dead, f"No gradient reaches parameters: {dead[:5]}"

    bad = [n for n, p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
    assert not bad, f"Non-finite gradients detected in: {bad[:5]}"


def test_temporal_order_matters():
    """Verify that reversing frame order changes output logits (proves temporal order sensitivity)."""
    model = SMSDT(embed_dim=256, depth=4, heads=8, grid=14).eval()
    # Assign non-zero weights to temporal_diff to simulate trained state
    for blk in model.blocks:
        torch.nn.init.normal_(blk.temporal_diff.proj.weight, std=0.2)

    x = torch.randn(1, 8, 3, 224, 224)
    with torch.no_grad():
        out_normal = model(x)[0]
        out_flipped = model(x.flip(1))[0]

    assert not torch.equal(out_normal, out_flipped), "Model is completely invariant to frame order!"
    assert abs(out_normal.item() - out_flipped.item()) > 1e-4, \
        f"Difference too small: {out_normal.item()} vs {out_flipped.item()}"


def test_temporal_diff_starts_as_identity():
    """Verify that TemporalDiff starts as an exact identity mapping before training."""
    td = TemporalDiff(256)
    z = torch.randn(2, 8, 196, 256)
    out = td(z)
    assert torch.allclose(out, z, atol=1e-6), \
        "TemporalDiff is not an identity mapping at initialization!"


def test_sbi_output_shapes_and_labels():
    """Verify that sbi_clip outputs correct shapes and non-empty masks."""
    clip_u8 = np.random.randint(0, 256, (8, 224, 224, 3), dtype=np.uint8)
    # Mock 106-point landmark centered in frame
    lmk106 = np.random.uniform(50, 174, (8, 106, 2)).astype(np.float32)

    fakes, masks = sbi_clip(clip_u8, lmk106)
    assert fakes.shape == (8, 224, 224, 3)
    assert fakes.dtype == np.uint8
    assert masks.shape == (8, 224, 224)
    assert masks.dtype == np.float32
    assert masks.max() > 0.0, "SBI mask should be non-empty"
