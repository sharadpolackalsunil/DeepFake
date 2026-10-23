"""Automated test suite for S-MSDTv3 (Step 5 of Execution Plan).

Tests:
1. Haar DWT and SRM high-pass residual filter operations and shape invariants.
2. ResidualStream forward pass, gated injection, and frequency auxiliary head.
3. EvidenceReadoutV3: K-query attention, soft perioral mouth prior, and Log-Sum-Exp pooling.
4. Full SMSDTv3 end-to-end forward shapes across T=1, T=4, and T=8 frames.
5. Near-zero initialization equivalence of high-frequency gate (< 1e-4).
6. Multi-task loss formulation in losses_v3.py (Focal BCE, pairwise ranking, mask loss, OHEM).
7. Parameter groups separation and weight decay exclusion on scalar parameters.
"""
import os
import sys

# Ensure repository root is on sys.path for direct execution
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
import torch.nn as nn
from transformers import CLIPVisionConfig

from smsdt.models.clip_encoder import CLIPFrameEncoder
from smsdt.models.v3_modules import haar_dwt, _srm_kernels, ResidualStream, EvidenceReadoutV3
from smsdt.models.smsdt_v3 import SMSDTv3
from smsdt.losses_v3 import focal_bce, detection_loss_v3
from smsdt.data.mouth_prior import compute_perioral_prior


@pytest.fixture
def dummy_clip_config():
    """Deterministic, lightweight 2-layer config for instant CPU/GPU execution without remote weights."""
    return CLIPVisionConfig(
        hidden_size=1024,
        intermediate_size=2048,
        num_hidden_layers=2,
        num_attention_heads=16,
        image_size=224,
        patch_size=14,
    )


def test_haar_dwt_and_srm_shapes():
    """Verify 2D Haar DWT and Spatial Rich Model (SRM) kernels produce exact shape contracts."""
    x = torch.randn(2, 3, 224, 224)
    ll, lh, hl, hh = haar_dwt(x)

    assert ll.shape == (2, 3, 112, 112), f"Expected LL (2,3,112,112), got {ll.shape}"
    assert lh.shape == (2, 3, 112, 112)
    assert hl.shape == (2, 3, 112, 112)
    assert hh.shape == (2, 3, 112, 112)

    # Verify SRM filter kernels
    srm = _srm_kernels()
    assert srm.shape == (3, 5, 5), f"Expected SRM shape (3,5,5), got {srm.shape}"


def test_residual_stream_forward_and_gate():
    """Verify ResidualStream extracts 21-channel high-frequency residuals and gates them cleanly."""
    dim = 128
    res = ResidualStream(dim=dim, c_r=64, grid=16, init_bias=-4.0)

    BT = 4
    x01 = torch.rand(BT, 3, 224, 224)
    s = torch.randn(BT, 256, dim)

    fused_s, aux_logit = res(x01, s)

    assert fused_s.shape == (BT, 256, dim), f"Expected fused_s shape ({BT}, 256, {dim}), got {fused_s.shape}"
    assert aux_logit.shape == (BT,), f"Expected aux_logit shape ({BT},), got {aux_logit.shape}"

    # Verify gate values are small at initialization (g ~ 0.018 for bias -4.0)
    with torch.no_grad():
        feats = res.residual_input(x01)
        r = res.cnn(feats)
        r = torch.nn.functional.adaptive_avg_pool2d(r, (16, 16))
        r = res.to_dim(r).flatten(2).transpose(1, 2)
        r = res.ln_r(r)
        g = torch.sigmoid(res.gate(torch.cat([s, r], dim=-1)))
        assert (g < 0.05).all(), f"Gate values not small at initialization: max {g.max().item()}"


def test_evidence_readout_v3_and_lse():
    """Verify EvidenceReadoutV3 combines multi-query cross-attention with LSE pooling."""
    dim = 128
    readout = EvidenceReadoutV3(dim=dim, heads=4, n_queries=4, r0=3.0)

    B, T, N = 2, 8, 256
    z = torch.randn(B, T, N, dim)
    prior = torch.rand(B, T, N)  # simulated soft perioral mouth prior

    logit, patch_logits = readout(z, prior=prior)

    assert logit.shape == (B,), f"Expected clip logit shape ({B},), got {logit.shape}"
    assert patch_logits.shape == (B, T, N), f"Expected patch logits ({B}, {T}, {N}), got {patch_logits.shape}"

    # Test LSE function stability with temperature scaling
    r = torch.tensor(3.0)
    lse_val = EvidenceReadoutV3.lse(patch_logits, r, dim=2)
    assert lse_val.shape == (B, T)
    assert torch.isfinite(lse_val).all()


def test_perioral_prior_generation():
    """Verify landmark-guided perioral prior generates a valid soft 16x16 Gaussian map."""
    B, T = 2, 4
    # Simulate 106-point facial landmarks centered around (112, 112) with mouth around (112, 150)
    lmk = torch.zeros(B, T, 106, 2)
    lmk[..., :52, :] = 100.0
    lmk[..., 52:72, 0] = 112.0  # mouth x
    lmk[..., 52:72, 1] = 150.0  # mouth y

    prior = compute_perioral_prior(lmk, grid_size=16, crop_size=224.0)

    assert prior.shape == (B, T, 256), f"Expected prior shape ({B}, {T}, 256), got {prior.shape}"
    assert (prior >= 0.0).all() and (prior <= 1.0).all(), "Prior values must fall strictly in [0, 1]"
    # Peak prior should be located around mouth rows (y ~ 150/224 * 16 ~ 10-11)
    assert prior.max() > 0.8, "Prior should have high confidence around the mouth landmarks"


def test_smsdt_v3_forward_shapes(dummy_clip_config):
    """Verify full SMSDTv3 end-to-end forward pass across T=1, T=4, and T=8 frames."""
    enc = CLIPFrameEncoder(config=dummy_clip_config, tune="layernorm", grad_ckpt=False)
    model = SMSDTv3(encoder=enc, dim=128, depth=2, heads=4, grid=16, n_queries=4)

    for T in [1, 4, 8]:
        B = 2
        frames = torch.randn(B, T, 3, 224, 224)
        logit, patch, aux = model(frames)

        assert logit.shape == (B,), f"T={T}: expected logit ({B},), got {logit.shape}"
        assert patch.shape == (B, T, 256), f"T={T}: expected patch logits ({B}, {T}, 256), got {patch.shape}"
        assert aux.shape == (B,), f"T={T}: expected aux ({B},), got {aux.shape}"


def test_smsdt_v3_zero_init_equivalence(dummy_clip_config):
    """Verify near-zero initialization equivalence of high-frequency gate (< 1e-4 max absolute error)."""
    enc = CLIPFrameEncoder(config=dummy_clip_config, tune="layernorm", grad_ckpt=False)
    model = SMSDTv3(encoder=enc, dim=128, depth=2, heads=4, grid=16, zero_gate=True)

    B, T = 2, 4
    x = torch.randn(B, T, 3, 224, 224)

    # Check that residual stream gate output is below 1e-4
    with torch.no_grad():
        x01 = (x.view(B * T, 3, 224, 224) * model.clip_std + model.clip_mean).clamp(0.0, 1.0)
        s = torch.randn(B * T, 256, 128)
        feats = model.res.residual_input(x01)
        r = model.res.cnn(feats)
        r = torch.nn.functional.adaptive_avg_pool2d(r, (16, 16))
        r = model.res.to_dim(r).flatten(2).transpose(1, 2)
        r = model.res.ln_r(r)
        g = torch.sigmoid(model.res.gate(torch.cat([s, r], dim=-1)))
        max_residual_impact = (g * r).abs().max().item()
        assert max_residual_impact < 1e-4, f"Residual impact exceeded 1e-4 bound: {max_residual_impact:.6f}"


def test_losses_v3_multi_task():
    """Verify focal BCE, pairwise ranking loss, difference mask loss, and OHEM."""
    B, T, N = 4, 4, 256
    logit = torch.tensor([0.2, 0.8, -0.5, 1.2], requires_grad=True)
    label = torch.tensor([0.0, 1.0, 0.0, 1.0])
    patch_logits = torch.randn(B, T, N, requires_grad=True)
    soft_mask = torch.zeros(B, T, N)
    soft_mask[1, :, 50:100] = 1.0
    soft_mask[3, :, 80:140] = 0.8
    aux_logit = torch.tensor([0.1, 0.9, -0.2, 1.1], requires_grad=True)

    total_loss, logs = detection_loss_v3(
        logit=logit,
        label=label,
        patch_logits=patch_logits,
        soft_mask=soft_mask,
        aux_logit=aux_logit,
        lam_rank=0.5,
        lam_mask=0.5,
        lam_freq=0.3,
        ohem_frac=0.5,
    )

    assert total_loss > 0.0, "Loss must be positive"
    for term in ["cls", "rank", "mask", "freq"]:
        assert term in logs, f"Missing loss log term: {term}"

    # Verify backpropagation through all loss terms
    total_loss.backward()
    assert logit.grad is not None and torch.isfinite(logit.grad).all()
    assert patch_logits.grad is not None and torch.isfinite(patch_logits.grad).all()
    assert aux_logit.grad is not None and torch.isfinite(aux_logit.grad).all()


def test_v3_param_groups_separation(dummy_clip_config):
    """Verify parameter groups divide into CLIP LayerNorms, Existing Head, and New V3 Modules,
    and scalar parameters are cleanly excluded from weight decay.
    """
    enc = CLIPFrameEncoder(config=dummy_clip_config, tune="layernorm", grad_ckpt=False)
    model = SMSDTv3(encoder=enc, dim=128, depth=2, heads=4, grid=16)

    groups = model.param_groups(
        lr_backbone_ln=5e-6,
        lr_head=1e-4,
        lr_new=3e-4,
        weight_decay=0.05,
    )

    group_names = [g["name"] for g in groups]
    assert any("ln" in name for name in group_names), "Missing LayerNorm group"
    assert any("head" in name for name in group_names), "Missing Head group"
    assert any("new" in name for name in group_names), "Missing New V3 group"

    # Verify scalar parameters have weight decay = 0.0
    decay_params = set()
    no_decay_params = set()
    for g in groups:
        if g["weight_decay"] > 0:
            decay_params.update(g["params"])
        else:
            no_decay_params.update(g["params"])

    # Check scalar parameters
    for scalar_name, scalar_param in [
        ("beta", model.beta),
        ("prior_gain", model.readout.prior_gain),
        ("log_r", model.readout.log_r),
        ("mix", model.readout.mix),
    ]:
        assert scalar_param in no_decay_params, f"Scalar parameter {scalar_name} received weight decay!"
        assert scalar_param not in decay_params, f"Scalar parameter {scalar_name} in decay params!"


if __name__ == "__main__":
    import sys
    print("Running S-MSDTv3 unit tests directly...")
    cfg = CLIPVisionConfig(hidden_size=1024, intermediate_size=2048, num_hidden_layers=2, num_attention_heads=16, image_size=224, patch_size=14)
    test_haar_dwt_and_srm_shapes()
    print("  * test_haar_dwt_and_srm_shapes: PASSED")
    test_residual_stream_forward_and_gate()
    print("  * test_residual_stream_forward_and_gate: PASSED")
    test_evidence_readout_v3_and_lse()
    print("  * test_evidence_readout_v3_and_lse: PASSED")
    test_perioral_prior_generation()
    print("  * test_perioral_prior_generation: PASSED")
    test_smsdt_v3_forward_shapes(cfg)
    print("  * test_smsdt_v3_forward_shapes: PASSED")
    test_smsdt_v3_zero_init_equivalence(cfg)
    print("  * test_smsdt_v3_zero_init_equivalence: PASSED")
    test_losses_v3_multi_task()
    print("  * test_losses_v3_multi_task: PASSED")
    test_v3_param_groups_separation(cfg)
    print("  * test_v3_param_groups_separation: PASSED")
    print("\nAll 8 tests in test_v3.py passed successfully!")
