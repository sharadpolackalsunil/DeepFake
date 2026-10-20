"""Unit tests for Phase 2: CLIP ViT-L/14 PEFT, SMSDTv2 architecture, and shape contracts."""
import pytest
import torch
import torch.nn as nn
from transformers import CLIPVisionConfig

from smsdt.losses import detection_loss
from smsdt.models.clip_encoder import CLIPFrameEncoder, CLIP_MEAN, CLIP_STD
from smsdt.models.smsdt_v2 import SMSDTv2


@pytest.fixture
def dummy_clip_config():
    """Lightweight 2-layer config for deterministic, instant unit testing without downloading 1.7GB weights."""
    return CLIPVisionConfig(
        hidden_size=1024,
        intermediate_size=2048,
        num_hidden_layers=2,
        num_attention_heads=16,
        image_size=224,
        patch_size=14,
    )


def test_only_layernorms_train(dummy_clip_config):
    """Confirm only LayerNorm parameters have requires_grad=True, staying under 0.1% of parameters."""
    enc = CLIPFrameEncoder(config=dummy_clip_config, tune="layernorm", grad_ckpt=False)

    trainable = [n for n, p in enc.named_parameters() if p.requires_grad]
    assert len(trainable) > 0, "No trainable parameters found"
    assert all("norm" in n.lower() for n in trainable), f"Non-norm parameters set to trainable: {trainable}"

    n_train = sum(p.numel() for p in enc.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in enc.parameters())
    ratio = n_train / n_all
    print(f"\nTrainable params: {n_train:,} / {n_all:,} ({100 * ratio:.3f}%)")
    assert ratio < 0.05, f"Trainable parameter fraction too high: {ratio:.4f}"


def test_smsdt_v2_shapes_and_readout(dummy_clip_config):
    """Confirm S-MSDTv2 produces clip logits (B,) and 16x16 patch logits (B, T, 256)."""
    enc = CLIPFrameEncoder(config=dummy_clip_config, tune="layernorm", grad_ckpt=False)
    model = SMSDTv2(encoder=enc, dim=128, depth=2, heads=4, grid=16)

    B, T = 2, 4
    x = torch.randn(B, T, 3, 224, 224)
    logit, patch = model(x)

    assert logit.shape == (B,), f"Expected logit shape ({B},), got {logit.shape}"
    assert patch.shape == (B, T, 256), f"Expected patch logits ({B}, {T}, 256), got {patch.shape}"


def test_smsdt_v2_gradient_flow(dummy_clip_config):
    """Verify backpropagation through patch EvidenceReadout and LayerNorms without gradient corruption."""
    enc = CLIPFrameEncoder(config=dummy_clip_config, tune="layernorm", grad_ckpt=False)
    model = SMSDTv2(encoder=enc, dim=128, depth=2, heads=4, grid=16)

    B, T = 2, 4
    x = torch.randn(B, T, 3, 224, 224)
    label = torch.tensor([0.0, 1.0])
    mask = torch.zeros(B, T, 16, 16)
    mask[1, :, 6:10, 6:10] = 1.0
    has_mask = torch.tensor([1.0, 1.0])

    logit, patch = model(x)
    loss, logs = detection_loss(
        logit=logit,
        label=label,
        patch_logits=patch,
        mask=mask,
        has_mask=has_mask,
        smoothing=0.05,
        lam_mask=0.5,
    )
    loss.backward()

    # Check for dead trainable parameters
    dead = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not dead, f"Trainable parameters with None gradient: {dead}"

    # Check for non-finite gradients (NaN or Inf)
    bad = [n for n, p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
    assert not bad, f"Non-finite gradients detected in: {bad}"

    # Verify frozen parameters received no gradient
    frozen_with_grad = [n for n, p in model.named_parameters() if not p.requires_grad and p.grad is not None]
    assert not frozen_with_grad, f"Frozen parameters received gradients: {frozen_with_grad}"


def test_param_groups_separation(dummy_clip_config):
    """Verify optimizer param groups correctly differentiate LayerNorm LR from head LR."""
    enc = CLIPFrameEncoder(config=dummy_clip_config, tune="layernorm", grad_ckpt=False)
    model = SMSDTv2(encoder=enc, dim=128, depth=2, heads=4, grid=16)

    groups = model.param_groups(lr_backbone_ln=1e-4, lr_head=5e-4, weight_decay=0.05)
    assert len(groups) >= 2, "Expected multiple parameter groups"

    lrs = {g["lr"] for g in groups}
    assert 1e-4 in lrs, "LayerNorm learning rate 1e-4 not found in param groups"
    assert 5e-4 in lrs, "Head learning rate 5e-4 not found in param groups"


def test_clip_normalize_picklable():
    """Verify ClipNormalize is serializable with pickle (essential for Windows DataLoader spawn)."""
    import pickle
    from smsdt.train_v2 import get_clip_normalize

    norm = get_clip_normalize()
    data = pickle.dumps(norm)
    restored = pickle.loads(data)
    x = torch.rand(4, 3, 224, 224)
    out1 = norm(x)
    out2 = restored(x)
    assert torch.allclose(out1, out2), "Restored normalizer produced different values"

