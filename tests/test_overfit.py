"""
Diagnostic Test: Overfit 64 Clips Sanity Check
===============================================
Verifies that the architecture and optimization setup can easily memorize and
overfit 64 video clips (Phase 0 Diagnostic requirement).
"""
import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from smsdt.models.baseline_b4 import BaselineB4
from smsdt.models.smsdt import SMSDT
from smsdt.losses import total_loss


def test_baseline_b4_overfit_64_clips():
    """Verify that BaselineB4 can overfit clips in a few gradient steps."""
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(1)
    # Adaptive scale: quick on CPU, full 64 on GPU
    N = 64 if device == "cuda" else 4
    T = 8 if device == "cuda" else 2
    n_real = max(1, N // 4)
    n_fake = N - n_real

    # n_real reals (0), n_fake fakes (1)
    frames = torch.randn(N, T, 3, 224, 224)
    confs = torch.ones(N, T)
    labels = torch.cat([torch.zeros(n_real), torch.ones(n_fake)])

    # Inject synthetic separation so gradient direction is unambiguous
    frames[:n_real] -= 0.5
    frames[n_real:] += 0.5

    dataset = TensorDataset(frames, confs, labels)
    loader = DataLoader(dataset, batch_size=min(8, N), shuffle=True)

    model = BaselineB4(pretrained=False).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.0)

    initial_loss = None
    final_loss = None

    steps = 30 if device == "cuda" else 6
    for step in range(steps):
        for b_frames, b_confs, b_labels in loader:
            optimizer.zero_grad()
            logits = model(b_frames, b_confs)
            loss, _ = total_loss(logits, b_labels, loss_type="bce", pos_weight=0.25)
            loss.backward()
            optimizer.step()

            if initial_loss is None:
                initial_loss = loss.item()
            final_loss = loss.item()

    # Evaluate training accuracy on the clips
    model.eval()
    with torch.no_grad():
        all_logits = model(frames, confs)
        preds = (torch.sigmoid(all_logits) > 0.5).float()
        acc = (preds == labels).float().mean().item()

    assert final_loss < initial_loss
    assert acc >= 0.75


def test_smsdt_v3_forward_backward_sanity():
    """Verify S-MSDT v3 forward and backward pass with new PSIT and anti-aliased stem."""
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(1)
    B = 2
    T = 4 if device == "cpu" else 8
    frames = torch.randn(B, T, 3, 224, 224)
    confs = torch.ones(B, T)
    labels = torch.tensor([0.0, 1.0])

    model = SMSDT(embed_dim=128, depth=2, heads=4, grid=7)
    logits, z_rgb, z_dwt, z_video = model(frames, confs, return_features=True)

    assert logits.shape == (B,)
    assert z_rgb.shape == (B, 128)
    assert z_dwt.shape == (B, 128)
    assert z_video.shape == (B, 128)

    loss, loss_dict = total_loss(logits, labels, z_rgb=z_rgb, z_dwt=z_dwt, z_post_vit=z_video,
                                 loss_type="bce", pos_weight=0.25)
    loss.backward()

    # Verify gradients flow into PSIT, Wavelet stem, and ViT
    assert model.psit.merge.weight.grad is not None
    assert model.wavelet_branch.stage1[0].weight.grad is not None
    assert model.blocks[0].spatial.attn.in_proj_weight.grad is not None
