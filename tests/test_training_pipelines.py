"""End-to-end smoke tests for GenD, S-MSDTv2, and S-MSDTv3 training and evaluation loops.

Verifies:
1. Model forward/backward passes and loss computation.
2. Scheduler creation and stepping (cosine_with_warmup).
3. EMA weight tracking.
4. Evaluation routines (eval.py evaluate) on mock batched data.
"""
import copy
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from smsdt.models.clip_encoder import CLIPFrameEncoder
from smsdt.models.gend import GenDModel, slerp
from smsdt.models.smsdt_v2 import SMSDTv2
from smsdt.models.smsdt_v3 import SMSDTv3
from smsdt.train import cosine_with_warmup
from smsdt.eval import evaluate as evaluate_metrics


class DummyDataset(torch.utils.data.Dataset):
    """Mock dataset mimicking PairedClipDataset / EvalClipDataset."""
    def __init__(self, n=4, K=2, T=1, S=224, is_eval=False):
        self.n = n
        self.K = K
        self.T = T
        self.S = S
        self.is_eval = is_eval

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        if self.is_eval:
            # EvalClipDataset yields (K, T, 3, S, S)
            return {
                "frames": torch.randn(self.K, self.T, 3, self.S, self.S),
                "label": torch.tensor(float(idx % 2)),
                "video_id": f"vid_{idx}",
                "method": "Deepfakes" if idx % 2 else "real",
                "prior": torch.ones(self.K, self.T, 256),
            }
        else:
            # PairedClipDataset with collate yields (2, T, 3, S, S) flattened into pairs
            return {
                "frames": torch.randn(2, self.T, 3, self.S, self.S),
                "label": torch.tensor([0.0, 1.0]),
                "prior": torch.ones(2, self.T, 256),
                "mask": torch.zeros(2, self.T, 14, 14),
                "has_mask": torch.tensor([1.0, 1.0]),
                "method": torch.tensor([0, 1]),
            }


def dummy_collate(batch):
    return {
        "frames": torch.cat([b["frames"] for b in batch]),  # (2B, T, 3, S, S)
        "label": torch.cat([b["label"] for b in batch]),    # (2B,)
        "prior": torch.cat([b["prior"] for b in batch]),    # (2B, T, 256)
        "mask": torch.cat([b["mask"] for b in batch]),
        "has_mask": torch.cat([b["has_mask"] for b in batch]),
        "method": torch.cat([b["method"] for b in batch]),
    }


@pytest.fixture(scope="module")
def shared_encoder():
    """Shared CLIPFrameEncoder to avoid multiple model weight downloads/loads."""
    return CLIPFrameEncoder(name="openai/clip-vit-large-patch14", tune="layernorm", grad_ckpt=False)


def test_gend_training_loop(shared_encoder):
    """Test full training step, EMA update, scheduler step, and validation eval for GenD."""
    device = torch.device("cpu")
    model = GenDModel(name="openai/clip-vit-large-patch14", tune="layernorm").to(device)
    # Re-use shared encoder to be fast
    model.encoder = shared_encoder

    ln_params = [p for p in model.encoder.parameters() if p.requires_grad]
    head_params = [p for p in model.head.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": ln_params, "lr": 1e-4},
        {"params": head_params, "lr": 5e-4},
    ])

    total_steps = 10
    scheduler = cosine_with_warmup(optimizer, total_steps=total_steps, warmup_frac=0.1, min_ratio=0.01)

    model_ema = copy.deepcopy(model)
    for p in model_ema.parameters():
        p.requires_grad = False

    # Simulate 1 training batch
    train_ds = DummyDataset(n=2, T=1)
    train_loader = DataLoader(train_ds, batch_size=1, collate_fn=dummy_collate)
    batch = next(iter(train_loader))

    frames = batch["frames"].squeeze(1)  # (2, 3, 224, 224)
    labels = batch["label"]              # (2,)

    # Slerp
    slerp_t = 0.5
    paired_frames = frames[torch.arange(frames.size(0)) ^ 1]
    target_labels = (1.0 - slerp_t) * labels + slerp_t * labels[torch.arange(frames.size(0)) ^ 1]

    logits, embeddings = model(frames, slerp_t=slerp_t, pair_x=paired_frames)
    assert logits.shape == (2,)
    assert embeddings.shape == (2, 1024)

    loss_dict = model.compute_loss(logits, target_labels, embeddings)
    loss = loss_dict["loss"]
    loss.backward()

    optimizer.step()
    optimizer.zero_grad()
    scheduler.step()

    # EMA step
    with torch.no_grad():
        for p_ema, p in zip(model_ema.parameters(), model.parameters()):
            p_ema.data.mul_(0.999).add_(p.data, alpha=0.001)

    # Validation evaluation check
    val_ds = DummyDataset(n=2, K=2, T=1, is_eval=True)
    val_loader = DataLoader(val_ds, batch_size=2)
    metrics = evaluate_metrics(model_ema, val_loader, device=device, top_k_ratio=1.0)
    assert "auc" in metrics


def test_smsdt_v2_training_loop(shared_encoder):
    """Test full training step and evaluation for SMSDTv2."""
    device = torch.device("cpu")
    model = SMSDTv2(encoder=shared_encoder, dim=64, depth=1, heads=2, max_frames=4, grid=16).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = cosine_with_warmup(optimizer, total_steps=10, warmup_frac=0.1)

    # 4 frames per clip
    frames = torch.randn(2, 4, 3, 224, 224)
    labels = torch.tensor([0.0, 1.0])

    logits, patch_logits = model(frames)
    assert logits.shape == (2,)
    assert patch_logits.shape == (2, 4, 256)

    loss = nn.BCEWithLogitsLoss()(logits, labels)
    loss.backward()
    optimizer.step()
    scheduler.step()

    # Eval test
    val_ds = DummyDataset(n=2, K=2, T=4, is_eval=True)
    val_loader = DataLoader(val_ds, batch_size=2)
    metrics = evaluate_metrics(model, val_loader, device=device, top_k_ratio=1.0)
    assert "auc" in metrics


def test_smsdt_v3_training_loop(shared_encoder):
    """Test full training step and evaluation for SMSDTv3."""
    device = torch.device("cpu")
    model = SMSDTv3(encoder=shared_encoder, dim=64, depth=1, heads=2, max_frames=2, grid=16).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = cosine_with_warmup(optimizer, total_steps=10, warmup_frac=0.1)

    frames = torch.randn(2, 2, 3, 224, 224)
    prior = torch.ones(2, 2, 256)
    labels = torch.tensor([0.0, 1.0])

    logit, patch_logits, aux_logit = model(frames, prior=prior)
    assert logit.shape == (2,)
    assert patch_logits.shape == (2, 2, 256)
    assert aux_logit.shape == (2,)

    loss = nn.BCEWithLogitsLoss()(logit, labels) + 0.3 * nn.BCEWithLogitsLoss()(aux_logit, labels)
    loss.backward()
    optimizer.step()
    scheduler.step()

    val_ds = DummyDataset(n=2, K=1, T=2, is_eval=True)
    val_loader = DataLoader(val_ds, batch_size=2)
    metrics = evaluate_metrics(model, val_loader, device=device, top_k_ratio=1.0)
    assert "auc" in metrics

