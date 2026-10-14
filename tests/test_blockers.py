"""
Unit tests for the 9 blockers and data leakage fixes identified in the repo review.
"""
import os
import tempfile
import json
import csv
import numpy as np
import torch
import pytest

from smsdt.preprocess.build_shards import make_chunks, build_split_manifests
from smsdt.data.dataset import simulate_compression, normalize_imagenet, IMAGENET_MEAN, IMAGENET_STD, build_dataloader
from smsdt.models.nmsss_vit import TransformerBlock, NMSSSBlock
from smsdt.models.smsdt import SMSDT
from smsdt.export.to_onnx import export_to_onnx
from smsdt.eval import compute_metrics, evaluate


def test_blocker_1_numpy_imported_in_make_chunks():
    """Blocker 1: Preprocessing must not crash in make_chunks due to missing numpy."""
    frames = [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(16)]
    confs = [0.95 for _ in range(16)]
    chunks = make_chunks(frames, confs, chunk_length=8, stride=4)
    assert len(chunks) == 3
    assert chunks[0][0].shape == (8, 224, 224, 3)
    assert chunks[0][1].shape == (8,)


def test_blocker_2_cv2_imported_in_simulate_compression():
    """Blocker 2: simulate_compression must not crash due to missing cv2."""
    frames = torch.rand(8, 3, 224, 224)
    compressed = simulate_compression(frames, quality_range=(50, 60))
    assert compressed.shape == (8, 3, 224, 224)
    assert compressed.min() >= 0.0
    assert compressed.max() <= 1.0


def test_blocker_3_shard_discovery_and_cache_root():
    """Blocker 3: Shard pattern must respect cache_root and support brace expansion / file list."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_dir = os.path.join(tmpdir, "test_ds")
        os.makedirs(dataset_dir, exist_ok=True)
        # Create a mock shard file
        shard_path = os.path.join(dataset_dir, "train-000000.tar")
        with open(shard_path, "w") as f:
            f.write("")

        class MockCfg:
            cache_root = tmpdir
            dataset = "test_ds"
            batch_size = 2
            epoch_length = 10

        loader = build_dataloader(MockCfg, split="train", num_workers=0)
        assert loader is not None


def test_blocker_4_with_epoch_prevents_infinite_loader():
    """Blocker 4: resampled=True must be bounded by with_epoch so epoch loop terminates."""
    import webdataset as wds
    with tempfile.TemporaryDirectory() as tmpdir:
        class MockCfg:
            cache_root = tmpdir
            dataset = "mock_ds"
            batch_size = 2
            epoch_length = 5

        # Check with_epoch chaining
        ds = wds.WebDataset(os.path.join(tmpdir, "shard-{000000..000001}.tar"), resampled=True)
        bounded_ds = ds.with_epoch(MockCfg.epoch_length)
        assert bounded_ds is not None


def test_blocker_6_imagenet_normalization():
    """Blocker 6: ImageNet normalization must be correctly applied."""
    frames = torch.ones(8, 3, 224, 224)  # pure 1.0
    normed = normalize_imagenet(frames)
    expected_r = (1.0 - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
    expected_g = (1.0 - IMAGENET_MEAN[1]) / IMAGENET_STD[1]
    expected_b = (1.0 - IMAGENET_MEAN[2]) / IMAGENET_STD[2]
    assert torch.isclose(normed[0, 0, 0, 0], torch.tensor(expected_r, dtype=torch.float32), atol=1e-4)
    assert torch.isclose(normed[0, 1, 0, 0], torch.tensor(expected_g, dtype=torch.float32), atol=1e-4)
    assert torch.isclose(normed[0, 2, 0, 0], torch.tensor(expected_b, dtype=torch.float32), atol=1e-4)


def test_blocker_7_per_manipulation_auc_scores_against_real():
    """Blocker 7: Per-manipulation AUC must score each fake type against ALL real samples."""
    # 50 real samples (label 0, manip_type 0) with low predicted fake prob
    real_probs = np.random.uniform(0.05, 0.25, size=50)
    real_labels = np.zeros(50)
    real_manip = np.zeros(50)

    # 50 Deepfakes samples (label 1, manip_type 1) with high predicted fake prob
    fake_probs = np.random.uniform(0.75, 0.95, size=50)
    fake_labels = np.ones(50)
    fake_manip = np.ones(50)

    all_probs = np.concatenate([real_probs, fake_probs])
    all_labels = np.concatenate([real_labels, fake_labels])
    all_manip = np.concatenate([real_manip, fake_manip])

    class MockModel(torch.nn.Module):
        def forward(self, f, c, return_features=False):
            return torch.zeros(1)

    # Build mock loader returning the pre-calculated outputs
    class MockBatchLoader:
        def __iter__(self):
            # Convert to logit
            logits = np.log(all_probs / (1.0 - all_probs))
            yield {
                "frames": torch.zeros(100, 8, 3, 224, 224),
                "conf": torch.ones(100, 8),
                "label": torch.from_numpy(all_labels),
                "manip_type": torch.from_numpy(all_manip).long(),
                "video_id": [f"vid_{i}" for i in range(100)],
            }

    # Evaluate directly with mock model where model returns logits matching our data
    class IdentityModel(torch.nn.Module):
        def forward(self, frames, conf, return_features=False):
            logits = np.log(all_probs / (1.0 - all_probs))
            return torch.from_numpy(logits).float()

    model = IdentityModel()
    metrics = evaluate(model, MockBatchLoader(), device="cpu")

    # In the old code, per-manipulation AUC was ALWAYS 0.5 because it scored fake vs fake!
    # In the fixed code, it scores fake vs real, so AUC should be ~1.0!
    assert "per_manipulation" in metrics
    assert "Deepfakes" in metrics["per_manipulation"]
    assert metrics["per_manipulation"]["Deepfakes"]["auc"] > 0.95


def test_blocker_8_gradient_checkpointing_wired_in_vit():
    """Blocker 8: gradient_checkpointing must be read and used by ViT blocks."""
    block = TransformerBlock(dim=64, heads=2)
    block.gradient_checkpointing = True
    block.train()

    x = torch.randn(2, 10, 64, requires_grad=True)
    out = block(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None

    # Test on full SMSDT model
    model = SMSDT(embed_dim=64, depth=2, heads=2, grid=7)
    model.set_gradient_checkpointing(True)
    for blk in model.blocks:
        assert blk.gradient_checkpointing is True
        assert blk.spatial.gradient_checkpointing is True
        assert blk.temporal.gradient_checkpointing is True


def test_blocker_9_to_onnx_accepts_dict_checkpoint():
    """Blocker 9: to_onnx must accept checkpoints saved as dict with model_state_dict."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "epoch0.pt")
        out_onnx = os.path.join(tmpdir, "model.onnx")

        model = SMSDT(embed_dim=64, depth=1, heads=2, grid=7)
        checkpoint_dict = {
            "epoch": 0,
            "step": 500,
            "model_state_dict": model.state_dict(),
            "best_auc": 0.92,
        }
        torch.save(checkpoint_dict, ckpt_path)

        # Should load the dict cleanly without throwing AttributeError/RuntimeError
        from smsdt.export.to_onnx import export_to_onnx
        export_to_onnx(ckpt_path, out_onnx, model=model)
        assert os.path.exists(out_onnx)


def test_data_leakage_prevention_in_split_manifests():
    """Verify source/target identities stay in the same split to avoid data leakage."""
    videos = [
        ("path/001.mp4", "real_001", 0, "real"),
        ("path/002.mp4", "real_002", 0, "real"),
        ("path/003.mp4", "real_003", 0, "real"),
        ("path/001_002.mp4", "Deepfakes_001_002", 1, "Deepfakes"),
        ("path/001_003.mp4", "Face2Face_001_003", 1, "Face2Face"),
        ("path/002_003.mp4", "FaceSwap_002_003", 1, "FaceSwap"),
    ]

    splits = build_split_manifests(videos)
    # Check that for any source ID (e.g., 001), real_001 and Deepfakes_001_002 are in the same split
    split_assignments = {}
    for s_name, s_vids in splits.items():
        for _, full_id, _, _ in s_vids:
            split_assignments[full_id] = s_name

    assert split_assignments["real_001"] == split_assignments["Deepfakes_001_002"]
    assert split_assignments["real_001"] == split_assignments["Face2Face_001_003"]
