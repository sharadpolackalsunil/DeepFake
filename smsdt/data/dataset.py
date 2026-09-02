"""
S-MSDT Dataset: WebDataset-based chunked face-crop loader.
Reads preprocessed T=8 aligned face chunks from webdataset .tar shards.
"""
import os
import glob
import numpy as np
import torch
from torch.utils.data import DataLoader
import webdataset as wds


# ImageNet normalization (EfficientNet pretrained weights expect this)
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# Manipulation type mapping
MANIP_TYPE_MAP = {
    "real": 0,
    "Deepfakes": 1,
    "Face2Face": 2,
    "FaceSwap": 3,
    "NeuralTextures": 4,
    "FaceShifter": 5,
    "DeepFakeDetection": 6,
}


def decode_sample(sample):
    """Decode and format a single webdataset sample.

    Handles:
    - uint8 (T,H,W,C) -> float32 (T,C,H,W) with ImageNet normalization
    - confidence array
    - label (binary)
    - manipulation type (string -> int)
    """
    # Frames: (T, 224, 224, 3) uint8 -> (T, 3, 224, 224) float32 normalized
    frames = torch.from_numpy(sample["frames.npy"].copy()).float() / 255.0
    if frames.ndim == 4 and frames.shape[-1] == 3:
        # (T, H, W, C) -> (T, C, H, W)
        frames = frames.permute(0, 3, 1, 2)

    # Apply ImageNet normalization
    frames = (frames - IMAGENET_MEAN) / IMAGENET_STD

    # Confidence: (T,) float32
    conf = torch.from_numpy(sample["conf.npy"].copy()).float()

    # Label: scalar
    label_raw = sample.get("label.cls", sample.get("cls", 0))
    if isinstance(label_raw, (bytes, str)):
        label_raw = int(label_raw)
    label = torch.tensor(label_raw, dtype=torch.float32)

    # Manipulation type: string -> int
    manip_raw = sample.get("manip.txt", sample.get("txt", "real"))
    if isinstance(manip_raw, bytes):
        manip_raw = manip_raw.decode("utf-8")
    manip_type = MANIP_TYPE_MAP.get(manip_raw.strip(), 0)

    return {
        "frames": frames,        # (T, 3, 224, 224)
        "conf": conf,            # (T,)
        "label": label,          # scalar
        "manip_type": manip_type, # int
    }


def build_dataloader(cfg, split="train", num_workers=12):
    """Build a WebDataset dataloader for a given split.

    Args:
        cfg: OmegaConf config with fields:
            - dataset: dataset name (e.g. "ffpp")
            - batch_size: micro-batch size
            - cache_root: path to cached shards (default: "data/cache")
        split: "train", "val", or "test"
        num_workers: number of dataloader workers

    Returns:
        DataLoader yielding batches of dicts with keys:
        frames (B,T,3,224,224), conf (B,T), label (B,), manip_type (B,)
    """
    cache_root = getattr(cfg, "cache_root", "data/cache")
    dataset_name = cfg.dataset
    batch_size = cfg.batch_size

    # Find all shard .tar files
    shard_dir = os.path.join(cache_root, dataset_name)
    shard_pattern = os.path.join(shard_dir, f"{split}-*.tar")
    shard_files = sorted(glob.glob(shard_pattern))

    if not shard_files:
        raise FileNotFoundError(
            f"No shard files found matching '{shard_pattern}'. "
            f"Run preprocessing first: python -m smsdt.preprocess.build_shards"
        )

    print(f"[Dataset] Found {len(shard_files)} shards for {split} split in {shard_dir}")

    # Build WebDataset pipeline
    is_train = (split == "train")
    dataset = (
        wds.WebDataset(shard_files, resampled=is_train, shardshuffle=is_train)
        .shuffle(1000 if is_train else 0)
        .decode()
        .map(decode_sample)
        .batched(batch_size, partial=not is_train)
    )

    loader = DataLoader(
        dataset,
        batch_size=None,     # WebDataset handles batching internally
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    return loader
