"""
S-MSDT Dataset: WebDataset-based chunked face-crop loader.
Reads preprocessed T=8 aligned face chunks from webdataset .tar shards.
"""
import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import webdataset as wds
import torchvision.transforms.functional as TF
import random
import io
import numpy as np
try:
    import cv2
except ImportError:
    cv2 = None

def simulate_compression(frames, quality_range=(30, 90)):
    """Simulates video compression artifacts by applying JPEG compression to frames.
    frames: (T, C, H, W) tensor, values in [0, 1]
    """
    if cv2 is None:
        return frames

    q = random.randint(quality_range[0], quality_range[1])
    compressed_frames = []

    # cv2 expects (H, W, C) in BGR and uint8 [0, 255]
    frames_np = (frames.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)

    for i in range(frames_np.shape[0]):
        # Encode to JPEG
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), q]
        result, encimg = cv2.imencode('.jpg', frames_np[i], encode_param)
        if result:
            decimg = cv2.imdecode(encimg, 1)
            compressed_frames.append(decimg)
        else:
            compressed_frames.append(frames_np[i])

    # Convert back to (T, C, H, W) float32 [0, 1]
    compressed_frames = np.stack(compressed_frames, axis=0)
    compressed_tensor = torch.from_numpy(compressed_frames).float() / 255.0
    return compressed_tensor.permute(0, 3, 1, 2)


def augment_video(frames, split="train", target_size=224, target_frames=8):
    """
    Applies spatial and temporal augmentations to video frames.
    frames: (T, H, W, C) numpy array or torch tensor
    """
    if isinstance(frames, np.ndarray):
        frames = torch.from_numpy(frames).float()
        if frames.max() > 1.0:
            frames = frames / 255.0

    # Convert to (T, C, H, W)
    if frames.shape[-1] == 3:
        frames = frames.permute(0, 3, 1, 2)

    T, C, H, W = frames.shape

    if split == "train":
        # 1. Temporal frame dropout/jitter
        if T > target_frames:
            # Randomly select a contiguous chunk with some jitter
            max_start = T - target_frames
            start_idx = random.randint(0, max_start)
            # Occasional frame drop out (stride)
            stride = random.choice([1, 1, 2]) if max_start >= target_frames else 1
            indices = list(range(start_idx, min(start_idx + target_frames * stride, T), stride))
            if len(indices) < target_frames:
                indices = list(range(start_idx, start_idx + target_frames))
            indices = indices[:target_frames]
            frames = frames[indices]
        elif T < target_frames:
            # Pad by repeating last frame
            pad_size = target_frames - T
            padding = frames[-1:].repeat(pad_size, 1, 1, 1)
            frames = torch.cat([frames, padding], dim=0)

        # 2. Simulated Compression
        if random.random() < 0.3:
            frames = simulate_compression(frames)

        # 3. Spatial Augmentation (Random Resize + Re-crop)
        # Resize to something slightly larger
        scale = random.uniform(1.0, 1.2)
        new_h, new_w = int(target_size * scale), int(target_size * scale)
        frames = F.interpolate(frames, size=(new_h, new_w), mode='bilinear', align_corners=False)

        # Random Crop
        i, j, h, w = torch.randint(0, new_h - target_size + 1, (1,)).item(), \
                     torch.randint(0, new_w - target_size + 1, (1,)).item(), \
                     target_size, target_size
        frames = frames[:, :, i:i+h, j:j+w]

        # Horizontal flip
        if random.random() < 0.5:
            frames = TF.hflip(frames)

    else:
        # Validation/Test: center crop and deterministic sampling
        if T > target_frames:
            start_idx = (T - target_frames) // 2
            frames = frames[start_idx:start_idx + target_frames]
        elif T < target_frames:
            pad_size = target_frames - T
            padding = frames[-1:].repeat(pad_size, 1, 1, 1)
            frames = torch.cat([frames, padding], dim=0)

        frames = F.interpolate(frames, size=(target_size, target_size), mode='bilinear', align_corners=False)

    return frames


def build_dataloader(cfg, split="train", num_workers=12):
    """
    Builds a PyTorch DataLoader reading from webdataset shards.
    """
    shard_pattern = f"data/cache/{cfg.dataset}/{split}-%06d.tar"

    def decode_and_format(sample):
        # We expect a .npy or .mp4 file in reality. For this code, we parse standard keys
        frames = sample.get("frames.npy", None)
        if frames is not None:
            # frames expected to be bytes of a numpy array in webdataset, but WDS `.decode("numpy")` handles this.
            # Assuming it's already decoded by `.decode()`
            pass
        else:
            # fallback mock data
            frames = np.zeros((16, 224, 224, 3), dtype=np.uint8)

        # Apply augmentation
        augmented_frames = augment_video(frames, split=split)

        # Confs and labels
        conf = sample.get("conf.npy", np.ones((8,)))
        label = sample.get("label.cls", 0)

        if isinstance(conf, np.ndarray):
            conf = torch.from_numpy(conf).float()

        # Align confidence length with target frames if necessary
        if conf.shape[0] > 8:
            conf = conf[:8]
        elif conf.shape[0] < 8:
            conf = F.pad(conf, (0, 8 - conf.shape[0]), value=1.0)

        return {
            "frames": augmented_frames,
            "conf": conf,
            "label": torch.tensor(label, dtype=torch.float32)
        }

    # Build WebDataset pipeline
    is_train = (split == "train")
    dataset = (
        wds.WebDataset(shard_pattern, resampled=True if split=="train" else False)
        .decode("numpy")
        .map(decode_and_format)
        .batched(cfg.batch_size)
    )

    loader = DataLoader(
        dataset,
        batch_size=None,     # WebDataset handles batching internally
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    return loader
