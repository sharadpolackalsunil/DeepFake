"""
S-MSDT Dataset: WebDataset-based chunked face-crop loader.
Reads preprocessed T=8 aligned face chunks from webdataset .tar shards.
"""
import os
import glob
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
try:
    import webdataset as wds
    from webdataset.gopen import gopen_schemes, gopen_file
    for _letter in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ":
        gopen_schemes[_letter] = gopen_file
except Exception:
    wds = None
import torchvision.transforms.functional as TF
import random
import io

MANIP_TYPE_MAP = {
    "real": 0,
    "Deepfakes": 1,
    "Face2Face": 2,
    "FaceSwap": 3,
    "NeuralTextures": 4,
    "FaceShifter": 5,
    "DeepFakeDetection": 6,
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def normalize_imagenet(frames):
    """Normalize (T, C, H, W) frames with ImageNet mean and std."""
    mean = torch.tensor(IMAGENET_MEAN, dtype=frames.dtype, device=frames.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=frames.dtype, device=frames.device).view(1, 3, 1, 1)
    return (frames - mean) / std


def simulate_compression(frames, quality_range=(15, 80)):
    """Simulates video compression artifacts by applying JPEG compression to frames.
    frames: (T, C, H, W) tensor, values in [0, 1] in RGB
    """
    q = random.randint(quality_range[0], quality_range[1])
    compressed_frames = []

    # cv2 expects (H, W, C) in BGR and uint8 [0, 255]
    frames_np = (frames.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)

    for i in range(frames_np.shape[0]):
        # frames is in RGB, cv2 expects BGR
        bgr_frame = cv2.cvtColor(frames_np[i], cv2.COLOR_RGB2BGR)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), q]
        result, encimg = cv2.imencode('.jpg', bgr_frame, encode_param)
        if result:
            dec_bgr = cv2.imdecode(encimg, 1)
            dec_rgb = cv2.cvtColor(dec_bgr, cv2.COLOR_BGR2RGB)
            compressed_frames.append(dec_rgb)
        else:
            compressed_frames.append(frames_np[i])

    # Convert back to (T, C, H, W) float32 [0, 1] RGB
    compressed_frames = np.stack(compressed_frames, axis=0)
    compressed_tensor = torch.from_numpy(compressed_frames).float() / 255.0
    return compressed_tensor.permute(0, 3, 1, 2)


def simulate_blur(frames, blur_prob=0.3):
    """Applies random Gaussian blur to frames to simulate lossy transmission/resolution decay."""
    if random.random() > blur_prob:
        return frames
    k = random.choice([3, 5])
    frames_np = (frames.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
    blurred_list = [cv2.GaussianBlur(f, (k, k), 0) for f in frames_np]
    blurred_arr = np.stack(blurred_list, axis=0)
    return torch.from_numpy(blurred_arr).float().permute(0, 3, 1, 2) / 255.0


def simulate_sbi_blend(frames, blend_prob=0.25):
    """Simulates self-blended image (SBI) boundary artifacts.
    Generates a feathered central mask and blends slightly altered face colors/textures.
    """
    if random.random() > blend_prob:
        return frames
    T, C, H, W = frames.shape
    y = torch.linspace(-1, 1, H, dtype=frames.dtype).view(1, 1, H, 1).repeat(T, 1, 1, W)
    x = torch.linspace(-1, 1, W, dtype=frames.dtype).view(1, 1, 1, W).repeat(T, 1, H, 1)
    radius = (x ** 2 + (y * 1.2) ** 2)
    mask = torch.clamp(1.0 - (radius - 0.2) / 0.35, 0.0, 1.0)

    color_shift = (torch.rand(1, C, 1, 1, dtype=frames.dtype) - 0.5) * 0.15
    transformed = torch.clamp(frames + color_shift, 0.0, 1.0)
    return frames * (1.0 - mask) + transformed * mask


def augment_video(frames, conf=None, split="train", target_size=224, target_frames=8):
    """
    Applies spatial, frequency, and temporal augmentations to video frames,
    keeping landmark confidence scores synchronously aligned, followed by ImageNet normalization.
    frames: (T, H, W, C) numpy array or torch tensor
    conf: (T,) landmark confidence scores (optional)
    """
    if isinstance(frames, np.ndarray):
        frames = torch.from_numpy(frames).float()
        if frames.max() > 1.0:
            frames = frames / 255.0

    # Convert to (T, C, H, W)
    if frames.shape[-1] == 3:
        frames = frames.permute(0, 3, 1, 2)

    if conf is not None and isinstance(conf, np.ndarray):
        conf = torch.from_numpy(conf).float()

    T, C, H, W = frames.shape

    if split == "train":
        # 1. Temporal frame dropout/jitter
        if T > target_frames:
            max_start = T - target_frames
            start_idx = random.randint(0, max_start)
            stride = random.choice([1, 1, 2]) if max_start >= target_frames else 1
            indices = list(range(start_idx, min(start_idx + target_frames * stride, T), stride))
            if len(indices) < target_frames:
                indices = list(range(start_idx, start_idx + target_frames))
            indices = indices[:target_frames]
            frames = frames[indices]
            if conf is not None:
                conf = conf[indices]
        elif T < target_frames:
            pad_size = target_frames - T
            padding = frames[-1:].repeat(pad_size, 1, 1, 1)
            frames = torch.cat([frames, padding], dim=0)
            if conf is not None:
                conf_pad = conf[-1:].repeat(pad_size)
                conf = torch.cat([conf, conf_pad], dim=0)

        # 2. Simulated Compression & Blur (robustness against lossy codecs)
        if random.random() < 0.4:
            frames = simulate_compression(frames)
        frames = simulate_blur(frames, blur_prob=0.3)

        # 3. Spatial Augmentation (Random Resize + Re-crop)
        scale = random.uniform(1.0, 1.25)
        new_h, new_w = int(target_size * scale), int(target_size * scale)
        frames = F.interpolate(frames, size=(new_h, new_w), mode='bilinear', align_corners=False)

        # Random Crop
        i = torch.randint(0, new_h - target_size + 1, (1,)).item()
        j = torch.randint(0, new_w - target_size + 1, (1,)).item()
        frames = frames[:, :, i:i+target_size, j:j+target_size]

        # Horizontal flip
        if random.random() < 0.5:
            frames = TF.hflip(frames)

    else:
        # Validation/Test: center crop and deterministic sampling
        if T > target_frames:
            start_idx = (T - target_frames) // 2
            frames = frames[start_idx:start_idx + target_frames]
            if conf is not None:
                conf = conf[start_idx:start_idx + target_frames]
        elif T < target_frames:
            pad_size = target_frames - T
            padding = frames[-1:].repeat(pad_size, 1, 1, 1)
            frames = torch.cat([frames, padding], dim=0)
            if conf is not None:
                conf_pad = conf[-1:].repeat(pad_size)
                conf = torch.cat([conf, conf_pad], dim=0)

        frames = F.interpolate(frames, size=(target_size, target_size), mode='bilinear', align_corners=False)

    # 5. Apply ImageNet Normalization
    frames = normalize_imagenet(frames)

    if conf is not None:
        if conf.shape[0] > target_frames:
            conf = conf[:target_frames]
        elif conf.shape[0] < target_frames:
            conf = F.pad(conf, (0, target_frames - conf.shape[0]), value=1.0)
        return frames, conf

    return frames



class SampleDecoder:
    """Top-level picklable sample transformer for WebDataset on Windows multiprocessing."""
    def __init__(self, split="train", bgr_to_rgb=True):
        self.split = split
        self.bgr_to_rgb = bgr_to_rgb

    def __call__(self, sample):
        frames = sample.get("frames.npy", None)
        if frames is None:
            # fallback mock data
            frames = np.zeros((16, 224, 224, 3), dtype=np.uint8)

        # Convert OpenCV BGR to RGB on-the-fly for existing preprocessed shards
        if self.bgr_to_rgb and isinstance(frames, np.ndarray) and frames.ndim == 4 and frames.shape[-1] == 3:
            frames = frames[..., [2, 1, 0]]

        # Confs, labels, manipulation type, and video_id
        conf = sample.get("conf.npy", np.ones((8,)))
        raw_label = sample.get("label.cls", 0)
        raw_manip = sample.get("manip.txt", b"real")
        key = sample.get("__key__", "unknown_000000")
        video_id = key.rsplit("_", 1)[0] if "_" in key else key

        # Apply augmentation, synchronized temporal frame/conf alignment, and ImageNet normalization
        augmented_frames, conf = augment_video(frames, conf=conf, split=self.split)

        if isinstance(raw_label, bytes):
            try:
                label_val = float(raw_label.decode("utf-8").strip())
            except Exception:
                label_val = 0.0
        elif isinstance(raw_label, (str, int)):
            label_val = float(raw_label)
        elif isinstance(raw_label, np.ndarray):
            label_val = float(raw_label.item())
        else:
            label_val = float(raw_label)

        if isinstance(raw_manip, bytes):
            manip_str = raw_manip.decode("utf-8").strip()
        else:
            manip_str = str(raw_manip)

        manip_id = MANIP_TYPE_MAP.get(manip_str, 0)

        return {
            "frames": augmented_frames,
            "conf": conf,
            "label": torch.tensor(label_val, dtype=torch.float32),
            "manip_type": torch.tensor(manip_id, dtype=torch.long),
            "video_id": video_id,
        }


def build_dataloader(cfg, split="train", num_workers=12):
    """
    Builds a PyTorch DataLoader reading from webdataset shards.
    """
    cache_root = getattr(cfg, "cache_root", "data/cache")
    if hasattr(cfg, "get"):
        cache_root = cfg.get("cache_root", "data/cache")
    dataset_name = getattr(cfg, "dataset", "ffpp")
    if hasattr(cfg, "get"):
        dataset_name = cfg.get("dataset", dataset_name)
    batch_size = getattr(cfg, "batch_size", 4)
    if hasattr(cfg, "get"):
        batch_size = cfg.get("batch_size", 4)
    bgr_to_rgb = getattr(cfg, "bgr_to_rgb", True)
    if hasattr(cfg, "get"):
        bgr_to_rgb = cfg.get("bgr_to_rgb", True)

    shard_dir = os.path.join(cache_root, dataset_name)
    shard_files = sorted(glob.glob(os.path.join(shard_dir, f"{split}-*.tar")))
    if shard_files:
        shard_spec = shard_files
    else:
        # WebDataset brace expansion format
        shard_spec = os.path.join(shard_dir, f"{split}-{{000000..000099}}.tar")

    if wds is None:
        raise ImportError("webdataset is not installed. Run 'pip install webdataset' to enable WebDataset shard loading.")

    # Build WebDataset pipeline
    is_train = (split == "train")
    dataset = wds.WebDataset(shard_spec, resampled=is_train, shardshuffle=is_train)
    if is_train:
        shuffle_buf = getattr(cfg, "shuffle_buf", 1000)
        if hasattr(cfg, "get"):
            shuffle_buf = cfg.get("shuffle_buf", 1000)
        if shuffle_buf > 0:
            dataset = dataset.shuffle(shuffle_buf)

    dataset = (
        dataset
        .decode()
        .map(SampleDecoder(split=split, bgr_to_rgb=bgr_to_rgb))
        .batched(batch_size)
    )

    if is_train:
        # If epoch_length is specified, bound the number of batches per epoch
        epoch_length = getattr(cfg, "epoch_length", None)
        if hasattr(cfg, "get"):
            epoch_length = cfg.get("epoch_length", None)
        if epoch_length is not None:
            dataset = dataset.with_epoch(epoch_length)


    loader = DataLoader(
        dataset,
        batch_size=None,     # WebDataset handles batching internally
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
    )
    return loader
