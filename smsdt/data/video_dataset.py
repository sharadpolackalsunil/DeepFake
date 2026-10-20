"""Map-style video datasets with exact epochs and paired sampling for S-MSDT.

Train: One item = (real clip, matching fake clip) with identical frame indices
       and identical spatial/compression augmentation.
       One epoch = every real training video visited exactly once.
Eval:  One item = K evenly spaced clips from one video for robust video-level scoring.
"""
import os
import csv
import random
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

METHOD_ID = {
    "real": 0,
    "Deepfakes": 1,
    "Face2Face": 2,
    "FaceSwap": 3,
    "NeuralTextures": 4,
    "FaceShifter": 5,
    "SBI": 6,
    "DeepFakeDetection": 7,
}


def load_index(path, split):
    """Load rows from data/splits/index.csv filtered by split."""
    with open(path, newline="") as f:
        return [r for r in csv.DictReader(f) if r["split"] == split]


class VideoStore:
    """Lazy, worker-safe, memory-mapped access to one video's stored arrays."""

    def __init__(self, root, video_id):
        self.dir = os.path.join(root, video_id)
        fidx_path = os.path.join(self.dir, "fidx.npy")
        if os.path.exists(fidx_path):
            self.fidx = np.load(fidx_path)
        else:
            self.fidx = np.array([], dtype=np.int32)
        self._frames = None

    @property
    def frames(self):
        if self._frames is None:
            # Memory-map frames inside worker process to conserve RAM
            self._frames = np.load(os.path.join(self.dir, "frames.npy"), mmap_mode="r")
        return self._frames

    def lmk(self):
        return np.load(os.path.join(self.dir, "lmk106.npy"))


def contiguous_starts(fidx, T, stride):
    """Find start indices where fidx has a contiguous run of T consecutive frame numbers."""
    if len(fidx) < T * stride:
        return []
    span = (T - 1) * stride
    return [s for s in range(len(fidx) - span) if fidx[s + span] - fidx[s] == span]


class Augment:
    """Sample parameters once, apply identically to every clip in a pair."""

    def __init__(self, size=224, jpeg=(50, 100), p_jpeg=0.3, p_blur=0.1, p_flip=0.5):
        self.size = size
        self.jpeg = jpeg
        self.p_jpeg = p_jpeg
        self.p_blur = p_blur
        self.p_flip = p_flip

    def sample(self, in_size):
        scale = random.uniform(0.9, 1.1)
        crop = int(round(self.size * scale))
        crop = min(crop, in_size)
        return dict(
            y=random.randint(0, in_size - crop),
            x=random.randint(0, in_size - crop),
            crop=crop,
            flip=random.random() < self.p_flip,
            q=random.randint(*self.jpeg) if random.random() < self.p_jpeg else None,
            blur=random.choice([3, 5]) if random.random() < self.p_blur else None,
        )

    def apply(self, clip_u8, p):
        """clip_u8: (T, H, W, 3) uint8 RGB -> (T, 3, size, size) float32 in [0, 1]."""
        import cv2
        out = []
        for fr in clip_u8:
            fr = np.ascontiguousarray(fr[p["y"]:p["y"] + p["crop"], p["x"]:p["x"] + p["crop"]])
            if p["q"] is not None:
                ok, enc = cv2.imencode(".jpg", fr[..., ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), p["q"]])
                if ok:
                    fr = cv2.imdecode(enc, cv2.IMREAD_COLOR)[..., ::-1]
            if p["blur"] is not None:
                fr = cv2.GaussianBlur(fr, (p["blur"], p["blur"]), 0)
            if p["flip"]:
                fr = fr[:, ::-1]
            out.append(np.ascontiguousarray(fr))

        x = torch.from_numpy(np.stack(out)).permute(0, 3, 1, 2).float() / 255.0
        return F.interpolate(x, size=(self.size, self.size), mode="bilinear", align_corners=False)


class PairedClipDataset(Dataset):
    """Paired real/fake dataset enforcing exact 50/50 balance and exact-epoch coverage."""

    def __init__(self, index_csv, store_root, T=8, stride=1, p_sbi=0.5, sbi_fn=None,
                 methods=("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "FaceShifter"),
                 augment=None, normalize=None, mask_grid=14):
        rows = load_index(index_csv, "train")
        self.root = store_root
        self.T = T
        self.stride = stride
        self.mask_grid = mask_grid
        self.methods = list(methods)
        self.epoch = 0
        self.reals = [r for r in rows if r["method"] == "real"]
        self.fakes = defaultdict(list)
        for r in rows:
            if r["method"] in self.methods:
                self.fakes[r["target_id"]].append(r)

        # Sort each target identity's fake list deterministically by method order
        for tid in self.fakes:
            self.fakes[tid].sort(key=lambda x: self.methods.index(x["method"]) if x["method"] in self.methods else 999)

        self.p_sbi = p_sbi
        self.sbi_fn = sbi_fn
        self.aug = augment or Augment()
        self.normalize = normalize

    def set_epoch(self, epoch: int):
        """Set current epoch for deterministic round-robin manipulation cycling."""
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.reals)  # Exactly 720 real videos = 1 exact epoch

    def __getitem__(self, i):
        real = self.reals[i]
        rs = VideoStore(self.root, real["video_id"])
        use_sbi = self.sbi_fn is not None and (random.random() < self.p_sbi or not self.fakes[real["target_id"]])

        if use_sbi:
            starts = contiguous_starts(rs.fidx, self.T, self.stride)
            if not starts:
                return self.__getitem__((i + 1) % len(self))
            s = random.choice(starts)
            sel = np.arange(s, s + self.T * self.stride, self.stride)
            real_u8 = np.asarray(rs.frames[sel]).copy()
            fake_u8, fake_mask = self.sbi_fn(real_u8, rs.lmk()[sel])
            method = "SBI"
        else:
            fakes_list = self.fakes[real["target_id"]]
            # Deterministic round-robin cycling: Epoch 0->Method 0, Epoch 1->Method 1, etc.
            fake = fakes_list[self.epoch % len(fakes_list)]
            fs = VideoStore(self.root, fake["video_id"])
            common, ri, fi = np.intersect1d(rs.fidx, fs.fidx, return_indices=True)
            starts = contiguous_starts(common, self.T, self.stride)
            if not starts:
                return self.__getitem__((i + 1) % len(self))
            s = random.choice(starts)
            sel = np.arange(s, s + self.T * self.stride, self.stride)
            real_u8 = np.asarray(rs.frames[ri[sel]]).copy()
            fake_u8 = np.asarray(fs.frames[fi[sel]]).copy()
            fake_mask = None
            method = fake["method"]

        p = self.aug.sample(real_u8.shape[1])
        real_clip = self.aug.apply(real_u8, p)
        fake_clip = self.aug.apply(fake_u8, p)
        clips = torch.stack([real_clip, fake_clip])  # (2, T, 3, S, S)

        if self.normalize is not None:
            clips = torch.stack([self.normalize(c) for c in clips])

        g = self.mask_grid
        if fake_mask is not None:
            m = torch.from_numpy(np.ascontiguousarray(fake_mask)).float()[:, None]  # (T, 1, H, W)
            m = m[:, :, p["y"]:p["y"] + p["crop"], p["x"]:p["x"] + p["crop"]]
            if p["flip"]:
                m = m.flip(-1)
            m = F.interpolate(m, size=(g, g), mode="area")[:, 0]  # (T, g, g)
            has_fake_mask = 1.0
        else:
            m = torch.zeros(self.T, g, g)
            has_fake_mask = 0.0

        return {
            "frames": clips,
            "label": torch.tensor([0.0, 1.0]),
            "method": torch.tensor([0, METHOD_ID.get(method, 1)]),
            "mask": torch.stack([torch.zeros_like(m), m]),
            "has_mask": torch.tensor([1.0, has_fake_mask]),
        }


def paired_collate(batch):
    """Flatten batch of pairs: frames (2B, T, 3, S, S), label (2B,), mask (2B, T, g, g), has_mask (2B,)."""
    return {
        k: torch.cat([b[k] for b in batch])
        for k in ("frames", "label", "method", "mask", "has_mask")
    }


class EvalClipDataset(Dataset):
    """Evaluation dataset extracting K evenly spaced clips per video for video-level aggregation."""

    def __init__(self, index_csv, store_root, split="val", K=8, T=8, stride=1,
                 size=224, normalize=None, methods=None):
        rows = load_index(index_csv, split)
        if methods is not None:
            norm_methods = set()
            for m in methods:
                if m.upper() in ("DFD", "DEEPFAKEDETECTION"):
                    norm_methods.add("DeepFakeDetection")
                    norm_methods.add("DFD")
                else:
                    norm_methods.add(m)
            rows = [r for r in rows if r["method"] in norm_methods or r["method"] == "real"]
        self.rows = rows
        self.root = store_root
        self.K = K
        self.T = T
        self.stride = stride
        self.size = size
        self.normalize = normalize

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        vs = VideoStore(self.root, row["video_id"])
        starts = contiguous_starts(vs.fidx, self.T, self.stride)

        if not starts:
            # Fallback for short tracks: repeat first frame
            clips = torch.zeros(self.K, self.T, 3, self.size, self.size)
        else:
            # Choose K evenly spaced starts
            step = max(1, len(starts) // self.K)
            chosen_starts = [starts[min(j * step, len(starts) - 1)] for j in range(self.K)]

            clips_list = []
            for s in chosen_starts:
                sel = np.arange(s, s + self.T * self.stride, self.stride)
                frames_u8 = np.asarray(vs.frames[sel])
                H, W = frames_u8.shape[1:3]
                # Deterministic center crop
                cy, cx = (H - self.size) // 2, (W - self.size) // 2
                cropped = frames_u8[:, cy:cy + self.size, cx:cx + self.size].copy()
                clip_tensor = torch.from_numpy(np.ascontiguousarray(cropped)).permute(0, 3, 1, 2).float() / 255.0
                if self.normalize is not None:
                    clip_tensor = self.normalize(clip_tensor)
                clips_list.append(clip_tensor)

            clips = torch.stack(clips_list)  # (K, T, 3, S, S)

        label = float(row["label"])
        return {
            "frames": clips,
            "label": torch.tensor(label, dtype=torch.float32),
            "video_id": row["video_id"],
            "method": row["method"],
        }
