# S‑MSDT on DGX Spark — Full Implementation & Training Plan

**Architecture:** Scale‑Invariance + Multi‑Scale Feature Differences + Wavelet Frequency Guard (S‑MSDT)
**Target hardware:** NVIDIA DGX Spark (GB10 Grace Blackwell Superchip)
**Goal:** Train, optimize, and deploy S‑MSDT for real‑time (≈8.1 ms/chunk, 123+ FPS) deepfake detection, end‑to‑end on a single Spark unit.

---

## 0. Hardware Reality Check (read this before writing any code)

DGX Spark is a **single‑node, memory‑bandwidth‑bound developer workstation**, not a multi‑GPU training cluster. Plan around its actual numbers instead of datacenter assumptions:

| Component | Spec |
|---|---|
| SoC | GB10 Grace Blackwell Superchip (NVIDIA + MediaTek, TSMC 3nm) |
| CPU | 20 Arm cores — 10× Cortex‑X925 (perf, ~4.0 GHz) + 10× Cortex‑A725 (eff, ~2.8 GHz), ARMv9.2‑A w/ SVE2, BF16, i8mm |
| GPU | Blackwell, 48 SMs, 6,144 CUDA cores, compute capability **sm_121** |
| Memory | 128 GB LPDDR5X, **unified/coherent** across CPU+GPU via NVLink‑C2C, ~273 GB/s aggregate bandwidth |
| Compute | ~1 PFLOP FP4 (sparse); no dedicated FP64 datacenter silicon |
| TDP | 140 W SoC |
| OS | DGX OS (Ubuntu 24.04 LTS base) |
| Networking | 10GbE (+ optional 200 Gbps ConnectX for multi‑Spark clustering) |

**Consequences for this project:**

1. **273 GB/s is the ceiling for everything** — data loading, activations, optimizer states, and weights all fight over the same bandwidth as the CPU. Treat the whole pipeline as bandwidth‑bound, not just the GPU kernels.
2. **This is one GPU.** There is no intra‑node data parallelism to exploit. Multi‑Spark scaling is only possible node‑to‑node over the 200 Gbps NIC (optional, covered in §11.5) — do not architect around it as a default.
3. **sm_121 is new.** Stock PyTorch/TensorRT wheels lag hardware releases. You will likely need NVIDIA's DGX Spark playbook containers or nightly builds — pin versions and verify `torch.cuda.get_device_capability()` reports `(12, 1)` before trusting any benchmark.
4. **This box is best used as:** (a) the full preprocessing + prototyping environment, (b) the fine‑tuning/distillation rig for a model pretrained faster elsewhere or trained here at reduced scale, and (c) the **exact target for latency optimization**, since your 8.1 ms budget is defined for this chip. Full from‑scratch pretraining on FF++ + DFDC + Celeb‑DF + WildDeepfake at large batch sizes will take **days, not hours** — §12 gives honest wall‑clock estimates so you don't get surprised.

---

## 1. Environment Setup

### 1.1 Base OS & driver check

```bash
# Confirm DGX OS + driver stack
cat /etc/os-release
nvidia-smi
# Architecture MUST read "Blackwell"; confirm compute capability later in Python
```

### 1.2 Container strategy (recommended over bare‑metal)

Use NVIDIA's NGC PyTorch container built for GB10 (aarch64 + sm_121), not a generic x86 wheel — DGX Spark is Arm, so any package without an aarch64 build will fail or silently fall back to CPU.

```bash
# Pull the NGC PyTorch container (check NGC catalog for the current DGX Spark / GB10 tag)
docker pull nvcr.io/nvidia/pytorch:<latest-gb10-tagged-release>

docker run --gpus all -it --rm \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v $(pwd):/workspace -w /workspace \
  nvcr.io/nvidia/pytorch:<tag> bash
```

If you must build bare‑metal instead:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip

# Use the PyTorch build channel that ships sm_121 kernels for aarch64 —
# check pytorch.org's "Blackwell / GB10" install instructions at build time,
# a stock stable wheel from PyPI may predate sm_121 support.
pip install torch torchvision torchaudio --index-url <gb10-compatible-index>

python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_capability())"
# Expect: True (12, 1)
```

### 1.3 Core dependencies

```bash
pip install timm==1.0.* opencv-python-headless==4.10.* \
    onnx onnxruntime-gpu tensorrt \
    pytorch-wavelets \
    einops webdataset lmdb pyarrow \
    albumentations decord \
    wandb tqdm hydra-core omegaconf \
    scikit-learn

# Face detection / tracking
pip install insightface   # provides SCRFD
pip install bytetrack      # or vendor the reference ByteTrack repo — pip package availability varies
```

> **aarch64 gotcha:** `decord`, some `onnxruntime-gpu` wheels, and older `insightface` builds may not publish aarch64 binaries. Where a wheel is missing, build from source inside the NGC container (which has the toolchain preinstalled) rather than fighting pip on bare metal.

### 1.4 Repo layout

```
smsdt/
├── configs/
│   ├── data.yaml
│   ├── model.yaml
│   └── train.yaml
├── data/
│   ├── raw/                  # symlinks to dataset roots
│   ├── cache/                # preprocessed crops (webdataset shards / LMDB)
│   └── splits/               # subject-disjoint train/val/test manifests
├── smsdt/
│   ├── preprocess/
│   │   ├── detect_track.py       # SCRFD + ByteTrack
│   │   ├── align_crop.py         # 5-point alignment, 224x224 crop
│   │   └── build_shards.py       # webdataset packer
│   ├── data/
│   │   └── dataset.py            # chunked T=8 dataset + on-the-fly DWT
│   ├── models/
│   │   ├── rgb_branch.py         # EfficientNetB4-Lite + FPN
│   │   ├── wavelet_branch.py     # Haar DWT stem
│   │   ├── tokenizer.py          # PSIT + frequency stem + fusion
│   │   ├── nmsss_vit.py          # decomposed spatial/temporal ViT
│   │   └── smsdt.py              # full assembled model
│   ├── train.py
│   ├── eval.py
│   └── export/
│       ├── to_onnx.py
│       └── build_trt_engine.py
├── scripts/
│   ├── run_preprocess.sh
│   ├── run_train.sh
│   └── run_export.sh
└── outputs/
    ├── checkpoints/
    └── logs/
```

---

## 2. Datasets

| Dataset | Purpose | Notes |
|---|---|---|
| FaceForensics++ (FF++, c23/c40) | Primary train set, 4 manipulation types | Use both c23 (light compression) and c40 (heavy) to teach the wavelet branch to survive compression |
| Celeb‑DF v2 | Cross‑dataset generalization eval | Held out entirely from training |
| DFDC (full or sampled subset) | Scale + diversity, adversarial perturbations | Large — sample a class‑balanced subset first if disk/time constrained |
| WildDeepfake | In‑the‑wild generalization eval | Held out |
| DeeperForensics‑1.0 | Perturbation robustness (blur, noise, compression) | Optional augmentation source |

**Splitting rule:** split by **identity/subject**, never by clip, so no identity leaks across train/val/test. Store manifests as CSVs in `data/splits/` with columns `video_id, subject_id, label, manipulation_type, split`.

**Storage estimate:** raw FF++ (all methods, c23+c40) + DFDC subset + Celeb‑DF + WildDeepfake will comfortably exceed 1–2 TB. Confirm the Spark's NVMe (up to 4 TB on Founder's Edition config) before committing to "download everything."

---

## 3. Phase 1 — Preprocessing Pipeline (implementation)

Goal: turn raw video into cached, ready‑to‑load `T=8` aligned face chunks so training never has to run SCRFD/ByteTrack live (that CPU work would eat into the 273 GB/s budget you need for the GPU).

### 3.1 Detection + tracking

```python
# smsdt/preprocess/detect_track.py
import cv2
from insightface.app import FaceAnalysis
# ByteTrack reference implementation vendored under third_party/bytetrack
from third_party.bytetrack.byte_tracker import BYTETracker

detector = FaceAnalysis(name="scrfd_10g_bnkps", providers=["CUDAExecutionProvider"])
detector.prepare(ctx_id=0, det_size=(320, 320))
tracker = BYTETracker(track_thresh=0.5, match_thresh=0.8, frame_rate=30)

def detect_and_track(video_path, out_path):
    cap = cv2.VideoCapture(video_path)
    records = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        faces = detector.get(frame)
        dets = [(f.bbox, f.det_score, f.kps) for f in faces]
        tracks = tracker.update(dets, frame.shape[:2])
        for t in tracks:
            records.append({
                "frame": frame_idx, "track_id": t.track_id,
                "bbox": t.tlbr.tolist(), "kps": t.kps.tolist(),
                "conf": float(t.score),
            })
        frame_idx += 1
    cap.release()
    return records
```

### 3.2 Alignment + crop

```python
# smsdt/preprocess/align_crop.py
import cv2
import numpy as np

REF_5PT = np.array([
    [38.2946, 51.6963], [73.5318, 51.5014],
    [56.0252, 71.7366], [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)  # standard ArcFace-style template, scaled to 112 -> resized to 224

def align_face(frame, kps, out_size=224):
    dst = REF_5PT * (out_size / 112.0)
    M, _ = cv2.estimateAffinePartial2D(kps.astype(np.float32), dst, method=cv2.LMEDS)
    aligned = cv2.warpAffine(frame, M, (out_size, out_size), borderValue=0.0)
    return aligned
```

### 3.3 Chunking + caching (webdataset shards)

Pack aligned `224x224` crops into fixed `T=8` chunks with sliding stride (e.g., stride=4 for training density, stride=8 non‑overlapping for eval), plus per‑frame SCRFD confidence, into webdataset `.tar` shards so the dataloader does sequential I/O instead of random small‑file reads (critical given LPDDR5X bandwidth is shared with the CPU).

```python
# smsdt/preprocess/build_shards.py
import webdataset as wds
import numpy as np, io

def write_shard(chunks, shard_path):
    with wds.TarWriter(shard_path) as sink:
        for i, c in enumerate(chunks):
            sink.write({
                "__key__": f"{c['video_id']}_{c['start_frame']:06d}",
                "frames.npy": c["frames"].astype(np.uint8),      # (T,224,224,3)
                "conf.npy": c["conf"].astype(np.float32),        # (T,)
                "label.cls": c["label"],                          # 0 real / 1 fake
                "manip.txt": c["manipulation_type"],
            })
```

Run with sharding parallelized across the 20 Arm cores, but **reserve headroom** — don't launch 20 workers; 12–14 leaves bandwidth for the GPU‑side wavelet/CNN compute during any concurrent GPU‑bound step, and avoids starving the CPU cores driving PCIe/NVMe I/O.

```bash
python -m smsdt.preprocess.build_shards \
  --dataset ffpp --workers 14 --chunk-stride 4 --out data/cache/ffpp/
```

---

## 4. Phase 2–5 — Model Implementation

### 4.1 Branch A — RGB stream (EfficientNetB4‑Lite + FPN)

```python
# smsdt/models/rgb_branch.py
import torch, torch.nn as nn
import timm

class RGBBranch(nn.Module):
    def __init__(self, out_channels=256):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b4", pretrained=True, features_only=True,
            out_indices=(2, 3, 4),  # roughly P2, P3, P4 strides
        )
        chs = self.backbone.feature_info.channels()
        self.lateral = nn.ModuleList([nn.Conv2d(c, out_channels, 1) for c in chs])
        self.smooth = nn.ModuleList([nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in chs])

    def forward(self, x):
        feats = self.backbone(x)                    # [P2, P3, P4], each (B, C, H, W)
        laterals = [l(f) for l, f in zip(self.lateral, feats)]
        # top-down fusion
        for i in range(len(laterals) - 1, 0, -1):
            up = nn.functional.interpolate(laterals[i], size=laterals[i - 1].shape[-2:], mode="nearest")
            laterals[i - 1] = laterals[i - 1] + up
        fpn_out = [s(l) for s, l in zip(self.smooth, laterals)]
        return fpn_out   # list of (B, 256, H_i, W_i)
```

> "Lite" note: swap `efficientnet_b4` for `efficientnet_lite0`/a distilled B4 checkpoint (or apply structured pruning + INT8 QAT later, §10) once you have a working baseline — start with the standard timm B4 for correctness, then shrink for the latency budget.

### 4.2 Branch B — Wavelet stream (2D Haar DWT)

Implement Haar DWT as fixed (non‑trainable) depthwise convolutions so it runs as a GPU kernel, not a CPU numpy op — this keeps it inside the same CUDA stream as everything else.

```python
# smsdt/models/wavelet_branch.py
import torch, torch.nn as nn

class HaarDWT2D(nn.Module):
    """Single-level 2D Haar DWT. Input (B,3,H,W) -> LL,LH,HL,HH each (B,3,H/2,W/2)."""
    def __init__(self):
        super().__init__()
        ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        lh = torch.tensor([[0.5, 0.5], [-0.5, -0.5]])
        hl = torch.tensor([[0.5, -0.5], [0.5, -0.5]])
        hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]])
        filt = torch.stack([ll, lh, hl, hh]).unsqueeze(1)   # (4,1,2,2)
        filt = filt.repeat(3, 1, 1, 1)                       # apply per RGB channel -> (12,1,2,2)
        self.register_buffer("filt", filt)

    def forward(self, x):
        B, C, H, W = x.shape
        out = nn.functional.conv2d(x, self.filt, stride=2, groups=C)  # (B, C*4, H/2, W/2)
        out = out.view(B, C, 4, H // 2, W // 2)
        ll, lh, hl, hh = out.unbind(dim=2)
        return ll, lh, hl, hh

class WaveletBranch(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        self.dwt = HaarDWT2D()
        self.proj = nn.Sequential(
            nn.Conv2d(9, 64, 3, padding=1), nn.GELU(),   # 9 = 3 bands (LH,HL,HH) x 3 RGB channels
            nn.Conv2d(64, embed_dim, 3, padding=1),
        )

    def forward(self, x):
        _, lh, hl, hh = self.dwt(x)          # discard LL per spec
        hf = torch.cat([lh, hl, hh], dim=1)  # (B, 9, H/2, W/2)
        return self.proj(hf)                 # (B, embed_dim, H/2, W/2)
```

### 4.3 Tokenizer + cross‑domain fusion

```python
# smsdt/models/tokenizer.py
import torch, torch.nn as nn

class PSIT(nn.Module):
    """Pyramidal Scale-Invariant Tokenizer: adaptive-pool each FPN level to 7x7, then merge."""
    def __init__(self, embed_dim=256, grid=7):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((grid, grid))
        self.merge = nn.Conv2d(embed_dim, embed_dim, 1)

    def forward(self, fpn_feats):
        pooled = [self.pool(f) for f in fpn_feats]     # each (B, D, 7, 7)
        summed = torch.stack(pooled, dim=0).sum(0)
        tokens = self.merge(summed).flatten(2).transpose(1, 2)   # (B, 49, D)
        return tokens

class FrequencyStem(nn.Module):
    def __init__(self, embed_dim=256, grid=7):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((grid, grid))

    def forward(self, wavelet_feat):
        pooled = self.pool(wavelet_feat)
        return pooled.flatten(2).transpose(1, 2)   # (B, 49, D)

class GatedFusion(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(embed_dim * 2, embed_dim), nn.Sigmoid())
        self.out = nn.Linear(embed_dim * 2, embed_dim)

    def forward(self, z_rgb, z_dwt):
        cat = torch.cat([z_rgb, z_dwt], dim=-1)
        g = self.gate(cat)
        fused = g * z_rgb + (1 - g) * z_dwt
        return self.out(torch.cat([fused, cat], dim=-1)[..., :fused.shape[-1]]) if False else fused
```

### 4.4 N‑MSSS decomposed ViT (spatial + temporal, with confidence‑gated frame differencing)

```python
# smsdt/models/nmsss_vit.py
import torch, torch.nn as nn
from einops import rearrange

class TransformerBlock(nn.Module):
    def __init__(self, dim=256, heads=8, mlp_ratio=4.0, drop=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim), nn.Dropout(drop),
        )

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x

class NMSSSBlock(nn.Module):
    """One spatial-attn -> temporal-diff -> temporal-attn stage."""
    def __init__(self, dim=256, heads=8):
        super().__init__()
        self.spatial = TransformerBlock(dim, heads)
        self.diff_proj = nn.Conv1d(dim * 2, dim, 1)   # applied per-token across the concat([f_t, f_t - f_{t-1}])
        self.temporal = TransformerBlock(dim, heads)

    def forward(self, x, conf):
        """
        x:    (B, T, N, D)  fused tokens per frame
        conf: (B, T)        SCRFD landmark confidence per frame
        """
        B, T, N, D = x.shape

        # --- Spatial self-attention: intra-frame ---
        x = rearrange(x, "b t n d -> (b t) n d")
        x = self.spatial(x)
        x = rearrange(x, "(b t) n d -> b t n d", b=B, t=T)

        # --- L2-normalized temporal subtraction (N-MSSS) ---
        x_norm = nn.functional.normalize(x, dim=-1)
        prev = torch.cat([x_norm[:, :1], x_norm[:, :-1]], dim=1)   # shift; first frame diffs against itself
        delta = x_norm - prev                                       # (B, T, N, D)
        cat = torch.cat([x_norm, delta], dim=-1)                    # (B, T, N, 2D)
        cat = rearrange(cat, "b t n d2 -> (b n) d2 t")
        gated = self.diff_proj(cat)                                  # (B*N, D, T)
        gated = rearrange(gated, "(b n) d t -> b t n d", b=B, n=N)
        conf_w = conf.view(B, T, 1, 1)
        x = x + gated * conf_w   # residual, gated by detector confidence so shaky-track frames are down-weighted

        # --- Temporal self-attention: inter-frame, per spatial location ---
        x = rearrange(x, "b t n d -> (b n) t d")
        x = self.temporal(x)
        x = rearrange(x, "(b n) t d -> b t n d", b=B, n=N)
        return x
```

### 4.5 Full assembled model

```python
# smsdt/models/smsdt.py
import torch, torch.nn as nn
from einops import rearrange
from .rgb_branch import RGBBranch
from .wavelet_branch import WaveletBranch
from .tokenizer import PSIT, FrequencyStem, GatedFusion
from .nmsss_vit import NMSSSBlock

class SMSDT(nn.Module):
    def __init__(self, embed_dim=256, depth=4, heads=8, num_classes=1):
        super().__init__()
        self.rgb_branch = RGBBranch(out_channels=embed_dim)
        self.wavelet_branch = WaveletBranch(embed_dim=embed_dim)
        self.psit = PSIT(embed_dim=embed_dim)
        self.freq_stem = FrequencyStem(embed_dim=embed_dim)
        self.fusion = GatedFusion(embed_dim=embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, 49, embed_dim))
        self.blocks = nn.ModuleList([NMSSSBlock(embed_dim, heads) for _ in range(depth)])
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim), nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(), nn.Linear(embed_dim // 2, num_classes),
        )

    def forward(self, frames, conf):
        """
        frames: (B, T, 3, 224, 224)
        conf:   (B, T)
        returns: (B,) chunk logit
        """
        B, T = frames.shape[:2]
        x = rearrange(frames, "b t c h w -> (b t) c h w")

        fpn = self.rgb_branch(x)                  # list of (B*T, D, h, w)
        z_rgb = self.psit(fpn)                     # (B*T, 49, D)

        wv = self.wavelet_branch(x)                # (B*T, D, H/2, W/2)
        z_dwt = self.freq_stem(wv)                  # (B*T, 49, D)

        z = self.fusion(z_rgb, z_dwt)               # (B*T, 49, D)
        z = rearrange(z, "(b t) n d -> b t n d", b=B, t=T)
        z = z + self.pos_embed

        for blk in self.blocks:
            z = blk(z, conf)

        pooled = z.mean(dim=2)          # (B, T, D) — collapse spatial tokens
        pooled = pooled.mean(dim=1)     # (B, D)   — collapse temporal (or use a learned temporal CLS token instead)
        logit = self.head(pooled).squeeze(-1)
        return logit
```

---

## 5. Training Configuration

### 5.1 Loss

```python
# focal loss to handle real/fake class imbalance across mixed datasets
import torch.nn.functional as F

def focal_bce(logits, targets, alpha=0.25, gamma=2.0):
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pt = p * targets + (1 - p) * (1 - targets)
    focal = ((1 - pt) ** gamma) * ce
    alpha_w = alpha * targets + (1 - alpha) * (1 - targets)
    return (alpha_w * focal).mean()
```

Optional auxiliary loss: a supervised contrastive term on the pooled wavelet tokens (`z_dwt`) alone, pulling same‑manipulation‑type fakes together — helps the frequency branch specialize instead of being drowned out by the (usually stronger) RGB gradient.

### 5.2 Optimizer / schedule

```yaml
# configs/train.yaml
optimizer: adamw
lr: 3e-4
weight_decay: 0.05
betas: [0.9, 0.999]
schedule: cosine
warmup_steps: 2000
total_steps: 150000
grad_clip: 1.0
label_smoothing: 0.05
```

### 5.3 Precision & memory strategy for GB10

- **Use `bfloat16` autocast**, not fp16 — Blackwell's Arm+CUDA combo handles BF16 natively and it avoids fp16 loss‑scaling headaches; save NVFP4/FP8 for the *inference* export (§10), not training.
- **Gradient checkpointing** on the ViT blocks (`torch.utils.checkpoint`) — trades compute for the activation memory you're sharing with the CPU side of the unified pool.
- **Batch size is bandwidth‑limited, not just capacity‑limited.** 128 GB sounds large, but at 273 GB/s you will be I/O‑bound well before you're capacity‑bound for a model this size. Start small and profile:

```python
# smsdt/train.py (core loop)
import torch, wandb
from torch.utils.checkpoint import checkpoint
from smsdt.models.smsdt import SMSDT
from smsdt.data.dataset import build_dataloader

def main(cfg):
    device = "cuda"
    model = SMSDT(embed_dim=cfg.model.embed_dim, depth=cfg.model.depth).to(device)
    model = torch.compile(model, mode="max-autotune")  # verify sm_121 support in your torch build first

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.train.lr, total_steps=cfg.train.total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=False)   # bf16 needs no loss scaling

    train_loader = build_dataloader(cfg.data, split="train", num_workers=12)
    val_loader = build_dataloader(cfg.data, split="val", num_workers=6)

    wandb.init(project="smsdt-dgx-spark", config=cfg)
    step = 0
    accum_steps = cfg.train.grad_accum

    model.train()
    for epoch in range(cfg.train.epochs):
        for i, batch in enumerate(train_loader):
            frames = batch["frames"].to(device, non_blocking=True)   # (B,T,3,224,224)
            conf = batch["conf"].to(device, non_blocking=True)       # (B,T)
            labels = batch["label"].float().to(device, non_blocking=True)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(frames, conf)
                loss = focal_bce(logits, labels) / accum_steps

            loss.backward()
            if (i + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                step += 1

            if step % 50 == 0:
                wandb.log({"train/loss": loss.item() * accum_steps, "lr": sched.get_last_lr()[0]}, step=step)

        val_auc = evaluate(model, val_loader, device)
        wandb.log({"val/auc": val_auc, "epoch": epoch})
        torch.save(model.state_dict(), f"outputs/checkpoints/epoch{epoch}.pt")
```

### 5.4 Practical batch‑size / accumulation guidance

| Config | Suggested start point |
|---|---|
| Per‑step micro‑batch (chunks of T=8, 224×224) | 4–8 |
| Gradient accumulation | 8–16 (effective batch 32–128) |
| Dataloader workers | 10–12 (leave headroom on the 20 Arm cores for the OS + GPU feeder threads) |
| `pin_memory` | True, but monitor host RAM pressure since it's the *same* pool the GPU uses |

Profile with `torch.cuda.memory_summary()` and `nsys profile` early — don't guess. Because CPU and GPU share the memory pool and bandwidth, an aggressive dataloader can visibly slow down GPU‑side training steps in a way it wouldn't on a discrete‑VRAM machine.

### 5.5 Augmentation

- Re‑encode at random bitrates/CRF (H.264) to teach the wavelet branch to survive real‑world compression, matching FF++'s c23/c40 splits.
- Random resize + re‑crop (breaks any fixed‑scale shortcut, reinforcing why PSIT's adaptive pooling matters).
- Temporal frame dropout / mild frame‑rate jitter (stress‑tests the temporal attention).
- Avoid heavy color/geometric augmentation on the *wavelet input path specifically* — you don't want to destroy the high‑frequency artifacts you're trying to detect. Apply photometric augmentation before the DWT split only if it approximates realistic capture noise, not synthetic corruption.

### 5.6 Curriculum

1. **Stage 1 (single‑dataset warm‑up):** FF++ only, c23, all 4 manipulation types — establishes a working baseline fast.
2. **Stage 2 (compression robustness):** add c40 + re‑encoding augmentation.
3. **Stage 3 (scale + diversity):** mix in DFDC subset, keep Celeb‑DF/WildDeepfake fully held out.
4. **Stage 4 (fine‑tune for Top‑K aggregation):** switch the training objective to operate on full videos with Top‑K chunk selection (§5.7) so train‑time objective matches inference‑time aggregation.

### 5.7 Matching training to Top‑K inference aggregation

```python
def video_level_loss(chunk_logits, video_label, k=5):
    """chunk_logits: (num_chunks,) for one video."""
    topk = torch.topk(chunk_logits, k=min(k, chunk_logits.numel())).values
    video_logit = topk.mean()
    return F.binary_cross_entropy_with_logits(video_logit.unsqueeze(0), video_label.unsqueeze(0))
```

Train Stage 4 with this video‑level Top‑K loss (sampling ~16–32 chunks per video per step) so the network is optimized for exactly the aggregation rule used at inference, rather than a per‑chunk objective that's only a proxy for it.

---

## 6. Evaluation Protocol

```python
# smsdt/eval.py
from sklearn.metrics import roc_auc_score

def evaluate(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for batch in loader:
            frames = batch["frames"].to(device)
            conf = batch["conf"].to(device)
            logits = model(frames, conf)
            all_logits.append(logits.float().cpu())
            all_labels.append(batch["label"].cpu())
    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    return roc_auc_score(labels, logits)
```

Report, at minimum:
- **Per‑dataset AUC / EER / accuracy** (FF++ c23, FF++ c40, DFDC, Celeb‑DF, WildDeepfake)
- **Cross‑manipulation generalization matrix** (train on FF++ subset X, test on manipulation types the model never saw)
- **Video‑level metrics using Top‑K aggregation** vs. naive full‑average, to justify the aggregation choice empirically
- **Perturbation robustness curve** (accuracy vs. JPEG quality / blur sigma / resize factor) using DeeperForensics perturbations

---

## 7. Realistic Wall‑Clock Expectations on One DGX Spark

These are planning‑order‑of‑magnitude numbers, not benchmarks — validate on your actual data pipeline early with a short profiling run rather than trusting any fixed number here.

| Phase | Rough expectation |
|---|---|
| Preprocessing (SCRFD+ByteTrack+align) for FF++ full set | Hours, dominated by CPU video decode — parallelize across Arm cores, cache aggressively |
| Stage 1 warm‑up (FF++ c23 only) to a usable baseline | Likely 1–2 days of wall clock at modest batch size |
| Full curriculum (Stages 1–4, all datasets) | Multiple days to ~1–2 weeks depending on effective batch size and how much of DFDC you include |
| TensorRT export + calibration | Minutes to low hours |
| Full eval suite across 5 datasets | Hours |

If full‑scale training time is unacceptable, the two levers that matter most given the 273 GB/s ceiling are: (1) **reduce token/sequence length** (smaller PSIT grid, e.g. 5×5) rather than reducing depth first, since attention over tokens is the main bandwidth consumer, and (2) **distill** from a larger model trained once on cloud/datacenter GPUs, then use the Spark purely for fine‑tuning + quantization + latency work, which plays to its actual strengths.

---

## 8. Latency Optimization for the 8.1 ms / 123 FPS Target

This is where DGX Spark is genuinely the *right* box, since sm_121 + TensorRT is exactly your deployment target.

### 8.1 Export to ONNX

```python
# smsdt/export/to_onnx.py
import torch
from smsdt.models.smsdt import SMSDT

model = SMSDT().eval().cuda()
model.load_state_dict(torch.load("outputs/checkpoints/best.pt"))

dummy_frames = torch.randn(1, 8, 3, 224, 224, device="cuda")
dummy_conf = torch.ones(1, 8, device="cuda")

torch.onnx.export(
    model, (dummy_frames, dummy_conf), "outputs/smsdt.onnx",
    input_names=["frames", "conf"], output_names=["logit"],
    opset_version=18, dynamic_axes={"frames": {0: "batch"}, "conf": {0: "batch"}},
)
```

### 8.2 Build a TensorRT engine targeting sm_121, with FP8/NVFP4 calibration

```bash
trtexec --onnx=outputs/smsdt.onnx \
  --saveEngine=outputs/smsdt_fp8.engine \
  --fp8 --int8 --calib=outputs/calib_cache.bin \
  --shapes=frames:1x8x3x224x224,conf:1x8 \
  --useCudaGraph --noDataTransfers
```

- Calibrate INT8/FP8 on a representative held‑out sample (~500–1000 chunks spanning real + all manipulation types) — an unrepresentative calibration set will quietly bias detection thresholds.
- Keep an FP16/BF16 fallback engine for the wavelet branch specifically if aggressive quantization visibly degrades high‑frequency artifact sensitivity in your ablations — the whole point of that branch is precision on subtle signals, so quantize it last and verify with the eval suite in §6, not just latency numbers.
- Use CUDA Graphs (`--useCudaGraph`) to cut per‑chunk kernel‑launch overhead, which matters proportionally more at your target ~123 FPS.

### 8.3 Benchmark against the phase budget

```bash
trtexec --loadEngine=outputs/smsdt_fp8.engine \
  --shapes=frames:1x8x3x224x224,conf:1x8 \
  --iterations=500 --avgRuns=100 --warmUp=200
```

Break the measured latency down per phase and compare against the target budget from the architecture spec:

| Phase | Budget |
|---|---|
| Ingestion/preprocess (SCRFD+ByteTrack+align, also export to its own TensorRT engine) | ~2.0 ms |
| Dual‑stream feature extraction (RGB + wavelet) | ~3.5 ms |
| Tokenization + fusion | ~0.3 ms |
| N‑MSSS decomposed ViT | ~2.2 ms |
| Output + aggregation | ~0.1 ms |
| **Total** | **~8.1 ms** |

If any phase blows its budget, the usual GB10‑specific culprits are: (a) the CNN backbone still running at fp32/bf16 instead of quantized, (b) attention sequence length larger than necessary (revisit the 7×7 grid), or (c) host↔device copies that weren't eliminated (`--noDataTransfers` and keeping SCRFD/ByteTrack on‑GPU end‑to‑end matters here since CPU↔GPU traffic shares the same 273 GB/s pool as everything else).

---

## 9. Deployment on DGX Spark

### 9.1 Local serving

```bash
# Package SCRFD engine, alignment kernel, and S-MSDT engine as a Triton ensemble
# so the whole pipeline runs as one scheduled DAG on-device.
docker run --gpus all -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v $(pwd)/outputs/triton_repo:/models \
  nvcr.io/nvidia/tritonserver:<gb10-tagged-release> \
  tritonserver --model-repository=/models
```

Triton `ensemble` config chains: `scrfd_bytetrack_engine → align_crop_kernel → smsdt_engine → topk_aggregator`.

### 9.2 Standalone runtime (no Triton) for a single‑box demo

A lighter option: a Python service using `tensorrt` + `pycuda`/`cuda-python` directly, reading from a video/RTSP source, running the four exported engines in sequence with CUDA streams, and exposing a simple REST/gRPC endpoint. Preferable if you want to avoid Triton's overhead for a single‑stream edge deployment.

---

## 10. Monitoring, Checkpointing, Reproducibility

- **Experiment tracking:** Weights & Biases (or TensorBoard if fully offline) — log loss, per‑dataset AUC, LR, gradient norm, and GPU memory/utilization (`nvidia-smi dmon` piped into your logger) every N steps.
- **Checkpointing:** save every epoch + best‑val‑AUC checkpoint; store the exact dataset manifest hash and augmentation config alongside each checkpoint so results are reproducible.
- **Config management:** Hydra/OmegaConf configs under `configs/`, one YAML per (data, model, train) axis, composed at run time — makes the Stage 1→4 curriculum in §5.6 trivial to express as config overrides rather than code branches.

---

## 11. Scaling Beyond One Spark (optional)

### 11.1 If you have 2+ DGX Spark units

The 200 Gbps ConnectX NIC option makes multi‑node DDP viable across Sparks for the training phase only (not needed for inference/latency work, which should stay single‑box to match the real deployment target):

```bash
torchrun --nnodes=2 --nproc_per_node=1 \
  --rdzv_backend=c10d --rdzv_endpoint=<node0_ip>:29500 \
  smsdt/train.py --config configs/train.yaml
```

Expect near‑linear scaling on compute‑bound phases (the ViT blocks) but sub‑linear on data loading unless your webdataset shards are also distributed across each node's local NVMe.

### 11.2 If you only have one Spark

Prefer the distillation route from §7 over trying to brute‑force full‑scale multi‑dataset training in one pass — pretrain (or partially train) a larger teacher off‑box if cloud/datacenter access is available, distill into the S‑MSDT student on the Spark, then spend the Spark's real advantage — CUDA‑native local iteration — on the latency/quantization work in §8, which is genuinely faster to iterate on locally than in the cloud.

---

## 12. Summary Checklist

- [ ] Verify `torch.cuda.get_device_capability() == (12, 1)` before trusting any benchmark
- [ ] Preprocess + cache all datasets to webdataset shards before touching the training loop
- [ ] Get Stage 1 (FF++ c23 only) to a working, sane‑looking AUC before adding complexity
- [ ] Match train‑time objective to inference‑time Top‑K aggregation (Stage 4 / §5.7)
- [ ] Hold Celeb‑DF and WildDeepfake out completely for generalization testing
- [ ] Export → TensorRT → benchmark against the 8.1 ms phase budget table (§8.3), iterate quantization per‑branch, not globally
- [ ] Deploy via Triton ensemble or a standalone TensorRT runtime service
