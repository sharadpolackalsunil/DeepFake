# S-MSDT: Scale-Invariant Multi-Scale Deepfake Transformer

**Architecture:** Scale-Invariance + Multi-Scale Feature Differences + Wavelet Frequency Guard  
**Target Hardware:** NVIDIA DGX Spark (GB10 Grace Blackwell Superchip)  
**Goal:** Real-time deepfake detection at ≈8.1 ms/chunk (123+ FPS)

---

## Project Structure

```
DEEPFAKE/
├── configs/
│   ├── data.yaml              # Dataset paths, batch size, chunk config
│   ├── model.yaml             # Model architecture hyperparameters
│   └── train.yaml             # Training, loss, optimizer config
├── data/
│   ├── raw/                   # Downloaded raw videos (symlinks or direct)
│   │   ├── ffpp_c23/          # FF++ c23 compression
│   │   └── ffpp_c40/          # FF++ c40 compression
│   ├── cache/                 # Preprocessed webdataset shards
│   └── splits/                # Subject-disjoint train/val/test manifests
├── Download_ff/
│   ├── download1.py           # FaceForensics++ v2 downloader (primary)
│   └── download2.py           # FaceForensics v1 downloader (supplementary)
├── smsdt/
│   ├── models/
│   │   ├── rgb_branch.py      # EfficientNetB4 + FPN
│   │   ├── wavelet_branch.py  # Haar DWT stem
│   │   ├── tokenizer.py       # PSIT + FrequencyStem + GatedFusion
│   │   ├── nmsss_vit.py       # Decomposed spatial/temporal ViT
│   │   └── smsdt.py           # Full assembled model
│   ├── data/
│   │   └── dataset.py         # WebDataset loader with ImageNet normalization
│   ├── preprocess/
│   │   ├── detect_track.py    # SCRFD + ByteTrack face detection
│   │   ├── align_crop.py      # 5-point face alignment (224×224)
│   │   └── build_shards.py    # Full preprocessing pipeline
│   ├── export/
│   │   ├── to_onnx.py         # ONNX export
│   │   └── build_trt_engine.py # TensorRT engine builder
│   ├── losses.py              # Focal BCE + Contrastive + Alignment losses
│   ├── train.py               # Training script
│   └── eval.py                # Evaluation with full metrics suite
├── scripts/                   # Shell scripts (for DGX Spark / Linux)
├── outputs/
│   ├── checkpoints/           # Saved model weights
│   └── logs/                  # Training logs
├── S-MSDT_DGX_Spark_Implementation_Plan.md
└── README.md                  # ← You are here
```

---

## Prerequisites

### Software
- Python 3.10+
- PyTorch 2.0+ (with CUDA support for GPU training)
- NVIDIA GPU with CUDA (DGX Spark with sm_121, or any modern GPU for development)

### Python Dependencies
```bash
pip install torch torchvision torchaudio
pip install timm==1.0.* opencv-python-headless==4.10.*
pip install einops webdataset
pip install albumentations
pip install omegaconf tqdm
pip install scikit-learn
pip install wandb          # optional, for experiment tracking
pip install insightface    # for face detection (SCRFD)
```

> **Note (DGX Spark):** Use NVIDIA's NGC PyTorch container for aarch64 + sm_121 support.
> See `S-MSDT_DGX_Spark_Implementation_Plan.md` §1 for full DGX Spark setup.

---

## ⚡ Dedicated Guide for RTX 5090 (32 GB VRAM) & 32 GB RAM PC

This setup guide is tailored for running and training S-MSDT locally on an **NVIDIA GeForce RTX 5090 (32 GB GDDR7)** paired with **32 GB of system RAM** on Windows (PowerShell) or Linux.

### Hardware & Configuration Tuning

| Component | Hardware Specification | Consideration | Tuning Applied |
|---|---|---|---|
| **GPU VRAM** | 32 GB GDDR7 (~1,792 GB/s) | High VRAM headroom | Increase micro-batch size from `4` → `8` (or `12`). Disable `gradient_checkpointing` for ~25% faster throughput without running out of memory. |
| **GPU Architecture** | Blackwell (`sm_120`) | Requires modern CUDA | Use PyTorch 2.6+ with CUDA 12.4/12.8. BF16 mixed-precision is natively accelerated on Blackwell Tensor Cores. |
| **System RAM** | 32 GB DDR4/DDR5 | Windows uses ~6–8 GB; high worker counts cause OOM page thrashing | Cap DataLoader `workers: 4` (default was 10-12 for DGX Spark). Cap preprocessing `workers: 4`. |
| **Effective Batch** | 32 chunks | `batch_size: 8` × `grad_accum: 4` = 32 | Preserves paper gradient dynamics while training ~2× faster. |

#### Step-by-Step Commands for RTX 5090 PC (PowerShell)

#### 1. Setup Virtual Environment & Blackwell CUDA
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install PyTorch with CUDA 12.4+ (or CUDA 12.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Verify RTX 5090 detection:
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0)); print('VRAM (GB):', round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2))"

# Install remaining dependencies:
pip install timm==1.0.* opencv-python-headless==4.10.* einops webdataset albumentations omegaconf tqdm scikit-learn insightface onnx onnxruntime-gpu
```

#### 2. Download Dataset
Use [`download1.py`](file:///e:/DEEPFAKE/Download_ff/download1.py) with the `--server EU2` flag:
```powershell
# Optional: Download 20 videos per class first for quick testing
python Download_ff/download1.py data/raw/ffpp_c23_mini -d all -c c23 -t videos -n 20 --server EU2

# Stage 1: Full FF++ c23 (all manipulations + originals)
python Download_ff/download1.py data/raw/ffpp_c23 -d all -c c23 -t videos --server EU2
```

#### 3. Preprocess Shards (Tuned for 32 GB System RAM)
Limit `--workers` to `4` so concurrent face detector processes stay well within the 32 GB RAM budget:
```powershell
python -m smsdt.preprocess.build_shards `
    --data-root data/raw/ffpp_c23 `
    --compression c23 `
    --out data/cache/ffpp `
    --chunk-length 8 `
    --chunk-stride 4 `
    --img-size 224 `
    --shard-size 500 `
    --workers 4
```

#### 4. Train Model (Tuned for 32 GB VRAM + 32 GB RAM)
Leverage the 32 GB VRAM on the RTX 5090 by bumping `batch_size=8`, disabling `gradient_checkpointing` for higher training speed, and setting `workers=4`:
```powershell
python -m smsdt.train `
    --override `
    data.batch_size=8 `
    train.grad_accum=4 `
    train.workers=4 `
    train.gradient_checkpointing=false
```

#### 5. Evaluate and Export
```powershell
# Evaluate on validation split
python -m smsdt.eval --ckpt outputs/checkpoints/best.pt --split val

# Evaluate on test split
python -m smsdt.eval --ckpt outputs/checkpoints/best.pt --split test

# Export to ONNX
python -m smsdt.export.to_onnx --ckpt outputs/checkpoints/best.pt --out outputs/smsdt.onnx
```

---

## Step-by-Step Guide (General / DGX Spark)

### Step 1: Download FaceForensics++ Dataset

Download FF++ with c23 (light compression) and c40 (heavy compression) using the provided scripts. Only the EU2 server is available.

```bash
# Stage 1: Download FF++ c23 (all manipulation types + originals)
python Download_ff/download1.py data/raw/ffpp_c23 -d all -c c23 -t videos --server EU2

# Stage 2: Download FF++ c40 (for compression robustness training)
python Download_ff/download1.py data/raw/ffpp_c40 -d all -c c40 -t videos --server EU2
```

**Optional — download a small subset first to test the pipeline:**
```bash
python Download_ff/download1.py data/raw/ffpp_c23_mini -d all -c c23 -t videos -n 20 --server EU2
```

The script will ask you to accept the Terms of Service (press Enter to continue).

**Output structure:**
```
data/raw/ffpp_c23/
├── original_sequences/youtube/c23/videos/*.mp4
├── manipulated_sequences/Deepfakes/c23/videos/*.mp4
├── manipulated_sequences/Face2Face/c23/videos/*.mp4
├── manipulated_sequences/FaceSwap/c23/videos/*.mp4
├── manipulated_sequences/NeuralTextures/c23/videos/*.mp4
└── manipulated_sequences/FaceShifter/c23/videos/*.mp4
```

> **Testing datasets** (Celeb-DF v2, DFDC, WildDeepfake) will be provided separately later. They are held out from training entirely.

---

### Step 2: Preprocess Videos into WebDataset Shards

This step extracts aligned face crops, chunks them into T=8 sequences, and packs them into `.tar` shards for efficient training I/O.

```bash
# Preprocess FF++ c23 (primary training data)
python -m smsdt.preprocess.build_shards \
    --data-root data/raw/ffpp_c23 \
    --compression c23 \
    --out data/cache/ffpp \
    --chunk-length 8 \
    --chunk-stride 4 \
    --img-size 224 \
    --shard-size 500 \
    --workers 1
```

**On Windows PowerShell** (use backtick `` ` `` for line continuation):
```powershell
python -m smsdt.preprocess.build_shards `
    --data-root data/raw/ffpp_c23 `
    --compression c23 `
    --out data/cache/ffpp `
    --chunk-length 8 `
    --chunk-stride 4 `
    --img-size 224 `
    --shard-size 500 `
    --workers 1
```

**Output:**
```
data/cache/ffpp/
├── train-000000.tar
├── train-000001.tar
├── ...
├── val-000000.tar
└── test-000000.tar
```

> **DGX Spark tip:** Use `--workers 12-14` to parallelize across Arm cores.
> Leave headroom for the GPU feeder threads.

---

### Step 3: Train the Model

```bash
# Train with default config files (configs/data.yaml, model.yaml, train.yaml)
python -m smsdt.train

# Train with custom overrides
python -m smsdt.train --override train.epochs=100 train.lr=1e-4 data.batch_size=8

# Train with WandB logging enabled
python -m smsdt.train --override train.use_wandb=true
```

**On Windows PowerShell:**
```powershell
python -m smsdt.train

# With overrides
python -m smsdt.train --override train.epochs=100 train.lr=1e-4
```

**Key config options** (edit `configs/train.yaml` or pass as overrides):

| Parameter | Default | Description |
|---|---|---|
| `train.epochs` | 50 | Number of training epochs |
| `train.lr` | 3e-4 | Learning rate |
| `train.grad_accum` | 8 | Gradient accumulation steps (effective batch = batch_size × grad_accum) |
| `train.alpha` | 0.8 | Focal loss alpha (tuned for FF++ 1:4 imbalance) |
| `train.lambda_contrastive` | 0.1 | Supervised contrastive loss weight |
| `train.lambda_align` | 0.05 | Alignment loss weight |
| `data.batch_size` | 4 | Micro-batch size per step |
| `train.workers` | 10 | Dataloader workers |

**Output:**
```
outputs/checkpoints/
├── epoch0.pt          # Full checkpoint (model + optimizer + scheduler)
├── epoch1.pt
├── ...
└── best.pt            # Best model weights (by validation AUC)
```

---

### Step 4: Evaluate the Model

```bash
# Evaluate best checkpoint on validation set
python -m smsdt.eval --ckpt outputs/checkpoints/best.pt --split val

# Evaluate on test set
python -m smsdt.eval --ckpt outputs/checkpoints/best.pt --split test

# Evaluate a specific epoch
python -m smsdt.eval --ckpt outputs/checkpoints/epoch10.pt --split val
```

**Output** (printed to stdout):
```
============================================================
  VAL Set Results
============================================================
  AUC:              0.9847
  EER:              0.0412 (threshold=0.6234)
  Accuracy @0.5:    0.9523
  Precision @0.5:   0.9612
  Recall @0.5:      0.9478
  F1 @0.5:          0.9544
  FPR @0.5:         0.0234

  Per-Manipulation Breakdown:
  Type                      AUC    F1@0.5    Count
  ------------------------------------------------
  Deepfakes               0.9921   0.9712      200
  Face2Face               0.9834   0.9456      200
  FaceSwap                0.9756   0.9389      200
  NeuralTextures          0.9678   0.9234      200
============================================================
```

---

### Step 5: Export to ONNX (for deployment)

```bash
# Export best model to ONNX
python -m smsdt.export.to_onnx --ckpt outputs/checkpoints/best.pt --out outputs/smsdt.onnx
```

---

### Step 6: Build TensorRT Engine (DGX Spark deployment)

```bash
# Build TensorRT engine (requires trtexec in PATH)
python -m smsdt.export.build_trt_engine --onnx outputs/smsdt.onnx --out outputs/smsdt.engine
```

Or directly with `trtexec`:
```bash
trtexec --onnx=outputs/smsdt.onnx \
    --saveEngine=outputs/smsdt_fp8.engine \
    --fp8 --int8 \
    --shapes=frames:1x8x3x224x224,conf:1x8 \
    --useCudaGraph --noDataTransfers
```

---

## Training Curriculum (Staged)

Following the implementation plan (§5.6):

| Stage | Dataset | Command |
|---|---|---|
| **1** — Warm-up | FF++ c23 only | `python -m smsdt.train` (default config) |
| **2** — Compression robustness | FF++ c23 + c40 | Preprocess c40, merge shards, retrain |
| **3** — Scale + diversity | FF++ + DFDC | Add DFDC data (to be provided later) |
| **4** — Top-K fine-tune | Same as Stage 3 | Switch to video-level Top-K loss |

---

## Architecture Overview

```
Input: (B, T=8, 3, 224, 224) face crop chunks + (B, T) confidence scores
  │
  ├─→ RGB Branch: EfficientNetB4 + FPN → (B*T, 256, h, w) multi-scale features
  │     └─→ PSIT: Adaptive pool → (B*T, 49, 256) tokens
  │
  ├─→ Wavelet Branch: Haar DWT → LH,HL,HH → Conv → (B*T, 256, H/2, W/2)
  │     └─→ FrequencyStem: Adaptive pool → (B*T, 49, 256) tokens
  │
  ├─→ Gated Fusion: σ(gate) × RGB + (1-σ(gate)) × Wavelet → (B*T, 49, 256)
  │
  ├─→ N-MSSS ViT (×4 blocks):
  │     ├─ Spatial Self-Attention (intra-frame)
  │     ├─ L2-normalized Temporal Differencing (confidence-gated)
  │     └─ Temporal Self-Attention (inter-frame)
  │
  └─→ Mean Pool → Classification Head → (B,) logit
```

---

## Loss Functions

| Loss | Weight | Purpose |
|---|---|---|
| **Focal BCE** | 1.0 | Primary classification loss with α=0.8 for FF++ 1:4 imbalance |
| **Supervised Contrastive** | 0.1 | Forces wavelet branch to learn manipulation-specific frequency signatures |
| **Cross-Domain Alignment** | 0.05 | Ensures RGB and wavelet branches are complementary |

---

## Metrics Reported

| Metric | Description |
|---|---|
| **AUC (AUROC)** | Primary metric — threshold-agnostic discrimination |
| **EER** | Equal Error Rate — optimal operating point |
| **Precision / Recall / F1** | At both EER threshold and 0.5 threshold |
| **FPR** | False Positive Rate — false alarm rate |
| **Per-manipulation breakdown** | AUC and F1 per forgery type |

---

## References

- [FaceForensics++](https://github.com/ondyari/FaceForensics)
- [S-MSDT Architecture](S-MSDT_DGX_Spark_Implementation_Plan.md)
- [NVIDIA DGX Spark](https://www.nvidia.com/en-us/products/workstations/dgx-spark/)