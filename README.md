# S-MSDT: Scale-Invariant Multi-Scale Deepfake Transformer

**Architecture:** Scale-Invariance + Multi-Scale Feature Differences + Wavelet Frequency Guard  
**Target Hardware:** NVIDIA GeForce RTX 5090 (32 GB GDDR7, Blackwell `sm_120`)  
**Operating System:** Windows 10 / 11 (PowerShell)  
**Goal:** High-throughput deepfake detection with real-time inference (<8 ms/chunk) and accelerated BF16 mixed-precision training.

---

## Project Structure

```
DEEPFAKE/
├── configs/
│   ├── data.yaml              # Dataset paths, batch size (8), chunk config (T=8)
│   ├── model.yaml             # Model architecture hyperparameters (embed_dim=256, depth=4)
│   └── train.yaml             # Training config tuned for RTX 5090 (grad_accum=4, grad_ckpt=false)
├── data/
│   ├── raw/                   # Downloaded raw videos
│   │   ├── ffpp_c23/          # FF++ c23 compression (light compression)
│   │   └── ffpp_c40/          # FF++ c40 compression (heavy compression)
│   ├── cache/                 # Preprocessed WebDataset shards (.tar)
│   └── splits/                # Train, validation, and test manifests
├── Download_ff/
│   ├── download1.py           # FaceForensics++ downloader (EU2 server)
│   └── download2.py           # Supplementary FaceForensics downloader
├── smsdt/
│   ├── models/
│   │   ├── rgb_branch.py      # EfficientNet-B4 + Feature Pyramid Network (FPN)
│   │   ├── wavelet_branch.py  # Haar Discrete Wavelet Transform (DWT 2D) stem
│   │   ├── tokenizer.py       # PSIT + FrequencyStem + Gated Cross-Domain Fusion
│   │   ├── nmsss_vit.py       # Decomposed Spatial/Temporal ViT + L2 Temporal Differencing
│   │   └── smsdt.py           # Full assembled S-MSDT model
│   ├── data/
│   │   └── dataset.py         # WebDataset loader with video augmentations & manipulation mapping
│   ├── preprocess/
│   │   ├── detect_track.py    # SCRFD face detection + ByteTrack tracking
│   │   ├── align_crop.py      # 5-point ArcFace landmark alignment (224×224)
│   │   └── build_shards.py    # Multi-threaded video-to-shard preprocessing pipeline
│   ├── losses.py              # Focal BCE + Supervised Contrastive + Cross-Domain Alignment losses
│   ├── train.py               # Crash-resilient training script with periodic checkpointing
│   └── eval.py                # Evaluation script with chunk-level, video-level & manipulation metrics
└── README.md                  # This documentation
```

---

## ⚡ NVIDIA GeForce RTX 5090 (32 GB VRAM) Hardware Tuning

This repository is optimized specifically for the **NVIDIA GeForce RTX 5090 (32 GB GDDR7)** running on **Windows** with 32 GB of system RAM.

### Hardware & Configuration Profile

| Hardware Parameter | Specification / Behavior | Tuning Applied in Configs | Rationale |
|---|---|---|---|
| **GPU VRAM** | 32 GB GDDR7 (~1,792 GB/s) | `batch_size: 8` | High VRAM capacity allows doubling micro-batch size from 4 to 8 per forward step. |
| **GPU Architecture** | Blackwell (`sm_120`) | Native `bfloat16` autocast | 5th-gen Tensor Cores accelerate BF16 mixed-precision natively with zero loss of numerical stability. |
| **Gradient Checkpointing** | Memory vs Compute tradeoff | `gradient_checkpointing: false` | 32 GB VRAM easily accommodates all activations in memory. Disabling activation recomputation makes training **~25% to 30% faster**. |
| **Effective Batch** | Gradient dynamics preservation | `batch_size: 8` × `grad_accum: 4` = 32 | Matches theoretical paper convergence with fewer gradient synchronization steps. |
| **System RAM (Windows)** | 32 GB DDR4 / DDR5 | `workers: 4` in Dataloader & Preprocessing | Windows uses the `spawn` process model. Capping worker processes at 4 prevents system memory paging and RAM exhaustion. |
| **Crash-Safe Checkpointing** | Power cut / unexpected exit protection | `save_interval_steps: 500` | Periodically writes `outputs/checkpoints/latest.pt` every 500 steps, enabling instant resume from intra-epoch stops. |

---

## Step-by-Step Guide for Windows (PowerShell)

### Step 1: Environment Setup & GPU Verification

Open **PowerShell** and navigate to the project directory:

```powershell
# 1. Enable script execution if restricted
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# 2. Create and activate a Python virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# 3. Upgrade pip and build tools
python -m pip install --upgrade pip setuptools wheel

# 4. Install PyTorch with Blackwell (sm_120) compatible CUDA 12.8
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# 5. Verify RTX 5090 detection and sm_120 compute capability:
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('Device Name:', torch.cuda.get_device_name(0)); print('Compute Capability:', torch.cuda.get_device_capability(0)); print('VRAM (GB):', round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2))"

# 6. Install all dependencies (pin onnxruntime-gpu==1.20.2 for CUDA 12.x compatibility)
pip install timm==1.0.* opencv-python-headless==4.10.* einops webdataset albumentations omegaconf tqdm scikit-learn insightface onnx "onnxruntime-gpu==1.20.2"

# 7. Verify ONNX Runtime GPU provider loads cleanly:
python -c "import torch, os; p = os.path.join(os.path.dirname(torch.__file__), 'lib'); os.add_dll_directory(p); import onnxruntime as ort; print('Available Providers:', ort.get_available_providers())"
```

> [!TIP]
> **Why `onnxruntime-gpu==1.20.2`?** Unpinned `pip install onnxruntime-gpu` pulls newer releases compiled for CUDA 13 (requiring `cublasLt64_13.dll`), whereas PyTorch cu128 provides CUDA 12.8 runtime libraries (`cublasLt64_12.dll`). Version `1.20.2` targets CUDA 12.x directly and seamlessly executes SCRFD face detection on the RTX 5090.

---

### Step 2: Download FaceForensics++ Dataset

Download the FaceForensics++ dataset using `Download_ff/download1.py`. The script uses the active `--server EU2` endpoint.

```powershell
# Option A: Quick Pipeline Test (downloads 20 videos per class first)
python Download_ff/download1.py data/raw/ffpp_c23_mini -d all -c c23 -t videos -n 20 --server EU2

# Option B: Full Training Dataset (FF++ c23: all manipulation methods + originals)
python Download_ff/download1.py data/raw/ffpp_c23 -d all -c c23 -t videos --server EU2

# Optional Stage 2: Heavy Compression Dataset (FF++ c40: for compression robustness)
python Download_ff/download1.py data/raw/ffpp_c40 -d all -c c40 -t videos --server EU2
```

> **Note:** The script prompts you to accept the FaceForensics Terms of Service on first run (press Enter to proceed).

**Expected Raw Folder Structure:**
```
data/raw/ffpp_c23/
├── original_sequences/
│   ├── youtube/c23/videos/*.mp4
│   └── actors/c23/videos/*.mp4
└── manipulated_sequences/
    ├── Deepfakes/c23/videos/*.mp4
    ├── Face2Face/c23/videos/*.mp4
    ├── FaceSwap/c23/videos/*.mp4
    ├── NeuralTextures/c23/videos/*.mp4
    ├── FaceShifter/c23/videos/*.mp4
    └── DeepFakeDetection/c23/videos/*.mp4
```

---

### Step 3: Preprocess Videos into WebDataset Shards

The preprocessing pipeline in `smsdt.preprocess.build_shards` features:
1. **GPU-Accelerated Face Detection:** Uses SCRFD (`det_10g.onnx` from `buffalo_l`) via `CUDAExecutionProvider` on the RTX 5090.
2. **Worker Model Caching:** The face detector ONNX session is created once per worker process and reused across all videos, eliminating session reload overhead.
3. **Alignment & Normalization:** Standard ArcFace 5-point affine transformation aligns and crops each face to 224×224 pixels with temporal landmark smoothing.
4. **Streaming Shard Writing (Memory-Safe):** Chunks are streamed to disk in 500-chunk `.tar` shards as videos finish. RAM usage is strictly bounded (<1 GB), preventing memory exhaustion on massive video collections.

Run in PowerShell as a single command:

```powershell
python -m smsdt.preprocess.build_shards --data-root "data/raw/ffpp_c23" --compression c23 --out "data/cache/ffpp" --chunk-length 8 --chunk-stride 4 --img-size 224 --shard-size 500 --workers 4
```

**Output Shard Files:**
```
data/cache/ffpp/
├── train-000000.tar
├── train-000001.tar
├── ...
├── val-000000.tar
└── test-000000.tar
```

---

### Step 4: Train the Model (Crash-Resilient)

`configs/data.yaml` and `configs/train.yaml` are pre-tuned for the RTX 5090. Simply launch training:

```powershell
# Resume training from latest checkpoint (default behavior)
python -m smsdt.train

# Start a fresh training run from scratch (ignores existing checkpoints)
python -m smsdt.train --from-scratch
```

#### Training with Custom Overrides
You can pass custom overrides directly from PowerShell without editing YAML files:

```powershell
# Example: Fast fine-tune / quick run with 30 epochs and custom learning rate
python -m smsdt.train --override train.epochs=30 train.lr=1e-4

# Start clean from scratch with overrides
python -m smsdt.train --from-scratch --override train.epochs=30
```


#### Understanding Checkpointing & Unexpected Crash Recovery

The training loop implements a dual-tier checkpointing mechanism:

1. **Intra-Epoch Periodic Checkpoints (`save_interval_steps: 500`):**
   Every 500 steps, model weights, optimizer state, scheduler state, current epoch, and global step are saved to `outputs/checkpoints/latest.pt`.
2. **End-of-Epoch Checkpoints:**
   At the conclusion of each epoch, weights are saved to `outputs/checkpoints/epoch{N}.pt` and `outputs/checkpoints/latest.pt`.
3. **Best Checkpoint (`best.pt`):**
   Whenever validation AUC reaches a new high, weights are saved to `outputs/checkpoints/best.pt`.

**How to resume if training stops unexpectedly:**
If training is interrupted (power outage, manual cancellation, or system reboot), simply rerun the training command:

```powershell
python -m smsdt.train
```

The script automatically detects `outputs/checkpoints/latest.pt`, outputs:
```
Loading checkpoint from outputs/checkpoints/latest.pt
Resuming training from epoch 12, step 6500 (Best AUC: 0.9842)
```
and resumes training from that exact step.

#### Why Gradient Checkpointing is Disabled (`false`)
> [!NOTE]
> **Gradient Checkpointing** is an in-memory GPU VRAM optimization, not weight saving to disk.
> - When `gradient_checkpointing: true`: Intermediate layer activations are discarded during the forward pass and recomputed during the backward pass to conserve VRAM on small GPUs (e.g., 8 GB / 12 GB cards).
> - When `gradient_checkpointing: false`: Activations are kept in memory. Because the RTX 5090 has **32 GB of VRAM**, activations fit with plenty of headroom. Disabling it makes training **~25% to 30% faster** without affecting your ability to save or resume weight checkpoints.

---

### Step 5: Comprehensive Model Evaluation

Evaluate trained checkpoints on either validation or test splits:

```powershell
# Evaluate the best model on the validation split
python -m smsdt.eval --ckpt outputs/checkpoints/best.pt --split val

# Evaluate the best model on the held-out test split
python -m smsdt.eval --ckpt outputs/checkpoints/best.pt --split test

# Evaluate a specific epoch checkpoint
python -m smsdt.eval --ckpt outputs/checkpoints/epoch10.pt --split val
```

#### Sample Evaluation Report
```
============================================================
  VAL Set Results
============================================================
  [Chunk-Level Metrics]
  AUC:              0.9847
  EER:              0.0412 (threshold=0.6234)
  Accuracy @0.5:    0.9523
  Precision @0.5:   0.9612
  Recall @0.5:      0.9478
  F1 @0.5:          0.9544
  FPR @0.5:         0.0234
  Precision @EER:   0.9580
  Recall @EER:      0.9580
  F1 @EER:          0.9580

  [Video-Level Metrics (Top-K Aggregation)]
  Video AUC:        0.9892
  Video EER:        0.0380
  Video Acc @0.5:   0.9643
  Video F1 @0.5:    0.9655

  Per-Manipulation Breakdown:
  Type                      AUC      EER   F1@0.5    Count
  --------------------------------------------------------
  Deepfakes              0.9921   0.0310   0.9712      140
  Face2Face              0.9834   0.0425   0.9456      140
  FaceSwap               0.9756   0.0480   0.9389      140
  NeuralTextures         0.9678   0.0550   0.9234      140
  FaceShifter            0.9810   0.0410   0.9510      140
  DeepFakeDetection      0.9850   0.0390   0.9580      140
============================================================
```

---

## S-MSDT Architecture Deep Dive

S-MSDT is built to eliminate compression artifacts and synthetic forgery fingerprints simultaneously across spatial and frequency domains:

```
Input: (B, T=8, 3, 224, 224) aligned face chunk + (B, T) SCRFD confidence scores
  │
  ├──► [RGB Branch]
  │      EfficientNet-B4 Backbone + Top-Down FPN (P2, P3, P4 levels)
  │      └─► PSIT (Pyramidal Scale-Invariant Tokenizer)
  │            Adaptive pooling across scales → (B*T, 49, 256) spatial tokens
  │
  ├──► [Wavelet Frequency Guard]
  │      Single-Level 2D Haar DWT (LH, HL, HH energy subbands, discarding low-pass LL)
  │      └─► Strided Patch Stem (Conv2D 16×16 → BatchNorm2D → GELU) → (B*T, 49, 256) frequency tokens
  │
  ├──► [Gated Cross-Domain Fusion]
  │      σ(Gate([z_rgb, z_dwt])) · z_rgb + (1 - σ(Gate)) · z_dwt → (B*T, 49, 256)
  │      Add learned 2D spatial positional embeddings
  │
  ├──► [N-MSSS ViT Backbone] (×4 Stages)
  │      ├─ 1. Spatial Self-Attention (intra-frame multi-head self-attention)
  │      ├─ 2. L2-Normalized Temporal Differencing (Zero-padded at t=0):
  │      │     Δ_t = Norm(x_t) - Norm(x_{t-1}) with Δ_0 = Norm(x_0), projected & SCRFD gated
  │      └─ 3. Temporal Self-Attention (inter-frame attention across time T)
  │
  └──► [Pooling & Head]
         Spatial Mean-Pool → Temporal Mean-Pool → LayerNorm → Linear → GELU → Dropout(0.1) → Linear → Logit
```

---

## Loss Functions & Mathematical Formulation

Training utilizes a multi-objective loss function designed specifically for FaceForensics++ class imbalance:

$$L_{\text{total}} = L_{\text{focal}} + \lambda_{\text{contrastive}} L_{\text{contrastive}} + \lambda_{\text{align}} L_{\text{align}}$$

| Loss Component | Weight | Formulation & Purpose |
|---|---|---|
| **Focal BCE ($L_{\text{focal}}$)** | $1.0$ | $\alpha = 0.2$, $\gamma = 2.0$. Down-weights easy background samples while properly balancing FF++ 1:4 real-to-fake class ratio ($4 \times 0.2 = 0.8$ vs $1 \times 0.8 = 0.8$). |
| **Supervised Contrastive ($L_{\text{contrastive}}$)** | $0.1$ | Applied to **post-ViT learned representations** ($z_{\text{post\_vit}}$). Directly guides the 4 transformer blocks to cluster genuine faces together and separate distinct manipulation methods in latent space. |
| **Cross-Domain Alignment ($L_{\text{align}}$)** | $0.05$ | Decorrelation / orthogonality penalty on pre-fusion tokens ($z_{\text{rgb}}$, $z_{\text{dwt}}$). Penalizes redundancy so the frequency branch specializes in high-frequency spectral artifacts rather than mimicking RGB features. |

---

## Training Curriculum (Staged Progression)

| Stage | Dataset | Command | Objective |
|---|---|---|---|
| **Stage 1: Warm-up** | FF++ c23 | `python -m smsdt.train` | Establish core dual-stream representations on clean compression. |
| **Stage 2: Robustness** | FF++ c23 + c40 | Merge c40 shards into cache, retrain | Train invariance against aggressive recompression and bit-rate degradation. |
| **Stage 3: Video-level Fine-tune** | FF++ (Top-K) | Train with `--override train.alpha=0.2 train.lr=5e-5` | Fine-tune chunk logits using Top-K aggregation ($K=3$) for optimal video-level classification. |

---

## References

- FaceForensics++: [Learning to Detect Manipulated Facial Images](https://github.com/ondyari/FaceForensics)
- SCRFD Face Detection: [InsightFace](https://github.com/deepinsight/insightface)
- Timm Models: [PyTorch Image Models](https://github.com/huggingface/pytorch-image-models)