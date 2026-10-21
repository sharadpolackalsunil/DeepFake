# S-MSDT: Scale-Invariant Multi-Scale Deepfake Transformer
## Complete Architecture Specification, Preprocessing Pipeline, and SOTA Roadmap Request

> **File Purpose:**  
> This document provides an exhaustive, mathematically precise, and verified architectural reference of the **S-MSDT** codebase directly from the `main` branch. It contains complete system details—hardware specifications, preprocessing pipeline, dual-stream feature extraction, spatio-temporal ViT modeling, multi-term loss formulations, empirical benchmark results, and identified technical doubts—designed to give any AI researcher or frontier LLM (e.g., Claude 3.7 / Claude 3.5 Sonnet) the exact context needed to diagnose our severe identity-overfitting bottleneck and formulate a concrete, phased roadmap to State-of-the-Art (SOTA) detection accuracy.

---

## 1. System & Hardware Profile

- **GPU:** NVIDIA GeForce RTX 5090 (31.5 GB / 32 GB GDDR7 dedicated VRAM, Blackwell architecture `sm_120`, native `bfloat16` Tensor Cores, ~1,792 GB/s memory bandwidth).
- **CPU:** Intel Core Ultra 9 285K (24 physical cores / 24 logical processors, 3.70 GHz base clock, 40 MB L2 cache, 2.4 MB L1 cache).
- **System Memory:** 64 GB DDR5 high-speed system RAM.
- **Operating Environment:** Windows 11 (PowerShell), CUDA 12.8+, PyTorch with native `bfloat16` mixed precision (`torch.autocast("cuda", dtype=torch.bfloat16)`).
- **Current Training & Evaluation Profile:**
  - **Training micro-batch size:** $B = 8$ video chunks.
  - **Validation batch size:** $B_{val} = 32$ video chunks (decoupled from training to fully leverage the RTX 5090 under `torch.no_grad()`).
  - **Temporal chunk length:** $T = 8$ frames per chunk.
  - **Spatial resolution:** $224 \times 224 \times 3$ aligned face crops.
  - **Gradient Accumulation steps:** $4$ (Effective training batch size $= 32$ chunks $= 256$ frames per optimizer step).
  - **DataLoader parallelism:** `workers: 8` (training) and `val_workers: 8` (validation) — scaled across the 24 cores of the Intel Core Ultra 9 285K to prevent GPU data starvation and CPU I/O wait.
  - **Optimized Non-Blocking Evaluation:** Evaluation predictions are aggregated on GPU in `bfloat16` and transferred to host memory in a single bulk transfer, completely eliminating per-batch CUDA pipeline stalls (`cudaDeviceSynchronize`).
  - **Gradient Checkpointing:** Disabled (`false`) because 32 GB VRAM comfortably accommodates all activations in memory without recomputation.
  - **Current VRAM Utilization:** ~6–8 GB during training ($B=8$), ~10–12 GB during validation ($B=32$).

---

## 2. Repository Layout & Verified File Paths

```
E:\DeepFake\
├── configs/
│   ├── data.yaml                 # Dataset paths, batch size, chunk length, image size
│   ├── model.yaml                # Model dimensions, heads, layers, grid size
│   └── train.yaml                # Schedulers, learning rates, loss weights, workers
├── data/
│   ├── cache/ffpp/               # Preprocessed WebDataset .tar shards (train-*.tar, val-*.tar, test-*.tar)
│   ├── raw/
│   │   └── ffpp_c23/             # FF++ light compression (used for training & evaluation)
│   └── splits/                   # train.csv, val.csv, test.csv (identity-disjoint actor splits)
├── Download_ff/
│   ├── download1.py              # Download utilities
│   └── download2.py
├── outputs/
│   ├── checkpoints/              # Active training run checkpoints (latest.pt, best.pt)
│   └── checkpoints_v1_epoch23/   # ARCHIVED best v1 model weights (best.pt evaluated in Section 8)
├── scripts/
│   ├── run_preprocess.sh         # Offline shard creation runner
│   └── run_train.sh              # Python -m smsdt.train runner
├── smsdt/
│   ├── data/
│   │   └── dataset.py            # WebDataset loader, SampleDecoder, augmentations
│   ├── models/
│   │   ├── nmsss_vit.py          # Decomposed Spatial-Temporal ViT with L2 differencing
│   │   ├── rgb_branch.py         # EfficientNet-B4 + FPN (P2, P3, P4)
│   │   ├── smsdt.py              # SMSDT top-level model & readout head
│   │   ├── tokenizer.py          # PSIT & Gated Cross-Domain Fusion
│   │   └── wavelet_branch.py     # 2D Haar DWT + strided patch stem
│   ├── preprocess/
│   │   ├── align_crop.py         # ArcFace 5-point affine alignment
│   │   ├── build_shards.py       # Preprocessing & WebDataset chunk serialization
│   │   └── detect_track.py       # SCRFD face detector
│   ├── __init__.py
│   ├── eval.py                   # Chunk & video-level metric evaluation suite
│   ├── losses.py                 # Focal BCE, SupCon, and Decorrelation losses
│   └── train.py                  # Distributed/mixed-precision training pipeline
└── tests/
    ├── test_blockers.py          # Unit tests
    └── test_improvements.py
```

> **Scope Note for Reviewer:**  
> All deployment and export modules (`smsdt/deploy/`, `smsdt/export/`, `triton_repo/`, `run_export.sh`) are intentionally excluded from this review. Our singular, non-negotiable objective is **maximizing model generalization and achieving State-of-the-Art (SOTA) detection accuracy**. Do not allocate attention or recommendations to export formats (ONNX/TensorRT) or deployment latency.

---

## 3. Dataset Specification: FaceForensics++ (FF++ c23)

### 3.1 Dataset Setup
- **Dataset:** FaceForensics++ (FF++) under $c23$ (light H.264 compression, cached in `data/cache/ffpp/`).
- **Directory Structure:**
  - `original_sequences/`: Pristine real videos (`youtube` and `actors`) $\rightarrow$ **Class 0 (Real)**
  - `manipulated_sequences/`: All 6 manipulation types $\rightarrow$ **Class 1 (Fake)**
    1. `Deepfakes` (autoencoder-based face replacement)
    2. `Face2Face` (facial reenactment / expression transfer)
    3. `FaceSwap` (graphics-based 3D landmark face swapping)
    4. `NeuralTextures` (neural rendering / GAN mouth & expression modification)
    5. `FaceShifter` (high-fidelity AEI-Net face swapping)
    6. `DeepFakeDetection` (actor-consented deepfakes benchmark)
- **Class Imbalance:** Up to 1:6 Real-to-Fake ratio across all manipulation methods.
- **Split Strategy:** Disjoint identity-aware split (720 train / 140 val / 140 test videos). Source and target actors are strictly quarantined so that no actor appearing in training ever appears in validation or testing.

---

## 4. End-to-End Preprocessing Pipeline (`smsdt/preprocess/`)

```
Raw Video (.mp4)
      │
      ▼
[detect_track.py]  ──► SCRFD Detector extracts faces & 5-point facial landmarks + confidence
      │
      ▼
[build_shards.py]   ──► Temporal Landmark EMA Smoothing (alpha=0.6)
      │
      ▼
[align_crop.py]    ──► ArcFace Affine Alignment to canonical 5-point template (224x224)
      │
      ▼
[build_shards.py]   ──► Temporal Chunking: T=8 frames, stride=4
      │
      ▼
WebDataset Shards  ──► Stored as train-*.tar / val-*.tar / test-*.tar containing:
                         ├── frames.npy: (8, 224, 224, 3) uint8
                         ├── conf.npy:   (8,) float32 detection scores
                         ├── label.cls:  0.0 or 1.0
                         └── manip.txt:  "real", "Deepfakes", etc.
```

### Component Details
1. **Face Detection & Tracking (`detect_track.py`):**
   - Utilizes SCRFD (`buffalo_l`) to localize faces and 5 key facial landmarks: left eye, right eye, nose tip, left mouth corner, right mouth corner.
   - Extracts detector confidence score $c_t \in [0.0, 1.0]$ for each frame.
2. **Landmark Temporal Smoothing (`build_shards.py`):**
   - Eliminates high-frequency frame-to-frame landmark jitter via Exponential Moving Average (EMA):
     $$\hat{kps}_t = \alpha \cdot \hat{kps}_{t-1} + (1 - \alpha) \cdot kps_t \quad (\alpha = 0.6)$$
3. **ArcFace Canonical Alignment (`align_crop.py`):**
   - Computes partial affine transformation matrix $M$ via `cv2.estimateAffinePartial2D` against the canonical 5-point ArcFace reference:
     $$\text{REF\_5PT} = \begin{bmatrix} 38.29 & 51.70 \\ 73.53 & 51.50 \\ 56.03 & 71.74 \\ 41.55 & 92.37 \\ 70.73 & 92.20 \end{bmatrix} \times \frac{224}{112}$$
   - Warps the face into an aligned $224 \times 224$ crop with eyes horizontally leveled.
4. **Chunking & WebDataset Serialization:**
   - Slices continuous video tracks into fixed chunks of length $T=8$ frames with stride $4$.
   - Writes binary shards (`.tar`) containing `frames.npy`, `conf.npy`, `label.cls`, and `manip.txt`.

---

## 5. Dataset Loader & Augmentations (`smsdt/data/dataset.py`)

- **I/O Engine:** `webdataset.WebDataset` with Windows drive letter resolution (`gopen_schemes`).
- **SampleDecoder & Color Handling:**
  - Decodes `frames.npy` and landmark confidences `conf.npy`.
  - Normalizes pixel values to $[0, 1]$ float32.
  - Normalizes with ImageNet parameters: $\mu = [0.485, 0.456, 0.406], \sigma = [0.229, 0.224, 0.225]$.
- **Training Augmentations (`augment_video`):**
  1. *Temporal Jitter:* Random start offset and temporal stride $\in \{1, 2\}$.
  2. *Simulated Compression:* Random JPEG re-encoding ($Q \in [15, 80]$) with 40% probability.
  3. *Simulated Blur:* Gaussian blur ($k \in \{3, 5\}$) with 30% probability.
  4. *Spatial Scaling & Re-crop:* Rescales $224 \rightarrow [224, 280]$ and extracts random $224 \times 224$ crop.
  5. *Horizontal Flip:* Random 50% horizontal reflection.
- **Validation Processing:**
  - Deterministic center frame chunk selection, center crop, no simulated JPEG or blur.
- **The "Epoch" Mechanics & What `resampled=True` Actually Means:**
  - Training currently uses:
    ```python
    dataset = wds.WebDataset(shard_spec, resampled=True, shardshuffle=True)
    ```
  - **What is `resampled`?**
    - **Sampling Without Replacement (`resampled=False` — Standard ML):** Imagine a deck of 100 cards (your training shards). You draw card #1, train on it, and set it aside on the table. You draw card #2, train, and set it aside. Once all 100 cards have been played, the deck is empty. That constitutes **1 true Epoch**: every single video and actor is seen **exactly once** without omission or repetition.
    - **Sampling With Replacement (`resampled=True` — Current Pipeline):** You draw a card, train on it, and **immediately put it back into the deck**. Because cards are returned to the deck, **the deck NEVER empties (it becomes an infinite stream)**. The DataLoader will run in an infinite loop forever unless artificially stopped.
  - **Why `train.py` was artificially capped at 1,000 steps:**
    - Because the stream never terminates on its own, `train.py` had to insert an artificial break:
      `if epoch_opt_steps >= steps_per_epoch: break  # steps_per_epoch: 1000`
    - At `grad_accum: 4` and micro-batch size $B=8$, each epoch executes $1,000 \times 4 \times 8 = 32,000$ video chunks.
  - **The Severe Consequences for Deepfake Detection:**
    1. **Severe Actor Sampling Variance:** Drawing shards *with replacement* means the model might randomly draw Actor #042 three or four times in a single epoch, while Actor #115 is **never drawn at all**. This heavily incentivizes the model to memorize the specific facial features of the few actors it encounters repeatedly.
    2. **Severe Under-Sampling:** FaceForensics++ $c23$ contains over 4,000 training videos (~80,000 to 120,000 chunks). An epoch of only 32,000 chunks sampled with replacement means the model only sees **~25% to 35% of the total dataset** in any given epoch!
    3. **Learning Rate Truncation:** `OneCycleLR` was set to `total_steps: 50000`, but after 23 epochs of 1,000 steps, training stopped at step 23,000—before the learning rate could complete its decay schedule.

---

## 6. Main Model Architecture (`smsdt/models/` on `main`)

Total Model Parameters: **~28.5M**.

```
Input: Video Chunk (B, T=8, 3, 224, 224) + Detector Confidences (B, T=8)
  │
  ├──► [Spatial RGB Branch] (rgb_branch.py)
  │      EfficientNet-B4 Backbone (Stages 2, 3, 4: 56, 160, 448 ch)
  │      └─► Top-Down Feature Pyramid Network (FPN) ──► P2, P3, P4 (each 256 ch)
  │            └─► PSIT (tokenizer.py): AdaptiveAvgPool(7x7) + Sum + 1x1 Conv
  │                  └─► z_rgb: (B*T, 49, 256) spatial tokens
  │
  ├──► [Frequency Wavelet Branch] (wavelet_branch.py)
  │      2D Haar DWT: Decomposes 224x224 RGB -> LL, LH, HL, HH (112x112)
  │      Discard LL -> Keep 9 High-Frequency Channels [LH, HL, HH]
  │      pool_energy: Take absolute values |LH|, |HL|, |HH|
  │      └─► Strided Patch Stem: Conv2d(9, 256, k=16, s=16) -> (B*T, 256, 7, 7)
  │            └─► FrequencyStem (tokenizer.py): Flatten -> (B*T, 49, 256) tokens
  │
  ├──► [Gated Cross-Domain Fusion] (tokenizer.py)
  │      Concat [z_rgb || z_dwt] -> Linear(512, 256) -> Sigmoid -> Gate g in [0, 1]
  │      Fused = g * z_rgb + (1 - g) * z_dwt -> Linear(256, 256) -> (B*T, 49, 256)
  │      Reshape to (B, T=8, N=49, 256) + Learnable Positional Embedding (1, 1, 49, 256)
  │
  ├──► [N-MSSS Decomposed ViT Backbone] (nmsss_vit.py) - 4 Blocks
  │      For each block:
  │        1. Spatial Self-Attention: Intra-frame MHSA over 49 tokens (heads=8)
  │        2. L2 Temporal Differencing:
  │             x_norm = x / ||x||_2
  │             delta = x_norm[t] - x_norm[t-1]  (zero-pad at t=0)
  │             cat = [x_norm || delta] -> Conv1d(512, 256, 1) across time
  │             Residual update gated by detector confidence: x = x + gated * conf_w
  │        3. Temporal Self-Attention: Inter-frame MHSA over T=8 frames (heads=8)
  │        4. Feed-Forward Network: LayerNorm -> Linear(256, 1024) -> GELU -> Linear(1024, 256)
  │
  └──► [Readout & Classification Head] (smsdt.py)
         Spatial Mean Pooling: (B, T, 49, 256) -> (B, T, 256)
         Temporal Mean Pooling: (B, T, 256) -> (B, 256)
         Classifier Head: LayerNorm(256) -> Linear(256, 128) -> GELU -> Dropout(0.1) -> Linear(128, 1)
         Output: (B,) Chunk-Level Binary Logit
```

### Detailed Mathematical Formulations

#### 1. Spatial RGB Branch (`rgb_branch.py`)
- **Backbone:** `efficientnet_b4` pretrained on ImageNet-1k.
- **Extracted Strides:** `out_indices=(2, 3, 4)`:
  - $C_2$: 56 channels, resolution $28 \times 28$
  - $C_3$: 160 channels, resolution $14 \times 14$
  - $C_4$: 448 channels, resolution $7 \times 7$
- **Top-Down FPN Pathway:**
  $$P_4 = \text{Conv}_{1\times 1}(C_4)$$
  $$P_3 = \text{Smooth}_3\left(\text{Conv}_{1\times 1}(C_3) + \text{NearestUp}_{2\times}(P_4)\right)$$
  $$P_2 = \text{Smooth}_2\left(\text{Conv}_{1\times 1}(C_2) + \text{NearestUp}_{2\times}(P_3)\right)$$
  All $P_l$ have channel dimension $D = 256$.

#### 2. Frequency Haar Wavelet Branch (`wavelet_branch.py`)
- Fixed Haar quadrature mirror filters:
  $$LL = \frac{1}{2}\begin{bmatrix}1 & 1 \\ 1 & 1\end{bmatrix}, \quad LH = \frac{1}{2}\begin{bmatrix}1 & 1 \\ -1 & -1\end{bmatrix}, \quad HL = \frac{1}{2}\begin{bmatrix}1 & -1 \\ 1 & -1\end{bmatrix}, \quad HH = \frac{1}{2}\begin{bmatrix}1 & -1 \\ -1 & 1\end{bmatrix}$$
- Decomposes $(B\cdot T, 3, 224, 224) \rightarrow 12$ subband channels at $112 \times 112$.
- Discards $LL$ (spatial approximation) to prevent spatial leakage into the frequency path.
- Retains 9 high-frequency channels $[LH_r, LH_g, LH_b, HL_r, HL_g, HL_b, HH_r, HH_g, HH_b]$.
- Computes absolute magnitude / spectral energy: $|LH|, |HL|, |HH|$ to avoid phase cancellation.
- Projects to latent dimension via Strided Patch Stem:
  $$\text{Conv2d}(9, 256, \text{kernel}=16, \text{stride}=16) + \text{BatchNorm} + \text{GELU} \longrightarrow (B\cdot T, 256, 7, 7)$$

#### 3. Pyramidal Scale-Invariant Tokenizer & Fusion (`tokenizer.py`)
- **PSIT:**
  $$z_{rgb} = \text{Conv}_{1\times 1}\left(\sum_{l=2}^{4} \text{AdaptiveAvgPool}_{7\times 7}(P_l)\right) \longrightarrow (B\cdot T, 49, 256)$$
- **Gated Cross-Domain Fusion:**
  $$g = \sigma\left(\text{Linear}_{512\rightarrow 256}([z_{rgb} \parallel z_{dwt}])\right) \in (0, 1)$$
  $$z_{fused} = \text{Linear}_{256\rightarrow 256}\left(g \odot z_{rgb} + (1 - g) \odot z_{dwt}\right)$$

#### 4. N-MSSS Spatio-Temporal ViT (`nmsss_vit.py`)
- 4 stacked `NMSSSBlock` layers with Pre-LayerNorm:
  1. **Intra-frame Spatial Attention:**
     $$z \leftarrow z + \text{MHSA}(\text{LN}(z)) \quad \text{over 49 tokens per frame independently}$$
  2. **Hyperspherical $L_2$ Temporal Differencing:**
     $$z_{norm, t} = \frac{z_t}{\|z_t\|_2}, \quad \Delta_t = z_{norm, t} - z_{norm, t-1} \quad (\Delta_0 = \mathbf{0})$$
     $$\text{gated} = \text{Conv1D}_{1\times 1}\left([z_{norm} \parallel \Delta]\right) \in (B, T, N, 256)$$
     $$z \leftarrow z + \text{gated} \odot c_t \quad (c_t \in [0, 1] \text{ detector confidence})$$
  3. **Inter-frame Temporal Attention:**
     $$z \leftarrow z + \text{MHSA}(\text{LN}(z)) \quad \text{over } T=8 \text{ frames for each spatial patch location}$$
  4. **Feed-Forward MLP:**
     $$z \leftarrow z + \text{MLP}(\text{LN}(z))$$

#### 5. Readout & Classification (`smsdt.py`)
- Global average pooling across spatial dimension ($49 \rightarrow 1$) and temporal dimension ($8 \rightarrow 1$):
  $$\bar{z} = \frac{1}{T \cdot N} \sum_{t=1}^T \sum_{n=1}^N z_{t, n} \in \mathbb{R}^{256}$$
- Classification:
  $$\hat{y} = \text{Linear}_{128\rightarrow 1}\left(\text{Dropout}_{0.1}\left(\text{GELU}\left(\text{Linear}_{256\rightarrow 128}(\text{LN}(\bar{z}))\right)\right)\right)$$

---

## 7. Multi-Term Loss Function (`smsdt/losses.py`)

$$\mathcal{L}_{total} = \mathcal{L}_{focal} + 0.1 \cdot \mathcal{L}_{contrastive} + 0.05 \cdot \mathcal{L}_{align}$$

1. **Class-Balanced Focal BCE Loss:**
   $$\mathcal{L}_{focal} = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$
   - $\alpha = 0.2$ for class 1 (Fake), $1 - \alpha = 0.8$ for class 0 (Real) to compensate for the real:fake imbalance across manipulation methods.
   - Focusing parameter $\gamma = 2.0$.
   - Label smoothing $= 0.05$.
2. **Supervised Contrastive Loss ($\tau=0.07$):**
   - Computed on post-ViT embeddings $\bar{z}$.
   - Pulls representations of the same manipulation type together; pushes reals and fakes apart.
3. **Cross-Domain Decorrelation Alignment Loss:**
   $$\mathcal{L}_{align} = \frac{1}{B} \sum_{i=1}^B \left|\frac{z_{rgb}^{(i)} \cdot z_{dwt}^{(i)}}{\|z_{rgb}^{(i)}\| \|z_{dwt}^{(i)}\|}\right|$$
   - Penalizes cosine similarity between pooled RGB and Wavelet embeddings to force feature orthogonality.

---

## 8. The Core Problem: Severe Identity Overfitting & Empirical Evaluation

### 7.1 Empirical Benchmark Results (Epoch 23 Best Checkpoint: `outputs/checkpoints_v1_epoch23/best.pt`)
Below are the exact evaluation metrics produced by evaluating our best `main` branch model trained for 23 epochs on our RTX 5090 using `smsdt.eval`:

#### Summary Comparison: Training vs. Validation vs. Test Set
| Metric | Training Set (Epoch 23) | Validation Set (N=307 videos) | Test Set (N=1,027 videos) | Core Forensic Diagnosis |
|---|---|---|---|---|
| **AUC (Frame/Chunk)** | **`0.9998`** | **`0.7649`** | **`0.7125`** | **~28.7 point generalization drop** from train to test |
| **AUC (Video-Level)** | `~1.000` | **`0.7818`** | **`0.7397`** | Video top-k aggregation provides minor +2.7% boost |
| **EER (Equal Error Rate)** | `< 0.01` | **`0.2997`** | **`0.3386`** | 1 in 3 decisions is erroneous at the equal error threshold |
| **EER Threshold** | `~0.50` | **`0.7417`** | **`0.9294`** | **Severe probability threshold drift** (shifts +0.187 from val to test) |
| **FPR @ 0.5 (False Positives)** | `< 0.01` | **`0.3510` (35.1%)** | **`0.4747` (47.5%)** | **Nearly 1 in 2 unseen pristine real faces is falsely flagged as Fake!** |
| **Accuracy @ 0.5** | `99.8%` | **`72.58%`** | **`73.17%`** | Artificially inflated by heavy Real:Fake test imbalance |
| **Precision @ 0.5** | `0.999` | **`0.9013`** | **`0.8949`** | High precision only due to dominance of fake samples |
| **Recall @ 0.5** | `0.999` | **`0.7436`** | **`0.7711`** | Misses ~23–25% of deepfake chunks at default threshold |
| **F1-Score @ 0.5** | `0.999` | **`0.8149`** | **`0.8284`** | Misleadingly high due to class imbalance |

---

#### 7.2 Per-Manipulation Breakdown (Each Fake Scored vs. ALL Real Samples)
Evaluating each manipulation family against pristine real faces reveals the exact structural weakness of our architecture:

| Manipulation Family | Method Category | Val AUC | Val EER | Val F1@0.5 | Test AUC | Test EER | Test F1@0.5 | Test Fake Chunks | Performance Verdict |
|---|---|---|---|---|---|---|---|---|---|
| **Deepfakes** | Autoencoder Face Swap | `0.8854` | `0.2012` | `0.8111` | **`0.7884`** | `0.2955` | `0.6018` | 12,304 | Moderate |
| **FaceShifter** | AEI-Net High-Fidelity Swap | `0.8748` | `0.2167` | `0.7966` | **`0.8017`** | `0.3055` | `0.6087` | 12,304 | Best performing |
| **DeepFakeDetection** | Diverse Actor Benchmarks | — | — | — | **`0.7190`** | `0.3072` | `0.8213` | 67,175 | Mediocre |
| **FaceSwap** | 3D Graphics Landmark Swap | `0.7495` | `0.3183` | `0.6543` | **`0.6569`** | `0.4060` | `0.4814` | 10,054 | Weak |
| **Face2Face** | Facial Expression Reenactment | `0.6671` | `0.3732` | `0.6037` | **`0.6618`** | `0.4025` | `0.5213` | 12,472 | Poor |
| **NeuralTextures** | Neural Rendering / GAN mouth | **`0.5886`** | **`0.4355`** | **`0.4926`** | **`0.5859`** | **`0.4578`** | **`0.4203`** | 10,054 | **Catastrophic Failure (Coin toss)** |

---

### 7.3 Critical Diagnostic Insights from the Empirical Data

1. **Catastrophic Failure on Facial Reenactment (`NeuralTextures` & `Face2Face`):**
   - In `NeuralTextures` (Test AUC `0.5859`), facial identity is **100% preserved**—only the mouth interior and subtle lip-sync textures are modified by a neural renderer.
   - In `Face2Face` (Test AUC `0.6618`), identity is also preserved, with only facial expressions transferred via 3D morphable models.
   - **Diagnosis:** Because our model achieves ~0.80 AUC on identity-swapping methods (`FaceShifter`, `Deepfakes`) but drops to `0.58` on expression/texture reenactment, **the network has learned to detect identity-level displacement rather than synthesis artifacts!**
2. **The 47.5% False Positive Rate on Unseen Identities:**
   - On the Test set, `FPR@0.5 = 0.4747`. Nearly half of pristine real videos of actors unseen during training are classified as Fake.
   - The network has memorized the facial features of the 720 training actors. Any unfamiliar face geometry triggers high forgery activations.
3. **Severe Probability Threshold Drift:**
   - The threshold that balances false positives and false negatives (EER threshold) is `0.7417` on validation and escalates to **`0.9294` on test**.
   - At the standard classification threshold of $0.5$, the model is completely uncalibrated because the logits are severely skewed toward predicting fake.
4. **The $7 \times 7$ Uniform Pooling Low-Pass Bottleneck:**
   - In `PSIT`, downsampling $P_2$ ($56 \times 56$) to $7 \times 7$ via `AdaptiveAvgPool2d` acts as an **$8 \times 8$ uniform box blur**.
   - Synthetic boundary seams and warping discontinuities are 1–3 pixels wide. An $8 \times 8$ box blur blends the seam into surrounding skin pixels, destroying the exact physical evidence of manipulation!
5. **Mean Pooling Readout Dilutes Anomaly Peaks:**
   - In `smsdt.py`, global average pooling over all 49 tokens and 8 frames averages subtle local anomalies (e.g. glitch in the left eye on frame 4) with 391 unmanipulated face patches.
6. **Decorrelation Loss Conflict:**
   - Forcing RGB and Wavelet embeddings to be strictly orthogonal ($\mathcal{L}_{align}$) penalizes both branches if they attempt to focus on the same physical forgery boundary.

---

## 9. Technical Doubts & Internal Architecture Critique

Before asking for architectural redesigns, our pair-programming analysis identified several subtle ambiguities and potential failure mechanisms in the current codebase that we request the reviewing LLM to evaluate:

1. **Hyperspherical $L_2$ Temporal Differencing Gating Mechanism (`nmsss_vit.py`):**
   - In each ViT block:
     $$z_{norm, t} = \frac{z_t}{\|z_t\|_2}, \quad \Delta_t = z_{norm, t} - z_{norm, t-1}$$
     $$\text{gated} = \text{Conv1D}([z_{norm} \parallel \Delta]), \quad z \leftarrow z + \text{gated} \odot c_t$$
   - **Doubt:** Is simple element-wise residual addition ($z + \text{gated} \odot c_t$) effective, or does adding raw normalized differences back into token space introduce high-frequency feature noise that corrupts spatial self-attention in the subsequent block? Would cross-attention between token representations and temporal difference tokens be strictly superior?
2. **Phase Erasure via Haar Energy Absolute Value (`wavelet_branch.py`):**
   - In `wavelet_branch.py`, `pool_energy: True` takes the absolute value $|LH|, |HL|, |HH|$ of the Haar subbands to measure "energy" before passing them to the strided patch stem `Conv2d(9, 256, k=16, s=16)`.
   - **Doubt:** In digital image forensics, phase angle coherence carries vital evidence of GAN and diffusion synthesis. Does taking the absolute magnitude completely eliminate phase discontinuities, reducing the wavelet branch to a rudimentary gradient magnitude/edge detector?
3. **Absence of Learnable Temporal Position Encodings:**
   - Spatial tokens receive a learnable spatial positional embedding `(1, 1, 49, 256)`.
   - However, along the temporal axis ($T=8$), **no temporal position embeddings exist**.
   - **Doubt:** Multi-head self-attention is permutation-invariant without position encodings. Does the temporal attention block treat the 8 frames as an unordered bag of frames? Without temporal position encoding, how can the model capture directional temporal dynamics (such as natural eye blink cadence or lip-sync rhythm)?
4. **Multi-Worker WebDataset Redundancy on Windows Multiprocessing:**
   - In `smsdt/data/dataset.py`:
     `dataset = wds.WebDataset(shard_spec, resampled=True, shardshuffle=True)`
   - **Doubt:** When `DataLoader` runs with `num_workers: 8` on Windows without explicit `.split_by_worker` or `shard_spec` partitioning, does each worker process read the identical shards or duplicate chunks across workers?
5. **ArcFace Affine Warping Inconsistency on Low-Confidence Frames:**
   - When SCRFD detection confidence drops ($c_t < 0.5$) or faces rotate rapidly, EMA landmark smoothing interpolates from previous frames, but `cv2.estimateAffinePartial2D` still warps the face to the canonical template.
   - **Doubt:** Could affine warping on noisy or interpolated landmarks introduce synthetic pixel stretch / shear distortions on real faces, causing the model to learn that "warping artifacts = real face"?
6. **Loss Weight Scale & Gradient Dominance:**
   - Focal Loss drops to $\sim 0.05 - 0.10$ after a few epochs.
   - Contrastive Loss ($\lambda=0.1$) hovers at $\sim 0.3 - 0.5$.
   - Alignment Loss ($\lambda=0.05$) is around $\sim 0.02$.
   - **Doubt:** Does Contrastive Loss dominate backpropagation in later epochs, pulling faces by identity rather than manipulation status?

---

## 10. Master Request for Claude / Frontier Research LLM

> ### **Prompt for Claude:**
> You are an elite AI researcher and principal computer vision architect specializing in deepfake forensics, video transformers, and domain generalization (CVPR, ICCV, ECCV, NeurIPS).
>
> We need a **deeply reasoned, mathematically rigorous, and concrete phase-by-phase implementation plan** to transform this architecture into a **State-of-the-Art (SOTA)** deepfake detector.
>
> ### **Execution Roles & Output Structure Guidelines:**
> To ensure seamless execution, please structure your recommendations so they can be immediately implemented by **Antigravity (an autonomous coding agent with full terminal & filesystem access)** and paired with the **User (the human operator on an RTX 5090)**:
>
> 1. **Antigravity Actionable Tasks:**
>    - For every architectural or pipeline modification, provide the **exact file path**, **exact mathematical equations**, **tensor input/output shapes**, and **clean PyTorch code snippets** that Antigravity can directly implement.
> 2. **User Tasks:**
>    - Clearly label any external actions that must be executed by the human user (e.g. executing PowerShell training commands, managing disk storage, downloading external datasets like CelebV-HQ or VoxCeleb2).
> 3. **Verification Step per Phase:**
>    - Include a lightweight verification check or unit test script for Antigravity to run after every phase to mathematically confirm tensor shapes, gradient flow, and loss behavior before advancing.
>
> ---
>
> ### **Key Architectural Areas to Address:**
>
> #### 1. Upstream Pipeline & Face Cropping: SCRFD + 5-Point ArcFace Alignment
> - **Tight Crop vs. Outer Seam Loss:** Our current pipeline aligns faces using 5 ArcFace landmarks into a standard $224 \times 224$ crop. However, deepfake face-swapping methods (Deepfakes, FaceSwap, FaceShifter) paste a synthetic face mask onto an existing head. The most obvious blending boundaries, skin-tone mismatches, and warping seams occur along the **jawline, hairline, ears, and neck**. Does our tight canonical crop discard the very regions containing the strongest forgery evidence?
> - **Affine Warping Artifacts:** Does `cv2.estimateAffinePartial2D` introduce synthetic interpolation artifacts that confuse the frequency/wavelet branch?
> - **Alternative Recommendations:** Should we switch to a loose bounding-box crop with a $1.3\times - 1.5\times$ margin, dense 3D facial mesh tracking (e.g. MediaPipe 468-point), or dual-scale crops (inner face + outer perimeter context)?
>
> #### 2. Spatial Backbone Audit: EfficientNet-B4 vs. Modern Foundation Models
> - **Identity Memorization Bias:** EfficientNet-B4 was pre-trained on ImageNet-1k classification, an objective designed to identify object classes. When applied to faces, does it naturally latch onto identity features (nose shape, eye distance, skin color) instead of high-frequency synthesis noise?
> - **SOTA Vision Foundation Backbones:** Recent deepfake literature (UnivFD, Forensics Adapter, Effort) demonstrates that self-supervised and multimodal foundation models generalize vastly better than supervised CNNs. Should we replace EfficientNet-B4 with:
>   - **DINOv2 (ViT-B/14 or ViT-L/14):** Self-supervised patch representations capture fine-grained surface geometry and pixel inconsistencies without category bias.
>   - **CLIP (ViT-L/14):** Pre-trained on massive image-text pairs; patch embeddings possess inherent sensitivity to non-photorealistic and synthetic artifacts.
>   - **ConvNeXt-V2 / Swin Transformer V2:** Modern pure-convolutional or hierarchical transformer architectures.
> - **Fine-Tuning Strategy:** Should the foundation backbone be frozen, tuned with LoRA / SVD adapters, or partially fine-tuned at low learning rates ($10^{-5}$)?
>
> #### 3. Spatio-Temporal ViT Robustness Across ALL Deepfake Types
> - **Generalization Across Generator Families:** Deepfakes span fundamentally distinct synthesis paradigms:
>   - Autoencoder latent code replacement (*Deepfakes, FaceSwap, FaceShifter*)
>   - 3D morphable model re-rendering (*Face2Face*)
>   - Neural texture rendering (*NeuralTextures*)
>   - Modern diffusion-based face inpainting (*Face0, Stable Diffusion, StyleGAN3*)
> - Is our decomposed Spatial-then-Temporal ViT (with Hyperspherical $L_2$ differencing $\Delta_t$) robust across all these generator types, or does it overfit to temporal jitter that only exists in older GAN models?
> - **Architectural Alternatives:** What temporal modeling paradigm is proven most robust to modern video deepfakes?
>   - Deformable 3D attention (e.g. Video Swin Transformer, TimeSformer)
>   - State-Space Models (e.g. **Mamba / Vision Mamba [Vim]**) for linear-complexity sequence modeling
>   - Cross-attention over temporal difference tokens vs. simple residual addition ($x + \text{gated} \times conf_t$)
>
> #### 4. Frequency Stream & Wavelet Reformulation
> - Does single-level 2D Haar DWT with a stride-16 patch stem provide meaningful frequency signal, or does it alias high-frequency fingerprints?
> - Does taking the absolute value $|LH|, |HL|, |HH|$ destroy phase information?
> - Would learnable wavelets, DCT phase/magnitude analysis, or shallow late-fusion be superior?
>
> #### 5. Tokenizer & Attention Pooling Upgrades
> - How do we replace the $8 \times 8$ box blur of `AdaptiveAvgPool2d((7, 7))` so that fine-grained 1-pixel boundary artifacts are preserved?
> - How should we replace global average pooling in the readout with attentional or anomaly-peak pooling without gradient saturation?
>
> #### 6. Hardware Maximization (RTX 5090 32GB + 64GB DDR5 + Intel Core Ultra 9 285K)
> - **Batch Size Scaling:** With 31.5 GB VRAM and ~70% currently unused during training, can / should we increase micro-batch size from $B=8$ to $B=16$ or $B=32$? How will a larger micro-batch size benefit contrastive and metric losses?
> - **Temporal Chunk Length ($T$):** Currently $T=8$ frames (~0.27s at 30 fps). Can / should we scale chunk length to $T=16$ or $T=32$ frames to capture physiological forgery anomalies (such as unnatural eye blinking, gaze drift, or lip-sync jitter)?
> - **Spatial Resolution Scaling:** Can we scale face crops from $224 \times 224$ up to $256 \times 256$ or $384 \times 384$? Does higher input resolution prevent subtle 1-to-2 pixel blending seams from disappearing?
> - **High-Capacity Architectures:** What dense token configurations (e.g. $14 \times 14 = 196$ tokens per frame, or multi-scale cross-attention) are now feasible given our 32 GB VRAM?
>
> #### 7. Managing Class Imbalance & Merging External Real Datasets
> - **Optimal Imbalance Strategy:** Currently, our 6 manipulation methods create an up to 1:6 Real-to-Fake imbalance. Is loss re-weighting (Focal $\alpha=0.2$ / BCE `pos_weight`) optimal, or should we employ balanced mini-batch sampling (enforcing 50% real / 50% fake chunks per forward step)?
> - **External Real Video Merging:** Can we import pristine real videos from diverse external datasets (such as **CelebV-HQ**, **VoxCeleb2**, **YouTube-Faces [YTF]**, or **MEAD**) to dramatically expand the real identity distribution from 720 actors to thousands?
> - **Preventing Domain/Codec Leakage:** When mixing external real videos with FaceForensics++, how can we ensure the network doesn't latch onto dataset shortcuts (e.g., classifying FF++ H.264 compression as fake and external videos as real)? What exact standardization steps (re-encoding to FF++ $c23$ CRF 23, resolution matching, identical landmark tracking) are mandatory?
>
> #### 8. Loss Function Formulation & Learning Rate Optimization Strategy
> - **Primary Classification Loss:**
>   - We currently use Class-Balanced Focal BCE ($\alpha=0.2, \gamma=2.0$). Does Focal Loss cause gradient instability on subtle edge-case forgeries, or does it focus excessively on hard identity examples rather than artifact representations?
>   - Is standard Binary Cross-Entropy with calibrated `pos_weight`, Label Smoothing BCE (smoothing=0.1), or Asymmetric Loss (ASL) better suited?
>   - Should deepfake detection be formulated as a **metric learning / hyperspherical compactness problem** (e.g., ArcFace, CosFace, or Sub-center ArcFace) where pristine *Real* faces are constrained into a compact hyperspherical manifold, and *Fakes* are pushed out via an angular margin?
> - **Auxiliary Losses Evaluation:**
>   - **Supervised Contrastive Loss ($\tau=0.07$):** Does SupCon help generalize across manipulation methods, or does it force the network to cluster embeddings by facial identity?
>   - **Cross-Domain Decorrelation Loss ($\mathcal{L}_{align}$):** We currently penalize cosine similarity between RGB and Wavelet token embeddings ($\lambda=0.05$). Is this decorrelation loss actively detrimental when an authentic physical artifact (such as a seam or frequency distortion) exists simultaneously in both spatial and frequency domains?
>   - What auxiliary losses reliably prevent identity memorization (e.g., Self-Supervised Masked Autoencoding [MAE] on real faces, Boundary Seam Supervision, or Frequency Consistency Loss)?
> - **Learning Rate Schedulers & Warming:**
>   - We currently use `OneCycleLR` (`max_lr: 3e-4, div_factor: 25, warmup_steps: 2000`). With 2000 warmup steps, the backbone begins at $1.2 \times 10^{-6}$ and remains effectively dormant for multiple epochs, followed by a rapid spike to $3 \times 10^{-4}$ which induces catastrophic forgetting of pretrained weights.
>   - What is the gold-standard scheduler for fine-tuning dual-branch deepfake transformers? (e.g., **Cosine Annealing with Linear Warmup** for 300–500 steps down to $\eta_{min}=10^{-6}$)?
> - **Backbone Freezing & Layer-wise Learning Rate Decay (LLRD):**
>   - Should we employ a **staged warm-start**: freezing the RGB backbone completely for Epoch 0–2 while training only the Wavelet branch, Tokenizer, and ViT blocks, and then unfreezing the backbone with a $10\times$ lower learning rate ($1 \times 10^{-5}$ for backbone vs. $1 \times 10^{-4}$–$2 \times 10^{-4}$ for transformer heads)?
>   - Should Layer-wise Learning Rate Decay (LLRD) be applied across transformer layers to preserve lower-level spatial/frequency primitives?
> - **Optimizer & Regularization:**
>   - Should we use **AdamW** with decoupled weight decay ($\lambda \in [10^{-4}, 10^{-2}]$), or is **Lion** / **SGD with Nesterov momentum** more resilient against identity memorization?
>   - What weight decay and gradient clipping (`clip_grad_norm_`) thresholds prevent gradient spikes in the temporal differencing and attention modules?
>
> #### 9. Diagnostic on Per-Manipulation Disparity & Pipeline Optimizations Just Implemented
> - **The Empirical Disparity (NeuralTextures 0.5859 vs. FaceShifter 0.8017):**
>   - In our 23-epoch benchmark, `NeuralTextures` collapsed to **`0.5859` Test AUC** (near random coin-toss) and `Face2Face` to **`0.6618` Test AUC**, whereas face-replacement methods (`FaceShifter` `0.8017`, `Deepfakes` `0.7884`) achieved reasonable scores.
>   - Why does our dual-stream architecture fail so completely on neural rendering and facial reenactment where identity is preserved?
>   - What exact mechanisms (e.g. optical flow phase consistency, lip/mouth micro-texture frequency analysis, or boundary-focused patch masking) must be added so the network detects subtle reenactment without relying on identity replacement shortcuts?
> - **Pipeline & DataLoader Adjustments Just Made:**
>   - We have just implemented the following runtime corrections:
>     1. **Worker Scaling:** Increased DataLoader workers from 4 to `workers: 8` and `val_workers: 8` across the 24 cores of the Intel Core Ultra 9 285K to eliminate GPU data-starvation.
>     2. **Decoupled Inference Batch Size:** Increased validation batch size from 8 to `val_batch_size: 32` to fully saturate the RTX 5090 tensor cores under `torch.no_grad()`.
>     3. **Non-Blocking CUDA Evaluation:** Replaced per-batch `.cpu()` calls with non-blocking GPU batch collection and a single bulk host transfer at loop completion, resolving CUDA pipeline stalls.
>     4. **Step Capping:** Added optional `max_val_batches` support for fast epoch-end validation checks.
>   - Given these throughput fixes, what **further corrections, data pipeline adjustments, or caching strategies** (e.g. memory-mapped shards, GPU-accelerated Albumentations/Kornia) are required to maximize training stability and throughput?
>
> #### 10. The 1,000-Step "Epoch" Arbitrary Cap vs. True Full-Dataset Epochs (`resampled=True` vs. `resampled=False`)
> - **The Core Discrepancy:**
>   - In machine learning, **1 epoch by definition equals 1 complete pass through 100% of the unique training samples without replacement**.
>   - However, our training uses:
>     ```python
>     dataset = wds.WebDataset(shard_spec, resampled=True, shardshuffle=True)
>     ```
>   - With `resampled=True`, shards are drawn randomly *with replacement*. The deck never empties, creating an infinite stream.
>   - To prevent an infinite loop, `train.py` artificially breaks after `steps_per_epoch: 1000` optimizer steps ($1,000 \times 4 \times 8 = 32,000$ chunks per epoch).
>   - In our 23-epoch benchmark (`checkpoints_v1_epoch23`), the model only trained for $23 \times 1,000 = 23,000$ optimizer steps total!
> - **Direct Audit Questions & Request for Suggestions from Claude:**
>   1. **Concrete Recommendation on `resampled=False`:**
>      - Should we immediately switch to `resampled=False` with `shardshuffle=True` so that every epoch is guaranteed to visit **100% of the training shards exactly once** without repeating or skipping actors?
>      - Please provide the exact PyTorch / WebDataset code to configure `resampled=False` safely with Windows multiprocessing (`num_workers: 8`), including necessary worker-splitting operators (e.g. `wds.split_by_worker`, `wds.split_by_node`) so workers do not duplicate shards or hang at epoch boundaries.
>   2. **True Step Count Calculation & Scheduler Synchronization:**
>      - If we switch to full-pass epochs, how should we dynamically calculate `steps_per_epoch` (e.g., total unique chunks across all shards divided by effective batch size $B_{eff} = 32$)?
>      - How should `total_steps` and scheduler cycles (CosineAnnealing or OneCycleLR) be calibrated to ensure the learning rate decays cleanly across the full dataset duration rather than terminating prematurely?
>   3. **Balancing Real vs. Fake under Non-Resampled Iteration:**
>      - With an up to 1:6 Real-to-Fake ratio in FF++ $c23$, if we iterate without replacement, pristine real samples will be exhausted far earlier than fake samples in each epoch. How should we structure shard composition or use dual-stream WebDataset merging (`wds.RandomMix` or separate Real/Fake loaders) so batches maintain a 50/50 balance without running out of real samples?
>
> ---
>
> ### **11. Phased SOTA Roadmap Structure:**
> Please organize your final actionable recommendations into clear, sequential phases:
> - **Phase 1: Immediate Anti-Overfitting & Training Dynamics (Within 24–48 Hours)**
>   - Actions for Antigravity (code fixes, loss reweighting, learning rate scheduler, pooling fix).
>   - Actions for User (commands to run, checkpoint evaluation).
> - **Phase 2: Representation & Foundation Backbone Upgrade**
>   - Actions for Antigravity (integration of foundation backbone like DINOv2 / CLIP, frequency stream redesign).
>   - Actions for User (model weight downloading, cache directory setup).
> - **Phase 3: Cross-Dataset Robustness & SOTA Generalization**
>   - Actions for Antigravity (Self-Blended Image synthesis [SBI], cross-compression augmentation, metric evaluation scripts).
>   - Actions for User (external dataset ingestion, cross-testing on Celeb-DF v2 and DFDC).
