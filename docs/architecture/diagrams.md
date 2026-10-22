# S-MSDT Architecture Guide

This document presents the complete end-to-end architectures for **S-MSDT v1** (Phase 1: Lightweight Edge Model) and **S-MSDT v2** (Phase 2: Vision Foundation Model) in a clean, human-readable box-flow format, from **raw video face detection (SCRFD)** down to the **final prediction**.

---

# ARCHITECTURE 1: S-MSDT v1 (Phase 1: Lightweight Edge Model)

> **Goal:** High-throughput, real-time detection (<8 ms per chunk) combining visible RGB textures with high-frequency noise analysis.

### End-to-End Visual Box Flowchart

```
                             ┌───────────────────────────────┐
                             │   0. RAW VIDEO INGESTION      │
                             │   Raw Uncut .mp4 Video File   │
                             └───────────────┬───────────────┘
                                             │
                                             ▼
                             ┌───────────────────────────────┐
                             │  SCRFD FACE DETECTOR & TRACK  │
                             │  (InsightFace SCRFD + ByteTrack)
                             │                               │
                             │ • Detects faces & confidence  │
                             │ • Extracts 106 mesh landmarks │
                             │ • ByteTrack assigns Track IDs │
                             │ • Loose 1.3× margin alignment │
                             │                               │
                             │ OUTPUT: Face Store (data/store)
                             └───────────────┬───────────────┘
                                             │
                                             ▼
                             ┌───────────────────────────────┐
                             │      1. INPUT VIDEO CLIP      │
                             │  8 Aligned Frames (224 × 224) │
                             └───────────────┬───────────────┘
                                             │
                     ┌───────────────────────┴───────────────────────┐
                     │                                               │
                     ▼                                               ▼
       ┌───────────────────────────┐                   ┌───────────────────────────┐
       │  2A. RGB SPATIAL STREAM   │                   │  2B. FREQUENCY STREAM     │
       │  (EfficientNet-B4 + FPN)  │                   │     (2D Haar Wavelet)     │
       │                           │                   │                           │
       │ • Visible texture shapes  │                   │ • Isolates LH, HL, HH     │
       │ • Multi-scale FPN pyramid │                   │ • Discards smooth colors  │
       │ • DensePSIT (14×14 grid)  │                   │ • Strided patch stem      │
       │                           │                   │                           │
       │ OUTPUT: 196 Spatial Toks  │                   │ OUTPUT: 196 Freq Tokens   │
       └─────────────┬─────────────┘                   └─────────────┬─────────────┘
                     │                                               │
                     └───────────────────────┬───────────────────────┘
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │  3. GATED DOMAIN FUSION   │
                               │  (Merge Textures + Noise) │
                               │                           │
                               │ • Learned Gate (g)        │
                               │ • 2D Spatial Pos Embed    │
                               │ • 1D Temporal Pos Embed   │
                               │                           │
                               │ OUTPUT: 196 Hybrid Tokens │
                               └─────────────┬─────────────┘
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │  4. SPATIO-TEMPORAL ViT   │
                               │  (4 Transformer Layers)   │
                               │                           │
                               │ • Spatial Self-Attention  │
                               │ • TemporalDiff (Δ0 = 0)   │
                               │ • Temporal Self-Attention │
                               │                           │
                               │ 💡 Catches frame flicker  │
                               └─────────────┬─────────────┘
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │  5. EVIDENCE READOUT HEAD │
                               │  (Top-10% Most Suspicious)│
                               │                           │
                               │ • Focuses on swap seams   │
                               │ • Prevents face dilution  │
                               └─────────────┬─────────────┘
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │     FINAL PREDICTION      │
                               │  REAL (0.0) vs FAKE (1.0) │
                               └───────────────────────────┘
```

### Component Details

| Step | Module | What It Does | Why It Matters |
| :---: | :--- | :--- | :--- |
| **0** | **SCRFD & ByteTrack** | InsightFace SCRFD detector + ByteTrack + 106-point landmarks (`2d106`) | Fast, accurate face localization; extracts 106 mesh landmarks; enforces a **1.3× loose margin crop** to keep jaw, ear, and forehead swap seams in frame. |
| **1** | **Input Video Clip** | 8 consecutive frames sampled from store (`data/store/`) | Formats the input chunk to `(B, 8, 3, 224, 224)` for model ingestion. |
| **2A**| **RGB Spatial Stream** | EfficientNet-B4 + FPN + DensePSIT | Generates 196 tokens ($14 \times 14$ grid) capturing multi-scale visual blending lines. |
| **2B**| **Frequency Stream** | 2D Haar Wavelet ($LH, HL, HH$) | Isolates high-frequency spectral residuals, catching invisible compression grids and sensor noise mismatches. |
| **3** | **Gated Domain Fusion** | $\text{Token} = g \cdot \text{RGB} + (1 - g) \cdot \text{Wavelet}$ | Learns whether the spatial or frequency view is more suspicious at each face patch. |
| **4** | **Spatio-Temporal ViT**| Spatial Attention $\to$ `TemporalDiff` $\to$ Temporal Attention | Models facial geometry across space and time; `TemporalDiff` ($\Delta_0 = 0$) catches inter-frame boundary flicker and warps. |
| **5** | **Evidence Readout** | Top-$k$ patch evidence pooling (Top-10%) | Focuses purely on the manipulated region (e.g., mouth or eye swap) instead of diluting evidence across the entire face. |

---

# ARCHITECTURE 2: S-MSDT v2 (Phase 2: Vision Foundation Model)

> **Goal:** State-of-the-Art generalization on unseen real-world deepfakes using an adapted foundation vision transformer with LayerNorm-only Parameter-Efficient Fine-Tuning (PEFT), localized difference-mask evidence readout, and temporal differencing.

### End-to-End Visual Box Flowchart

```
                             ┌───────────────────────────────┐
                             │   0. RAW VIDEO INGESTION      │
                             │   Raw Uncut .mp4 Video File   │
                             └───────────────┬───────────────┘
                                             │
                                             ▼
                             ┌───────────────────────────────┐
                             │  SCRFD FACE DETECTOR & TRACK  │
                             │  (InsightFace SCRFD + ByteTrack)
                             │                               │
                             │ • Detects faces & confidence  │
                             │ • Extracts 106 mesh landmarks │
                             │ • ByteTrack assigns Track IDs │
                             │ • Loose 1.3× margin alignment │
                             │                               │
                             │ OUTPUT: Face Store (data/store)
                             └───────────────┬───────────────┘
                                             │
                                             ▼
                             ┌───────────────────────────────┐
                             │      1. INPUT VIDEO CLIP      │
                             │  T=8 Frames (CLIP Normalized) │
                             │  Shape: (B, T, 3, 224, 224)   │
                             └───────────────┬───────────────┘
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │ 2. CLIP ViT-L/14 BACKBONE │
                               │ (Pretrained Foundation)   │
                               │                           │
                               │ • Attention/MLP: FROZEN   │
                               │ • LayerNorms: TUNED (PEFT)│
                               │ • Params: 304M (<0.03% lr)│
                               └─────────────┬─────────────┘
                                             │
                     ┌───────────────────────┴───────────────────────┐
                     │                                               │
                     ▼                                               ▼
       ┌───────────────────────────┐                   ┌───────────────────────────┐
       │  3A. SPATIAL PATCH PATH   │                   │   3B. GLOBAL CLS PATH     │
       │  (Localized Seam Traces)  │                   │   (Whole Scene Semantics) │
       │                           │                   │                           │
       │ • 16×16 Grid (256 tokens) │                   │ • 1024-dim Scene Vector   │
       │ • Projected: 1024 → 384   │                   │ • L2 Normalization        │
       │ • 2D Spatial Pos Embed    │                   │ • Temporal Mean Pooling   │
       │ • 1D Chronological Time   │                   │ • Linear Head (1024 → 1)  │
       │                           │                   │                           │
       │ OUTPUT: (B, T, 256, 384)  │                   │ OUTPUT: Logit_cls (Scene) │
       └─────────────┬─────────────┘                   └─────────────┬─────────────┘
                     │                                               │
                     ▼                                               │
       ┌───────────────────────────┐                                 │
       │  4. SPATIO-TEMPORAL ViT   │                                 │
       │  (L=2 NMSSS Transformer)  │                                 │
       │                           │                                 │
       │ • Factorized Spatial MHA  │                                 │
       │ • TemporalDiff (Δ0 = 0)   │                                 │
       │ • Factorized Temporal MHA │                                 │
       │                           │                                 │
       │ OUTPUT: Inter-frame Dynamics                                │
       └─────────────┬─────────────┘                                 │
                     │                                               │
                     ▼                                               │
       ┌───────────────────────────┐                                 │
       │  5. EVIDENCE READOUT HEAD │                                 │
       │  (Patch Anomaly Detector) │                                 │
       │                           │                                 │
       │ • Linear Map: 384 → 1     │                                 │
       │ • Localized Patch Anomaly │                                 │
       │ • Difference-Mask Loss    │                                 │
       │ • Top-5% Patch Pooling    │                                 │
       │                           │                                 │
       │ OUTPUT: Logit_tok (Seams) │                                 │
       └─────────────┬─────────────┘                                 │
                     │                                               │
                     └───────────────────────┬───────────────────────┘
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │   6. DUAL READOUT FUSION  │
                               │   (Combine Both Streams)  │
                               │                           │
                               │ • Balance: w = σ(β)       │
                               │ • Final = w·Tok + (1-w)·CLS│
                               └─────────────┬─────────────┘
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │ 7. FROZEN CALIBRATION     │
                               │ (Validation Platt Scaling)│
                               │                           │
                               │ • p = σ(a · u + b)        │
                               │ • Frozen τ* = 0.7507      │
                               │ • 0 Test-Set Snooping     │
                               └─────────────┬─────────────┘
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │     FINAL PREDICTION      │
                               │  REAL (0.0) vs FAKE (1.0) │
                               └───────────────────────────┘
```

### Component Details

| Step | Module | Implementation | What It Does & Why It Matters |
| :---: | :--- | :--- | :--- |
| **0** | **SCRFD & ByteTrack** | InsightFace SCRFD + ByteTrack | Detects face bounding boxes, tracks identities across frames, extracts 106-point mesh landmarks, and enforces a **loose 1.3× margin alignment** so swap seams at jawline and hairline remain inside the field of view. |
| **1** | **CLIP Video Input** | `(B, T=8, 3, 224, 224)` | Consecutive face frames normalized with CLIP mean `(0.481, 0.458, 0.408)` and std `(0.269, 0.261, 0.276)`. |
| **2** | **CLIP ViT-L/14 Backbone**| 24-layer ViT (`transformers`) | Retains rich pre-trained semantic representations. **LayerNorm-only tuning (PEFT):** All attention and MLP layers ($>99.97\%$ parameters) are frozen; only LayerNorm parameters ($\approx 96\text{k}$ params) are updated with learning rate $1\times 10^{-4}$ to adapt domain statistics without catastrophic forgetting. |
| **3A**| **Spatial Patch Path** | `nn.LayerNorm(1024) -> Linear(1024, 384)` | Extracts $16 \times 16 = 256$ dense patch tokens per frame, projects to latent dimension $D=384$, and injects learnable 2D spatial embeddings and 1D temporal positional embeddings. |
| **3B**| **Global CLS Path** | `Linear(1024, 1)` on $L_2$-normalized CLS | Averages the global $1024$-dim CLS token across time ($\frac{1}{T}\sum_t \text{Norm}(CLS_t)$) to evaluate scene-level lighting, identity, and macro-semantic consistency. |
| **4** | **Spatio-Temporal ViT**| $L=2$ `NMSSSBlock` layers ($H=6$ heads) | **Factorized Spatio-Temporal Attention:** Decouples intra-frame spatial self-attention from inter-frame temporal self-attention. Features **replicate-padded `TemporalDiff`** ($\Delta_t = \text{Norm}(x_t) - \text{Norm}(x_{t-1})$ with $\Delta_0 = 0$ and zero-initialized projection), guaranteeing detection of inter-frame expression and boundary flicker without acting as a first-frame marker. |
| **5** | **Evidence Readout Head** | `Linear(384, 1)` + Top-$k$ Pooling | Maps each patch token to an anomaly logit ($B, T, 256$). Supervised during training with difference-derived manipulation masks $\|fake_{u8} - real_{u8}\|$ ($w_{\text{pos}} = 3.0$). Top-$5\%$ most suspicious patches are aggregated into the video token logit (`Logit_tok`), preventing subtle manipulation from being diluted across the face. |
| **6** | **Dual Readout Fusion** | $\text{Logit} = \sigma(\beta) \cdot \text{Logit}_{\text{tok}} + (1 - \sigma(\beta)) \cdot \text{Logit}_{\text{cls}}$ | Learned scalar parameter $\beta$ dynamically balances localized boundary evidence with whole-face identity consistency. |
| **7** | **Frozen Calibration** | Class-balanced Platt scaling $(a, b, \tau^*)$ | Evaluates probabilities via $p = \sigma(a \cdot u + b)$ with threshold $\tau^* = 0.7507$ fitted strictly on validation to eliminate test-set snooping (oracle performance) across test benchmarks. |

---

### Two-Stage Training Curriculum

```
[Stage A: Static PEFT Adaptation]
 • Input: T=1 frame per sample (B=32)
 • Objective: Adapt CLIP LayerNorms + train spatial projection & EvidenceReadout
 • Losses: Focal BCE + Difference-Mask BCE (w_pos = 3.0)
 • Output Checkpoint: outputs/checkpoints_v2_stage_a_fixed/best_ema.pt
      │
      ▼
[Stage B: Video Spatio-Temporal Training]
 • Input: T=8 consecutive frames (B=8)
 • Initialization: Preloaded Stage A PEFT weights
 • Sampling: Worker-safe stateless balanced sampling (25% NT, 25% F2F, 20% FS, 15% DF, 15% FSwap)
 • Optimizer: Decoupled weight decay + Dynamic EMA warmup d_t = min(0.999, (1+t)/(10+t))
 • Output Checkpoint: outputs/checkpoints_v2_fixed/best_ema.pt
```

---

# SIDE-BY-SIDE ARCHITECTURAL COMPARISON

```
┌──────────────────────────────────────┬──────────────────────────────────────┐
│         S-MSDT v1 (Phase 1)          │         S-MSDT v2 (Phase 2 Fixed)    │
├──────────────────────────────────────┼──────────────────────────────────────┤
│ • Preprocessing: SCRFD + 1.3× Loose  │ • Preprocessing: SCRFD + 1.3× Loose  │
│ • Facial Landmarks: 106 Points       │ • Facial Landmarks: 106 Points       │
│ • Backbone: EfficientNet-B4 + Wavelet│ • Backbone: OpenAI CLIP ViT-L/14     │
│ • Parameters: ~26.7 Million          │ • Parameters: ~305 Million           │
│ • Trainable Params: 100% (26.7M)     │ • Trainable Params: <1.5% (~4.5M)    │
│ • Backbone Tuning: Full (Diff LR)    │ • Backbone Tuning: LayerNorm (<0.03%)│
│ • Spatial Resolution: 14×14 (196 tok)│ • Spatial Resolution: 16×16 (256 tok)│
│ • Spectral Representation: 2D Haar   │ • Spectral Representation: Pretrained│
│ • Temporal Modeling: 4 NMSSS Blocks  │ • Temporal Modeling: 2 NMSSS Blocks  │
│ • Temporal Differencing: L2 (Δ0 = 0) │ • Temporal Differencing: L2 (Δ0 = 0) │
│ • Readout Pooling: Top-10% Patches   │ • Readout: Dual (Top-5% Tok + CLS)   │
│ • Mask Supervision: None             │ • Mask Supervision: Difference-Masks │
│ • Calibration: Raw 0.50 / Post-hoc   │ • Calibration: Frozen Platt (τ=0.751)│
├──────────────────────────────────────┼──────────────────────────────────────┤
│ BENCHMARK METRICS (Official Test)    │ BENCHMARK METRICS (Official Test)    │
│ • FF++ Macro Test AUC: 0.8510        │ • FF++ Macro Test AUC: 0.9385        │
│ • NeuralTextures AUC:  0.8220        │ • NeuralTextures AUC:  0.8230        │
│ • Face2Face AUC:       0.8900        │ • Face2Face AUC:       0.9650        │
│ • FaceSwap AUC:        0.8050        │ • FaceSwap AUC:        0.9700        │
│ • Deepfakes AUC:       0.9450        │ • Deepfakes AUC:       0.9900        │
│ • FaceShifter AUC:     0.7930        │ • FaceShifter AUC:     0.9440        │
│ • DFD Alone Video AUC: 0.8650        │ • DFD Alone Video AUC: 0.8534        │
│ • Celeb-DF v2 AUC:     —             │ • Celeb-DF v2 AUC:     0.8754        │
│ • Celeb-DF Recall@0.5: —             │ • Celeb-DF Recall@0.5: 69.4%         │
│ • Celeb-DF Calib Prec: —             │ • Celeb-DF Calib Prec: 93.1%         │
└──────────────────────────────────────┴──────────────────────────────────────┘
```

