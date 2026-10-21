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

> **Goal:** State-of-the-Art generalization ($\ge 0.965$ AUC on unseen real-world deepfakes like Celeb-DF v2) using an adapted foundation vision transformer.

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
                             │  8 Aligned Frames (CLIP Norm) │
                             └───────────────┬───────────────┘
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │ 2. CLIP ViT-L/14 BACKBONE │
                               │ (Pretrained Foundation)   │
                               │                           │
                               │ • Attention/MLP: FROZEN   │
                               │ • LayerNorms: TUNED       │
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
       │ • Projected to 384 dim    │                   │ • L2 Normalization        │
       │ • Spatio-Temporal ViT     │                   │ • Temporal Mean Pooling   │
       │ • TemporalDiff (Δ0 = 0)   │                   │ • Linear Projection Head  │
       │ • Evidence Readout (5%)   │                   │                           │
       │                           │                   │                           │
       │ OUTPUT: Logit_tok (Seams) │                   │ OUTPUT: Logit_cls (Scene) │
       └─────────────┬─────────────┘                   └─────────────┬─────────────┘
                     │                                               │
                     └───────────────────────┬───────────────────────┘
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │   4. DUAL READOUT FUSION  │
                               │   (Combine Both Scores)   │
                               │                           │
                               │ • Learned Weight: w=σ(β)  │
                               │ • Final = w·Tok + (1-w)·CLS│
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
| **0** | **SCRFD & ByteTrack** | InsightFace SCRFD detector + ByteTrack + 106-point landmarks | Detects face bounding boxes and tracks identities across frames; applies **1.3× loose margin alignment** to preserve swap boundaries. |
| **1** | **CLIP Video Input** | 8 frames normalized with `CLIP_MEAN` / `CLIP_STD` | Standard foundation model input normalization. |
| **2** | **CLIP ViT-L/14 Backbone**| 24-layer Vision Transformer with frozen attention | Retains rich pre-trained world knowledge; fine-tuning only LayerNorms ($<0.03\%$) prevents overfitting to dataset-specific artifacts. |
| **3A**| **Spatial Patch Path** | 256 patch tokens through Spatio-Temporal ViT | Focuses on micro-level boundary seams and temporal motion warps. |
| **3B**| **Global CLS Path** | 1024-dim CLS token averaged across time | Evaluates macro-level semantic and scene-lighting consistency. |
| **4** | **Dual Readout Fusion** | $\text{Final} = w \cdot \text{Logit}_{\text{tok}} + (1 - w) \cdot \text{Logit}_{\text{cls}}$ | Ensembles localized boundary evidence with whole-face identity consistency. |

---

# SIDE-BY-SIDE COMPARISON

```
┌──────────────────────────────────────┬──────────────────────────────────────┐
│         S-MSDT v1 (Phase 1)          │         S-MSDT v2 (Phase 2)          │
├──────────────────────────────────────┼──────────────────────────────────────┤
│ • Face Preprocessing: SCRFD + 1.3×   │ • Face Preprocessing: SCRFD + 1.3×   │
│ • Landmarks: 106 Points (2d106)      │ • Landmarks: 106 Points (2d106)      │
│ • Parameters: 26.7 Million           │ • Parameters: ~305 Million           │
│ • Latency: <8 ms per chunk           │ • Latency: ~30 ms per chunk          │
│ • Backbone: EfficientNet-B4 + Wavelet│ • Backbone: OpenAI CLIP ViT-L/14     │
│ • Spatial Grid: 14×14 (196 tokens)   │ • Spatial Grid: 16×16 (256 tokens)   │
│ • Tuning: Full with differential LR  │ • Tuning: LayerNorm-Only (<0.03%)    │
│ • Target: High-throughput Edge Cams  │ • Target: Cloud Verification API     │
│ • Benchmark: FF++ c23 AUC ≥ 0.95     │ • Benchmark: Celeb-DF v2 AUC ≥ 0.965 │
└──────────────────────────────────────┴──────────────────────────────────────┘
```
