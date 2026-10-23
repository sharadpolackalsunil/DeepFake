# S-MSDT: Scale-Invariant Multi-Scale Deepfake Transformer

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.5+](https://img.shields.io/badge/PyTorch-2.5+-ee4c2c.svg)](https://pytorch.org/)
[![CUDA 12.8](https://img.shields.io/badge/CUDA-12.8-76b900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Hardware](https://img.shields.io/badge/GPU-RTX%205090%20(32GB)-green.svg)](https://www.nvidia.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **State-of-the-Art Deepfake Detection Architecture** integrating dual-stream spatial-frequency representation, spatio-temporal attention, and vision foundation backbones with parameter-efficient LayerNorm adaptation.

---

## Table of Contents
1. [Architectural Overview](#architectural-overview)
2. [Multi-Phase Roadmap & Implementation Status Tracker](#multi-phase-roadmap--implementation-status-tracker)
3. [Data Protocol & Stateless Balanced Fake Sampling](#data-protocol--stateless-balanced-fake-sampling)
4. [Training Formulation, Two-Stage Progression & Loss Functions](#training-formulation-two-stage-progression--loss-functions)
5. [Scientific Evaluation, Calibration & Testing Protocol](#scientific-evaluation-calibration--testing-protocol)
6. [Live Training Results (Phase 1 Baseline)](#live-training-results-phase-1-baseline)
7. [Hardware Configuration (NVIDIA RTX 5090)](#hardware-configuration-nvidia-rtx-5090)
8. [Quick Start & Command Reference](#quick-start--command-reference)
9. [Repository Structure](#repository-structure)

---

## Architectural Overview

S-MSDT is designed to detect both boundary blending artifacts and inter-frame temporal flicker without succumbing to identity memorization or domain shifts:

```
Input: Video Clip (B, T=8, C=3, H=224, W=224)
  │
  ├──► [Spatial RGB Branch]
  │      Phase 1: EfficientNet-B4 + Top-Down FPN → DensePSIT (14×14 grid = 196 tokens)
  │      Phase 2: CLIP ViT-L/14 (16×16 patch tokens = 256 tokens) with LayerNorm-only tuning
  │      Phase 3: CLIP ViT-L/14 coupled with raw-pixel Forensic ResidualStream
  │
  ├──► [Forensic ResidualStream Branch (S-MSDTv3)]
  │      3 Spatial Rich Models (SRM) linear high-pass filters + stationary 2D Haar DWT
  │      Zero-initialized 1x1 conv bottleneck injecting spatial-frequency tampering residuals
  │
  ├──► [Dynamic Landmark-Guided Perioral Prior]
  │      SCRFD 106-point facial landmarks isolated to 20 mouth/jaw reenactment points
  │      Soft Gaussian spotlight prior dynamically projected onto 16×16 patch token grid
  │
  ├──► [N-MSSS Spatio-Temporal Transformer]
  │      L=2 Spatio-Temporal blocks with decoupled spatial attention and TemporalDiff
  │      Inter-frame token velocity: Δ_t = LayerNorm(z_t) - LayerNorm(z_{t-1}) with Δ_0 = 0
  │
  └──► [Evidence Readout Head V3]
         Continuous Log-Sum-Exp anomaly pooling: smooth interpolation between seams and diffuse texture
```

---

## Multi-Phase Roadmap & Implementation Status Tracker

```mermaid
graph LR
    P0["Phase 0: Complete<br/>Offline 1.3x Store (9.4k videos)"] --> P1["Phase 1: Complete<br/>EfficientNet-B4 Baseline<br/>Master AUC: 0.862"]
    P1 --> P2["Phase 2: Complete<br/>S-MSDTv2 Stage B Fixed<br/>FF++ Test Macro: 0.9385"]
    P2 --> P3["Phase 3 / S-MSDTv3: Complete<br/>Hybrid ResidualStream + LSE<br/>Celeb-DF v2: 0.8813 | FF++ Macro: 0.9349"]
```

### Master Implementation Status vs. Execution Roadmap

Following [`SOTA Roadmap`](docs/roadmap/SOTA_Roadmap.md) and the comprehensive scientific report in [`research.md`](research.md), the repository documents all audited project milestones:

| Step | Roadmap Component | Status | Codebase Modifications & Implementations |
| :---: | :--- | :---: | :--- |
| **1** | **D2 Worker Freeze Verification** | ✅ **Complete** | Confirmed that Windows multiprocessing with `persistent_workers: true` isolated worker dataset memory, freezing fake method rotation on Method 0 (`Deepfakes`). |
| **2** | **Hygiene Bug Fixes (D1, D2, D3)** | ✅ **Complete** | • **D1:** Added difference-derived manipulation masks $\|fake_{u8} - real_{u8}\|$ with $w_{\text{pos}}=3.0$ in [`video_dataset.py`](smsdt/data/video_dataset.py) & [`losses.py`](smsdt/losses.py).<br>• **D2:** Implemented worker-safe stateless balanced sampling `pick_fake` (25% NT, 25% F2F, 20% FS, 15% DF, 15% FSwap).<br>• **D3:** Added dynamic EMA warmup $d_t = \min(0.999, \frac{1+t}{10+t})$ in [`train_v2.py`](smsdt/train_v2.py).<br>• Excluded scalars (`beta`, `mix`, `p.ndim < 2`) from weight decay in [`smsdt_v2.py`](smsdt/models/smsdt_v2.py).<br>• Filtered 584 DFD videos from validation split (leaving 840 pure FF++ videos).<br>• Accelerated validation with `val_start_epoch: 10` (skips epochs 1–9, saves `last.pt` atomically). |
| **3** | **Scientific Testing & Calibration Engine** | ✅ **Complete** | Integrated directly into [`eval_cross.py`](smsdt/eval_cross.py) without repo clutter:<br>• Class-balanced Platt scaling (`fit_platt`) via L-BFGS.<br>• Optimal decision threshold search via Youden's $J = \text{TPR} - \text{FPR}$ (`youden_tau`).<br>• 1,000-iteration stratified bootstrap 95% Confidence Intervals (`boot_ci`).<br>• Biometric utility computation ($\text{TPR} @ 1\%$ and $5\%$ FPR).<br>• 15-bin Expected Calibration Error (`compute_ece`).<br>• Stage A ($T=1$), Stage B ($T=8$), and Stage C (v3) auto-detection from checkpoint metadata.<br>• Relabeled test-tuned EER as `[Oracle] Acc@EER`.<br>• CLI flags: `--fit-calib`, `--calib`, `--ci`, `--save-scores`. |
| **4** | **Re-Run Stage B Baseline ("v2-fixed")** | ✅ **Complete** | Trained Stage A Fixed & Stage B Fixed (40 epochs) with worker-safe sampling and difference masks. Val Macro AUC reached **0.9535** (EER dropped to **0.1214**, F1 to **0.9363**). |
| **5** | **S-MSDTv3 Architectural Modules** | ✅ **Complete** | Implemented dual-stream high-frequency injection (`ResidualStream` with SRM + Haar DWT in [`v3_modules.py`](smsdt/models/v3_modules.py)), Dynamic Landmark Perioral Attention Prior ([`mouth_prior.py`](smsdt/data/mouth_prior.py)), and `EvidenceReadoutV3` continuous Log-Sum-Exp pooling. |
| **6** | **Stage C Two-Phase Training** | ✅ **Complete** | Trained 28-epoch curriculum via [`smsdt/train_v3.py`](smsdt/train_v3.py) using `configs/train_v3_stage_c.yaml` with pairwise margin ranking loss ($\mathcal{L}_{\text{rank}}$). Checkpoint saved to `outputs/checkpoints_v3_stage_c/best_ema.pt`. |
| **7** | **Scientific Research Report** | ✅ **Complete** | Authored publication-grade academic report in [**`research.md`**](research.md) detailing theory, architecture, data engineering, calibration protocols, and cross-generational comparisons. |
| **8** | **Full Benchmark & Robustness Suite** | ✅ **Complete** | Evaluated Stage C across all benchmark suites under frozen calibration:<br>• **FF++ Test Macro AUC:** **0.9349** (DF: **0.9890**, FS: **0.9660**, F2F: **0.9580**, FSh: **0.9420**, NT: **0.8190**)<br>• **Celeb-DF v2 Test Video AUC:** **0.8813** [95% CI: 0.8514–0.9094] (*New Project Record*)<br>• **DFD Isolated Test AUC:** **0.8522** [95% CI: 0.8325–0.8696]<br>• **Master Test Video AUC ($N=3,687$):** **0.8697** [95% CI: 0.8537–0.8858] with **79.31% Acc @ 0.50** and **96.7% Precision**. |
| **9** | **Camera-Ready Results Tables** | ✅ **Complete** | Master cross-generational comparison tables, sensitivity sweeps, and confusion matrices documented in [`results.md`](results.md) and [`research.md`](research.md). |

### Phase 0: Leak-Free Offline Preprocessing (Complete)
- **Problem Solved:** Runtime on-the-fly face detection throttled training and tight ArcFace crops cut off boundary swap seams sitting at the jaw and hairline.
- **Implementation:** [`smsdt/preprocess/build_store.py`](smsdt/preprocess/build_store.py) converts raw `.mp4` videos into an aligned, memory-mapped frame store (`data/store/`).
- **Specifications:** Loose **1.3× face crops** (preserving jawline context), 106-point facial landmarks (`lmk106.npy`), and frame indexing (`fidx.npy`) storing 128 frames at $256 \times 256$ uint8.
- **Dataset Scaled:** **9,425 / 9,431 videos** successfully preprocessed across all FaceForensics++ splits.

### Phase 1: In-Domain Baseline & Anti-Overfitting (Complete)
- **Problem Solved:** Original pipeline suffered from token dilution ($7 \times 7 = 49$ tokens), initial frame marker bugs ($\Delta_0 \ne 0$), unbounded class imbalance, and worker handle leaks on Windows.
- **Key Enhancements:**
  - **DensePSIT:** Expanded token resolution to $14 \times 14 = 196$ spatial tokens.
  - **Temporal Integrity:** Enforced $\Delta_0 = 0$ at $t=0$ and added temporal positional encodings.
  - **Exact-Epoch Paired Dataset:** Every real video is matched 1:1 with an aligned manipulation clip.
  - **Stateless Balanced Fake Sampling:** Eliminates multiprocessing worker isolation bugs, ensuring continuous, balanced exposure to all 5 manipulation types every epoch.
  - **Optimization:** Cosine warmup scheduler, decoupled weight decay, and Model EMA ($\beta=0.999$).
- **Benchmarked Result:** Master Test Video AUC **0.862** across $N=3,687$ videos (FF++ c23: **0.851**, DFD: **0.865**).

### Phase 2: S-MSDTv2 with Foundation Backbone (Complete)
- **Problem Solved:** Classical CNN backbones overfit to compression and fail on subtle temporal manipulation.
- **Architecture (`SMSDTv2`):**
  - **Backbone:** OpenAI CLIP ViT-L/14 yielding $16 \times 16 = 256$ dense patch tokens.
  - **LayerNorm-Only Tuning (`LNCLIP-DF`):** Freezes all multi-head attention weights ($>99.97\%$ parameters), updating only LayerNorm parameters ($<0.03\%$ parameters, $\approx 96\text{k}$ params) to preserve semantic representations.
  - **Temporal Fusion:** 2-layer Spatio-Temporal Transformer with `TemporalDiff` and dual readout (`EvidenceReadout` + normalized CLS head).
- **Two-Stage Training Progression:**
  - **Stage A (Spatial Alignment, $T=1$):** Converged at **0.8278 Val AUC** (`Deepfakes`: 0.991, `FaceSwap`: 0.970).
  - **Stage B Fixed (Video Differencing, $T=8$):** Reached **0.9535 Val Macro AUC** and **0.9385 FF++ Test Macro AUC** (`FaceSwap`: **0.9700**, `Deepfakes`: **0.9900**, `FaceShifter`: **0.9440**, `Face2Face`: **0.9650**, `NeuralTextures`: **0.8230**). Checkpoint: `outputs/checkpoints_v2_fixed/best_ema.pt`.

### Phase 3: S-MSDTv3 Stage C (Complete)
- **Problem Solved:** Contrastive vision foundation models (CLIP ViT-L/14) enforce invariance to high-frequency pixel noise, creating a blind spot for expression reenactment and neural rendering textures (`NeuralTextures`).
- **Architecture (`SMSDTv3`):**
  - **Dual-Branch Forensic Residual Stream (`ResidualStream`):** Bypasses semantic layers by computing 3 SRM linear high-pass filters and 4 stationary 2D Haar DWT subbands directly on raw pixels, fused through a zero-initialized conv bottleneck.
  - **Dynamic Landmark-Guided Perioral Prior:** Isolates 20 mouth and jaw landmarks (`52 to 71`), dynamically projecting soft Gaussian spotlight attention priors onto the $16 \times 16$ patch token grid.
  - **Continuous Log-Sum-Exp Anomaly Pooling (`EvidenceReadoutV3`):** Smoothly interpolates between max-pooling (for sharp boundary seams) and mean-pooling (for diffuse facial texturing).
  - **Pairwise Margin Ranking Loss ($\mathcal{L}_{\text{rank}}$):** Optimizes the ROC area on paired authentic/manipulated videos of the same identity with margin $1.0$.
- **Empirical Milestones:**
  - **FF++ Test Macro AUC:** **0.9349** across all 5 manipulation methods (`NeuralTextures` sustained at **0.8190**).
  - **Celeb-DF v2 Zero-Shot Test Video AUC:** **0.8813** [95% CI: 0.8514–0.9094] (*New Project Record*).
  - **DFD Isolated Test Video AUC:** **0.8522** [95% CI: 0.8325–0.8696].
  - **Master Test Video AUC ($N=3,687$):** **0.8697** [95% CI: 0.8537–0.8858], **79.31% Accuracy @ 0.50**, **96.7% Precision**. Checkpoint: `outputs/checkpoints_v3_stage_c/best_ema.pt`.

---

## Data Protocol & Stateless Balanced Fake Sampling

### The Multiprocessing Isolation Discovery & Fix
Previously, manipulation methods were indexed via an epoch-modulo formula:
$$\text{fake\_method} = \text{self.fakes}[\text{target\_id}][\text{epoch} \pmod 5]$$

**The Root Cause (D2 Diagnostic):** Under PyTorch multiprocessing on Windows with `persistent_workers: true`, worker processes maintain isolated memory spaces. Updating `dataset.set_epoch(epoch)` in the main process thread **never crossed process boundaries** to persistent workers. Consequently, persistent workers remained permanently frozen on Method 0 (`Deepfakes`), starving `NeuralTextures` and `Face2Face` and causing `NeuralTextures` validation AUC to collapse to ~0.54.

### The Fixed Architecture: Worker-Safe Stateless Balanced Sampling (`pick_fake`)
We replaced the stateful `set_epoch()` dependency with **stateless weighted sampling** implemented directly inside `__getitem__`:

$$\mathcal{P}(\text{method}) = \frac{w_{\text{method}}}{\sum_{m} w_m}$$

Because sampling is stateless and driven by worker-local random generators, every worker process continuously and uniformly streams all 5 manipulation methods across every batch and epoch without cross-process synchronization:

| Manipulation Method | Weight ($w_m$) | Sampling Ratio | Role & Strategic Objective |
| :--- | :---: | :---: | :--- |
| **`NeuralTextures`** | **0.25** | **25%** | Boosted 1.25× to overcome foundation model semantic bias against neural facial rendering. |
| **`Face2Face`** | **0.25** | **25%** | Boosted 1.25× to detect facial expression and reenactment deformation artifacts. |
| **`FaceShifter`** | **0.20** | **20%** | Trains robust detection of high-fidelity GAN face-swapping blending seams. |
| **`Deepfakes`** | **0.15** | **15%** | Autoencoder face-swapping baseline (converges rapidly). |
| **`FaceSwap`** | **0.15** | **15%** | Classical 3D graphics face replacement (converges rapidly). |

### Difference-Derived Patch Mask Supervision (D1 Fix)
In original FaceForensics++ data, pristine/fake pairs lacked ground-truth blending masks, leaving the patch localization head unguided. We implemented on-the-fly difference masks:

$$M_{\text{raw}} = \max_{c \in \{R,G,B\}} \left| X_{\text{fake}}^{(c)} - X_{\text{real}}^{(c)} \right|, \qquad M = \text{Downsample}\left(\frac{M_{\text{raw}}}{\tau_{95}}\right) > 0.1$$

This generates pixel-aligned spatial supervision targeting the exact manipulated pixels (e.g., mouth region for `NeuralTextures`, boundary seams for `FaceShifter`).

### Guarantees
1. **Continuous Multi-Method Exposure:** All 5 manipulation types are trained in parallel every single epoch, eliminating catastrophic forgetting and worker starvation.
2. **Exact 50/50 Real/Fake Batch Balance:** Every training batch consists of exact 1:1 real and manipulated clip pairs for pristine facial anchoring.
3. **Multiprocessing Stability:** Fully compatible with `num_workers: 4` and `persistent_workers: true` on Windows workstations.

---

## Training Formulation, Two-Stage Progression & Loss Functions

### 1. Two-Stage Training Paradigm (Stage A vs. Stage B)

S-MSDTv2 employs a decoupled two-stage curriculum that separates spatial forensic token calibration from spatio-temporal video modeling:

| Feature | Phase A (Stage A) | Phase B (Stage B) |
| :--- | :--- | :--- |
| **Input Shape** | 1 Frame ($T=1$) | 8 Contiguous Frames ($T=8$) |
| **Batch Size** | 64 pairs = 128 frames / step | 8 pairs = 16 clips (128 frames) / step |
| **Trainable Params** | `<0.03%` (CLIP LayerNorms only, ~76k params) | TemporalDiff + Spatio-Temporal Blocks + Dual Readout Head |
| **Supervision** | Frame-level classification ($\mathcal{L}_{\text{cls}}$) | Video classification + 16×16 patch localization ($\mathcal{L}_{\text{cls}} + 0.5 \cdot \mathcal{L}_{\text{mask}}$) |
| **Primary Goal** | Shift CLIP from semantics to spatial artifacts | Model inter-frame flicker, temporal jitter, and patch localization |
| **Hardware Throughput** | ~11 steps/epoch (~2.5 sec/epoch on RTX 5090) | ~90 steps/epoch (~50-55 sec/epoch on RTX 5090) |

#### Why Train Separately?
1. **Preventing Gradient Shock:** The temporal transformer and patch readout head start from random initialization. Training both simultaneously from scratch sends erratic, high-variance gradients into CLIP's LayerNorms, corrupting the visual backbone before spatial forensic tokens stabilize.
2. **Preventing Video Memorization:** Video training operates with smaller clip batches ($B=16$), which risks overfitting to specific actor identities if the backbone is tuned concurrently. Stage A uses large frame batches ($B=128$) to enforce invariant spatial boundaries.
3. **Cracking NeuralTextures (NT):**
   - NeuralTextures only alters the **mouth cavity and facial expressions** via neural rendering; it contains no face-swap boundary seams.
   - On static frames ($T=1$), NT easily mimics pristine faces. Stage A dedicates **25% balanced sampling** to NT, forcing CLIP LayerNorms to become sensitive to neural rendering noise.
   - Stage B introduces **`TemporalDiff` ($\Delta X_t = X_t - X_{t-1}$)** and **Patch-Mask Supervision**: this exposes NT's inter-frame oral jitter and explicitly drives the 16×16 spatial attention to lock onto the mouth patches.

---

### 2. Multi-Task Loss Formulation

The primary detection objective optimizes a balanced classification and localized patch-mask loss:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{cls}} + \lambda_{\text{mask}} \cdot \mathcal{L}_{\text{mask}}$$

#### Component 1: Video Classification Loss ($\mathcal{L}_{\text{cls}}$)
Binary Cross-Entropy with Logits with **Label Smoothing ($\epsilon = 0.05$)**:

$$\text{target} = y \cdot (1 - \epsilon) + 0.5 \cdot \epsilon \quad \implies \quad y_{\text{real}} \to 0.025, \quad y_{\text{fake}} \to 0.975$$

$$\mathcal{L}_{\text{cls}} = -\sum \left[ \text{target} \cdot \log(\sigma(\hat{y})) + (1 - \text{target}) \cdot \log(1 - \sigma(\hat{y})) \right]$$

*Rationale:* Prevents gradient saturation and overconfidence on easily detected manipulation methods (`FaceSwap`, `Deepfakes`), maintaining strong gradient updates for subtle reenactments (`NeuralTextures`, `Face2Face`).

#### Component 2: Patch-Mask Localization Loss ($\mathcal{L}_{\text{mask}}$)
Spatial supervision across the $16 \times 16 = 256$ patch token grid:

$$\mathcal{L}_{\text{mask}} = -\frac{1}{N_{\text{patches}}} \sum_{i=1}^{256} \left[ w_{\text{pos}} \cdot m_i \cdot \log(\sigma(p_i)) + (1 - m_i) \cdot \log(1 - \sigma(p_i)) \right]$$

- **Supervision Signal ($m_i$):** Generated from pixel-aligned difference masks between fake and pristine counterparts ($|fake_{u8} - real_{u8}| > 0.1$).
- **Positive Class Weight ($w_{\text{pos}} = 3.0$):** Corrects spatial class imbalance, as forged facial regions (e.g., mouth, eyes) occupy `<20%` of total face patches.

#### Stage Weighting:
- **Phase A:** $\lambda_{\text{mask}} = 0.0$ (Pure frame classification).
- **Phase B:** $\lambda_{\text{mask}} = 0.5$ (Joint video classification and spatial patch localization).

---

### 3. Accelerated Validation & Split Isolation Protocol

- **Strict Identity Isolation:** The dataset split enforces 0 overlap between train, val, and test partitions in both video IDs (`0 overlap`) and target actor identities (`0 overlap`).
- **Pure In-Domain FF++ Evaluation:** DeepFakeDetection (DFD) actor reals and fakes are filtered from validation, leaving exactly $N=840$ videos (140 reals + 140 per each of the 5 FF++ methods).
- **Validation Acceleration (`val_start_epoch: 10`):** Validation is completely bypassed for Epochs 1 to 9, allowing the model to train at 100% throughput with atomic `last.pt` checkpoints. Validation activates at Epoch 10, computing exact per-manipulation AUC breakdowns and saving `best_ema.pt`.

---

## Scientific Evaluation, Calibration & Testing Protocol

Following the strategic engineering analysis in [`Strategic Plan v2`](docs/roadmap/Strategic_Plan_v2.md), the repository adopts strict peer-review benchmark protocols designed for IEEE TIFS, CVPR, and ICCV.

### 1. Eliminating Test-Set Snooping: Frozen Platt Calibration
- **The Core Flaw in Legacy Pipelines:** Standard evaluations often fit EER decision thresholds on the *test set itself*. In top-tier computer vision venues, tuning thresholds on the evaluation split is considered test-set snooping (oracle performance).
- **The Protocol:**
  1. **Calibration Split:** Class-balanced Platt scaling is fitted strictly on FF++ validation mean logits $u$:
     $$p = \sigma(a \cdot u + b), \quad \text{where } a, b \text{ are fitted via weighted logistic regression}$$
  2. **Frozen Operating Threshold:** Threshold $\tau$ is chosen on validation by maximizing **Youden's J statistic**:
     $$J = \text{TPR} - \text{FPR} \quad \iff \quad \arg\max_\tau \text{Balanced Accuracy}(\tau)$$
  3. **Evaluation Transfer:** The parameter tuple $(a, b, \tau)$ is permanently frozen and applied unchanged across FF++ test, DFD, and Celeb-DF v2.
  4. **Oracle Labeling:** Any metric requiring test-set threshold selection is explicitly designated as `oracle_*` and documented as an upper bound rather than an operational metric.

---

### 2. Standardized Cross-Dataset Benchmark Protocol (Celeb-DF v2)
- **Zero In-Domain Leakage:** The model is trained purely on FaceForensics++ c23 (5 methods + reals). No Celeb-DF v2, DFD, or synthetic pseudo-fakes are present in training, guaranteeing an honest out-of-distribution evaluation.
- **Official Test Manifest:** Evaluated on the official 518 test videos (178 real, 340 fake, 65.6% fake prevalence).
- **Multi-Clip Video Inference:** Evaluates $K=8$ clips of $T=8$ frames per video with 1.3× loose face crops, aggregating clip predictions via mean logit pooling.
- **Reporting Hierarchy:**
  - **Primary Headline Metric:** Video-Level AUC with **95% Confidence Intervals** computed via 2,000 stratified video-level bootstrap resamples.
  - **Secondary Operational Metrics:** AP, EER, TPR @ FPR = 1% & 5%, and Balanced Accuracy at frozen $\tau$.
  - **Metric Separation:** Video-level metrics and frame-level metrics are reported in distinct tables to ensure fair comparisons against DeepfakeBench baselines.

---

### 3. Macro Model Selection & Regression Floor Guards
- **Macro FF++ Val Metric:** Checkpoint selection is governed by the unweighted macro-mean across all five manipulation methods:
  $$\text{Macro AUC} = \frac{1}{5} \left( \text{AUC}_{\text{Deepfakes}} + \text{AUC}_{\text{Face2Face}} + \text{AUC}_{\text{FaceSwap}} + \text{AUC}_{\text{NeuralTextures}} + \text{AUC}_{\text{FaceShifter}} \right)$$
  This eliminates evaluation bias from DFD, which previously constituted ~79% of validation fake videos.
- **Floor Guards on Established Methods:**
  $$\text{Save Checkpoint } \iff \text{Macro AUC} > \text{Best} \quad \mathbf{AND} \quad \text{AUC}_{\text{FaceSwap}} \ge 0.975 \quad \mathbf{AND} \quad \text{AUC}_{\text{Deepfakes}} \ge 0.975$$
  Guarantees that performance on established autoencoder and 3D swaps never regresses while optimizing for hard reenactment methods.

---

### 4. Forensic Diagnostic & Robustness Suite
To verify genuine forensic feature learning rather than shortcut learning or identity memorization:

| Diagnostic Tool | Target Pathology | Verification Mechanism |
| :--- | :--- | :--- |
| **`tools/diag_identity_probe.py`** | Identity Memorization (D3) | Measures whether intra-identity real/fake pairs cluster by actor identity rather than manipulation status. |
| **`tools/diag_domain_shift.py`** | Domain Bias (D4) | Audits false-positive stability across YouTube compression reals vs. Actor studio reals. |
| **`tools/eval_robustness.py`** | Perturbation Resilience | Evaluates degradation curves under JPEG compression ($Q \in [30, 90]$), Gaussian blur ($\sigma \in [1, 7]$), and downscaling ($s \in [0.25, 0.75]$). |
| **Method Counter Tracking** | Worker Multiprocessing Starvation (D2) | Real-time logging of manipulation methods trained per epoch to ensure persistent workers maintain uniform exposure. |
| **Bootstrap CI Engine** | Statistical Significance | 2,000 stratified video-level resamplings to establish true significance over standard errors ($\text{SE} \approx 0.015$). |

---

## Live Training Results (Phase 1 Baseline)

Active training log running on **NVIDIA GeForce RTX 5090** (50 Epochs, 90 batches/epoch, batch size 8 pairs = 16 video clips per step):

| Epoch | Active Method | Train Loss | Train AUC | Val AUC | Val EER | Val F1@0.5 | Deepfakes | Face2Face | FaceSwap | NeuralTextures | FaceShifter | DFD |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **00** | Deepfakes | 0.7021 | 0.5210 | 0.4801 | 0.5000 | 0.0000 | 0.5117 | 0.4652 | 0.4861 | 0.4820 | 0.4188 | 0.4913 |
| **01** | Face2Face | 0.5629 | 0.8048 | 0.5814 | 0.4462 | 0.0000 | 0.7095 | 0.5938 | 0.5151 | 0.6046 | 0.4359 | 0.5958 |
| **02** | FaceSwap | 0.4944 | 0.8599 | 0.6234 | 0.4154 | 0.0000 | 0.7087 | 0.7299 | 0.6139 | 0.6106 | 0.3914 | 0.6423 |
| **03** | NeuralTextures | 0.6106 | 0.7430 | 0.6619 | 0.3923 | 0.0000 | 0.7792 | 0.8212 | 0.7247 | 0.7228 | 0.4380 | 0.6086 |
| **04** | FaceShifter | 0.5282 | 0.8307 | 0.7025 | 0.3462 | 0.0000 | 0.8042 | 0.8276 | 0.6795 | 0.7580 | 0.5217 | 0.6787 |
| **05** | Deepfakes | 0.4028 | 0.9208 | 0.7225 | 0.3269 | 0.0000 | 0.8517 | 0.7838 | 0.5822 | 0.7634 | 0.5867 | 0.7362 |
| **07** | FaceSwap | 0.3937 | 0.9245 | 0.7520 | 0.3115 | 0.0338 | 0.8539 | 0.8206 | 0.6308 | 0.7391 | 0.5873 | 0.7908 |
| **09** | FaceShifter | 0.4411 | 0.8982 | 0.7666 | 0.2923 | 0.1849 | 0.8287 | 0.8415 | 0.7149 | 0.7633 | 0.6304 | 0.7830 |
| **12** | FaceSwap | 0.3564 | 0.9447 | 0.7920 | 0.2923 | 0.5339 | 0.8862 | 0.8467 | 0.6424 | 0.7770 | 0.6407 | 0.8425 |
| **14** | FaceShifter | 0.3654 | 0.9396 | 0.8017 | 0.2654 | 0.6384 | 0.8777 | 0.8571 | 0.7252 | 0.7906 | 0.6649 | 0.8297 |
| **17** | FaceSwap | 0.3209 | 0.9586 | 0.8109 | 0.2654 | 0.7148 | 0.8938 | 0.8505 | 0.6849 | 0.7759 | 0.6646 | 0.8666 |
| **19** | FaceShifter | 0.3397 | 0.9518 | 0.8217 | 0.2500 | 0.7871 | 0.9004 | 0.8686 | 0.7291 | 0.8074 | 0.6864 | 0.8569 |
| **23** | NeuralTextures | 0.4394 | 0.9011 | 0.8267 | 0.2500 | 0.7852 | 0.9097 | 0.8698 | 0.7251 | 0.7941 | 0.7054 | 0.8658 |
| **24** | FaceShifter | 0.3350 | 0.9523 | 0.8384 | 0.2346 | 0.8398 | 0.9128 | 0.8925 | 0.7534 | 0.8102 | 0.7187 | 0.8699 |
| **25** | Deepfakes | 0.2250 | 0.9878 | 0.8291 | 0.2577 | 0.7449 | 0.8979 | 0.8776 | 0.7601 | 0.7899 | 0.7112 | 0.8619 |
| **39 (Best)** | FaceShifter | 0.2185 | 0.9892 | **0.8566** | **0.2154** | **0.8542** | **0.9250** | **0.9020** | **0.8010** | **0.8270** | **0.7820** | **0.8720** |

### Official Master Test Benchmarks (Detailed Report in [results.md](results.md) and [research.md](research.md))

Evaluated across `N = 3,687` test videos on NVIDIA GeForce RTX 5090 (strict scientific reporting with zero unfixed runs):

| Model / Milestone | Architecture & Components | FF++ Macro AUC | Deepfakes | FaceSwap | Face2Face | FaceShifter | NeuralTextures | DFD Test AUC | Celeb-DF v2 AUC | Master Video AUC | Precision @ 0.50 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Phase 1 Baseline** | EfficientNet-B4 + Wavelet + DensePSIT | 0.8510 | 0.9450 | 0.8050 | 0.8900 | 0.7930 | **0.8220** | 0.8650 | 0.8560 | 0.8620 | 95.9% |
| **Phase 2 Stage B Fixed** | CLIP ViT-L/14 + TempDiff (LN-tuned) | **0.9385** | **0.9900** | **0.9700** | **0.9650** | **0.9440** | **0.8230** | 0.8534 | 0.8754 | **0.8714** | **99.9%** |
| **Phase 3 Stage C (v3)** | CLIP ViT-L + ResidualStream + Prior + LSE | **0.9349** | **0.9890** | **0.9660** | **0.9580** | **0.9420** | **0.8190** | **0.8522** | **0.8813** | **0.8697** | **96.7%** |

*For complete confusion matrices, threshold sweep tables, Celeb-DF v2 comparisons against published baselines, and methodological derivations, see [results.md](results.md) and [research.md](research.md).*

---

## Hardware Configuration (NVIDIA RTX 5090)

The codebase is hardened specifically for Windows workstations powered by high-end GPUs:

| Hardware Component | Profile | Optimization Applied |
| :--- | :--- | :--- |
| **GPU** | NVIDIA GeForce RTX 5090 32 GB (`sm_120`) | Native PyTorch `bfloat16` autocast |
| **CPU** | Intel Core Ultra 9 285K (24 cores) | `workers: 4` per worker pool prevents memory thrashing |
| **Windows Memory Model** | Windows `spawn` architecture | Detached numpy frame buffers (`.copy()`) prevent `cudaErrorAlreadyMapped` |
| **Handle Management** | Windows Pagefile Commitment | `val_workers: 0`, `prefetch_factor: 2` eliminates Error 1455 handle exhaustion |
| **Weight EMA** | Exponential Moving Average | $\beta = 0.999$ smoothed checkpoints saved to `outputs/checkpoints_v3_stage_c/best_ema.pt` |

---

## Quick Start & Command Reference

### 1. Phase 1 Training (In-Domain Baseline)
```powershell
# Launch Phase 1 training (50 epochs)
python -m smsdt.train --config-dir configs
```

### 2. Phase 2 Training (S-MSDTv2 "v2-fixed" with CLIP ViT-L/14)
```powershell
# Step 1: Install vision foundation dependencies
pip install "transformers>=4.40" safetensors

# Step 2: Cache OpenAI CLIP ViT-L/14 weights
python -c "from transformers import CLIPVisionModel; CLIPVisionModel.from_pretrained('openai/clip-vit-large-patch14')"

# Step 3: Train Stage A Fixed (Spatial Alignment, T=1, ~1.5 mins on RTX 5090)
# Skips validation for Epochs 1-9; validates and saves at Epoch 10
python -m smsdt.train_v2 --config configs/train_v2_stage_a.yaml

# Step 4: Train Stage B Fixed (Video Differencing with Balanced Sampling & Diff Masks, T=8, ~25 mins)
# Warm-starts from Stage A fixed checkpoint; outputs to outputs/checkpoints_v2_fixed/
python -m smsdt.train_v2 --config configs/train_v2_stage_b.yaml
```

### 3. Phase 3 Training (S-MSDTv3 Stage C with ResidualStream & LSE Readout)
```powershell
# Launch Stage C training fusing CLIP semantic tokens with SRM + 2D Haar DWT residuals
python -m smsdt.train_v3 --config configs/train_v3_stage_c.yaml
```

### 4. Scientific Cross-Dataset Evaluation Suite (Stage C)
```powershell
# Benchmark FaceForensics++ c23 Master Test Split (N = 3,687 videos)
python -m smsdt.eval_cross --ckpt outputs/checkpoints_v3_stage_c/best_ema.pt --split test --dataset-name "FF++ c23" --out-csv outputs/benchmark_v3.csv --save-scores outputs/scores/ffpp_test_v3.csv --ci

# Benchmark Celeb-DF v2 Zero-Shot Cross-Dataset (N = 518 videos)
python -m smsdt.eval_cross --ckpt outputs/checkpoints_v3_stage_c/best_ema.pt --index data/splits/celebdf_test.csv --store data/store_celebdf --split test --dataset-name "Celeb-DF v2" --out-csv outputs/benchmark_v3.csv --save-scores outputs/scores/celebdf_v3.csv --ci

# Benchmark Google/Jigsaw DFD Standalone (N = 2,987 videos)
python -m smsdt.eval_cross --ckpt outputs/checkpoints_v3_stage_c/best_ema.pt --split test --methods DeepFakeDetection --dataset-name "DFD" --out-csv outputs/benchmark_v3.csv --save-scores outputs/scores/dfd_v3.csv --ci

# Perturbation Robustness Testing (JPEG, Blur, Rescaling)
python -m tools.eval_robustness --ckpt outputs/checkpoints_v3_stage_c/best_ema.pt --split test --out-csv outputs/robustness_v3.csv

# Diagnostic Probes (Identity Memorization & Domain Shift)
python -m tools.diag_identity_probe --ckpt outputs/checkpoints_v3_stage_c/best_ema.pt
python -m tools.diag_domain_shift --ckpt outputs/checkpoints_v3_stage_c/best_ema.pt
```

---

## Repository Structure

```
DEEPFAKE/
├── configs/
│   ├── data.yaml                 # Store root (data/store), index manifest, chunking (T=8)
│   ├── model.yaml                # Model hyperparameters (embed_dim=256, depth=4, grid=14)
│   ├── train.yaml                # Phase 1 training config (epochs=50, p_sbi=0.0, workers=4)
│   ├── train_v2_stage_a.yaml     # Phase 2 Stage A config (CLIP ViT-L/14, T=1, batch=32)
│   ├── train_v2_stage_b.yaml     # Phase 2 Stage B config (CLIP ViT-L/14, T=8, batch=8)
│   └── train_v3_stage_c.yaml     # Phase 3 Stage C config (S-MSDTv3, T=8, batch=8)
├── data/
│   ├── splits/index.csv          # Identity-disjoint split manifest (train, val, test)
│   ├── store/                    # Preprocessed loose 1.3x FF++ frame store
│   └── store_celebdf/            # Preprocessed loose 1.3x Celeb-DF v2 frame store
├── smsdt/
│   ├── models/
│   │   ├── rgb_branch.py         # EfficientNet-B4 + FPN extractor
│   │   ├── wavelet_branch.py     # 2D Haar Discrete Wavelet Transform stem
│   │   ├── tokenizer.py          # DensePSIT (14x14) + Gated Fusion
│   │   ├── nmsss_vit.py          # Spatio-Temporal ViT + L2 Temporal Differencing
│   │   ├── clip_encoder.py       # OpenAI CLIP ViT-L/14 with LayerNorm-only tuning
│   │   ├── smsdt.py              # Full S-MSDT v1 architecture
│   │   ├── smsdt_v2.py           # Full S-MSDT v2 foundation model architecture
│   │   ├── smsdt_v3.py           # Full S-MSDTv3 Stage C hybrid architecture
│   │   └── v3_modules.py         # ResidualStream (SRM + Haar DWT) & EvidenceReadoutV3 (LSE)
│   ├── data/
│   │   ├── video_dataset.py      # Map-style PairedClipDataset with stateless balanced sampling
│   │   ├── mouth_prior.py        # Dynamic landmark-guided perioral Gaussian spotlight prior
│   │   └── sbi_video.py          # Temporally consistent Self-Blended Images (SBI)
│   ├── preprocess/
│   │   └── build_store.py        # Multi-worker 1.3x loose frame & 106-landmark extractor
│   ├── losses.py                 # Multi-task detection loss (BCE + patch-mask BCE)
│   ├── losses_v3.py              # S-MSDTv3 loss (BCE + patch-mask + pairwise ranking L_rank)
│   ├── train.py                  # Phase 1 training orchestrator
│   ├── train_v2.py               # Phase 2 two-stage training orchestrator
│   ├── train_v3.py               # Phase 3 Stage C training orchestrator
│   ├── eval.py                   # In-epoch validation evaluator
│   └── eval_cross.py             # Cross-dataset benchmark & calibration runner
├── tools/
│   ├── eval_robustness.py        # Robustness curves under JPEG, blur, and scaling
│   ├── diag_identity_probe.py    # D3 identity memorization probe
│   └── diag_domain_shift.py      # D4 domain shift auditor
├── tests/
│   ├── test_blockers.py          # Unit tests for Phase 1 components
│   └── test_v3.py                # Unit tests for S-MSDTv3 Stage C architecture
├── research.md                   # Comprehensive scientific report (11 chapters)
└── results.md                    # Master empirical evaluation logs and sensitivity sweeps
```

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.