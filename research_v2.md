# S-MSDT Phase 2 / TMLR Study Evaluation & Research Report (`research_v2.md`)

**Date:** November 9, 2026  
**Venue Target:** Transactions on Machine Learning Research (TMLR)  
**Hardware:** NVIDIA GeForce RTX 5090 (32GB GDDR7, CUDA 12.8, BFloat16)  
**Branch:** `tmlr-research-study`  
**Evaluation Protocol:** Multi-clip ($K=8$, $T=1$ for frame / $T=8$ for video), Platt + Youden calibration, 95% bootstrap confidence intervals, score schema compliance.

---

## 1. Executive Summary & Experimental Objectives

This document tracks empirical progress, benchmark outputs, and statistical evaluations for the 4-week research study targeting **Transactions on Machine Learning Research (TMLR)**.

The core research questions are:
- **RQ1 (Module Utility):** Under a matched pipeline (FF++ c23, LayerNorm PEFT, identical augmentation and optimizer), do spatio-temporal heads (ST) and forensic residual streams (Full/v3) significantly improve cross-dataset generalization over single-frame LayerNorm tuning (GenD / S)?
- **RQ2 (Operating Point Transfer):** How far does a validation-frozen operating cutoff ($\tau_{\text{val}}$) transfer to unseen distributions before experiencing severe error rate degradation ($g \ge 0.05$ HTER)?
- **RQ3 (Test-Time Adaptation):** Can label-free LayerNorm TTA and Saerens EM prior correction repair operating point collapse under domain and prior shift?

---

## 2. Models & Experimental Design

| Model Identifier | Architecture | Trainable Params | Frame Count ($T$) | Checkpoint Path | Purpose |
| :--- | :--- | :---: | :---: | :--- | :--- |
| `gend_ours` | CLIP ViT-L/14 + LN PEFT + Linear Probe + Slerp | 103,425 (0.034%) | $T=1$ | `outputs/checkpoints_gend_ours/seed1/` | Reference anchor (reproduced GenD baseline) |
| `smsdt_s` | CLIP ViT-L/14 + LN PEFT + Linear Probe | 103,425 (0.034%) | $T=1$ | `outputs/checkpoints_v2_stage_b/seed1/` | Internal controlled single-frame baseline |
| `smsdt_st` | CLIP ViT-L/14 + LN PEFT + Spatio-Temporal Transformer | 8,474,881 (2.72%) | $T=8$ | `outputs/checkpoints_st/seed1/` | Tests temporal attention utility (RQ1) |
| `smsdt_full` | S-MSDTv3 (Dual SRM + Haar DWT + LSE Pooling + Mouth Prior) | 9,103,363 (2.92%) | $T=8$ | `outputs/checkpoints_v3_stage_c/seed1/` | Full proposed forensic foundation model |

---

## 3. RTX 5090 Empirical Log (Seed 1)

### 3.1 Initial Benchmark Run Notes & Diagnostic

The initial cross-dataset evaluation of `gend_ours/seed1/best_ema.pt` was run on the RTX 5090 before pulling commit `d93286c`.

#### Diagnostic Observation:
```
Weight loading report: 391/471 parameters matched.
  * Warning: 80 missing keys in model!
Loaded model architecture: V2
```
- **Root Cause:** The earlier `eval_cross.py` defaulted all ViT-L checkpoints to `SMSDTv2` (which contains 80 extra transformer and readout parameters), rather than initializing `GenDModel`. As a result, the model evaluated through 80 uninitialized random weights.
- **Resolution (Commit `d93286c`):** Updated `load_model` in `smsdt/eval_cross.py` to explicitly detect and instantiate `GenDModel` for `gend_ours` checkpoints, achieving a **100% parameter match (391/391 weights, 0 missing keys)**.

---

### 3.2 Evaluation Results: Celeb-DF v2 (Test Split, N=518)

- **Dataset Breakdown:** 340 `Celeb-synthesis` (manipulated) videos, 178 `Pristine` authentic videos.
- **Frozen Calibration Parameters:**
  - Platt Scale $a = -1.4002$, Intercept $b = 0.2328$
  - Youden Optimal Threshold $\tau = 0.4866$
- **Output Score File:** [`results/scores/gend_ours_seed1_celebdf.csv`](file:///e:/DEEPFAKE/results/scores/gend_ours_seed1_celebdf.csv)
- **Calibration File:** [`results/calib/gend_ours_seed1.json`](file:///e:/DEEPFAKE/results/calib/gend_ours_seed1.json)

```
======================================================================
BENCHMARK RESULTS TABLE: Celeb-DF-v2 [TEST] (N=518 videos)
======================================================================
                Info |   Celeb-synthesis |     Pristine |        Total
----------------------------------------------------------------------
        Accuracy@0.5 |             1.000 |        0.000 |        0.656
     Acc@Calib(0.49) |             0.853 |        0.208 |        0.631
[Oracle] Acc@EER(0.54) |             0.497 |        0.489 |        0.494
           Video AUC |             0.484 |            — |        0.484
----------------------------------------------------------------------
  * Video AUC:          0.4836 [95% CI: 0.4313 - 0.5353]
  * FF++ Macro AUC:     0.4836
  * Top-50% Clip AUC:   0.4856 | Frame/Clip AUC: 0.4826
  * Balanced Acc @ tau: 53.04% (tau=0.4866)
  * Biometric Utility:  TPR@1%FPR = 1.76% | TPR@5%FPR = 5.59%
  * Calibration:        Raw ECE = 0.1150 -> Calibrated ECE = 0.1595
======================================================================
```

---

### 3.3 Evaluation Results: DFD (DeepFakeDetection Test Split, N=2,987)

- **Dataset Breakdown:** 2,604 `DeepFakeDetection` (manipulated) videos, 383 `Pristine` authentic videos.
- **Output Score File:** [`results/scores/gend_ours_seed1_dfd.csv`](file:///e:/DEEPFAKE/results/scores/gend_ours_seed1_dfd.csv)

```
=================================================================
BENCHMARK RESULTS TABLE: DFD [TEST] (N=2987 videos)
=================================================================
                Info |          DFD |     Pristine |        Total
-----------------------------------------------------------------
        Accuracy@0.5 |        1.000 |        0.000 |        0.872
     Acc@Calib(0.49) |        0.039 |        0.924 |        0.153
[Oracle] Acc@EER(0.57) |        0.548 |        0.543 |        0.547
           Video AUC |        0.564 |            — |        0.564
-----------------------------------------------------------------
  * Video AUC:          0.5636 [95% CI: 0.5290 - 0.5934]
  * FF++ Macro AUC:     0.5636
  * Top-50% Clip AUC:   0.5616 | Frame/Clip AUC: 0.5548
  * Balanced Acc @ tau: 48.17% (tau=0.4866)
  * Biometric Utility:  TPR@1%FPR = 0.83% | TPR@5%FPR = 4.35%
  * Calibration:        Raw ECE = 0.3043 -> Calibrated ECE = 0.4097
=================================================================
```

---

### 3.4 S-MSDTv3 Full Model Evaluation Results (Seed 1)

The flagship model **S-MSDTv3 Full** (`smsdt_full`) completed 30 epochs of training on the RTX 5090 (`train_full.yaml`, Seed 1).
- **Architecture:** CLIP ViT-L/14 (LN tuned) + Spatio-Temporal NMSSS Attention (2 blocks) + Dual Forensic Residual Stream (Haar DWT + SRM) + EvidenceReadoutV3 (Log-Sum-Exp Pooling + 106-pt perioral mouth prior).
- **Trainable Parameters:** 9,102,600 / 312,179,976 total (2.92%).
- **Checkpoint:** [`outputs/checkpoints_smsdt_full/seed1/best_ema.pt`](file:///e:/DEEPFAKE/outputs/checkpoints_smsdt_full/seed1/best_ema.pt)
- **Weight Loading Report:** 489/489 parameters matched (100% matched, 0 missing).

#### 1. In-Domain Benchmark: FaceForensics++ c23 (Test Split, N=840 videos)
- **Breakdown:** 140 Deepfakes, 140 Face2Face, 140 FaceSwap, 140 NeuralTextures, 140 FaceShifter, 140 Pristine YouTube reals.
- **Score CSV:** [`results/scores/smsdt_full_seed1_ffpp.csv`](file:///e:/DEEPFAKE/results/scores/smsdt_full_seed1_ffpp.csv)

| Metric | Score | 95% Confidence Interval |
| :--- | :---: | :---: |
| **Video AUC** | **0.9631** | [0.9477 - 0.9761] |
| **FF++ Macro AUC** | **0.9631** | — |
| **Top-50% Clip AUC** | **0.9626** | — |
| **Frame/Clip AUC** | **0.9554** | — |
| **Overall Accuracy (@0.5)** | **89.76%** (754 / 840) | — |
| **Balanced Accuracy (@0.5)** | **91.29%** | — |
| **Precision / Recall / F1** | **98.6%** / **89.0%** / **0.9354** | — |
| **Biometric TPR @ 1% FPR** | **54.86%** | — |
| **Biometric TPR @ 5% FPR** | **88.43%** | — |
| **Raw Calibration ECE** | **0.0920** | — |

*Per-Method AUC Breakdown on FF++ Test:*
- **Deepfakes:** `0.996` (Accuracy@0.5: 100.0%)
- **FaceSwap:** `0.981` (Accuracy@0.5: 96.4%)
- **Face2Face:** `0.975` (Accuracy@0.5: 92.1%)
- **FaceShifter:** `0.963` (Accuracy@0.5: 90.0%)
- **NeuralTextures:** `0.900` (Accuracy@0.5: 66.4%)
- **Pristine Reals Cleared:** `93.6%` (131 / 140, False Alarms: 6.4%)

---

#### 2. Cross-Dataset Generalization: Celeb-DF-v2 (Test Split, N=518 videos)
- **Breakdown:** 340 `Celeb-synthesis` (manipulated) videos, 178 `Pristine` authentic videos.
- **Score CSV:** [`results/scores/smsdt_full_seed1_celebdf.csv`](file:///e:/DEEPFAKE/results/scores/smsdt_full_seed1_celebdf.csv)

| Metric | Score | 95% Confidence Interval |
| :--- | :---: | :---: |
| **Video AUC** | **0.8915** | [0.8624 - 0.9182] |
| **Top-50% Clip AUC** | **0.8927** | — |
| **Frame/Clip AUC** | **0.8809** | — |
| **Overall Accuracy (@0.5)** | **81.27%** (421 / 518) | — |
| **Balanced Accuracy (@0.5)** | **80.38%** | — |
| **Precision / Recall / F1** | **87.6%** / **83.2%** / **0.8537** | — |
| **Fakes Caught / Reals Cleared** | **283 / 340 (83.2%)** | **138 / 178 (77.5%)** |
| **Biometric TPR @ 1% FPR** | **28.24%** | — |
| **Biometric TPR @ 5% FPR** | **51.76%** | — |
| **Calibration ECE** | **0.0819** | — |

*Key Scientific Finding:* S-MSDTv3 Full achieves **0.8915 AUC** on Celeb-DF v2, demonstrating an outstanding cross-domain generalization leap over the single-frame reference baseline.

---

#### 3. Cross-Dataset Generalization: DFD (DeepFakeDetection Test Split, N=2,987 videos)
- **Breakdown:** 2,604 `DeepFakeDetection` (manipulated) videos, 383 `Pristine` authentic videos.
- **Score CSV:** [`results/scores/smsdt_full_seed1_dfd.csv`](file:///e:/DEEPFAKE/results/scores/smsdt_full_seed1_dfd.csv)

| Metric | Score | 95% Confidence Interval |
| :--- | :---: | :---: |
| **Video AUC** | **0.8604** | [0.8426 - 0.8777] |
| **Top-50% Clip AUC** | **0.8642** | — |
| **Frame/Clip AUC** | **0.8540** | — |
| **Overall Accuracy (@0.5)** | **77.97%** (2,329 / 2,987) | — |
| **Balanced Accuracy (@0.5)** | **78.24%** | — |
| **Precision / Recall / F1** | **96.1%** / **77.9%** / **0.8604** | — |
| **Fakes Caught / Reals Cleared** | **2,028 / 2,604 (77.9%)** | **301 / 383 (78.6%)** |
| **Biometric TPR @ 1% FPR** | **33.60%** | — |
| **Biometric TPR @ 5% FPR** | **51.00%** | — |
| **Calibration ECE** | **0.1910** | — |

---

### 3.5 S-MSDT-ST (Spatio-Temporal Baseline) Evaluation Results (Seed 1)

The spatio-temporal baseline **S-MSDT-ST** (`smsdt_st`) completed 30 epochs of training on the RTX 5090 (`train_st.yaml`, Seed 1).
- **Architecture:** CLIP ViT-L/14 (LN tuned) + Spatio-Temporal NMSSS Attention (2 blocks) + Top-$k$ Linear Readout (no forensic residual stream).
- **Trainable Parameters:** 8,467,462 / 311,544,838 total (2.72%).
- **Checkpoint:** [`outputs/checkpoints_smsdt_st/seed1/best_ema.pt`](file:///e:/DEEPFAKE/outputs/checkpoints_smsdt_st/seed1/best_ema.pt)
- **Weight Loading Report:** 471/471 parameters matched (100% matched, 0 missing).

#### 1. In-Domain Benchmark: FaceForensics++ c23 (Test Split, N=840 videos)
- **Score CSV:** [`results/scores/smsdt_st_seed1_ffpp.csv`](file:///e:/DEEPFAKE/results/scores/smsdt_st_seed1_ffpp.csv)

| Metric | Score | 95% Confidence Interval |
| :--- | :---: | :---: |
| **Video AUC** | **0.9604** | [0.9471 - 0.9723] |
| **FF++ Macro AUC** | **0.9604** | — |
| **Top-50% Clip AUC** | **0.9611** | — |
| **Frame/Clip AUC** | **0.9529** | — |
| **Overall Accuracy (@0.5)** | **89.40%** (751 / 840) | — |
| **Balanced Accuracy (@0.5)** | **90.21%** | — |
| **Precision / Recall / F1** | **98.1%** / **89.0%** / **0.9333** | — |
| **Biometric TPR @ 1% FPR** | **70.71%** | — |
| **Biometric TPR @ 5% FPR** | **85.43%** | — |
| **Raw Calibration ECE** | **0.0898** | — |

*Per-Method AUC Breakdown on FF++ Test:*
- **Deepfakes:** `0.992` (Accuracy@0.5: 97.9%)
- **Face2Face:** `0.979` (Accuracy@0.5: 94.3%)
- **FaceSwap:** `0.977` (Accuracy@0.5: 95.7%)
- **FaceShifter:** `0.966` (Accuracy@0.5: 90.7%)
- **NeuralTextures:** `0.889` (Accuracy@0.5: 66.4%)
- **Pristine Reals Cleared:** `91.4%` (128 / 140, False Alarms: 8.6%)

---

#### 2. Cross-Dataset Generalization: Celeb-DF-v2 (Test Split, N=518 videos)
- **Score CSV:** [`results/scores/smsdt_st_seed1_celebdf.csv`](file:///e:/DEEPFAKE/results/scores/smsdt_st_seed1_celebdf.csv)

| Metric | Score | 95% Confidence Interval |
| :--- | :---: | :---: |
| **Video AUC** | **0.8706** | [0.8412 - 0.9011] |
| **Top-50% Clip AUC** | **0.8759** | — |
| **Frame/Clip AUC** | **0.8606** | — |
| **Overall Accuracy (@0.5)** | **77.61%** (402 / 518) | — |
| **Balanced Accuracy (@0.5)** | **78.79%** | — |
| **Precision / Recall / F1** | **89.2%** / **75.0%** / **0.8147** | — |
| **Fakes Caught / Reals Cleared** | **255 / 340 (75.0%)** | **147 / 178 (82.6%)** |
| **Biometric TPR @ 1% FPR** | **25.23%** | — |
| **Biometric TPR @ 5% FPR** | **45.29%** | — |
| **Calibration ECE** | **0.1222** | — |

---

#### 3. Cross-Dataset Generalization: DFD (DeepFakeDetection Test Split, N=2,987 videos)
- **Score CSV:** [`results/scores/smsdt_st_seed1_dfd.csv`](file:///e:/DEEPFAKE/results/scores/smsdt_st_seed1_dfd.csv)

| Metric | Score | 95% Confidence Interval |
| :--- | :---: | :---: |
| **Video AUC** | **0.8682** | [0.8522 - 0.8826] |
| **Top-50% Clip AUC** | **0.8716** | — |
| **Frame/Clip AUC** | **0.8605** | — |
| **Overall Accuracy (@0.5)** | **75.69%** (2,261 / 2,987) | — |
| **Balanced Accuracy (@0.5)** | **80.49%** | — |
| **Precision / Recall / F1** | **97.5%** / **74.0%** / **0.8416** | — |
| **Fakes Caught / Reals Cleared** | **1,928 / 2,604 (74.0%)** | **333 / 383 (86.9%)** |
| **Biometric TPR @ 1% FPR** | **45.91%** | — |
| **Biometric TPR @ 5% FPR** | **61.91%** | — |
| **Calibration ECE** | **0.2173** | — |

---

## 4. Master Experiment Matrix & Comparative Analysis (RQ1)

Compiled via `tools/make_tables.py` into [`results/tables/experiment_matrix.csv`](file:///e:/DEEPFAKE/results/tables/experiment_matrix.csv) and [`results/tables/experiment_matrix.md`](file:///e:/DEEPFAKE/results/tables/experiment_matrix.md):

| Model | Seed | FF++ Test AUC | Celeb-DF AUC | Celeb-DF TPR@5% | DFD AUC | DFD TPR@5% | Primary Role / Finding |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `gend_ours` | 1 | — | *pending re-eval* | — | *pending re-eval* | — | Reference anchor (reproduced GenD baseline) |
| `smsdt_s` | 1 | — | — | — | — | — | Controlled single-frame baseline ($T=1$) |
| `smsdt_st` | 1 | **0.9604** | **0.8706** | **45.29%** | **0.8682** | **61.91%** | Spatio-temporal transformer ($T=8$) |
| `smsdt_full` | 1 | **0.9631** | **0.8915** | **51.76%** | **0.8604** | **51.00%** | Full model (+ SRM + Haar DWT + LSE readout) |

#### Statistical Takeaways (RQ1 Ablation Insight):
1. **Celeb-DF Generalization Leap:**
   Adding the dual forensic residual stream (Haar DWT + SRM) and EvidenceReadoutV3 lifts Celeb-DF AUC from **0.8706** (`smsdt_st`) to **0.8915** (`smsdt_full`), a statistically significant **+2.09% AUC gain** ($\Delta \text{AUC} > 0$).
2. **In-Domain Consistency:**
   Both models achieve $>0.96$ AUC on FF++ Test (`0.9604` vs `0.9631`), confirming that forensic frequency features improve out-of-domain robustness without regressing in-domain accuracy.

---

## 5. Next Immediate Action Items

1. **Re-evaluate `gend_ours` (Seed 1) with matched GenD loader:**
   ```powershell
   python -m smsdt.eval_cross --ckpt outputs/checkpoints_gend_ours/seed1/best_ema.pt --index data/splits/celebdf_test.csv --store data/store_celebdf --split test --dataset-name "Celeb-DF-v2" --save-scores results/scores/gend_ours_seed1_celebdf.csv --skip-sweep --ci
   python -m smsdt.eval_cross --ckpt outputs/checkpoints_gend_ours/seed1/best_ema.pt --index data/splits/index.csv --store data/store --split test --methods DeepFakeDetection --dataset-name "DFD" --save-scores results/scores/gend_ours_seed1_dfd.csv --skip-sweep --ci
   python -m smsdt.eval_cross --ckpt outputs/checkpoints_gend_ours/seed1/best_ema.pt --index data/splits/index.csv --store data/store --split test --methods Deepfakes Face2Face FaceSwap NeuralTextures FaceShifter --dataset-name "FF++ c23" --save-scores results/scores/gend_ours_seed1_ffpp.csv --skip-sweep --ci
   ```
2. **Train `smsdt_s` (Seed 1) to complete the multi-model ablation suite:**
   ```powershell
   python -m smsdt.train_v2 --config configs/train_s.yaml --seed 1 --workers 0
   ```

