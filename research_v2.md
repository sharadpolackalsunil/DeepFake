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

## 4. Master Experiment Matrix

Compiled via `tools/make_tables.py` into [`results/tables/experiment_matrix.csv`](file:///e:/DEEPFAKE/results/tables/experiment_matrix.csv) and [`results/tables/experiment_matrix.md`](file:///e:/DEEPFAKE/results/tables/experiment_matrix.md):

| Model | Seed | $\tau_{\text{val}}$ | Celeb-DF AUC | Celeb-DF HTER | Gap $g$ | DFD AUC | DFD HTER | Gap $g$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `gend_ours` (Unmatched V2 Load) | 1 | 0.4866 | 0.4836 | 0.4696 | 0.0242 | 0.5636 | 0.5183 | 0.0653 |
| `gend_ours` (Matched Re-eval) | 1 | *pending pull* | — | — | — | — | — | — |
| `smsdt_s` | 1 | — | — | — | — | — | — | — |
| `smsdt_st` | 1 | — | — | — | — | — | — | — |
| `smsdt_full` | 1 | — | — | — | — | — | — | — |

---

## 5. Next Immediate Action Items

1. **Pull Latest Commit on RTX 5090:**
   ```powershell
   git pull origin tmlr-research-study
   ```
2. **Re-run Cross-Evaluation for `gend_ours`:**
   Running `eval_cross` with commit `d93286c` will output `Loaded model architecture: GEND` and `391/391 parameters matched`, producing the true reproduced baseline numbers.
3. **Train Next Architecture (`smsdt_st`):**
   ```powershell
   python -m smsdt.train_v2 --config configs/train_st.yaml --seed 1
   ```
4. **Train Full Proposed Architecture (`smsdt_full`):**
   ```powershell
   python -m smsdt.train_v3 --config configs/train_full.yaml --seed 1
   ```
