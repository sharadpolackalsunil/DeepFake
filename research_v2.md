# S-MSDT Phase 2 / TMLR Study Evaluation & Research Report (`research_v2.md`)

**Date:** November 12, 2026  
**Venue Target:** Transactions on Machine Learning Research (TMLR)  
**Hardware:** NVIDIA GeForce RTX 5090 (32GB GDDR7, CUDA 12.8, BFloat16)  
**Branch:** `feat/eval-and-benchmarks`  
**Evaluation Protocol:** Multi-clip ($K=8$, $T=1$ for frame / $T=8$ for video), Platt + Youden calibration, 95% bootstrap confidence intervals, score schema compliance.

---

> **Audit & Execution Update (2026-11-12, branch `feat/eval-and-benchmarks` @ `973647d`).** Read [`docs/novelty_audit.md`](file:///e:/DEEPFAKE/docs/novelty_audit.md),
> [`docs/decisions.md`](file:///e:/DEEPFAKE/docs/decisions.md) (D1–D15 & Section 6), [`docs/credibility_plan.md`](file:///e:/DEEPFAKE/docs/credibility_plan.md), [`docs/research_v2_corrections.md`](file:///e:/DEEPFAKE/docs/research_v2_corrections.md), and [`docs/real_shot_benchmark.md`](file:///e:/DEEPFAKE/docs/real_shot_benchmark.md) for pre-registration context.
> Repository state on `feat/eval-and-benchmarks`: **`52`** per-video score CSVs ([`results/scores/`](file:///e:/DEEPFAKE/results/scores)),
> **`13`** Platt calibration JSONs ([`results/calib/`](file:///e:/DEEPFAKE/results/calib)), **`30`** ablation CSVs
> ([`results/scores_ablation/`](file:///e:/DEEPFAKE/results/scores_ablation): `18` readout + `12` branch), **`26`** files in [`results/tables/`](file:///e:/DEEPFAKE/results/tables),
> **`30`** files in [`results/tables_ablation/`](file:///e:/DEEPFAKE/results/tables_ablation), and **`6`** sensitivity files in [`results/tables_resample/`](file:///e:/DEEPFAKE/results/tables_resample).
> - **D14 Pure-DFD Fix Across All Audit Tools:** Earlier DFD score CSVs mixed `140` FF++ test reals (`video_id = real_NNN`) labelled `dfd_actor` into the `243` pure DFD reals (`383` total). [`tools/stats.py`](file:///e:/DEEPFAKE/tools/stats.py) (`PURE_NEGATIVE_ID_PREFIX = {"dfd": "dfd_real"}`) now restricts DFD across `tools.stats`, `tools.threshold_transfer`, and `tools.fpr_control` to **`243` reals / `18` actor identities (`2,847` videos, `273` clusters)**; DFD remains `dev_overlap` (exploratory).
> - **Gate G1 Resolved (§9):** Official author checkpoint `hf:yermandy/GenD_CLIP_L_14` (`gend_official_seed0`) achieves `0.9452` FF++ Test, `0.9371` FF++ Val, `0.8840` pure DFD (`0.9076` raw unstripped), and **`0.9327` Celeb-DF v2** — above every S-MSDT variant (`smsdt_full`: `0.8882`, `smsdt_s`: `0.8826`, `gend_ours`: `0.8294`). Under dev-split `source_quantile` calibration (`--source ffppval`), `gend_official` inflates Celeb-DF FPR **`3.15x`** to **`15.73%`** at $\alpha=0.05$ (`31.46%` at $\alpha=0.10$) and pure-DFD FPR **`3.87x`** to **`19.34%`**, showing cross-domain FPR inflation also occurs for the official GenD checkpoint.
> - **H1 & H2 Post-PR-#12 Reruns (§§4.3, 5):** Updated `tools.stats` ($N_{\text{boot}}=10{,}000$, Bonferroni $m=2$, D14 pure negatives) leaves **H1 `undetermined (underpowered)`** (`ST vs S` Celeb-DF: `-0.0122`, `97.5% CI [-0.0312, +0.0068]`; `Full vs S` Celeb-DF: `+0.0056`, `97.5% CI [-0.0142, +0.0262]`; pure DFD `Full vs S` is `-0.0325`, `worse`). Updated `tools.threshold_transfer` using dev-split Platt calibration (`tau_cal`) and **D13 resolved identity clusters** (`121` real identities on Celeb-DF) yields **H2 `undetermined`** (`smsdt_s` seed 1 clears $g \ge 0.05$ at `0.0984`, `95% CI [0.0796, 0.1283]`, while seeds 2–3 stay at `0.0259` and `0.0389`).
> - **FPR Control Audit — H4a/H4b, H5, Prior Stress & Source Resampling (§10):** Re-run via [`tools/fpr_control.py`](file:///e:/DEEPFAKE/tools/fpr_control.py) (`--source ffppval`, `--confirmatory-seeds 3 4 5`, pure DFD `243` reals / `18` clusters). On Celeb-DF ($\alpha=0.05$), `smsdt_full` mean FPR is `12.17%` (`2.43x`; `seed3` confirmatory passes `h4_failure` at `15.73%`, cons CI `[0.1046, 0.2180]`), `smsdt_st` is `8.43%` (`1.69x`; `0/3`), `smsdt_s` is `5.81%` (`1.16x`), and `gend_official` is `15.73%` (`3.15x`); on pure DFD ($\alpha=0.05$), `smsdt_full` rises to `13.44%` (`seed1` `18.11%`, `h4_failure = True`) and `gend_official` reaches `19.34%`. Including source-quantile sampling uncertainty (`--resample-source`, [`results/tables_resample/`](file:///e:/DEEPFAKE/results/tables_resample)) drops `h4_failures` from `6/52` to `0/52`. Formal **H4a is `undetermined (fewer than 3 confirmatory seeds)`**, paired **H4b** `Full - S` is `+0.0674`, `+0.0337`, `+0.0899`, and **H5 remains `falsified (degenerate fits)`** (`12` pure-DFD rows flagged `degenerate = True`).
> - **Eval-Time Readout & D15 Branch Ablations (§10.4):** Swapping `smsdt_full` token readout at evaluation time does not change FPR (`11.99%–12.17%` vs `12.17%` learned; Celeb-DF AUC `0.8845–0.8886` vs `0.8882`). Under D15 `--branch-ablation`, `token_only` keeps both Celeb-DF AUC (`0.8843`) and FPR inflation (`11.80%`, mean TPR `67.94%`), whereas `cls_only` drops Celeb-DF AUC to `0.7456` and TPR to `31.18%` (`4.87%` FPR).
> - **Real-Shot Budget Transfer Benchmark — H7 Development Evidence Only (§11):** [`tools/real_shot.py`](file:///e:/DEEPFAKE/tools/real_shot.py) evaluates few-shot authentic calibration on Celeb-DF (`121` real identities) and DFD (`18` real actors). Original **H7-v1 is `not supported`** (shape transfer `loc` fails and over-corrects `gend_official`); **H7-v2 receives development support on Celeb-DF only** (`n = 20` authentic identities with identity-level `conformal` keeps mean FPR `0.0391–0.0543` across all `13` runs at `0.78–0.96` oracle TPR), while DFD (`n <= 10`) is `undetermined`. Confirmatory testing requires sign-off of [`docs/decisions.md`](file:///e:/DEEPFAKE/docs/decisions.md) Section 6 and a new unseen dataset (`>= 40` real identities).
> - **H3a, H6 & Robustness (§§6–8):** **H3a `falsified`** on HTER under calibrated EM (`prior_sweep.py`; Bayes accuracy improves at extreme priors $\pi \in \{0.1, 0.9\}$); **H6 `falsified`** (online consistency TTA collapses to `0.5409` AUC; single model/seed/dataset); robustness perturbation curve verified on `smsdt_full_seed1` Celeb-DF only.
> - **Paper-Consistency Audit (2026-11-12, D16-D19, [`docs/decisions.md`](file:///e:/DEEPFAKE/docs/decisions.md)):** (D16) one TPR@FPR definition (`smsdt/metric_defs.py`); the old §4.1 matrix mixed ROC-interpolated eval-log values (seed-1 rows) with percentile-table values (seeds 2-3), up to 4.1 points apart on Celeb-DF. §4.1 is now regenerated from the score files with pure DFD. (D17) `smsdt_s` has **777,478** trainable parameters, not 103,425 (`docs/param_counts.md`); `gend_ours` vs `smsdt_s` is not a matched-footprint comparison. (D18) Celeb-DF's 121 real "identities" are 51 celebrities + 70 unlabelled YouTube-real clips; §11 adds identity-weighted FPR and per-source strata. (D19, proposed) seeds 4-5 dropped in favour of the H7 confirmatory dataset.
> - **Provenance Note:** Section 3 and §4.1–4.2 retain the per-run raw RTX 5090 evaluation logs (where DFD CSVs had `2,987` rows including `140` mislabelled FF++ test reals) for traceability; their TPR@1%/5% values use the pre-D16 ROC-interpolation rule and their DFD rows include the 140 mislabelled FF++ reals. **Do not quote §3 numbers in the paper; use §4.1 (regenerated) and §§4.3–12.**

---

## 1. Executive Summary & Experimental Objectives

This document tracks empirical progress, benchmark outputs, and statistical evaluations for the 4-week research study targeting **Transactions on Machine Learning Research (TMLR)**.

The core research questions are:
- **RQ1 (Module Utility):** Under a matched pipeline (FF++ c23, LayerNorm PEFT, identical augmentation and optimizer), do spatio-temporal heads (ST) and forensic residual streams (Full/v3) significantly improve cross-dataset generalization over single-frame LayerNorm tuning (GenD / S)?
- **RQ2 (Operating Point Transfer):** How far does a validation-frozen operating cutoff ($\tau_{\text{val}}$) transfer to unseen distributions before experiencing severe error rate degradation ($g \ge 0.05$ HTER)?
- **RQ3 (Test-Time Adaptation):** Can label-free LayerNorm TTA and Saerens EM prior correction repair operating point collapse under domain and prior shift?

---

## 2. Models & Experimental Design

*Parameter counts from `python -m tools.count_params` ([`docs/param_counts.md`](file:///e:/DEEPFAKE/docs/param_counts.md)); checkpoint directories from each config's `output_dir`.*

| Model Identifier | Architecture | Trainable Params | Frame Count ($T$) | Checkpoint Path | Purpose |
| :--- | :--- | :---: | :---: | :--- | :--- |
| `gend_ours` | CLIP ViT-L/14 + LN PEFT + Linear Probe + Slerp | 103,425 (0.034%) | $T=1$ | `outputs/checkpoints_gend_ours/seed1/` | GenD-style in-house baseline (not a reproduction; Celeb-DF 0.83 vs 0.933 for the official weights) |
| `smsdt_s` | CLIP ViT-L/14 + LN PEFT + `SMSDTv2` head with depth 0 (projection + attention-MIL/top-k readout + CLS head) | 777,478 (0.256%) | $T=1$ | `outputs/checkpoints_smsdt_s/seed1/` | Internal controlled single-frame baseline |
| `smsdt_st` | CLIP ViT-L/14 + LN PEFT + Spatio-Temporal Transformer | 8,467,462 (2.72%) | $T=8$ | `outputs/checkpoints_smsdt_st/seed1/` | Tests temporal attention utility (RQ1) |
| `smsdt_full` | S-MSDTv3 (Dual SRM + Haar DWT + LSE Pooling + Mouth Prior) | 9,102,600 (2.92%) | $T=8$ | `outputs/checkpoints_smsdt_full/seed1/` | Full proposed forensic foundation model |

---

## 3. RTX 5090 Empirical Log (Seed 1)

### 3.1 Initial Benchmark Diagnostic & Fix

- **Initial Anomaly:** The first cross-dataset evaluation of `gend_ours/seed1/best_ema.pt` was run before commit `d93286c`, where `eval_cross.py` defaulted all ViT-L checkpoints to `SMSDTv2` (evaluating through 80 uninitialized random weights, yielding near-random ~0.48-0.56 AUC).
- **Resolution:** `load_model` in `smsdt/eval_cross.py` was updated to explicitly instantiate `GenDModel`, achieving a **100% parameter match (393/393 weights matched)** and auto-detecting $T=1$ frame evaluation.

---

### 3.2 GenD-Style In-House Baseline Evaluation Results (Seed 1, Not a Reproduction)

Evaluated with the matched `GenDModel` architecture (CLIP ViT-L/14 + LN PEFT + Linear Probe + Slerp, 393/393 parameters matched, $T=1$ frame) trained under our in-house FF++ c23 recipe (Celeb-DF `0.83` vs `0.9327` for official weights `gend_official`; see §9).
- **Trainable Parameters:** 103,425 / 303,180,801 total (0.034%).
- **Checkpoint:** [`outputs/checkpoints_gend_ours/seed1/best_ema.pt`](file:///e:/DEEPFAKE/outputs/checkpoints_gend_ours/seed1/best_ema.pt)

#### 1. In-Domain Benchmark: FaceForensics++ c23 (Test Split, N=840 videos)
- **Score CSV:** [`results/scores/gend_ours_seed1_ffpp.csv`](file:///e:/DEEPFAKE/results/scores/gend_ours_seed1_ffpp.csv)

| Metric | Score | 95% Confidence Interval |
| :--- | :---: | :---: |
| **Video AUC** | **0.9047** | [0.8823 - 0.9235] |
| **FF++ Macro AUC** | **0.9047** | — |
| **Top-50% Clip AUC** | **0.9033** | — |
| **Frame/Clip AUC** | **0.8775** | — |
| **Overall Accuracy (@0.5)** | **78.21%** (657 / 840) | — |
| **Balanced Accuracy (@0.5)** | **83.50%** | — |
| **Precision / Recall / F1** | **97.8%** / **75.6%** / **0.8525** | — |
| **Biometric TPR @ 1% FPR** | **63.86%** | — |
| **Biometric TPR @ 5% FPR** | **72.71%** | — |
| **Raw Calibration ECE** | **0.2644** | — |

*Per-Method AUC Breakdown on FF++ Test:*
- **Deepfakes:** `0.995` (Accuracy@0.5: 97.9%)
- **FaceSwap:** `0.985` (Accuracy@0.5: 95.7%)
- **FaceShifter:** `0.948` (Accuracy@0.5: 82.1%)
- **Face2Face:** `0.919` (Accuracy@0.5: 77.1%)
- **NeuralTextures:** `0.676` (Accuracy@0.5: 25.0%)
- **Pristine Reals Cleared:** `91.4%` (128 / 140, False Alarms: 8.6%)

---

#### 2. Cross-Dataset Generalization: Celeb-DF-v2 (Test Split, N=518 videos)
- **Score CSV:** [`results/scores/gend_ours_seed1_celebdf.csv`](file:///e:/DEEPFAKE/results/scores/gend_ours_seed1_celebdf.csv)

| Metric | Score | 95% Confidence Interval |
| :--- | :---: | :---: |
| **Video AUC** | **0.8321** | [0.7973 - 0.8617] |
| **Top-50% Clip AUC** | **0.8311** | — |
| **Frame/Clip AUC** | **0.7895** | — |
| **Overall Accuracy (@0.5)** | **69.88%** (362 / 518) | — |
| **Balanced Accuracy (@0.5)** | **74.65%** | — |
| **Precision / Recall / F1** | **91.8%** / **59.4%** / **0.7214** | — |
| **Fakes Caught / Reals Cleared** | **202 / 340 (59.4%)** | **160 / 178 (89.9%)** |
| **Biometric TPR @ 1% FPR** | **31.18%** | — |
| **Biometric TPR @ 5% FPR** | **52.59%** | — |
| **Calibration ECE** | **0.1940** | — |

---

#### 3. Cross-Dataset Generalization: DFD (DeepFakeDetection Test Split, N=2,987 videos)
- **Score CSV:** [`results/scores/gend_ours_seed1_dfd.csv`](file:///e:/DEEPFAKE/results/scores/gend_ours_seed1_dfd.csv)

| Metric | Score | 95% Confidence Interval |
| :--- | :---: | :---: |
| **Video AUC** | **0.8776** | [0.8631 - 0.8909] |
| **Top-50% Clip AUC** | **0.8758** | — |
| **Frame/Clip AUC** | **0.8457** | — |
| **Overall Accuracy (@0.5)** | **73.28%** (2,189 / 2,987) | — |
| **Balanced Accuracy (@0.5)** | **81.56%** | — |
| **Precision / Recall / F1** | **98.5%** / **70.4%** / **0.8213** | — |
| **Fakes Caught / Reals Cleared** | **1,834 / 2,604 (70.4%)** | **355 / 383 (92.7%)** |
| **Biometric TPR @ 1% FPR** | **55.86%** | — |
| **Biometric TPR @ 5% FPR** | **69.13%** | — |
| **Calibration ECE** | **0.3100** | — |

---

### 3.3 S-MSDT-S (Spatial-Only Baseline) Evaluation Results (Seed 1)

The spatial-only baseline **S-MSDT-S** (`smsdt_s`) completed 30 epochs of training on the RTX 5090 (`train_s.yaml`, Seed 1).
- **Architecture:** CLIP ViT-L/14 with LayerNorm PEFT + `SMSDTv2` head with 0 temporal blocks (1024->384 projection, attention-MIL/top-k readout, CLS head; $T=1$ frame).
- **Trainable Parameters:** 777,478 / 303,854,854 total (0.256%) [corrected 2026-11-12, D17; previously misreported as 103,425].
- **Checkpoint:** [`outputs/checkpoints_smsdt_s/seed1/best_ema.pt`](file:///e:/DEEPFAKE/outputs/checkpoints_smsdt_s/seed1/best_ema.pt)
- **Weight Loading Report:** 415/415 parameters matched (100% matched, 0 missing).
- **Evaluation Mode:** Auto-detected Phase A checkpoint $\rightarrow$ evaluated with $T=1$ frame per clip.

#### 1. In-Domain Benchmark: FaceForensics++ c23 (Test Split, N=840 videos)
- **Score CSV:** [`results/scores/smsdt_s_seed1_ffpp.csv`](file:///e:/DEEPFAKE/results/scores/smsdt_s_seed1_ffpp.csv)

| Metric | Score | 95% Confidence Interval |
| :--- | :---: | :---: |
| **Video AUC** | **0.9587** | [0.9463 - 0.9707] |
| **FF++ Macro AUC** | **0.9587** | — |
| **Top-50% Clip AUC** | **0.9562** | — |
| **Frame/Clip AUC** | **0.9395** | — |
| **Overall Accuracy (@0.5)** | **86.31%** (725 / 840) | — |
| **Balanced Accuracy (@0.5)** | **89.50%** | — |
| **Precision / Recall / F1** | **98.7%** / **84.7%** / **0.9116** | — |
| **Biometric TPR @ 1% FPR** | **78.29%** | — |
| **Biometric TPR @ 5% FPR** | **84.14%** | — |
| **Raw Calibration ECE** | **0.1265** | — |

*Per-Method AUC Breakdown on FF++ Test:*
- **Deepfakes:** `0.991` (Accuracy@0.5: 97.9%)
- **FaceSwap:** `0.988` (Accuracy@0.5: 95.7%)
- **Face2Face:** `0.981` (Accuracy@0.5: 92.1%)
- **FaceShifter:** `0.965` (Accuracy@0.5: 85.0%)
- **NeuralTextures:** `0.868` (Accuracy@0.5: 52.9%)
- **Pristine Reals Cleared:** `94.3%` (132 / 140, False Alarms: 5.7%)

---

#### 2. Cross-Dataset Generalization: Celeb-DF-v2 (Test Split, N=518 videos)
- **Score CSV:** [`results/scores/smsdt_s_seed1_celebdf.csv`](file:///e:/DEEPFAKE/results/scores/smsdt_s_seed1_celebdf.csv)

| Metric | Score | 95% Confidence Interval |
| :--- | :---: | :---: |
| **Video AUC** | **0.8837** | [0.8547 - 0.9109] |
| **Top-50% Clip AUC** | **0.8801** | — |
| **Frame/Clip AUC** | **0.8546** | — |
| **Overall Accuracy (@0.5)** | **78.57%** (407 / 518) | — |
| **Balanced Accuracy (@0.5)** | **81.80%** | — |
| **Precision / Recall / F1** | **94.6%** / **71.5%** / **0.8141** | — |
| **Fakes Caught / Reals Cleared** | **243 / 340 (71.5%)** | **164 / 178 (92.1%)** |
| **Biometric TPR @ 1% FPR** | **28.82%** | — |
| **Biometric TPR @ 5% FPR** | **50.29%** | — |
| **Calibration ECE** | **0.1555** | — |

---

#### 3. Cross-Dataset Generalization: DFD (DeepFakeDetection Test Split, N=2,987 videos)
- **Score CSV:** [`results/scores/smsdt_s_seed1_dfd.csv`](file:///e:/DEEPFAKE/results/scores/smsdt_s_seed1_dfd.csv)

| Metric | Score | 95% Confidence Interval |
| :--- | :---: | :---: |
| **Video AUC** | **0.8742** | [0.8593 - 0.8880] |
| **Top-50% Clip AUC** | **0.8762** | — |
| **Frame/Clip AUC** | **0.8528** | — |
| **Overall Accuracy (@0.5)** | **75.76%** (2,263 / 2,987) | — |
| **Balanced Accuracy (@0.5)** | **81.20%** | — |
| **Precision / Recall / F1** | **97.8%** / **73.9%** / **0.8416** | — |
| **Fakes Caught / Reals Cleared** | **1,924 / 2,604 (73.9%)** | **339 / 383 (88.5%)** |
| **Biometric TPR @ 1% FPR** | **53.25%** | — |
| **Biometric TPR @ 5% FPR** | **67.47%** | — |
| **Calibration ECE** | **0.2288** | — |

---

### 3.4 S-MSDTv3 Full Model Evaluation Results (Seed 1)

The full model **S-MSDTv3 Full** (`smsdt_full`) completed 30 epochs of training on the RTX 5090 (`train_full.yaml`, Seed 1).
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

*Observation [corrected 2026-11-10]:* single run. S-MSDTv3 Full scores 0.8915 AUC on Celeb-DF v2 (95% CI 0.862-0.918). The single-frame model `smsdt_s` scores 0.8837 (CI 0.855-0.911) in the same pipeline; the intervals overlap. Note also that 40 of 178 Celeb-DF reals (22.5%, Wilson CI 17.0-29.1%) are flagged at 0.5, against 9 of 140 FF++ reals (6.4%): see H4 in `docs/decisions.md`.

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

### 3.6 Multi-Seed Replication Log: Seed 2 Results

All four architectures completed 30 epochs of training on the RTX 5090 for **Seed 2** and were evaluated under the standardized cross-dataset protocol:

#### Summary of Seed 2 Evaluations:
1. **`gend_ours_seed2` (GenD-style in-house baseline, $T=1$):**
   - **FF++ Test Video AUC:** `0.9040` (Macro: 0.9040, Acc@0.5: 78.45%, ECE: 0.2693)
   - **Celeb-DF v2 Video AUC:** `0.8273` (Acc@0.5: 70.08%, TPR@5%FPR: 51.18%, ECE: 0.1858)
   - **DFD Video AUC:** `0.8820` (Acc@0.5: 73.95%, TPR@5%FPR: 69.01%, ECE: 0.3104)
   - *Scores:* `results/scores/gend_ours_seed2_{celebdf,dfd,ffpp}.csv`

2. **`smsdt_s_seed2` (Spatial-Only Baseline, $T=1$):**
   - **FF++ Test Video AUC:** `0.9613` (Macro: 0.9613, Acc@0.5: 86.07%, ECE: 0.1322)
   - **Celeb-DF v2 Video AUC:** `0.8825` (Acc@0.5: 74.90%, TPR@5%FPR: 50.59%, ECE: 0.1579)
   - **DFD Video AUC:** `0.8807` (Acc@0.5: 77.47%, TPR@5%FPR: 70.55%, ECE: 0.2316)
   - *Scores:* `results/scores/smsdt_s_seed2_{celebdf,dfd,ffpp}.csv`

3. **`smsdt_st_seed2` (Spatio-Temporal Baseline, $T=8$):**
   - **FF++ Test Video AUC:** `0.9665` (Macro: 0.9665, Acc@0.5: 89.05%, ECE: 0.1080)
   - **Celeb-DF v2 Video AUC:** `0.8779` (Acc@0.5: 73.55%, TPR@5%FPR: 42.32%, ECE: 0.1468)
   - **DFD Video AUC:** `0.8748` (Acc@0.5: 75.66%, TPR@5%FPR: 59.96%, ECE: 0.2132)
   - *Scores:* `results/scores/smsdt_st_seed2_{celebdf,dfd,ffpp}.csv`

4. **`smsdt_full_seed2` (S-MSDTv3 Full, $T=8$):**
   - **FF++ Test Video AUC:** `0.9621` (Macro: 0.9621, Acc@0.5: 89.17%, ECE: 0.0947)
   - **Celeb-DF v2 Video AUC:** **`0.8960`** (Acc@0.5: 81.66%, TPR@5%FPR: 55.00%, ECE: 0.0864)
   - **DFD Video AUC:** `0.8689` (Acc@0.5: 76.03%, TPR@5%FPR: 58.76%, ECE: 0.2154)
   - *Scores:* `results/scores/smsdt_full_seed2_{celebdf,dfd,ffpp}.csv`

---

### 3.7 Multi-Seed Replication Log: Seed 3 Results

All four architectures completed 30 epochs of training on the RTX 5090 for **Seed 3** and were evaluated under the standardized cross-dataset protocol:

#### Summary of Seed 3 Evaluations:
1. **`gend_ours_seed3` (GenD-style in-house baseline, $T=1$):**
   - **FF++ Test Video AUC:** `0.9045` (95% CI: [0.8821 - 0.9237], Acc@0.5: 77.86%, TPR@5%FPR: 73.29%, ECE: 0.2687)
     - *Per-Method AUC on FF++:* Deepfakes: 0.994, FaceSwap: 0.986, FaceShifter: 0.949, Face2Face: 0.918, NeuralTextures: 0.675
   - **Celeb-DF v2 Video AUC:** `0.8288` (95% CI: [0.7927 - 0.8591], Acc@0.5: 69.88%, TPR@5%FPR: 51.47%, ECE: 0.1990)
   - **DFD Video AUC:** `0.8758` (95% CI: [0.8614 - 0.8892], Acc@0.5: 73.42%, TPR@5%FPR: 69.51%, ECE: 0.3126)
   - *Scores:* `results/scores/gend_ours_seed3_{celebdf,dfd,ffpp}.csv`

2. **`smsdt_s_seed3` (Spatial-Only Baseline, $T=1$):**
   - **FF++ Test Video AUC:** `0.9681` (95% CI: [0.9571 - 0.9784], Acc@0.5: 87.86%, TPR@5%FPR: 86.71%, ECE: 0.1299)
     - *Per-Method AUC on FF++:* Deepfakes: 0.996, FaceSwap: 0.990, Face2Face: 0.986, FaceShifter: 0.971, NeuralTextures: 0.897
   - **Celeb-DF v2 Video AUC:** `0.8815` (95% CI: [0.8521 - 0.9086], Acc@0.5: 77.99%, TPR@5%FPR: 51.74%, ECE: 0.1590)
   - **DFD Video AUC:** `0.8974` (95% CI: [0.8830 - 0.9097], Acc@0.5: 76.83%, TPR@5%FPR: 68.74%, ECE: 0.2242)
   - *Scores:* `results/scores/smsdt_s_seed3_{celebdf,dfd,ffpp}.csv`

3. **`smsdt_st_seed3` (Spatio-Temporal Baseline, $T=8$):**
   - **FF++ Test Video AUC:** `0.9618` (95% CI: [0.9492 - 0.9732], Acc@0.5: 88.57%, TPR@5%FPR: 86.29%, ECE: 0.1095)
     - *Per-Method AUC on FF++:* Deepfakes: 0.996, Face2Face: 0.984, FaceSwap: 0.976, FaceShifter: 0.964, NeuralTextures: 0.889
   - **Celeb-DF v2 Video AUC:** `0.8628` (95% CI: [0.8349 - 0.8908], Acc@0.5: 77.61%, TPR@5%FPR: 54.41%, ECE: 0.1273)
   - **DFD Video AUC:** `0.8908` (95% CI: [0.8764 - 0.9047], Acc@0.5: 80.65%, TPR@5%FPR: 68.06%, ECE: 0.1650)
   - *Scores:* `results/scores/smsdt_st_seed3_{celebdf,dfd,ffpp}.csv`

4. **`smsdt_full_seed3` (S-MSDTv3 Full, $T=8$):**
   - **FF++ Test Video AUC:** `0.9625` (95% CI: [0.9502 - 0.9741], Acc@0.5: 89.88%, TPR@5%FPR: 87.86%, ECE: 0.0917)
     - *Per-Method AUC on FF++:* Deepfakes: 0.996, FaceSwap: 0.979, Face2Face: 0.978, FaceShifter: 0.967, NeuralTextures: 0.892
   - **Celeb-DF v2 Video AUC:** `0.8772` (95% CI: [0.8475 - 0.9046], Acc@0.5: 80.12%, TPR@5%FPR: 50.88%, ECE: 0.0957)
   - **DFD Video AUC:** `0.8655` (95% CI: [0.8483 - 0.8813], Acc@0.5: 78.00%, TPR@5%FPR: 55.61%, ECE: 0.1883)
   - *Scores:* `results/scores/smsdt_full_seed3_{celebdf,dfd,ffpp}.csv`

---

## 4. Master Experiment Matrix & Comparative Analysis (RQ1)

Compiled via `tools/make_tables.py` into [`results/tables/experiment_matrix.csv`](file:///e:/DEEPFAKE/results/tables/experiment_matrix.csv) and [`results/tables/experiment_matrix.md`](file:///e:/DEEPFAKE/results/tables/experiment_matrix.md):

### 4.1 Master Experiment Matrix (All 13 Runs, regenerated 2026-11-12)

Generated by `python -m tools.make_tables` from `results/scores/` (D14 pure reals: DFD = `243` actor reals; D16 canonical TPR at the conservative empirical 5%-FPR point). Supersedes the hand-assembled matrix, whose TPR columns mixed two definitions and whose DFD columns used `383` mixed reals.

| Model | Seed | FF++ Test AUC | Celeb-DF AUC | Celeb-DF TPR@5% | DFD AUC (pure) | DFD TPR@5% (pure) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `gend_official` | 0 | 0.9452 | 0.9327 | 73.82% | 0.8840 | 64.98% |
| `gend_ours` | 1 | 0.9047 | 0.8321 | 52.06% | 0.8657 | 69.93% |
| `gend_ours` | 2 | 0.9040 | 0.8273 | 51.18% | 0.8688 | 70.97% |
| `gend_ours` | 3 | 0.9045 | 0.8288 | 51.47% | 0.8623 | 70.16% |
| `smsdt_s` | 1 | 0.9587 | 0.8837 | 50.29% | 0.8667 | 65.09% |
| `smsdt_s` | 2 | 0.9613 | 0.8825 | 50.59% | 0.8775 | 69.16% |
| `smsdt_s` | 3 | 0.9681 | 0.8815 | 51.47% | 0.8876 | 68.36% |
| `smsdt_st` | 1 | 0.9604 | 0.8706 | 45.29% | 0.8607 | 58.76% |
| `smsdt_st` | 2 | 0.9664 | 0.8779 | 42.06% | 0.8559 | 57.60% |
| `smsdt_st` | 3 | 0.9618 | 0.8628 | 54.41% | 0.8701 | 65.55% |
| `smsdt_full` | 1 | 0.9631 | 0.8914 | 51.76% | 0.8321 | 46.51% |
| `smsdt_full` | 2 | 0.9621 | 0.8960 | 55.00% | 0.8577 | 56.34% |
| `smsdt_full` | 3 | 0.9625 | 0.8772 | 50.88% | 0.8445 | 52.04% |

Mean +/- sd per architecture: [`results/tables/experiment_matrix_agg.md`](file:///e:/DEEPFAKE/results/tables/experiment_matrix_agg.md).

---

### 4.2 Canonical 3-Seed Aggregated Performance (Mean $\pm$ Std Dev)

| Model Architecture | Seeds / Runs | In-Domain FF++ Test AUC | Cross-Dataset Celeb-DF v2 AUC | Cross-Dataset DFD AUC (pure, `243` reals) | Role / Summary |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`gend_official`** | 1 (`seed0`) | `0.9452` | **`0.9327`** | **`0.8840`** | Official author weights (`hf:yermandy/GenD_CLIP_L_14`) |
| **`gend_ours`** | 3 | $0.9044 \pm 0.0004$ | $0.8294 \pm 0.0025$ | $0.8656 \pm 0.0033$ | GenD-style in-house baseline (not a reproduction; Celeb-DF 0.83 vs 0.933 for official weights) |
| **`smsdt_s`** | 3 | $0.9627 \pm 0.0049$ | $0.8826 \pm 0.0011$ | $0.8773 \pm 0.0105$ | Controlled single-frame baseline ($T=1$) |
| **`smsdt_st`** | 3 | $0.9629 \pm 0.0031$ | $0.8704 \pm 0.0076$ | $0.8622 \pm 0.0072$ | Spatio-temporal attention ($T=8$) |
| **`smsdt_full`** | 3 | $0.9626 \pm 0.0005$ | $0.8882 \pm 0.0098$ | $0.8447 \pm 0.0128$ | Full S-MSDTv3 ($T=8$) |

#### Statistical Takeaways (3-Seed Summary):
1. **In-House Baseline vs. Published GenD (Celeb-DF v2):**
   - `smsdt_full` achieves a 3-seed mean of **$0.8882 \pm 0.0098$** (`+0.059` AUC over our in-house GenD-style baseline at $0.8294 \pm 0.0025$). **This is not a comparison with GenD, and not a recipe-only comparison either (D17):** `smsdt_s` trains 777,478 parameters (projection, attention-MIL readout and CLS head on top of the LayerNorms) against `gend_ours`'s 103,425, so the gap mixes training recipe and head architecture. Official GenD (`gend_official`) scores **`0.9327`** on Celeb-DF, above every S-MSDT variant (`smsdt_s`: `0.8826`, `smsdt_st`: `0.8704`, `smsdt_full`: `0.8882`).
2. **Run-to-Run Seed Spread:**
   - Across the 3 training seeds, seed $\text{sd} \le 0.0098$ on Celeb-DF (and $\le 0.0049$ on FF++). Three seeds characterize empirical variance under our pipeline rather than proving architectural causality, and no ablation isolates loss weighting or curriculum sampling.
   - In-domain on FF++ c23, `smsdt_s`, `smsdt_st`, and `smsdt_full` all cluster around `0.9626 – 0.9629`.

*(DFD columns in §§4.1–4.2 now use the D14 pure-DFD reals (`243` reals / `18` actor identities, `2,847` videos); DFD stays `dev_overlap` exploratory.)*

---

### 4.3 Hypothesis H1: Canonical 3-Seed Paired Cluster Bootstrap ΔAUC & Equivalence Tests

Compiled via [`tools/stats.py`](file:///e:/DEEPFAKE/tools/stats.py) into [`results/tables/hypothesis_h1_stats.csv`](file:///e:/DEEPFAKE/results/tables/hypothesis_h1_stats.csv), [`results/tables/hypothesis_h1_stats.md`](file:///e:/DEEPFAKE/results/tables/hypothesis_h1_stats.md), and [`results/tables/hypothesis_h1_meta.json`](file:///e:/DEEPFAKE/results/tables/hypothesis_h1_meta.json) ($N_{\text{boot}}=10{,}000$, equivalence margin $\delta = 0.01$, Bonferroni family $m=2 \rightarrow 97.5\%$ CI on confirmatory pairs, D14 pure-source negatives):

- **Pre-Registered H1 Verdict:** **`undetermined (underpowered)`**

| Dataset | Role | Kind | Model A | Model B | Comparison | AUC (A) | AUC (B) | $\Delta\text{AUC}$ (A-B) | CI Level | CI Lower | CI Upper | Classification ($\delta=0.01$) | $N$ Mult | Clusters / Samples |
| :--- | :--- | :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CELEBDF** | `confirmatory` | **`confirmatory`** | `smsdt_st` | `smsdt_s` | **H1: ST vs S** | 0.8704 | 0.8826 | **-0.0122** | `0.9750` | -0.0312 | 0.0068 | `inconclusive` | `inf` | 124 / 518 |
| **DFD** *(pure)* | `dev_overlap` | `exploratory` | `smsdt_st` | `smsdt_s` | H1: ST vs S | 0.8622 | 0.8773 | -0.0151 | `0.9500` | -0.0254 | -0.0047 | `inconclusive` | `inf` | 273 / 2847 |
| **FFPP** | `in_domain` | `exploratory` | `smsdt_st` | `smsdt_s` | H1: ST vs S | 0.9629 | 0.9627 | 0.0002 | `0.9500` | -0.0066 | 0.0072 | `equivalent` | `0.4957` | 140 / 840 |
| **FFPPVAL** | `unknown` | `exploratory` | `smsdt_st` | `smsdt_s` | H1: ST vs S | 0.9482 | 0.9377 | 0.0105 | `0.9500` | 0.0035 | 0.0175 | `inconclusive` | `inf` | 178 / 1304 |
| **CELEBDF** | `confirmatory` | **`confirmatory`** | `smsdt_full` | `smsdt_s` | **H1: Full vs S** | 0.8882 | 0.8826 | **+0.0056** | `0.9750` | -0.0142 | 0.0262 | `inconclusive` | `21.0170` | 124 / 518 |
| **DFD** *(pure)* | `dev_overlap` | `exploratory` | `smsdt_full` | `smsdt_s` | H1: Full vs S | 0.8447 | 0.8773 | -0.0325 | `0.9500` | -0.0450 | -0.0207 | **`worse`** | `inf` | 273 / 2847 |
| **FFPP** | `in_domain` | `exploratory` | `smsdt_full` | `smsdt_s` | H1: Full vs S | 0.9625 | 0.9627 | -0.0001 | `0.9500` | -0.0078 | 0.0073 | `equivalent` | `0.5892` | 140 / 840 |
| **FFPPVAL** | `unknown` | `exploratory` | `smsdt_full` | `smsdt_s` | H1: Full vs S | 0.9410 | 0.9377 | 0.0034 | `0.9500` | -0.0041 | 0.0107 | `inconclusive` | `1.2449` | 178 / 1304 |
| **CELEBDF** | `confirmatory` | `exploratory` | `smsdt_full` | `smsdt_st` | Exploratory: Full vs ST | 0.8882 | 0.8704 | +0.0178 | `0.9500` | 0.0050 | 0.0318 | `inconclusive` | `inf` | 124 / 518 |
| **DFD** *(pure)* | `dev_overlap` | `exploratory` | `smsdt_full` | `smsdt_st` | Exploratory: Full vs ST | 0.8447 | 0.8622 | -0.0175 | `0.9500` | -0.0284 | -0.0068 | `inconclusive` | `inf` | 273 / 2847 |
| **FFPP** | `in_domain` | `exploratory` | `smsdt_full` | `smsdt_st` | Exploratory: Full vs ST | 0.9625 | 0.9629 | -0.0004 | `0.9500` | -0.0059 | 0.0054 | `equivalent` | `0.3455` | 140 / 840 |
| **FFPPVAL** | `unknown` | `exploratory` | `smsdt_full` | `smsdt_st` | Exploratory: Full vs ST | 0.9410 | 0.9482 | -0.0071 | `0.9500` | -0.0140 | -0.0002 | `inconclusive` | `5.6907` | 178 / 1304 |
| **CELEBDF** | `confirmatory` | `exploratory` | `smsdt_s` | `gend_ours` | Exploratory: S vs `gend_ours` | 0.8826 | 0.8294 | +0.0532 | `0.9500` | 0.0210 | 0.0879 | `better` | `inf` | 124 / 518 |
| **DFD** *(pure)* | `dev_overlap` | `exploratory` | `smsdt_s` | `gend_ours` | Exploratory: S vs `gend_ours` | 0.8773 | 0.8656 | +0.0117 | `0.9500` | -0.0122 | 0.0372 | `inconclusive` | `inf` | 273 / 2847 |
| **FFPP** | `in_domain` | `exploratory` | `smsdt_s` | `gend_ours` | Exploratory: S vs `gend_ours` | 0.9627 | 0.9044 | +0.0583 | `0.9500` | 0.0453 | 0.0714 | `better` | `inf` | 140 / 840 |
| **FFPPVAL** | `unknown` | `exploratory` | `smsdt_s` | `gend_ours` | Exploratory: S vs `gend_ours` | 0.9377 | 0.9007 | +0.0370 | `0.9500` | 0.0198 | 0.0536 | `better` | `inf` | 178 / 1304 |
| **CELEBDF** | `confirmatory` | `exploratory` | `smsdt_full` | `gend_ours` | Exploratory: Full vs `gend_ours` | 0.8882 | 0.8294 | +0.0588 | `0.9500` | 0.0239 | 0.0957 | `better` | `inf` | 124 / 518 |
| **DFD** *(pure)* | `dev_overlap` | `exploratory` | `smsdt_full` | `gend_ours` | Exploratory: Full vs `gend_ours` | 0.8447 | 0.8656 | -0.0208 | `0.9500` | -0.0498 | 0.0092 | `inconclusive` | `inf` | 273 / 2847 |
| **FFPP** | `in_domain` | `exploratory` | `smsdt_full` | `gend_ours` | Exploratory: Full vs `gend_ours` | 0.9625 | 0.9044 | +0.0582 | `0.9500` | 0.0431 | 0.0733 | `better` | `inf` | 140 / 840 |
| **FFPPVAL** | `unknown` | `exploratory` | `smsdt_full` | `gend_ours` | Exploratory: Full vs `gend_ours` | 0.9410 | 0.9007 | +0.0404 | `0.9500` | 0.0220 | 0.0585 | `better` | `inf` | 178 / 1304 |
| **CELEBDF** | `confirmatory` | `exploratory` | `smsdt_s` | `gend_official` | **G1: S vs `gend_official`** | 0.8826 | 0.9327 | **-0.0501** | `0.9500` | -0.0852 | -0.0210 | **`worse`** | `inf` | 124 / 518 |
| **DFD** *(pure)* | `dev_overlap` | `exploratory` | `smsdt_s` | `gend_official` | G1: S vs `gend_official` | 0.8773 | 0.8840 | -0.0067 | `0.9500` | -0.0266 | 0.0142 | `inconclusive` | `38.5854` | 273 / 2847 |
| **FFPP** | `in_domain` | `exploratory` | `smsdt_s` | `gend_official` | G1: S vs `gend_official` | 0.9627 | 0.9452 | +0.0175 | `0.9500` | 0.0060 | 0.0293 | `inconclusive` | `inf` | 140 / 840 |
| **FFPPVAL** | `unknown` | `exploratory` | `smsdt_s` | `gend_official` | G1: S vs `gend_official` | 0.9545 | 0.9371 | +0.0174 | `0.9500` | 0.0073 | 0.0276 | `inconclusive` | `inf` | 140 / 840 (`464` dropped) |
| **CELEBDF** | `confirmatory` | `exploratory` | `gend_ours` | `gend_official` | **G1: `gend_ours` vs `gend_official`** | 0.8294 | 0.9327 | **-0.1033** | `0.9500` | -0.1393 | -0.0723 | **`worse`** | `inf` | 124 / 518 |
| **DFD** *(pure)* | `dev_overlap` | `exploratory` | `gend_ours` | `gend_official` | G1: `gend_ours` vs `gend_official` | 0.8656 | 0.8840 | -0.0184 | `0.9500` | -0.0324 | -0.0060 | `inconclusive` | `inf` | 273 / 2847 |
| **FFPP** | `in_domain` | `exploratory` | `gend_ours` | `gend_official` | G1: `gend_ours` vs `gend_official` | 0.9044 | 0.9452 | -0.0408 | `0.9500` | -0.0568 | -0.0247 | `worse` | `inf` | 140 / 840 |
| **FFPPVAL** | `unknown` | `exploratory` | `gend_ours` | `gend_official` | G1: `gend_ours` vs `gend_official` | 0.8988 | 0.9371 | -0.0383 | `0.9500` | -0.0527 | -0.0232 | `worse` | `inf` | 140 / 840 (`464` dropped) |

#### Scientific Reading of H1:
1. **No Established AUC Gain From Extra ST/Full Modules:** On confirmatory Celeb-DF v2, `smsdt_full vs smsdt_s` gives $\Delta\text{AUC} = +0.0056$ ($97.5\%\text{ CI }[-0.0142, +0.0262]$) and `smsdt_st vs smsdt_s` gives $\Delta\text{AUC} = -0.0122$ ($97.5\%\text{ CI }[-0.0312, +0.0068]$). Both cross zero and span outside $[-0.01, +0.01]$, so under the pre-registered rule H1 is **`undetermined (underpowered)`** (and resolving equivalence for Full vs S would require $\approx 21\times$ the cluster count). On pure exploratory DFD (`2,847` videos, `243` reals), `smsdt_full` actually scores lower than `smsdt_s` ($\Delta\text{AUC} = -0.0325$, $95\%\text{ CI }[-0.0450, -0.0207]$, **`worse`**). Overall, the added ST and Full modules deliver **no demonstrable cross-dataset AUC gain** over single-frame `smsdt_s`.
2. **Gate G1 External Standing (`gend_official`):** Against official author weights (`0.9327` on Celeb-DF), both `smsdt_s` ($\Delta\text{AUC} = -0.0501$, $95\%\text{ CI }[-0.0852, -0.0210]$) and `gend_ours` ($\Delta\text{AUC} = -0.1033$, $95\%\text{ CI }[-0.1393, -0.0723]$) classify as **`worse`**. Our in-house recipe improves over `gend_ours`, not over published GenD.
3. **In-Domain Saturation:** On FF++ test, `ST vs S` ($+0.0002$), `Full vs S` ($-0.0001$), and `Full vs ST` ($-0.0004$) all fall inside $[-\delta, +\delta]$ and classify as `equivalent`.

---

## 5. Hypothesis H2: Operating Point Transfer & Threshold Degradation Audit

Compiled via [`tools/threshold_transfer.py`](file:///e:/DEEPFAKE/tools/threshold_transfer.py) into [`results/tables/threshold_transfer_audit.csv`](file:///e:/DEEPFAKE/results/tables/threshold_transfer_audit.csv), [`results/tables/threshold_transfer_audit.md`](file:///e:/DEEPFAKE/results/tables/threshold_transfer_audit.md), and [`results/tables/threshold_transfer_details.json`](file:///e:/DEEPFAKE/results/tables/threshold_transfer_details.json) using dev-split Platt calibration parameters (`results/calib/*.json`), **D13** resolved Celeb-DF identity clusters (`121` real identities instead of the 3-valued `cluster_id`), and **D14** pure DFD negatives (`243` reals / `18` actors).

Evaluates the degradation gap $g = \text{HTER}(\tau_{\text{cal}}) - \text{HTER}(\tau^*) \ge 0$ (`tau_cal` mapped through each run's FF++ val Platt scale & Youden cutoff), where **H2** tests whether a validation-frozen threshold loses $\ge 0.05$ HTER under unseen distribution shift ($g_{\text{CI,low}} \ge 0.05$ in confirmatory seeds):

- **Pre-Registered H2 Verdict:** **`undetermined`** (`{'smsdt_s': 'undetermined', 'gend_ours': 'undetermined'}`)

| Model | Seed | Dataset | Role | $\tau_{\text{cal}}$ | HTER@0.50 (raw) | $\text{HTER}(\tau_{\text{cal}})$ | $\text{HTER}(\tau^*)$ | Gap $g$ | $g$ CI Low | $g$ CI High | $\text{FPR}(\tau_{\text{cal}})$ | $\text{FNR}(\tau_{\text{cal}})$ | FPR Infl vs FF++val |
| :--- | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `gend_official` | `seed0` | `FFPP` | `in_domain` | 0.6660 | 0.1364 | 0.1114 | 0.0943 | 0.0171 | 0.0071 | 0.0300 | 0.0357 | 0.1871 | 1.0000x |
| `gend_official` | `seed0` | `FFPPVAL` | `unknown` | 0.6660 | 0.1586 | 0.1221 | 0.1214 | 0.0007 | 0.0000 | 0.0150 | 0.0643 | 0.1800 | 1.0000x |
| `gend_ours` | `seed1` | `FFPP` | `in_domain` | 0.2613 | 0.1650 | 0.2007 | 0.1557 | 0.0450 | 0.0186 | 0.0857 | 0.2643 | 0.1371 | 1.0000x |
| `gend_ours` | `seed1` | `FFPPVAL` | `unknown` | 0.2613 | 0.1923 | 0.1798 | 0.1798 | 0.0000 | 0.0000 | 0.0251 | 0.2429 | 0.1168 | 1.0000x |
| `gend_ours` | `seed2` | `FFPP` | `in_domain` | 0.2771 | 0.1693 | 0.1900 | 0.1586 | 0.0314 | 0.0107 | 0.0700 | 0.2357 | 0.1443 | 1.0000x |
| `gend_ours` | `seed2` | `FFPPVAL` | `unknown` | 0.2771 | 0.1931 | 0.1881 | 0.1877 | 0.0004 | 0.0000 | 0.0332 | 0.2500 | 0.1263 | 1.0000x |
| `gend_ours` | `seed3` | `FFPP` | `in_domain` | 0.3993 | 0.1614 | 0.1743 | 0.1564 | 0.0179 | 0.0064 | 0.0479 | 0.1500 | 0.1986 | 1.0000x |
| `gend_ours` | `seed3` | `FFPPVAL` | `unknown` | 0.3993 | 0.1921 | 0.1842 | 0.1842 | 0.0000 | 0.0000 | 0.0242 | 0.1786 | 0.1899 | 1.0000x |
| `smsdt_full` | `seed1` | `FFPP` | `in_domain` | 0.6282 | 0.0871 | 0.1164 | 0.0793 | 0.0371 | 0.0243 | 0.0579 | 0.0286 | 0.2043 | 1.0000x |
| `smsdt_full` | `seed1` | `FFPPVAL` | `unknown` | 0.6282 | 0.1496 | 0.1181 | 0.1181 | 0.0000 | 0.0000 | 0.0119 | 0.0429 | 0.1933 | 1.0000x |
| `smsdt_full` | `seed2` | `FFPP` | `in_domain` | 0.4452 | 0.1050 | 0.1000 | 0.0943 | 0.0057 | 0.0000 | 0.0300 | 0.0857 | 0.1143 | 1.0000x |
| `smsdt_full` | `seed2` | `FFPPVAL` | `unknown` | 0.4452 | 0.1249 | 0.1207 | 0.1203 | 0.0004 | 0.0000 | 0.0189 | 0.1143 | 0.1271 | 1.0000x |
| `smsdt_full` | `seed3` | `FFPP` | `in_domain` | 0.6637 | 0.0864 | 0.1143 | 0.0814 | 0.0329 | 0.0179 | 0.0507 | 0.0214 | 0.2071 | 1.0000x |
| `smsdt_full` | `seed3` | `FFPPVAL` | `unknown` | 0.6637 | 0.1420 | 0.1185 | 0.1185 | 0.0000 | 0.0000 | 0.0162 | 0.0214 | 0.2156 | 1.0000x |
| `smsdt_s` | `seed1` | `FFPP` | `in_domain` | 0.6937 | 0.1050 | 0.1214 | 0.1029 | 0.0186 | 0.0086 | 0.0450 | 0.0071 | 0.2357 | 1.0000x |
| `smsdt_s` | `seed1` | `FFPPVAL` | `unknown` | 0.6937 | 0.1501 | 0.1252 | 0.1244 | 0.0009 | 0.0000 | 0.0155 | 0.0357 | 0.2148 | 1.0000x |
| `smsdt_s` | `seed2` | `FFPP` | `in_domain` | 0.5636 | 0.0979 | 0.1043 | 0.0907 | 0.0136 | 0.0036 | 0.0364 | 0.0214 | 0.1871 | 1.0000x |
| `smsdt_s` | `seed2` | `FFPPVAL` | `unknown` | 0.5636 | 0.1223 | 0.1115 | 0.1110 | 0.0004 | 0.0000 | 0.0133 | 0.0571 | 0.1658 | 1.0000x |
| `smsdt_s` | `seed3` | `FFPP` | `in_domain` | 0.5632 | 0.0900 | 0.1014 | 0.0786 | 0.0229 | 0.0121 | 0.0414 | 0.0357 | 0.1671 | 1.0000x |
| `smsdt_s` | `seed3` | `FFPPVAL` | `unknown` | 0.5632 | 0.1150 | 0.1020 | 0.1020 | 0.0000 | 0.0000 | 0.0143 | 0.0571 | 0.1469 | 1.0000x |
| `smsdt_st` | `seed1` | `FFPP` | `in_domain` | 0.5799 | 0.0979 | 0.1021 | 0.0857 | 0.0164 | 0.0043 | 0.0343 | 0.0571 | 0.1471 | 1.0000x |
| `smsdt_st` | `seed1` | `FFPPVAL` | `unknown` | 0.5799 | 0.1313 | 0.1142 | 0.1142 | 0.0000 | 0.0000 | 0.0181 | 0.0857 | 0.1426 | 1.0000x |
| `smsdt_st` | `seed2` | `FFPP` | `in_domain` | 0.7924 | 0.0914 | 0.1200 | 0.0821 | 0.0379 | 0.0236 | 0.0593 | 0.0143 | 0.2257 | 1.0000x |
| `smsdt_st` | `seed2` | `FFPPVAL` | `unknown` | 0.7924 | 0.1258 | 0.1148 | 0.1144 | 0.0004 | 0.0000 | 0.0216 | 0.0286 | 0.2010 | 1.0000x |
| `smsdt_st` | `seed3` | `FFPP` | `in_domain` | 0.5004 | 0.0914 | 0.0993 | 0.0914 | 0.0079 | 0.0000 | 0.0236 | 0.0429 | 0.1557 | 1.0000x |
| `smsdt_st` | `seed3` | `FFPPVAL` | `unknown` | 0.5004 | 0.1228 | 0.1100 | 0.1096 | 0.0004 | 0.0000 | 0.0170 | 0.0929 | 0.1271 | 1.0000x |
| **`gend_official`** | `seed0` | `CELEBDF` | `confirmatory` | 0.6660 | 0.2826 | 0.1641 | 0.1434 | **0.0207** | 0.0086 | 0.0552 | 0.2135 | 0.1147 | 3.3208x |
| **`gend_official`** | `seed0` | `DFD` *(pure)* | `dev_overlap` | 0.6660 | 0.2760 | 0.1991 | 0.1751 | **0.0240** | 0.0099 | 0.0445 | 0.2058 | 0.1924 | 3.2007x |
| `gend_ours` | `seed1` | `CELEBDF` | `confirmatory` | 0.2613 | 0.2535 | 0.2628 | 0.2453 | 0.0174 | 0.0018 | 0.0652 | 0.3315 | 0.1941 | 1.3648x |
| `gend_ours` | `seed1` | `DFD` *(pure)* | `dev_overlap` | 0.2613 | 0.1808 | 0.2831 | 0.1719 | **0.1112** | **0.0793** | 0.1511 | 0.4156 | 0.1505 | 1.7115x |
| `gend_ours` | `seed2` | `CELEBDF` | `confirmatory` | 0.2771 | 0.2534 | 0.2663 | 0.2404 | 0.0258 | 0.0061 | 0.0647 | 0.3090 | 0.2235 | 1.2360x |
| `gend_ours` | `seed2` | `DFD` *(pure)* | `dev_overlap` | 0.2771 | 0.1728 | 0.2742 | 0.1682 | **0.1060** | **0.0756** | 0.1456 | 0.3909 | 0.1575 | 1.5638x |
| `gend_ours` | `seed3` | `CELEBDF` | `confirmatory` | 0.3993 | 0.2508 | 0.2478 | 0.2404 | 0.0074 | 0.0036 | 0.0367 | 0.1573 | 0.3382 | 0.8809x |
| `gend_ours` | `seed3` | `DFD` *(pure)* | `dev_overlap` | 0.3993 | 0.1748 | 0.1998 | 0.1721 | 0.0277 | 0.0085 | 0.0544 | 0.1646 | 0.2350 | 0.9218x |
| `smsdt_full` | `seed1` | `CELEBDF` | `confirmatory` | 0.6282 | 0.1962 | 0.2134 | 0.1831 | 0.0303 | 0.0144 | 0.0608 | 0.1180 | 0.3088 | 2.7528x |
| `smsdt_full` | `seed1` | `DFD` *(pure)* | `dev_overlap` | 0.6282 | 0.2608 | 0.2492 | 0.2420 | 0.0071 | 0.0004 | 0.0242 | 0.1646 | 0.3337 | 3.8409x |
| `smsdt_full` | `seed2` | `CELEBDF` | `confirmatory` | 0.4452 | 0.1878 | 0.1895 | 0.1881 | 0.0014 | 0.0000 | 0.0309 | 0.1966 | 0.1824 | 1.7205x |
| `smsdt_full` | `seed2` | `DFD` *(pure)* | `dev_overlap` | 0.4452 | 0.2262 | 0.2276 | 0.2167 | 0.0109 | 0.0047 | 0.0347 | 0.1975 | 0.2577 | 1.7284x |
| `smsdt_full` | `seed3` | `CELEBDF` | `confirmatory` | 0.6637 | 0.2077 | 0.2323 | 0.2065 | 0.0258 | 0.0115 | 0.0635 | 0.1292 | 0.3353 | **6.0300x** |
| `smsdt_full` | `seed3` | `DFD` *(pure)* | `dev_overlap` | 0.6637 | 0.2345 | 0.2344 | 0.2222 | 0.0122 | 0.0059 | 0.0272 | 0.1358 | 0.3329 | **6.3374x** |
| **`smsdt_s`** | **`seed1`** | **`CELEBDF`** | `confirmatory` | 0.6937 | 0.1820 | 0.2753 | 0.1769 | **0.0984** | **0.0796** | 0.1283 | 0.0506 | 0.5000 | 1.4157x |
| `smsdt_s` | `seed1` | `DFD` *(pure)* | `dev_overlap` | 0.6937 | 0.2046 | 0.2037 | 0.1886 | 0.0151 | 0.0079 | 0.0284 | 0.0453 | 0.3621 | 1.2675x |
| `smsdt_s` | `seed2` | `CELEBDF` | `confirmatory` | 0.5636 | 0.2153 | 0.2219 | 0.1960 | 0.0259 | 0.0130 | 0.0627 | 0.0674 | 0.3765 | 1.1798x |
| `smsdt_s` | `seed2` | `DFD` *(pure)* | `dev_overlap` | 0.5636 | 0.1693 | 0.1780 | 0.1657 | 0.0123 | 0.0049 | 0.0306 | 0.0700 | 0.2861 | 1.2243x |
| `smsdt_s` | `seed3` | `CELEBDF` | `confirmatory` | 0.5632 | 0.1864 | 0.2116 | 0.1728 | 0.0389 | 0.0221 | 0.0628 | 0.0674 | 0.3559 | 1.1798x |
| `smsdt_s` | `seed3` | `DFD` *(pure)* | `dev_overlap` | 0.5632 | 0.1784 | 0.1791 | 0.1782 | 0.0009 | 0.0000 | 0.0175 | 0.0741 | 0.2842 | 1.2963x |
| `smsdt_st` | `seed1` | `CELEBDF` | `confirmatory` | 0.5799 | 0.2121 | 0.2276 | 0.1887 | 0.0389 | 0.0125 | 0.0698 | 0.1404 | 0.3147 | 1.6386x |
| `smsdt_st` | `seed1` | `DFD` *(pure)* | `dev_overlap` | 0.5799 | 0.2080 | 0.2189 | 0.2027 | 0.0162 | 0.0117 | 0.0334 | 0.1317 | 0.3061 | 1.5364x |
| **`smsdt_st`** | **`seed2`** | **`CELEBDF`** | `confirmatory` | 0.7924 | 0.2363 | 0.2927 | 0.2001 | **0.0926** | **0.0561** | 0.1402 | 0.0618 | 0.5235 | 2.1629x |
| `smsdt_st` | `seed2` | `DFD` *(pure)* | `dev_overlap` | 0.7924 | 0.2498 | 0.2244 | 0.2156 | 0.0088 | 0.0050 | 0.0225 | 0.0741 | 0.3748 | 2.5926x |
| `smsdt_st` | `seed3` | `CELEBDF` | `confirmatory` | 0.5004 | 0.2081 | 0.2290 | 0.2004 | 0.0286 | 0.0140 | 0.0537 | 0.1404 | 0.3176 | 1.5125x |
| `smsdt_st` | `seed3` | `DFD` *(pure)* | `dev_overlap` | 0.5004 | 0.2419 | 0.2209 | 0.1849 | 0.0360 | 0.0157 | 0.0594 | 0.2263 | 0.2154 | 2.4375x |

#### Scientific Findings & Pre-Registration Audit (H2):
1. **Why Earlier Runs Showed $g \le 0.0374$:** The prior run fell back to raw `0.5000` (`tau_val = 0.5000` on every S-MSDT model) because dev-split Platt calibration JSONs had not yet been generated. Re-running with actual dev calibration (`results/calib/*.json`), D13 identity clusters, and D14 pure DFD changes the picture substantially.
2. **H2 Is Formally `undetermined` (Not Falsified):** With Celeb-DF's `121` resolved identity clusters (**D13**), `smsdt_s` seed 1 still clears the H2 degradation threshold ($g = 0.0984$, $95\%\text{ CI }[0.0796, 0.1283]$, lower CI $\ge 0.05$, with `FNR = 0.5000` at $\tau_{\text{cal}}=0.6937$), while seeds 2–3 stay below `0.05` (`0.0259 [0.0130, 0.0627]` and `0.0389 [0.0221, 0.0628]`); `gend_ours` stays below `0.05` on Celeb-DF (`0.0074–0.0258`, though `0.1112 [0.0793, 0.1511]` and `0.1060 [0.0756, 0.1456]` on exploratory pure DFD seeds 1–2). Exploratory `smsdt_st` seed 2 also exceeds `0.05` on Celeb-DF ($g = 0.0926$, CI $[0.0561, 0.1402]$). Additional confirmatory unseen data/seeds are required before issuing a confirm/falsify verdict for H2.
3. **Threshold rule used here:** every FPR in this section is at the run's FF++-val Platt + Youden cutoff (`tau_cal`). §§9–11 use the FF++-val `(1 - α)` source quantile instead, so `gend_official` reads `21.35%` here and `15.73%` there; both are correct for their rule. Label the rule in every paper table.
4. **HTER Still Masks Large Asymmetric FPR Shifts:** Even where $g < 0.05$, `FPR` under the frozen calibrated cutoff inflates sharply relative to `FF++val` (`smsdt_full` seed 3: `12.92%` on Celeb-DF and `13.58%` on pure DFD vs `2.14%` = `6.03x` / `6.34x`; `gend_official` seed 0: `21.35%` on Celeb-DF and `20.58%` on pure DFD vs `6.43%` = `3.32x` / `3.20x`), motivating the explicit FPR-control audit in §10.

---

## 6. Hypothesis H3: Operational Prior Shift & Saerens EM Correction Audit

Compiled via [`tools/prior_sweep.py`](file:///e:/DEEPFAKE/tools/prior_sweep.py) into [`results/tables/prior_shift_sweep.csv`](file:///e:/DEEPFAKE/results/tables/prior_shift_sweep.csv) and [`results/tables/prior_shift_sweep.md`](file:///e:/DEEPFAKE/results/tables/prior_shift_sweep.md) across all **13** Celeb-DF v2 evaluation runs using **Platt-calibrated posteriors** (`results/calib/*.json`) under simulated base-rate shifts $\pi \in \{0.1, 0.3, 0.5, 0.7, 0.9\}$ (50 Monte Carlo subsamplings per condition).

### Calibrated 3-Seed Mean by Architecture (`prior_shift_sweep.csv`, Celeb-DF v2)

| Architecture (`N` runs) | Target Prior $\pi$ | Mean Prior Err $|\hat{\pi}-\pi|$ | Frozen HTER ($\tau=0.50$) | Post-EM HTER | HTER Reduction | Frozen Acc | Bayes-EM Acc | Bayes Acc Gain |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `gend_official` (`1`) | `0.1 / 0.5 / 0.9` | `0.1923 / 0.2670 / 0.1000` | `0.1632 / 0.1647 / 0.1608` | `0.1748 / 0.2299 / 0.5000` | `-0.0115 / -0.0653 / -0.3392` | `0.7967 / 0.8353 / 0.8760` | `0.8534 / 0.6624 / 0.8995` | `+0.0568 / -0.1730 / +0.0234` |
| `gend_ours` (`3`) | `0.1 / 0.5 / 0.9` | `0.0868 / 0.1783 / 0.2071` | `0.2630 / 0.2598 / 0.2655` | `0.4603 / 0.2527 / 0.3778` | `-0.1974 / +0.0071 / -0.1122` | `0.7347 / 0.7402 / 0.7453` | `0.9013 / 0.7137 / 0.7319` | `+0.1667 / -0.0265 / -0.0134` |
| `smsdt_s` (`3`) | `0.1 / 0.5 / 0.9` | `0.0708 / 0.1871 / 0.2221` | `0.2361 / 0.2371 / 0.2346` | `0.4489 / 0.2869 / 0.1993` | `-0.2128 / -0.0498 / +0.0353` | `0.9030 / 0.7629 / 0.6246` | `0.9102 / 0.7404 / 0.7742` | `+0.0072 / -0.0225 / +0.1495` |
| `smsdt_st` (`3`) | `0.1 / 0.5 / 0.9` | `0.0493 / 0.1440 / 0.1940` | `0.2534 / 0.2524 / 0.2490` | `0.4368 / 0.2807 / 0.2241` | `-0.1833 / -0.0283 / +0.0248` | `0.8576 / 0.7476 / 0.6421` | `0.9124 / 0.7498 / 0.7899` | `+0.0547 / +0.0022 / +0.1478` |
| `smsdt_full` (`3`) | `0.1 / 0.5 / 0.9` | **`0.0144 / 0.0382 / 0.0521`** | `0.2100 / 0.2124 / 0.2105` | `0.3159 / 0.2135 / 0.2298` | `-0.1059 / -0.0011 / -0.0193` | `0.8395 / 0.7876 / 0.7376` | **`0.9201 / 0.7974 / 0.9114`** | **`+0.0806 / +0.0097 / +0.1738`** |

*(Full 65-row breakdown across all $\pi \in \{0.1, 0.3, 0.5, 0.7, 0.9\}$ is in [`results/tables/prior_shift_sweep.md`](file:///e:/DEEPFAKE/results/tables/prior_shift_sweep.md).)*

#### Reading of H3a Results:
1. **H3a Is Formally `falsified` on HTER Reduction:** Even with calibrated posteriors, Saerens EM does not reduce HTER consistently by $\ge 0.03$; it worsens fixed-threshold HTER under low priors across all architectures (e.g., `-0.1059` to `-0.2128` at $\pi=0.10$).
2. **Bayes-Decision Accuracy (Not HTER) Is the Appropriate EM Metric:** As established in [`prior_sweep.py`](file:///e:/DEEPFAKE/tools/prior_sweep.py), HTER is prior-invariant and EM re-weights posteriors monotonically, so EM is designed to maximize **Bayes-decision accuracy** (or minimize expected cost) under shifted base rates rather than lowering HTER at $\tau=0.50$. On this metric (`acc_gain`), `smsdt_full` achieves accurate prior tracking (`0.0144–0.0521` mean error) and large accuracy gains under strong skew (`+8.06 pts` at $\pi=0.10$, `+17.38 pts` at $\pi=0.90$), and `smsdt_s` / `smsdt_st` gain `+14.95` / `+14.78 pts` at $\pi=0.90$, whereas `gend_official` (`0.23–0.27` prior error) loses accuracy at intermediate priors (`-9.28` to `-17.30 pts`).

---

## 7. Test-Time Adaptation (TTA) & Clip-Consistency Benchmark (RQ3 / H3b / H6)

Compiled via [`tools/eval_tta.py`](file:///e:/DEEPFAKE/tools/eval_tta.py) into [`results/tables/tta_summary.csv`](file:///e:/DEEPFAKE/results/tables/tta_summary.csv) and [`results/tables/tta_summary.md`](file:///e:/DEEPFAKE/results/tables/tta_summary.md).

Evaluates label-free LayerNorm adaptation ($lr = 1\times 10^{-4}$, batch size 2 videos, $K=4$) on a **single model (`smsdt_full`), single seed (`seed1`), and single target dataset (`Celeb-DF-v2` for consistency; `Celeb-DF-v2` and `DFD` for legacy entropy)** under two objectives (`entropy` and `consistency`) and two modes (`online` streaming vs `batch_episodic` reset):

| Model | Target Dataset | Adaptation Mode | Objective | Pre-TTA AUC | Post-TTA AUC | ΔAUC | Mean Metric/Ent | Pred Positive % | Ground Truth % | Collapse Audit |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `smsdt_full_seed1` | Celeb-DF-v2 | **Online** | `entropy` | 0.8880 | 0.3386 | **-0.5494** | 0.1187 | 6.6% | 65.6% | **COLLAPSED (FAILED)** |
| `smsdt_full_seed1` | Celeb-DF-v2 | **Batch-Episodic** | `entropy` | 0.8880 | 0.8880 | +0.0000* | 0.2679 | 61.0% | 65.6% | *Invalid (pre-reset logits)* |
| `smsdt_full_seed1` | DFD | **Online** | `entropy` | 0.8583 | 0.2390 | **-0.6193** | 0.0252 | 0.6% | 87.2% | **COLLAPSED (FAILED)** |
| `smsdt_full_seed1` | DFD | **Batch-Episodic** | `entropy` | 0.8583 | 0.8583 | +0.0000* | 0.2091 | 70.7% | 87.2% | *Invalid (pre-reset logits)* |
| `smsdt_full_seed1` | Celeb-DF-v2 | **Online** | `consistency` | 0.8880 | 0.5409 | **-0.3472** | 1.0326 | 10.8% | 65.6% | **COLLAPSED (FAILED)** |
| `smsdt_full_seed1` | Celeb-DF-v2 | **Batch-Episodic** | `consistency` | 0.8880 | 0.8877 | **-0.0003** | 1.3669 | 60.4% | 65.6% | **NO COLLAPSE / NO GAIN** |

*\*Methodological Note: The legacy batch-episodic `entropy` rows (`+0.0000`) measured logits **before** the parameter reset and are **not** a valid test of adaptation; only the updated `consistency` runner measures post-adaptation weights prior to reset (`0.8877`, $\Delta\text{AUC} = -0.0003$).*

#### Key Scientific Findings (RQ3 / TTA / H6):
1. **H6 Is Formally `falsified`:** Online consistency TTA **collapses** (`0.5409` AUC vs `0.8880` pre-TTA, predicted-positive rate `10.8%` vs `65.6%` true positives), failing the pre-registered no-collapse condition; batch-episodic consistency changes AUC by `-0.0003` (`0.8877` vs `0.8880`), yielding neither collapse nor adaptation gain. Scope: **single model, single seed, and single dataset** (`smsdt_full_seed1` on `Celeb-DF-v2`).
2. **Online Entropy Collapse (H3b):** Unconstrained online entropy minimization rapidly collapses predictions to near-zero positives (`6.6%` on Celeb-DF, `0.6%` on DFD; $\Delta\text{AUC} \le -0.55$). Because batch-episodic entropy measured pre-reset logits (`+0.0000`), only batch-episodic consistency provides a clean reset check (`-0.0003` $\Delta\text{AUC}$).

---

## 8. Perturbation Robustness Benchmark (Corrected Pipeline)

Compiled via [`tools/eval_robustness.py`](file:///e:/DEEPFAKE/tools/eval_robustness.py) into [`results/tables/robustness_summary.csv`](file:///e:/DEEPFAKE/results/tables/robustness_summary.csv) and [`results/tables/robustness_summary.md`](file:///e:/DEEPFAKE/results/tables/robustness_summary.md).

Evaluated on a **single model (`smsdt_full`), single seed (`seed1`), and Celeb-DF-v2 only** ($N=518$ videos, $K=4, T=8$) **after fixing the uint8 normalization order bug** (perturbations applied on $[0, 255]$ uint8 RGB frames prior to CLIP mean/std normalization):

| Perturbation Type | Severity / Level | Test AUC | AUC Degradation (Clean $-$ Perturbed) |
| :--- | :---: | :---: | :---: |
| **Clean Baseline** | None | **0.8884** | +0.0000 |
| **JPEG Compression** | Quality 90 | **0.8972** | -0.0088 |
| **JPEG Compression** | Quality 70 | **0.8829** | +0.0055 |
| **JPEG Compression** | Quality 50 | **0.8482** | +0.0402 |
| **JPEG Compression** | Quality 30 | **0.8048** | +0.0836 |
| **Gaussian Blur** | $\sigma = 0.5$ | **0.8920** | -0.0035 |
| **Gaussian Blur** | $\sigma = 1.0$ | **0.8732** | +0.0152 |
| **Gaussian Blur** | $\sigma = 2.0$ | **0.7987** | +0.0898 |
| **Downscaling** | Factor 0.75 | **0.8907** | -0.0023 |
| **Downscaling** | Factor 0.50 | **0.8734** | +0.0151 |
| **Downscaling** | Factor 0.25 | **0.8217** | +0.0667 |

#### Scientific Findings (Single Model / Single Seed / Celeb-DF-v2 Only):
1. **Graceful Degradation Curve:** Once the earlier CLIP normalization bug (which had caused an artificial drop to $0.6373$ at JPEG 90) was resolved, `smsdt_full_seed1` exhibits smooth degradation under web compression. Moderate perturbations (JPEG Q=70: $0.8829$; Blur $\sigma=1.0$: $0.8732$; Downscale 0.50x: $0.8734$) preserve $\ge 98\%$ of clean discriminative performance.
2. **Severe High-Frequency Loss:** Larger drops occur under aggressive artifacts (JPEG Q=30 drops by $0.0836$; Blur $\sigma=2.0$ drops by $0.0898$; Downscale 0.25x drops by $0.0667$). Note: scope is limited to `smsdt_full_seed1` on Celeb-DF-v2.

---

## 9. Gate G1 Resolution: Official GenD Benchmark (`gend_official`)

Evaluated directly via [`smsdt/eval_cross.py`](file:///e:/DEEPFAKE/smsdt/eval_cross.py) using the official author Hugging Face weights (`hf:yermandy/GenD_CLIP_L_14`, $T=1$) across all four evaluation splits:
- **FF++ c23 Val (`N=840` pure FF++):** [`results/scores/gend_official_seed0_ffppval.csv`](file:///e:/DEEPFAKE/results/scores/gend_official_seed0_ffppval.csv) — **0.9371 Video AUC** (dev Platt calibration saved in [`results/calib/gend_official_seed0.json`](file:///e:/DEEPFAKE/results/calib/gend_official_seed0.json), $\tau_{\text{cal}}=0.6660$)
- **FF++ c23 Test (`N=840`):** [`results/scores/gend_official_seed0_ffpp.csv`](file:///e:/DEEPFAKE/results/scores/gend_official_seed0_ffpp.csv) — **0.9452 Video AUC**
- **DFD Test (`N=2,847` pure; `2,987` raw CSV):** [`results/scores/gend_official_seed0_dfd.csv`](file:///e:/DEEPFAKE/results/scores/gend_official_seed0_dfd.csv) — **0.8840 Pure DFD Video AUC** (`0.9076` raw before D14 stripped the `140` FF++ test reals)
- **Celeb-DF-v2 Test (`N=518`):** [`results/scores/gend_official_seed0_celebdf.csv`](file:///e:/DEEPFAKE/results/scores/gend_official_seed0_celebdf.csv) — **0.9327 Video AUC**

### 9.1 Celeb-DF-v2 Performance & Score Distribution (`gend_official_seed0_celebdf.csv`)

| Metric | Official GenD (`hf:yermandy/GenD_CLIP_L_14`) | Notes |
| :--- | :---: | :--- |
| **Video AUC** | **0.9327** | Above every S-MSDT variant (`smsdt_full`: `0.8882`, `smsdt_s`: `0.8826`, `gend_ours`: `0.8294`) |
| **Top-50% Clip AUC** | **0.9277** | Frame/Clip AUC: `0.8841` |
| **Biometric Utility** | TPR@1%FPR = `51.76%` \| TPR@5%FPR = `73.82%` | Raw ECE = `0.1362` |
| **Oracle Acc@EER (0.82)** | `84.2%` (`84.1%` Fake / `84.3%` Real) | High rank-order separability |
| **Uncalibrated $\tau=0.500$** | Overall Acc: `80.12%` \| Bal Acc: `71.74%` \| F1: `0.8668` | Recall `98.53%` (`335/340`); Raw FPR `55.06%` (`98/178`) — softmax is not calibrated at `0.5` |

| Class Subset ($N$) | `score_mean` [min / median / tail / max] | `logit_mean` [min / median / tail / max] | Raw $\tau=0.50$ Rate |
| :--- | :---: | :---: | :---: |
| **Pristine Reals ($N=178$)** | `0.0555` / `0.5339` / p95=`0.8956` / `0.9749` | `-2.9609` / `+0.2136` / p95=`+2.4001` / `+3.7031` | FPR = `0.5506` (`98 / 178`) |
| **Manipulated Fakes ($N=340$)** | `0.1669` / `0.9474` / p05=`0.6906` / `0.9961` | `-1.6501` / `+3.0998` / p05=`+1.0679` / `+5.6094` | TPR = `0.9853` (`335 / 340`) |

### 9.2 Like-for-Like Source-Quantile Transfer on `gend_official` (`--source ffppval`)

Evaluating `gend_official` under the exact same dev-split `source_quantile` protocol (`results/scores/gend_official_seed0_ffppval.csv`) provides a direct like-for-like comparison with S-MSDT:

| Target Dataset | Budget ($\alpha$) | Source Logit Cutoff | Achieved FPR | Conservative 95% CI | FPR / $\alpha$ | TPR | H4 Failure (`ci_low_cons > 2α`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Celeb-DF-v2** | `0.05` | `1.8057` | **`0.1573`** | `[0.1050, 0.2180]` | **`3.15x`** | `0.8647` | **`True`** |
| **Celeb-DF-v2** | `0.10` | `1.2140` | **`0.3146`** | `[0.2499, 0.3861]` | **`3.15x`** | `0.9382` | **`True`** |
| **DFD** *(pure, 243 reals)* | `0.05` | `1.8057` | **`0.1934`** | `[0.1487, 0.2477]` | **`3.87x`** | `0.7976` | **`True`** |
| **DFD** *(pure, 243 reals)* | `0.10` | `1.2140` | **`0.2963`** | `[0.2424, 0.3565]` | **`2.96x`** | `0.8468` | **`True`** |

#### Gate G1 Scientific Conclusion:
1. **No Superiority Over Official GenD (Gate G1 Branch B):** Official author weights (`0.9327` Celeb-DF AUC) outperform our in-house FF++-only `gend_ours` (`0.8294`) and every S-MSDT variant (`smsdt_full`: `0.8882`, `smsdt_s`: `0.8826`, `smsdt_st`: `0.8704`). Accordingly, S-MSDT cannot be framed as a new best detector; our `+0.059` AUC delta over `gend_ours` compares our training recipe against our own baseline rather than published GenD.
2. **Like-for-Like Threshold Transfer Fails on Official GenD Too:** At a raw `0.50` cutoff, official GenD's `55.06%` FPR only shows that its softmax logits are not calibrated around `0.5` (`reals` median logit `+0.2136`). However, even after computing proper dev-split `FF++val` quantiles (`source_quantile`), `gend_official` inflates Celeb-DF FPR **`3.15x`** to **`15.73%`** at $\alpha=0.05$ (`31.46%` at $\alpha=0.10$) and pure-DFD FPR **`3.87x`** to **`19.34%`** at $\alpha=0.05$ (`29.63%` at $\alpha=0.10$, all four `h4_failure = True`) — **higher FPR inflation than every S-MSDT variant under `ffppval`**. Platt calibration (`tau_cal = 0.6660`) similarly increases FPR from `6.43%` on `FFPPVAL` to `21.35%` on `Celeb-DF` (`3.32x`) and `20.58%` on pure `DFD` (`3.20x`). Thus cross-domain FPR inflation under frozen source thresholds **also occurs for the official GenD checkpoint**.

---

## 10. False Positive Rate Control Under Domain Shift (Hypotheses H4 & H5)

Compiled via [`tools/fpr_control.py`](file:///e:/DEEPFAKE/tools/fpr_control.py) (`--scores-dir results/scores --source ffppval --confirmatory-seeds 3 4 5`) across all `13` runs (`12` multi-seed trained runs + external `gend_official_seed0`) at target false-alarm budgets $\alpha \in \{0.05, 0.10\}$.

To eliminate sub-binomial CIs caused by coarse real-video clustering (`121` Celeb-DF clusters for `178` reals) and contaminated DFD negatives (now **`18` actor identities for `243` pure DFD reals** under **D14**), [`tools/fpr_control.py`](file:///e:/DEEPFAKE/tools/fpr_control.py) reports **cluster-conservative bounds** (`fpr_ci_low_cons = min(boot_low, wilson_low)`, `fpr_ci_high_cons = max(boot_high, wilson_high)`) and evaluates H4 failures using `fpr_ci_low_cons > 2 * alpha`. Note that DFD stays exploratory (`dev_overlap`) and has `< 30` real clusters (`18`), so conservative Wilson bounds govern DFD intervals.

Outputs stored in:
- [`results/tables/fpr_control.csv`](file:///e:/DEEPFAKE/results/tables/fpr_control.csv) & [`results/tables/fpr_control.md`](file:///e:/DEEPFAKE/results/tables/fpr_control.md) (`208` policy rows across `13 runs × 2 datasets × 4 policies × 2 alphas`)
- [`results/tables/fpr_control_h4.csv`](file:///e:/DEEPFAKE/results/tables/fpr_control_h4.csv) (`52` `source_quantile` rows testing `fpr_ci_low_cons > 2 * alpha`)
- [`results/tables/fpr_control_h4b.csv`](file:///e:/DEEPFAKE/results/tables/fpr_control_h4b.csv) (`12` paired delta-FPR rows for `ST - S` and `Full - S` on Celeb-DF-v2)
- [`results/tables/fpr_control_prior_stress.csv`](file:///e:/DEEPFAKE/results/tables/fpr_control_prior_stress.csv) (`624` subsampled prior-stress rows across $\pi_{\text{fake}} \in \{0.50, 0.66, 0.91\}$)
- [`results/tables/fpr_control_verdicts.json`](file:///e:/DEEPFAKE/results/tables/fpr_control_verdicts.json) (formal H4a `undetermined` and H5 `falsified` verdicts)
- [`results/tables_resample/`](file:///e:/DEEPFAKE/results/tables_resample) (`6` sensitivity files under `--resample-source` incorporating dev-threshold estimation uncertainty from the `140` FF++ val reals)

### 10.1 Seed-by-Seed Dev-Quantile Transfer at $\alpha = 0.05$ (`--source ffppval`, Celeb-DF-v2 Confirmatory)

| Model | Seed (Role) | Logit Cutoff ($\tau_{\text{FF++val},0.95}$) | Target ($\alpha$) | Achieved FPR | Conservative 95% CI | FPR / $\alpha$ | TPR (Recall) | H4 Check (`ci_low_cons > 2α`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`gend_official`** | `0` *(ext)* | `1.8057` | 0.0500 | **0.1573** | **[0.1050, 0.2180]** | **3.15x** | 0.8647 | **H4 FAILURE** |
| `gend_ours` | `1` *(exp)* | `0.3439` | 0.0500 | 0.0169 | [0.0000, 0.0484] | 0.34x | 0.4029 | Bounded |
| `gend_ours` | `2` *(exp)* | `0.3113` | 0.0500 | 0.0169 | [0.0000, 0.0484] | 0.34x | 0.4088 | Bounded |
| `gend_ours` | `3` *(conf)* | `0.3052` | 0.0500 | 0.0169 | [0.0000, 0.0484] | 0.34x | 0.4000 | Bounded |
| `smsdt_s` | `1` *(exp)* | `1.0547` | 0.0500 | 0.0506 | [0.0226, 0.0933] | 1.01x | 0.5088 | Bounded |
| `smsdt_s` | `2` *(exp)* | `0.4858` | 0.0500 | 0.0562 | [0.0270, 0.1003] | 1.12x | 0.5941 | Bounded |
| `smsdt_s` | `3` *(conf)* | `0.4436` | 0.0500 | 0.0674 | [0.0311, 0.1141] | 1.35x | 0.6118 | Bounded |
| `smsdt_st` | `1` *(exp)* | `1.4438` | 0.0500 | 0.0955 | [0.0517, 0.1476] | 1.91x | 0.5853 | Bounded |
| `smsdt_st` | `2` *(exp)* | `1.4692` | 0.0500 | 0.0787 | [0.0410, 0.1277] | 1.57x | 0.5235 | Bounded |
| `smsdt_st` | `3` *(conf)* | `1.2402` | 0.0500 | 0.0787 | [0.0422, 0.1277] | 1.57x | 0.5588 | Bounded |
| `smsdt_full` | `1` *(exp)* | `1.3628` | 0.0500 | 0.1180 | [0.0699, 0.1736] | 2.36x | 0.7000 | Bounded (`ci_low_cons <= 0.10`) |
| `smsdt_full` | `2` *(exp)* | `1.8874` | 0.0500 | 0.0899 | [0.0494, 0.1410] | 1.80x | 0.6500 | Bounded |
| **`smsdt_full`** | **`3` *(conf)*** | `1.1027` | 0.0500 | **0.1573** | **[0.1046, 0.2180]** | **3.15x** | 0.6912 | **H4 FAILURE** |

### 10.2 Full H4a Summary & Paired H4b Comparisons (`fpr_control_h4.csv` & `fpr_control_h4b.csv`)

#### H4a (`source_quantile` across `52` rows, `--source ffppval`)

| Target Dataset (`role`) | Model | Budget ($\alpha$) | Mean FPR | Min–Max FPR Across Seeds | Mean FPR / $\alpha$ | Mean TPR | H4 Failures |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Celeb-DF-v2** (`confirmatory`) | `gend_official` | 0.05 | **`0.1573`** | `0.1573` | **`3.15x`** | `0.8647` | **`1 / 1`** |
| **Celeb-DF-v2** (`confirmatory`) | `gend_ours` | 0.05 | `0.0169` | `0.0169 – 0.0169` | `0.34x` | `0.4039` | `0 / 3` |
| **Celeb-DF-v2** (`confirmatory`) | `smsdt_s` | 0.05 | `0.0581` | `0.0506 – 0.0674` | `1.16x` | `0.5716` | `0 / 3` |
| **Celeb-DF-v2** (`confirmatory`) | `smsdt_st` | 0.05 | `0.0843` | `0.0787 – 0.0955` | `1.69x` | `0.5559` | `0 / 3` |
| **Celeb-DF-v2** (`confirmatory`) | `smsdt_full` | 0.05 | **`0.1217`** | `0.0899 – 0.1573` | **`2.43x`** | `0.6804` | **`1 / 3`** *(conf seed3)* |
| **Celeb-DF-v2** (`confirmatory`) | `gend_official` | 0.10 | **`0.3146`** | `0.3146` | **`3.15x`** | `0.9382` | **`1 / 1`** |
| **Celeb-DF-v2** (`confirmatory`) | `gend_ours` | 0.10 | `0.0393` | `0.0393 – 0.0393` | `0.39x` | `0.4667` | `0 / 3` |
| **Celeb-DF-v2** (`confirmatory`) | `smsdt_s` | 0.10 | `0.0805` | `0.0618 – 0.1067` | `0.81x` | `0.6549` | `0 / 3` |
| **Celeb-DF-v2** (`confirmatory`) | `smsdt_st` | 0.10 | `0.1311` | `0.1067 – 0.1461` | `1.31x` | `0.6627` | `0 / 3` |
| **Celeb-DF-v2** (`confirmatory`) | `smsdt_full` | 0.10 | `0.1479` | `0.1348 – 0.1685` | `1.48x` | `0.7480` | `0 / 3` |
| **DFD** *(dev_overlap, pure 243 reals)* | `gend_official` | 0.05 | **`0.1934`** | `0.1934` | **`3.87x`** | `0.7976` | **`1 / 1`** |
| **DFD** *(dev_overlap, pure 243 reals)* | `gend_ours` | 0.05 | `0.0110` | `0.0082 – 0.0123` | `0.22x` | `0.5698` | `0 / 3` |
| **DFD** *(dev_overlap, pure 243 reals)* | `smsdt_s` | 0.05 | `0.0535` | `0.0453 – 0.0617` | `1.07x` | `0.6807` | `0 / 3` |
| **DFD** *(dev_overlap, pure 243 reals)* | `smsdt_st` | 0.05 | `0.0960` | `0.0905 – 0.1070` | `1.92x` | `0.6723` | `0 / 3` |
| **DFD** *(dev_overlap, pure 243 reals)* | `smsdt_full` | 0.05 | **`0.1344`** | `0.0658 – 0.1811` | **`2.69x`** | `0.6601` | **`1 / 3`** *(seed1)* |
| **DFD** *(dev_overlap, pure 243 reals)* | `gend_official` | 0.10 | **`0.2963`** | `0.2963` | **`2.96x`** | `0.8468` | **`1 / 1`** |
| **DFD** *(dev_overlap, pure 243 reals)* | `gend_ours` | 0.10 | `0.0288` | `0.0288 – 0.0288` | `0.29x` | `0.6240` | `0 / 3` |
| **DFD** *(dev_overlap, pure 243 reals)* | `smsdt_s` | 0.10 | `0.0933` | `0.0741 – 0.1111` | `0.93x` | `0.7244` | `0 / 3` |
| **DFD** *(dev_overlap, pure 243 reals)* | `smsdt_st` | 0.10 | `0.1783` | `0.1317 – 0.2263` | `1.78x` | `0.7341` | `0 / 3` |
| **DFD** *(dev_overlap, pure 243 reals)* | `smsdt_full` | 0.10 | `0.1797` | `0.1317 – 0.2222` | `1.80x` | `0.7071` | `0 / 3` |

**Total H4 Failures:** **`6 / 52`** rows (`smsdt_full` `seed3` on Celeb-DF $\alpha=0.05$, `smsdt_full` `seed1` on exploratory pure DFD $\alpha=0.05$, plus all `4` `gend_official` rows across Celeb-DF $\alpha\in\{0.05,0.10\}$ and pure DFD $\alpha\in\{0.05,0.10\}$). Note that switching to `--resample-source` sensitivity ([`results/tables_resample/`](file:///e:/DEEPFAKE/results/tables_resample)) to account for threshold estimation variance on the ~`140` `FF++val` dev reals widens every CI (`smsdt_full seed3`: `[0.1046, 0.2180] -> [0.0966, 0.2180]`; `gend_official`: `[0.1050, 0.2180] -> [0.0179, 0.3060]`) and reduces `h4_failures` from `6/52` to **`0/52`**, illustrating why ~140 source reals cannot certify a tight tail operating point across shifts.

#### Paired H4b Differences (`fpr_control_h4b.csv`, Celeb-DF-v2, $N=178$ reals)

| Contrast | Seed | Budget ($\alpha$) | $\text{FPR}_A$ | $\text{FPR}_B$ | Paired $\Delta\text{FPR}$ ($A-B$) | 95% Cluster CI | Excludes 0? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `smsdt_st` $-$ `smsdt_s` | `1` *(exp)* | 0.05 | 0.0955 | 0.0506 | **+0.0449** | [+0.0055, +0.0905] | Yes |
| `smsdt_st` $-$ `smsdt_s` | `2` *(exp)* | 0.05 | 0.0787 | 0.0562 | +0.0225 | [-0.0126, +0.0612] | No |
| `smsdt_st` $-$ `smsdt_s` | `3` *(conf)* | 0.05 | 0.0787 | 0.0674 | +0.0112 | [-0.0225, +0.0462] | No |
| `smsdt_st` $-$ `smsdt_s` | `1` *(exp)* | 0.10 | 0.1461 | 0.0618 | **+0.0843** | [+0.0402, +0.1340] | Yes |
| `smsdt_st` $-$ `smsdt_s` | `2` *(exp)* | 0.10 | 0.1067 | 0.1067 | 0.0000 | [-0.0444, +0.0424] | No |
| `smsdt_st` $-$ `smsdt_s` | `3` *(conf)* | 0.10 | 0.1404 | 0.0730 | **+0.0674** | [+0.0275, +0.1098] | Yes |
| `smsdt_full` $-$ `smsdt_s` | `1` *(exp)* | 0.05 | 0.1180 | 0.0506 | **+0.0674** | [+0.0262, +0.1149] | Yes |
| `smsdt_full` $-$ `smsdt_s` | `2` *(exp)* | 0.05 | 0.0899 | 0.0562 | +0.0337 | [-0.0061, +0.0765] | No |
| `smsdt_full` $-$ `smsdt_s` | `3` *(conf)* | 0.05 | 0.1573 | 0.0674 | **+0.0899** | [+0.0447, +0.1377] | Yes |
| `smsdt_full` $-$ `smsdt_s` | `1` *(exp)* | 0.10 | 0.1348 | 0.0618 | **+0.0730** | [+0.0316, +0.1212] | Yes |
| `smsdt_full` $-$ `smsdt_s` | `2` *(exp)* | 0.10 | 0.1404 | 0.1067 | +0.0337 | [-0.0160, +0.0808] | No |
| `smsdt_full` $-$ `smsdt_s` | `3` *(conf)* | 0.10 | 0.1685 | 0.0730 | **+0.0955** | [+0.0484, +0.1453] | Yes |

### 10.3 Complete H5 Policy Comparison & Degenerate GMM Audit (`fpr_control.csv`, `fpr_control_prior_stress.csv`, `fpr_control_verdicts.json`)

Aggregate H5 verdicts stored in [`results/tables/fpr_control_verdicts.json`](file:///e:/DEEPFAKE/results/tables/fpr_control_verdicts.json) (evaluated on Celeb-DF-v2 confirmatory across all 13 runs):
- `target_gmm@alpha=0.05`: **`falsified (degenerate fits)`** (`err_policy = 0.0680` vs `err_source = 0.0449`, `degenerate_rate = 1.0` in worst prior-stress cell)
- `target_gmm@alpha=0.10`: **`falsified`** (`err_policy = 0.0826` vs `err_source = 0.0576`, `degenerate_rate = 0.0`)
- `target_gmm_anchored@alpha=0.05`: **`falsified (degenerate fits)`** (`err_policy = 0.0672` vs `err_source = 0.0449`, `degenerate_rate = 1.0`)
- `target_gmm_anchored@alpha=0.10`: **`falsified`** (`err_policy = 0.0816` vs `err_source = 0.0576`, `degenerate_rate = 0.0`)

Policy means at $\alpha = 0.05$ (`3` seeds per trained model, `1` run for `gend_official`):

| Dataset | Model | `source_quantile` FPR / Error / TPR | `target_gmm` FPR / Error / TPR *(degenerate frac)* | `oracle` FPR / Error / TPR |
| :--- | :--- | :---: | :---: | :---: |
| **Celeb-DF-v2** | `gend_official` *(1)* | `0.1573` / `0.1073` / `0.8647` | `0.0225` / `0.0275` / `0.6294` *(0.0)* | `0.0449` / `0.0051` / `0.7265` |
| **Celeb-DF-v2** | `gend_ours` | `0.0169` / `0.0331` / `0.4039` | `0.1573` / `0.1073` / `0.6627` *(0.0)* | `0.0449` / `0.0051` / `0.5225` |
| **Celeb-DF-v2** | `smsdt_s` | `0.0581` / `0.0081` / `0.5716` | `0.0412` / `0.0096` / `0.5029` *(0.0)* | `0.0449` / `0.0051` / `0.5059` |
| **Celeb-DF-v2** | `smsdt_st` | `0.0843` / `0.0343` / `0.5559` | `0.1086` / `0.0732` / `0.5980` *(0.0)* | `0.0449` / `0.0051` / `0.4814` |
| **Celeb-DF-v2** | `smsdt_full` | `0.1217` / `0.0717` / `0.6804` | **`0.0000`** / `0.0500` / **`0.1539`** *(0.0)* | `0.0449` / `0.0051` / `0.5373` |
| **DFD** | `gend_official` *(1)* | `0.1934` / `0.1434` / `0.7976` | `0.0000` / `0.0500` / `0.4528` *(0.0)* | `0.0494` / `0.0006` / `0.6525` |
| **DFD** | `gend_ours` | `0.0110` / `0.0390` / `0.5698` | `0.0288` / `0.0212` / `0.6206` *(0.0)* | `0.0494` / `0.0006` / `0.7038` |
| **DFD** | `smsdt_s` | `0.0535` / `0.0067` / `0.6807` | `0.0000` / `0.0500` / `0.3510` *(0.0)* | `0.0494` / `0.0006` / `0.6822` |
| **DFD** | `smsdt_st` | `0.0960` / `0.0460` / `0.6723` | **`0.0000`** / `0.0500` / **`0.0000`** **`(1.00, deg=True)`** | `0.0494` / `0.0006` / `0.6212` |
| **DFD** | `smsdt_full` | `0.1344` / `0.0844` / `0.6601` | **`0.0000`** / `0.0500` / **`0.0000`** **`(1.00, deg=True)`** | `0.0494` / `0.0006` / `0.5408` |

**Why GMM Collapses (`outside_support` Degeneracy):** Under PR #11's fixed degeneracy check, [`tools/fpr_control.py`](file:///e:/DEEPFAKE/tools/fpr_control.py) explicitly flags `degenerate = True` whenever the fitted GMM quantile threshold lies outside the target logit support (`threshold > max(logit)` or `< min(logit)`). With pure DFD (`243` pure actor reals, `2,604 / 2,847 = 91.46%` fake), all **`12`** `TPR = 0.0000 / FPR = 0.0000` rows in `fpr_control.csv` (`smsdt_full` seeds 1–3 and `smsdt_st` seeds 1–3 under both `target_gmm` and `target_gmm_anchored` on DFD at $\alpha=0.05$) are now flagged **`degenerate = True`**: on fake-heavy targets ($\approx 91.5\%$ fake on pure DFD), the estimated "real" Gaussian component absorbs wide scores and pushes its upper quantile beyond every logit in the dataset (`0` samples classified positive). On `target_gmm_anchored`, Celeb-DF `gend_official` shifts slightly (`0.0337` FPR / `0.6529` TPR at $\alpha=0.05$) and DFD `gend_official` shifts slightly (`0.0000` FPR / `0.4670` TPR at $\alpha=0.05$), while S-MSDT / `gend_ours` match `target_gmm` within `0.001` on the full datasets.

#### Prior-Stress Sensitivity (`fpr_control_prior_stress.csv`, `624` Rows across $\pi_{\text{fake}} \in \{0.50, 0.66, 0.91\}$ at $\alpha = 0.05$)

| Dataset | Model | Policy | $\pi_{\text{fake}} = 0.50$ (`err / tpr / deg`) | $\pi_{\text{fake}} = 0.66$ (`err / tpr / deg`) | $\pi_{\text{fake}} = 0.91$ (`err / tpr / deg`) |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Celeb-DF-v2** | `gend_official` | `source_quantile` | `0.1073 / 0.8613 / 0.00` | `0.1071 / 0.8647 / 0.00` | `0.1101 / 0.8647 / 0.00` |
| **Celeb-DF-v2** | `gend_official` | `target_gmm` | `0.0518 / 0.7843 / 0.00` | `0.0255 / 0.6214 / 0.00` | `0.0497 / 0.1678 / 0.00` |
| **Celeb-DF-v2** | `gend_ours` | `source_quantile` | `0.0331 / 0.4024 / 0.00` | `0.0332 / 0.4039 / 0.00` | `0.0355 / 0.4039 / 0.00` |
| **Celeb-DF-v2** | `gend_ours` | `target_gmm` | `0.1179 / 0.6718 / 0.00` | `0.1069 / 0.6627 / 0.00` | `0.0939 / 0.6379 / 0.00` |
| **Celeb-DF-v2** | `smsdt_s` | `source_quantile` | `0.0081 / 0.5696 / 0.00` | `0.0083 / 0.5716 / 0.00` | `0.0260 / 0.5716 / 0.00` |
| **Celeb-DF-v2** | `smsdt_s` | `target_gmm` | `0.0208 / 0.6018 / 0.00` | `0.0117 / 0.5002 / 0.00` | `0.0448 / 0.2439 / 0.00` |
| **Celeb-DF-v2** | `smsdt_st` | `source_quantile` | `0.0343 / 0.5525 / 0.00` | `0.0341 / 0.5559 / 0.00` | `0.0419 / 0.5559 / 0.00` |
| **Celeb-DF-v2** | `smsdt_st` | `target_gmm` | `0.1758 / 0.7684 / 0.00` | `0.0738 / 0.5959 / 0.00` | `0.0415 / 0.3007 / 0.00` |
| **Celeb-DF-v2** | `smsdt_full` | `source_quantile` | `0.0717 / 0.6783 / 0.00` | `0.0715 / 0.6804 / 0.00` | `0.0775 / 0.6804 / 0.00` |
| **Celeb-DF-v2** | `smsdt_full` | `target_gmm` | `0.0543 / 0.3033 / 0.00` | `0.0500 / 0.1497 / 0.00` | **`0.0500 / 0.0008 / 0.94`** |
| **DFD** | `smsdt_s` | `source_quantile` | `0.0067 / 0.6791 / 0.00` | `0.0067 / 0.6821 / 0.00` | `0.0067 / 0.6807 / 0.00` |
| **DFD** | `smsdt_s` | `target_gmm` | `0.0266 / 0.5966 / 0.00` | `0.0392 / 0.5273 / 0.00` | `0.0498 / 0.3544 / 0.00` |
| **DFD** | `smsdt_st` | `source_quantile` | `0.0460 / 0.6679 / 0.00` | `0.0460 / 0.6726 / 0.00` | `0.0460 / 0.6722 / 0.00` |
| **DFD** | `smsdt_st` | `target_gmm` | `0.1982 / 0.6781 / 0.00` | `0.0475 / 0.3153 / 0.09` | **`0.0500 / 0.0000 / 1.00`** |
| **DFD** | `smsdt_full` | `source_quantile` | `0.0844 / 0.6574 / 0.00` | `0.0844 / 0.6616 / 0.00` | `0.0844 / 0.6600 / 0.00` |
| **DFD** | `smsdt_full` | `target_gmm` | `0.2141 / 0.5711 / 0.00` | `0.0478 / 0.2112 / 0.13` | **`0.0500 / 0.0000 / 1.00`** |

*(At $\alpha=0.05$ under extreme skew $\pi_{\text{fake}}=0.91$, `smsdt_full` degenerates in `93.7%` of Celeb-DF draws and `100%` of DFD draws, while `smsdt_st` degenerates in `100%` of DFD draws; `target_gmm_anchored` exhibits identical degenerate rates at $\pi_{\text{fake}}=0.91$.)*

#### Pre-Registered Verdicts for H4 & H5:
1. **Hypothesis H4 (FPR Inflation Under Shift):**
   - **H4a:** **`undetermined (fewer than 3 confirmatory seeds)`** (`fpr_control_verdicts.json`). Under dev-split `ffppval` source thresholds and cluster-conservative bounds across Celeb-DF and pure DFD, `smsdt_full` passes `h4_failure` (`fpr_ci_low_cons > 0.10`) on the lone confirmatory seed (`seed3`: Celeb-DF `0.1573`, cons CI `[0.1046, 0.2180]`), whereas exploratory `seed1` (`0.1180 [0.0699, 0.1736]`) and `seed2` (`0.0899 [0.0494, 0.1410]`) have lower bounds below `0.10`, and `smsdt_st` passes on `0/3` seeds on Celeb-DF and DFD. *(When `--resample-source` bootstrap uncertainty in the source quantile threshold is enabled in [`results/tables_resample/`](file:///e:/DEEPFAKE/results/tables_resample), conservative lower bounds widen such that `0 / 52` rows pass `h4_failure`, including confirmatory `smsdt_full` seed 3 -> `False`; thus seed-3 Celeb-DF point/cluster failure is borderline relative to source-calibration finite-sample variance, making confirmatory seeds `4–5` essential.)*
   - **H4b & General Finding:** Paired **`Full - S`** increases Celeb-DF FPR on all 3 seeds (`+0.0674`, `+0.0337`, `+0.0899` at $\alpha=0.05$), with CIs excluding `0` on seeds 1 and 3, while `ST - S` crosses `0` on seeds 2 and 3. Critically, `gend_official` **also** inflates Celeb-DF FPR **`3.15x`** (`0.1573`, cons CI `[0.1050, 0.2180]`) and pure-DFD FPR **`3.87x`** (`0.1934`) under the exact same dev `ffppval` protocol — showing that **cross-domain FPR inflation under frozen source thresholds also occurs for the official GenD checkpoint**, rather than being a flaw confined to S-MSDT alone.
2. **Hypothesis H5 (Label-Free GMM Thresholding) — `falsified (degenerate fits)`:**
   Both `target_gmm` and `target_gmm_anchored` are formally **falsified** across main-table fits (including `12/12` degenerate DFD fits on `smsdt_st` and `smsdt_full` at $\alpha=0.05$) and prior-stress sweeps (`93.7%–100%` degenerate threshold-outside-support collapse at high fake prevalence $\pi_{\text{fake}} = 0.91$).

### 10.4 Eval-Time Readout & Branch Ablations on `smsdt_full` (`results/scores_ablation/`, `results/tables_ablation/`)

To test whether log-sum-exp (LSE) patch pooling or the dual-branch readout causes `smsdt_full`'s tail false-positive rate inflation over `smsdt_s`, all three seeds of `smsdt_full` were re-evaluated with `--readout-ablation {mean_patch,lse_only,att_only}` (D12) and `--branch-ablation {token_only,cls_only}` (D15) across pure `FF++val` (`840` pure FF++ videos) and `Celeb-DF-v2` (`518` videos), and audited via [`tools/fpr_control.py`](file:///e:/DEEPFAKE/tools/fpr_control.py) (`--source ffppval`, $\alpha = 0.05$):

| Ablation Mode | FF++ Val Pure (`840`) Mean AUC | Celeb-DF (`518`) Mean `score_mean` AUC [Seeds 1 / 2 / 3] | Celeb-DF $\alpha=0.05$ Mean FPR | Per-Seed Celeb-DF FPR [Cons 95% CI] | Mean Celeb-DF TPR |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`learned` (baseline `smsdt_full`)** | `0.9561` | `0.8882` `[0.8914 / 0.8960 / 0.8772]` | **`12.17%`** | `11.80% [0.0699, 0.1736]` / `8.99% [0.0494, 0.1410]` / `15.73% [0.1046, 0.2180]` | `68.04%` |
| **`mean_patch` (LSE disabled)** | `0.9559` | `0.8886` `[0.8918 / 0.8966 / 0.8775]` | **`11.99%`** | `11.80% [0.0699, 0.1736]` / `9.55% [0.0543, 0.1476]` / `14.61% [0.0944, 0.2054]` | `68.82%` |
| **`lse_only` ($\beta=1.0$)** | `0.9563` | `0.8885` `[0.8929 / 0.8942 / 0.8785]` | **`12.17%`** | `9.55% [0.0555, 0.1476]` / `12.36% [0.0778, 0.1800]` / `14.61% [0.0966, 0.2054]` | `67.75%` |
| **`att_only` ($\beta=0.0$)** | `0.9551` | `0.8845` `[0.8874 / 0.8921 / 0.8740]` | **`11.99%`** | `12.36% [0.0745, 0.1800]` / `8.99% [0.0494, 0.1410]` / `14.61% [0.0944, 0.2054]` | `68.04%` |
| **`token_only` (D15 branch)** | `0.9551` | `0.8843` `[0.8871 / 0.8917 / 0.8742]` | **`11.80%`** | `11.80% [0.0699, 0.1736]` / `8.99% [0.0494, 0.1410]` / `14.61% [0.0944, 0.2054]` | `67.94%` |
| **`cls_only` (D15 branch)** | `0.8706` | `0.7456` `[0.7935 / 0.7287 / 0.7147]` | **`4.87%`** | `6.74% [0.0339, 0.1141]` / `2.81% [0.0056, 0.0641]` / `5.06% [0.0225, 0.0933]` | `31.18%` |

*(Conclusion: Swapping the token readout at evaluation time does not change FPR (11.99 to 12.17%). Under D15 `--branch-ablation`, `token_only` closely tracks baseline `smsdt_full` across both FF++ Val [`0.9551`] and Celeb-DF [`0.8843` AUC, `11.80%` mean FPR, `67.94%` mean TPR; `0/3` seeds pass `h4_failure` because seed 3's conservative lower bound `0.0944` sits just below `0.10`], whereas `cls_only` degrades substantially in discrimination [`0.8706` FF++ Val AUC, `0.7456` Celeb-DF AUC, `31.18%` mean TPR] while exhibiting `4.87%` mean FPR.)*

---

## 11. Real-Shot Budget Transfer Benchmark (H7 — Development Evidence Only)

> [!IMPORTANT]
> **Development-Set Evidence Only:** All H7 numbers in this section come from Celeb-DF-v2 (`121` real identities, `500` identity-disjoint calibration draws) and exploratory DFD (`18` actor identities), both treated strictly as **development benchmarks** ([`docs/real_shot_benchmark.md`](file:///e:/DEEPFAKE/docs/real_shot_benchmark.md)). They must not be cited as confirmatory validation until evaluated on a held-out dataset with `>= 40` real identities after author sign-off in [`docs/decisions.md`](file:///e:/DEEPFAKE/docs/decisions.md).

Because frozen source quantiles fail on `gend_official` (`15.73%` Celeb-DF FPR) and `smsdt_full` (`15.73%` on seed 3), and label-free target GMMs collapse under prior skew (**H5 `falsified`**), we benchmarked label-efficient **target-real calibration** under budgets of `n` identity-disjoint real videos (`α ∈ {0.05, 0.10}`).

### 11.1 What Failed First: Adaptive Shift Estimation (H7-v1 — `not supported`)
Before finite-sample calibration, we tested an adaptive shift estimator (`adaptive_shift`) that picked among `loc` (mean-shift transfer), `shrink` (variance-shrunk transfer), and `ls` (location-scale transfer). Even though `loc` was selected (`0.037` error at `n=10` vs `0.055` for `shrink` and `0.076` for `ls`), **H7-v1 is `not supported`**:
- It failed to halve source error (`0.043`).
- It severely over-corrected `gend_official` (`FPR = 0.008`, `TPR = 0.29` vs `0.75` oracle; `loc_ucb` dropped `TPR` to `0.08`).
- It under-corrected `smsdt_full` (`P(FPR > 2α) = 0.31`).

### 11.2 Split-Conformal Target-Real Calibration (H7-v2 — Development Support on Celeb-DF)
In contrast, split-conformal quantile calibration (`conformal`) on `n = 20` target real videos (`k = ceil((1 - α)(n + 1))` order statistic; abstains cleanly when `(1 - α)(n + 1) > n`) controls `P(FPR > 2α)` on Celeb-DF while preserving high oracle-relative TPR:

| Model (Celeb-DF `n=20, α=0.05`) | `source_quantile` FPR (`P(>2α)`) | `target_gmm` FPR (`P(>2α)`) | `naive_quantile` FPR | `conformal` FPR (`P(>2α)`) | `conformal` / Oracle TPR |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`gend_official`** | `0.157 (1.00)` | `0.022 (0.00)` | `0.082` | **`0.042 (0.08)`** | `0.585 / 0.747` |
| **`gend_ours`** | `0.017 (0.00)` | `0.157 (1.00)` | `0.078` | **`0.042 (0.10)`** | `0.441 / 0.532` |
| **`smsdt_s`** | `0.058 (0.00)` | `0.041 (0.00)` | `0.088` | **`0.048 (0.14)`** | `0.485 / 0.547` |
| **`smsdt_st`** | `0.084 (0.12)` | `0.108 (0.42)` | `0.085` | **`0.043 (0.11)`** | `0.415 / 0.490` |
| **`smsdt_full`** | `0.121 (0.68)` | `0.000 (0.00)` | `0.089` | **`0.047 (0.12)`** | `0.486 / 0.552` |

- **Operational envelope (`Celeb-DF-v2`):** Across individual runs at `n=20, α=0.05`, conformal mean FPR spans `0.039–0.054` while retaining `0.78–0.96` of oracle TPR. At `α=0.10, n=20`, conformal achieves `0.081–0.092` FPR with `P(FPR > 2α) = 0.03–0.07`. By comparison, `naive_quantile` violates the target budget (`0.112` mean FPR at `n=10`, `0.085` at `n=20` across `0.077–0.096`, with `P(FPR > 2α) = 0.46` / `0.32`), and `np_umbrella` abstains below `n=59` identities (`0.017` FPR, `0.34` TPR at `n=60`).
- **DFD (`18` actor identities, exploratory `n <= 10`):** With only `18` actor clusters (`<= 10` calibration draws), `conformal` abstains at both `n=5` and `n=10`, so all H7-v2 verdicts on DFD are **`undetermined`**.
- **Identity-weighted FPR and real-source strata (D18, exploratory, 2026-11-12):** Celeb-DF's `121` real clusters are `51` celebrities (`108` Celeb-real videos) plus `70` YouTube-real clips with no identity label. Averaging the flag rate per held-out identity instead of per video, conformal at `n=20, α=0.05` gives `0.043–0.052` FPR across the `13` runs (`P(FPR_idw > 2α) = 0.12–0.16`). Restricting the reals to one source at a time (`--real-strata`, `results/tables/real_shot_strata_dev_celebdf.md`), conformal still gives `0.041–0.052` FPR (identity-weighted `0.044–0.051`) on Celeb-real (`51` identities) and YouTube-real (`70`), but the TPR/oracle ratio drops to `0.73–0.93`, below the `0.75` H7b-v2 bar for some runs. The zero-shot source threshold is worse on YouTube-real for `gend_official` (`0.187` vs `0.140`).
- **Note on the rule at `n = 20`:** at `α = 0.05`, `k = ceil(0.95 × 21) = 20 = n`, so the conformal threshold is the **maximum** of the 20 calibration reals. The guarantee is on the expected FPR over calibration draws; any single draw can overshoot (hence `P(FPR > 2α)` of `0.08–0.16`). `np_umbrella` is the high-probability version and needs `n ≥ 59`.
- **Confirmation Plan:** Pre-register `(unseen dataset with >= 40 real identities, budget n, α, confirmatory seed list)` alongside author sign-off in [`docs/decisions.md`](file:///e:/DEEPFAKE/docs/decisions.md) Sections 5 & 6 before running the confirmatory H7 evaluation.

---

## 12. Complete Artifact Inventory & Study Status Checklist

### 12.1 Saved Result Artifacts (`feat/eval-and-benchmarks`)
- **Per-Video Score CSVs ([`results/scores/`](file:///e:/DEEPFAKE/results/scores), `52` files):**
  - `gend_ours_seed{1,2,3}_{ffpp,ffppval,celebdf,dfd}.csv` (`12` files)
  - `smsdt_s_seed{1,2,3}_{ffpp,ffppval,celebdf,dfd}.csv` (`12` files)
  - `smsdt_st_seed{1,2,3}_{ffpp,ffppval,celebdf,dfd}.csv` (`12` files)
  - `smsdt_full_seed{1,2,3}_{ffpp,ffppval,celebdf,dfd}.csv` (`12` files)
  - `gend_official_seed0_{ffpp,ffppval,celebdf,dfd}.csv` (`4` external baseline files)
- **Platt Calibration JSONs ([`results/calib/`](file:///e:/DEEPFAKE/results/calib), `13` files):**
  - `gend_official_seed0.json` + `{gend_ours,smsdt_s,smsdt_st,smsdt_full}_seed{1,2,3}.json` (all fit on pure `ffppval`)
- **Readout & Branch-Ablation Scores & Tables ([`results/scores_ablation/`](file:///e:/DEEPFAKE/results/scores_ablation), `30` CSVs; [`results/tables_ablation/`](file:///e:/DEEPFAKE/results/tables_ablation), `30` files):**
  - `{mean_patch,lse_only,att_only,token_only,cls_only}/smsdt_full_seed{1,2,3}_{ffppval,celebdf}.csv` and corresponding `fpr_control*` tables
- **Source-Resample Sensitivity Tables ([`results/tables_resample/`](file:///e:/DEEPFAKE/results/tables_resample), `6` files):**
  - `fpr_control.{csv,md}`, `fpr_control_h4.csv`, `fpr_control_h4b.csv`, `fpr_control_prior_stress.csv`, `fpr_control_verdicts.json` (`--resample-source` sensitivity check)
- **Aggregated Summary Tables & JSON ([`results/tables/`](file:///e:/DEEPFAKE/results/tables), `26` files):**
  - [`hypothesis_h1_stats.{csv,md}`](file:///e:/DEEPFAKE/results/tables/hypothesis_h1_stats.csv), [`hypothesis_h1_meta.json`](file:///e:/DEEPFAKE/results/tables/hypothesis_h1_meta.json), [`threshold_transfer_audit.{csv,md}`](file:///e:/DEEPFAKE/results/tables/threshold_transfer_audit.csv), [`threshold_transfer_details.json`](file:///e:/DEEPFAKE/results/tables/threshold_transfer_details.json), [`prior_shift_sweep.{csv,md}`](file:///e:/DEEPFAKE/results/tables/prior_shift_sweep.csv)
  - [`tta_summary.{csv,md}`](file:///e:/DEEPFAKE/results/tables/tta_summary.csv), [`robustness_summary.{csv,md}`](file:///e:/DEEPFAKE/results/tables/robustness_summary.csv), [`experiment_matrix.{csv,md}`](file:///e:/DEEPFAKE/results/tables/experiment_matrix.csv)
  - [`fpr_control.{csv,md}`](file:///e:/DEEPFAKE/results/tables/fpr_control.csv), [`fpr_control_h4.csv`](file:///e:/DEEPFAKE/results/tables/fpr_control_h4.csv), [`fpr_control_h4b.csv`](file:///e:/DEEPFAKE/results/tables/fpr_control_h4b.csv), [`fpr_control_prior_stress.csv`](file:///e:/DEEPFAKE/results/tables/fpr_control_prior_stress.csv), [`fpr_control_verdicts.json`](file:///e:/DEEPFAKE/results/tables/fpr_control_verdicts.json), plus H7 development benchmark tables in [`results/tables/`](file:///e:/DEEPFAKE/results/tables).

### 12.2 Study Status Checklist
- [x] Multi-seed model training (12 models across seeds 1–3; seeds 4–5 confirmatory training pending per [`docs/decisions.md`](file:///e:/DEEPFAKE/docs/decisions.md)).
- [x] D14 FF++ Real Contamination Fix (`EvalClipDataset` isolates `dfd_*` actor reals in FF++ `test`/`val`: `840` pure FF++ videos, `243` pure DFD reals across `18` actor clusters) + full refresh of `52` score CSVs & `13` dev-split Platt JSONs.
- [x] **Gate G1 Resolved:** Official GenD (`yermandy/GenD_CLIP_L_14`) evaluated across FF++ (`0.9452`), pure DFD (`0.8840`, `19.34%` dev `source_quantile` FPR), and Celeb-DF v2 (`0.9327`, `15.73%` dev `source_quantile` FPR = `3.15x`).
- [x] **Hypothesis H1:** Re-run with `10,000` resamples, Bonferroni ($m=2$), pure `273`-cluster DFD negatives -> **`undetermined (underpowered)`**; extra modules show no AUC gain.
- [x] **Hypothesis H2:** Re-run with dev-split Platt calibration (`tau_cal`) and D13 `121`-cluster Celeb-DF CIs -> **`undetermined`** (`smsdt_s` seed 1 $g=0.0984$, CI `[0.0796, 0.1283]`).
- [x] **Hypothesis H3a:** Re-run with calibrated posteriors -> **`falsified`** on HTER; exploratory Bayes-decision accuracy gains documented at extreme priors.
- [x] **Hypotheses H4 & H5:** Re-run with `--source ffppval --confirmatory-seeds 3 4 5`, pure DFD (`243` reals / `12` degenerate GMM fits), conservative cluster CIs, and `--resample-source` sensitivity -> **H4a `undetermined`** (`Full` supported on seed 3 without `--resample-source`, `0/52` with `--resample-source`; `ST` not established; **H4b** positive on `Full - S`), **H5 `falsified (degenerate fits)`**.
- [x] **Hypothesis H6:** **`falsified`** on `smsdt_full_seed1` Celeb-DF-v2 (`0.5409` online consistency collapse; `-0.0003` batch-episodic; legacy entropy pre-reset caveat documented).
- [x] **Eval-Time Readout & Branch Ablations (D12 & D15):** Tested `mean_patch`, `lse_only`, `att_only`, `token_only`, and `cls_only` across seeds 1–3; swapping eval-time token readout leaves FPR unchanged (`11.99%–12.17%`).
- [x] **Real-Shot Calibration (H7 — Development Only):** Benchmarked `adaptive_shift` (H7-v1 `not supported`) and split-conformal `n=20` target-real calibration on Celeb-DF dev (`0.042–0.048` FPR, `P(FPR > 2α) = 0.08–0.14`; DFD `undetermined`).
- [x] **`research_v2.md` Overclaiming Scrubbed:** All items in [`docs/research_v2_corrections.md`](file:///e:/DEEPFAKE/docs/research_v2_corrections.md) (including Addendum 2026-11-11) applied.
- [ ] **Final Confirmatory & Manuscript Alignment:** Author sign-off in [`docs/decisions.md`](file:///e:/DEEPFAKE/docs/decisions.md) Sections 5 & 6, confirmatory seeds `4–5` + unseen confirmatory dataset (`>= 40` real identities), and corresponding alignment of `paper/`.

