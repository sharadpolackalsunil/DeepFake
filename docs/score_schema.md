# Score File Schema (CSV Specification)

Path structure:
`results/scores/{model}_{seed}_{date}/{dataset}.csv`

Example:
`results/scores/smsdt_s_seed1_20261109/celebdf.csv`

---

## Required Header & Columns

| Column Name | Type | Description | Allowed Values / Examples |
|:---|:---|:---|:---|
| `video_id` | string | Unique video identifier matching dataset split | e.g. `id0_0000`, `real_043`, `Deepfakes_001_002` |
| `dataset` | string | Dataset name | `ffpp_c23`, `dfd`, `celebdf`, `dfdc`, `wdf`, `ffiw` |
| `split` | string | Split partition | `val`, `test` |
| `label` | integer | Ground truth binary label | `0` (real / authentic), `1` (fake / manipulated) |
| `real_source` | string | Specific provenance for authentic negatives | `ffpp_youtube`, `dfd_actor`, `celebdf_real`, `dfdc_real`, `wdf_real`, `other`; empty string `""` for fakes |
| `manipulation` | string | Specific manipulation method for positive fakes | `Deepfakes`, `Face2Face`, `FaceSwap`, `NeuralTextures`, `FaceShifter`, `other`; empty string `""` for reals |
| `cluster_id` | string | Identity or video group for cluster bootstrap | e.g., target actor/celebrity ID, original video ID group |
| `n_clips` | integer | Number of evaluated clips in this video | e.g., `8` |
| `logit_mean` | float | Pre-sigmoid mean logit across all clips | e.g., `-1.4258`, `3.1092` |
| `score_mean` | float | Sigmoid probability computed on `logit_mean` | `[0.0, 1.0]` (headline metric aggregation) |
| `score_topk50` | float | Top-50% most anomalous clip score mean | `[0.0, 1.0]` (exploratory comparison) |

---

## Evaluation Rules
1. **Headline aggregation:** Always `score_mean` across clips.
2. **Pure Negatives:**
   - FF++ per-method evaluations compare fakes against `real_source == 'ffpp_youtube'` only ($N=140$).
   - DFD evaluations compare DFD fakes against `real_source == 'dfd_actor'` only ($N=243$).
   - Mixed-negative numbers (combining multiple sources) are clearly labeled as `mixed`.


---

## Additional Files (2026-11-10)

The audit tools look up files by name. Flat layout under `results/scores/`:

| File | Contents | Used by |
|:---|:---|:---|
| `{model}_seed{n}_{dataset}.csv` | Test-split scores (`dataset` in `ffpp`, `dfd`, `celebdf`, `dfdc`, `wdf`, `ffiw`) | all tools |
| `{model}_seed{n}_ffppval.csv` | **Dev-split** scores of the same run (the split used for `--fit-calib`) | `tools/fpr_control.py` (source thresholds) |
| `gend_official_seed0_{dataset}.csv` | Official GenD weights through this pipeline (`--ckpt hf:yermandy/GenD_CLIP_L_14`); `seed0` = no training seed | `tools/stats.py` (G1 rows) |
| `results/calib/{model}_seed{n}.json` | Frozen Platt + Youden calibration `{a, b, tau}`, tau on the CALIBRATED scale | `tools/threshold_transfer.py`, `tools/prior_sweep.py` |

`logit_mean` is required for every file: frozen thresholds and FPR control work on logits.
