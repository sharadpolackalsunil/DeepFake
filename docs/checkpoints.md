# Master Model Checkpoint & Weights Registry

This registry tracks the exact storage location, architecture specifications, parameter counts, configuration files, and evaluation outputs for all models in the S-MSDT and TMLR research study.

---

## 1. Quick Reference: Model Checkpoint Locations

| Model | Architecture Class | Total Params | Trainable Params | Input Shape | Checkpoint Folder | Best Weights File | Training Config |
| :--- | :--- | :---: | :---: | :---: | :--- | :--- | :--- |
| **GenD (Ours)** | `GenDModel` | 303.18M | 103,425 (0.034%) | $(B, 1, 3, 224, 224)$ | `outputs/checkpoints_gend_ours/seed{seed}/` | `best_ema.pt` / `last.pt` | [`configs/train_gend.yaml`](file:///e:/DEEPFAKE/configs/train_gend.yaml) |
| **S-MSDT (S)** | `SMSDTv2` ($L=0$) | 303.18M | 103,425 (0.034%) | $(B, 1, 3, 224, 224)$ | `outputs/checkpoints_smsdt_s/seed{seed}/` | `best_ema.pt` / `last.pt` | [`configs/train_s.yaml`](file:///e:/DEEPFAKE/configs/train_s.yaml) |
| **S-MSDT (ST)** | `SMSDTv2` ($L=2$) | 311.54M | 8,474,881 (2.72%) | $(B, 8, 3, 224, 224)$ | `outputs/checkpoints_st/seed{seed}/` | `best_ema.pt` / `last.pt` | [`configs/train_st.yaml`](file:///e:/DEEPFAKE/configs/train_st.yaml) |
| **S-MSDTv3 (Full)** | `SMSDTv3` | 312.18M | 9,103,363 (2.92%) | $(B, 8, 3, 224, 224)$ | `outputs/checkpoints_v3_stage_c/seed{seed}/` | `best_ema.pt` / `last.pt` | [`configs/train_full.yaml`](file:///e:/DEEPFAKE/configs/train_full.yaml) |
| **S-MSDT v1 (Phase 1)** | `SMSDT` | 26.68M | 26,678,659 (100%) | $(B, 8, 3, 224, 224)$ | `outputs/checkpoints/` | `best_ema.pt` | [`configs/train.yaml`](file:///e:/DEEPFAKE/configs/train.yaml) |

---

## 2. Detailed Model Specifications

### Model 1: `gend_ours` (GenD Baseline Reimplementation)
- **Architecture:** CLIP ViT-L/14 with LayerNorm-only Parameter-Efficient Fine-Tuning (PEFT) + $L_2$-normalized linear probe head + Slerp spherical interpolation augmentation.
- **Loss:** $\mathcal{L}_{\text{BCE}} + 0.5 \mathcal{L}_{\text{align}} + 0.5 \mathcal{L}_{\text{unif}}$.
- **Input:** $T=1$ frame per clip ($224 \times 224$).
- **Weights Stored At:**
  - Seed 1: `outputs/checkpoints_gend_ours/seed1/best_ema.pt` (and `last.pt`)
  - Seed 2: `outputs/checkpoints_gend_ours/seed2/best_ema.pt`
  - Seed 3: `outputs/checkpoints_gend_ours/seed3/best_ema.pt`
- **Output Scores:**
  - Celeb-DF v2: `results/scores/gend_ours_seed{seed}_celebdf.csv`
  - DFD: `results/scores/gend_ours_seed{seed}_dfd.csv`
  - Calibration: `results/calib/gend_ours_seed{seed}.json`

### Model 2: `smsdt_s` (S-MSDT Spatial-Only)
- **Architecture:** CLIP ViT-L/14 with LayerNorm-only PEFT, linear token projection, 0 temporal blocks ($L=0$), dual readout head (`EvidenceReadout` + global CLS head).
- **Input:** $T=1$ frame per clip ($224 \times 224$).
- **Weights Stored At:**
  - Seed 1: `outputs/checkpoints_smsdt_s/seed1/best_ema.pt` (and `last.pt`)
  - Seed 2: `outputs/checkpoints_smsdt_s/seed2/best_ema.pt`
  - Seed 3: `outputs/checkpoints_smsdt_s/seed3/best_ema.pt`
- **Output Scores:**
  - Celeb-DF v2: `results/scores/smsdt_s_seed{seed}_celebdf.csv`
  - DFD: `results/scores/smsdt_s_seed{seed}_dfd.csv`
  - Calibration: `results/calib/smsdt_s_seed{seed}.json`

### Model 3: `smsdt_st` (S-MSDT Spatio-Temporal)
- **Architecture:** CLIP ViT-L/14 with LayerNorm-only PEFT + 2 Spatio-Temporal transformer blocks ($L=2$, $H=6$, dim=384) with factorized spatial attention and TemporalDiff velocity ($\Delta_t$).
- **Input:** $T=8$ frames per clip ($224 \times 224$).
- **Weights Stored At:**
  - Seed 1: `outputs/checkpoints_st/seed1/best_ema.pt`
  - Seed 2: `outputs/checkpoints_st/seed2/best_ema.pt`
  - Seed 3: `outputs/checkpoints_st/seed3/best_ema.pt`
- **Output Scores:**
  - Celeb-DF v2: `results/scores/smsdt_st_seed{seed}_celebdf.csv`
  - DFD: `results/scores/smsdt_st_seed{seed}_dfd.csv`
  - Calibration: `results/calib/smsdt_st_seed{seed}.json`

### Model 4: `smsdt_full` (S-MSDTv3 Foundation Forensic Model)
- **Architecture:** Full proposed architecture incorporating:
  1. Dual-stream raw-pixel `ResidualStream` (3 SRM filters + 2-level 2D Haar DWT via zero-init gate).
  2. 2 Spatio-Temporal transformer blocks ($L=2$, $H=6$, dim=384) with TemporalDiff.
  3. Dynamic SCRFD 106-point landmark perioral mouth spotlight prior (indices 52..71).
  4. Continuous Log-Sum-Exp anomaly pooling readout (`EvidenceReadoutV3`).
- **Input:** $T=8$ frames per clip ($224 \times 224$).
- **Weights Stored At:**
  - Seed 1: `outputs/checkpoints_v3_stage_c/seed1/best_ema.pt`
  - Seed 2: `outputs/checkpoints_v3_stage_c/seed2/best_ema.pt`
  - Seed 3: `outputs/checkpoints_v3_stage_c/seed3/best_ema.pt`
- **Output Scores:**
  - Celeb-DF v2: `results/scores/smsdt_full_seed{seed}_celebdf.csv`
  - DFD: `results/scores/smsdt_full_seed{seed}_dfd.csv`
  - Calibration: `results/calib/smsdt_full_seed{seed}.json`

---

## 3. Training & Evaluation Run Tracker

| Model | Seed | Training Status | Epochs Completed | Best Val AUC | Checkpoint Size | Celeb-DF AUC | DFD AUC | Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `gend_ours` | 1 | ✅ Completed | 30 / 30 | 0.9047 (test) | ~1.2 GB | 0.8321 | 0.8776 | Reference GenD anchor: FF++ Test AUC 0.9047, Celeb-DF 0.8321, DFD 0.8776 |
| `gend_ours` | 2 | ✅ Completed | 30 / 30 | 0.9040 (test) | ~1.2 GB | 0.8273 | 0.8820 | Seed 2: FF++ Test AUC 0.9040, Celeb-DF 0.8273, DFD 0.8820 |
| `gend_ours` | 3 | ✅ Completed | 30 / 30 | 0.9045 (test) | ~1.2 GB | 0.8288 | 0.8758 | Seed 3: FF++ Test AUC 0.9045, Celeb-DF 0.8288, DFD 0.8758 |
| `smsdt_s` | 1 | ✅ Completed | 30 / 30 | 0.9514 | ~1.2 GB | 0.8837 | 0.8742 | Controlled single-frame baseline: FF++ Test AUC 0.9587, Celeb-DF 0.8837, DFD 0.8742 |
| `smsdt_s` | 2 | ✅ Completed | 30 / 30 | 0.9613 | ~1.2 GB | 0.8825 | 0.8807 | Seed 2: FF++ Test AUC 0.9613, Celeb-DF 0.8825, DFD 0.8807 |
| `smsdt_s` | 3 | ✅ Completed | 30 / 30 | 0.9681 | ~1.2 GB | 0.8815 | 0.8974 | Seed 3: FF++ Test AUC 0.9681, Celeb-DF 0.8815, DFD 0.8974 |
| `smsdt_st` | 1 | ✅ Completed | 30 / 30 | 0.9584 | ~1.2 GB | 0.8706 | 0.8682 | Spatio-temporal baseline: FF++ Test AUC 0.9604, Celeb-DF 0.8706, DFD 0.8682 |
| `smsdt_st` | 2 | ✅ Completed | 30 / 30 | 0.9664 | ~1.2 GB | 0.8779 | 0.8748 | Seed 2: FF++ Test AUC 0.9664, Celeb-DF 0.8779, DFD 0.8748 |
| `smsdt_st` | 3 | ✅ Completed | 30 / 30 | 0.9618 | ~1.2 GB | 0.8628 | 0.8908 | Seed 3: FF++ Test AUC 0.9618, Celeb-DF 0.8628, DFD 0.8908 |
| `smsdt_full`| 1 | ✅ Completed | 30 / 30 | 0.9591 | ~1.2 GB | 0.8915 | 0.8604 | Flagship model (SRM+DWT+LSE): FF++ Test AUC 0.9631, Celeb-DF 0.8915, DFD 0.8604 |
| `smsdt_full`| 2 | ✅ Completed | 30 / 30 | 0.9621 | ~1.2 GB | 0.8960 | 0.8689 | Seed 2: FF++ Test AUC 0.9621, Celeb-DF 0.8960, DFD 0.8689 (Peak cross-dataset) |
| `smsdt_full`| 3 | ✅ Completed | 30 / 30 | 0.9625 | ~1.2 GB | 0.8772 | 0.8655 | Seed 3: FF++ Test AUC 0.9625, Celeb-DF 0.8772, DFD 0.8655 |

---

## 4. Checkpoint File Format Standard

Every saved checkpoint (`best_ema.pt` and `last.pt`) is an atomic Python dictionary containing:
```python
{
    "epoch": int,                  # Completed epoch number
    "val_auc": float,              # Validation AUC on FF++ c23 val split
    "model": dict,                 # model.state_dict()
    "model_ema": dict,             # model_ema.state_dict() (Exponential Moving Average weights)
    "optimizer": dict,             # optimizer.state_dict()
    "scheduler": dict,             # scheduler.state_dict()
    "cfg": dict,                   # Full YAML configuration used for training
}
```

This guarantees bit-identical reproduction and allows `smsdt/eval_cross.py` to auto-detect architecture, temporal window size ($T$), backbone type, and normalizations directly from the file itself.

