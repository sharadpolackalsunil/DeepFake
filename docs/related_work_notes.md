# Related Work Notes & Gate G0 Novelty Matrix

This document tracks literature analysis and supports the **Gate G0 Novelty Kill-Switch** decision.

---

## 1. Gate G0 Evaluation Matrix

Criteria evaluated across recent (2024–2026) literature:
1. **Video Faces:** Evaluates on multi-frame video deepfakes with inter-frame temporal context (not purely static single images).
2. **CLIP / LN-only Tuning:** Specifically investigates LayerNorm-only parameter-efficient fine-tuning on foundation ViTs.
3. **Per-Source Real/Fake Decomposition:** Distinguishes FPR and FNR failure modes across individual real camera sources and fake manipulation algorithms.
4. **Label-Free Adaptation with Prior Failure Analysis:** Evaluates test-time adaptation (TTA) and explicit class-prior shift correction while auditing collapse mechanisms.

| Paper / Citation | Key Focus & Modality | Video Faces? | CLIP / LN-only? | Per-Source Decomp? | Label-Free TTA w/ Prior Failure? | Notes & Scope |
|:---|:---|:---:|:---:|:---:|:---:|:---|
| **GenD / LNCLIP-DF** (arXiv:2508.06248 v3) | Generalizable deepfake detection via LN tuning + Slerp | ❌ (Frame-level) | ✅ Yes | ❌ No | ❌ No | Evaluates single frames with multi-dataset cross-evaluation. Does not analyze threshold transfer or TTA. |
| **T²A** (IJCAI 2025, arXiv:2505.18787) | Online test-time adaptation for deepfake detection | ⚠️ Partial (CNN image) | ❌ No (CNN backbones) | ❌ No | ⚠️ Partial (Entropy + neg learning) | Tested on Xception, EfficientNet-B4, F3-Net, RECCE. Addresses fake-dominant shift, but no ViT LN-only adaptation. |
| **CDTS** (arXiv:2606.29484) | Calibrator drift under deepfake transfer | ⚠️ (Images / Gen) | ❌ No | ❌ No | ❌ No | Examines calibration (Platt, isotonic) across 21 generators; observes ECE increase as competence drops. Does not study video temporal modeling or TTA repair. |
| **Speech Threshold Shift** (arXiv:2606.21584) | Threshold transfer failure & pseudo-label sensitivity | ❌ No (Speech audio) | ❌ No | ⚠️ (Audio sources) | ⚠️ (Pseudo-labeling) | Demonstrates severe HTER degradation under transfer. Pure audio modality; serves as conceptual inspiration for face video forensics. |
| **FCG / Plug-and-Play / VLAForge** (2024–2026) | Frequency / forensic feature guidance | ⚠️ Frame/Video | ❌ No | ❌ No | ❌ No | Benchmark references for cross-dataset AUC tables. |

---

## 2. Gate G0 Ruling (Day 3 Check)
- **Finding:** No single paper in the literature covers all four dimensions simultaneously.
- **Decision:**
  - **RQ1 (Architectural Efficacy):** Matched-baseline study on whether temporal blocks and forensic streams yield statistically sound gains over LN-tuned CLIP.
  - **RQ2 (Threshold Transfer):** First comprehensive audit of validation-frozen operating point transfer on CLIP video deepfake detectors with explicit per-source FPR/FNR decomposition.
  - **RQ3 (Label-Free Repair):** Systematic evaluation of entropy-minimization TTA and EM prior-shift correction with explicit collapse characterization.
