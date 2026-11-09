# TMLR Submission Checklist & Author Logistics

## 1. Author Information & Order
**Author criteria:** Substantial contribution to conception/design, acquisition/analysis of data, drafting or critical revision, and final approval with accountability.

| Author | Role / Contribution | OpenReview Profile | TMLR Quota Budget | Status |
|:---|:---|:---|:---|:---|
| **Sharad** (Sh) | Preprocessing, baseline reproduction, ST/Full training, TTA implementation | Complete | Confirmed available | Verified |
| **Aditya Raj** (AR) | Evaluation harness, statistical testing, threshold audit, tables & figures | Complete | Confirmed available | Verified |
| **Aditya Kumar** (AK) | Manuscript drafting, literature review, experiment synthesis, TMLR packaging | Complete | Confirmed available | Verified |

*Note: The author list and order is final and cannot be modified after submission on OpenReview.*

---

## 2. TMLR Format & Submission Requirements
- [x] Unmodified TMLR LaTeX style file (`jmlr.org/tmlr`)
- [ ] Main text word/page floor verified (maximum 12 pages excluding references and appendix)
- [ ] Appendix placed strictly after references
- [ ] First-page footnote disclosing assistive LLM usage included
- [ ] Complete double-blind anonymization (no author names, institutions, non-anonymous repo URLs, or self-revealing citations)
- [ ] Supplementary material archive (`paper/supp/supplement.zip`) < 100 MB, containing configs, splits, per-video score CSVs, calibration JSONs, and reproduction code
- [ ] 3 to 5 candidate Action Editors identified with computer vision / deepfake forensics expertise and zero conflicts of interest:
  - Candidate 1: Expert in visual forensics / face manipulation
  - Candidate 2: Expert in test-time adaptation / domain generalization
  - Candidate 3: Expert in foundation model adaptation / parameter-efficient fine-tuning
- [ ] Human subjects / IRB statement and dataset license terms prepared for OpenReview form
- [ ] No parallel submission to any other archival conference or journal
