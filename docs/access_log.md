# Dataset Access Log

This log tracks external dataset access requests, contacts, verification dates, and inclusion status for the TMLR study.

| Dataset | Role in Study | Status | Request Date | Verification Date | Contact / Source | Notes |
|:---|:---|:---|:---|:---|:---|:---|
| **FF++ c23** | Train & Dev | **On Disk** | Pre-existing | 2026-11-09 | Local preprocessed store (`data/store`) | 720 train / 140 val / 140 test pairs; 106 mesh landmarks; 1.3× loose crops |
| **DFD** | Dev & Seen Test | **On Disk** | Pre-existing | 2026-11-09 | Local store | 120 val / 243 test reals; actor-disjoint splits verified |
| **Celeb-DF v2** | Seen Test | **On Disk** | Pre-existing | 2026-11-09 | Local preprocessed store (`data/store_celebdf`) | Official test split (178 real, 340 fake); celebrity IDs logged |
| **FF++ c40** | Robustness (Exploratory) | **Pending Download** | 2026-11-09 | Pending | Official FaceForensics++ repo | Heavy compression evaluation |
| **DFDC test subset** | Confirmatory | **Access Requested** | 2026-11-09 | Pending | Kaggle / Deepfake Detection Challenge | Target: test subset with original-video grouping metadata |
| **WildDeepfake** | Confirmatory | **Access Requested** | 2026-11-09 | Pending | Official WildDeepfake repo | Real-world in-the-wild web clips; video IDs logged |
| **FFIW** | Confirmatory Fallback 1 | **Access Requested** | 2026-11-09 | Pending | Face Forensics in the Wild repo | Fallback if DFDC or WildDeepfake access fails |
| **DF40 subset** | Confirmatory Fallback 2 | **Access Requested** | 2026-11-09 | Pending | DeepFake40 repo | Secondary fallback |

---

### Access Rules (Frozen Protocol)
1. Any dataset not obtained and verified by end of **Day 7** is dropped and stated as such in the paper.
2. At least 2 confirmatory datasets ($K_c \ge 2$) are required for confirmatory claims.
3. If $K_c < 2$, all H1–H3 claims are strictly designated as **exploratory**.

