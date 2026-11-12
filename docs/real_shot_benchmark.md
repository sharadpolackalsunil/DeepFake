# Real-shot budget transfer: a benchmark for holding a false-alarm budget on a new domain

Status: development results on Celeb-DF and DFD (exploratory for this question, since
both have been inspected). Confirmation requires a new external dataset; see section 7.
Tool: `tools/real_shot.py`. Pre-registration: `docs/decisions.md`, section 6.

## 1. The question

A deployer fixes a false-alarm budget alpha: no more than 5% of authentic videos may be
flagged. Our audits show that thresholds do not carry this budget to a new domain.
Every zero-shot option fails for at least one detector:

| Zero-shot policy (Celeb-DF, alpha = 0.05) | Detector it fails for | Achieved FPR |
|---|---|---|
| Source dev quantile (`source_quantile`) | official GenD | 15.7% (3.15x) |
| | `smsdt_full` | 12.2% mean (2.4x) |
| Label-free mixture (`target_gmm`) | `gend_ours` | 15.7% |
| | `smsdt_st` | 10.9% |

Label-free TTA and EM prior correction fail as well (H3a, H5 and H6 are falsified).
A related audit of speech deepfake detectors reached the same conclusion with labels
withheld: no unlabelled correction was safe across targets (arXiv 2606.21584).

Labelling fakes from a new domain is expensive. Collecting a few authentic videos from
it usually is not: the deployer has its own recordings. That makes the practical
question: **how many authentic identities from the deployment domain are needed to hold
the budget, and with which estimator?**

## 2. Estimators compared

Let `n` be the number of authentic target identities, with one video drawn per identity.
Flag a video if its logit exceeds the threshold.

| Policy | Uses | Guarantee |
|---|---|---|
| `source_quantile` | source dev reals only | none under shift |
| `target_gmm` | unlabelled target set | none (degenerate under fake-heavy targets) |
| `naive_quantile` | (1 - alpha) empirical quantile of n target reals | none; optimistic for small n |
| `conformal` | k-th smallest of n target reals, k = ceil((n+1)(1-alpha)) | E[FPR] <= alpha if identities are exchangeable; needs n >= 19 at alpha = 0.05, n >= 9 at 0.10 |
| `np_umbrella` | smallest k with P(Bin(n, 1-alpha) >= k) <= delta (Tong, Feng & Li 2018) | P(FPR > alpha) <= delta; needs n >= 59 at alpha = delta = 0.05 |
| `loc/ls/shrink_transfer` | source real-score shape, with target median (and MAD) estimated from n reals | exact only if the real-class shift is a location (scale) map |
| `*_ucb` | (1 - delta) bootstrap upper quantile of the transfer threshold | approximate, under the same assumption |
| `oracle` | (1 - alpha) quantile of the evaluation reals | reference only (uses test labels) |

**Why one video per identity.** Videos of the same person are correlated. If you draw
calibration reals per identity and evaluate on disjoint identities, the conformal
exchangeability argument applies at the identity level: a new identity's video is
exchangeable with the n calibration videos. Drawing several videos per identity would
silently break the guarantee.

**Why the source-shape transfer was tried.** Conformal needs at least 19 identities. A
location-scale model could in principle work with 5 to 10. Section 4 shows that it does
not.

## 3. Protocol

For each run, target dataset, alpha and n, the tool repeats the following R = 500
times. All settings were fixed before the run.

1. Draw n real identity clusters without replacement and take one random video from each.
2. Evaluate on all reals of the remaining identities plus all fakes. At least 20 identities
   are held out (8 on DFD, which has only 18).
3. Record FPR and TPR for every policy on the same split.

Report mean FPR (video-weighted and identity-weighted), mean |FPR - alpha|,
P(FPR_eval > 2 alpha), mean TPR, the abstain rate, and the oracle TPR on the same splits.

`P(FPR_eval > 2 alpha)` is the practical failure rate. The evaluation FPR is itself an
estimate from about 100 to 160 held-out reals. Even a threshold whose true FPR is exactly
alpha shows P(FPR_eval > alpha) near 0.5. The synthetic tests check the guarantees
against the population FPR instead.

Identity clusters are resolved with the `tools/stats.py` rule. Celeb-DF has 121 real
identities and DFD has 18 real actors, the latter after the D14 fix that removes the 140
FF++ reals that were mislabelled as DFD.

## 4. Development results (Celeb-DF, 13 runs, alpha = 0.05)

These are means over seeds; `results/tables/real_shot_curve_dev_celebdf.csv` has every
row. Celeb-DF has been inspected before, so these numbers are development evidence only.

**n = 20 real identities**

| Detector | Source FPR (P > 2a) | GMM FPR (P > 2a) | Naive FPR | **Conformal FPR (P > 2a)** | Conformal TPR / oracle TPR |
|---|---|---|---|---|---|
| official GenD | 0.157 (1.00) | 0.022 (0.00) | 0.082 | **0.042 (0.08)** | 0.585 / 0.747 |
| `gend_ours` | 0.017 (0.00) | 0.157 (1.00) | 0.078 | **0.042 (0.10)** | 0.441 / 0.532 |
| `smsdt_s` | 0.058 (0.00) | 0.041 (0.00) | 0.088 | **0.048 (0.14)** | 0.485 / 0.547 |
| `smsdt_st` | 0.084 (0.12) | 0.108 (0.42) | 0.085 | **0.043 (0.11)** | 0.415 / 0.490 |
| `smsdt_full` | 0.121 (0.68) | 0.000 (0.00) | 0.089 | **0.047 (0.12)** | 0.486 / 0.552 |

Per run, conformal at n = 20 gives mean FPR 0.039 to 0.054 in all 13 runs. Its TPR is
0.78 to 0.96 of the oracle's. At alpha = 0.10 and n = 20 it gives FPR 0.081 to 0.092,
with P(FPR > 2 alpha) of 0.03 to 0.07.

**Identity weighting and real-source strata (D18, exploratory).** The 121 Celeb-DF real
clusters are 51 celebrities (108 videos) plus 70 unlabelled YouTube-real clips. The
guarantee covers a new identity's video, so the tool now also reports the FPR averaged per
held-out identity (`mean_fpr_idw`). For conformal at n = 20 this is 0.043 to 0.052 across
the 13 runs. With the reals restricted to one source (`--real-strata`), conformal gives FPR
0.041 to 0.052 on each source, but the TPR/oracle ratio falls to 0.73 to 0.93. At alpha =
0.05 and n = 20 the conformal threshold is the maximum of the 20 calibration reals.

**What failed**

- **Source-shape transfer** (my proposed estimator). The selection rule picked `loc` as
  primary, with error 0.037 at n = 10 against 0.055 for `shrink` and 0.076 for `ls`.
  - It did not halve the source error: 0.037 against 0.043 on average.
  - For official GenD it over-corrects: FPR 0.008, TPR 0.29, against an oracle of 0.75.
  - For `smsdt_full` it under-corrects: P(FPR > 2 alpha) = 0.31 at n = 10.
  - The real-score distribution changes shape between FF++ and Celeb-DF, not just
    position, so the location-scale assumption does not hold.
  - The original H7a and H7b (v1) are therefore **not supported** on development data.
- **Bootstrap upper bounds** (`*_ucb`). They keep gross violations rare (P(FPR > 2 alpha)
  of 0.00 to 0.06 at n = 20), but keep only 0.07 to 0.87 of the oracle TPR. For official
  GenD, `loc_ucb` reaches a TPR of 0.08.
- **NP umbrella** abstains below 59 identities. At 60 it holds FPR at 0.017 but is very
  conservative (TPR 0.34).
- **Naive empirical quantile** overshoots the budget. Mean FPR is 0.112 at n = 10 and
  0.085 at n = 20 (0.077 to 0.096 per run). Mean P(FPR > 2 alpha) is 0.46 at n = 10 and
  0.32 at n = 20. The order-statistic correction matters.

**DFD** (exploratory; 18 actor identities, so only n = 5 and n = 10 were run):
- Conformal abstains at both sizes.
- No estimator with 10 or fewer identities improves on the source threshold (error
  0.055, TPR 0.66) without a large TPR loss. `loc_ucb` lowers the error to 0.041 but its
  TPR falls to 0.44; `target_gmm` reaches 0.043 with TPR 0.26.
- DFD cannot test the n = 20 claim.

## 5. What this supports, and what it does not

Supported by development data:
- No zero-shot policy holds the budget across detectors.
- 20 authentic identities with the identity-level conformal threshold held it for every
  detector tested, at a TPR cost of 4 to 22% relative to the oracle.

Not supported:
- That fewer than about 20 identities suffice.
- That source-shape transfer helps.
- Anything beyond Celeb-DF, which is one unseen dataset.

The method is standard split-conformal calibration. The candidate contribution is the
**benchmark**:
- the problem setting (real-shot budget transfer for video deepfake detectors);
- the identity-level protocol;
- real-shot curves for 13 detector runs, including the official GenD weights;
- the finding that the finite-sample rule, not a cleverer estimator, is what works.

Prior art has not been read in full:
- Conformal FPR control already exists for machine-generated text detection
  (arXiv 2505.05084), and with limited false positives in general (arXiv 2202.07650).
- arXiv 2606.21584 audits threshold transfer for speech, without labelled target reals.
- arXiv 2609.10737 transfers conformal calibration with no target labels.

Read all four before claiming novelty for the setting.

## 6. Paper framing (if H7-v2 is confirmed)

The working title is "How many real videos does a deepfake detector need to keep its
false-alarm promise?"

1. Fixed-budget transfer fails for every detector family we tested, including the
   strongest public checkpoint. HTER and AUC hide this.
2. Label-free repairs (mixture thresholds, entropy and consistency TTA, EM) are unsafe
   across detectors (H3a, H5 and H6 falsified).
3. Cheap supervision on the authentic class alone fixes it. About 20 identities with an
   identity-level conformal threshold work, with a finite-sample guarantee.
4. Shape-transfer shortcuts fail because the shape of the real-score distribution
   changes. This is a negative result with a diagnosis.
5. The S-MSDT ablations (H1) show that the extra modules give no AUC gain, which
   supports point 1: the gap is calibration, not capacity.

## 7. Confirmation plan

Do these in order:

1. All three authors sign off section 6 of `docs/decisions.md`.
2. Score at least one new dataset with at least 40 real identities: DFDC preview,
   FakeAVCeleb or WildDeepfake. Nobody opens the scores before sign-off.
3. Run seeds 4 and 5 for the four trained models.
4. Run the tool on the new data:
   `python -m tools.real_shot --targets <new> --checks-on <new> --primary loc`
5. Report the v2 verdicts as they come out, including failures.
