# S-MSDT to TMLR: Decisions & Pre-Registration Log

**Status:** `pre-registered`  
**Date:** 2026-11-09  
**Repository Branch:** `tmlr-research-study`

---

## 1. Pre-Registration Sign-Off

All three authors (**Sharad**, **Aditya Raj**, **Aditya Kumar**) confirm and freeze the following definitions, hypotheses, and decision rules before any confirmatory dataset is evaluated:

### 1.1 Pre-Registered Hypotheses & Numeric Falsification Rules

| Hypothesis | Pre-Registered Statement | Supported If | Falsified If |
|:---|:---|:---|:---|
| **H1** | Neither ST nor Full improves AUC over S by more than $\delta$ on unseen datasets | For both ST and Full on every confirmatory dataset, comparison is classified as **equivalent** or **worse** | Any ST or Full comparison on any confirmatory dataset is classified as **better** |
| **H2** | A validation-frozen threshold loses at least $0.05$ HTER vs. the per-dataset optimum on at least one unseen dataset (for S and for GenD-ours) | Lower bound of the $95\%$ CI of $g = \text{HTER}(\tau_{\text{val}}) - \min_\tau \text{HTER}(\tau)$ is $\ge 0.05$ on at least one confirmatory dataset | Upper bound of the $95\%$ CI of $g$ is $< 0.05$ on every confirmatory dataset |
| **H3a** | EM prior-shift correction lowers frozen-threshold HTER by at least $0.03$ on a majority ($> 50\%$) of confirmatory datasets, with zero collapse | Both conditions hold across all confirmatory datasets | Fewer than a majority reach $0.03$ reduction, OR any collapse occurs in any seed |
| **H3b** | LayerNorm-only entropy-minimization TTA lowers frozen-threshold HTER by at least $0.03$ on a majority of confirmatory datasets, with zero collapse | Both conditions hold across all confirmatory datasets | Fewer than a majority reach $0.03$ reduction, OR any collapse occurs in any seed |

### 1.2 Formal Statistical Definitions
- **Seed-Averaged Metric:** All reported metrics are averaged over seeds $1, 2, 3, 4, 5$ on each evaluation dataset.
- **$\Delta\text{AUC}(A, S)$:** $\overline{\text{AUC}}(A) - \overline{\text{AUC}}(S)$, computed over identical paired cluster bootstrap resamples.
- **Cluster Bootstrap:** Resamples identity clusters with $10,000$ iterations. Confidence intervals for H1 comparisons utilize Bonferroni-adjusted family coverage ($1 - 0.05/m$, where $m$ is total pairwise comparisons).
- **Equivalence Classification with Margin $\delta$:**
  - $\delta = 0.01$ (frozen default, unless empirical seed standard deviation of S on Celeb-DF exceeds $0.005$, in which case $\delta = \min(2 \times \text{sd}, 0.02)$).
  - Given adjusted interval $[L, U]$:
    - **Better:** $L > \delta$
    - **Equivalent:** $L \ge -\delta$ and $U \le +\delta$
    - **Worse:** $U < -\delta$
    - **Inconclusive:** Otherwise
- **HTER Metric:** $\text{HTER}(\tau) = 0.5 \times (\text{FPR}(\tau) + \text{FNR}(\tau))$. $\tau_{\text{val}}$ is the Platt-calibrated Youden threshold fitted strictly on dev set.
- **Collapse Definition:** Post-adaptation AUC drops by $> 0.03$ relative to pre-adaptation AUC, OR $\ge 95\%$ of test samples are predicted as one class while that class represents $\le 85\%$ of the ground truth.

---

## 2. Institutional Checks & Prior Draft Confirmation (Task 0.5)

- **Prior IEEE TIFS Draft Status:** Confirmed **NEVER submitted** to any journal, conference, workshop, or archival repository. No dual-submission conflict exists.
- **GenD Reference Repository (Task 0.7):**
  - Cloned into `third_party/GenD`
  - Upstream commit hash: `387a42266dd385fbe8f7626c5d3aa03eea3bfaab`
- **Initial S-MSDT Benchmark Records (Task 0.8):**
  - Audited against `results.md` and `research.md`:
    - Phase 1 Baseline: FF++ Macro AUC = 0.8510
    - Phase 2 Stage B Fixed (ST): FF++ Macro AUC = 0.9385, Celeb-DF v2 Video AUC = 0.8754
    - Phase 3 Stage C (Full): FF++ Macro AUC = 0.9349, Celeb-DF v2 Video AUC = 0.8813

---

## 3. Pre-Written Gate Branch Framings

### Gate G1 (Reproduction Gate — Day 7)
- **Branch A ($M \ge R - 0.02$):**
  *"Our reimplementation reproduces the published GenD baseline within the margin of error ($M$ vs. $R$). Therefore, any observed performance delta in S-MSDT variants is directly attributable to the added architectural modules rather than training or data pipeline discrepancies."*
- **Branch B ($M \le R - 0.05$):**
  *"Our pipeline incurs an offset of $\Delta = R - M$ relative to the official codebase. Consequently, absolute performance is pipeline-constrained, and claims are formulated strictly on matched-pipeline relative comparisons, using `gend_official` as the external reference anchor."*
- **Branch C (Between $R - 0.05$ and $R - 0.02$):**
  *"Baseline reproduction is partial. Both absolute numbers and relative deltas are reported transparently alongside pipeline sensitivity analysis."*

---

## 4. Deviation Log (Append-Only)
*No deviations recorded prior to pre-registration freeze.*

### 2026-11-10: audit of analyses run 2026-11-09 to 2026-11-10 (see docs/novelty_audit.md)

| ID | Deviation from section 1 | Effect | Action |
|:---|:---|:---|:---|
| D1 | Results and claims reported with 2 of the 5 pre-registered seeds | All multi-seed claims are provisional | Train seeds 3-5; label 2-seed tables "interim" |
| D2 | H1 run with 2,000 resamples and unadjusted 95% CIs (pre-registered: 10,000, Bonferroni) | Intervals too narrow for the family | `tools/stats.py` defaults fixed |
| D3 | H1 table tested Full vs `gend_ours`, Full vs ST, ST vs S, S vs `gend_ours`; pre-registered H1 is ST vs S **and Full vs S**; Full vs S was never computed | H1 verdict was not evaluated | Confirmatory pairs fixed; others labelled exploratory |
| D4 | DFD evaluated against 383 mixed reals (140 FF++ YouTube + 243 DFD actors), contrary to `docs/score_schema.md` | DFD AUC and FPR not pure-source | Pure negatives by default |
| D5 | H2: tau (calibrated scale) compared with raw `score_mean`; runs without calibration files audited at 0.5 and labelled tau_val; no CI for g | H2 not evaluated as registered | `tools/calib_utils.py`; refuse silent fallback; cluster CI for g |
| D6 | H3a: raw `score_mean` used as posteriors in Saerens EM; one model, one dataset | H3a evidence incomplete | Calibrated posteriors; all runs. Note: HTER is prior-invariant, so H3a as registered cannot succeed by construction; registered outcome kept, accuracy reported as exploratory |
| D7 | Gate G1 decided Branch B (M = 0.830, 2-seed mean of `gend_ours`, vs R = 0.946, GenD 5-seed mean, Table 4 of arXiv 2508.06248v3; M <= R - 0.05) but `gend_official` was not evaluated | Claims "vs GenD" have no valid anchor | `hf:` checkpoint path in `smsdt/eval_cross.py`; run it |
| D8 | DFD reals and fakes are in the dev split used for calibration, so DFD is not an unseen dataset | DFD cannot be confirmatory for H1/H2 | Dataset roles: FF++ in-domain, DFD dev-overlap, Celeb-DF (+ new external sets) confirmatory |
| D9 | `research_v2.md` states single-run results as "Key Scientific Finding" and "statistically proven" | Overclaims | Corrected in research_v2.md (audit note at top) |
| D10 | `fpr_control` bootstrapped on the raw `cluster_id` column; a coarse column collapses the CI to a point (reported CIs such as [0.222, 0.229] at n=178 reals) | Over-confident H4 verdicts | Clusters now resolved as in `tools/stats.py`; Wilson bounds reported; H4 verdicts use the wider (`fpr_ci_low_cons`) |
| D11 | H4 reported as "confirmed" counting seeds 1-2, which were seen before registration | Exploratory seeds counted as confirmatory | `h4a_verdict` counts only `--confirmatory-seeds` (default 3 4 5); fewer than 3 gives "undetermined" |
| D12 | GMM thresholds outside the observed logit range (TPR = FPR = 0) were not flagged degenerate | H5 diagnostics hid collapse | `outside_support` now sets `degenerate` |

### 2026-11-10: Gate G1 decision
Branch B. Until `gend_official` has been evaluated through this pipeline, no sentence may
compare S-MSDT with GenD; `gend_ours` is called "our GenD-style re-implementation".

---

## 5. Additional Pre-Registrations (PROPOSED 2026-11-10, effective on sign-off)

These must be signed off by all three authors **before** seeds 3-5 are trained and before
`tools/fpr_control.py` or the consistency TTA objective is run on any real score file.
Seeds 1-2 have been looked at; for H4 they are exploratory only.

Sign-off: Sharad [ ]  Aditya Raj [ ]  Aditya Kumar [ ]  Date: ________

Common settings: confirmatory runs = seeds 3, 4, 5; confirmatory datasets = Celeb-DF plus
any external dataset whose scores have not been inspected by the date of sign-off; source
threshold set on dev-split reals of the same run (`<run>_ffppval.csv`); primary budget
alpha = 0.05 (alpha = 0.10 secondary); pure-source negatives; cluster bootstrap by identity.

| Hypothesis | Statement | Supported if | Falsified if |
|:---|:---|:---|:---|
| **H4a** (origin: exploratory, seed 1) | With the threshold set for alpha = 0.05 on dev-split reals, ST and Full exceed twice the budget on unseen data | For both ST and Full, the lower 95% CI of achieved FPR is > 0.10 on Celeb-DF in at least 2 of 3 confirmatory seeds | Otherwise |
| **H4b** | Under the same thresholds, ST and Full raise the target FPR relative to S (paired, same reals) | Paired FPR delta (Full - S) and (ST - S) has lower 95% CI > 0 on Celeb-DF in at least 2 of 3 seeds | Upper 95% CI < 0.02 in at least 2 of 3 seeds |
| **H5** | A label-free threshold (`target_gmm` or `target_gmm_anchored`, both reported) halves the FPR error of `source_quantile` | For the policy in question: prior-stress mean abs(FPR - alpha) at alpha = 0.05 <= 0.5 x that of `source_quantile` on a strict majority of confirmatory datasets, in at least 2 of 3 seeds, with degenerate/swapped rate <= 0.05 at every stress level (fake fractions 0.5, 0.66, 0.91) | Otherwise |
| **H6** | Clip-consistency LayerNorm TTA (lambda_e = 1, lambda_d = 0.1, lr 1e-4, both H3b modes) lowers frozen-threshold HTER by >= 0.03 and beats entropy-only TTA by >= 0.01, with zero collapse | Both hold on a majority of confirmatory datasets in at least 2 of 3 seeds | Otherwise; any collapse (section 1.2 definition) falsifies |

Fixed tool settings (not to be tuned): mixture min_weight 0.02, min_separation 1.0;
fpr_control n_boot 1000, n_repeats 100; TTA lambda_d = 0 reported as an ablation only.

