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

