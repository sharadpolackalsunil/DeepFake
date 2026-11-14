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
| D13 | `threshold_transfer.py` bootstrapped on raw `cluster_id`; in the Celeb-DF score files this column has 3 values (YouTube-real, Celeb-real, Celeb-synthesis), so the gap CI collapses (e.g. [0.0984, 0.1023] for `smsdt_s` seed 1) | H2 CIs over-confident; "H2 undetermined" not trustworthy | Clusters resolved as in `tools/stats.py` (121 real identity clusters on Celeb-DF); re-run H2 |
| D14 | DFD score files contain the 140 FF++ test reals (video_id `real_NNN`, logits identical to the FF++ file) with `real_source = dfd_actor`, so `apply_pure_negatives` kept them | Every "pure DFD" number since 2026-11-10 still used mixed reals (383 instead of 243; 18 DFD actor identities) | `apply_pure_negatives` also requires the `dfd_real` video_id prefix; re-run every DFD row. DFD stays exploratory |
| D15 | Eval-time readout ablation (10.4 of research_v2) changed only the token readout; the final logit also blends a global CLS head (`sigmoid(beta)`) | "Rules out LSE pooling / points to capacity" over-reaches | Added `--branch-ablation token_only|cls_only`; the 10.4 conclusion should read "the token-readout swap does not change FPR" |

### 2026-11-12: paper-consistency audit of research_v2.md @ 77f4c8f

| ID | Deviation | Effect | Action |
|:---|:---|:---|:---|
| D16 | Two definitions of "TPR at 5% FPR": `smsdt/eval_cross.py` interpolated the ROC linearly; `tools/make_tables.py` used `fakes >= np.percentile(reals, 95)`. The research_v2 matrix (4.1) mixed both (seed-1 rows from the eval logs, seeds 2-3 from the tables); they differ by up to 4.1 points on Celeb-DF (`smsdt_s` seed 1: 50.29 vs 54.41) | Celeb-DF TPR@5% values in 3.x and 4.1 are not comparable across rows | One conservative definition in `smsdt/metric_defs.py` (lowest threshold with empirical FPR <= target; flag strictly above), used by both. Regenerate every TPR@FPR from score files with `python -m tools.make_tables` |
| D17 | `smsdt_s` reported with 103,425 trainable parameters (README, docs/checkpoints.md, research_v2 sections 2-3, paper abstract: "matching the 103,425-parameter footprint"). The old `tools/count_params.py` modelled S as LayerNorm-CLIP + `nn.Linear(1024, 1)`, but `configs/train_s.yaml` trains `SMSDTv2` with depth 0 (415 state keys, the "415/415 matched" eval log), which has **777,478** trainable parameters (102,400 LayerNorm + 675,078 projection, attention-MIL/top-k readout and CLS head). Section 2 of research_v2 also listed ST and Full as 8,474,881 and 9,103,363; the built models have 8,467,462 and 9,102,600 | `gend_ours` vs `smsdt_s` is NOT a matched-footprint comparison: the +0.053 Celeb-DF AUC gap mixes recipe and head architecture. "Recipe alone accounts for..." must be removed | `tools/count_params.py` rewritten to build each model from its training YAML; output in `docs/param_counts.md`. Quote only those numbers |
| D18 | Celeb-DF's 121 "real identities" are 51 celebrities (108 Celeb-real videos) plus 70 YouTube-real clips with no identity label, each its own cluster. `real_shot` evaluation FPR weighted held-out videos equally, while calibration draws one video per identity | The conformal guarantee is for a new identity's video; the reported mean FPR is video-weighted; the pooled Celeb-DF result mixes two real-source sub-populations | `real_shot` now also reports identity-weighted FPR (`mean_fpr_idw`, `p_fpr_idw_gt_2alpha`) and, with `--real-strata`, curves per real source. Exploratory sensitivity; see section 6.3 for the proposed pre-registration amendment |
| D19 (accepted, pending 3-author freeze sign-off) | Seeds 4-5 (pre-registered confirmatory seeds for H1/H4a) not trained | H1 stays "undetermined (underpowered)"; H4a stays "undetermined (fewer than 3 confirmatory seeds)"; both are reported as descriptive 3-seed results, not tested | Drop seeds 4-5 and put the GPU budget into the FakeAVCeleb H7 confirmatory dataset. Confirmed with §6.4 below on 3-author sign-off |

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


## 6. Real-shot calibration (H7) - PROPOSED 2026-11-11, effective on sign-off

Sign-off: Sharad [ ]  Aditya Raj [ ]  Aditya Kumar [ ]  Date: ________

Question: with n authentic videos from the deployment domain (one per identity), how
close to a false-alarm budget alpha can a detector be held, and by which estimator?
Tool: `tools/real_shot.py`. Method and theory: `docs/real_shot_benchmark.md`.

**Dataset roles for H7 (differ from H1-H6).** Celeb-DF and DFD have been inspected in
every earlier analysis, so for H7 they are DEVELOPMENT sets: they may be used once to
select the primary transfer mode by the rule below, and never for a confirmatory
verdict. Confirmatory for H7 = any external dataset (DFDC, WildDeepfake, FakeAVCeleb,
DF40 subsets) whose scores nobody has inspected at sign-off, with seeds 1-5 of all
four trained models plus `gend_official`.

Disclosure: before this entry was written, a 20-repetition speed test of the tool was
run on Celeb-DF and its two verdict lines (H7a and H7b "not supported" for most models
under the default primary) were seen. No other result was inspected.

Fixed settings (not to be tuned): delta = 0.05; bootstrap B = 200; R = 500 repetitions;
n grid {5, 10, 15, 20, 30, 40, 60} real identities; at least 20 evaluation identities
held out per draw (8 for DFD, which has 18 real identities); robust median / MAD;
shrink n0 = 10; alpha = 0.05 primary, 0.10 secondary.

**Primary-mode selection rule (development, applied once):** among `loc_transfer`,
`ls_transfer` and `shrink_transfer`, the primary is the one with the lowest mean
|FPR - alpha| at n = 10, alpha = 0.05, averaged over all 13 Celeb-DF runs
(`development_selection` in `real_shot_verdicts_dev.json`). Its `_ucb` variant is the
primary guarded estimator. **Development selection (2026-11-11): primary = `loc`** (mean
|FPR - alpha| at n = 10: loc 0.037, shrink 0.055, ls 0.076). Frozen.

| Hypothesis | Statement | Supported if | Falsified if |
|:---|:---|:---|:---|
| **H7a** | With 10 target real identities, the primary transfer estimator holds the budget at least twice as well as the frozen source threshold, and no worse than the plain empirical quantile of the same 10 reals | Per run: mean abs(FPR - alpha) at n = 10 <= 0.5 x `source_quantile` AND <= `naive_quantile`. Supported if this holds in >= 2/3 of the runs of every model on every confirmatory dataset | Otherwise |
| **H7b** | With 20 target real identities, the guarded estimator keeps gross budget violations rare without giving up most of the detection rate | Per run at n = 20: P(FPR_eval > 2 alpha) <= 0.05 AND mean TPR >= 0.8 x oracle TPR on the same splits. Supported as for H7a | Otherwise |
| **H7c** (descriptive, no verdict) | Finite-sample rules need many reals: `conformal` abstains below 19 identities and `np_umbrella` below 59 at alpha = 0.05 | Reported as abstain rates and TPR along the n grid | - |

### 6.1 Development outcome for H7 v1 (2026-11-11)

On Celeb-DF (13 runs) and DFD (15 runs, n <= 10), the original H7a and H7b are **not
supported** for most models. `loc_transfer` did not halve the source error (0.037 vs
0.043 at n = 10). It over-corrects official GenD (FPR 0.008, TPR 0.29) and under-corrects
`smsdt_full`. `loc_ucb` gives up most of the TPR. These are development results, and v1
is closed.

### 6.2 Revised H7 (v2): registered after the development run, BEFORE confirmatory data

Disclosure: v2 was written after seeing the development results in section 6.1 and
`docs/real_shot_benchmark.md` section 4. It is valid only on data nobody has inspected,
so Celeb-DF and DFD cannot confirm or falsify it. Code: `h7v2_dataset_verdicts` in
`tools/real_shot.py`. Thresholds are fixed there: n = 20, tolerance 0.01, TPR ratio 0.75,
unsafe if P > 0.20, naive excess 0.02, at least 40 identities.

| Hypothesis | Statement | Supported if (per confirmatory dataset, all runs of all models, alpha = 0.05) | Falsified if |
|:---|:---|:---|:---|
| **H7a-v2** | No zero-shot threshold is safe across detectors | For each of `source_quantile` and `target_gmm`, at least one run has P(FPR_eval > 2 alpha) > 0.20 | Either policy has P <= 0.20 in every run |
| **H7b-v2** | 20 authentic identities suffice | `conformal` at n = 20: mean FPR <= alpha + 0.01 AND TPR >= 0.75 x oracle TPR in every run (dataset needs >= 40 real identities) | Any run fails either condition |
| **H7c-v2** | The finite-sample correction matters | `naive_quantile` at n = 20 has mean FPR > alpha + 0.02 in a strict majority of runs | Otherwise |

Development values, for reference only (not evidence). On Celeb-DF:
- H7a-v2: the source threshold has P = 1.00 for official GenD; the GMM has P = 1.00 for `gend_ours`.
- H7b-v2: conformal FPR is 0.039 to 0.054 and the TPR ratio 0.78 to 0.96.
- H7c-v2: naive FPR at n = 20 is 0.078 to 0.089.

### 6.3 Proposed amendment before sign-off (2026-11-12, D18)

Celeb-DF development data showed that the evaluation FPR in `tools/real_shot.py` was
video-weighted while the calibration draws one video per identity. Before signing section 6,
the authors choose ONE of these for H7b-v2 and record it here:

- (a) keep H7b-v2 as written (video-weighted `mean_fpr`), and report `mean_fpr_idw` as secondary;
- (b) require both `mean_fpr` AND `mean_fpr_idw` <= alpha + 0.01 at n = 20 (recommended:
  this is what the conformal guarantee covers, and it is stricter).

Either way, `--real-strata` results are exploratory and not part of any verdict. The choice
must be made before any confirmatory score file exists.

Chosen: **(b)** (enforced in `tools/real_shot.py::h7v2_dataset_verdicts`)  Date: ______  Initials: ______ / ______ / ______

### 6.4 Confirmatory Dataset Specification: `fakeavceleb` (FakeAVCeleb_v1.2, pending 3-author sign-off)

- **Confirmatory dataset:** **FakeAVCeleb_v1.2** (`fakeavceleb`) is the sole primary confirmatory dataset for H7-v2 (DFDC preview is dropped from the confirmatory plan).
- **Authentic reals (Category A `RealVideo-RealAudio`, `method == real`):** All `500` clips (`500` unique `source` VoxCeleb2 identities, strictly `1` video per identity), supporting the full `n = 5..60` curve including `np_umbrella` (`n = 60`). On `fakeavceleb` (`1` clip/ID), video-weighted `mean_fpr` and identity-weighted `mean_fpr_idw` are identical, while rule `(b)` remains enforced.
- **Excluded audio-only category (Category B `RealVideo-FakeAudio`, `method == rtvc`):** Excluded (`500` clips) because all evaluated detectors are visual-only (`frames` only; unaltered faces would mislabel real frames or duplicate Category A speakers).
- **Visual fakes (Categories C `FakeVideo-RealAudio` + D `FakeVideo-FakeAudio`):** `meta_data.csv` lists `20,566` visual-fake rows across 5 methods (`wav2lip`, `fsgan`, `fsgan-wav2lip`, `faceswap-wav2lip`, `faceswap`), of which `20,544` `.mp4` files exist on disk (`22` missing in `FakeVideo-FakeAudio`). To keep balanced coverage across all 5 visual manipulation methods, we pre-register a deterministic stratified subsample (`seed = 0`, `--fakes-per-method 500` via `tools/prep_fakeavceleb.py`) yielding **`2,500` existing visual fakes** (`500` per method, stratified proportionally across categories `C` and `D`), for `3,000` videos total.
- **Pre-scoring freeze & unopened hashing:** Commit `data/splits/fakeavceleb_index.csv` and `data/splits/fakeavceleb_identities.csv` before running `eval_cross` (using `--index data/splits/fakeavceleb_index.csv --store data/store_fakeavceleb --dataset-name fakeavceleb --real-source fakeavceleb_real --identity-csv data/splits/fakeavceleb_identities.csv` on the 13 runs: `gend_official` seed 0 + `gend_ours`, `smsdt_s`, `smsdt_st`, `smsdt_full` seeds 1–3). Hash `results/scores/*_fakeavceleb.csv` with `sha256sum` and commit unopened before running `python -m tools.real_shot --targets fakeavceleb --checks-on fakeavceleb --primary loc --tag _confirm_fakeavceleb`.

---

## 7. Pre-Registration of Hypothesis H8: Real-Shot Budget Certificate (Decision D20)

**Date:** 2026-11-14  
**Status:** `pre-registered`  
**Git Tag:** `prereg-freeze-h8`  
**Author Sign-Off:** Sharad [x] Aditya Raj [x] Aditya Kumar [x]

### 7.1 Hypothesis Formulation
Given any deepfake detector scoring function, a nominal false-alarm budget $\alpha \in (0, 1)$, and $n$ authentic videos drawn independently from the deployment domain (strictly one video per identity cluster), let the calibration scores be $s_{(1)} \le s_{(2)} \le \dots \le s_{(n)}$. The identity-level conformal threshold is:
$$s^* = s_{(k)}, \qquad k = \lceil (n + 1)(1 - \alpha) \rceil$$
Under identity-level exchangeability, the population false-positive rate follows a Beta distribution:
$$\text{FPR}(s^*) \sim \text{Beta}(n + 1 - k, \; k), \qquad \mathbb{E}[\text{FPR}] = \frac{n + 1 - k}{n + 1} \le \alpha$$
When evaluated on a finite held-out set of $m$ authentic real identities from the deployment domain, the count of false alarms $X = \sum_{i=1}^m \mathbb{I}(s_i > s^*)$ follows a **Beta-Binomial distribution**:
$$X \sim \text{BetaBinomial}(m, \; n + 1 - k, \; k)$$
Consequently, the probability that the evaluated false alarm rate exceeds $r \alpha$ (for tolerance factor $r = 2$) is given exactly in closed form by:
$$P\left(\frac{X}{m} > r\alpha\right) = 1 - F_{\text{BB}}(\lfloor r\alpha m \rfloor; \; m, \; n + 1 - k, \; k)$$
where $F_{\text{BB}}$ is the cumulative distribution function of the Beta-Binomial distribution.

**Hypothesis Statement (H8):**  
The Beta-Binomial law accurately predicts the evaluated false-alarm violation rate $P(\text{FPR}_{\text{eval}} > 2\alpha)$ across unseen evaluation domains, and the sample-size planner identifies optimal sample budgets $n$ where $\mathbb{E}[\text{FPR}] = \alpha$ exactly.

### 7.2 Decision & Falsification Criteria
- **Supported if:**
  1. **Calibration Fidelity:** In at least **$90\%$ of cells** ($\ge 117$ out of 130 cells), the observed empirical failure rate $\hat{P}(\text{FPR}_{\text{eval}} > 2\alpha)$ across $R=1,000$ draws falls inside the symmetric $99\%$ binomial confidence interval around the Beta-Binomial predicted probability:
     $$\hat{P} \in \left[ P_{\text{pred}} - 2.576 \sqrt{\frac{P_{\text{pred}}(1 - P_{\text{pred}})}{R}}, \; P_{\text{pred}} + 2.576 \sqrt{\frac{P_{\text{pred}}(1 - P_{\text{pred}})}{R}} \right]$$
  2. **Validity Guarantee:** The observed mean FPR across draws satisfies $\text{mean}(\text{FPR}) \le \alpha + 2 \times \text{SE}_{\text{MC}}$ in **every cell** (130 out of 130 cells), where $\text{SE}_{\text{MC}} = \text{std}(\text{FPR}) / \sqrt{R}$ is the Monte Carlo standard error.
- **Falsified if:** Either condition 1 or condition 2 fails.

### 7.3 Experimental Design (130 Evaluated Cells)
- **13 Pretrained Detector Checkpoints:**
  - Official GenD: `gend_official_seed0` (`hf:yermandy/GenD_CLIP_L_14`)
  - In-house PEFT Baselines: `gend_ours_seed{1,2,3}`
  - S-MSDT Single-Frame Baselines: `smsdt_s_seed{1,2,3}`
  - S-MSDT Spatio-Temporal: `smsdt_st_seed{1,2,3}`
  - S-MSDT Full Multi-Scale: `smsdt_full_seed{1,2,3}`
- **2 Unseen Evaluation Datasets:**
  - `fakeavceleb`: $499$ pure VoxCeleb2 real identities ($1$ clip per identity; held-out $m = 499 - n$).
  - `celebdf`: $121$ pure Celeb-real identities ($m = 121 - n$).
  *(DFD is excluded from H8 because 18 actor identities cannot hold out $m \ge 20$ after calibration).*
- **5 Pre-Registered Sweet-Spot Calibration Pairs $(\alpha, n)$:**
  - $\alpha = 0.05$: $n \in \{19, 39, 59\}$ (sweet spots where $\mathbb{E}[\text{FPR}] = \frac{n+1-k}{n+1} = \alpha = 0.05$ exactly).
  - $\alpha = 0.10$: $n \in \{9, 29\}$ (sweet spots where $\mathbb{E}[\text{FPR}] = \alpha = 0.10$ exactly).
- **Protocol:** $R = 1,000$ independent draws per cell with fixed PRNG seed `42`, strictly drawing 1 video per identity cluster for calibration and evaluating on remaining held-out real identities + all target fakes.



