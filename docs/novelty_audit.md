# Novelty Audit of the TMLR Study (2026-11-10)

Branch `novelty-audit` on top of `tmlr-research-study` (f4c1665).
Purpose: establish what the study can currently claim, fix the analysis bugs that change
those claims, and make the remaining novelty candidates testable under the
pre-registration. Nothing here was run on the real score files (they are not in the
repository); all numbers quoted from the study come from `research_v2.md`.

---

## 1. Bottom line

1. **Under the pre-registration, none of H1, H2, H3a currently supports a positive claim.**
   `research_v2.md` sections 4 to 6 report the opposite in places. Those passages are
   corrected in this branch.
2. **The headline "S-MSDT beats GenD by +6.4 AUC" is not a claim about GenD.** `gend_ours`
   scores about 0.83 on Celeb-DF against a published 0.946 (5-seed mean). By the
   pre-registered gate G1 this is Branch B, which requires the official weights
   (`gend_official`) as the external anchor. That evaluation was never run. This branch
   adds it.
3. **Three analysis bugs change conclusions** (section 3): the calibrated threshold was
   applied on the wrong scale, the DFD numbers use mixed reals, and the H1 table tests
   comparisons other than the pre-registered ones.
4. **Novelty is still available, but not where the draft puts it.** The strongest
   candidate is a measured trade-off visible in the existing seed-1 numbers: higher-
   capacity heads raise cross-dataset AUC slightly while making the false-alarm rate at a
   frozen threshold unstable under shift (section 4, N1). It is exploratory and must be
   confirmed on seeds 3 to 5. Two label-free methods that target exactly this failure are
   implemented and pre-registered (N2, N3).
5. This patch cannot guarantee novelty. It makes the candidates falsifiable. If they fail,
   the study is still a publishable, honest matched-pipeline negative result for TMLR,
   whose criterion is that claims are supported by evidence. Check TMLR's current
   reviewing guidelines yourselves.

---

## 2. Pre-registration versus the current write-up

| Item | Pre-registered rule (docs/decisions.md) | What the data shows (seeds 1-2) | What research_v2.md says | Status |
|---|---|---|---|---|
| G1 | Branch B if M <= R - 0.05: report relative to `gend_official` | M = 0.830 vs R = 0.946: Branch B | Calls `gend_ours` the "literature reference anchor" | **Branch B; `gend_official` missing** |
| H1 | Supported if ST and Full vs **S** are equivalent or worse on every unseen dataset; falsified if any is better | ST vs S on Celeb-DF: -0.013 [-0.039, +0.011]; Full vs S was **not computed** (2-seed means 0.8937 vs 0.8831) | "Statistically proven +6.40% improvement" (vs `gend_ours`, not S) | **Undetermined; underpowered** |
| H1 details | 10,000 resamples, Bonferroni, 5 seeds | 2,000 resamples, unadjusted, 2 seeds | Not mentioned | **Deviations, now logged** |
| H2 | Supported if lower CI of g >= 0.05 on an unseen dataset | g <= 0.017 everywhere, but tau_val was applied on the wrong scale and for S/ST/Full no calibration file was found (tau = 0.5) | "Zero threshold catastrophe", "superior generalization efficiency" | **Not yet evaluated correctly; likely falsified** |
| H3a | EM lowers frozen-threshold HTER by >= 0.03 on a majority, no collapse | Best +0.010; harm at pi = 0.1 and 0.9 | "Accurate prior recovery", "stable operating point" | **Falsified, and ill-posed (section 3.4)** |
| H3b | LayerNorm entropy TTA lowers HTER by >= 0.03 | Not run | Listed as next step | Pending |

Celeb-DF is the only truly unseen dataset so far. DFD reals and fakes are in the dev split
used for calibration, so DFD is "dev-overlap", not confirmatory. FF++ is in-domain.

---

## 3. Bugs found and fixed in this branch

### 3.1 Calibrated threshold applied to raw scores (`tools/threshold_transfer.py`)
`eval_cross --fit-calib` stores `tau` as the Youden threshold on the Platt-calibrated
probability sigmoid(a * logit_mean + b). The audit compared it with the raw `score_mean`.
For the earlier Stage B calibration (a = 0.647, b = -0.587, tau = 0.751) the correct rule
is logit_mean >= 2.61, while the buggy rule is logit_mean >= 1.10, a very different
operating point. Runs without a calibration file were silently audited at 0.5 and labelled
`tau_val` (that is why every S/ST/Full row shows tau_val = 0.5000).
**Fix:** `tools/calib_utils.py` converts (a, b, tau) into a logit threshold; runs without a
calibration file are refused unless `--allow-raw-threshold`, and then labelled `raw_0.5`
and excluded from H2. The oracle search is now exact; g has a cluster-bootstrap CI as the
pre-registration requires; per-source FPRs carry Wilson CIs.

### 3.2 Mixed negatives for DFD (`tools/stats.py`, `tools/threshold_transfer.py`)
DFD files contain 383 reals = 140 FF++ YouTube + 243 DFD actors, against the schema's
pure-negative rule. **Fix:** `apply_pure_negatives` (DFD fakes vs `dfd_actor` reals only;
FF++ vs `ffpp_youtube`); `--mixed-negatives` restores the old behaviour for comparison.

### 3.3 H1 tested the wrong comparisons, without the pre-registered settings (`tools/stats.py`)
- Confirmatory pairs are now exactly ST vs S and Full vs S on confirmatory datasets.
  Everything else (Full vs ST, S vs `gend_ours`, `gend_official` pairs) is exploratory.
- Bonferroni-adjusted level 1 - 0.05/m over the confirmatory family; 10,000 resamples by
  default; the pre-registered delta rule (0.01, or min(2 sd_S, 0.02) if sd_S > 0.005).
- Runs are aligned on `video_id` (a run missing a video previously misaligned every row
  after it) and labels are checked across runs.
- Single-class bootstrap resamples were left as delta = 0, biasing the interval toward
  zero; they are now redrawn.
- The pre-registered H1 verdict is computed and written to `hypothesis_h1_meta.json`, with
  a rough factor by which the evaluation set must grow to resolve each comparison.

### 3.4 Prior-shift sweep (`tools/prior_sweep.py`)
- Saerens EM assumes posteriors calibrated under a balanced source prior; the sweep fed
  raw `score_mean`. It now uses Platt-calibrated posteriors when a calibration file exists.
- **H3a is ill-posed.** HTER = (FPR + FNR)/2 does not depend on the class prior, and EM
  only re-weights posteriors monotonically, so EM cannot be expected to lower HTER at a
  fixed threshold; the pre-registered outcome was predictable. EM targets the Bayes
  decision under the new prior, i.e. accuracy or expected cost. The sweep now also reports
  `acc_frozen`, `acc_em_bayes` and `acc_gain`. Keep H3a as registered (falsified) and
  report the accuracy result as exploratory.

---

## 4. Where novelty can still come from

| ID | Candidate | Type | Evidence now | What decides it | Prior-art risk |
|---|---|---|---|---|---|
| N1 | **AUC versus false-alarm stability:** higher-capacity heads (ST, Full) raise cross-dataset AUC slightly but inflate the FPR at a frozen threshold under shift | Finding (H4) | Seed 1, raw 0.5 (below) | H4 on seeds 3-5 with dev-split thresholds and FPR budgets | Medium: "EER hides deployment failure" is argued for speech (arXiv 2606.21584); check for face video |
| N2 | **Label-free FPR control:** set the threshold on unlabelled target logits (2-component mixture, optionally anchored at the source) to hit a false-alarm budget | Method (H5) | Synthetic only | H5 on Celeb-DF (+ any new external set), class-prior stress 0.5/0.66/0.91 | Medium: mixture-based threshold setting is old in general; video deepfake + FPR budget not found in a quick search |
| N3 | **Clip-consistency LayerNorm TTA:** the K clips of a video must agree; use their disagreement as a label-free loss | Method (H6) | Unit tests only | H6 vs entropy-only TTA (H3b) | Medium: T²A (IJCAI 2025) does deepfake TTA; read it to see if it uses clip structure |
| N4 | **Matched-pipeline module utility** with an explicit power analysis: what Celeb-DF-sized evaluations can and cannot resolve | Negative result + methodology (H1) | ST vs S inconclusive; CI half-width about 0.025 | H1 with 5 seeds + more external data | Low as a finding; novelty is modest |
| N5 | **Training recipe over architecture:** S beats `gend_ours` by about +0.05 on Celeb-DF in the same pipeline | Finding | Seeds 1-2, CI excludes 0 | **Only if** `gend_official` in our pipeline is not far above S | High dependency on G1 |

### N1 evidence (exploratory, seed 1, raw threshold 0.5, from research_v2.md)

| Model | Celeb-DF AUC | FPR on FF++ reals | FPR on Celeb-DF reals | Ratio |
|---|---|---|---|---|
| `gend_ours` | 0.832 | 8.6% (12/140) [5.0, 14.4] | 10.1% (18/178) [6.5, 15.4] | 1.2 |
| S | 0.884 | 5.7% (8/140) [2.9, 10.9] | 7.9% (14/178) [4.7, 12.8] | 1.4 |
| ST | 0.871 | 8.6% (12/140) [5.0, 14.4] | 17.4% (31/178) [12.5, 23.7] | 2.0 |
| Full | 0.892 | 6.4% (9/140) [3.4, 11.8] | 22.5% (40/178) [17.0, 29.1] | 3.5 |

Brackets: Wilson 95% intervals. Full and S have nearly the same in-domain FPR, but on
Celeb-DF reals their intervals do not overlap. The pattern was found by looking at the
data, so it is a hypothesis, not a result. It is registered as H4 before seeds 3 to 5 are
trained, with thresholds set on dev-split reals at fixed FPR budgets, which is what a
deployer would do.

Why this could be the paper: it explains why the AUC gains of the larger heads do not
translate into deployable gains, gives a mechanism to test (target reals of a new domain
score higher under larger heads), and motivates N2 and N3 as label-free fixes. It also
turns the existing H2 result (small HTER gaps) into evidence for the argument instead of
against it: HTER looks stable because FNR falls as FPR rises.

---

## 5. Run order (Windows, from the repo root)

```powershell
# 0. Author sign-off on the new pre-registrations in docs/decisions.md (H4, H5, H6) BEFORE step 3.

# 1. Dev-split calibration and dev-split scores for every existing run (needed by H2 and H4/H5)
foreach ($run in "gend_ours_seed1","gend_ours_seed2","smsdt_s_seed1","smsdt_s_seed2","smsdt_st_seed1","smsdt_st_seed2","smsdt_full_seed1","smsdt_full_seed2") {
  $m = $run -replace "_seed\d+",""; $s = $run -replace ".*_seed",""
  python -m smsdt.eval_cross --ckpt "outputs/checkpoints_$m/seed$s/best_ema.pt" --split val `
    --fit-calib "results/calib/$run.json" --save-scores "results/scores/${run}_ffppval.csv" --skip-sweep
}
# (adjust --index/--store so the val split includes the DFD dev videos, as in the original calibration)

# 2. Gate G1: official GenD weights through our pipeline (needs transformers + Hugging Face access)
python -m smsdt.eval_cross --ckpt hf:yermandy/GenD_CLIP_L_14 --T 1 --index <celebdf index> --store <celebdf store> `
  --save-scores results/scores/gend_official_seed0_celebdf.csv --skip-sweep
# repeat for ffpp and dfd; also fit its calibration on val -> results/calib/gend_official_seed0.json

# 3. Seeds 3-5 for all four models (pre-registered: 5 seeds)

# 4. Re-run the audits
python -m tools.stats                 # H1 (pure negatives, Bonferroni, 10k resamples)
python -m tools.threshold_transfer    # H2 (calibrated scale, CI of g, FPR inflation)
python -m tools.prior_sweep           # H3a (calibrated posteriors) + accuracy gain
python -m tools.fpr_control           # H4, H5
# H3b and H6: TTA runs with smsdt.tta.TTAAdapter(objective="entropy" | "consistency")
```

G1 diagnostics to record in `docs/decisions.md` (one line each):
- A = official weights on our crops (step 2). B = official weights on the official GenD
  preprocessed Celeb-DF (their Hugging Face dataset, their code). C = `gend_ours`.
- A vs B isolates preprocessing (our 256 store with 224 centre crop versus their landmark
  alignment at scale 1.3 and CLIPProcessor resize). B vs paper checks their harness.
  A vs C isolates training. If A is near 0.95, our recipe and our claims against GenD do
  not hold; if A is near 0.83, our preprocessing caps every model and absolute numbers
  must be reported as pipeline-constrained (Branch B text).

---

## 6. Decision tree to a paper framing

| Outcome | Paper |
|---|---|
| H4 confirmed, H5 or H6 supported | **Main story:** larger heads buy AUC at the cost of false-alarm stability; label-free FPR control (and/or clip-consistency TTA) restores the budget. Methods paper with a measured failure. |
| H4 confirmed, H5 and H6 falsified | Analysis paper: the trade-off, its mechanism (per-source real-score shift), why HTER/EER audits hide it, and two label-free fixes that fail with documented failure modes. |
| H4 not confirmed | Matched-pipeline study (N4, and N5 if G1 allows): modules do not reliably help once seeds, baselines and pure negatives are controlled; with a power analysis of what Celeb-DF-sized evaluations can resolve. |
| `gend_official` in our pipeline clearly above S | Every claim against GenD is dropped. The paper is about our pipeline's relative comparisons and the operating-point findings only. |

Required in every branch: 5 seeds, at least one more external dataset (DFDC or
WildDeepfake; Celeb-DF alone gives CI half-widths near 0.025 AUC, too wide for delta =
0.01), and the deviation log.

---

## 7. What this branch does not do

- It does not run anything on the real score files or checkpoints (not available here).
- It does not train seeds 3 to 5 or run TTA.
- The official GenD path is unit-tested with a stub model only; the download from Hugging
  Face could not be tested from this environment.
- Prior-art checks above are title/abstract level. Read GenD, T²A, FCG and the speech
  threshold-transfer paper in full before writing any novelty sentence.
