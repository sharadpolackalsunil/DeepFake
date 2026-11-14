# Completion plan: from `feat/eval-and-benchmarks` to a TMLR submission

Written 2026-11-12 against `77f4c8f` plus the `fix/paper-consistency` patch. Target:
submit by **2026-11-30**. TMLR has no deadline, so the date is ours to move, but every
step below is ordered so that nothing waits on anything it does not need.

## 0. Where the study stands

The paper is no longer "S-MSDT generalises better". Official GenD beats every S-MSDT
variant (Celeb-DF AUC 0.9327 vs at most 0.8882), and the extra modules give no
demonstrable AUC gain (H1 undetermined, DFD worse). The contribution is a deployment
benchmark:

> How many authentic videos from a new domain does a deepfake detector need to keep its
> false-alarm budget, and which threshold rule works?

| Claim | Status | Evidence | Usable in the paper as |
|---|---|---|---|
| Source-split thresholds do not carry a 5% budget to a new domain | Shown on Celeb-DF and DFD for every detector family, including official GenD (3.15x) | §§9-10 | Motivation, with CIs |
| Label-free repairs (GMM, entropy/consistency TTA, EM) are unsafe | H3a, H5, H6 falsified | §§6-7, 10.3 | Negative results |
| About 20 authentic identities + conformal order statistic hold the budget | Development support only (Celeb-DF, inspected) | §11, `docs/real_shot_benchmark.md` | **Main claim, only after confirmation (step 3)** |
| Source-shape transfer helps with fewer reals | Not supported (H7-v1) | §11.1 | Negative result with diagnosis (fig 5) |
| Extra ST / forensic modules improve cross-dataset AUC | Undetermined; DFD worse | §4.3 | Descriptive ablation |
| FPR inflation lives in the token pathway, not the readout | Shown on 3 seeds | §10.4 | Ablation paragraph |
| `gend_ours` vs `smsdt_s` isolates the training recipe | **False** (D17: 103,425 vs 777,478 trainable) | `docs/param_counts.md` | Remove |

## 1. Apply the fixes (day 1, 1-2 hours)

```bash
git checkout feat/eval-and-benchmarks && git pull
git checkout -b fix/paper-consistency
git am smsdt-paper-consistency.patch
python -m pytest tests/test_paper_fixes.py tests/test_real_shot.py -q
python -m tools.make_tables            # experiment_matrix{,_agg}.{csv,md}
python -m tools.count_params           # docs/param_counts.md (offline, about 1 min)
python -m tools.paper_figures          # paper/figures/fig1..fig5 (.pdf, .png)
```

The patch already contains the regenerated tables. Rerunning them is a check that your
machine gives the same numbers.

What changes and why: see `docs/decisions.md` D16-D19 and section 6.3. In short:

- **D16:** one TPR@FPR definition (conservative empirical point). The §4.1 matrix was
  regenerated; seed 2-3 Celeb-DF TPR@5% values change by up to 4.1 points.
- **D17:** `smsdt_s` has 777,478 trainable parameters. README, `docs/checkpoints.md`,
  research_v2 §§2-3 are fixed. The paper still says 103,425 in four places (step 4).
- **D18:** identity-weighted FPR and per-real-source strata for H7.
- **D19 (proposed):** drop seeds 4-5.
- **New `smsdt.eval_cross` flags `--real-source` and `--identity-csv`.** Without them, a
  new dataset's reals are tagged `ffpp_youtube`, and identities are guessed from file
  names. Both would silently break the confirmatory run.

## 2. Freeze the protocol (days 1-2; all three authors)

Nothing in step 3 may start before this is done and committed.

1. Read `docs/decisions.md` section 6 (H7 v1 and v2) and **6.3** (choose (a) or (b) for
   the identity-weighted FPR; (b) is recommended).
2. Decide D19: sign it (seeds 4-5 dropped) or delete the row and train them.
3. Fill in the sign-off lines in sections 1, 5 and 6, then commit.
4. Tag the commit: `git tag prereg-freeze-h7 && git push origin prereg-freeze-h7`.
5. Write the confirmatory dataset name, the fake subsample size and the score-file hash
   procedure (step 3.4) into section 6 **before** downloading anything.

## 3. Confirmatory dataset (days 1-8; GPU)

### 3.1 Choose the dataset

Requirements: never scored by this project; at least **40 real identities** with an
identity label per real video; at least a few hundred fakes; licence allows research use.

| Candidate | Why | Risk |
|---|---|---|
| FakeAVCeleb | Real videos come from VoxCeleb2 speakers with speaker IDs, so identities are labelled | Access is by request form; ask today |
| DFDC (preview or test) | Easy to get; paid actors | Check that actor IDs are available for real videos; without them, each real video is its own cluster, which weakens the identity argument |
| WildDeepfake | Already wired into `scripts/run_confirmatory.sh` | No identity labels, mixed sources; use only as a second, clearly labelled set |

Verify the identity metadata of whichever you pick before freezing; do not rely on this
table. Request access to two datasets in parallel so a delay on one does not block.

### 3.2 Prepare

1. Extract faces with the same pipeline as FF++ / Celeb-DF (`smsdt/preprocess/`), and add
   rows to the split index with split `test`.
2. Write `data/splits/<dataset>_identities.csv` with columns `video_id,identity` for every
   real video (fakes optional).
3. Pre-register a fake subsample if there are many fakes (e.g. 1,000 drawn with seed 0).
   Write the list to a file and commit it with the freeze.
4. Add the dataset key to `DEFAULT_DATASET_ROLES` in `tools/stats.py` as `confirmatory`.

### 3.3 Score all 13 runs (one pass, then do not look)

For each run (official GenD seed 0 plus `gend_ours`, `smsdt_s`, `smsdt_st`,
`smsdt_full` seeds 1-3):

```bash
python -m smsdt.eval_cross --ckpt <ckpt or hf:yermandy/GenD_CLIP_L_14> \
  --index data/splits/index.csv --split test --methods <dataset methods> \
  --dataset-name <Dataset> --real-source <dataset>_real \
  --identity-csv data/splits/<dataset>_identities.csv \
  --calib results/calib/<model>_seed<k>.json \
  --save-scores results/scores/<model>_seed<k>_<dataset>.csv --skip-sweep
```

### 3.4 Before opening any score

```bash
sha256sum results/scores/*_<dataset>.csv > results/scores/<dataset>_sha256.txt
git add results/scores/*_<dataset>.csv results/scores/<dataset>_sha256.txt
git commit -m "results: confirmatory <dataset> scores (unopened)"
```

## 4. Run the registered analysis (day 8-9, about 15 minutes of CPU)

```bash
python -m tools.real_shot --targets <dataset> --checks-on <dataset> --primary loc \
  --tag _confirm_<dataset> --real-strata
python -m tools.fpr_control --source ffppval   # zero-shot FPR table, all datasets incl. the new one
python -m tools.paper_figures --target <dataset> --tag _confirm_<dataset> \
  --out paper/figures/confirm
```

Report the H7-v2 verdicts in `results/tables/real_shot_verdicts_confirm_<dataset>.json`
as they come out. If H7b-v2 fails, the paper still stands as a benchmark. In that case
it reports the smallest n at which conformal holds on the new dataset, and the claim
becomes "n depends on the domain; here is how to measure it".


## 5. Rewrite the paper (days 2-12, in parallel with step 3)

The LaTeX in `paper/` still argues the old story and contains claims the current results
contradict. Rewrite it; do not patch it sentence by sentence.

**Delete or correct** (found by grep on 2026-11-12):

- `main.tex` title and abstract: "matching the 103,425-parameter footprint", "recipe
  alone accounts for +5.83%", "added head capacity systematically inflates false alarms
  ... 16.9-22.5%", "55.1% FPR at threshold 0.50" (an uncalibrated cutoff).
- `03_method.tex:19` and `04_experiments.tex:18`: 103,425 trainable for `smsdt_s`.
- `01_introduction.tex:20`, `04_experiments.tex:75`, `06_discussion.tex:8`: "identical
  trainable parameters" and "optimization recipe dominates".
- Any "in-house reproduction of GenD" (it is a GenD-style baseline, not a reproduction).
- Any DFD number from the 2,987-video files, and any TPR@FPR not taken from
  `results/tables/experiment_matrix.csv`.

**New structure** (working title: *How many real videos does a deepfake detector need to
keep its false-alarm promise?*):

| Section | Content | Source |
|---|---|---|
| 1 Introduction | Deployment budget problem; contributions as a benchmark | this file §0 |
| 2 Related work | Detector generalisation (GenD, CLIP PEFT); conformal and NP thresholds; calibration transfer; speech-deepfake threshold audit | prior-art reading (step 6) |
| 3 Problem and protocol | Real-shot setting; identity-disjoint repeated splits; metrics incl. identity-weighted FPR | `docs/real_shot_benchmark.md` §§2-3 |
| 4 Detectors | 5 detector families, 13 runs; parameter table from `docs/param_counts.md` | research_v2 §2 |
| 5 Zero-shot thresholds fail | Fig 1; label-free repairs fail | research_v2 §§6-7, 9-10 |
| 6 Real-shot calibration | Figs 2-4; main table at n = 20; confirmatory dataset first, Celeb-DF as development | step 4 outputs |
| 7 Why shape transfer fails | Fig 5; negative result | §11.1 |
| 8 Architecture ablations | H1, §10.4 token vs CLS pathway | research_v2 §§4.3, 10.4 |
| 9 Limitations | Exchangeability; video vs identity weighting; few datasets; n = 20 is the sample max at alpha = 0.05 | this plan |
| Appendix | Pre-registration and full deviation log D1-D19; all per-run tables | `docs/decisions.md` |

Say in the paper that H7-v2 was written after the development run, and that Celeb-DF is
development data. TMLR reviewers check claims against evidence. A disclosed revision is
fine; an undisclosed one is not.

**Owners (suggested):** one author writes §§1-3 and 9, one writes §§4-5 and 8, one writes
§§6-7 and runs the figures. Swap drafts on day 10 for a hostile read.

## 6. Prior art (days 1-3; one author)

Read in full, and write a short paragraph for each on how it differs from our setting:

- arXiv 2606.21584: threshold-transfer audit for speech deepfake detectors (no labelled
  target reals);
- arXiv 2505.05084: conformal FPR control for machine-generated text detection;
- arXiv 2202.07650: conformal prediction sets with limited false positives;
- arXiv 2609.10737: conformal calibration transfer without target labels;
- Tong, Feng & Li (2018): Neyman-Pearson umbrella algorithm.

Search once more for "conformal", "false positive rate" and "deepfake video" in 2026
papers. If someone already published real-shot calibration for video deepfakes, the
paper becomes a replication-plus-benchmark, and the framing in §1 must say so.

## 7. Submission (days 12-18)

- TMLR template (`paper/tmlr.sty`), anonymised. Remove the repository URL and team names,
  and use an anonymous code link.
- Code and score files: release the score CSVs and `tools/`. They are what makes the
  benchmark reusable.
- Broader-impact statement: deepfake detection is dual-use; false alarms on authentic
  people have real costs, which is the point of the paper.
- Final check: every number in the PDF must trace to a file under `results/tables/`.
  Grep the .tex for numbers and spot-check twenty of them.

## 8. Timeline

| Dates | Step | Who |
|---|---|---|
| Nov 13 | Apply patch; request dataset access (two datasets) | Aditya |
| Nov 13-14 | Sign-off and freeze (step 2) | all three |
| Nov 13-15 | Prior-art reading (step 6) | one author |
| Nov 14-20 | Dataset prep and scoring (step 3) | GPU owner |
| Nov 13-24 | Paper rewrite (step 5) | all, by section |
| Nov 21 | Registered analysis (step 4) | GPU owner |
| Nov 22-26 | Results sections, figures, appendix | all |
| Nov 27-29 | Internal hostile read, number trace | all |
| Nov 30 | Submit | Aditya |

If dataset access is not granted by Nov 18, choose between waiting and submitting with
Celeb-DF as development-only evidence. In the second case the paper must say the H7 result
is unconfirmed, which makes acceptance less likely. Waiting is better.
