# Making the core finding credible

Candidate finding (H4): ST and Full inflate false-alarm rate under dataset shift without
a cross-dataset AUC gain over S. Everything below is ordered by how much it raises
credibility per unit of effort. Commands assume the repo layout in `docs/score_schema.md`.

## 0. Do not touch results until sign-off
All three authors sign H4a/H4b/H5/H6 in `docs/decisions.md` (date the signatures). Seeds
looked at before signing (1, 2) stay exploratory. This is what makes seeds 4-5 confirmatory.

## 1. Fix the measurement (hours)
- `python -m pytest tests/test_credibility_fixes.py` (new tests).
- Export dev-split scores for every run and use them as the threshold source, not FF++ test:
  `python -m smsdt.eval_cross --ckpt <ckpt> --split val --save-scores results/scores/<run>_ffppval.csv`
- `bash scripts/run_credibility.sh` re-runs stats, threshold transfer, prior sweep and
  FPR control with the fixed CIs. Check `n_real_clusters` (should be > 30) in `fpr_control.csv`.

## 2. Add the missing comparator: official GenD on FF++ (hours)
`python -m smsdt.eval_cross --ckpt hf:yermandy/GenD_CLIP_L_14 --split test --dataset-name ffpp --save-scores results/scores/gend_official_seed0_ffpp.csv`
(and `--split val` for the dev file). Then `fpr_control` can compute its FPR inflation.
- Official GenD inflates too: the finding is a general shift effect; reframe as a benchmarking caution.
- Official GenD stays near alpha: the finding is specific to the larger heads. Stronger.

## 3. Find the cause (1-2 days)
Cheap, no retraining: for each Full checkpoint run
`--readout-ablation mean_patch|lse_only|att_only` on FF++ and Celeb-DF, then `fpr_control`.
If `mean_patch` removes the inflation, LSE pooling is the cause, not capacity.
Decisive, needs training (3 seeds each): Full with mean readout; ST without TemporalDiff;
S evaluated as 8-frame average. Pre-register which outcome means what before running.

## 4. More data (days)
- Seeds 4 and 5 for the four models (confirmatory, with the signed protocol).
- One more unseen dataset (DFDC preview or WildDeepfake; DF40 subsets if licences allow).
  Do not tune anything on it. Report per-dataset, not pooled.
- Report FPR at alpha in {0.01, 0.05, 0.10} only where reals >= 10/alpha (Celeb-DF: 0.05, 0.10).

## 5. Pre-empt reviewers
- Matched-AUC comparison: compare FPR at equal source TPR, not only at equal source FPR.
- Calibration controls: temperature scaling on the dev split; does the inflation survive?
- State the Celeb-DF real-video limits (178 test reals, 124 identity clusters) and use the
  conservative CI for every verdict.
- Prior art: read (not only skim) T2A 2505.18787, AIGI calibration 2602.01973, DF40 and
  the speech-threshold audit 2606.21584 before claiming novelty in the introduction.

## 6. Decision table after steps 1-4
| Outcome | Paper |
|---|---|
| H4a supported for Full and ST; official GenD does not inflate; ablation points to capacity | "Capacity buys no AUC but costs false-alarm stability" + negative results on label-free fixes |
| H4a supported; ablation points to LSE readout | "Pooling choice drives threshold instability" (narrower) |
| Official GenD inflates as well | Benchmarking caution on fixed-budget transfer; S-MSDT as case study |
| H4a not supported on seeds 3-5 | Negative-results paper (H2, H3a, H5, H6 falsified); drop the finding |
