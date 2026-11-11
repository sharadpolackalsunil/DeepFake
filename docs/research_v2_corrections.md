# Required corrections to `research_v2.md` (2026-11-11 review)

Apply these before any number is copied into a manuscript. Each item names the section,
the problem and the replacement. Re-run the tools first (`scripts/run_credibility.sh`);
where a replacement needs a new number it says `[RERUN]`.

| Section | Problem | Replace with |
|---|---|---|
| 2, 3.2, 4.2 table | `gend_ours` called "reproduced GenD baseline" / "literature reference anchor" | "GenD-style in-house baseline (not a reproduction; Celeb-DF 0.83 vs 0.933 for the official weights)". |
| 4.2, takeaway 1 | "+5.88% improvement over GenD" | "+0.059 AUC over our in-house GenD-style baseline. This is a recipe comparison, not a comparison with GenD: official GenD scores 0.933 on Celeb-DF, above every S-MSDT variant." |
| 4.2, takeaways 2-3 | "proving the gains are intrinsic to the architecture", "balanced loss and curriculum sampling provide +5.32%" | Delete. Three seeds cannot prove this and no ablation isolates loss or curriculum. State: "seed sd <= 0.0098 (3 seeds)". |
| 4.2, table | "Peaks at 0.8960" | Report mean +- sd only. |
| 4.3 | Legacy analysis (2,000 resamples, no Bonferroni, mixed DFD reals, missing Full vs S) | `[RERUN]` `python -m tools.stats`; report the two confirmatory pairs (ST vs S, Full vs S) with Bonferroni, 10,000 resamples. Expected from the point estimates: Full vs S about +0.006 and ST vs S about -0.012 on Celeb-DF, i.e. no AUC gain from the extra modules. Say so. |
| 5 | H2 verdict from the legacy run (tau_val = 0.5000 for every smsdt model means the raw-0.5 fallback was used; only seed 1 shown; "g <= 0.0374" not visible in the table) | `[RERUN]` `python -m tools.threshold_transfer` with dev-split calibration files; show all 36 rows; give the lower CI of g; then state the verdict. Until then write "H2: not yet evaluated as registered". |
| 6 | One model, one dataset, raw scores | Keep the correction note; add that Bayes-decision accuracy (not HTER) is the right metric, from the fixed `prior_sweep.py`. |
| 7 | "Partial mitigation" for online consistency TTA at 0.54 AUC | "H6 falsified: online consistency TTA collapses (AUC 0.54, predicted-positive 10.8%); batch-episodic changes AUC by -0.0003. Single model, seed and dataset." Also state that the batch-episodic entropy rows (+0.0000) measured logits before the reset and are not a valid test. |
| 8 | Fine | Add "single model, seed 1, Celeb-DF only". |
| 9 | "55.1% false-alarm catastrophe" compared with source-quantile thresholds of our models; unverified recipe explanation | Compare like with like: `[RERUN]` official GenD FF++ scores, then its source-quantile FPR on Celeb-DF. State the raw-0.5 FPR only as "its softmax is not calibrated at 0.5". Remove the list "multi-source mixture, cycling LR, 32 frames, cross-dataset checkpoint selection" unless each item is cited to the GenD paper or config (`third_party/GenD`). |
| 10.1-10.2 | CIs narrower than binomial (e.g. [0.2222, 0.2286] at n=178) | `[RERUN]` `python -m tools.fpr_control` (fixed clustering; reports `fpr_ci_low_cons`). |
| 10, verdict 1 | "H4 CONFIRMED" | Use the `h4a` block of `fpr_control_verdicts.json`. Seeds 1-2 are exploratory. Wilson bounds from the reported point estimates: Full lower bound > 0.10 on seeds 1-3; ST only on seed 3. Until seeds 4-5 exist write "H4a supported for Full, not established for ST". Report H4b from `fpr_control_h4b.csv`. |
| 10.3 | `degenerate_rate = 0.0` next to TPR = 0.000 / FPR = 0.000 | `[RERUN]`; the fixed tool now flags thresholds outside the logit support as degenerate. The H5 verdict stays "falsified" but the explanation must come from the re-run. |
| 11.2 | "Overclaiming scrubbed" ticked | Untick until the rows above are done. |
