# Re-run every audit after the 2026-11-10 fixes (see docs/novelty_audit.md, section 5).
# Prerequisites: results/scores/*.csv (test + *_ffppval dev scores), results/calib/*.json.
# Usage (repo root):  powershell -ExecutionPolicy Bypass -File scripts/run_novelty_audit.ps1
$ErrorActionPreference = "Stop"

Write-Host "== H1: pre-registered comparisons vs S (pure negatives, Bonferroni, 10k resamples)"
python -m tools.stats

Write-Host "== H2: threshold transfer on the calibrated scale, CI of g, FPR inflation"
python -m tools.threshold_transfer

Write-Host "== H3a: prior sweep with calibrated posteriors (+ exploratory accuracy gain)"
python -m tools.prior_sweep

Write-Host "== H4/H5: FPR control under shift (needs *_ffppval.csv dev scores)"
python -m tools.fpr_control

Write-Host "Done. Tables in results/tables/. Record verdicts in docs/decisions.md."
