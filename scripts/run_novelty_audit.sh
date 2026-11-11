#!/usr/bin/env bash
# Re-run every audit after the 2026-11-10 fixes (see docs/novelty_audit.md, section 5).
set -euo pipefail
python -m tools.stats
python -m tools.threshold_transfer
python -m tools.prior_sweep
python -m tools.fpr_control
echo "Done. Tables in results/tables/. Record verdicts in docs/decisions.md."
