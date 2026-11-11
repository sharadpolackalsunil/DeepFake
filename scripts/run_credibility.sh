#!/usr/bin/env bash
# Re-run the analyses after the credibility fixes (see docs/credibility_plan.md).
set -euo pipefail
python -m pytest tests/test_credibility_fixes.py -q
python -m tools.stats
python -m tools.threshold_transfer
python -m tools.prior_sweep
python -m tools.fpr_control --confirmatory-seeds 3 4 5
echo "Check results/tables/fpr_control_verdicts.json (h4a), fpr_control_h4b.csv, and n_real_clusters."
