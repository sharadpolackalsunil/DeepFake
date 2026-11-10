#!/usr/bin/env bash
# ==============================================================================
# S-MSDT TMLR STUDY: FULL EXPERIMENT REPRODUCTION SCRIPT
# ==============================================================================
# Regenerates every manuscript table and figure from raw score CSVs in results/scores/
# and frozen calibration files in results/calib/.
# ==============================================================================
set -euo pipefail

echo "======================================================================"
echo "S-MSDT TMLR: REPRODUCIBILITY SUITE"
echo "======================================================================"

echo "[1/4] Running unit test suite..."
python -m pytest tests/test_stats.py tests/test_tta.py -v

echo "[2/4] Verifying parameter counts..."
python tools/count_params.py

echo "[3/4] Generating master experiment matrix and paper tables..."
python tools/make_tables.py

echo "[4/4] Generating figures (landmark overlays, threshold transfer, prior sweep)..."
python tools/overlay_landmarks.py

echo "======================================================================"
echo "Reproduction complete. Output tables in results/tables/ and figures in results/figs/."
echo "======================================================================"

