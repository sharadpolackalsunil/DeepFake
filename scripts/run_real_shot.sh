#!/usr/bin/env bash
# Real-shot budget transfer benchmark (docs/real_shot_benchmark.md).
# Development (already inspected data; never confirmatory for H7):
set -euo pipefail
python -m pytest tests/test_real_shot.py -q
python -m tools.real_shot --targets celebdf --checks-on celebdf --primary loc --tag _dev_celebdf
python -m tools.real_shot --targets dfd --checks-on dfd --primary loc --ns 5 10 --min-eval-clusters 8 --tag _dev_dfd
# Confirmatory (after sign-off of docs/decisions.md section 6; replace <new> with the dataset key):
# python -m tools.real_shot --targets <new> --checks-on <new> --primary loc --tag _confirmatory
