"""Post-hoc (NOT pre-registered) check of H8 criterion (ii).

The registered H8 run seeds every cell with 42, so all detectors at a given
(target, alpha, n) are evaluated on the same 1,000 random splits and their Monte
Carlo errors are correlated. This script re-runs the FakeAVCeleb cells with fresh
seeds and records, per seed, the mean standardized excess z = (mean FPR - alpha)/SE
and the number of cells failing criterion (ii). It does not change the registered verdict.

Usage: python tools/h8_reseed_check.py   -> results/tables/h8_reseed_check.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.budget_certificate import evaluate_h8_cell, load_dataset_scores  # noqa: E402

SEEDS = [42, 101, 202, 303]


def main() -> None:
    reg = pd.read_csv(ROOT / "results" / "tables" / "h8_certificate_eval.csv")
    rows = []
    for dataset in ["fakeavceleb", "celebdf"]:
        cells = reg[reg.dataset == dataset][["model", "alpha", "n"]].values.tolist()
        for seed in SEEDS:
            zs, fails = [], 0
            for model, alpha, n in cells:
                df = load_dataset_scores(model, dataset, scores_dir=str(ROOT / "results" / "scores"))
                r = evaluate_h8_cell(df, alpha, int(n), R=1000, seed=seed)
                zs.append((r["mean_fpr"] - r["expected_fpr"]) / r["se_mc"])
                fails += int(not r["validity_pass"])
            rows.append(dict(dataset=dataset, seed=seed, cells=len(cells), mean_z=float(np.mean(zs)),
                             max_z=float(np.max(zs)), criterion_ii_failures=fails))
            print(rows[-1])
    pd.DataFrame(rows).to_csv(ROOT / "results" / "tables" / "h8_reseed_check.csv", index=False)


if __name__ == "__main__":
    main()
