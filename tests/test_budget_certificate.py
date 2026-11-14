"""
Unit tests for the Real-Shot Budget Certificate (tools/budget_certificate.py).
Verifies:
1. Conformal rank and expected FPR calculations.
2. Beta-Binomial evaluation violation rate predictions.
3. Sweet-spot budget selection (n in {19, 39, 59} for alpha=0.05).
4. Triage threshold calculation and classification.
5. Full 130-cell evaluation artifact integrity.
"""
import os
import json
import pytest
import numpy as np
from tools.budget_certificate import (
    conformal_rank,
    expected_fpr,
    predict_eval_violation_rate,
    sweet_spot_ns,
    sample_size_planner,
    compute_triage_thresholds,
    triage_classify,
)


def test_conformal_rank_and_expected_fpr():
    # n=19, alpha=0.05 -> rank k = ceil(20*0.95) = 19, expected FPR = (20-19)/20 = 0.05
    k1 = conformal_rank(19, 0.05)
    assert k1 == 19
    assert abs(expected_fpr(19, 0.05) - 0.05) < 1e-6

    # n=39, alpha=0.05 -> rank k = ceil(40*0.95) = 38, expected FPR = (40-38)/40 = 0.05
    k2 = conformal_rank(39, 0.05)
    assert k2 == 38
    assert abs(expected_fpr(39, 0.05) - 0.05) < 1e-6

    # n=20, alpha=0.05 -> rank k = ceil(21*0.95) = ceil(19.95) = 20, expected FPR = (21-20)/21 = 1/21 ~ 0.0476
    k3 = conformal_rank(20, 0.05)
    assert k3 == 20
    assert abs(expected_fpr(20, 0.05) - (1.0 / 21.0)) < 1e-6


def test_sweet_spot_ns():
    budgets_5 = sweet_spot_ns(alpha=0.05, max_n=60)
    assert budgets_5 == [19, 39, 59]

    budgets_10 = sweet_spot_ns(alpha=0.10, max_n=30)
    assert budgets_10 == [9, 19, 29]


def test_predict_eval_violation_rate():
    # At m=704, n=19, alpha=0.05, r=2.0 (i.e. target threshold is 2*0.05 = 0.10)
    p_viol = predict_eval_violation_rate(m=704, n=19, alpha=0.05, r=2.0)
    assert p_viol is not None
    assert 0.0 <= p_viol <= 1.0


def test_sample_size_planner():
    plan_df = sample_size_planner(alpha=0.05, delta=0.05, r=2.0)
    assert "is_sweet_spot" in plan_df.columns
    assert "meets_delta" in plan_df.columns
    sweet_spots = plan_df[plan_df["is_sweet_spot"]]["n"].tolist()
    assert sweet_spots[:3] == [19, 39, 59]


def test_triage_classification():
    calib = np.linspace(0.1, 0.9, 100)
    thrs = compute_triage_thresholds(calib, alpha=0.05, beta=0.10)
    assert "thr_alert" in thrs
    assert "thr_review" in thrs
    assert thrs["thr_alert"] >= thrs["thr_review"]

    test_scores = np.array([0.05, 0.85, 0.95])
    classes = triage_classify(test_scores, thrs["thr_alert"], thrs["thr_review"])
    assert len(classes) == 3
    assert set(classes).issubset({"PASS", "REVIEW", "FLAG"})


def test_h8_eval_artifacts_exist_and_pass():
    csv_path = "results/tables/h8_certificate_eval.csv"
    verdict_path = "results/tables/h8_certificate_verdict.json"

    assert os.path.exists(csv_path), f"Missing {csv_path}"
    assert os.path.exists(verdict_path), f"Missing {verdict_path}"

    with open(verdict_path, "r") as f:
        data = json.load(f)

    assert data.get("total_cells") == 130
    assert data.get("ci_pass_cells") == 130
    assert data.get("ci_pass_rate") == 1.0
