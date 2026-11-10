"""Test suite for research tools: threshold transfer audit, prior sweep, and table generation."""
import os
import sys
import numpy as np
import pandas as pd
import pytest

from tools.threshold_transfer import compute_rates, find_oracle_tau, audit_threshold_transfer
from tools.prior_sweep import saerens_em_prior, resample_to_prior, run_prior_sweep
from tools.make_tables import compute_dataset_metrics, build_experiment_matrix


@pytest.fixture
def sample_score_df():
    """Create realistic mock score DataFrame conforming to score_schema.md."""
    np.random.seed(42)
    n = 100
    labels = np.array([0] * 50 + [1] * 50)
    # Real logits mean ~ -2.0, Fake logits mean ~ +2.0
    logits = np.concatenate([
        np.random.normal(-2.0, 1.0, 50),
        np.random.normal(2.0, 1.0, 50)
    ])
    scores = 1.0 / (1.0 + np.exp(-logits))
    topk50 = scores + 0.05 * np.random.uniform(0, 1, n)

    df = pd.DataFrame({
        "video_id": [f"vid_{i:03d}" for i in range(n)],
        "dataset": ["Celeb-DF-v2"] * n,
        "split": ["test"] * n,
        "label": labels,
        "real_source": ["celebdf_real" if l == 0 else "" for l in labels],
        "manipulation": ["" if l == 0 else "Celeb-DF" for l in labels],
        "cluster_id": [f"id{i%10}" for i in range(n)],
        "n_clips": [8] * n,
        "logit_mean": logits,
        "score_mean": scores,
        "score_topk50": topk50
    })
    return df


def test_threshold_transfer_audit(sample_score_df):
    """Verify threshold transfer metrics, operating point gap g, and error decomposition."""
    res = audit_threshold_transfer(sample_score_df, tau_val=0.50)
    assert "tau_val" in res
    assert "tau_oracle" in res
    assert "gap_g" in res
    assert res["gap_g"] >= 0.0
    assert "fpr_by_source" in res
    assert "fnr_by_manipulation" in res
    assert "celebdf_real" in res["fpr_by_source"]
    assert "Celeb-DF" in res["fnr_by_manipulation"]


def test_prior_sweep_and_em(sample_score_df):
    """Verify Saerens EM prior estimation and prior shift sweep."""
    # Test EM convergence on shifted posteriors
    true_target_prior = 0.20
    rng = np.random.RandomState(42)
    resampled = resample_to_prior(sample_score_df, target_prior=true_target_prior, rng=rng)
    actual_fraction = float(np.mean(resampled["label"] == 1))
    assert abs(actual_fraction - true_target_prior) < 0.05

    # Run EM prior estimator
    pi_hat, adj_posteriors = saerens_em_prior(resampled["score_mean"].values, pi_source=0.50)
    assert 0.0 < pi_hat < 1.0
    assert len(adj_posteriors) == len(resampled)

    # Test sweep across multiple priors
    sweep_results = run_prior_sweep(sample_score_df, priors=[0.2, 0.5, 0.8], n_repeats=5, random_seed=42)
    assert len(sweep_results) == 3
    assert "target_prior_pi" in sweep_results.columns
    assert "mean_hter_frozen" in sweep_results.columns
    assert "mean_hter_em_corrected" in sweep_results.columns


def test_make_tables_workflow(sample_score_df, tmp_path):
    """Verify table generation engine reads score files and outputs master experiment matrix."""
    scores_dir = tmp_path / "scores"
    calib_dir = tmp_path / "calib"
    scores_dir.mkdir()
    calib_dir.mkdir()

    # Save mock score files for 2 models
    for m in ["gend_ours", "smsdt_st"]:
        for s in [1, 2]:
            path = scores_dir / f"{m}_seed{s}_celebdf.csv"
            sample_score_df.to_csv(path, index=False)

    df_matrix = build_experiment_matrix(scores_dir=str(scores_dir), calib_dir=str(calib_dir))
    assert len(df_matrix) == 4
    assert "Model" in df_matrix.columns
    assert "celebdf_auc" in df_matrix.columns
