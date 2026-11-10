"""Unit tests for tools/stats.py statistical testing and paired bootstrap."""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import pytest

from tools.stats import fast_auc, delong_roc_variance, paired_cluster_boot_dauc


def test_fast_auc_analytical():
    # Perfect separation
    y = np.array([0, 0, 0, 1, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    assert pytest.approx(fast_auc(y, s), rel=1e-5) == 1.0

    # Inverted separation
    assert pytest.approx(fast_auc(y, 1.0 - s), rel=1e-5) == 0.0

    # Half separation
    s_half = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    assert pytest.approx(fast_auc(y, s_half), rel=1e-5) == 0.5


def test_delong_variance_positive():
    rng = np.random.RandomState(42)
    y = np.array([0] * 50 + [1] * 50)
    s = np.concatenate([rng.normal(0, 1, 50), rng.normal(1.5, 1, 50)])
    auc, var = delong_roc_variance(y, s)
    assert 0.7 < auc < 1.0
    assert var > 0.0
    assert var < 0.05


def test_paired_cluster_boot_identical_models():
    """When Model A == Model B, CI should be centered at 0 and classified as equivalent."""
    rng = np.random.RandomState(123)
    n_clusters = 30
    records = []
    
    for c in range(n_clusters):
        c_id = f"person_{c}"
        label = 1 if c % 2 == 0 else 0
        n_vids = rng.randint(2, 6)
        for v in range(n_vids):
            base_score = 0.8 if label == 1 else 0.2
            noise = rng.normal(0, 0.05)
            score = float(np.clip(base_score + noise, 0.0, 1.0))
            records.append({
                "video_id": f"{c_id}_vid_{v}",
                "cluster_id": c_id,
                "label": label,
                "score_mean": score
            })
            
    df = pd.DataFrame(records)
    # 5 identical seeds for both models
    dfs_a = [df.copy() for _ in range(5)]
    dfs_b = [df.copy() for _ in range(5)]
    
    res = paired_cluster_boot_dauc(dfs_a, dfs_b, n_boot=1000, delta=0.01)
    
    assert pytest.approx(res["mean_delta"], abs=1e-6) == 0.0
    assert res["ci_lower"] <= 0.0 <= res["ci_upper"]
    assert res["classification"] == "equivalent"


def test_paired_cluster_boot_superior_model():
    """When Model A is strictly superior to Model B by > delta, classified as better."""
    rng = np.random.RandomState(456)
    n_clusters = 50
    records_a = []
    records_b = []
    
    for c in range(n_clusters):
        c_id = f"actor_{c}"
        label = 1 if c % 2 == 0 else 0
        n_vids = rng.randint(3, 7)
        for v in range(n_vids):
            # Model A has high discrimination
            score_a = float(np.clip(rng.normal(0.85 if label == 1 else 0.15, 0.05), 0, 1))
            # Model B has much worse discrimination
            score_b = float(np.clip(rng.normal(0.60 if label == 1 else 0.40, 0.15), 0, 1))
            
            records_a.append({
                "video_id": f"{c_id}_v_{v}",
                "cluster_id": c_id,
                "label": label,
                "score_mean": score_a
            })
            records_b.append({
                "video_id": f"{c_id}_v_{v}",
                "cluster_id": c_id,
                "label": label,
                "score_mean": score_b
            })
            
    df_a = pd.DataFrame(records_a)
    df_b = pd.DataFrame(records_b)
    
    dfs_a = [df_a.copy() for _ in range(5)]
    dfs_b = [df_b.copy() for _ in range(5)]
    
    res = paired_cluster_boot_dauc(dfs_a, dfs_b, n_boot=1000, delta=0.01)
    
    assert res["mean_delta"] > 0.05
    assert res["ci_lower"] > 0.01
    assert res["classification"] == "better"
