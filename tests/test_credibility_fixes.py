"""Tests for the credibility fixes: bootstrap clustering, conservative CIs, H4a/H4b, GMM support check."""
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.fpr_control import (evaluate_policies, gmm_threshold, h4a_verdict, h4b_table,
                               resolve_clusters)


def _target(rng, n_r=178, n_f=340, coarse_cluster=False):
    r = rng.normal(0, 1, n_r)
    f = rng.normal(2.5, 1.2, n_f)
    ids = [f"id{i % 60}_{i:04d}" for i in range(n_r + n_f)]
    df = pd.DataFrame({"video_id": ids, "label": [0] * n_r + [1] * n_f, "logit_mean": np.r_[r, f]})
    # cluster_id that is a coarse label (NOT an identity grouping)
    df["cluster_id"] = ["real" if l == 0 else "fake" for l in df["label"]] if coarse_cluster else df["video_id"]
    return df


def _source(rng):
    return pd.DataFrame({"label": [0] * 140 + [1] * 700,
                         "logit_mean": np.r_[rng.normal(-1, 1, 140), rng.normal(3, 1, 700)]})


def test_coarse_cluster_id_is_not_used_for_bootstrap():
    rng = np.random.default_rng(0)
    t = _target(rng, coarse_cluster=True)
    cl = resolve_clusters(t)
    assert len(np.unique(cl)) > 30


def test_ci_width_matches_binomial_scale_when_cluster_id_is_coarse():
    """Regression: a coarse cluster_id collapsed the CI to a point. With the resolver the
    95% CI on ~20% FPR over 178 reals must be wider than +-0.03."""
    rng = np.random.default_rng(1)
    t, s = _target(rng, coarse_cluster=True), _source(rng)
    q = evaluate_policies(t, s, alphas=[0.05], n_boot=300)
    q = q[q.policy == "source_quantile"].iloc[0]
    assert q.fpr_ci_high - q.fpr_ci_low > 0.06
    assert q.fpr_ci_low_cons <= q.fpr_ci_low and q.fpr_ci_high_cons >= q.fpr_ci_high
    assert q.n_real_clusters > 30


def test_warns_when_reals_have_few_clusters():
    rng = np.random.default_rng(2)
    t = _target(rng).drop(columns=["video_id"])  # no identity information available
    t["cluster_id"] = ["a" if i % 2 else "b" for i in range(len(t))]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        evaluate_policies(t, _source(rng), alphas=[0.05], n_boot=5)
    assert any("distinct clusters" in str(x.message) for x in w)


def test_gmm_threshold_outside_support_is_degenerate():
    rng = np.random.default_rng(3)
    # almost all fakes: real component fitted on a thin tail pushes the threshold outside support
    x = np.r_[rng.normal(0, 0.05, 3), rng.normal(5, 1, 400)]
    thr, info = gmm_threshold(x, 0.05)
    if thr > x.max() or thr < x.min():
        assert info["degenerate"] and info["outside_support"]


def _h4_df(rows):
    return pd.DataFrame(rows, columns=["model", "seed", "dataset", "alpha", "fpr_ci_low_cons"])


def test_h4a_counts_only_confirmatory_seeds():
    rows = []
    for m in ("smsdt_st", "smsdt_full"):
        for sd in ("seed1", "seed2", "seed3"):
            rows.append([m, sd, "celebdf", 0.05, 0.15])
    v = h4a_verdict(_h4_df(rows), confirmatory_seeds=(3, 4, 5))
    assert v["supported"].startswith("undetermined")  # only seed 3 is confirmatory


def test_h4a_supported_and_not_supported():
    def mk(st_low):
        rows = []
        for sd in ("seed3", "seed4", "seed5"):
            rows.append(["smsdt_full", sd, "celebdf", 0.05, 0.15])
            rows.append(["smsdt_st", sd, "celebdf", 0.05, st_low])
        return _h4_df(rows)
    assert h4a_verdict(mk(0.12))["supported"] == "supported"
    assert h4a_verdict(mk(0.08))["supported"] == "not supported"


def test_h4b_paired_delta_sign():
    rng = np.random.default_rng(4)
    ids = [f"id{i % 50}_{i}" for i in range(400)]
    lab = np.r_[np.zeros(160), np.ones(240)].astype(int)
    base = rng.normal(0, 1, 400)
    big = base + np.where(lab == 0, 0.8, 0.0)  # model A has shifted reals
    def df(u):
        return pd.DataFrame({"video_id": ids, "label": lab, "logit_mean": u, "cluster_id": ids})
    src = _source(rng)
    runs = {("smsdt_full", "seed1", "celebdf"): df(big), ("smsdt_s", "seed1", "celebdf"): df(base),
            ("smsdt_full", "seed1", "ffpp"): src, ("smsdt_s", "seed1", "ffpp"): src}
    out = h4b_table(runs, pairs=(("smsdt_full", "smsdt_s"),), alphas=(0.05,), n_boot=200)
    assert len(out) == 1 and out.iloc[0]["delta"] > 0 and out.iloc[0]["ci_low"] > 0


def test_readout_ablation_switch_changes_output():
    import torch
    from smsdt.models.v3_modules import EvidenceReadoutV3
    torch.manual_seed(0)
    r = EvidenceReadoutV3(dim=32, heads=4, n_queries=2).eval()
    r.mix.data.fill_(0.5)
    z = torch.randn(2, 2, 16, 32)
    with torch.no_grad():
        a, _ = r(z)
        r.readout_mode = "mean_patch"
        b, _ = r(z)
        r.readout_mode = "att_only"
        c, _ = r(z)
        r.readout_mode = "lse_only"
        d, _ = r(z)
    assert not torch.allclose(a, b) and not torch.allclose(a, c) and not torch.allclose(c, d)
