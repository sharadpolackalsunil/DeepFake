"""Tests for tools/real_shot.py (synthetic; guarantees checked against population FPR)."""
import os
import sys

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm, t as student_t

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.real_shot import (SourceRealModel, conformal_rank, few_shot_threshold, h7_run_checks,
                             ls_transfer_threshold, min_n_np_umbrella, np_umbrella_rank, real_shot_curve,
                             select_primary, source_standardised_quantile, transfer_threshold)
from tools.stats import apply_pure_negatives


# --- order-statistic rules -------------------------------------------------

def test_conformal_rank():
    assert conformal_rank(19, 0.05) == 19
    assert conformal_rank(18, 0.05) is None
    assert conformal_rank(99, 0.05) == 95


def test_np_umbrella_min_n():
    assert min_n_np_umbrella(0.05, 0.05) == 59
    assert np_umbrella_rank(58, 0.05) is None
    assert np_umbrella_rank(59, 0.05) == 59
    k = np_umbrella_rank(200, 0.05)
    assert k is not None and k < 200


def _pop_fpr_normal(thr, mu=0.0, sd=1.0):
    return 0.0 if not np.isfinite(thr) else float(norm.sf((thr - mu) / sd))


def test_conformal_marginal_guarantee():
    rng = np.random.default_rng(0)
    for n in (19, 40):
        fprs = [_pop_fpr_normal(few_shot_threshold("conformal", rng.normal(size=n), 0.05, None, rng))
                for _ in range(4000)]
        assert np.mean(fprs) <= 0.05 + 0.003


def test_np_umbrella_high_probability_guarantee():
    rng = np.random.default_rng(1)
    fprs = np.array([_pop_fpr_normal(few_shot_threshold("np_umbrella", rng.normal(size=80), 0.05, None, rng))
                     for _ in range(3000)])
    assert np.mean(fprs > 0.05) <= 0.05 + 0.012


def test_ls_transfer_exact_under_location_scale_shift():
    rng = np.random.default_rng(2)
    src = rng.standard_t(5, size=20000)
    q = source_standardised_quantile(src, 0.05)
    tgt = 1.7 + 2.3 * rng.standard_t(5, size=20000)          # same shape, new location/scale
    thr = ls_transfer_threshold(tgt, q)
    true = 1.7 + 2.3 * student_t.ppf(0.95, 5)
    assert abs(thr - true) < 0.1


def test_ls_transfer_is_biased_when_shape_changes():
    """Honest failure mode: a heavier target tail breaks the location-scale assumption."""
    rng = np.random.default_rng(3)
    src = rng.normal(size=20000)
    q = source_standardised_quantile(src, 0.01)
    tgt = rng.standard_t(2, size=20000)
    thr = ls_transfer_threshold(tgt, q)
    assert float(np.mean(tgt > thr)) > 0.015


def test_ucb_more_conservative_than_point():
    rng = np.random.default_rng(4)
    src = SourceRealModel(rng.normal(size=500), 0.05)
    x = rng.normal(size=15)
    for mode in ("loc", "ls", "shrink"):
        a = few_shot_threshold(f"{mode}_transfer", x, 0.05, src, rng)
        b = few_shot_threshold(f"{mode}_ucb", x, 0.05, src, rng)
        assert b >= a


def test_shrink_interpolates_between_loc_and_ls():
    rng = np.random.default_rng(8)
    src = SourceRealModel(rng.normal(size=2000), 0.05)
    x_small = 2.0 * rng.normal(size=5)
    x_big = 2.0 * rng.normal(size=2000)
    for x in (x_small, x_big):
        lo, ls, sh = (transfer_threshold(x, src, m) for m in ("loc", "ls", "shrink"))
        assert min(lo, ls) - 1e-9 <= sh <= max(lo, ls) + 1e-9
    # with many reals shrink is close to ls
    assert abs(transfer_threshold(x_big, src, "shrink") - transfer_threshold(x_big, src, "ls")) < 0.05


def test_loc_ucb_population_violation_small_under_shift():
    rng = np.random.default_rng(5)
    src_x = rng.normal(size=5000)
    src = SourceRealModel(src_x, 0.05)
    fprs = np.array([_pop_fpr_normal(few_shot_threshold("loc_ucb", 0.5 + rng.normal(size=20), 0.05, src, rng),
                                     mu=0.5) for _ in range(600)])
    assert np.mean(fprs > 0.05) <= 0.12      # nominal delta 0.05; bootstrap of the median is approximate


# --- curve protocol --------------------------------------------------------

def _target(rng, n_ids=80, per_id=2, shift=0.8):
    rows = []
    for i in range(n_ids):
        for j in range(per_id):
            rows.append({"video_id": f"celebdf_Celeb-real_id{i}_{j:04d}", "label": 0,
                         "logit_mean": shift + rng.normal()})
    for k in range(300):
        rows.append({"video_id": f"celebdf_Celeb-synthesis_id{k % n_ids}_id0_{k:04d}", "label": 1,
                     "logit_mean": 3 + rng.normal()})
    df = pd.DataFrame(rows)
    df["cluster_id"] = np.where(df.label == 0, "Celeb-real", "Celeb-synthesis")  # coarse, as in the real files
    return df


def _source(rng):
    return pd.DataFrame({"label": [0] * 140 + [1] * 700,
                         "logit_mean": np.r_[rng.normal(size=140), rng.normal(3, 1, 700)]})


def test_curve_shapes_and_identity_disjointness():
    rng = np.random.default_rng(6)
    c = real_shot_curve(_target(rng), _source(rng), ns=(10, 20), alphas=(0.05,), n_rep=60)
    assert set(c.policy) >= {"source_quantile", "conformal", "np_umbrella", "ls_transfer", "ls_ucb", "oracle"}
    np_row = c[(c.policy == "np_umbrella") & (c.n_draw == 20)].iloc[0]
    assert np_row.abstain_rate == 1.0                       # n = 20 < 59
    cf10 = c[(c.policy == "conformal") & (c.n_draw == 10)].iloc[0]
    assert cf10.abstain_rate == 1.0                         # n = 10 < 19
    assert c.n_real_clusters.iloc[0] == 80                  # coarse cluster_id was not used


def test_source_threshold_fails_and_transfer_recovers_under_shift():
    rng = np.random.default_rng(7)
    c = real_shot_curve(_target(rng, shift=0.8), _source(rng), ns=(10,), alphas=(0.05,), n_rep=150)
    src = c[(c.policy == "source_quantile")].iloc[0]
    lo = c[(c.policy == "loc_transfer")].iloc[0]
    assert src.mean_fpr > 0.12                              # shifted reals blow the budget
    assert lo.mean_abs_err < 0.5 * src.mean_abs_err
    sel = select_primary(c, n=10)
    assert sel["primary"] in ("loc", "ls", "shrink")
    checks = h7_run_checks(c, primary="loc", n_small=10, n_mid=10)
    assert checks["h7a_pass"]


def test_pure_negatives_drop_mislabelled_ffpp_reals_from_dfd():
    df = pd.DataFrame({
        "video_id": ["real_012", "real_953", "dfd_real_01__kitchen_pan", "DFD_01_02__x"],
        "label": [0, 0, 0, 1],
        "real_source": ["dfd_actor", "dfd_actor", "dfd_actor", np.nan],
        "logit_mean": [0.1, 0.2, 0.3, 2.0],
    })
    out, info = apply_pure_negatives(df, "dfd")
    assert info["n_real_kept"] == 1 and info["n_real_removed"] == 2
    assert set(out.video_id) == {"dfd_real_01__kitchen_pan", "DFD_01_02__x"}


def test_threshold_transfer_uses_resolved_clusters():
    """Regression for D13: Celeb-DF's cluster_id has 3 values; the gap CI must not collapse."""
    from tools.threshold_transfer import audit_threshold_transfer
    rng = np.random.default_rng(9)
    df = _target(rng)
    df["score"] = 1 / (1 + np.exp(-df["logit_mean"]))
    res = audit_threshold_transfer(df, tau_val=0.75, score_col="score", n_boot=200)
    lo, hi = res["gap_g_ci95"]
    assert hi - lo > 0.01          # collapsed to ~0 width before the fix


def test_smsdt_v3_branch_mode_switch():
    import torch
    from smsdt.models import smsdt_v3
    src = open(smsdt_v3.__file__).read()
    assert 'branch_mode' in src and 'cls_only' in src and 'token_only' in src


def test_resample_source_widens_fpr_ci():
    from tools.fpr_control import evaluate_policies
    rng = np.random.default_rng(10)
    t, src = _target(rng), _source(rng)
    a = evaluate_policies(t, src, alphas=[0.05], n_boot=300)
    b = evaluate_policies(t, src, alphas=[0.05], n_boot=300, resample_source=True)
    qa = a[a.policy == "source_quantile"].iloc[0]
    qb = b[b.policy == "source_quantile"].iloc[0]
    assert (qb.fpr_ci_high - qb.fpr_ci_low) > (qa.fpr_ci_high - qa.fpr_ci_low)


def test_h7v2_verdicts_on_synthetic_curve():
    from tools.real_shot import h7v2_dataset_verdicts
    rows = []
    for model, src_p, gmm_p in (("m1", 0.9, 0.0), ("m2", 0.0, 0.8)):
        base = {"model": model, "seed": "seed1", "alpha": 0.05, "n_draw": 20, "n_real_clusters": 100}
        rows += [dict(base, policy="source_quantile", p_fpr_gt_2alpha=src_p, mean_fpr=0.1, mean_tpr=0.8),
                 dict(base, policy="target_gmm", p_fpr_gt_2alpha=gmm_p, mean_fpr=0.1, mean_tpr=0.5),
                 dict(base, policy="conformal", p_fpr_gt_2alpha=0.1, mean_fpr=0.045, mean_tpr=0.50),
                 dict(base, policy="oracle", p_fpr_gt_2alpha=0.0, mean_fpr=0.05, mean_tpr=0.60),
                 dict(base, policy="naive_quantile", p_fpr_gt_2alpha=0.3, mean_fpr=0.09, mean_tpr=0.7)]
    v = h7v2_dataset_verdicts(pd.DataFrame(rows))
    assert v["h7a_v2"] == "supported" and v["h7b_v2"] == "supported" and v["h7c_v2"] == "supported"
    small = pd.DataFrame(rows).assign(n_real_clusters=18)
    assert h7v2_dataset_verdicts(small)["h7b_v2"].startswith("undetermined")
