"""Tests for the 2026-11-10 audit changes: H1 statistics fixes, calibrated threshold
transfer, FPR control (H4/H5), official GenD adapter (G1) and clip-consistency TTA (H6).
All synthetic; no data or GPU needed."""
import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from tools.stats import (align_runs_on_video_id, apply_pure_negatives, bonferroni_level,
                         h1_verdict, paired_cluster_boot_dauc, required_size_multiplier)
from tools.calib_utils import (calibrated_probs, frozen_logit_threshold, resolve_decision_scores,
                               wilson_interval)
from tools.threshold_transfer import (audit_threshold_transfer, bootstrap_gap_ci, compute_rates,
                                      find_oracle_tau, h2_verdict)
from tools.prior_sweep import run_prior_sweep
from tools.fpr_control import (evaluate_policies, gmm_threshold, h4_failure, h5_verdict,
                               paired_fpr_delta, prior_stress, quantile_threshold)


def _scores(n_real=120, n_fake=240, shift=2.0, seed=0, n_clusters=40, prefix="v"):
    rng = np.random.default_rng(seed)
    u = np.r_[rng.normal(-shift / 2, 1, n_real), rng.normal(shift / 2, 1, n_fake)]
    y = np.r_[np.zeros(n_real), np.ones(n_fake)].astype(int)
    return pd.DataFrame({
        "video_id": [f"{prefix}{i:04d}" for i in range(len(u))],
        "label": y,
        "logit_mean": u,
        "score_mean": 1 / (1 + np.exp(-u)),
        "cluster_id": [f"c{i % n_clusters}" for i in range(len(u))],
        "real_source": ["celebdf_real" if l == 0 else "" for l in y],
        "manipulation": ["" if l == 0 else "Celeb-DF" for l in y],
    })


# ----------------------------------------------------------------- H1 statistics

def test_alignment_drops_missing_and_rejects_label_mismatch():
    a = _scores()
    b = a.copy()
    b = b.drop(index=[3, 7]).sample(frac=1.0, random_state=1)  # missing + shuffled
    aa, bb, dropped = align_runs_on_video_id([a], [b])
    assert dropped == 2
    assert list(aa[0]["video_id"]) == list(bb[0]["video_id"])
    c = a.copy()
    c.loc[0, "label"] = 1 - c.loc[0, "label"]
    with pytest.raises(ValueError):
        align_runs_on_video_id([a], [c])


def test_paired_bootstrap_is_alignment_invariant():
    a = _scores(seed=0)
    b = a.copy()
    b["score_mean"] = b["score_mean"] + np.random.default_rng(5).normal(0, 0.2, len(b))
    r1 = paired_cluster_boot_dauc([a], [b], n_boot=300, random_seed=3)
    r2 = paired_cluster_boot_dauc([a], [b.sample(frac=1.0, random_state=9)], n_boot=300, random_seed=3)
    assert r1["mean_delta"] == pytest.approx(r2["mean_delta"])
    assert r1["ci_lower"] == pytest.approx(r2["ci_lower"])


def test_degenerate_resamples_are_redrawn_not_zeroed():
    # 4 single-class clusters: about 1 in 8 resamples is single-class. A is perfect,
    # B is noisy, so every valid resample has delta > 0. Zero-filled resamples would put
    # the lower CI bound at exactly 0.
    rows = []
    rng = np.random.default_rng(0)
    for c in range(4):
        lab = c % 2
        for v in range(30):
            rows.append({"video_id": f"c{c}_{v}", "cluster_id": f"c{c}", "label": lab})
    df = pd.DataFrame(rows)
    a = df.copy(); a["score_mean"] = a["label"] + 0.01 * rng.random(len(a))
    b = df.copy(); b["score_mean"] = rng.random(len(b))
    r = paired_cluster_boot_dauc([a], [b], n_boot=400, random_seed=1)
    assert r["ci_lower"] > 0.0


def test_pure_negatives_dfd_keeps_actor_reals_only():
    df = pd.DataFrame({
        # ids follow the real score files (D14: the dfd_real prefix is also required)
        "video_id": ["real_012", "dfd_real_01__kitchen_pan", "dfd_real_02__walk", "DFD_01_02__a", "DFD_01_03__b", "DFD_02_01__c"],
        "label": [0, 0, 0, 1, 1, 1],
        "real_source": ["ffpp_youtube", "dfd_actor", "dfd_actor", "", "", ""],
        "score_mean": [0.1, 0.2, 0.3, 0.7, 0.8, 0.9],
    })
    out, info = apply_pure_negatives(df, "dfd")
    assert info["n_real_kept"] == 2 and info["n_real_removed"] == 1
    assert set(out.loc[out.label == 0, "real_source"]) == {"dfd_actor"}
    out_c, info_c = apply_pure_negatives(df, "celebdf")
    assert len(out_c) == len(df)
    with pytest.raises(ValueError):
        apply_pure_negatives(df.assign(real_source=["x"] * 3 + [""] * 3), "dfd")


def test_h1_verdict_and_helpers():
    assert h1_verdict(["equivalent", "worse"]) == "supported"
    assert h1_verdict(["equivalent", "better"]) == "falsified"
    assert h1_verdict(["inconclusive", "equivalent"]).startswith("undetermined")
    assert bonferroni_level(4) == pytest.approx(0.9875)
    # half-width 0.024, delta 0.01, point 0.002 -> (0.024 / 0.008)^2 = 9
    assert required_size_multiplier(0.024, 0.002, 0.01) == pytest.approx(9.0)
    assert math.isinf(required_size_multiplier(0.02, 0.015, 0.01))


# ----------------------------------------------------------------- calibration / H2

def test_frozen_logit_threshold_matches_calibrated_rule():
    calib = {"a": 0.647, "b": -0.5865, "tau": 0.7507}
    df = _scores()
    p = calibrated_probs(df, calib)
    thr = frozen_logit_threshold(calib)
    assert np.array_equal(p >= calib["tau"], df["logit_mean"].values >= thr)
    # the old bug: tau compared with raw score_mean gives a different decision set
    assert not np.array_equal(df["score_mean"].values >= calib["tau"], p >= calib["tau"])


def test_resolve_decision_scores_refuses_silent_raw_fallback():
    df = _scores()
    with pytest.raises(FileNotFoundError):
        resolve_decision_scores(df, None)
    s, thr, src = resolve_decision_scores(df, None, allow_raw=True)
    assert src == "raw_0.5" and thr == 0.5


def test_exact_oracle_matches_brute_force():
    df = _scores(seed=2)
    y, s = df["label"].values, df["logit_mean"].values
    tau, fpr, fnr, h = find_oracle_tau(y, s)
    brute = min(compute_rates(y, s, t)[2] for t in np.unique(s))
    assert h == pytest.approx(brute)
    assert compute_rates(y, s, tau)[2] == pytest.approx(h)


def test_gap_ci_and_h2_verdict():
    df = _scores(seed=4)
    y, s = df["label"].values, df["logit_mean"].values
    lo, hi, _ = bootstrap_gap_ci(y, s, tau_val=1.5, clusters=df["cluster_id"].values, n_boot=200)
    assert 0 <= lo <= hi
    res = audit_threshold_transfer(df, tau_val=1.5, score_col="logit_mean", n_boot=100)
    assert res["gap_g_ci95"][0] <= res["gap_g"] + 1e-9 or res["gap_g"] >= 0
    assert h2_verdict({"CELEBDF": (0.06, 0.12)}) == "supported"
    assert h2_verdict({"CELEBDF": (0.00, 0.03), "DFDC": (0.01, 0.04)}) == "falsified"
    assert h2_verdict({"CELEBDF": (0.00, 0.07)}) == "undetermined"


def test_wilson_interval_contains_point():
    lo, hi = wilson_interval(40, 178)
    assert lo < 40 / 178 < hi


def test_prior_sweep_reports_bayes_accuracy():
    df = _scores(seed=6)
    out = run_prior_sweep(df, tau_val=0.5, priors=[0.2, 0.8], n_repeats=5)
    assert {"acc_frozen", "acc_em_bayes", "acc_gain"} <= set(out.columns)


# ----------------------------------------------------------------- FPR control (H4/H5)

def test_quantile_threshold_respects_budget():
    rng = np.random.default_rng(0)
    r = rng.normal(0, 1, 140)
    for a in (0.01, 0.05, 0.10):
        thr = quantile_threshold(r, a)
        assert np.mean(r >= thr) <= a + 1e-12


def test_gmm_recovers_shifted_real_mode():
    rng = np.random.default_rng(1)
    u = np.r_[rng.normal(-0.5, 1.0, 400), rng.normal(3.0, 1.0, 600)]
    y = np.r_[np.zeros(400), np.ones(600)].astype(int)
    thr, info = gmm_threshold(u, 0.05)
    assert not info["degenerate"]
    assert abs(np.mean(u[y == 0] >= thr) - 0.05) < 0.03
    assert info["estimated_fake_fraction"] == pytest.approx(0.6, abs=0.05)


def test_gmm_flags_unimodal_target_as_degenerate():
    u = np.random.default_rng(2).normal(0, 1, 500)
    _, info = gmm_threshold(u, 0.05)
    assert info["degenerate"]


def test_source_quantile_inflates_and_label_free_corrects_under_real_shift():
    src = _scores(n_real=300, n_fake=900, shift=4.0, seed=3, prefix="s")
    tgt = _scores(n_real=178, n_fake=340, shift=4.0, seed=4, prefix="t")
    tgt.loc[tgt.label == 0, "logit_mean"] += 1.2  # target reals look more fake
    res = evaluate_policies(tgt, src, alphas=[0.05], n_boot=50)
    by = res.set_index("policy")
    assert by.loc["source_quantile", "fpr"] > 0.15
    assert h4_failure(by.loc["source_quantile", "fpr_ci_low"], 0.05)
    assert by.loc["target_gmm", "abs_fpr_error"] < by.loc["source_quantile", "abs_fpr_error"]
    assert by.loc["oracle", "fpr"] <= 0.05 + 1e-12
    st = prior_stress(tgt, src, fake_fractions=[0.66], alphas=[0.05], n_repeats=10)
    assert set(st["policy"]) >= {"source_quantile", "target_gmm"}


def test_h5_verdict_rules():
    good = {"celebdf": {"err_policy": 0.01, "err_source": 0.10, "degenerate_rate": 0.0}}
    assert h5_verdict(good) == "supported"
    bad = {"celebdf": {"err_policy": 0.08, "err_source": 0.10, "degenerate_rate": 0.0}}
    assert h5_verdict(bad) == "falsified"
    deg = {"celebdf": {"err_policy": 0.01, "err_source": 0.10, "degenerate_rate": 0.2}}
    assert h5_verdict(deg).startswith("falsified")


def test_paired_fpr_delta_sign():
    a = _scores(seed=7)
    b = a.copy()
    b.loc[b.label == 0, "logit_mean"] += 1.0
    r = paired_fpr_delta(b, 0.0, a, 0.0, n_boot=200)
    assert r["delta"] > 0 and r["ci_low"] > 0


# ----------------------------------------------------------------- official GenD adapter (G1)

def test_official_gend_adapter_binary_logit_and_frame_mean():
    from smsdt.models.gend_official import OfficialGenDAdapter

    class Stub(nn.Module):
        def forward(self, x):
            z_fake = x.mean(dim=(1, 2, 3))
            return torch.stack([torch.zeros_like(z_fake), z_fake], dim=1)

    ad = OfficialGenDAdapter(Stub())
    x = torch.randn(2, 4, 3, 8, 8)
    logit, extra = ad(x)
    assert extra is None and logit.shape == (2,)
    expected = x.mean(dim=(2, 3, 4)).mean(dim=1)
    assert torch.allclose(logit, expected, atol=1e-6)
    z = torch.tensor([[0.2, 1.7]])
    p = z.softmax(-1)[0]
    assert OfficialGenDAdapter.binary_logit(z).item() == pytest.approx(math.log(p[1] / p[0]), abs=1e-6)


# ----------------------------------------------------------------- clip-consistency TTA (H6)

def test_consistency_loss_zero_variance_when_clips_agree():
    from smsdt.tta import clip_consistency_loss
    logits = torch.tensor([[2.0, 2.0, 2.0], [-1.0, -1.0, -1.0]])
    loss, parts = clip_consistency_loss(logits, lambda_entropy=0.0, lambda_div=0.0)
    assert parts["consistency"] == pytest.approx(0.0)
    disagree = torch.tensor([[2.0, -2.0, 0.0]])
    _, parts2 = clip_consistency_loss(disagree, 0.0, 0.0)
    assert parts2["consistency"] > 1.0


def test_consistency_tta_updates_only_layernorm_and_returns_video_logits():
    from smsdt.tta import TTAAdapter

    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(6, 6)
            self.norm = nn.LayerNorm(6)
            self.head = nn.Linear(6, 1)
            for p in list(self.proj.parameters()) + list(self.head.parameters()):
                p.requires_grad_(False)

        def forward(self, x):
            return self.head(self.norm(self.proj(x))).squeeze(-1)

    torch.manual_seed(0)
    m = Toy()
    before = {k: v.clone() for k, v in m.state_dict().items()}
    ad = TTAAdapter(m, lr=1e-2, objective="consistency", lambda_div=0.0)
    x = torch.randn(4, 5, 6)  # 4 videos x 5 clips
    video_logits, loss = ad.adapt_batch(x)
    assert video_logits.shape == (4,)
    after = m.state_dict()
    assert not torch.equal(before["norm.weight"], after["norm.weight"])
    assert torch.equal(before["proj.weight"], after["proj.weight"])
    assert torch.equal(before["head.weight"], after["head.weight"])
    ad.reset()
    assert torch.equal(before["norm.weight"], m.state_dict()["norm.weight"])
