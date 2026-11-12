"""Tests for the paper-consistency fixes (D16 metric definition, D17 parameter counts,
D18 identity-weighted FPR and real-source strata)."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from smsdt.metric_defs import threshold_at_fpr, tpr_at_fpr, tpr_fpr_at_fpr


# --- D16: one TPR@FPR definition -------------------------------------------

def test_threshold_achieves_at_most_target_fpr():
    rng = np.random.default_rng(0)
    for n in (7, 19, 20, 140, 178, 243):
        r = rng.normal(size=n)
        for a in (0.01, 0.05, 0.10):
            t = threshold_at_fpr(r, a)
            assert np.mean(r > t) <= a + 1e-12
            # lowest such threshold: lowering it to the next real would break the budget
            below = np.sort(r)[np.sort(r) < t]
            if len(below):
                assert np.mean(r > below[-1]) > a


def test_tpr_matches_brute_force():
    rng = np.random.default_rng(1)
    y = np.r_[np.zeros(178), np.ones(340)].astype(int)
    s = np.r_[rng.normal(size=178), rng.normal(1.5, 1, 340)]
    best = 0.0
    for t in np.r_[np.sort(s), np.inf]:
        if np.mean(s[y == 0] > t) <= 0.05:
            best = max(best, np.mean(s[y == 1] > t))
    assert tpr_at_fpr(y, s, 0.05) == pytest.approx(best)
    tpr, fpr = tpr_fpr_at_fpr(y, s, 0.05)
    assert fpr <= 0.05


def test_eval_cross_and_tables_agree():
    from smsdt.eval_cross import tpr_at_fpr as eval_tpr
    from tools.make_tables import compute_dataset_metrics
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"label": np.r_[np.zeros(178), np.ones(340)].astype(int),
                       "score_mean": np.r_[rng.uniform(0, 0.7, 178), rng.uniform(0.2, 1, 340)]})
    a = eval_tpr(df.score_mean.values, df.label.values, 0.05)
    b = compute_dataset_metrics(df)["tpr_5fpr"]
    assert a == pytest.approx(b)


# --- D14 applied to the experiment matrix ----------------------------------

def test_matrix_loader_uses_pure_dfd(tmp_path):
    from tools.make_tables import load_scores_pure
    df = pd.DataFrame({"video_id": ["real_012", "dfd_real_01__a", "DFD_01_02__x"],
                       "label": [0, 0, 1], "real_source": ["dfd_actor", "dfd_actor", np.nan],
                       "score_mean": [0.1, 0.2, 0.9]})
    p = tmp_path / "m_seed1_dfd.csv"
    df.to_csv(p, index=False)
    out = load_scores_pure(str(p), "dfd")
    assert list(out.video_id) == ["dfd_real_01__a", "DFD_01_02__x"]


def test_aggregate_matrix():
    from tools.make_tables import aggregate_matrix
    m = pd.DataFrame({"Model": ["a_seed1", "a_seed2", "b_seed0"],
                      "celebdf_auc": [0.80, 0.90, 0.93], "celebdf_tpr5": [0.5, 0.6, 0.7]})
    g = aggregate_matrix(m).set_index("arch")
    assert g.loc["a", "n_seeds"] == 2
    assert g.loc["a", "celebdf_auc"].startswith("0.8500 +/- 0.0707")
    assert g.loc["b", "celebdf_auc"] == "0.9300"


# --- D17: parameter counts from the trained architectures -------------------

def test_param_counts_match_checkpoint_keys():
    pytest.importorskip("transformers")
    from tools.count_params import count_all
    rows = {r["arch"]: r for r in count_all()}
    # state-key counts match the eval logs' "Weight loading report" (393/415/471/489)
    assert [rows[a]["n_state_keys"] for a in ("gend_ours", "smsdt_s", "smsdt_st", "smsdt_full")] == [393, 415, 471, 489]
    assert rows["gend_ours"]["trainable"] == 103_425
    assert rows["smsdt_s"]["trainable"] == 777_478          # not 103,425
    assert rows["smsdt_st"]["trainable"] == 8_467_462
    assert rows["smsdt_full"]["trainable"] == 9_102_600


# --- D18: identity-weighted FPR and real-source strata ------------------------

def _target(rng, celeb_ids=40, celeb_per=3, yt=40):
    rows = []
    for i in range(celeb_ids):
        for j in range(celeb_per):
            rows.append({"video_id": f"celebdf_Celeb-real_id{i}_{j:04d}", "label": 0,
                         "logit_mean": rng.normal(), "cluster_id": "Celeb-real"})
    for k in range(yt):
        rows.append({"video_id": f"celebdf_YouTube-real_{k:05d}", "label": 0,
                     "logit_mean": 1.0 + rng.normal(), "cluster_id": "YouTube-real"})
    for k in range(200):
        rows.append({"video_id": f"celebdf_Celeb-synthesis_id{k % 40}_id1_{k:04d}", "label": 1,
                     "logit_mean": 3 + rng.normal(), "cluster_id": "Celeb-synthesis"})
    return pd.DataFrame(rows)


def _source(rng):
    return pd.DataFrame({"label": [0] * 140 + [1] * 700,
                         "logit_mean": np.r_[rng.normal(size=140), rng.normal(3, 1, 700)]})


def test_identity_weighted_fpr_columns():
    from tools.real_shot import real_shot_curve
    rng = np.random.default_rng(3)
    c = real_shot_curve(_target(rng), _source(rng), ns=(20,), alphas=(0.05,), n_rep=40)
    assert {"mean_fpr_idw", "p_fpr_idw_gt_2alpha"} <= set(c.columns)
    assert c.n_real_clusters.iloc[0] == 80
    # one video per identity: identity-weighted equals video-weighted
    t1 = _target(rng, celeb_per=1)
    c1 = real_shot_curve(t1, _source(rng), ns=(20,), alphas=(0.05,), n_rep=20)
    assert np.allclose(c1.mean_fpr, c1.mean_fpr_idw)


def test_conformal_identity_weighted_guarantee_holds():
    """E[FPR_idw] <= alpha (+ MC slack) for one-video-per-identity calibration."""
    from tools.real_shot import real_shot_curve
    rng = np.random.default_rng(4)
    t = _target(rng, celeb_ids=60, celeb_per=3, yt=0)
    c = real_shot_curve(t, _source(rng), ns=(19,), alphas=(0.05,), n_rep=400)
    assert c[c.policy == "conformal"].mean_fpr_idw.iloc[0] <= 0.05 + 0.012


def test_strata_curves_split_by_real_source():
    from tools.real_shot import strata_curves
    rng = np.random.default_rng(5)
    st = strata_curves(_target(rng), _source(rng), ns=(10,), alphas=(0.05,), n_rep=20)
    assert set(st.real_stratum) == {"Celeb-real", "YouTube-real"}
    n_cl = st.groupby("real_stratum").n_real_clusters.first()
    assert n_cl["Celeb-real"] == 40 and n_cl["YouTube-real"] == 40


# --- new confirmatory dataset: explicit real_source and identity labels ------

def test_identity_map_and_cli_flags(tmp_path):
    from smsdt import eval_cross
    p = tmp_path / "ids.csv"
    p.write_text("video_id,identity\nv1,actor_07\nv2,actor_07\nv3,actor_11\n", encoding="utf-8")
    m = eval_cross.load_identity_map(str(p))
    assert m == {"v1": "actor_07", "v2": "actor_07", "v3": "actor_11"}
    bad = tmp_path / "bad.csv"
    bad.write_text("vid,who\nv1,a\n", encoding="utf-8")
    with pytest.raises(ValueError):
        eval_cross.load_identity_map(str(bad))
    src = open(eval_cross.__file__).read()
    assert "--real-source" in src and "--identity-csv" in src


def test_paper_figures_build_from_tables(tmp_path):
    pytest.importorskip("matplotlib")
    from tools import paper_figures
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    tables = os.path.join(root, "results", "tables")
    if not os.path.exists(os.path.join(tables, "real_shot_curve_dev_celebdf.csv")):
        pytest.skip("development tables not present")
    paper_figures.fig_zero_shot(tables, "celebdf", 0.05, str(tmp_path))
    paper_figures.fig_real_shot(tables, "_dev_celebdf", 0.05, str(tmp_path))
    names = {p.name for p in tmp_path.iterdir()}
    assert {"fig1_zero_shot_fpr.pdf", "fig2_real_shot_fpr.pdf", "fig4_real_shot_fail.png"} <= names
