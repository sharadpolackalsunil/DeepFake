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


def test_h7v2_verdicts_enforce_idw_amendment_6_3_b():
    from tools.real_shot import h7v2_dataset_verdicts
    rows = []
    for model, idw_fpr in (("m1", 0.050), ("m2", 0.070)):
        base = {"model": model, "seed": "seed1", "alpha": 0.05, "n_draw": 20, "n_real_clusters": 100}
        rows += [
            dict(base, policy="source_quantile", p_fpr_gt_2alpha=0.5, mean_fpr=0.1, mean_fpr_idw=0.1, mean_tpr=0.8),
            dict(base, policy="target_gmm", p_fpr_gt_2alpha=0.5, mean_fpr=0.1, mean_fpr_idw=0.1, mean_tpr=0.5),
            dict(base, policy="conformal", p_fpr_gt_2alpha=0.1, mean_fpr=0.048, mean_fpr_idw=idw_fpr, mean_tpr=0.50),
            dict(base, policy="oracle", p_fpr_gt_2alpha=0.0, mean_fpr=0.05, mean_fpr_idw=0.05, mean_tpr=0.60),
            dict(base, policy="naive_quantile", p_fpr_gt_2alpha=0.3, mean_fpr=0.09, mean_fpr_idw=0.09, mean_tpr=0.7),
        ]
    v = h7v2_dataset_verdicts(pd.DataFrame(rows))
    assert v["h7b_v2"] == "not supported (m2_seed1)"


def test_prep_fakeavceleb_filters_and_stratifies(tmp_path):
    from tools.prep_fakeavceleb import VISUAL_FAKE_METHODS, prepare_fakeavceleb
    from tools.stats import DEFAULT_DATASET_ROLES, PURE_NEGATIVE_SOURCES

    assert DEFAULT_DATASET_ROLES["fakeavceleb"] == "confirmatory"
    assert PURE_NEGATIVE_SOURCES["fakeavceleb"] is None

    raw = tmp_path / "FakeAVCeleb_v1.2"
    rows = []
    # 3 real Category A (same filename 00010.mp4 under different identity folders)
    for i in range(3):
        rel = f"RealVideo-RealAudio/African/men/id0000{i}"
        (raw / rel).mkdir(parents=True, exist_ok=True)
        (raw / rel / "00010.mp4").write_bytes(b"mp4")
        rows.append({"source": f"id0000{i}", "target1": "-", "target2": "-", "method": "real",
                     "category": "A", "type": "RealVideo-RealAudio", "race": "African", "gender": "men",
                     "path": "00010.mp4", "Unnamed: 9": f"FakeAVCeleb/{rel}"})
    # 2 Category B (RealVideo-FakeAudio) -> must be excluded
    for i in range(2):
        rel = f"RealVideo-FakeAudio/African/men/id0000{i}"
        (raw / rel).mkdir(parents=True, exist_ok=True)
        (raw / rel / "00010_rtvc.mp4").write_bytes(b"mp4")
        rows.append({"source": f"id0000{i}", "target1": "-", "target2": "-", "method": "rtvc",
                     "category": "B", "type": "RealVideo-FakeAudio", "race": "African", "gender": "men",
                     "path": "00010_rtvc.mp4", "Unnamed: 9": f"FakeAVCeleb/{rel}"})
    # 3 fakes per visual method across C & D, plus 1 missing .mp4 in D
    for m in VISUAL_FAKE_METHODS:
        for j, cat_type in enumerate((("C", "FakeVideo-RealAudio"), ("D", "FakeVideo-FakeAudio"), ("D", "FakeVideo-FakeAudio"))):
            cat, tname = cat_type
            rel = f"{tname}/African/men/id0000{j}"
            (raw / rel).mkdir(parents=True, exist_ok=True)
            fname = f"fake_{m}_{j}.mp4"
            (raw / rel / fname).write_bytes(b"mp4")
            rows.append({"source": f"id0000{j}", "target1": f"id0000{(j+1)%3}", "target2": "-", "method": m,
                         "category": cat, "type": tname, "race": "African", "gender": "men",
                         "path": fname, "Unnamed: 9": f"FakeAVCeleb/{rel}"})
    rows.append({"source": "id00000", "target1": "id00001", "target2": "-", "method": "wav2lip",
                 "category": "D", "type": "FakeVideo-FakeAudio", "race": "African", "gender": "men",
                 "path": "missing.mp4", "Unnamed: 9": "FakeAVCeleb/FakeVideo-FakeAudio/African/men/id00000"})
    pd.DataFrame(rows).to_csv(raw / "meta_data.csv", index=False)

    out_idx = tmp_path / "fakeavceleb_index.csv"
    out_ids = tmp_path / "fakeavceleb_identities.csv"
    stats = prepare_fakeavceleb(raw, out_idx, out_ids, fakes_per_method=2, seed=0)
    assert stats["n_reals"] == 3 and stats["n_real_identities"] == 3
    assert stats["n_fakes"] == 2 * len(VISUAL_FAKE_METHODS)
    assert stats["n_missing_fakes_skipped"] == 1
    idx_df = pd.read_csv(out_idx)
    id_df = pd.read_csv(out_ids)
    assert "rtvc" not in set(idx_df["method"])
    assert idx_df["video_id"].is_unique and id_df["video_id"].is_unique
    assert len(idx_df) == len(id_df) == 3 + 2 * len(VISUAL_FAKE_METHODS)


def test_prep_fakeavceleb_prune_from_store(tmp_path):
    from tools.prep_fakeavceleb import prune_from_store
    store_dir = tmp_path / "store"
    # Create fakeavceleb_index.csv with 3 videos:
    # v1: valid fidx with 8 contiguous frames
    # v2: valid fidx but only 4 frames (too short)
    # v3: missing directory / fidx
    idx_path = tmp_path / "fakeavceleb_index.csv"
    id_path = tmp_path / "fakeavceleb_identities.csv"

    v1_dir = store_dir / "favc_real_id0_001"
    v1_dir.mkdir(parents=True)
    np.save(v1_dir / "fidx.npy", np.arange(10, dtype=np.int32))

    v2_dir = store_dir / "favc_real_id1_002"
    v2_dir.mkdir(parents=True)
    np.save(v2_dir / "fidx.npy", np.arange(4, dtype=np.int32))

    # v3 not created in store

    pd.DataFrame({
        "video_id": ["favc_real_id0_001", "favc_real_id1_002", "favc_fake_c_003"],
        "split": ["test", "test", "test"],
        "label": [0, 0, 1],
        "method": ["real", "real", "wav2lip"],
        "target_id": ["id0", "id1", "id2"],
        "source_id": ["", "", "id2"],
        "path": ["a.mp4", "b.mp4", "c.mp4"]
    }).to_csv(idx_path, index=False)

    pd.DataFrame({
        "video_id": ["favc_real_id0_001", "favc_real_id1_002", "favc_fake_c_003"],
        "identity": ["id0", "id1", "fake_id2"]
    }).to_csv(id_path, index=False)

    res = prune_from_store(store_dir, idx_path, id_path, min_contiguous_frames=8)
    assert res["initial_total"] == 3
    assert res["pruned_total"] == 1
    assert res["dropped_count"] == 2
    assert res["n_reals"] == 1
    assert res["n_real_identities"] == 1

    pruned_idx = pd.read_csv(idx_path)
    pruned_ids = pd.read_csv(id_path)
    assert list(pruned_idx["video_id"]) == ["favc_real_id0_001"]
    assert list(pruned_ids["video_id"]) == ["favc_real_id0_001"]


def test_eval_cross_fakeavceleb_fallbacks():
    src = open("smsdt/eval_cross.py", encoding="utf-8").read()
    assert 'elif "fakeavceleb" in args.dataset_name.lower():' in src
    assert 'v_id.startswith("favc_")' in src


def test_make_tables_preferred_order_has_fakeavceleb():
    src = open("tools/make_tables.py", encoding="utf-8").read()
    assert '"fakeavceleb_auc"' in src
    assert '"fakeavceleb_tpr5"' in src

