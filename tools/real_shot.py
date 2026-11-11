"""Real-shot calibration: holding a false-alarm budget on a new domain with a few real videos.

Motivation (see docs/real_shot_benchmark.md). A deepfake detector deployed on a new
domain needs a threshold that keeps the false-alarm rate (FPR on authentic videos) at a
budget alpha. Our audits show that a threshold fixed on the source dev split does not
transfer: official GenD flags 3.15x the budget on Celeb-DF, and label-free fixes
(mixture models, entropy TTA, EM) fail or collapse. Labelling fakes in a new domain is
hard, but collecting a handful of AUTHENTIC videos from that domain usually is not
(the deployer's own recordings). This tool measures how many real videos (more
precisely, real identities) are needed to hold the budget, and compares estimators.

Policies (predict "fake" if logit > threshold; `n` = number of target real identities):

  source_quantile  n = 0. (1 - alpha) quantile of source dev reals. Deployment default.
  target_gmm       n = 0. Label-free mixture fit on unlabelled target logits (from
                   tools/fpr_control.py). Uses the whole unlabelled target set.
  naive_quantile   Empirical (1 - alpha) quantile of the n target reals (linear
                   interpolation). Common practice, no guarantee.
  conformal        Split-conformal order statistic: k = ceil((n + 1)(1 - alpha)); threshold
                   = k-th smallest target real. E[FPR] <= alpha for exchangeable reals.
                   If k > n the policy abstains (threshold = +inf).
  np_umbrella      Neyman-Pearson umbrella order statistic (Tong, Feng & Li, 2018): the
                   smallest k with P(Bin(n, 1 - alpha) >= k) <= delta. Then
                   P(FPR > alpha) <= delta. Needs n >= log(delta) / log(1 - alpha)
                   (59 at alpha = delta = 0.05); abstains below that.
  *_transfer       PROPOSED (source-shape transfer). Borrow the SHAPE of the real-score
                   distribution from the source dev reals; estimate only its position from
                   the n target reals (robust median / MAD):
                     loc_transfer    shift only:  med_t + (q_src - med_src)
                     ls_transfer     shift+scale: med_t + mad_t * q_std
                     shrink_transfer scale ratio shrunk toward 1 with weight n / (n + 10)
                   Exact when the real-class shift is a (location-)scale map; no
                   distribution-free guarantee.
  *_ucb            PROPOSED (guarded). (1 - delta) quantile of the *_transfer threshold over
                   B bootstrap resamples of the n target reals. Aims at
                   P(FPR > alpha) <= delta with far fewer reals than np_umbrella, under
                   the shape assumption.
  oracle           (1 - alpha) quantile of the evaluation reals themselves. Reference only.

Protocol. For each run, target dataset, alpha and n: draw n distinct real identity
clusters, take one random video from each (calibration set; one per identity keeps the
draws close to independent), and evaluate on the reals of the REMAINING identities plus
all fakes. Repeat R times. Report mean FPR, P(FPR > alpha), P(FPR > 2 alpha),
mean |FPR - alpha|, mean TPR, the abstain rate, and the oracle TPR on the same splits.

Note on violation rates: the evaluation FPR is itself an estimate from about 100-180
held-out reals, so even a threshold with true FPR exactly alpha shows P(FPR_eval > alpha)
near 0.5. P(FPR_eval > 2 alpha) is the robust practical failure rate; the synthetic
tests check the guarantees against the true (population) FPR.

All hyperparameters are fixed in advance (delta = 0.05, B = 200, R = 500, n grid
below). Do not tune them on confirmatory datasets.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import glob
import json
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import binom

DEFAULT_NS = (5, 10, 15, 20, 30, 40, 60)
DEFAULT_ALPHAS = (0.05, 0.10)
DELTA = 0.05
N_BOOT_UCB = 200
MAD_FLOOR = 1e-3
FEW_SHOT_POLICIES = ("naive_quantile", "conformal", "np_umbrella",
                     "loc_transfer", "ls_transfer", "shrink_transfer",
                     "loc_ucb", "ls_ucb", "shrink_ucb")
PROPOSED_POINT = ("loc_transfer", "ls_transfer", "shrink_transfer")
ZERO_SHOT_POLICIES = ("source_quantile", "target_gmm")
ALL_POLICIES = ZERO_SHOT_POLICIES + FEW_SHOT_POLICIES + ("oracle",)


# ---------------------------------------------------------------------------
# Order-statistic rules (finite-sample guarantees)
# ---------------------------------------------------------------------------

def conformal_rank(n: int, alpha: float) -> Optional[int]:
    """1-based rank k of the split-conformal threshold, or None if n is too small."""
    k = int(math.ceil((n + 1) * (1.0 - alpha)))
    return k if k <= n else None


def np_umbrella_rank(n: int, alpha: float, delta: float = DELTA) -> Optional[int]:
    """Smallest 1-based rank k with P(Bin(n, 1 - alpha) >= k) <= delta.

    With threshold = k-th smallest of n i.i.d. reals, P(FPR > alpha) <= delta
    (Tong, Feng & Li, Science Advances 2018, Proposition 1). None if infeasible.
    """
    for k in range(1, n + 1):
        # P(at least k of n reals fall below the true (1 - alpha) quantile)
        if binom.sf(k - 1, n, 1.0 - alpha) <= delta:
            return k
    return None


def min_n_np_umbrella(alpha: float, delta: float = DELTA) -> int:
    """Smallest n for which np_umbrella does not abstain (k = n)."""
    return int(math.ceil(math.log(delta) / math.log(1.0 - alpha)))


def order_stat_threshold(reals: np.ndarray, k: Optional[int]) -> float:
    if k is None:
        return float("inf")
    return float(np.sort(np.asarray(reals, dtype=np.float64))[k - 1])


# ---------------------------------------------------------------------------
# Location-scale transfer (proposed)
# ---------------------------------------------------------------------------

def robust_loc_scale(x: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(x, dtype=np.float64)
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med))) * 1.4826
    return med, max(mad, MAD_FLOOR)


TRANSFER_MODES = ("loc", "ls", "shrink")
SHRINK_N0 = 10


class SourceRealModel:
    """Summary of the source real-score distribution used by the transfer estimators."""

    def __init__(self, source_reals: np.ndarray, alpha: float):
        x = np.asarray(source_reals, dtype=np.float64)
        self.med, self.mad = robust_loc_scale(x)
        self.q_raw = float(np.quantile(x, 1.0 - alpha))           # source (1 - alpha) quantile
        self.q_std = (self.q_raw - self.med) / self.mad            # in source-standardised units


def source_standardised_quantile(source_reals: np.ndarray, alpha: float) -> float:
    return SourceRealModel(source_reals, alpha).q_std


def transfer_threshold(target_reals: np.ndarray, src: "SourceRealModel", mode: str = "shrink",
                       n0: int = SHRINK_N0) -> float:
    """Source-shape transfer of the (1 - alpha) real quantile.

    loc    : target = source + shift. thr = med_t + (q_src - med_src).
    ls     : target = loc + scale * source. thr = med_t + mad_t * q_std.
    shrink : as ls, but the log scale ratio is shrunk toward 0 with weight n / (n + n0),
             so with few reals it behaves like `loc` and with many like `ls`.
    """
    x = np.asarray(target_reals, dtype=np.float64)
    n = len(x)
    if n < 3:
        return float("inf")
    med_t, mad_t = robust_loc_scale(x)
    if mode == "loc":
        return med_t + (src.q_raw - src.med)
    if mode == "ls":
        return med_t + mad_t * src.q_std
    if mode == "shrink":
        w = n / (n + n0)
        scale = src.mad * (mad_t / src.mad) ** w
        return med_t + scale * src.q_std
    raise ValueError(mode)


def _transfer_rows(xb: np.ndarray, src: "SourceRealModel", mode: str, n0: int = SHRINK_N0) -> np.ndarray:
    """Vectorised transfer_threshold over the rows of xb (shape B x n)."""
    n = xb.shape[1]
    med = np.median(xb, axis=1)
    mad = np.maximum(np.median(np.abs(xb - med[:, None]), axis=1) * 1.4826, MAD_FLOOR)
    if mode == "loc":
        return med + (src.q_raw - src.med)
    if mode == "ls":
        return med + mad * src.q_std
    if mode == "shrink":
        w = n / (n + n0)
        return med + src.mad * (mad / src.mad) ** w * src.q_std
    raise ValueError(mode)


def transfer_ucb_threshold(target_reals: np.ndarray, src: "SourceRealModel", mode: str = "shrink",
                           delta: float = DELTA, n_boot: int = N_BOOT_UCB,
                           rng: Optional[np.random.Generator] = None) -> float:
    """(1 - delta) bootstrap quantile of the transfer threshold over resampled target reals."""
    x = np.asarray(target_reals, dtype=np.float64)
    if len(x) < 3:
        return float("inf")
    rng = rng or np.random.default_rng(0)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    return float(np.quantile(_transfer_rows(x[idx], src, mode), 1.0 - delta))


def ls_transfer_threshold(target_reals: np.ndarray, q_src: float) -> float:
    """Location-scale transfer given a source-standardised quantile (kept for tests)."""
    if len(target_reals) < 3:
        return float("inf")
    med, mad = robust_loc_scale(target_reals)
    return med + mad * q_src


# ---------------------------------------------------------------------------
# Policy dispatch
# ---------------------------------------------------------------------------

def few_shot_threshold(policy: str, calib_reals: np.ndarray, alpha: float, src: "SourceRealModel",
                       rng: np.random.Generator, delta: float = DELTA) -> float:
    n = len(calib_reals)
    if policy == "naive_quantile":
        return float(np.quantile(calib_reals, 1.0 - alpha)) if n >= 1 else float("inf")
    if policy == "conformal":
        return order_stat_threshold(calib_reals, conformal_rank(n, alpha))
    if policy == "np_umbrella":
        return order_stat_threshold(calib_reals, np_umbrella_rank(n, alpha, delta))
    for mode in TRANSFER_MODES:
        if policy == f"{mode}_transfer":
            return transfer_threshold(calib_reals, src, mode)
        if policy == f"{mode}_ucb":
            return transfer_ucb_threshold(calib_reals, src, mode, delta=delta, rng=rng)
    raise ValueError(policy)


def rates(y: np.ndarray, u: np.ndarray, thr: float) -> Tuple[float, float]:
    pred = u > thr
    neg, pos = y == 0, y == 1
    fpr = float(pred[neg].mean()) if neg.any() else float("nan")
    tpr = float(pred[pos].mean()) if pos.any() else float("nan")
    return fpr, tpr


# ---------------------------------------------------------------------------
# Real-shot curves
# ---------------------------------------------------------------------------

def real_clusters(df: pd.DataFrame) -> np.ndarray:
    """Identity clusters for the target reals (same rule as tools/fpr_control.py)."""
    from tools.fpr_control import resolve_clusters
    return resolve_clusters(df)


def real_shot_curve(
    target_df: pd.DataFrame,
    source_df: pd.DataFrame,
    ns: Sequence[int] = DEFAULT_NS,
    alphas: Sequence[float] = DEFAULT_ALPHAS,
    n_rep: int = 500,
    logit_col: str = "logit_mean",
    min_eval_clusters: int = 20,
    delta: float = DELTA,
    random_seed: int = 0,
) -> pd.DataFrame:
    """One row per (policy, n, alpha) with summary statistics over n_rep draws."""
    from tools.fpr_control import gmm_threshold, quantile_threshold

    y = target_df["label"].values.astype(int)
    u = target_df[logit_col].values.astype(np.float64)
    clusters = np.asarray(real_clusters(target_df)).astype(str)
    real_idx = np.where(y == 0)[0]
    fake_idx = np.where(y == 1)[0]
    real_cl = clusters[real_idx]
    uniq = np.unique(real_cl)
    by_cl = {c: real_idx[real_cl == c] for c in uniq}
    src_real = source_df.loc[source_df["label"] == 0, logit_col].values.astype(np.float64)
    rng = np.random.default_rng(random_seed)

    # Zero-shot thresholds do not depend on the draw.
    zero = {}
    for a in alphas:
        zero[("source_quantile", a)] = quantile_threshold(src_real, a)
        zero[("target_gmm", a)] = gmm_threshold(u, a)[0]
    src_model = {a: SourceRealModel(src_real, a) for a in alphas}

    rows = []
    max_n = len(uniq) - min_eval_clusters
    for n in ns:
        if n > max_n:
            continue
        rec = {(p, a): {"fpr": [], "tpr": [], "abst": []} for p in ALL_POLICIES for a in alphas}
        for _ in range(n_rep):
            pick = rng.choice(uniq, size=n, replace=False)
            calib = np.array([rng.choice(by_cl[c]) for c in pick])
            eval_real = np.concatenate([by_cl[c] for c in uniq if c not in set(pick)])
            ev = np.concatenate([eval_real, fake_idx])
            ye, ue = y[ev], u[ev]
            cal_u = u[calib]
            for a in alphas:
                thrs = {p: zero[(p, a)] for p in ZERO_SHOT_POLICIES}
                for p in FEW_SHOT_POLICIES:
                    thrs[p] = few_shot_threshold(p, cal_u, a, src_model[a], rng, delta)
                thrs["oracle"] = float(np.quantile(u[eval_real], 1.0 - a))
                for p, t in thrs.items():
                    f, tp = rates(ye, ue, t)
                    r = rec[(p, a)]
                    r["fpr"].append(f)
                    r["tpr"].append(tp)
                    r["abst"].append(not np.isfinite(t))
        for (p, a), r in rec.items():
            f = np.array(r["fpr"])
            t = np.array(r["tpr"])
            rows.append({
                "policy": p, "n_real_identities": n if p in FEW_SHOT_POLICIES else 0,
                "n_draw": n, "alpha": a,
                "mean_fpr": float(f.mean()), "sd_fpr": float(f.std()),
                "p_fpr_gt_alpha": float((f > a).mean()),
                "p_fpr_gt_2alpha": float((f > 2 * a).mean()),
                "mean_abs_err": float(np.abs(f - a).mean()),
                "mean_tpr": float(t.mean()),
                "abstain_rate": float(np.mean(r["abst"])),
                "n_rep": n_rep,
                "n_real_clusters": int(len(uniq)),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Pre-registered verdicts (H7, docs/decisions.md)
# ---------------------------------------------------------------------------

def _get(curve: pd.DataFrame, policy: str, n: int, alpha: float, col: str) -> float:
    g = curve[(curve.policy == policy) & (curve.n_draw == n) & np.isclose(curve.alpha, alpha)]
    return float(g[col].iloc[0]) if len(g) else float("nan")


def select_primary(curve: pd.DataFrame, alpha: float = 0.05, n: int = 10) -> Dict:
    """Development-set selection rule (fixed in advance, docs/decisions.md H7):
    among the proposed point estimators, the one with the lowest mean |FPR - alpha| at
    n target real identities, averaged over all runs in `curve`."""
    g = curve[curve.policy.isin(PROPOSED_POINT) & (curve.n_draw == n) & np.isclose(curve.alpha, alpha)]
    if g.empty:
        return {"primary": None, "errors": {}}
    err = g.groupby("policy")["mean_abs_err"].mean().to_dict()
    best = min(err, key=err.get)
    return {"primary": best.replace("_transfer", ""), "errors": err, "alpha": alpha, "n": n}


def h7_run_checks(curve: pd.DataFrame, primary: str = "shrink", alpha: float = 0.05,
                  n_small: int = 10, n_mid: int = 20) -> Dict:
    """ORIGINAL H7 (v1, registered 2026-11-11 before the development run). Not supported on
    the development sets (see docs/decisions.md section 6); kept so the record is complete.
    Per-run checks (primary = frozen transfer mode):

    h7a: <primary>_transfer mean |FPR - alpha| at n_small <= 0.5 x source_quantile's AND
         <= naive_quantile's at the same n.
    h7b: <primary>_ucb at n_mid has P(FPR_eval > 2 alpha) <= 0.05 AND mean TPR >= 0.8 x the
         oracle TPR at the same n (oracle = exactly alpha on the evaluation reals).
    """
    e_ls = _get(curve, f"{primary}_transfer", n_small, alpha, "mean_abs_err")
    e_src = _get(curve, "source_quantile", n_small, alpha, "mean_abs_err")
    e_nv = _get(curve, "naive_quantile", n_small, alpha, "mean_abs_err")
    v_ucb = _get(curve, f"{primary}_ucb", n_mid, alpha, "p_fpr_gt_2alpha")
    t_ucb = _get(curve, f"{primary}_ucb", n_mid, alpha, "mean_tpr")
    t_or = _get(curve, "oracle", n_mid, alpha, "mean_tpr")
    return {
        "primary": primary,
        "transfer_err": e_ls, "source_err": e_src, "naive_err": e_nv,
        "h7a_pass": bool(np.isfinite(e_ls) and e_ls <= 0.5 * e_src and e_ls <= e_nv),
        "ucb_viol_2a": v_ucb, "ucb_tpr": t_ucb, "oracle_tpr": t_or,
        "h7b_pass": bool(np.isfinite(v_ucb) and np.isfinite(t_or) and v_ucb <= 0.05 and t_ucb >= 0.8 * t_or),
    }


def h7_verdict(checks: Dict[Tuple[str, str], Dict], key: str, min_frac: float = 2 / 3) -> str:
    """Supported if the check passes in at least `min_frac` of runs of every model."""
    by_model: Dict[str, List[bool]] = {}
    for (model, _seed), c in checks.items():
        by_model.setdefault(model, []).append(c[key])
    if not by_model:
        return "undetermined (no runs)"
    ok = {m: (sum(v) / len(v)) >= min_frac for m, v in by_model.items()}
    return "supported" if all(ok.values()) else "not supported (" + ", ".join(m for m, o in ok.items() if not o) + ")"


# ---------------------------------------------------------------------------
# Revised H7 (v2): registered after the development run, BEFORE any confirmatory data
# ---------------------------------------------------------------------------

H7V2_N = 20                 # real identities for the sufficiency claim
H7V2_FPR_TOL = 0.01         # mean FPR <= alpha + tol (eval-set noise)
H7V2_TPR_RATIO = 0.75       # TPR >= ratio x oracle TPR
H7V2_UNSAFE_P = 0.20        # a zero-shot policy is "unsafe" on a run if P(FPR_eval > 2 alpha) > this
H7V2_NAIVE_EXCESS = 0.02    # naive empirical quantile overshoots alpha by more than this
H7V2_MIN_IDENTITIES = 40    # datasets with fewer real identities cannot test H7b-v2


def h7v2_dataset_verdicts(curve: pd.DataFrame, alpha: float = 0.05, n: int = H7V2_N) -> Dict:
    """Dataset-level verdicts for revised H7 (all runs of all models on one dataset).

    H7a-v2 (zero-shot is unsafe): for EACH zero-shot policy (source_quantile, target_gmm)
        at least one run has P(FPR_eval > 2 alpha) > 0.20.
    H7b-v2 (n real identities suffice): conformal at n identities has, in EVERY run,
        mean FPR <= alpha + 0.01 and mean TPR >= 0.75 x oracle TPR. Needs >= 40 identities.
    H7c-v2 (the finite-sample correction matters): naive_quantile at n has mean FPR >
        alpha + 0.02 in a strict majority of runs.
    """
    a_ = np.isclose(curve.alpha, alpha)
    runs = curve[["model", "seed"]].drop_duplicates().itertuples(index=False)
    per_run = []
    for model, seed in runs:
        g = curve[a_ & (curve.model == model) & (curve.seed == seed)]
        get = lambda pol, col, nn=n: _get(g, pol, nn, alpha, col)  # noqa: E731
        per_run.append({
            "model": model, "seed": seed,
            "src_p2a": get("source_quantile", "p_fpr_gt_2alpha"),
            "gmm_p2a": get("target_gmm", "p_fpr_gt_2alpha"),
            "conf_fpr": get("conformal", "mean_fpr"),
            "conf_tpr": get("conformal", "mean_tpr"),
            "oracle_tpr": get("oracle", "mean_tpr"),
            "naive_fpr": get("naive_quantile", "mean_fpr"),
        })
    pr = pd.DataFrame(per_run)
    n_id = int(curve["n_real_clusters"].max()) if len(curve) else 0
    if pr.empty or pr["conf_fpr"].isna().all():
        return {"h7a_v2": "undetermined", "h7b_v2": "undetermined", "h7c_v2": "undetermined",
                "n_real_identities": n_id, "per_run": []}
    a_ok = bool((pr.src_p2a > H7V2_UNSAFE_P).any() and (pr.gmm_p2a > H7V2_UNSAFE_P).any())
    b_rows = (pr.conf_fpr <= alpha + H7V2_FPR_TOL) & (pr.conf_tpr >= H7V2_TPR_RATIO * pr.oracle_tpr)
    if n_id < H7V2_MIN_IDENTITIES:
        b = f"undetermined ({n_id} real identities < {H7V2_MIN_IDENTITIES})"
    else:
        b = "supported" if bool(b_rows.all()) else "not supported (" + ", ".join(
            f"{m}_{s}" for m, s, ok in zip(pr.model, pr.seed, b_rows) if not ok) + ")"
    c_ok = bool((pr.naive_fpr > alpha + H7V2_NAIVE_EXCESS).mean() > 0.5)
    return {"h7a_v2": "supported" if a_ok else "not supported",
            "h7b_v2": b, "h7c_v2": "supported" if c_ok else "not supported",
            "n_real_identities": n_id, "per_run": pr.round(4).to_dict(orient="records")}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    from tools.stats import apply_pure_negatives, dataset_key, DEFAULT_DATASET_ROLES, _parse_score_filename
    from tools.make_tables import df_to_markdown

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scores-dir", default="results/scores")
    ap.add_argument("--source", default="ffppval")
    ap.add_argument("--targets", nargs="*", default=None,
                    help="Target dataset keys (default: every non-FF++ dataset found).")
    ap.add_argument("--ns", nargs="*", type=int, default=list(DEFAULT_NS))
    ap.add_argument("--alphas", nargs="*", type=float, default=list(DEFAULT_ALPHAS))
    ap.add_argument("--n-rep", type=int, default=500)
    ap.add_argument("--min-eval-clusters", type=int, default=20)
    ap.add_argument("--out-dir", default="results/tables")
    ap.add_argument("--tag", default="", help="Suffix for output files (e.g. _exploratory).")
    ap.add_argument("--primary", default="loc", choices=["loc", "ls", "shrink"],
                    help="Frozen primary transfer mode for the H7 checks (set from the development "
                         "selection, then never changed).")
    ap.add_argument("--checks-on", nargs="*", default=None,
                    help="Datasets on which to compute H7 checks (default: confirmatory role or unknown).")
    args = ap.parse_args()

    runs = {}
    for sf in sorted(glob.glob(os.path.join(args.scores_dir, "*.csv"))):
        parsed = _parse_score_filename(sf)
        if parsed is None:
            continue
        model, seed, dname = parsed
        df = pd.read_csv(sf)
        if "logit_mean" not in df.columns:
            continue
        df, _ = apply_pure_negatives(df, dname)
        runs[(model, seed, dataset_key(dname))] = df

    src_key = dataset_key(args.source)
    curves, checks, notes = [], {}, []
    for (model, seed, ds), tdf in sorted(runs.items()):
        if ds.startswith("ffpp"):
            continue
        if args.targets and ds not in args.targets:
            continue
        sdf = runs.get((model, seed, src_key))
        if sdf is None:
            notes.append(f"{model}_{seed}/{ds}: no {src_key} source file")
            continue
        c = real_shot_curve(tdf, sdf, ns=args.ns, alphas=args.alphas, n_rep=args.n_rep,
                            min_eval_clusters=args.min_eval_clusters)
        if c.empty:
            notes.append(f"{model}_{seed}/{ds}: too few real identity clusters")
            continue
        c.insert(0, "role", DEFAULT_DATASET_ROLES.get(ds, "unknown"))
        c.insert(0, "dataset", ds)
        c.insert(0, "seed", seed)
        c.insert(0, "model", model)
        curves.append(c)
        check_here = (ds in args.checks_on) if args.checks_on is not None else (
            DEFAULT_DATASET_ROLES.get(ds) == "confirmatory" or ds not in DEFAULT_DATASET_ROLES)
        if check_here:
            checks[(model, seed, ds)] = h7_run_checks(c, primary=args.primary)

    if not curves:
        print("No runs evaluated.", *notes, sep="\n  ")
        return
    curve = pd.concat(curves, ignore_index=True)
    os.makedirs(args.out_dir, exist_ok=True)
    curve.to_csv(os.path.join(args.out_dir, f"real_shot_curve{args.tag}.csv"), index=False)

    by_ds: Dict[str, Dict] = {}
    for (model, seed, ds), c in checks.items():
        by_ds.setdefault(ds, {})[(model, seed)] = c
    verdicts = {ds: {"h7a_v1": h7_verdict(v, "h7a_pass"), "h7b_v1": h7_verdict(v, "h7b_pass"),
                     "runs_v1": {f"{m}_{s}": c for (m, s), c in v.items()}}
                for ds, v in by_ds.items()}
    for ds, g in curve.groupby("dataset"):
        if ds in by_ds:
            verdicts[ds].update(h7v2_dataset_verdicts(g))
    selection = {ds: select_primary(g) for ds, g in curve.groupby("dataset")}
    meta = {"primary": args.primary, "development_selection": selection, "delta": DELTA, "n_boot_ucb": N_BOOT_UCB, "n_rep": args.n_rep,
            "min_n_np_umbrella": {str(a): min_n_np_umbrella(a) for a in args.alphas},
            "notes": notes, "verdicts": verdicts}
    with open(os.path.join(args.out_dir, f"real_shot_verdicts{args.tag}.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Compact markdown: mean over seeds per (model, dataset, policy, n) at each alpha.
    agg = (curve.groupby(["dataset", "model", "alpha", "policy", "n_draw"], as_index=False)
           [["mean_fpr", "p_fpr_gt_2alpha", "mean_abs_err", "mean_tpr", "abstain_rate"]].mean())
    with open(os.path.join(args.out_dir, f"real_shot_curve{args.tag}.md"), "w", encoding="utf-8") as f:
        f.write("# Real-shot calibration curves (mean over seeds)\n\n")
        f.write("n_draw = number of target real identities used for calibration (zero-shot policies "
                "are repeated at every n on the same evaluation splits).\n\n")
        for (ds, a), g in agg.groupby(["dataset", "alpha"]):
            f.write(f"## {ds}, alpha = {a}\n\n")
            f.write(df_to_markdown(g.drop(columns=["dataset", "alpha"]).round(4)) + "\n\n")
        f.write("## H7 verdicts\n\n")
        for ds, v in verdicts.items():
            f.write(f"- {ds}: v1 (original) H7a {v['h7a_v1']}; H7b {v['h7b_v1']}. "
                    f"v2 (revised) H7a {v.get('h7a_v2')}; H7b {v.get('h7b_v2')}; H7c {v.get('h7c_v2')}\n")
    keys = ("h7a_v1", "h7b_v1", "h7a_v2", "h7b_v2", "h7c_v2", "n_real_identities")
    print(json.dumps({ds: {k: v.get(k) for k in keys} for ds, v in verdicts.items()}, indent=2))
    for n in notes:
        print("note:", n)


if __name__ == "__main__":
    main()
