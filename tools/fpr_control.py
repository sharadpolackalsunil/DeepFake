"""False-positive-rate control under dataset shift (hypotheses H4 and H5).

Question. A deployer fixes a false-alarm budget alpha (for example 5% of real videos
flagged). The threshold is set on in-domain reals. On an unseen dataset, is the
achieved FPR still near alpha? HTER and EER audits can look fine while the FPR drifts,
because FNR falls when FPR rises. This tool measures that drift directly and tests
label-free ways to correct it.

Thresholding policies (all predict "fake" if logit >= threshold):

  source_quantile      Baseline / deployment default. Threshold = (1 - alpha) quantile of
                       the SOURCE real logits (dev split preferred; FF++ test reals as a
                       labelled proxy). Uses no target information.
  target_gmm           Label-free (proposed). Fit a 2-component Gaussian mixture to the
                       UNLABELLED target logits; the lower-mean component is taken as the
                       real class; threshold = mu_r + sigma_r * z_(1 - alpha).
  target_gmm_anchored  Label-free (proposed). As target_gmm, but EM is initialised at the
                       source real and fake components, which keeps the component
                       identity stable when the target is very imbalanced.
  oracle               Reference only. (1 - alpha) quantile of the TRUE target real logits.
                       Uses target labels; never a deployable method.

Outputs per (run, target dataset, alpha): achieved FPR, TPR, |FPR - alpha|, FPR / alpha,
degeneracy flags for the mixture fits, and cluster-bootstrap CIs in which the label-free
thresholds are re-estimated on every resample. A class-prior stress test subsamples the
target to fixed fake fractions (default 0.5, 0.66, 0.91).

Pre-registration: see docs/decisions.md (H4, H5). Tune nothing on confirmatory datasets.
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm

POLICIES = ("source_quantile", "target_gmm", "target_gmm_anchored", "oracle")
LABEL_FREE = ("target_gmm", "target_gmm_anchored")


# ---------------------------------------------------------------------------
# Threshold estimators
# ---------------------------------------------------------------------------

def quantile_threshold(real_logits: np.ndarray, alpha: float) -> float:
    """Smallest threshold whose empirical FPR on `real_logits` is <= alpha.

    With n reals the achievable FPRs are k/n; the threshold is placed just above the
    k-th largest real logit, k = floor(alpha * n).
    """
    r = np.sort(np.asarray(real_logits, dtype=np.float64))[::-1]
    n = len(r)
    if n == 0:
        raise ValueError("No real logits to set a quantile threshold.")
    k = int(math.floor(alpha * n))
    if k >= n:
        return float(r[-1])
    return float(np.nextafter(r[k], np.inf))


def _fit_gmm_1d(x: np.ndarray, means_init=None, variances_init=None, weights_init=None,
                random_state: int = 0, max_iter: int = 300):
    """Two-component 1-D Gaussian mixture by EM (closed form, log-space, no sklearn)."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if means_init is None:
        q = np.quantile(x, [0.25, 0.75])
        means = np.array([q[0], q[1]], dtype=np.float64)
    else:
        means = np.asarray(means_init, dtype=np.float64).copy()
    var0 = max(float(np.var(x)), 1e-6)
    variances = (np.asarray(variances_init, dtype=np.float64).copy()
                 if variances_init is not None else np.array([var0 / 2, var0 / 2]))
    weights = (np.asarray(weights_init, dtype=np.float64).copy()
               if weights_init is not None else np.array([0.5, 0.5]))
    floor = 1e-4 * var0
    prev_ll = -np.inf
    converged = False
    log2pi = math.log(2 * math.pi)
    for _ in range(max_iter):
        logd = (np.log(weights)[None, :] - 0.5 * (log2pi + np.log(variances))[None, :]
                - 0.5 * (x[:, None] - means[None, :]) ** 2 / variances[None, :])
        mx = logd.max(axis=1, keepdims=True)
        lse = mx[:, 0] + np.log(np.exp(logd - mx).sum(axis=1))
        ll = float(lse.sum())
        resp = np.exp(logd - lse[:, None])
        nk = resp.sum(axis=0) + 1e-12
        weights = nk / n
        means = (resp * x[:, None]).sum(axis=0) / nk
        variances = np.maximum((resp * (x[:, None] - means) ** 2).sum(axis=0) / nk, floor)
        if abs(ll - prev_ll) < 1e-7 * max(1.0, abs(ll)):
            converged = True
            break
        prev_ll = ll
    return means, variances, weights, converged


def gmm_threshold(
    target_logits: np.ndarray,
    alpha: float,
    anchor: Optional[Dict] = None,
    min_weight: float = 0.02,
    min_separation: float = 1.0,
) -> Tuple[float, Dict]:
    """Label-free threshold from a 2-component mixture on unlabelled target logits.

    Args:
        target_logits: logits of the unlabelled target set
        alpha: target FPR
        anchor: optional {'real_mean','real_var','fake_mean','fake_var'} from the source
            to initialise EM (target_gmm_anchored)
        min_weight: a component with weight below this is flagged degenerate

    Returns:
        (threshold, info) where info has the fitted parameters and flags:
        degenerate (a component collapsed), swapped (for anchored fits: the component
        initialised as real ended with the higher mean), converged.
    """
    x = np.asarray(target_logits, dtype=np.float64)
    if anchor is not None:
        means0 = [anchor["real_mean"], anchor["fake_mean"]]
        vars0 = [max(anchor["real_var"], 1e-4), max(anchor["fake_var"], 1e-4)]
        means, variances, weights, conv = _fit_gmm_1d(x, means_init=means0, variances_init=vars0)
        swapped = bool(means[0] > means[1])
    else:
        means, variances, weights, conv = _fit_gmm_1d(x)
        swapped = False
    r = int(np.argmin(means))
    mu_r, sd_r, w_r = float(means[r]), float(math.sqrt(variances[r])), float(weights[r])
    thr = mu_r + sd_r * float(norm.ppf(1.0 - alpha))
    # Degenerate if a component collapsed, or the two components are not separated
    # (a unimodal fit gives no information about where the real class sits).
    sep = abs(float(means[1 - r]) - mu_r) / math.sqrt(0.5 * (float(variances[0]) + float(variances[1])))
    degenerate = bool(min(weights) < min_weight or sep < min_separation)
    info = {
        "real_mean": mu_r, "real_sd": sd_r, "real_weight": w_r,
        "fake_mean": float(means[1 - r]), "fake_sd": float(math.sqrt(variances[1 - r])),
        "estimated_fake_fraction": float(1.0 - w_r),
        "separation": float(sep),
        "degenerate": degenerate, "swapped": swapped, "converged": bool(conv),
    }
    return float(thr), info


def source_anchor(source_df: pd.DataFrame, logit_col: str = "logit_mean") -> Dict:
    """Mean and variance of source real and fake logits (for the anchored mixture)."""
    r = source_df.loc[source_df["label"] == 0, logit_col].values.astype(np.float64)
    f = source_df.loc[source_df["label"] == 1, logit_col].values.astype(np.float64)
    if len(r) < 2 or len(f) < 2:
        raise ValueError("Source needs at least two reals and two fakes for the anchor.")
    return {"real_mean": float(r.mean()), "real_var": float(r.var(ddof=1)),
            "fake_mean": float(f.mean()), "fake_var": float(f.var(ddof=1))}


def policy_threshold(
    policy: str,
    alpha: float,
    target_logits: np.ndarray,
    target_labels: Optional[np.ndarray],
    source_real_logits: np.ndarray,
    anchor: Optional[Dict],
) -> Tuple[float, Dict]:
    """Threshold for one policy. Only `oracle` reads target_labels."""
    if policy == "source_quantile":
        return quantile_threshold(source_real_logits, alpha), {}
    if policy == "target_gmm":
        return gmm_threshold(target_logits, alpha)
    if policy == "target_gmm_anchored":
        return gmm_threshold(target_logits, alpha, anchor=anchor)
    if policy == "oracle":
        if target_labels is None:
            raise ValueError("oracle needs target labels")
        return quantile_threshold(target_logits[target_labels == 0], alpha), {}
    raise ValueError(f"Unknown policy {policy}")


def rates_at(labels: np.ndarray, logits: np.ndarray, thr: float) -> Tuple[float, float]:
    """(FPR, TPR) for the rule fake if logit >= thr."""
    labels = np.asarray(labels)
    neg = logits[labels == 0]
    pos = logits[labels == 1]
    fpr = float(np.mean(neg >= thr)) if len(neg) else float("nan")
    tpr = float(np.mean(pos >= thr)) if len(pos) else float("nan")
    return fpr, tpr


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_policies(
    target_df: pd.DataFrame,
    source_df: pd.DataFrame,
    alphas: List[float] = (0.05, 0.10),
    logit_col: str = "logit_mean",
    n_boot: int = 1000,
    cluster_col: Optional[str] = "cluster_id",
    random_seed: int = 0,
) -> pd.DataFrame:
    """All policies x alphas on one target set, with cluster-bootstrap CIs.

    In each resample the label-free thresholds are re-estimated from the resampled
    target logits (they are functions of the target sample); the source threshold is
    fixed; the oracle is recomputed from the resampled reals.
    """
    y = target_df["label"].values.astype(int)
    u = target_df[logit_col].values.astype(np.float64)
    src_real = source_df.loc[source_df["label"] == 0, logit_col].values.astype(np.float64)
    anchor = source_anchor(source_df, logit_col)
    if cluster_col and cluster_col in target_df.columns:
        clusters = target_df[cluster_col].astype(str).values
    else:
        clusters = np.arange(len(target_df)).astype(str)
    uniq = np.unique(clusters)
    idx_by = {c: np.where(clusters == c)[0] for c in uniq}
    rng = np.random.default_rng(random_seed)

    rows = []
    boot_store = {(p, a): [] for p in POLICIES for a in alphas}
    boot_deg = {(p, a): [] for p in POLICIES for a in alphas}
    for _ in range(n_boot):
        for _try in range(1000):
            pick = rng.choice(uniq, size=len(uniq), replace=True)
            idx = np.concatenate([idx_by[c] for c in pick])
            if 0 < y[idx].sum() < len(idx):
                break
        yb, ub = y[idx], u[idx]
        for a in alphas:
            for p in POLICIES:
                thr, binfo = policy_threshold(p, a, ub, yb, src_real, anchor)
                boot_store[(p, a)].append(rates_at(yb, ub, thr))
                boot_deg[(p, a)].append(bool(binfo.get("degenerate", False) or binfo.get("swapped", False)))

    for a in alphas:
        for p in POLICIES:
            thr, info = policy_threshold(p, a, u, y, src_real, anchor)
            fpr, tpr = rates_at(y, u, thr)
            bs = np.array(boot_store[(p, a)]) if n_boot > 0 else np.empty((0, 2))
            row = {
                "policy": p,
                "alpha": a,
                "label_free": p in LABEL_FREE,
                "uses_target_labels": p == "oracle",
                "threshold_logit": thr,
                "fpr": fpr,
                "tpr": tpr,
                "abs_fpr_error": abs(fpr - a),
                "fpr_over_alpha": fpr / a,
                "fpr_ci_low": float(np.quantile(bs[:, 0], 0.025)) if len(bs) else float("nan"),
                "fpr_ci_high": float(np.quantile(bs[:, 0], 0.975)) if len(bs) else float("nan"),
                "abs_err_ci_high": float(np.quantile(np.abs(bs[:, 0] - a), 0.975)) if len(bs) else float("nan"),
                "tpr_ci_low": float(np.quantile(bs[:, 1], 0.025)) if len(bs) else float("nan"),
                "tpr_ci_high": float(np.quantile(bs[:, 1], 0.975)) if len(bs) else float("nan"),
                "n_reals": int((y == 0).sum()),
                "n_fakes": int((y == 1).sum()),
                "fpr_boot_median": float(np.median(bs[:, 0])) if len(bs) else float("nan"),
                "boot_degenerate_rate": float(np.mean(boot_deg[(p, a)])) if n_boot > 0 else float("nan"),
                "degenerate": info.get("degenerate", False),
                "swapped": info.get("swapped", False),
                "est_fake_fraction": info.get("estimated_fake_fraction", float("nan")),
            }
            rows.append(row)
    return pd.DataFrame(rows)


def prior_stress(
    target_df: pd.DataFrame,
    source_df: pd.DataFrame,
    fake_fractions: List[float] = (0.5, 0.66, 0.91),
    alphas: List[float] = (0.05, 0.10),
    n_repeats: int = 100,
    logit_col: str = "logit_mean",
    random_seed: int = 0,
) -> pd.DataFrame:
    """Subsample the target to fixed fake fractions and re-run every policy.

    Keeps all reals when possible and subsamples fakes (or reals) to hit the fraction.
    Reports mean and 95th percentile of |FPR - alpha| and the degeneracy rate.
    """
    rng = np.random.default_rng(random_seed)
    reals = target_df[target_df["label"] == 0]
    fakes = target_df[target_df["label"] == 1]
    src_real = source_df.loc[source_df["label"] == 0, logit_col].values.astype(np.float64)
    anchor = source_anchor(source_df, logit_col)
    rows = []
    for pi in fake_fractions:
        n_r = len(reals)
        n_f = int(round(pi / (1 - pi) * n_r))
        if n_f > len(fakes):
            n_f = len(fakes)
            n_r = int(round((1 - pi) / pi * n_f))
        if n_r < 20 or n_f < 20:
            continue
        rec = {(p, a): {"err": [], "deg": [], "tpr": []} for p in POLICIES for a in alphas}
        for _ in range(n_repeats):
            sr = reals.iloc[rng.choice(len(reals), size=n_r, replace=False)]
            sf = fakes.iloc[rng.choice(len(fakes), size=n_f, replace=False)]
            sub = pd.concat([sr, sf])
            yy = sub["label"].values.astype(int)
            uu = sub[logit_col].values.astype(np.float64)
            for a in alphas:
                for p in POLICIES:
                    thr, info = policy_threshold(p, a, uu, yy, src_real, anchor)
                    fpr, tpr = rates_at(yy, uu, thr)
                    rec[(p, a)]["err"].append(abs(fpr - a))
                    rec[(p, a)]["tpr"].append(tpr)
                    rec[(p, a)]["deg"].append(bool(info.get("degenerate", False) or info.get("swapped", False)))
        for (p, a), d in rec.items():
            rows.append({
                "fake_fraction": pi, "n_reals": n_r, "n_fakes": n_f,
                "policy": p, "alpha": a,
                "mean_abs_fpr_error": float(np.mean(d["err"])),
                "p95_abs_fpr_error": float(np.quantile(d["err"], 0.95)),
                "mean_tpr": float(np.mean(d["tpr"])),
                "degenerate_rate": float(np.mean(d["deg"])),
            })
    return pd.DataFrame(rows)


def paired_fpr_delta(
    target_a: pd.DataFrame, thr_a: float,
    target_b: pd.DataFrame, thr_b: float,
    n_boot: int = 2000, logit_col: str = "logit_mean",
    cluster_col: str = "cluster_id", random_seed: int = 0,
) -> Dict:
    """Paired cluster bootstrap of FPR(A) - FPR(B) on the same target reals, each model
    at its own fixed (source) threshold. Used for the H4 capacity comparison."""
    ra = target_a[target_a["label"] == 0][["video_id", logit_col] + ([cluster_col] if cluster_col in target_a else [])]
    rb = target_b[target_b["label"] == 0][["video_id", logit_col]]
    m = ra.merge(rb, on="video_id", suffixes=("_a", "_b"))
    if m.empty:
        raise ValueError("No shared real videos between the two runs.")
    fa = (m[f"{logit_col}_a"].values >= thr_a).astype(float)
    fb = (m[f"{logit_col}_b"].values >= thr_b).astype(float)
    cl = m[cluster_col].astype(str).values if cluster_col in m else m["video_id"].astype(str).values
    uniq = np.unique(cl)
    idx_by = {c: np.where(cl == c)[0] for c in uniq}
    rng = np.random.default_rng(random_seed)
    d = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([idx_by[c] for c in rng.choice(uniq, size=len(uniq), replace=True)])
        d[b] = fa[idx].mean() - fb[idx].mean()
    return {"fpr_a": float(fa.mean()), "fpr_b": float(fb.mean()), "delta": float(fa.mean() - fb.mean()),
            "ci_low": float(np.quantile(d, 0.025)), "ci_high": float(np.quantile(d, 0.975)),
            "n_reals": int(len(m))}


# ---------------------------------------------------------------------------
# Pre-registered verdicts
# ---------------------------------------------------------------------------

def h4_failure(fpr_ci_low: float, alpha: float, factor: float = 2.0) -> bool:
    """H4 'FPR-transfer failure' on one dataset: the lower 95% CI bound of the achieved
    FPR under source_quantile exceeds factor * alpha."""
    return bool(fpr_ci_low > factor * alpha)


def h5_verdict(per_dataset: Dict[str, Dict]) -> str:
    """H5 rule for one label-free policy (docs/decisions.md).

    per_dataset[ds] = {'err_policy': mean |FPR-alpha| (stress mean), 'err_source': same for
    source_quantile, 'degenerate_rate': max over stress levels}.
    Supported if on a strict majority of confirmatory datasets the policy halves the error
    of source_quantile AND no stress level has degenerate_rate > 0.05.
    Falsified otherwise.
    """
    if not per_dataset:
        return "not_evaluated"
    if any(v["degenerate_rate"] > 0.05 for v in per_dataset.values()):
        return "falsified (degenerate fits)"
    wins = sum(1 for v in per_dataset.values() if v["err_policy"] <= 0.5 * v["err_source"])
    return "supported" if wins > len(per_dataset) / 2 else "falsified"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse
    import glob
    from tools.make_tables import df_to_markdown
    from tools.stats import apply_pure_negatives, dataset_key, DEFAULT_DATASET_ROLES, _parse_score_filename

    ap = argparse.ArgumentParser(description="FPR control under dataset shift (H4, H5)")
    ap.add_argument("--scores-dir", default="results/scores")
    ap.add_argument("--source", default="ffppval",
                    help="Dataset token of the source file per run: 'ffppval' (dev split scores, preferred) "
                         "or 'ffpp' (FF++ test reals, used as a labelled proxy)")
    ap.add_argument("--alphas", nargs="*", type=float, default=[0.05, 0.10])
    ap.add_argument("--fake-fractions", nargs="*", type=float, default=[0.5, 0.66, 0.91])
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--n-repeats", type=int, default=100)
    ap.add_argument("--out-dir", default="results/tables")
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

    main_rows, stress_rows, notes = [], [], []
    for (model, seed, ds), tdf in sorted(runs.items()):
        if ds.startswith("ffpp"):
            continue
        source_key = dataset_key(args.source)
        sdf = runs.get((model, seed, source_key))
        source_label = source_key
        if sdf is None and source_key != "ffpp":
            sdf = runs.get((model, seed, "ffpp"))
            source_label = "ffpp_test_proxy"
        if sdf is None:
            notes.append(f"{model}_{seed}/{ds}: no source file")
            continue
        res = evaluate_policies(tdf, sdf, alphas=args.alphas, n_boot=args.n_boot)
        res.insert(0, "source", source_label)
        res.insert(0, "role", DEFAULT_DATASET_ROLES.get(ds, "unknown"))
        res.insert(0, "dataset", ds)
        res.insert(0, "seed", seed)
        res.insert(0, "model", model)
        main_rows.append(res)
        st = prior_stress(tdf, sdf, fake_fractions=args.fake_fractions, alphas=args.alphas,
                          n_repeats=args.n_repeats)
        st.insert(0, "dataset", ds)
        st.insert(0, "seed", seed)
        st.insert(0, "model", model)
        stress_rows.append(st)

    if not main_rows:
        print("No (run, target) pairs with a source file. Export dev-split scores with "
              "`smsdt.eval_cross --split val --save-scores results/scores/<run>_ffppval.csv`.")
        for n in notes:
            print("  -", n)
        return

    main_df = pd.concat(main_rows, ignore_index=True)
    stress_df = pd.concat(stress_rows, ignore_index=True) if stress_rows else pd.DataFrame()

    # H4: source_quantile failure per (model, seed, dataset) at each alpha
    h4 = main_df[main_df["policy"] == "source_quantile"].copy()
    h4["h4_failure"] = [h4_failure(lo, a) for lo, a in zip(h4["fpr_ci_low"], h4["alpha"])]

    # H5: per label-free policy and alpha, over confirmatory datasets, pooled over runs
    h5 = {}
    if not stress_df.empty:
        conf = stress_df[stress_df["dataset"].map(lambda d: DEFAULT_DATASET_ROLES.get(d, "unknown")) == "confirmatory"]
        for pol in LABEL_FREE:
            for a in args.alphas:
                per = {}
                for ds, g in conf[conf["alpha"] == a].groupby("dataset"):
                    gp = g[g["policy"] == pol]
                    gs = g[g["policy"] == "source_quantile"]
                    if gp.empty or gs.empty:
                        continue
                    per[ds] = {"err_policy": float(gp["mean_abs_fpr_error"].mean()),
                               "err_source": float(gs["mean_abs_fpr_error"].mean()),
                               "degenerate_rate": float(gp["degenerate_rate"].max())}
                h5[f"{pol}@alpha={a}"] = {"verdict": h5_verdict(per), "per_dataset": per}

    os.makedirs(args.out_dir, exist_ok=True)
    main_df.to_csv(os.path.join(args.out_dir, "fpr_control.csv"), index=False)
    stress_df.to_csv(os.path.join(args.out_dir, "fpr_control_prior_stress.csv"), index=False)
    h4.to_csv(os.path.join(args.out_dir, "fpr_control_h4.csv"), index=False)
    with open(os.path.join(args.out_dir, "fpr_control_verdicts.json"), "w", encoding="utf-8") as f:
        json.dump({"h5": h5, "notes": notes,
                   "h4_failures": int(h4["h4_failure"].sum()), "h4_rows": int(len(h4))}, f, indent=2)
    cols = ["model", "seed", "dataset", "role", "source", "policy", "alpha", "fpr", "fpr_ci_low",
            "fpr_ci_high", "fpr_over_alpha", "tpr", "degenerate", "swapped"]
    with open(os.path.join(args.out_dir, "fpr_control.md"), "w", encoding="utf-8") as f:
        f.write("# FPR control under dataset shift (H4, H5)\n\n")
        f.write("- source_quantile: threshold from source reals only (deployment default).\n")
        f.write("- target_gmm / target_gmm_anchored: label-free, fitted on unlabelled target logits.\n")
        f.write("- oracle: uses target labels; reference only.\n")
        f.write(f"- H4 failures (lower CI of FPR > 2 alpha under source_quantile): "
                f"{int(h4['h4_failure'].sum())} of {len(h4)} rows\n")
        f.write(f"- H5 verdicts: { {k: v['verdict'] for k, v in h5.items()} }\n\n")
        f.write(df_to_markdown(main_df[cols]) + "\n")
    print(df_to_markdown(main_df[cols]))
    print("H5:", {k: v["verdict"] for k, v in h5.items()})
    for n in notes:
        print("note:", n)


if __name__ == "__main__":
    main()
