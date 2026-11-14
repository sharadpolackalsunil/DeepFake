"""Real-Shot Budget Certificate: Closed-Form Guarantee and Sample-Size Planner.

Implements the parameter-free Real-Shot Budget Certificate for video deepfake detection
under identity-level exchangeability, as pre-registered in docs/decisions.md (Decision D20, H8).

Mathematical Formulation:
  Threshold:
    s* = s_{(k)},  where k = ceil((n + 1)(1 - alpha)).
  Population Guarantee (Beta law):
    FPR(s*) ~ Beta(n + 1 - k, k),  E[FPR] = (n + 1 - k) / (n + 1) <= alpha.
  Evaluated Failure Rate on m held-out authentic identities (Beta-Binomial law):
    X ~ BetaBinomial(m, n + 1 - k, k),
    P(FPR_eval > r * alpha) = 1 - F_BB(floor(r * alpha * m); m, n + 1 - k, k).
  Sweet-Spot Planner:
    n = j / alpha - 1  (for integer j >= 1), where E[FPR] equals alpha exactly.
  3-Way Triage Output:
    Alert budget alpha, Review budget beta = 2 * alpha.
    Flag if score > s_{(k_alpha)}, Review if s_{(k_beta)} < score <= s_{(k_alpha)}, Pass if score <= s_{(k_beta)}.
"""

import os
import sys
import math
import argparse
import json
from typing import Dict, List, Optional, Tuple, Sequence

import numpy as np
import pandas as pd
from scipy.stats import betabinom

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tools.fpr_control import resolve_clusters


# ---------------------------------------------------------------------------
# Core Analytical Formulas
# ---------------------------------------------------------------------------

def conformal_rank(n: int, alpha: float) -> Optional[int]:
    """Rank k (1-indexed) of the conformal order statistic.
    Abstains (returns None) if k > n.
    """
    if n <= 0 or alpha <= 0 or alpha >= 1:
        return None
    k = math.ceil((n + 1) * (1.0 - alpha))
    return k if k <= n else None


def expected_fpr(n: int, alpha: float) -> Optional[float]:
    """Theoretical expectation of the population FPR under exchangeability."""
    k = conformal_rank(n, alpha)
    if k is None:
        return None
    return float((n + 1 - k) / (n + 1))


def predict_eval_violation_rate(m: int, n: int, alpha: float, r: float = 2.0) -> Optional[float]:
    """Exact probability that evaluated FPR on m held-out identities exceeds r * alpha.
    Governed by BetaBinomial(m, n + 1 - k, k).
    """
    k = conformal_rank(n, alpha)
    if k is None or m <= 0:
        return None
    a = n + 1 - k
    b = k
    viol_thresh = math.floor(r * alpha * m)
    p_viol = 1.0 - float(betabinom.cdf(viol_thresh, m, a, b))
    return float(np.clip(p_viol, 0.0, 1.0))


def binomial_ci(p: float, R: int = 1000, conf: float = 0.99) -> Tuple[float, float]:
    """Symmetric binomial confidence interval around predicted probability p."""
    # z for 99% two-tailed is ~2.575829
    z = 2.5758293 if conf == 0.99 else 1.959964
    se = math.sqrt(max(0.0, p * (1.0 - p)) / R)
    lower = max(0.0, p - z * se)
    upper = min(1.0, p + z * se)
    return lower, upper


def sweet_spot_ns(alpha: float, max_n: int = 120) -> List[int]:
    """Return calibration sizes n where E[FPR] == alpha exactly."""
    spots = []
    j = 1
    while True:
        n = int(round(j / alpha - 1))
        if n > max_n:
            break
        if n > 0 and conformal_rank(n, alpha) is not None:
            exp_f = expected_fpr(n, alpha)
            if exp_f is not None and abs(exp_f - alpha) < 1e-6:
                spots.append(n)
        j += 1
    return spots


def sample_size_planner(alpha: float = 0.05, delta: float = 0.05, r: float = 2.0,
                        m_eval: int = 400, max_n: int = 100) -> pd.DataFrame:
    """Generate sample-size planning table showing predicted failure rates across n."""
    rows = []
    sweet_set = set(sweet_spot_ns(alpha, max_n=max_n))
    for n in range(1, max_n + 1):
        k = conformal_rank(n, alpha)
        if k is None:
            continue
        exp_f = expected_fpr(n, alpha)
        p_fail = predict_eval_violation_rate(m_eval, n, alpha, r=r)
        is_sweet = n in sweet_set
        satisfies_delta = (p_fail is not None and p_fail <= delta)
        rows.append({
            "n": n,
            "k": k,
            "expected_fpr": exp_f,
            "is_sweet_spot": is_sweet,
            "p_fpr_gt_2alpha": p_fail,
            "meets_delta": satisfies_delta,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Triage Policy (Flag / Review / Pass)
# ---------------------------------------------------------------------------

def compute_triage_thresholds(calib_reals: np.ndarray, alpha: float = 0.05,
                              beta: float = 0.10) -> Dict[str, float]:
    """Calculate dual alert (alpha) and review (beta) thresholds from n real videos."""
    n = len(calib_reals)
    sorted_reals = np.sort(calib_reals)
    k_alpha = conformal_rank(n, alpha)
    k_beta = conformal_rank(n, beta)
    thr_alert = sorted_reals[k_alpha - 1] if k_alpha is not None else float("inf")
    thr_review = sorted_reals[k_beta - 1] if k_beta is not None else float("inf")
    return {
        "thr_alert": float(thr_alert),
        "thr_review": float(thr_review),
        "k_alpha": k_alpha,
        "k_beta": k_beta,
    }


def triage_classify(scores: np.ndarray, thr_alert: float, thr_review: float) -> np.ndarray:
    """Classify video scores into 3 operational bands:
      'FLAG' (score > thr_alert)
      'REVIEW' (thr_review < score <= thr_alert)
      'PASS' (score <= thr_review)
    """
    labels = np.full(len(scores), "PASS", dtype=object)
    labels[scores > thr_review] = "REVIEW"
    labels[scores > thr_alert] = "FLAG"
    return labels


# ---------------------------------------------------------------------------
# Confirmatory H8 Evaluation Harness (130 Pre-Registered Cells)
# ---------------------------------------------------------------------------

MODELS = [
    "gend_official_seed0",
    "gend_ours_seed1", "gend_ours_seed2", "gend_ours_seed3",
    "smsdt_s_seed1", "smsdt_s_seed2", "smsdt_s_seed3",
    "smsdt_st_seed1", "smsdt_st_seed2", "smsdt_st_seed3",
    "smsdt_full_seed1", "smsdt_full_seed2", "smsdt_full_seed3",
]

H8_CONFIGS = [
    (0.05, 19),
    (0.05, 39),
    (0.05, 59),
    (0.10, 9),
    (0.10, 29),
]

DATASETS = ["fakeavceleb", "celebdf"]


def load_dataset_scores(model: str, dataset: str, scores_dir: str = "results/scores") -> pd.DataFrame:
    """Load pre-registered score CSV and extract resolved identity clusters."""
    path = os.path.join(scores_dir, f"{model}_{dataset}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing score CSV: {path}")
    df = pd.read_csv(path)
    # Filter pure reals if dataset == celebdf (cluster_id containing celeb_real or valid IDs)
    df["cluster_id"] = resolve_clusters(df)
    return df


def evaluate_h8_cell(
    target_df: pd.DataFrame,
    alpha: float,
    n: int,
    R: int = 1000,
    seed: int = 42,
    score_col: str = "score_mean"
) -> Dict:
    """Evaluate one experimental cell over R draws with one video per identity."""
    y = target_df["label"].values.astype(int)
    scores = target_df[score_col].values.astype(np.float64)
    clusters = target_df["cluster_id"].values.astype(str)

    real_idx = np.where(y == 0)[0]
    fake_idx = np.where(y == 1)[0]
    real_clusters_unique = np.unique(clusters[real_idx])
    n_total_real = len(real_clusters_unique)

    if n >= n_total_real:
        raise ValueError(f"Calibration size n={n} >= total real identities {n_total_real}")

    m_eval = n_total_real - n
    k = conformal_rank(n, alpha)
    if k is None:
        raise ValueError(f"Conformal rank is None for n={n}, alpha={alpha}")

    # Theoretical prediction
    p_pred = predict_eval_violation_rate(m_eval, n, alpha, r=2.0)
    exp_f = expected_fpr(n, alpha)
    ci_low, ci_high = binomial_ci(p_pred, R=R, conf=0.99)

    # Pre-index cluster rows
    cl_to_idx = {c: real_idx[clusters[real_idx] == c] for c in real_clusters_unique}

    rng = np.random.RandomState(seed)

    violation_count = 0
    fpr_list = []
    tpr_list = []
    oracle_tpr_list = []

    viol_thresh = math.floor(2.0 * alpha * m_eval)
    fake_scores = scores[fake_idx]

    for _ in range(R):
        # 1. Draw 1 random video per identity
        sampled_reals = np.array([scores[rng.choice(cl_to_idx[c])] for c in real_clusters_unique])

        # 2. Split into n calibration and m held-out
        perm = rng.permutation(n_total_real)
        calib_scores = sampled_reals[perm[:n]]
        eval_scores = sampled_reals[perm[n:]]

        # 3. Conformal threshold
        calib_sorted = np.sort(calib_scores)
        thr = calib_sorted[k - 1]

        # 4. Evaluated FPR on m held-out identities
        fp_count = np.sum(eval_scores > thr)
        fpr = fp_count / m_eval
        fpr_list.append(fpr)
        if fp_count > viol_thresh:
            violation_count += 1

        # 5. Evaluated TPR on all fakes
        tpr = float(np.mean(fake_scores > thr))
        tpr_list.append(tpr)

        # 6. Oracle TPR on held-out reals at (1 - alpha)
        oracle_thr = float(np.quantile(eval_scores, 1.0 - alpha))
        oracle_tpr_list.append(float(np.mean(fake_scores > oracle_thr)))

    obs_p = violation_count / R
    mean_fpr = float(np.mean(fpr_list))
    sd_fpr = float(np.std(fpr_list))
    se_mc = sd_fpr / math.sqrt(R)
    mean_tpr = float(np.mean(tpr_list))
    mean_oracle_tpr = float(np.mean(oracle_tpr_list))
    retained_tpr_ratio = mean_tpr / mean_oracle_tpr if mean_oracle_tpr > 0 else float("nan")

    # Verdict criteria
    ci_pass = (ci_low <= obs_p <= ci_high)
    validity_pass = (mean_fpr <= alpha + 2.0 * se_mc)
    cell_pass = ci_pass and validity_pass

    return {
        "alpha": alpha,
        "n": n,
        "k": k,
        "m_eval": m_eval,
        "n_total_real": n_total_real,
        "expected_fpr": exp_f,
        "pred_p_viol": p_pred,
        "ci_low_99": ci_low,
        "ci_high_99": ci_high,
        "obs_p_viol": obs_p,
        "mean_fpr": mean_fpr,
        "sd_fpr": sd_fpr,
        "se_mc": se_mc,
        "mean_tpr": mean_tpr,
        "oracle_tpr": mean_oracle_tpr,
        "retained_tpr_ratio": retained_tpr_ratio,
        "ci_pass": ci_pass,
        "validity_pass": validity_pass,
        "cell_pass": cell_pass,
    }


def run_full_h8_evaluation(
    scores_dir: str = "results/scores",
    out_csv: str = "results/tables/h8_certificate_eval.csv",
    out_json: str = "results/tables/h8_certificate_verdict.json",
    out_md: str = "results/tables/h8_certificate_summary.md",
    R: int = 1000,
    seed: int = 42,
) -> Dict:
    """Run the complete 130-cell confirmatory evaluation across 13 models, 2 datasets, and 5 configs."""
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    results = []

    print(f"=== Starting Confirmatory H8 Evaluation ({len(MODELS)} models x {len(DATASETS)} datasets x {len(H8_CONFIGS)} configs = 130 cells) ===")

    cell_count = 0
    for dname in DATASETS:
        for model in MODELS:
            try:
                target_df = load_dataset_scores(model, dname, scores_dir=scores_dir)
            except Exception as e:
                print(f"Error loading {model} on {dname}: {e}")
                continue

            for alpha, n in H8_CONFIGS:
                cell_count += 1
                res = evaluate_h8_cell(target_df, alpha, n, R=R, seed=seed)
                res["model"] = model
                res["dataset"] = dname
                res["cell_id"] = f"{dname}_{model}_a{int(alpha*100)}_n{n}"
                results.append(res)
                status_str = "PASS" if res["cell_pass"] else "FAIL"
                print(f"[{cell_count:3d}/130] {dname:11s} | {model:20s} | a={alpha:.2f}, n={n:2d} | "
                      f"Pred={res['pred_p_viol']:.4f} Obs={res['obs_p_viol']:.4f} "
                      f"[{res['ci_low_99']:.4f}, {res['ci_high_99']:.4f}] | "
                      f"FPR={res['mean_fpr']:.4f} | {status_str}")

    df_res = pd.DataFrame(results)
    df_res.to_csv(out_csv, index=False)

    total_cells = len(df_res)
    ci_passes = int(df_res["ci_pass"].sum())
    validity_passes = int(df_res["validity_pass"].sum())
    full_passes = int(df_res["cell_pass"].sum())

    ci_rate = ci_passes / total_cells if total_cells > 0 else 0.0
    validity_rate = validity_passes / total_cells if total_cells > 0 else 0.0

    # Decision criteria: >= 90% inside 99% CI AND 100% mean FPR <= alpha + 2*SE
    h8_supported = (ci_rate >= 0.90) and (validity_passes == total_cells)

    verdict_data = {
        "hypothesis": "H8",
        "title": "Real-Shot Budget Certificate Beta-Binomial Prediction",
        "total_cells": total_cells,
        "ci_pass_cells": ci_passes,
        "ci_pass_rate": ci_rate,
        "ci_threshold_required": 0.90,
        "validity_pass_cells": validity_passes,
        "validity_pass_rate": validity_rate,
        "validity_required_cells": total_cells,
        "full_pass_cells": full_passes,
        "h8_verdict": "SUPPORTED" if h8_supported else "FALSIFIED",
        "datasets": DATASETS,
        "models": MODELS,
        "configs": H8_CONFIGS,
        "R": R,
        "seed": seed,
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(verdict_data, f, indent=2)

    # Generate Markdown Summary
    md_content = rf"""# Hypothesis H8 Confirmatory Verdict: Real-Shot Budget Certificate

- **Hypothesis H8 Verdict:** **`{verdict_data['h8_verdict']}`**
- **Calibration Fidelity (Beta-Binomial 99% CI Coverage):** **`{ci_passes}/{total_cells}`** cells (**`{ci_rate*100:.1f}%`**, required $\\ge 90\%$)
- **Validity Guarantee (Mean FPR $\\le \\alpha + 2\\text{{SE}}$):** **`{validity_passes}/{total_cells}`** cells (**`{validity_rate*100:.1f}%`**, required $100\%$)
- **Total Evaluated Cells:** **`{total_cells}`** across 13 models, 2 datasets (`fakeavceleb`, `celebdf`), and 5 sweet-spot $(\\alpha, n)$ configurations ($R=1,000$ draws per cell).

## Aggregate Observed vs Predicted Failure Rates by Configuration

| Dataset | Budget $\\alpha$ | Reals $n$ | Held-out $m$ | Predicted $P(\\text{{FPR}} > 2\\alpha)$ | Mean Observed $P(\\text{{FPR}} > 2\\alpha)$ | Mean Empirical FPR | Mean Retained TPR |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
"""
    for dname in DATASETS:
        for alpha, n in H8_CONFIGS:
            sub = df_res[(df_res["dataset"] == dname) & np.isclose(df_res["alpha"], alpha) & (df_res["n"] == n)]
            p_pred = sub["pred_p_viol"].iloc[0]
            m_eval = sub["m_eval"].iloc[0]
            obs_p_mean = sub["obs_p_viol"].mean()
            mean_fpr = sub["mean_fpr"].mean()
            mean_ret_tpr = sub["retained_tpr_ratio"].mean()
            md_content += f"| `{dname}` | `{alpha:.2f}` | `{n:2d}` | `{m_eval:3d}` | **`{p_pred:.4f}`** | **`{obs_p_mean:.4f}`** | `{mean_fpr:.4f}` | `{mean_ret_tpr*100:.1f}%` |\n"

    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"\n=== Evaluation Complete: H8 is {verdict_data['h8_verdict']} ===")
    print(f"Results written to:\n - {out_csv}\n - {out_json}\n - {out_md}")
    return verdict_data


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Real-Shot Budget Certificate Tool & H8 Harness")
    parser.add_argument("--evaluate-h8", action="store_true", help="Execute pre-registered 130-cell confirmatory evaluation")
    parser.add_argument("--plan", action="store_true", help="Display sample-size planning table")
    parser.add_argument("--alpha", type=float, default=0.05, help="Nominal false-alarm budget (default: 0.05)")
    parser.add_argument("--delta", type=float, default=0.05, help="Tail violation tolerance (default: 0.05)")
    parser.add_argument("--m-eval", type=int, default=480, help="Held-out evaluation identities (default: 480)")
    parser.add_argument("--scores-dir", type=str, default="results/scores", help="Score CSV directory")
    args = parser.parse_args()

    if args.plan:
        planner_df = sample_size_planner(alpha=args.alpha, delta=args.delta, m_eval=args.m_eval)
        print(planner_df.to_string(index=False))
        return

    if args.evaluate_h8:
        run_full_h8_evaluation(scores_dir=args.scores_dir)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
