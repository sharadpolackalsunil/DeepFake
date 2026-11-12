"""Generate the paper figures from result tables (no hand-typed numbers).

    python -m tools.paper_figures [--tables results/tables] [--scores results/scores] \
        [--target celebdf] [--tag _dev_celebdf] [--out paper/figures]

Figures (PDF + PNG):
  fig1_zero_shot_fpr     FPR at alpha=0.05 per detector and seed: FF++-val source quantile
                         and label-free GMM, with conservative cluster CIs (fpr_control.csv).
  fig2_real_shot_fpr     Mean held-out FPR vs n real identities per policy (real_shot_curve).
  fig3_real_shot_tpr     TPR / oracle TPR vs n.
  fig4_real_shot_fail    P(FPR > 2 alpha) vs n.
  fig5_real_shape        Real-score densities, source (FF++ val) vs target, after median/MAD
                         standardisation: the shape change behind the shape-transfer failure.

Rerun on the confirmatory dataset by passing its --target and --tag.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd

MODEL_ORDER = ["gend_official", "gend_ours", "smsdt_s", "smsdt_st", "smsdt_full"]
MODEL_LABEL = {"gend_official": "GenD (official)", "gend_ours": "GenD-style (ours)",
               "smsdt_s": "S-MSDT-S", "smsdt_st": "S-MSDT-ST", "smsdt_full": "S-MSDT-Full"}
POLICY_STYLE = {  # label, colour, linestyle
    "source_quantile": ("Source quantile (n=0)", "#7f7f7f", ":"),
    "target_gmm": ("Label-free GMM (n=0)", "#bcbd22", ":"),
    "naive_quantile": ("Naive empirical quantile", "#d62728", "--"),
    "conformal": ("Conformal order statistic", "#1f77b4", "-"),
    "np_umbrella": ("NP umbrella", "#9467bd", "-."),
    "loc_transfer": ("Shape transfer (loc)", "#ff7f0e", "--"),
    "oracle": ("Oracle (uses test reals)", "#2ca02c", "-"),
}


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.dpi": 150, "savefig.bbox": "tight"})
    return plt


def _save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"))


def fig_zero_shot(tables, target, alpha, out):
    plt = _plt()
    d = pd.read_csv(os.path.join(tables, "fpr_control.csv"))
    d = d[(d.dataset == target) & (np.isclose(d.alpha, alpha))
          & d.policy.isin(["source_quantile", "target_gmm"])]
    models = [m for m in MODEL_ORDER if m in set(d.model)]
    if not models:
        print(f"fig_zero_shot: no models found for dataset '{target}' in {tables}/fpr_control.csv")
        return
    fig, ax = plt.subplots(figsize=(6.2, 2.6))
    width = 0.38
    for j, (pol, col) in enumerate((("source_quantile", "#7f7f7f"), ("target_gmm", "#bcbd22"))):
        for i, m in enumerate(models):
            g = d[(d.model == m) & (d.policy == pol)].sort_values("seed")
            xs = i + (j - 0.5) * width + np.linspace(-0.1, 0.1, max(len(g), 1))[:len(g)]
            lo = g.fpr - g.get("fpr_ci_low_cons", g.fpr_ci_low)
            hi = g.get("fpr_ci_high_cons", g.fpr_ci_high) - g.fpr
            ax.errorbar(xs, g.fpr, yerr=[lo.clip(lower=0), hi.clip(lower=0)], fmt="o", ms=3.5,
                        color=col, capsize=2, lw=0.8,
                        label=POLICY_STYLE[pol][0] if i == 0 else None)
    ax.axhline(alpha, color="k", lw=0.8)
    ax.axhline(2 * alpha, color="k", lw=0.6, ls="--")
    ax.text(len(models) - 0.45, alpha, r"$\alpha$", va="bottom", fontsize=8)
    ax.text(len(models) - 0.45, 2 * alpha, r"$2\alpha$", va="bottom", fontsize=8)
    ax.set_xticks(range(len(models)), [MODEL_LABEL[m] for m in models], rotation=15)
    ax.set_ylabel(f"FPR on {target} reals")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    ax.set_title(f"Zero-shot thresholds, target budget alpha = {alpha}", fontsize=9)
    _save(fig, out, "fig1_zero_shot_fpr")
    plt.close(fig)


def _curve(tables, tag):
    c = pd.read_csv(os.path.join(tables, f"real_shot_curve{tag}.csv"))
    return c


def fig_real_shot(tables, tag, alpha, out):
    plt = _plt()
    c = _curve(tables, tag)
    c = c[np.isclose(c.alpha, alpha)]
    models = [m for m in MODEL_ORDER if m in set(c.model)]
    if not models:
        print(f"fig_real_shot: no models found in real_shot_curve{tag}.csv")
        return
    oracle = c[c.policy == "oracle"][["model", "seed", "n_draw", "mean_tpr"]].rename(
        columns={"mean_tpr": "oracle_tpr"})
    c = c.merge(oracle, on=["model", "seed", "n_draw"], how="left")
    c["tpr_ratio"] = c.mean_tpr / c.oracle_tpr
    agg = (c.groupby(["model", "policy", "n_draw"], as_index=False)
           [["mean_fpr", "p_fpr_gt_2alpha", "tpr_ratio", "abstain_rate"]].mean())
    specs = (("fig2_real_shot_fpr", "mean_fpr", "Mean held-out FPR", alpha),
             ("fig3_real_shot_tpr", "tpr_ratio", "TPR / oracle TPR", None),
             ("fig4_real_shot_fail", "p_fpr_gt_2alpha", r"P(FPR > 2$\alpha$)", None))
    for name, col, ylab, ref in specs:
        fig, axes = plt.subplots(1, len(models), figsize=(1.95 * len(models), 2.3), sharey=True)
        axes = np.atleast_1d(axes)
        for ax, m in zip(axes, models):
            g = agg[agg.model == m]
            for pol, (lab, colr, ls) in POLICY_STYLE.items():
                if pol == "oracle" and col == "tpr_ratio":
                    continue
                h = g[(g.policy == pol) & (g.abstain_rate < 0.5)].sort_values("n_draw")
                if len(h):
                    ax.plot(h.n_draw, h[col], ls=ls, color=colr, marker="o", ms=2.5, lw=1.1, label=lab)
            if ref is not None:
                ax.axhline(ref, color="k", lw=0.6)
            ax.set_title(MODEL_LABEL[m], fontsize=8)
            ax.set_xlabel("n real identities")
        axes[0].set_ylabel(ylab)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.subplots_adjust(bottom=0.32, wspace=0.12)
        fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=7,
                   bbox_to_anchor=(0.5, -0.02))
        _save(fig, out, name)
        plt.close(fig)


def fig_shape(scores, target, out, runs=("gend_official_seed0", "smsdt_full_seed1", "smsdt_s_seed1")):
    from tools.stats import apply_pure_negatives
    plt = _plt()
    runs = [r for r in runs if os.path.exists(os.path.join(scores, f"{r}_{target}.csv"))]
    if not runs:
        print(f"fig_shape: no score files found for target='{target}' in {scores}")
        return
    fig, axes = plt.subplots(1, len(runs), figsize=(2.3 * len(runs), 2.2), sharey=True)
    axes = np.atleast_1d(axes)
    grid = np.linspace(-5, 8, 400)
    for ax, r in zip(axes, runs):
        for ds, col in (("ffppval", "#7f7f7f"), (target, "#1f77b4")):
            df = pd.read_csv(os.path.join(scores, f"{r}_{ds}.csv"))
            df, _ = apply_pure_negatives(df, ds)
            x = df.loc[df.label == 0, "logit_mean"].values.astype(float)
            med = np.median(x)
            mad = 1.4826 * np.median(np.abs(x - med)) + 1e-6
            z = (x - med) / mad
            bw = 1.06 * z.std() * len(z) ** (-0.2)
            dens = np.exp(-0.5 * ((grid[:, None] - z[None]) / bw) ** 2).sum(1) / (len(z) * bw * np.sqrt(2 * np.pi))
            q95 = np.quantile(z, 0.95)
            ax.plot(grid, dens, color=col, lw=1.2)
            ax.axvline(q95, color=col, lw=0.8, ls="--")
        ax.set_title(r.replace("_seed", " seed "), fontsize=8)
        ax.set_xlim(-4, 6)
        ax.set_xlabel("(logit - median) / MAD")
    axes[0].set_ylabel("density")
    from matplotlib.lines import Line2D
    fig.legend([Line2D([], [], color="#7f7f7f"), Line2D([], [], color="#1f77b4")],
               ["FF++ val reals (source)", f"{target} reals (target)"], loc="lower center",
               ncol=2, frameon=False, fontsize=7, bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(bottom=0.3)
    fig.suptitle("Standardised real-score shape: dashed = 95th percentile", fontsize=8, y=1.03)
    _save(fig, out, "fig5_real_shape")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tables", default="results/tables")
    ap.add_argument("--scores", default="results/scores")
    ap.add_argument("--target", default="celebdf")
    ap.add_argument("--tag", default="_dev_celebdf")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out", default="paper/figures")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fig_zero_shot(a.tables, a.target, a.alpha, a.out)
    fig_real_shot(a.tables, a.tag, a.alpha, a.out)
    fig_shape(a.scores, a.target, a.out)
    print("figures written to", a.out)


if __name__ == "__main__":
    main()
