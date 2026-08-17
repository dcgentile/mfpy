"""
Tables and figures for the distributional-fidelity experiment.

    python -m mfpy.experiments.analyze_distributional_fidelity
"""

from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message=".*intrinsic dimension is not defined.*")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

RESULT_DIR = os.path.join(REPO_ROOT, "results", "distributional_fidelity")
TABLE_DIR = os.path.join(RESULT_DIR, "tables")
FIG_DIR = os.path.join(RESULT_DIR, "figures")

TL_BINS = [0, 20, 40, 60, 80, 101]


def _write(df, name):
    os.makedirs(TABLE_DIR, exist_ok=True)
    df.to_csv(os.path.join(TABLE_DIR, f"{name}.csv"), float_format="%.3f")
    with open(os.path.join(TABLE_DIR, f"{name}.md"), "w") as fh:
        fh.write(df.to_markdown(floatfmt=".3f") + "\n")
    print(f"wrote {name}")


def tables(toy, lang, per_cluster):
    ok = toy[toy.status == "ok"]

    t1 = ok.groupby("sigma").agg(
        n=("sigma", "size"),
        fid_adp=("adp3_fidelity_norm", "mean"),
        fid_oracle_seg=("oracle3_fidelity_norm", "mean"),
        fid_kmeans=("kmeans3_fidelity_norm", "mean"),
        cov_adp=("adp3_coverage_norm", "mean"),
        cov_oracle_seg=("oracle3_coverage_norm", "mean"),
        cov_kmeans=("kmeans3_coverage_norm", "mean"),
    )
    _write(t1.round(3), "table1_fidelity_by_sigma")

    noisy = ok[ok.sigma > 0].copy()
    noisy["tl_bin"] = pd.cut(noisy.transition_length, TL_BINS)
    t2 = (
        noisy.groupby(["sigma", "tl_bin"], observed=True)[
            [
                "adp3_fidelity_norm",
                "oracle3_fidelity_norm",
                "kmeans3_fidelity_norm",
            ]
        ]
        .mean()
        .round(3)
    )
    _write(t2, "table2_fidelity_by_transition_length")

    # Head-to-head win rates: the fraction of trajectories on which each variant is
    # distributionally closer to the true laws than k-means.
    wins = ok.assign(
        adp_beats_kmeans=ok.adp3_fidelity_norm < ok.kmeans3_fidelity_norm,
        oracle_beats_kmeans=ok.oracle3_fidelity_norm < ok.kmeans3_fidelity_norm,
    )
    t3 = wins.groupby("sigma").agg(
        n=("sigma", "size"),
        adp_beats_kmeans=("adp_beats_kmeans", "mean"),
        n_oracle=("oracle3_fidelity_norm", "count"),
        oracle_beats_kmeans=("oracle_beats_kmeans", "sum"),
    )
    t3["oracle_beats_kmeans"] = (t3.oracle_beats_kmeans / t3.n_oracle).fillna(np.nan)
    _write(t3.round(3), "table3_win_rate_vs_kmeans")

    _write(
        lang.set_index("core_halfwidth")[
            [
                "adp_k",
                "adp3_fidelity_norm",
                "kmeans3_fidelity_norm",
                "adp3_coverage_norm",
                "kmeans3_coverage_norm",
            ]
        ].round(3),
        "table4_langevin_core_sweep",
    )
    _write(per_cluster.set_index(["method", "cluster"]).round(3), "table5_langevin_per_cluster")


def figure_langevin(c=0.0):
    """Recovered cluster laws against the true basin laws, ADP vs k-means."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from mfpy.experiments.distributional_fidelity import _profiles
    from mfpy.experiments.vmeasure import (
        LANG_Q,
        LANG_W,
        LANGEVIN_PATH,
        cluster_trajectory,
        kmeans_labels,
        langevin_point_labels,
    )

    os.makedirs(FIG_DIR, exist_ok=True)
    X = np.loadtxt(LANGEVIN_PATH).ravel()
    truth = langevin_point_labels(X, c)
    pred, _, _ = cluster_trajectory(X, LANG_W, LANG_Q)
    km = kmeans_labels(X, 3)

    bins = np.linspace(X.min(), X.max(), 120)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, (title, labels) in zip(
        axes, [("Segment clustering (ADP)", pred), ("$k$-means on values", km)]
    ):
        for lab, (_, mass) in sorted(
            _profiles(X, labels).items(), key=lambda kv: -kv[1][1]
        ):
            ax.hist(
                X[labels == lab],
                bins=bins,
                alpha=0.55,
                label=f"cluster {int(lab)} ({mass:.0%})",
            )
        ax.axvline(0.0, color="k", lw=0.8, ls="--")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("$x$")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("count")
    fig.suptitle(
        "Langevin: k-means clusters are truncated slabs; segment clusters carry the "
        "tails across the barrier",
        fontsize=10,
    )
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG_DIR, f"langevin_cluster_laws.{ext}"), dpi=150)
    plt.close(fig)
    print("wrote langevin_cluster_laws")


def figure_toy(toy):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(FIG_DIR, exist_ok=True)
    ok = toy[(toy.status == "ok") & (toy.sigma > 0)]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, sigma in zip(axes, sorted(ok.sigma.unique())):
        d = ok[ok.sigma == sigma].sort_values("transition_length")
        for col, label, style in [
            ("oracle3_fidelity_norm", "ADP on true change points", "-"),
            ("adp3_fidelity_norm", "ADP on detected change points", "-"),
            ("kmeans3_fidelity_norm", "$k$-means on values ($k=3$)", "--"),
        ]:
            s = d[["transition_length", col]].dropna()
            ax.plot(
                s.transition_length,
                s[col].rolling(9, center=True, min_periods=3).mean(),
                style,
                label=label,
            )
        ax.set_title(rf"$\sigma = {int(sigma)}$", fontsize=10)
        ax.set_xlabel("transition length $\\ell$")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("$W_2$ error / A-to-B separation")
    axes[0].legend(fontsize=8)
    fig.suptitle(
        "Distributional error of recovered cluster laws (lower is better)", fontsize=10
    )
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG_DIR, f"toy_fidelity_vs_tl.{ext}"), dpi=150)
    plt.close(fig)
    print("wrote toy_fidelity_vs_tl")


def main():
    toy = pd.read_csv(os.path.join(RESULT_DIR, "distfid_toy.csv"))
    lang = pd.read_csv(os.path.join(RESULT_DIR, "distfid_langevin.csv"))
    per_cluster = pd.read_csv(
        os.path.join(RESULT_DIR, "distfid_langevin_per_cluster.csv")
    )
    tables(toy, lang, per_cluster)
    figure_toy(toy)
    figure_langevin()


if __name__ == "__main__":
    main()
