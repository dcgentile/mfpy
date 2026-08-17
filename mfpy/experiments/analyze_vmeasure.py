"""
Tables and figures for the V-measure experiment. Reads `results/vmeasure/*.csv`.

    python -m mfpy.experiments.analyze_vmeasure
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

RESULT_DIR = os.path.join(REPO_ROOT, "results", "vmeasure")
TABLE_DIR = os.path.join(RESULT_DIR, "tables")
FIG_DIR = os.path.join(RESULT_DIR, "figures")

TL_BINS = [0, 20, 40, 60, 80, 101]


def _write(df, name, float_format="%.3f"):
    os.makedirs(TABLE_DIR, exist_ok=True)
    df.to_csv(os.path.join(TABLE_DIR, f"{name}.csv"), float_format=float_format)
    with open(os.path.join(TABLE_DIR, f"{name}.md"), "w") as fh:
        fh.write(df.to_markdown(floatfmt=".3f"))
        fh.write("\n")
    print(f"wrote {name}")


def tables(toy, lang):
    ok = toy[toy.status == "ok"]

    t1 = (
        toy.assign(failed=toy.status != "ok")
        .groupby("sigma")
        .agg(
            n=("sigma", "size"),
            n_adp_failed=("failed", "sum"),
            mean_adp_k=("adp_k", "mean"),
            v3_adp=("adp_v3", "mean"),
            v2_adp=("adp_v2", "mean"),
            v3_oracle_seg=("oracle_v3", "mean"),
            v2_oracle_seg=("oracle_v2", "mean"),
            v3_kmeans=("kmeans_v3", "mean"),
            v2_kmeans=("kmeans_v2", "mean"),
        )
    )
    _write(t1, "table1_vmeasure_by_sigma")

    noisy = ok[ok.sigma > 0].copy()
    noisy["tl_bin"] = pd.cut(noisy.transition_length, TL_BINS)
    t2 = (
        noisy.groupby(["sigma", "tl_bin"], observed=True)[
            ["adp_v3", "adp_v2", "oracle_v3", "oracle_v2", "kmeans_v3", "kmeans_v2"]
        ]
        .mean()
        .round(3)
    )
    _write(t2, "table2_vmeasure_by_transition_length")

    # How much of the ADP shortfall is attributable to change-point localisation?
    gap = noisy.dropna(subset=["oracle_v3"]).copy()
    gap["localisation_gap_v3"] = gap.oracle_v3 - gap.adp_v3
    t3 = (
        gap.groupby("sigma")
        .agg(
            n=("sigma", "size"),
            v3_adp=("adp_v3", "mean"),
            v3_oracle_seg=("oracle_v3", "mean"),
            localisation_gap=("localisation_gap_v3", "mean"),
            frac_of_shortfall=(
                "localisation_gap_v3",
                lambda s: float(
                    s.mean() / (1.0 - gap.loc[s.index, "adp_v3"].mean())
                ),
            ),
        )
        .round(3)
    )
    _write(t3, "table3_localisation_gap")

    _write(lang.set_index("core_halfwidth"), "table4_langevin_core_sweep")
    return t1, t2, t3


def figures(toy):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(FIG_DIR, exist_ok=True)
    ok = toy[(toy.status == "ok") & (toy.sigma > 0)]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, sigma in zip(axes, sorted(ok.sigma.unique())):
        d = ok[ok.sigma == sigma].sort_values("transition_length")
        for col, label, style in [
            ("oracle_v3", "ADP on true change points", "-"),
            ("adp_v3", "ADP on detected change points", "-"),
            ("kmeans_v3", "$k$-means on values ($k=3$)", "--"),
        ]:
            s = d[["transition_length", col]].dropna()
            roll = s[col].rolling(9, center=True, min_periods=3).mean()
            ax.plot(s.transition_length, roll, style, label=label)
        ax.set_title(rf"$\sigma = {int(sigma)}$")
        ax.set_xlabel("transition length $\\ell$")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("V-measure (3-class target)")
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle(
        "Clustering quality is limited by change-point localisation, not by the "
        "clustering step",
        fontsize=10,
    )
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG_DIR, f"vmeasure_vs_tl.{ext}"), dpi=150)
    plt.close(fig)
    print("wrote vmeasure_vs_tl")


def main():
    toy = pd.read_csv(os.path.join(RESULT_DIR, "vmeasure_toy.csv"))
    lang = pd.read_csv(os.path.join(RESULT_DIR, "vmeasure_langevin.csv"))
    tables(toy, lang)
    figures(toy)


if __name__ == "__main__":
    main()
