"""
Tables and figure for the variance-change experiment.

    python -m mfpy.experiments.analyze_variance_change
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

RESULT_DIR = os.path.join(REPO_ROOT, "results", "variance_change")
TABLE_DIR = os.path.join(RESULT_DIR, "tables")
FIG_DIR = os.path.join(RESULT_DIR, "figures")

CELL = ["kind", "separation", "transition_length", "w", "extractor", "detector"]


def _write(df, name):
    os.makedirs(TABLE_DIR, exist_ok=True)
    df.to_csv(os.path.join(TABLE_DIR, f"{name}.csv"), float_format="%.3f")
    with open(os.path.join(TABLE_DIR, f"{name}.md"), "w") as fh:
        fh.write(df.to_markdown(floatfmt=".3f") + "\n")
    print(f"wrote {name}")


def best_q(df):
    """
    Collapse the q sweep: for each cell take the q with the highest mean F1 over reps.

    Both detectors get their own best q, so neither is handicapped by a threshold tuned
    for the other.
    """
    g = (
        df.groupby(CELL + ["q"])
        .f1_tau10.agg(mean="mean", std="std", n="size")
        .reset_index()
    )
    return g.loc[g.groupby(CELL)["mean"].idxmax()].drop(columns="q")


def tables(best):
    inst = best[(best.transition_length == 0) & (best.extractor == "peak")]
    _write(
        inst.pivot_table(
            index=["kind", "separation"], columns=["w", "detector"], values="mean"
        ).round(3),
        "table1_instantaneous_f1",
    )
    _write(
        inst.pivot_table(
            index=["kind", "separation"], columns=["w", "detector"], values="std"
        ).round(3),
        "table2_instantaneous_f1_std",
    )

    grad = best[(best.transition_length > 0) & (best.extractor == "gradient")]
    _write(
        grad[grad.w == 50]
        .pivot_table(
            index=["kind", "separation"],
            columns=["transition_length", "detector"],
            values="mean",
        )
        .round(3),
        "table3_gradual_f1_w50",
    )

    # Head-to-head: the ratio that carries the argument.
    piv = best.pivot_table(
        index=["kind", "transition_length", "w", "extractor", "separation"],
        columns="detector",
        values="mean",
    )
    piv["w2_minus_mean"] = piv["w2"] - piv["mean"]
    _write(
        piv.groupby(level=["kind", "w"])[["w2", "mean", "w2_minus_mean"]]
        .mean()
        .round(3),
        "table4_head_to_head_by_kind_and_window",
    )

    # Window dependence, which is the 3(e)-relevant part.
    _write(
        best[(best.transition_length == 0) & (best.extractor == "peak")]
        .groupby(["kind", "detector", "w"])["mean"]
        .mean()
        .unstack()
        .round(3),
        "table5_window_dependence",
    )


def figure(best):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(FIG_DIR, exist_ok=True)
    inst = best[(best.transition_length == 0) & (best.extractor == "peak")]

    kinds = ["mean", "variance", "both"]
    titles = {
        "mean": "Change in mean only",
        "variance": "Change in variance only",
        "both": "Change in both",
    }
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, kind in zip(axes, kinds):
        d = inst[inst.kind == kind]
        for det, colour, marker in [("w2", "C0", "o"), ("mean", "C1", "s")]:
            for w, ls in [(25, ":"), (50, "--"), (100, "-")]:
                s = d[(d.detector == det) & (d.w == w)].sort_values("separation")
                ax.errorbar(
                    s.separation,
                    s["mean"],
                    yerr=s["std"] / np.sqrt(s["n"]),
                    color=colour,
                    ls=ls,
                    marker=marker,
                    ms=4,
                    lw=1.2,
                    capsize=2,
                )
        ax.set_xscale("log")
        ax.set_xlabel("state separation $W_2(A,B)$")
        ax.set_title(titles[kind], fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(r"$F_1$ ($\tau=10$)")

    handles = [
        plt.Line2D([], [], color="C0", marker="o", ls="-", label="Wasserstein statistic"),
        plt.Line2D([], [], color="C1", marker="s", ls="-", label="mean statistic (CUSUM-type)"),
        plt.Line2D([], [], color="0.4", ls=":", label="$w=25$"),
        plt.Line2D([], [], color="0.4", ls="--", label="$w=50$"),
        plt.Line2D([], [], color="0.4", ls="-", label="$w=100$"),
    ]
    axes[0].legend(handles=handles, fontsize=7, loc="upper left")
    fig.suptitle(
        "Instantaneous changes, 10 replicates per point, error bars are standard errors",
        fontsize=10,
    )
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG_DIR, f"variance_change.{ext}"), dpi=150)
    plt.close(fig)
    print("wrote variance_change")


def main():
    df = pd.read_csv(os.path.join(RESULT_DIR, "variance_change.csv"))
    best = best_q(df)
    tables(best)
    figure(best)


if __name__ == "__main__":
    main()
