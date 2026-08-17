"""
Tables and figures for the q-w interaction experiment.

    python -m mfpy.experiments.analyze_qw_interaction
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

RESULT_DIR = os.path.join(REPO_ROOT, "results", "qw_interaction")
TABLE_DIR = os.path.join(RESULT_DIR, "tables")
FIG_DIR = os.path.join(RESULT_DIR, "figures")

TL_BINS = [0, 25, 50, 75, 101]
METRIC = "f1_tau10"


def _write(df, name):
    os.makedirs(TABLE_DIR, exist_ok=True)
    df.to_csv(os.path.join(TABLE_DIR, f"{name}.csv"), float_format="%.3f")
    with open(os.path.join(TABLE_DIR, f"{name}.md"), "w") as fh:
        fh.write(df.to_markdown(floatfmt=".3f") + "\n")
    print(f"wrote {name}")


def tables(base, res):
    _write(base.pivot_table(index="w", columns="q", values=METRIC).round(3),
           "table1_baseline_f1_by_w_and_q")

    for method in ("baseline", "resample_pointwise"):
        _write(
            res[res.method == method]
            .pivot_table(index="w", columns="q", values=METRIC)
            .round(3),
            f"table2_{method}_f1_by_w_and_q",
        )

    # Best q per w, and the price of getting w wrong at a fixed q.
    rows = []
    for label, d in [("baseline (300 traj)", base)] + [
        (f"{m} (60 traj)", res[res.method == m])
        for m in ("baseline", "resample_pointwise")
    ]:
        g = d.pivot_table(index="w", columns="q", values=METRIC)
        for w in g.index:
            rows.append(
                {
                    "arm": label,
                    "w": w,
                    "best_q": g.loc[w].idxmax(),
                    "f1_at_best_q": g.loc[w].max(),
                }
            )
        worst_case = g.min(axis=0)
        rows.append(
            {
                "arm": label,
                "w": "worst-case over w",
                "best_q": worst_case.idxmax(),
                "f1_at_best_q": worst_case.max(),
            }
        )
    _write(
        pd.DataFrame(rows).set_index(["arm", "w"]).round(3),
        "table3_best_q_per_w",
    )

    # Where the resampling cutoff lands on the observed statistic's own quantile scale.
    rows = []
    obs = res[res.method == "baseline"].groupby(["w", "q"]).cutoff.median().unstack()
    nul = res[res.method == "resample_pointwise"].groupby(["w", "q"]).cutoff.median().unstack()
    for w in obs.index:
        c = nul.loc[w, 0.5]
        rows.append(
            {
                "w": w,
                "null_cutoff_at_q_0.5": c,
                "equivalent_observed_quantile": float(
                    np.interp(c, obs.loc[w].values, obs.loc[w].index.values)
                ),
                "baseline_cutoff_at_q_0.75": obs.loc[w, 0.75],
            }
        )
    _write(pd.DataFrame(rows).set_index("w").round(3), "table4_cutoff_alignment")

    # Interaction with transition length, at each rule's recommended q.
    for method, q in (("baseline", 0.75), ("resample_pointwise", 0.5)):
        d = res[(res.method == method) & (res.q == q)].copy()
        d["tl_bin"] = pd.cut(d.transition_length, TL_BINS)
        _write(
            d.pivot_table(index="w", columns="tl_bin", values=METRIC, observed=True).round(3),
            f"table5_{method}_f1_by_w_and_tl",
        )


def figures(base, res):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(FIG_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.9))
    grids = [
        ("Fixed quantile (300 traj)", base.pivot_table(index="w", columns="q", values=METRIC)),
        (
            "Pointwise permutation null (60 traj)",
            res[res.method == "resample_pointwise"].pivot_table(
                index="w", columns="q", values=METRIC
            ),
        ),
    ]
    for ax, (title, g) in zip(axes, grids):
        im = ax.imshow(g.values, aspect="auto", origin="lower", vmin=0, vmax=0.7,
                       cmap="viridis")
        ax.set_xticks(range(len(g.columns)))
        ax.set_xticklabels([f"{c:g}" for c in g.columns], rotation=90, fontsize=7)
        ax.set_yticks(range(len(g.index)))
        ax.set_yticklabels(g.index)
        for i in range(len(g.index)):
            j = int(np.argmax(g.values[i]))
            ax.plot(j, i, "r*", ms=11)
        ax.set_xlabel("$q$")
        ax.set_ylabel("$w$")
        ax.set_title(title, fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[2]
    obs = res[res.method == "baseline"].groupby(["w", "q"]).cutoff.median().unstack()
    nul = res[res.method == "resample_pointwise"].groupby(["w", "q"]).cutoff.median().unstack()
    ax.plot(obs.index, obs[0.75].values, "o-", label="observed statistic, $q=0.75$")
    ax.plot(nul.index, nul[0.5].values, "s-", label="permutation null, $q=0.5$")
    ax.set_xlabel("$w$")
    ax.set_ylabel("cutoff")
    ax.set_title("The two cutoffs move in opposite directions", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle(
        "Red stars mark the best $q$ at each $w$. The fixed quantile's optimum barely "
        "moves; the permutation null's does.",
        fontsize=9,
    )
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG_DIR, f"qw_interaction.{ext}"), dpi=150)
    plt.close(fig)
    print("wrote qw_interaction")


def main():
    base = pd.read_csv(os.path.join(RESULT_DIR, "qw_interaction_baseline.csv"))
    res = pd.read_csv(os.path.join(RESULT_DIR, "qw_interaction_resample.csv"))
    tables(base, res)
    figures(base, res)


if __name__ == "__main__":
    main()
