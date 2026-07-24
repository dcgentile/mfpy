"""
Aggregate the toy-model sweeps into paper-ready tables and figures.

Reads the CSVs written by ``toy_sweep.py`` and ``toy_threshold_sweep.py``.

Because the estimator has a transition-length-dependent localisation bias, no single
tolerance is treated as privileged: the tau curve is the primary detection result, and
signed localisation error is reported alongside as a threshold-free diagnostic.

Usage:
    python mfpy/experiments/analyze_toy_sweep.py [--results DIR]
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

TL_BINS = [0, 10, 25, 50, 75, 100]
TL_LABELS = ["1-10", "11-25", "26-50", "51-75", "76-100"]

SETTING_ORDER = ["oracle", "w10", "w25", "w50", "w100"]
SETTING_LABEL = {
    "oracle": r"$w=2\ell$ (oracle)",
    "w10": r"$w=10$",
    "w25": r"$w=25$",
    "w50": r"$w=50$",
    "w100": r"$w=100$",
}
EST_ORDER = ["gradient", "run_edge", "half_max"]
EST_LABEL = {
    "gradient": "gradient (incumbent)",
    "run_edge": "run edge",
    "half_max": "half max",
}


def _write(df, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, f"{name}.csv"))
    with open(os.path.join(out_dir, f"{name}.md"), "w") as f:
        f.write(df.to_markdown(floatfmt=".3f"))
        f.write("\n")
    return df


def add_bins(df):
    df = df.copy()
    df["tl_bin"] = pd.cut(
        df["transition_length"], bins=TL_BINS, labels=TL_LABELS, include_lowest=True
    )
    if "window_setting" in df.columns:
        df["window_setting"] = pd.Categorical(
            df["window_setting"], categories=SETTING_ORDER, ordered=True
        )
    if "estimator" in df.columns:
        df["estimator"] = pd.Categorical(
            df["estimator"], categories=EST_ORDER, ordered=True
        )
    return df


# --------------------------------------------------------------------------- tables


def table_detection(df, out_dir):
    """F1 at several tolerances, by estimator and window."""
    cols = [c for c in df.columns if c.startswith("f1_tau")]
    cols.sort(key=lambda c: int(c.split("tau")[1]))
    t = df.groupby(["estimator", "window_setting"], observed=True)[cols].mean()
    return _write(t, out_dir, "table1_detection_by_tolerance")


def table_localisation(df, out_dir):
    """Localisation error by estimator and transition-length bin."""
    t = (
        df.groupby(["estimator", "tl_bin"], observed=True)[
            ["median_abs_error", "median_signed_error"]
        ]
        .median()
        .unstack("tl_bin")
    )
    return _write(t, out_dir, "table2_localisation_by_tl")


def table_estimator_equivalence(df, out_dir):
    """
    Do the three extraction rules differ? Pairwise mean absolute gap in localisation
    error, per window setting. Small numbers here are the evidence that the bias is
    not caused by the choice of extraction rule.
    """
    piv = df.pivot_table(
        index=["window_setting", "sigma", "transition_length"],
        columns="estimator",
        values="median_abs_error",
        observed=True,
    )
    rows = {}
    for a, b in [("gradient", "run_edge"), ("gradient", "half_max"), ("run_edge", "half_max")]:
        d = (piv[a] - piv[b]).abs()
        rows[f"{a} vs {b}"] = {
            "mean |gap| (steps)": d.mean(),
            "median |gap| (steps)": d.median(),
            "p90 |gap| (steps)": d.quantile(0.9),
        }
    t = pd.DataFrame(rows).T
    return _write(t, out_dir, "table3_estimator_equivalence")


def table_window_choice(df, out_dir):
    """Detection F1 at a generous tolerance, by window and transition-length bin."""
    col = "f1_tau50"
    t = (
        df[df["estimator"] == "gradient"]
        .groupby(["window_setting", "tl_bin"], observed=True)[col]
        .mean()
        .unstack("tl_bin")
    )
    return _write(t, out_dir, "table4_window_choice")


def table_threshold(thr, out_dir):
    """Empirically best q against the predicted q* = 1 - ramp fraction."""
    best = thr.loc[
        thr.groupby(["sigma", "transition_length"])["median_abs_error"].idxmin()
    ]
    t = (
        best.groupby("transition_length")[["q", "q_star", "median_abs_error"]]
        .median()
        .rename(
            columns={
                "q": "best q (empirical)",
                "q_star": "q* = 1 - ramp fraction",
                "median_abs_error": "median |error| at best q",
            }
        )
    )
    subset = t.loc[[1, 5, 10, 20, 40, 60, 80, 100]]

    # what the fixed q=0.95 costs, versus the grid point nearest q*
    fixed = thr[np.isclose(thr["q"], 0.95)].groupby("transition_length")[
        "median_abs_error"
    ].median()
    subset = subset.join(fixed.rename("median |error| at q=0.95"))
    return _write(subset, out_dir, "table5_threshold_diagnosis"), best


def prior_tests(df, out_dir):
    """
    Re-test the stated prior under boundary scoring: is detection better for abrupt
    transitions (small tl) and tight states (small sigma)?

    Uses F1 at tau=25, a tolerance generous enough that the results are not dominated
    by the localisation bias documented separately.
    """
    metric = "f1_tau25"
    g = df[df["estimator"] == "gradient"]

    rows = []
    for (setting, sigma), sub in g.groupby(["window_setting", "sigma"], observed=True):
        if sub[metric].nunique() <= 1:
            rho, p = np.nan, np.nan
        else:
            rho, p = spearmanr(sub["transition_length"], sub[metric])
        rows.append(
            {
                "window_setting": setting,
                "sigma": sigma,
                "rho_F1_vs_tl": rho,
                "p_value": p,
                "mean_F1": sub[metric].mean(),
            }
        )
    t1 = pd.DataFrame(rows).set_index(["window_setting", "sigma"])

    rows = []
    for setting, sub in g.groupby("window_setting", observed=True):
        rho, p = spearmanr(sub["sigma"], sub[metric])
        rows.append(
            {"window_setting": setting, "rho_F1_vs_sigma": rho, "p_value": p}
        )
    t2 = pd.DataFrame(rows).set_index("window_setting")

    _write(t1, out_dir, "prior_tests_tl")
    _write(t2, out_dir, "prior_tests_sigma")
    return t1, t2


# -------------------------------------------------------------------------- figures


def fig_tolerance_curves(curve, fig_dir):
    """Primary detection result: F1 as a function of tau, per estimator."""
    ests = [e for e in EST_ORDER if e in set(curve["estimator"])]
    fig, axes = plt.subplots(1, len(ests), figsize=(4.8 * len(ests), 4), sharey=True)
    if len(ests) == 1:
        axes = [axes]

    for ax, est in zip(axes, ests):
        sub = curve[curve["estimator"] == est]
        for setting in SETTING_ORDER:
            s = (
                sub[sub["window_setting"] == setting]
                .groupby("tau")["f1"]
                .mean()
                .sort_index()
            )
            if s.empty:
                continue
            ax.plot(s.index, s.values, marker="o", ms=3, label=SETTING_LABEL[setting])
        ax.set_title(EST_LABEL[est])
        ax.set_xlabel(r"tolerance $\tau$ (time steps)")
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel(r"mean $F_1$")
    axes[-1].legend(loc="lower right", fontsize=9)
    fig.suptitle(
        "Detection accuracy against true ramp boundaries, as a function of tolerance",
        y=1.02,
    )
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(
            os.path.join(fig_dir, f"tolerance_curves.{ext}"),
            dpi=200,
            bbox_inches="tight",
        )
    plt.close(fig)


def fig_localisation_bias(df, fig_dir):
    """The central finding: signed localisation error grows linearly with ramp length."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    ax = axes[0]
    g = df[df["estimator"] == "gradient"]
    for setting in SETTING_ORDER:
        s = (
            g[g["window_setting"] == setting]
            .groupby("transition_length")["median_signed_error"]
            .median()
            .sort_index()
        )
        if s.empty:
            continue
        ax.plot(s.index, s.values, lw=1.6, label=SETTING_LABEL[setting], alpha=0.9)
    lo, hi = 0, df["transition_length"].max()
    ax.plot(
        [lo, hi], [0, 0.4 * hi], "k--", lw=1.2, alpha=0.7, label=r"$0.4\,\ell$ reference"
    )
    ax.axhline(0, color="k", lw=0.8, alpha=0.5)
    ax.set_xlabel(r"transition length $\ell$")
    ax.set_ylabel("median signed error (steps)")
    ax.set_title("Change points fall inside the ramp,\nby an amount proportional to its length")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)

    ax = axes[1]
    for est in EST_ORDER:
        s = (
            df[(df["estimator"] == est) & (df["window_setting"] == "w25")]
            .groupby("transition_length")["median_abs_error"]
            .median()
            .sort_index()
        )
        if s.empty:
            continue
        ax.plot(s.index, s.values, lw=1.6, label=EST_LABEL[est], alpha=0.9)
    ax.set_xlabel(r"transition length $\ell$")
    ax.set_ylabel("median |error| (steps)")
    ax.set_title("All three extraction rules share the bias\n" r"($w=25$)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(
            os.path.join(fig_dir, f"localisation_bias.{ext}"),
            dpi=200,
            bbox_inches="tight",
        )
    plt.close(fig)


def fig_threshold_diagnosis(thr, best, fig_dir):
    """The quantile threshold explains the bias, and q* removes it."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    ax = axes[0]
    for sigma, m in zip([0, 5, 20], ["o", "s", "^"]):
        s = best[best["sigma"] == sigma]
        ax.scatter(s["q_star"], s["q"], s=16, marker=m, alpha=0.6, label=rf"$\sigma={sigma}$")
    lims = [0.65, 1.005]
    ax.plot(lims, lims, "k--", lw=1.2, alpha=0.7, label="identity")
    ax.set_xlim(*lims)
    ax.set_ylim(*lims)
    ax.set_xlabel(r"$q^* = 1 - $ fraction of trajectory in transition")
    ax.set_ylabel("empirically best $q$")
    ax.set_title("The threshold that minimises localisation error\nis predicted by the ramp fraction")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    fixed = (
        thr[np.isclose(thr["q"], 0.95)]
        .groupby("transition_length")["median_abs_error"]
        .median()
    )
    tuned = best.groupby("transition_length")["median_abs_error"].median()
    ax.plot(fixed.index, fixed.values, lw=1.8, label=r"fixed $q=0.95$")
    ax.plot(tuned.index, tuned.values, lw=1.8, label=r"$q$ at its empirical optimum")
    ax.set_xlabel(r"transition length $\ell$")
    ax.set_ylabel("median |error| (steps)")
    ax.set_title("Cost of the fixed threshold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(
            os.path.join(fig_dir, f"threshold_diagnosis.{ext}"),
            dpi=200,
            bbox_inches="tight",
        )
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--results", default=os.path.join(REPO_ROOT, "results", "toy_sweep")
    )
    args = ap.parse_args()

    df = add_bins(
        pd.read_csv(os.path.join(args.results, "toy_sweep_per_trajectory.csv"))
    )
    curve = add_bins(
        pd.read_csv(os.path.join(args.results, "toy_sweep_tolerance_curve.csv"))
    )
    thr_path = os.path.join(args.results, "toy_threshold_sweep.csv")
    thr = pd.read_csv(thr_path) if os.path.exists(thr_path) else None

    tables_dir = os.path.join(args.results, "tables")
    fig_dir = os.path.join(args.results, "figures")
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    t_det = table_detection(df, tables_dir)
    t_loc = table_localisation(df, tables_dir)
    t_eq = table_estimator_equivalence(df, tables_dir)
    table_window_choice(df, tables_dir)
    t_tl, t_sig = prior_tests(df, tables_dir)

    fig_tolerance_curves(curve, fig_dir)
    fig_localisation_bias(df, fig_dir)

    print("Table 1 - mean F1 by estimator and window, at several tolerances\n")
    print(t_det.round(3).to_string())
    print("\n\nTable 2 - localisation error by estimator and ramp length\n")
    print(t_loc.round(1).to_string())
    print("\n\nTable 3 - do the extraction rules differ?\n")
    print(t_eq.round(2).to_string())

    if thr is not None:
        t_thr, best = table_threshold(thr, tables_dir)
        fig_threshold_diagnosis(thr, best, fig_dir)
        print("\n\nTable 5 - threshold diagnosis\n")
        print(t_thr.round(3).to_string())
        rho, p = spearmanr(best["q_star"], best["q"])
        print(f"\nSpearman(best q, q*) = {rho:.3f} (p = {p:.2e})")

    print("\n\nPrior test: F1(tau=25) vs transition length\n")
    print(t_tl.round(4).to_string())
    print("\n\nPrior test: F1(tau=25) vs sigma\n")
    print(t_sig.round(4).to_string())
    print(f"\nwrote tables to {tables_dir}\nwrote figures to {fig_dir}")


if __name__ == "__main__":
    main()
