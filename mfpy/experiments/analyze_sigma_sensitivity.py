"""
Tables and figure for the sigma-sensitivity experiment.

    python -m mfpy.experiments.analyze_sigma_sensitivity
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

RESULT_DIR = os.path.join(REPO_ROOT, "results", "sigma_sensitivity")
TABLE_DIR = os.path.join(RESULT_DIR, "tables")
FIG_DIR = os.path.join(RESULT_DIR, "figures")

CODE_SIGMA = 0.5  # what mfpy/clustering.py computes
PAPER_SIGMA = 1.0  # what the manuscript says


def _write(df, name):
    os.makedirs(TABLE_DIR, exist_ok=True)
    df.to_csv(os.path.join(TABLE_DIR, f"{name}.csv"), float_format="%.4g")
    with open(os.path.join(TABLE_DIR, f"{name}.md"), "w") as fh:
        fh.write(df.to_markdown(floatfmt=".3f") + "\n")
    print(f"wrote {name}")


def tables(toy, toy_med, lang, lang_med):
    # Table 1: the two fixed values against the scale-free heuristic, on the toy set.
    rows = []
    for noise, g in toy.groupby("noise_sigma"):
        best = g.loc[g.groupby("transition_length").v3.idxmax()]
        med = toy_med[toy_med.noise_sigma == noise]
        for label, d in [
            (f"fixed sigma = {CODE_SIGMA} (code)", g[g.sigma == CODE_SIGMA]),
            (f"fixed sigma = {PAPER_SIGMA} (manuscript)", g[g.sigma == PAPER_SIGMA]),
            ("median heuristic", med),
            ("best sigma per trajectory (ceiling)", best),
        ]:
            rows.append(
                {
                    "noise_sigma": noise,
                    "rule": label,
                    "mean_sigma": d.sigma.mean(),
                    "v3": d.v3.mean(),
                    "v2": d.v2.mean(),
                    "fidelity_norm": d.fidelity_norm.mean(),
                }
            )
    _write(
        pd.DataFrame(rows).set_index(["noise_sigma", "rule"]).round(3),
        "table1_toy_sigma_rules",
    )

    # Table 2: Langevin, same comparison.
    rows = []
    for label, d in [
        (f"fixed sigma = {CODE_SIGMA} (code)", lang[lang.sigma == CODE_SIGMA]),
        (f"fixed sigma = {PAPER_SIGMA} (manuscript)", lang[lang.sigma == PAPER_SIGMA]),
        ("median heuristic", lang_med),
        ("best sigma on the grid (ceiling)", lang.loc[[lang.v3.idxmax()]]),
    ]:
        rows.append(
            {
                "rule": label,
                "sigma": d.sigma.mean(),
                "eigengap_k": d.eigengap_k.mean(),
                "v3": d.v3.mean(),
                "v2": d.v2.mean(),
                "fidelity_norm": d.fidelity_norm.mean(),
            }
        )
    _write(pd.DataFrame(rows).set_index("rule").round(3), "table2_langevin_sigma_rules")

    # Table 3: eigengap-implied k across sigma (toy, noisy cells only) -- does sigma
    # change how many states a practitioner would infer?
    noisy = toy[toy.noise_sigma > 0]
    _write(
        noisy.groupby("sigma")
        .agg(
            eigengap_k_median=("eigengap_k", "median"),
            eigengap_k_mean=("eigengap_k", "mean"),
            frac_k_eq_3=("eigengap_k", lambda s: (s == 3).mean()),
            v3=("v3", "mean"),
            ari_vs_code_default=("ari_vs_code_default", "mean"),
        )
        .round(3),
        "table3_toy_eigengap_by_sigma",
    )

    _write(
        lang.set_index("sigma")[
            ["eigengap_k", "ari_vs_code_default", "v3", "v2", "fidelity_norm"]
        ].round(3),
        "table4_langevin_sigma_sweep",
    )


def figure(toy, toy_med, lang, lang_med):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(FIG_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    for noise, g in toy[toy.noise_sigma > 0].groupby("noise_sigma"):
        m = g.groupby("sigma").v3.mean()
        ax.semilogx(m.index, m.values, marker=".", label=rf"toy, noise $\sigma={noise}$")
    lm = lang.groupby("sigma").v3.mean()
    ax.semilogx(lm.index, lm.values, marker=".", label="Langevin")
    for x, lab, style in [
        (CODE_SIGMA, "code ($\\sigma=1/2$)", "-"),
        (PAPER_SIGMA, "manuscript ($\\sigma=1$)", ":"),
    ]:
        ax.axvline(x, color="k", lw=0.8, ls=style)
        ax.annotate(lab, (x, 0.02), rotation=90, fontsize=7, ha="right")
    ax.set_xlabel(r"similarity bandwidth $\sigma$")
    ax.set_ylabel("V-measure (3-class)")
    ax.set_title("Quality vs $\\sigma$ (spectral, $K=3$)", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1]
    labels, vals = [], []
    for noise in sorted(toy.noise_sigma.unique()):
        g = toy[toy.noise_sigma == noise]
        med = toy_med[toy_med.noise_sigma == noise]
        labels.append(f"toy\nnoise {int(noise)}")
        vals.append(
            (
                g[g.sigma == CODE_SIGMA].v3.mean(),
                g[g.sigma == PAPER_SIGMA].v3.mean(),
                med.v3.mean(),
            )
        )
    labels.append("Langevin")
    vals.append(
        (
            lang[lang.sigma == CODE_SIGMA].v3.mean(),
            lang[lang.sigma == PAPER_SIGMA].v3.mean(),
            lang_med.v3.mean(),
        )
    )
    vals = np.array(vals)
    x = np.arange(len(labels))
    for i, name in enumerate(
        [r"$\sigma=1/2$ (code)", r"$\sigma=1$ (manuscript)", "median heuristic"]
    ):
        ax.bar(x + (i - 1) * 0.27, vals[:, i], width=0.26, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("V-measure (3-class)")
    ax.set_title("A scale-free $\\sigma$ removes the sensitivity", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG_DIR, f"sigma_sensitivity.{ext}"), dpi=150)
    plt.close(fig)
    print("wrote sigma_sensitivity")


def main():
    toy = pd.read_csv(os.path.join(RESULT_DIR, "sigma_toy.csv"))
    toy_med = pd.read_csv(os.path.join(RESULT_DIR, "sigma_toy_median_heuristic.csv"))
    lang = pd.read_csv(os.path.join(RESULT_DIR, "sigma_langevin.csv"))
    lang_med = pd.read_csv(
        os.path.join(RESULT_DIR, "sigma_langevin_median_heuristic.csv")
    )
    tables(toy, toy_med, lang, lang_med)
    figure(toy, toy_med, lang, lang_med)


if __name__ == "__main__":
    main()
