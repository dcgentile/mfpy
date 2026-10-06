"""
Persist and plot the Langevin clustering under each threshold rule.

``block_null_vmeasure.py`` computes the clustering but writes out only its scores, so
there is no artefact showing *what* the method actually produced on the Langevin
trajectory -- which segments were cut, which cluster each got, and where the third
(transition) cluster does or does not appear. This script writes that out.

For each threshold rule it saves

- ``langevin_segments_<tag>.csv`` -- one row per segment: start, end, length, ADP cluster
  label, and the segment's mean and standard deviation, so a cluster can be identified
  with a well without reading the figure.
- ``langevin_labels_<tag>.npy`` -- the per-point cluster label array (length T).

and draws a single comparison figure: the trajectory coloured by cluster under each
rule, with the change points marked.

Tags: ``fixed`` (manuscript q = 0.86), ``shuffle`` (published full-shuffle null),
``blockL60`` etc. (block null at that block length).

Usage
-----
    python mfpy/experiments/langevin_clustering_output.py [--blocks 50 60 75] [--R 30]
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, HERE)

from vmeasure import (  # noqa: E402
    LANG_W,
    LANG_Q,
    LANGEVIN_PATH,
    langevin_point_labels,
    score,
)
from toy_sweep import fast_metric_derivative_1d  # noqa: E402
from resampling_threshold import null_distributions  # noqa: E402
from block_permutation_null import block_null_pooled  # noqa: E402
from block_null_vmeasure import cluster_from_cutoff  # noqa: E402

RESULT_DIR = os.path.join(REPO_ROOT, "results", "block_permutation_null")
SEG_DIR = os.path.join(RESULT_DIR, "clusterings")
FIG_DIR = os.path.join(RESULT_DIR, "figures")
Q_NULL = 0.5


def cutoff_for(X, gammadot, method, L, R, seed):
    if method == "fixed":
        return float(np.quantile(gammadot, LANG_Q))
    if method == "shuffle":
        _, pooled = null_distributions(X, LANG_W, R, np.random.default_rng(seed))
        return float(np.quantile(pooled, Q_NULL))
    pooled = block_null_pooled(
        X, LANG_W, R, np.random.default_rng(seed), L, scheme="circular"
    )
    return float(np.quantile(pooled, Q_NULL))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", type=int, nargs="+", default=[50, 60, 75])
    ap.add_argument("--R", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(SEG_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    X = np.loadtxt(LANGEVIN_PATH).ravel()
    T = X.shape[0]
    gammadot = fast_metric_derivative_1d(X, LANG_W)
    truth2 = langevin_point_labels(X, 0.0)

    settings = [("fixed", None, "fixed"), ("shuffle", 1, "shuffle")]
    settings += [("block", L, f"blockL{L}") for L in args.blocks]

    panels = []
    summary = []
    for method, L, tag in settings:
        cutoff = cutoff_for(X, gammadot, method, L, args.R, args.seed)
        pred, changes, _ = cluster_from_cutoff(X, gammadot, cutoff)

        seg_rows = []
        for i in range(len(changes) - 1):
            a, b = int(changes[i]), int(changes[i + 1])
            seg_rows.append({
                "segment": i,
                "start": a,
                "end": b,
                "length": b - a,
                "cluster": int(pred[a]),
                "mean": float(X[a:b].mean()),
                "std": float(X[a:b].std()),
            })
        seg = pd.DataFrame(seg_rows)
        seg.to_csv(os.path.join(SEG_DIR, f"langevin_segments_{tag}.csv"), index=False)
        np.save(os.path.join(SEG_DIR, f"langevin_labels_{tag}.npy"), pred)

        # Cluster composition, computed per *point* rather than per segment: the
        # segment-averaged mean is misleading here because segments differ in length by
        # an order of magnitude. ``frac_core`` is the fraction of a cluster's points with
        # |x| < 0.4, i.e. sitting between the wells -- this is what identifies the
        # transition cluster, and it separates cleanly (~0.1 for a well, ~0.65+ for the
        # transition state).
        comp_rows = []
        for c in np.unique(pred):
            m = pred == c
            comp_rows.append({
                "cluster": int(c),
                "n_segments": int((seg["cluster"] == c).sum()),
                "n_points": int(m.sum()),
                "mean_x": float(X[m].mean()),
                "frac_core": float((np.abs(X[m]) < 0.4).mean()),
            })
        comp = pd.DataFrame(comp_rows).sort_values("mean_x")
        v2 = score(truth2, pred)[2]
        summary.append({
            "tag": tag, "method": method, "block_length": L, "cutoff": cutoff,
            "n_segments": len(seg), "k": int(seg["cluster"].nunique()),
            "v2_c0": v2,
            "cluster_mean_x": ";".join(f"{c}:{m:.3f}" for c, m in
                                      zip(comp["cluster"], comp["mean_x"])),
            "cluster_n_points": ";".join(f"{c}:{n}" for c, n in
                                        zip(comp["cluster"], comp["n_points"])),
            "cluster_frac_core": ";".join(f"{c}:{f:.2f}" for c, f in
                                         zip(comp["cluster"], comp["frac_core"])),
            "has_transition_cluster": bool((comp["frac_core"] > 0.4).any()),
        })
        panels.append((tag, pred, changes, len(seg), v2))
        print(f"{tag:>10}: {len(seg)} segments, k={seg['cluster'].nunique()}, "
              f"V2={v2:.3f}", flush=True)
        print(comp.to_string(index=False), flush=True)

    pd.DataFrame(summary).to_csv(
        os.path.join(SEG_DIR, "langevin_cluster_composition.csv"), index=False
    )

    # ------------------------------------------------------------------ figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # A window long enough to contain several well-to-well transitions.
    lo, hi = 0, 20000
    colors = np.array(["tab:blue", "tab:orange", "tab:green", "tab:red",
                       "tab:purple", "tab:brown"])

    fig, axes = plt.subplots(len(panels), 1, figsize=(12, 2.1 * len(panels)),
                             sharex=True, sharey=True)
    for ax, (tag, pred, changes, n_seg, v2) in zip(np.atleast_1d(axes), panels):
        sl = slice(lo, hi)
        ax.scatter(np.arange(lo, hi), X[sl], c=colors[pred[sl] % len(colors)],
                   s=0.6, linewidths=0)
        for ch in changes[(changes >= lo) & (changes < hi)]:
            ax.axvline(ch, color="k", lw=0.3, alpha=0.35)
        ax.set_ylabel("x")
        ax.set_title(f"{tag}: {n_seg} segments, k={len(np.unique(pred))}, "
                     f"V2(c=0)={v2:.3f}", fontsize=9, loc="left")
    np.atleast_1d(axes)[-1].set_xlabel(f"time step (showing {lo}-{hi} of {T})")
    fig.suptitle(
        "Langevin clustering under each threshold rule"
        "—generated by `langevin_clustering_output.py`", fontsize=10
    )
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG_DIR, f"langevin_clustering.{ext}"), dpi=150)
    print(f"wrote {FIG_DIR}/langevin_clustering.{{pdf,png}}")
    print(f"wrote {SEG_DIR}/")


if __name__ == "__main__":
    main()
