"""
Langevin clustering under the block-permutation null.

``block_permutation_null.py`` establishes the segmentation half of the story: a block
null with L ~ the within-well autocorrelation time puts the cutoff back in the region
the fixed-quantile rule occupies, and the segment count returns to O(157). This script
closes the loop by running the rest of the manuscript pipeline -- pairwise Wasserstein
distances between segments, ADP clustering, V-measure against the core-set ground truth
-- so the comparison is against the numbers ``results/vmeasure/FINDINGS.md`` section 6
actually reports (V2 at c = 0: 0.685 fixed quantile, 0.621 full shuffle; ADP 3 clusters
-> 2).

Everything except the source of the cutoff is imported from ``vmeasure``, so this is the
same pipeline, not a reimplementation.

Usage
-----
    python mfpy/experiments/block_null_vmeasure.py [--blocks 25 50 100] [--R 30]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, HERE)

from vmeasure import (  # noqa: E402
    LANG_W,
    LANG_Q,
    LANG_CORES,
    LANGEVIN_PATH,
    _adp_labels,
    _pairwise_segment_distances,
    langevin_point_labels,
    score_both_targets,
)
from toy_sweep import fast_metric_derivative_1d  # noqa: E402
from resampling_threshold import (  # noqa: E402
    identify_change_points_cutoff,
    null_distributions,
)
from block_permutation_null import block_null_pooled  # noqa: E402
import boundary_estimators as be  # noqa: E402

RESULT_DIR = os.path.join(REPO_ROOT, "results", "block_permutation_null")
Q_NULL = 0.5


def cluster_from_cutoff(X, gammadot, cutoff, Z=1.65):
    changes = identify_change_points_cutoff(gammadot, cutoff, estimator="gradient")
    changes = np.unique(np.append(changes, X.shape[0]))
    D = _pairwise_segment_distances(X, changes)
    segment_labels, n_distinct = _adp_labels(D, Z=Z)
    point_labels = np.empty(X.shape[0], dtype=int)
    for i in range(len(changes) - 1):
        point_labels[changes[i] : changes[i + 1]] = segment_labels[i]
    return point_labels, changes, n_distinct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", type=int, nargs="+", default=[25, 50, 100, 200])
    ap.add_argument("--R", type=int, default=30)
    ap.add_argument("--scheme", default="circular")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=RESULT_DIR)
    args = ap.parse_args()

    X = np.loadtxt(LANGEVIN_PATH).ravel()
    T = X.shape[0]
    gammadot = fast_metric_derivative_1d(X, LANG_W)

    settings = [("fixed_quantile", None), ("full_shuffle", 1)]
    settings += [("block", L) for L in args.blocks]

    rows = []
    for method, L in settings:
        t0 = time.time()
        if method == "fixed_quantile":
            cutoff = float(np.quantile(gammadot, LANG_Q))
        elif method == "full_shuffle":
            _, pooled = null_distributions(
                X, LANG_W, args.R, np.random.default_rng(args.seed)
            )
            cutoff = float(np.quantile(pooled, Q_NULL))
        else:
            pooled = block_null_pooled(
                X, LANG_W, args.R, np.random.default_rng(args.seed), L,
                scheme=args.scheme,
            )
            cutoff = float(np.quantile(pooled, Q_NULL))

        pred, changes, n_distinct = cluster_from_cutoff(X, gammadot, cutoff)
        n_seg = int(len(changes) - 1)

        for c in LANG_CORES:
            truth3 = langevin_point_labels(X, c)
            row = {
                "method": method,
                "block_length": L,
                "R": args.R,
                "core": c,
                "cutoff": cutoff,
                "n_segments": n_seg,
                "n_distinct_segments": n_distinct,
            }
            row.update(score_both_targets(truth3, pred, "adp"))
            rows.append(row)

        base = [
            r for r in rows
            if r["method"] == method and r["block_length"] == L and r["core"] == 0.0
        ][0]
        print(f"{method:>15} L={L}: cutoff={cutoff:.4f} n_seg={n_seg} "
              f"k={base['adp_k']} V2(c=0)={base['adp_v2']:.3f} "
              f"[{time.time()-t0:.0f}s]", flush=True)

    df = pd.DataFrame(rows)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "langevin_block_vmeasure.csv")
    df.to_csv(path, index=False)
    print(f"wrote {path} ({len(df)} rows)")


if __name__ == "__main__":
    main()
