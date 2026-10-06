"""
Automatic block-length selection for the block-permutation null.

The gap left open by `results/block_permutation_null/FINDINGS.md`: the block length `L`
should be the *within-state* integrated autocorrelation time, but estimating that needs a
state assignment, which is roughly what the method is trying to produce. In that
experiment the state assignment was supplied by hand -- a sign-and-threshold rule on the
Langevin, ground truth on the toy -- so `L` was principled but not automatic.

This module closes the loop with a two-pass estimator:

    pass 1   segment the series with the incumbent fixed-quantile rule at (w, q)
    pass 2   estimate tau_int inside the segments pass 1 returned, take the median,
             set L, and re-threshold with the block null at that L

The pilot segmentation does not need to be good. It needs only to cut the series
often enough that most segments lie inside a single state, which the fixed-quantile rule
does by construction: its documented failure mode (`results/toy_sweep/FINDINGS.md`
finding 4) is that change points land *inside* transitions, i.e. it over-cuts the
transitions, not that it merges states. Short segments are discarded before estimating
tau_int, since a segment shorter than a few correlation times carries no usable ACF.

Why not Politis & White (2004)
------------------------------
Their automatic block-length selector is the standard answer to "how long should the
blocks be", and it is fully data-driven. It is not directly usable here for two reasons.
It targets the block length that minimises MSE of a *variance estimator for the sample
mean under stationarity*, whereas what this pipeline needs is the shape of the null
distribution of a windowed two-sample statistic. And it is a global estimator: applied to
a series that contains change points it inherits exactly the contamination documented in
`FINDINGS.md` §2, where the whole-trajectory tau_int of the Langevin is 1676 -- a
metastability time, not a noise time -- against a within-well value of 52-75. Any global
stationary-series selector will return the former. Segmenting first is the adaptation
that makes the estimate mean what we need it to mean. Their selector is reported
alongside ours as a diagnostic, not as the operating point.

Usage
-----
    python mfpy/experiments/auto_block_length.py --dataset langevin
    python mfpy/experiments/auto_block_length.py --dataset prinz --plot
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

import boundary_estimators as be  # noqa: E402
from toy_sweep import fast_metric_derivative_1d  # noqa: E402
from block_permutation_null import (  # noqa: E402
    block_null_pooled,
    integrated_autocorr_time,
)
from resampling_threshold import identify_change_points_cutoff  # noqa: E402

# Minimum segment length, in multiples of the pilot tau_int, for a segment to be used in
# the tau_int estimate. A segment shorter than a few correlation times has an ACF that is
# all noise; Sokal's window would truncate at lag 1 and return ~1 regardless of the truth.
MIN_SEGMENT_MULTIPLE = 8
ABS_MIN_SEGMENT = 50

# Each pilot segment is trimmed by this many window widths at both ends before its
# autocorrelation is measured. This is not a tuning knob: the pipeline's documented
# boundary bias (`results/toy_sweep/FINDINGS.md` finding 2) is that detected change
# points land *inside* transitions, so both ends of every pilot segment carry transition
# material, and a ramp is smooth and therefore strongly autocorrelated. Left untrimmed
# the estimate is contaminated upward by exactly the structure it is supposed to exclude
# -- on the toy data, whose within-state process is iid, the untrimmed estimator returns
# tau_int up to 10 where the truth is 1. Trimming one window width is the natural choice
# because w is the scale the statistic is computed on and therefore the resolution limit
# of a detected boundary.
TRIM_WINDOWS = 1


def pilot_change_points(x, w, q):
    """Incumbent fixed-quantile segmentation. Deliberately the manuscript's own rule."""
    gammadot = fast_metric_derivative_1d(x, w)
    changes = be.identify_change_points(gammadot, q, estimator="gradient")
    return np.unique(np.append(changes, x.shape[0])), gammadot


def within_segment_tau(x, changes, min_len, trim):
    """Median tau_int over segment interiors at least ``min_len`` long after trimming."""
    taus, lens = [], []
    for a, b in zip(changes[:-1], changes[1:]):
        a, b = int(a) + trim, int(b) - trim
        if b - a >= min_len:
            taus.append(integrated_autocorr_time(x[a:b])[0])
            lens.append(b - a)
    return np.asarray(taus), np.asarray(lens)


def estimate_block_length(x, w, q, verbose=True):
    """
    Two-pass automatic block length.

    Returns ``(L, info)``. ``info`` carries the pilot segment count, the number of
    segments actually used, the median/IQR of the within-segment tau_int, and the
    contaminated whole-series tau_int for comparison.
    """
    x = np.asarray(x, dtype=float).ravel()
    changes, gammadot = pilot_change_points(x, w, q)
    trim = int(TRIM_WINDOWS * w)

    # First cut with a generous absolute floor, then refine the floor using the estimate
    # itself so it scales with the process rather than being hard-coded.
    taus, lens = within_segment_tau(x, changes, ABS_MIN_SEGMENT, trim)
    tau0 = float(np.median(taus)) if taus.size else 1.0
    min_len = max(ABS_MIN_SEGMENT, int(MIN_SEGMENT_MULTIPLE * tau0))
    taus, lens = within_segment_tau(x, changes, min_len, trim)

    if taus.size == 0:  # every segment too short: the process has no usable dependence
        tau_hat = 1.0
    else:
        tau_hat = float(np.median(taus))

    L = int(max(1, round(tau_hat)))
    tau_global = integrated_autocorr_time(x)[0]

    info = {
        "pilot_segments": int(len(changes) - 1),
        "segments_used": int(taus.size),
        "trim": trim,
        "min_segment_len": int(min_len),
        "median_segment_len": float(np.median(lens)) if lens.size else np.nan,
        "tau_within_median": tau_hat,
        "tau_within_q25": float(np.percentile(taus, 25)) if taus.size else np.nan,
        "tau_within_q75": float(np.percentile(taus, 75)) if taus.size else np.nan,
        "tau_global": float(tau_global),
        "L": L,
    }
    if verbose:
        print(f"  pilot: {info['pilot_segments']} segments, "
              f"{info['segments_used']} used (>= {min_len} steps)")
        print(f"  tau_int within segments: median {tau_hat:.1f} "
              f"[{info['tau_within_q25']:.1f}, {info['tau_within_q75']:.1f}]  "
              f"| whole series: {tau_global:.1f}")
        print(f"  --> L = {L}")
    return L, info, changes, gammadot


def auto_threshold_change_points(x, w, q, q_null=0.5, R=30, seed=0, L=None):
    """
    Full automatic pipeline: estimate L, build the block null, return change points.

    Pass ``L`` to override the estimate (used to compare against an oracle block length).
    """
    x = np.asarray(x, dtype=float).ravel()
    if L is None:
        L, info, _, gammadot = estimate_block_length(x, w, q)
    else:
        info = {"L": L, "tau_within_median": np.nan}
        gammadot = fast_metric_derivative_1d(x, w)

    pooled = block_null_pooled(x, w, R, np.random.default_rng(seed), L,
                               scheme="circular")
    cutoff = float(np.quantile(pooled, q_null))
    changes = identify_change_points_cutoff(gammadot, cutoff, estimator="gradient")
    changes = np.unique(np.append(changes, x.shape[0]))
    return changes, cutoff, L, info, gammadot


DATASETS = {
    # name: (path, w, q) -- w and q are the manuscript settings for each dataset
    "langevin": (os.path.join(REPO_ROOT, "data", "Langevin", "langevin.txt"), 215, 0.86),
    "prinz": (os.path.join(REPO_ROOT, "data", "prinz", "prinz.txt"), 16, 0.5),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="langevin", choices=sorted(DATASETS))
    ap.add_argument("--R", type=int, default=30)
    args = ap.parse_args()

    path, w, q = DATASETS[args.dataset]
    x = np.loadtxt(path).ravel()
    print(f"{args.dataset}: T={x.shape[0]}, w={w}, q={q}")
    changes, cutoff, L, info, g = auto_threshold_change_points(x, w, q, R=args.R)
    valid = g[w : x.shape[0] - w]
    print(f"  auto-L cutoff={cutoff:.4f} "
          f"(obs quantile {(valid <= cutoff).mean():.3f}), "
          f"{len(changes) - 1} segments")
    print(pd.Series(info).to_string())


if __name__ == "__main__":
    main()
