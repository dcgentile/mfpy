"""
A/B test: resampling-calibrated threshold vs. the fixed self-quantile threshold.

Motivation
----------
Reviewer major comment 1(b): thresholding the test statistic at its own q-quantile is
"not appropriate" because multiple change points "upwardly bias the quantiles"; the
suggested fix is "a resampling method; see e.g. Matteson and James 2014".

That paper (Matteson & James, JASA 2014; methodology also described in the companion
ecp package paper, James & Matteson, JSS 2015) is E-Divisive: a bisection algorithm built
on the multivariate energy-distance statistic, unrelated in mechanics to our sliding
Wasserstein-derivative + quantile pipeline. But its threshold-calibration idea transfers
directly. E-Divisive never reads a quantile off the observed statistic. Instead, for each
candidate change point it permutes the observations *within the current segmentation*
(a null of complete exchangeability -- no change), recomputes the statistic on the
permutation, and repeats R times to build a null distribution. The candidate is accepted
only if its observed statistic exceeds enough of that null distribution (Algorithm 3 in
the JSS paper). Critically, this threshold is calibrated from data with no change
structure, so it does not care what fraction of the real series is "elevated" -- which is
exactly the failure mode ``toy_threshold_sweep.py`` diagnosed (FINDINGS.md, finding 4):
q=0.95 read off the observed statistic silently rises as more of the trajectory sits in
transition, so change points land further inside each ramp as tl grows.

Design
------
For each trajectory x, draw R permutations of the raw values (full exchangeability --
destroys temporal order, hence destroys any change structure, while preserving the
marginal distribution of the readings). Compute the same windowed statistic on each
permuted copy and take its max over time. Pooling R such maxima gives a null distribution
for "how large does this statistic get, anywhere in a same-length series, when there is no
change." The run-detection cutoff is the q-quantile of *that* null distribution, in place
of ``np.quantile(gammadot_real, q)``.

This is a familywise-max permutation threshold, the natural single-shot analogue of
E-Divisive's per-candidate permutation test adapted to a threshold-then-extract pipeline
rather than a bisection. One set of R permutations per trajectory is reused across every
q in QUANTILES (the null max distribution doesn't depend on q), so cost scales with
R per trajectory, not R x len(QUANTILES).

Everything downstream of the cutoff -- run detection, boundary extraction, scoring -- is
byte-identical to ``toy_threshold_sweep.py`` / ``boundary_estimators.py``. Same
trajectories, same w, same QUANTILES grid, same estimator; only the source of the cutoff
differs. Baseline and resampling rows share a key (sigma, tl, w, estimator, q) for direct
paired comparison.

Usage
-----
    python mfpy/experiments/resampling_threshold.py [--jobs N] [--out DIR] [--window 25]
                                                      [--R 199] [--limit N]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

sys.path.insert(0, HERE)
import boundary_estimators as be  # noqa: E402
from toy_sweep import (  # noqa: E402
    discover_trajectories,
    fast_metric_derivative_1d,
)
from toy_scoring import (  # noqa: E402
    load_ground_truth,
    localisation_stats,
    score_change_points,
    N_RAMPS,
    N_POINTS_METASTABLE,
)

QUANTILES = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.925, 0.95, 0.975, 0.99]
SCORE_TAUS = (5, 10, 25)


def predicted_optimal_q(tl):
    """1 - (fraction of the trajectory spent in transition); same as toy_threshold_sweep."""
    total = N_POINTS_METASTABLE + N_RAMPS * tl
    return 1.0 - (N_RAMPS * tl) / total


def find_runs_cutoff(statistic, cutoff):
    """Same run-detection logic as ``boundary_estimators.find_runs``, but taking an
    explicit cutoff rather than computing ``np.quantile(statistic, q)`` internally."""
    statistic = np.asarray(statistic).ravel()
    candidates = np.where(statistic > cutoff)[0]

    runs = []
    in_sequence = False
    seq_start = 0
    N = candidates.shape[0]

    for n in range(N - 1):
        curr = candidates[n]
        nxt = candidates[n + 1]

        if not in_sequence and nxt - curr == 1:
            in_sequence = True
            seq_start = curr

        if in_sequence and nxt - curr > 1:
            in_sequence = False
            runs.append((int(seq_start), int(curr)))

    return runs


def identify_change_points_cutoff(statistic, cutoff, estimator="gradient"):
    """Same as ``boundary_estimators.identify_change_points``, but the run-detection
    cutoff is supplied directly instead of derived from a quantile of ``statistic``."""
    statistic = np.asarray(statistic).ravel()
    gradient = np.gradient(statistic)

    changes = [0]
    for start, end in find_runs_cutoff(statistic, cutoff):
        changes.extend(be._extract(statistic, gradient, start, end, estimator))
    changes.append(statistic.shape[0] - 1)

    return np.sort(np.unique(changes))


def null_distributions(x, w, R, rng):
    """
    Build two null references from R permutations of x under full exchangeability:

    - ``maxes``: max_t statistic(permutation(x)) per permutation -- a family-wise
      (max-type) null, the natural single-shot analogue of a permutation significance
      test applied once over the whole series.
    - ``pooled``: every statistic value from every permutation, restricted to the valid
      (non-padded) region -- a pointwise null describing "how large does the statistic
      get, typewise, when there is no change", with no max-type multiple-testing
      correction baked in.
    """
    T = x.shape[0]
    maxes = np.empty(R)
    pooled = []
    for r in range(R):
        xp = rng.permutation(x)
        g_null = fast_metric_derivative_1d(xp, w)
        valid = g_null[w : T - w] if T > 2 * w else g_null
        maxes[r] = valid.max() if valid.size else 0.0
        pooled.append(valid)
    pooled = np.concatenate(pooled) if pooled else np.empty(0)
    return maxes, pooled


def run_one(job):
    sigma, tl, traj_path, gt_path, window, estimator, R, seed = job

    x = np.loadtxt(traj_path)
    ramps = load_ground_truth(gt_path, transition_length=tl)
    T = x.shape[0]

    gammadot = fast_metric_derivative_1d(x, window)

    rng = np.random.default_rng(seed)
    null_maxes, null_pooled = null_distributions(x, window, R, rng)

    rows = []
    for q in QUANTILES:
        for method in ("baseline", "resample_max", "resample_pointwise"):
            if method == "baseline":
                cps = be.identify_change_points(gammadot, q, estimator=estimator)
                cutoff = float(np.quantile(gammadot, q))
            elif method == "resample_max":
                cutoff = float(np.quantile(null_maxes, q))
                cps = identify_change_points_cutoff(gammadot, cutoff, estimator=estimator)
            else:
                cutoff = float(np.quantile(null_pooled, q))
                cps = identify_change_points_cutoff(gammadot, cutoff, estimator=estimator)

            loc = localisation_stats(cps, ramps, trajectory_length=T)
            row = {
                "sigma": sigma,
                "transition_length": tl,
                "w": window,
                "estimator": estimator,
                "method": method,
                "q": q,
                "q_star": predicted_optimal_q(tl),
                "cutoff": cutoff,
                "n_predicted_cps": loc["n"],
                "median_signed_error": loc["median_signed_error"],
                "median_abs_error": loc["median_abs_error"],
            }
            for tau in SCORE_TAUS:
                s = score_change_points(cps, ramps, tau=tau, trajectory_length=T)
                row[f"f1_tau{tau}"] = s["f1"]
                row[f"precision_tau{tau}"] = s["precision"]
                row[f"recall_tau{tau}"] = s["recall"]
            rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=max(1, os.cpu_count() or 2))
    ap.add_argument(
        "--out", default=os.path.join(REPO_ROOT, "results", "resampling_threshold")
    )
    ap.add_argument("--tmp", default="/tmp/resampling_threshold")
    ap.add_argument("--window", type=int, default=25)
    ap.add_argument("--estimator", default="gradient", choices=be.ESTIMATORS)
    ap.add_argument("--R", type=int, default=199, help="permutations per trajectory")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument(
        "--append", action="store_true", help="append to existing CSVs instead of overwriting"
    )
    args = ap.parse_args()

    os.makedirs(args.tmp, exist_ok=True)
    trajectories = discover_trajectories()
    if args.offset:
        trajectories = trajectories[args.offset :]
    if args.limit:
        trajectories = trajectories[: args.limit]

    jobs = [
        (sigma, tl, traj, gt, args.window, args.estimator, args.R, 1000 * sigma + tl)
        for (sigma, tl, traj, gt) in trajectories
    ]
    print(
        f"{len(jobs)} trajectories x {args.R} permutations x {len(QUANTILES)} quantiles "
        f"x 2 methods at w={args.window}",
        flush=True,
    )

    rows = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futures = [ex.submit(run_one, j) for j in jobs]
        for i, fut in enumerate(as_completed(futures), 1):
            rows.extend(fut.result())
            if i % 25 == 0:
                print(f"  {i}/{len(jobs)} ({time.time()-t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows).sort_values(["method", "sigma", "transition_length", "q"])

    os.makedirs(args.out, exist_ok=True)
    tmp_path = os.path.join(args.tmp, "resampling_threshold.csv")
    final_path = os.path.join(args.out, "resampling_threshold.csv")

    if args.append and os.path.exists(final_path):
        prior = pd.read_csv(final_path)
        df = pd.concat([prior, df], ignore_index=True)
        df = df.drop_duplicates(subset=["method", "sigma", "transition_length", "q"])
        df = df.sort_values(["method", "sigma", "transition_length", "q"])

    df.to_csv(tmp_path, index=False)
    df.to_csv(final_path, index=False)
    print(f"\nwrote {final_path} ({len(df)} rows) in {time.time()-t0:.0f}s")

    # empirically best q per (method, sigma, tl), by absolute localisation error
    scored = df.dropna(subset=["median_abs_error"])
    if len(scored):
        best = scored.loc[
            scored.groupby(["method", "sigma", "transition_length"])[
                "median_abs_error"
            ].idxmin()
        ]
        print("\nbest q vs q_star by method and transition length (median over sigma):")
        summary = (
            best.groupby(["method", "transition_length"])[
                ["q", "q_star", "median_abs_error"]
            ]
            .median()
            .round(3)
        )
        print(summary.to_string())

        for method in df["method"].unique():
            sub = best[best["method"] == method]
            if len(sub) > 1:
                corr = np.corrcoef(sub["q_star"], sub["q"])[0, 1]
                print(f"\ncorr(best q, q_star) for {method}: {corr:.3f} (n={len(sub)})")


if __name__ == "__main__":
    main()
