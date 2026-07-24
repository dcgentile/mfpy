"""
Diagnose the quantile threshold in ``identify_change_points``.

Why
---
Scoring against true ramp boundaries (rather than mere containment in a ramp) reveals
that the pipeline places its change points well *inside* each ramp, by an amount that
grows roughly linearly with the ramp length ``tl`` and is essentially identical under
all three boundary-extraction rules in ``boundary_estimators``. Because changing the
extraction rule does not move the bias, the cause must lie upstream, in stage 1: the
supra-threshold run is itself a shrunken image of the ramp.

That points directly at reviewer comment 1(b) -- thresholding at a fixed quantile of the
test statistic is inappropriate when the change structure is what inflates the quantile.
The longer the ramps, the larger the fraction of the trajectory spent in transition, and
the further ``q = 0.95`` cuts into the elevated region.

Prediction
----------
If that account is right, the appropriate threshold is not a constant but

    q* ~ 1 - (fraction of the trajectory spent in transition)
       = 1 - 19*tl / (5000 + 19*tl)

and setting ``q`` there should remove the localisation bias. This script tests that by
sweeping ``q`` across every trajectory at fixed ``w`` and comparing the empirically
best ``q`` against ``q*``.

Usage
-----
    python mfpy/experiments/toy_threshold_sweep.py [--jobs N] [--out DIR] [--window 25]
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
    """1 - (fraction of the trajectory spent in transition)."""
    total = N_POINTS_METASTABLE + N_RAMPS * tl
    return 1.0 - (N_RAMPS * tl) / total


def run_one(job):
    sigma, tl, traj_path, gt_path, window, estimator = job

    x = np.loadtxt(traj_path)
    ramps = load_ground_truth(gt_path, transition_length=tl)
    T = x.shape[0]

    gammadot = fast_metric_derivative_1d(x, window)

    rows = []
    for q in QUANTILES:
        cps = be.identify_change_points(gammadot, q, estimator=estimator)
        loc = localisation_stats(cps, ramps, trajectory_length=T)
        row = {
            "sigma": sigma,
            "transition_length": tl,
            "w": window,
            "estimator": estimator,
            "q": q,
            "q_star": predicted_optimal_q(tl),
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
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "results", "toy_sweep"))
    ap.add_argument("--window", type=int, default=25)
    ap.add_argument("--estimator", default="gradient", choices=be.ESTIMATORS)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    trajectories = discover_trajectories()
    if args.limit:
        trajectories = trajectories[: args.limit]

    jobs = [
        (sigma, tl, traj, gt, args.window, args.estimator)
        for (sigma, tl, traj, gt) in trajectories
    ]
    print(
        f"{len(jobs)} trajectories x {len(QUANTILES)} quantiles "
        f"= {len(jobs)*len(QUANTILES)} fits at w={args.window}",
        flush=True,
    )

    rows = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futures = [ex.submit(run_one, j) for j in jobs]
        for i, fut in enumerate(as_completed(futures), 1):
            rows.extend(fut.result())
            if i % 50 == 0:
                print(f"  {i}/{len(jobs)} ({time.time()-t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows).sort_values(["sigma", "transition_length", "q"])
    path = os.path.join(args.out, "toy_threshold_sweep.csv")
    df.to_csv(path, index=False)
    print(f"\nwrote {path} ({len(df)} rows) in {time.time()-t0:.0f}s")

    # empirically best q per trajectory, by absolute localisation error
    best = df.loc[df.groupby(["sigma", "transition_length"])["median_abs_error"].idxmin()]
    corr = np.corrcoef(best["q_star"], best["q"])[0, 1]
    print(f"\ncorrelation between empirically best q and q* = 1 - ramp fraction: {corr:.3f}")
    print("\nbest q vs q* by transition length (median over sigma):")
    summary = (
        best.groupby("transition_length")[["q", "q_star", "median_abs_error"]]
        .median()
        .loc[[1, 10, 20, 40, 60, 80, 100]]
        .round(3)
    )
    print(summary.to_string())


if __name__ == "__main__":
    main()
