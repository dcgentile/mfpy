"""
The q-w interaction (open item from `toy_threshold_sweep.py` and
`results/resampling_threshold/FINDINGS.md`).

Why
---
Two threshold results in this repo were established at a single window size, `w = 25`:

- the fixed-quantile diagnosis (`q* = 1 - transition fraction`, `toy_threshold_sweep.py`);
- the recommendation to replace it with a pointwise permutation null at a single global
  `q = 0.5` (`results/resampling_threshold/FINDINGS.md`).

Neither says anything about whether the best `q` moves with `w`. That matters for the
manuscript because `w` and `q` are the method's only two tuning parameters and we advise
readers on both: if the recommended `q` shifts with `w`, the advice has to be stated jointly
rather than as two independent rules of thumb.

Design
------
Grid: `w in {10, 25, 50, 100}` x `q in QUANTILES` x two threshold rules
(`baseline` = quantile of the observed statistic, `resample_pointwise` = quantile of a
permutation null pooled over time and `R` permutations). Segmentation, boundary extraction
(`gradient`) and scoring are shared with the existing experiments so the numbers are
directly comparable.

The `oracle` window (`w = 2 * tl`) from `toy_sweep.py` is deliberately excluded: it was
already shown to lose to a flat `w = 25` at every noise level, and including it would
conflate "does q depend on w" with "should w depend on tl".

Cost
----
The baseline arm is nearly free -- one statistic per (trajectory, w), then every `q` is a
requantisation of it. The resampling arm recomputes `R` permutations per (trajectory, w)
and dominates the runtime, which is why `--stratified` exists.

Usage
-----
    python -m mfpy.experiments.qw_interaction --stratified 60 --start 0 --stop 15
    python -m mfpy.experiments.qw_interaction --baseline-only          # all 300, fast
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from mfpy.experiments import boundary_estimators as be  # noqa: E402
from mfpy.experiments.resampling_threshold import (  # noqa: E402
    QUANTILES,
    identify_change_points_cutoff,
    null_distributions,
    predicted_optimal_q,
)
from mfpy.experiments.toy_scoring import (  # noqa: E402
    load_ground_truth,
    localisation_stats,
    score_change_points,
)
from mfpy.experiments.toy_sweep import (  # noqa: E402
    DATA_DIR,
    discover_trajectories,
    fast_metric_derivative_1d,
)

RESULT_DIR = os.path.join(REPO_ROOT, "results", "qw_interaction")

WINDOWS = [10, 25, 50, 100]
SCORE_TAUS = (5, 10, 25)
METHODS = ("baseline", "resample_pointwise")


def stratify(jobs, n):
    """
    Evenly spaced transition lengths within each noise level.

    Trajectories are one per (noise, tl) cell on a 3 x 100 grid, so taking every k-th tl
    inside each noise level gives balanced coverage of both axes.
    """
    by_noise = {}
    for job in jobs:
        by_noise.setdefault(job[0], []).append(job)
    per = max(1, n // max(1, len(by_noise)))
    out = []
    for noise in sorted(by_noise):
        group = sorted(by_noise[noise], key=lambda j: j[1])
        idx = np.unique(np.linspace(0, len(group) - 1, per).round().astype(int))
        out.extend(group[i] for i in idx)
    return out


def run_one(sigma, tl, traj_path, gt_path, windows, R, seed, methods):
    x = np.loadtxt(traj_path).ravel()
    ramps = load_ground_truth(gt_path, transition_length=tl)
    T = x.shape[0]

    rows = []
    for w in windows:
        if T <= 2 * w:
            continue
        gammadot = fast_metric_derivative_1d(x, w)

        pooled = None
        if "resample_pointwise" in methods:
            _, pooled = null_distributions(
                x, w, R, np.random.default_rng(seed + w)
            )

        for method in methods:
            for q in QUANTILES:
                if method == "baseline":
                    cps = be.identify_change_points(gammadot, q, estimator="gradient")
                    cutoff = float(np.quantile(gammadot, q))
                else:
                    cutoff = float(np.quantile(pooled, q))
                    cps = identify_change_points_cutoff(
                        gammadot, cutoff, estimator="gradient"
                    )

                loc = localisation_stats(cps, ramps, trajectory_length=T)
                row = {
                    "sigma": sigma,
                    "transition_length": tl,
                    "w": w,
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
    ap.add_argument("--stratified", type=int, default=None)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stop", type=int, default=None)
    ap.add_argument("--R", type=int, default=199)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--baseline-only", action="store_true")
    ap.add_argument("--windows", nargs="+", type=int, default=WINDOWS)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--out", default=RESULT_DIR)
    args = ap.parse_args()

    methods = ("baseline",) if args.baseline_only else METHODS

    jobs = discover_trajectories(DATA_DIR)
    if args.stratified:
        jobs = stratify(jobs, args.stratified)
    jobs = jobs[args.start : args.stop]

    os.makedirs(args.out, exist_ok=True)
    tmp = "/tmp/qw"
    os.makedirs(tmp, exist_ok=True)

    rows, t0 = [], time.time()
    for n, (sigma, tl, traj_path, gt_path) in enumerate(jobs, 1):
        rows.extend(
            run_one(sigma, tl, traj_path, gt_path, args.windows, args.R, args.seed, methods)
        )
        if n % 5 == 0:
            print(f"[{n}/{len(jobs)}] {time.time() - t0:.0f}s", flush=True)

    df = pd.DataFrame(rows)
    name = f"qw_interaction{args.suffix}.csv"
    local = os.path.join(tmp, name)
    df.to_csv(local, index=False)
    shutil.copy(local, os.path.join(args.out, name))
    print(f"{name}: {len(df)} rows, {len(jobs)} trajectories")


if __name__ == "__main__":
    main()
