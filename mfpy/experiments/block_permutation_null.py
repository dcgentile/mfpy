"""
Block / circular permutation null for the resampling threshold.

Motivation
----------
``results/resampling_threshold/FINDINGS.md`` adopted ``resample_pointwise``: the run
cutoff is the ``q``-quantile of a permutation null built by fully shuffling the raw
values. ``results/vmeasure/FINDINGS.md`` section 6 then showed that this fails on the
paper's own Langevin trajectory -- segment count 157 -> 463, ADP 3 clusters -> 2,
V-measure 0.685 -> 0.621 -- and diagnosed the cause as the exchangeability assumption.
A full shuffle is a valid "no change" null only when the within-state process is iid,
which holds for the toy generator (independent Laplace draws inside each block) and
fails for a Langevin trajectory, whose within-well dynamics are strongly autocorrelated.
Destroying that autocorrelation makes the two half-window empirical measures look far
more alike than they really are, the null collapses, the cutoff lands too low, and the
trajectory is oversegmented.

The standard repair is a block bootstrap: resample contiguous blocks of length L rather
than individual points, so that dependence up to lag L survives into the null. This
module implements it and sweeps L. ``L = 1`` reduces exactly to the existing full
shuffle and is retained as a control -- if the L = 1 column does not reproduce the
published numbers, the harness is wrong, not the method.

Two block schemes are provided:

``circular``
    Blocks are drawn from the series wrapped end-to-end, so every observation appears
    in the same number of candidate blocks and the null series has the same length as
    the original with no edge deficit. This is the default.
``moving``
    Blocks are drawn from the un-wrapped series (Kuensch's moving-block bootstrap).
    Points near either end are under-represented, which for T >> L is a small effect;
    it is included so the choice of scheme can be shown not to drive the result.

Both draw ceil(T / L) blocks with replacement and truncate to length T, matching the
usual construction.

Choosing L
----------
``integrated_autocorr_time`` estimates tau_int from the trajectory's own
autocorrelation function using Sokal's automatic windowing (window at the first
M with M >= c * tau_int, c = 5). The principled default is L ~ tau_int; the sweep
brackets it so the sensitivity is visible rather than assumed.

What is measured
----------------
For the Langevin trajectory this script deliberately stops short of clustering. The
failure reported in ``vmeasure`` is a *segmentation* failure -- the cutoff lands too low
and the trajectory is cut into three times too many pieces -- and segment count is a
sufficient statistic for it, computable without ``dadapy``. Three quantities are
reported per L:

- ``cutoff`` -- the q-quantile of the pooled block-permutation null.
- ``cutoff_obs_quantile`` -- where that cutoff sits on the *observed* statistic's own
  quantile scale. This is the alignment diagnostic from
  ``results/qw_interaction/FINDINGS.md``: the fixed-quantile rule's own optimum is
  ~0.75-0.86, so a null whose cutoff translates to ~0.5 is by construction going to
  oversegment.
- ``n_segments`` -- segments produced downstream, against the fixed-quantile
  baseline's 157.

For the toy trajectories the full detection score is available, so F1 at tau = 10 is
reported directly and the L = 1 column is checked against the published result.

Usage
-----
    python mfpy/experiments/block_permutation_null.py --langevin
    python mfpy/experiments/block_permutation_null.py --toy [--jobs N] [--R 199]
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
)
from resampling_threshold import (  # noqa: E402
    identify_change_points_cutoff,
    null_distributions,
)

RESULT_DIR = os.path.join(REPO_ROOT, "results", "block_permutation_null")
LANGEVIN_PATH = os.path.join(REPO_ROOT, "data", "Langevin", "langevin.txt")

# Manuscript settings.
LANG_W, LANG_Q = 215, 0.86
TOY_W, TOY_Q_FIXED = 25, 0.95
TOY_Q_NULL = 0.5  # the setting resampling_threshold recommends for the pointwise null

SCORE_TAUS = (5, 10, 25)


# --------------------------------------------------------------------------------------
# block resampling
# --------------------------------------------------------------------------------------


def block_permutation(x, L, rng, scheme="circular"):
    """
    One block-bootstrap replicate of ``x`` with block length ``L``.

    ``L = 1`` is an ordinary iid resample of the values. Note this is a resample *with
    replacement*, whereas ``null_distributions`` uses a permutation (without
    replacement). For the pooled null these differ only at O(1/T); the L = 1 control
    below quantifies that rather than assuming it.
    """
    x = np.asarray(x).ravel()
    T = x.shape[0]
    L = int(max(1, min(L, T)))
    n_blocks = int(np.ceil(T / L))

    if scheme == "circular":
        starts = rng.integers(0, T, size=n_blocks)
        idx = (starts[:, None] + np.arange(L)[None, :]) % T
    elif scheme == "moving":
        starts = rng.integers(0, max(1, T - L + 1), size=n_blocks)
        idx = starts[:, None] + np.arange(L)[None, :]
    else:
        raise ValueError(f"unknown scheme: {scheme}")

    return x[idx.ravel()[:T]]


def block_null_pooled(x, w, R, rng, L, scheme="circular"):
    """
    Pointwise null pooled over R block-bootstrap replicates.

    Mirrors ``null_distributions``'s ``pooled`` branch exactly -- same statistic, same
    restriction to the valid (non-padded) region -- so the only difference from the
    published experiment is where the replicate comes from. The max-type null is not
    rebuilt here: ``results/resampling_threshold/FINDINGS.md`` established that it fails
    for a structural multiple-testing reason that block resampling does not touch.
    """
    x = np.asarray(x).ravel()
    T = x.shape[0]
    pooled = []
    for _ in range(R):
        xp = block_permutation(x, L, rng, scheme=scheme)
        g_null = fast_metric_derivative_1d(xp, w)
        valid = g_null[w : T - w] if T > 2 * w else g_null
        pooled.append(valid)
    return np.concatenate(pooled) if pooled else np.empty(0)


def integrated_autocorr_time(x, c=5.0, max_lag=None):
    """
    tau_int by Sokal's automatic windowing.

    tau_int = 1 + 2 * sum_{k=1}^{M} rho(k), with M the smallest window satisfying
    M >= c * tau_int(M). Returns (tau_int, M).
    """
    x = np.asarray(x, dtype=float).ravel()
    x = x - x.mean()
    T = x.shape[0]
    if max_lag is None:
        max_lag = min(T - 1, 10000)

    n = 1 << (2 * T - 1).bit_length()
    f = np.fft.rfft(x, n)
    acf = np.fft.irfft(f * np.conjugate(f), n)[:max_lag + 1].real
    if acf[0] <= 0:
        return 1.0, 0
    acf /= acf[0]

    taus = 1.0 + 2.0 * np.cumsum(acf[1:])
    windows = np.arange(1, taus.size + 1)
    ok = np.where(windows >= c * taus)[0]
    M = int(windows[ok[0]]) if ok.size else int(windows[-1])
    return float(1.0 + 2.0 * np.sum(acf[1:M + 1])), M


# --------------------------------------------------------------------------------------
# Langevin: segmentation diagnostic
# --------------------------------------------------------------------------------------


def n_segments_for_cutoff(gammadot, cutoff, T):
    changes = identify_change_points_cutoff(gammadot, cutoff, estimator="gradient")
    changes = np.unique(np.append(changes, T))
    return int(len(changes) - 1), changes


def run_langevin(block_lengths, R=199, w=LANG_W, q=LANG_Q, q_null=TOY_Q_NULL,
                 schemes=("circular", "moving"), seed=0):
    x = np.loadtxt(LANGEVIN_PATH).ravel()
    T = x.shape[0]
    gammadot = fast_metric_derivative_1d(x, w)
    valid = gammadot[w : T - w]

    tau_int, sokal_M = integrated_autocorr_time(x)
    print(f"Langevin: T={T}, w={w}, tau_int={tau_int:.1f} (Sokal window M={sokal_M})",
          flush=True)

    def obs_quantile(cut):
        """Where a cutoff sits on the observed statistic's own quantile scale."""
        return float((valid <= cut).mean())

    rows = []

    # incumbent: fixed quantile of the observed statistic
    cut_fixed = float(np.quantile(gammadot, q))
    n_fixed, _ = n_segments_for_cutoff(gammadot, cut_fixed, T)
    rows.append({
        "method": "fixed_quantile", "scheme": "-", "block_length": np.nan,
        "q": q, "cutoff": cut_fixed, "cutoff_obs_quantile": obs_quantile(cut_fixed),
        "n_segments": n_fixed, "tau_int": tau_int,
    })
    print(f"  fixed quantile q={q}: cutoff={cut_fixed:.4f} "
          f"(obs quantile {obs_quantile(cut_fixed):.3f}), {n_fixed} segments", flush=True)

    # published full-shuffle pointwise null, rebuilt with the original code path
    _, pooled = null_distributions(x, w, R, np.random.default_rng(seed))
    cut_shuffle = float(np.quantile(pooled, q_null))
    n_shuffle, _ = n_segments_for_cutoff(gammadot, cut_shuffle, T)
    rows.append({
        "method": "resample_pointwise", "scheme": "full_shuffle", "block_length": 1,
        "q": q_null, "cutoff": cut_shuffle, "cutoff_obs_quantile": obs_quantile(cut_shuffle),
        "n_segments": n_shuffle, "tau_int": tau_int,
    })
    print(f"  full shuffle q={q_null}: cutoff={cut_shuffle:.4f} "
          f"(obs quantile {obs_quantile(cut_shuffle):.3f}), {n_shuffle} segments",
          flush=True)

    for scheme in schemes:
        for L in block_lengths:
            t0 = time.time()
            rng = np.random.default_rng(seed)
            pooled_L = block_null_pooled(x, w, R, rng, L, scheme=scheme)
            cut = float(np.quantile(pooled_L, q_null))
            n_seg, _ = n_segments_for_cutoff(gammadot, cut, T)
            rows.append({
                "method": "block_pointwise", "scheme": scheme, "block_length": L,
                "q": q_null, "cutoff": cut, "cutoff_obs_quantile": obs_quantile(cut),
                "n_segments": n_seg, "tau_int": tau_int,
            })
            print(f"  {scheme} L={L:>5}: cutoff={cut:.4f} "
                  f"(obs quantile {obs_quantile(cut):.3f}), {n_seg} segments "
                  f"[{time.time()-t0:.0f}s]", flush=True)

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# toy: detection score
# --------------------------------------------------------------------------------------


def run_one_toy(job):
    sigma, tl, traj_path, gt_path, w, R, block_lengths, scheme, seed = job

    x = np.loadtxt(traj_path).ravel()
    ramps = load_ground_truth(gt_path, transition_length=tl)
    T = x.shape[0]
    gammadot = fast_metric_derivative_1d(x, w)

    def emit(method, L, q, cps, cutoff):
        loc = localisation_stats(cps, ramps, trajectory_length=T)
        row = {
            "sigma": sigma, "transition_length": tl, "w": w, "R": R,
            "method": method, "scheme": scheme if method == "block_pointwise" else "-",
            "block_length": L, "q": q, "cutoff": cutoff,
            "n_predicted_cps": loc["n"],
            "median_signed_error": loc["median_signed_error"],
            "median_abs_error": loc["median_abs_error"],
        }
        for tau in SCORE_TAUS:
            s = score_change_points(cps, ramps, tau=tau, trajectory_length=T)
            row[f"f1_tau{tau}"] = s["f1"]
            row[f"precision_tau{tau}"] = s["precision"]
            row[f"recall_tau{tau}"] = s["recall"]
        return row

    rows = []

    cps = be.identify_change_points(gammadot, TOY_Q_FIXED, estimator="gradient")
    rows.append(emit("fixed_quantile", np.nan, TOY_Q_FIXED, cps,
                     float(np.quantile(gammadot, TOY_Q_FIXED))))

    _, pooled = null_distributions(x, w, R, np.random.default_rng(seed))
    cut = float(np.quantile(pooled, TOY_Q_NULL))
    rows.append(emit("resample_pointwise", 1, TOY_Q_NULL,
                     identify_change_points_cutoff(gammadot, cut, estimator="gradient"),
                     cut))

    for L in block_lengths:
        rng = np.random.default_rng(seed)
        pooled_L = block_null_pooled(x, w, R, rng, L, scheme=scheme)
        cut = float(np.quantile(pooled_L, TOY_Q_NULL))
        rows.append(emit("block_pointwise", L, TOY_Q_NULL,
                         identify_change_points_cutoff(gammadot, cut, estimator="gradient"),
                         cut))

    return rows


def run_toy(block_lengths, R=199, w=TOY_W, scheme="circular", jobs=None, limit=None):
    trajectories = discover_trajectories()
    if limit:
        trajectories = trajectories[:limit]

    job_list = [
        (sigma, tl, traj, gt, w, R, block_lengths, scheme, 1000 * sigma + tl)
        for (sigma, tl, traj, gt) in trajectories
    ]
    print(f"toy: {len(job_list)} trajectories x {R} replicates x "
          f"{len(block_lengths) + 2} thresholds at w={w}", flush=True)

    rows = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=jobs or (os.cpu_count() or 2)) as ex:
        futures = [ex.submit(run_one_toy, j) for j in job_list]
        for i, fut in enumerate(as_completed(futures), 1):
            rows.extend(fut.result())
            if i % 10 == 0:
                print(f"  {i}/{len(job_list)} ({time.time()-t0:.0f}s)", flush=True)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langevin", action="store_true")
    ap.add_argument("--toy", action="store_true")
    ap.add_argument("--R", type=int, default=199)
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--scheme", default="circular", choices=["circular", "moving"])
    ap.add_argument("--out", default=RESULT_DIR)
    ap.add_argument("--tmp", default="/tmp/block_permutation_null")
    ap.add_argument("--blocks", type=int, nargs="+", default=None)
    args = ap.parse_args()

    if not (args.langevin or args.toy):
        args.langevin = args.toy = True

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.tmp, exist_ok=True)

    if args.langevin:
        blocks = args.blocks or [1, 5, 10, 25, 50, 100, 200, 400, 800, 1600]
        df = run_langevin(blocks, R=args.R)
        path = os.path.join(args.out, "langevin_block_null.csv")
        df.to_csv(path, index=False)
        print(f"wrote {path}")

    if args.toy:
        blocks = args.blocks or [1, 5, 10, 25, 50]
        df = run_toy(blocks, R=args.R, scheme=args.scheme,
                     jobs=args.jobs, limit=args.limit)
        path = os.path.join(args.out, "toy_block_null.csv")
        df.to_csv(path, index=False)
        print(f"wrote {path} ({len(df)} rows)")


if __name__ == "__main__":
    main()
