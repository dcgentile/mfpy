"""
V-measure evaluation of the segment clustering (action item 4 / reviewer 3(b)).

The reviewer asks whether the clustering step recovers the metastable states, not just
whether the change points are in the right place. V-measure (Rosenberg & Hirschberg 2007)
is the natural score here because it is invariant to label permutation and does not
require the predicted and true label sets to have the same cardinality -- which matters,
since ADP chooses its own number of clusters.

Two targets are reported for every trajectory:

``three_class``
    Ground truth is {state A, transition, state B}. This is the full target: the pipeline
    emits ramp segments as segments in their own right, so it *can* in principle recover a
    transition cluster, and the manuscript claims it does on both the toy and Langevin
    data.

``two_class``
    Ground truth is {state A, state B} with all transition points deleted from both the
    truth and the prediction before scoring. This isolates metastable-state recovery from
    ramp handling: a method that lumps ramps into the neighbouring states is not penalised.

Deleting points rather than relabelling them is deliberate. Relabelling ramp points to
their neighbouring state would invent a ground truth the generator does not assert, and
would make completeness depend on ramp length.

K-means on the raw values, at the number of clusters ADP found, is scored alongside as a
reference. The manuscript already contrasts the two qualitatively (figures
``toy-kmeans-clustering`` and ``langevin-kmeans-clustering``); this makes the contrast a
number.

Ground truth
------------
Toy: read directly from ``*_GroundTruth.txt`` via ``toy_scoring.load_ground_truth``. The
ramp intervals are exact, so the point labels are exact.

Langevin: the generator does not ship labels, so truth is defined geometrically from the
double-well potential (wells at -1 and +1, barrier at 0): state A is ``x < -c``, state B is
``x > +c``, transition is ``|x| <= c``, for a core-set half-width ``c``. This is a
*position-based* proxy, not a committor: a point sitting at ``x = 0.1`` on its way back
into the left well is called "transition" here even though a dynamical labelling would
assign it to A. Results are therefore reported across a sweep of ``c``, and the c-sweep is
the honest way to read them -- no single value is privileged.

Usage
-----
    python -m mfpy.experiments.vmeasure --toy          # 300 toy trajectories
    python -m mfpy.experiments.vmeasure --langevin     # Langevin trajectory, c sweep
    python -m mfpy.experiments.vmeasure --toy --limit 12   # quick smoke test
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message=".*intrinsic dimension is not defined.*")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from mfpy.experiments.toy_scoring import (  # noqa: E402
    N_RAMPS,
    load_ground_truth,
)
from mfpy.experiments.toy_sweep import (  # noqa: E402
    DATA_DIR,
    discover_trajectories,
    fast_metric_derivative_1d,
)

RESULT_DIR = os.path.join(REPO_ROOT, "results", "vmeasure")
LANGEVIN_PATH = os.path.join(REPO_ROOT, "data", "Langevin", "langevin.txt")

# Parameters as used in the manuscript figures.
TOY_W, TOY_Q = 25, 0.95
LANG_W, LANG_Q = 215, 0.86

# Core-set half-widths for the Langevin ground truth. c = 0 makes "transition" empty, so
# the three-class score degenerates to two-class; it is included as the limiting case.
LANG_CORES = [0.0, 0.2, 0.4, 0.6, 0.8]

STATE_A, TRANSITION, STATE_B = 0, 1, 2


# --------------------------------------------------------------------------------------
# ground truth
# --------------------------------------------------------------------------------------


def toy_point_labels(ramps, T):
    """
    Per-point ground-truth labels for a toy trajectory.

    The generator alternates: state A block, ramp A->B, state B block, ramp B->A, ...
    beginning in state A. ``ramps[i]`` is the half-open interval of the i-th ramp, so the
    stable block before ramp 0 is A, the block between ramps 0 and 1 is B, the block
    between ramps 1 and 2 is A, and so on. The trailing block after the last ramp is B,
    since there are 19 ramps (odd) and the sequence starts in A.
    """
    ramps = np.asarray(ramps, dtype=int)
    if len(ramps) != N_RAMPS:
        raise ValueError(f"expected {N_RAMPS} ramps, got {len(ramps)}")

    labels = np.empty(T, dtype=int)
    cursor = 0
    state = STATE_A
    for start, end in ramps:
        labels[cursor:start] = state
        labels[start:end] = TRANSITION
        cursor = end
        state = STATE_B if state == STATE_A else STATE_A
    labels[cursor:T] = state
    return labels


def langevin_point_labels(x, c):
    """Core-set ground truth for the double-well Langevin trajectory."""
    labels = np.full(x.shape[0], TRANSITION, dtype=int)
    labels[x < -c] = STATE_A
    labels[x > c] = STATE_B
    return labels


# --------------------------------------------------------------------------------------
# pipeline
# --------------------------------------------------------------------------------------


def _pairwise_segment_distances(X, changes):
    """
    W2 between every pair of segments, using 1-D order statistics.

    Equivalent to ``mfpy.distances.compute_pairwise_segment_distances`` but without the
    per-call ``ot.emd2_1d`` overhead. Segments have different lengths, so the order
    statistics are compared on a common quantile grid via linear interpolation of the
    inverse CDF -- which is what the exact 1-D optimal coupling amounts to.
    """
    changes = np.asarray(changes, dtype=int)
    n = len(changes) - 1
    grid = np.linspace(0.0, 1.0, 512, endpoint=False) + 0.5 / 512

    quantiles = np.empty((n, grid.size))
    for i in range(n):
        seg = np.sort(X[changes[i] : changes[i + 1]])
        if seg.size == 0:
            quantiles[i] = np.nan
            continue
        pos = grid * seg.size - 0.5
        quantiles[i] = np.interp(pos, np.arange(seg.size), seg)

    diff = quantiles[:, None, :] - quantiles[None, :, :]
    D = np.sqrt(np.nanmean(diff**2, axis=2))
    np.fill_diagonal(D, 0.0)
    return np.nan_to_num(D, nan=0.0)


MIN_DISTINCT_SEGMENTS = 5

# ADP signals degenerate input in three different ways: our own guard (too few distinct
# segments), an internal assertion when the log-density comes out nan, and a curve_fit
# failure inside the 2-NN dimension estimate. All three mean the same thing here -- the
# segment set is too degenerate to support a density-based clustering -- so they are
# caught together and recorded rather than silently dropped.
ADP_FAILURES = (ValueError, AssertionError, RuntimeError, np.linalg.LinAlgError)


def _adp_labels(D, Z=1.65):
    """
    ADP on a segment distance matrix, with exact duplicates collapsed first.

    At ``sigma = 0`` the toy trajectory is piecewise constant, so every state-A block is
    the *same* measure as every other state-A block and their W2 distance is exactly 0.
    ADP's 2-NN intrinsic-dimension estimator takes logs of nearest-neighbour distances and
    fails outright on these ties. Collapsing duplicate segments to one representative,
    clustering those, and broadcasting the labels back is exact -- identical measures must
    receive the same label under any distance-based clustering -- and leaves the noisy
    cells (sigma = 5, 20), where no ties occur, bit-identical to the naive path.

    Raises ``ValueError`` if too few distinct segments survive for ADP to be meaningful.
    """
    from dadapy.data import Data

    _, rep_index, inverse = np.unique(
        np.round(D, 12), axis=0, return_index=True, return_inverse=True
    )
    order = np.argsort(rep_index)
    rep_index = rep_index[order]
    remap = np.empty(order.size, dtype=int)
    remap[order] = np.arange(order.size)
    inverse = remap[inverse]

    if rep_index.size < MIN_DISTINCT_SEGMENTS:
        raise ValueError(
            f"only {rep_index.size} distinct segments; ADP is not defined here"
        )

    D_rep = D[np.ix_(rep_index, rep_index)]
    rep_labels = Data(distances=D_rep).compute_clustering_ADP(Z=Z)
    return np.asarray(rep_labels)[inverse], rep_index.size


def cluster_trajectory(X, w, q, Z=1.65, threshold="quantile", R=199, seed=0):
    """
    Run the manuscript pipeline and return (point_labels, change_points, n_distinct).

    Change points come from ``boundary_estimators.identify_change_points`` with the
    ``gradient`` rule, which is a validated exact reproduction of
    ``mfpy.distances.identify_change_points``. The trailing index is extended from
    ``T - 1`` to ``T`` so the segments tile the whole trajectory.

    ``threshold`` selects where the cutoff comes from:

    ``"quantile"``
        ``q``-quantile of the observed statistic. The manuscript's incumbent.
    ``"resample_pointwise"``
        ``q``-quantile of a permutation null pooled over all time points and ``R``
        permutations, per ``results/resampling_threshold/FINDINGS.md``. Use ``q = 0.5``:
        this is a quantile of the *null*, not of the observed statistic, so the two ``q``
        scales are not comparable.
    """
    from mfpy.experiments.boundary_estimators import identify_change_points
    from mfpy.experiments.resampling_threshold import (
        identify_change_points_cutoff,
        null_distributions,
    )

    X = np.asarray(X, dtype=float).ravel()
    gammadot = fast_metric_derivative_1d(X, w)

    if threshold == "quantile":
        changes = identify_change_points(gammadot, q, estimator="gradient")
    elif threshold == "resample_pointwise":
        _, pooled = null_distributions(X, w, R, np.random.default_rng(seed))
        cutoff = float(np.quantile(pooled, q))
        changes = identify_change_points_cutoff(gammadot, cutoff, estimator="gradient")
    else:
        raise ValueError(f"unknown threshold: {threshold}")

    changes = np.unique(np.append(changes, X.shape[0]))
    D = _pairwise_segment_distances(X, changes)
    segment_labels, n_distinct = _adp_labels(D, Z=Z)

    point_labels = np.empty(X.shape[0], dtype=int)
    for i in range(len(changes) - 1):
        point_labels[changes[i] : changes[i + 1]] = segment_labels[i]
    return point_labels, changes, n_distinct


def kmeans_labels(X, k, seed=0):
    from sklearn.cluster import KMeans

    X = np.asarray(X, dtype=float).reshape(-1, 1)
    k = max(1, min(int(k), X.shape[0]))
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X).labels_


# --------------------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------------------


def score(truth, pred):
    """Homogeneity, completeness, V-measure. Returns nan if truth has one class."""
    from sklearn.metrics import homogeneity_completeness_v_measure

    truth = np.asarray(truth)
    pred = np.asarray(pred)
    if truth.size == 0 or len(np.unique(truth)) < 2:
        return (np.nan, np.nan, np.nan)
    return homogeneity_completeness_v_measure(truth, pred)


def score_both_targets(truth3, pred, prefix):
    """Score a prediction against the three-class and two-class targets."""
    out = {}
    h, c, v = score(truth3, pred)
    out[f"{prefix}_h3"], out[f"{prefix}_c3"], out[f"{prefix}_v3"] = h, c, v

    keep = truth3 != TRANSITION
    h, c, v = score(truth3[keep], pred[keep])
    out[f"{prefix}_h2"], out[f"{prefix}_c2"], out[f"{prefix}_v2"] = h, c, v
    out[f"{prefix}_k"] = int(len(np.unique(pred)))
    return out


# --------------------------------------------------------------------------------------
# drivers
# --------------------------------------------------------------------------------------


def run_toy(
    limit=None,
    w=TOY_W,
    q=TOY_Q,
    log_path=None,
    threshold="quantile",
    R=199,
    start=0,
    stop=None,
):
    jobs = discover_trajectories(DATA_DIR)
    if limit:
        jobs = jobs[:: max(1, len(jobs) // limit)][:limit]
    jobs = jobs[start:stop]

    rows = []
    t0 = time.time()
    for n, (sigma, tl, traj_path, gt_path) in enumerate(jobs, 1):
        X = np.loadtxt(traj_path).ravel()
        ramps = load_ground_truth(gt_path, transition_length=tl)
        truth3 = toy_point_labels(ramps, X.shape[0])

        row = {
            "sigma": sigma,
            "transition_length": tl,
            "T": X.shape[0],
            "transition_fraction": float(np.mean(truth3 == TRANSITION)),
            "threshold": threshold,
            "q": q,
            "status": "ok",
        }
        try:
            pred, changes, n_distinct = cluster_trajectory(
                X, w, q, threshold=threshold, R=R
            )
        except ADP_FAILURES as exc:
            row["status"] = f"adp failed: {type(exc).__name__}: {exc}"
            rows.append(row)
            continue

        row["n_segments"] = len(changes) - 1
        row["n_distinct_segments"] = n_distinct
        row.update(score_both_targets(truth3, pred, "adp"))

        # Reference 1: k-means on raw values at the oracle number of classes (3).
        row.update(score_both_targets(truth3, kmeans_labels(X, 3), "kmeans"))

        # Reference 2: the same clustering step run on the *true* ramp boundaries.
        # The gap between this and `adp_*` is the part of the V-measure loss caused by
        # change-point localisation error rather than by the clustering itself.
        try:
            oracle_changes = np.unique(
                np.concatenate([[0], ramps.ravel(), [X.shape[0]]])
            )
            oracle_seg, _ = _adp_labels(_pairwise_segment_distances(X, oracle_changes))
            oracle_pred = np.empty(X.shape[0], dtype=int)
            for i in range(len(oracle_changes) - 1):
                oracle_pred[oracle_changes[i] : oracle_changes[i + 1]] = oracle_seg[i]
            row.update(score_both_targets(truth3, oracle_pred, "oracle"))
        except ADP_FAILURES:
            row["oracle_status"] = "failed"

        rows.append(row)

        if log_path and n % 10 == 0:
            msg = f"[{n}/{len(jobs)}] {time.time() - t0:.0f}s"
            with open(log_path, "a") as fh:
                fh.write(msg + "\n")
            print(msg, flush=True)

    return pd.DataFrame(rows)


def run_langevin(w=LANG_W, q=LANG_Q, cores=LANG_CORES, threshold="quantile", R=199):
    X = np.loadtxt(LANGEVIN_PATH).ravel()
    pred, changes, n_distinct = cluster_trajectory(X, w, q, threshold=threshold, R=R)
    km = kmeans_labels(X, 3)

    rows = []
    for c in cores:
        truth3 = langevin_point_labels(X, c)
        row = {
            "core_halfwidth": c,
            "threshold": threshold,
            "q": q,
            "T": X.shape[0],
            "n_segments": len(changes) - 1,
            "n_distinct_segments": n_distinct,
            "transition_fraction": float(np.mean(truth3 == TRANSITION)),
        }
        row.update(score_both_targets(truth3, pred, "adp"))
        row.update(score_both_targets(truth3, km, "kmeans"))
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--toy", action="store_true")
    ap.add_argument("--langevin", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=RESULT_DIR)
    ap.add_argument(
        "--threshold", default="quantile", choices=["quantile", "resample_pointwise"]
    )
    ap.add_argument(
        "--q",
        type=float,
        default=None,
        help="threshold level; defaults to the manuscript value for --threshold "
        "quantile and to 0.5 (the validated global optimum) for resample_pointwise",
    )
    ap.add_argument("--R", type=int, default=199, help="permutations for the null")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stop", type=int, default=None)
    ap.add_argument(
        "--suffix", default="", help="appended to output filenames"
    )
    args = ap.parse_args()

    if not (args.toy or args.langevin):
        args.toy = args.langevin = True

    os.makedirs(args.out, exist_ok=True)
    tmp = "/tmp/vmeasure"
    os.makedirs(tmp, exist_ok=True)

    def _q(default):
        if args.q is not None:
            return args.q
        return 0.5 if args.threshold == "resample_pointwise" else default

    def save(df, name):
        name = f"{name}{args.suffix}.csv"
        local = os.path.join(tmp, name)
        df.to_csv(local, index=False)
        shutil.copy(local, os.path.join(args.out, name))
        print(f"{name}: {len(df)} rows")

    if args.toy:
        save(
            run_toy(
                limit=args.limit,
                q=_q(TOY_Q),
                log_path=os.path.join(tmp, "progress.log"),
                threshold=args.threshold,
                R=args.R,
                start=args.start,
                stop=args.stop,
            ),
            "vmeasure_toy",
        )

    if args.langevin:
        save(
            run_langevin(q=_q(LANG_Q), threshold=args.threshold, R=args.R),
            "vmeasure_langevin",
        )


if __name__ == "__main__":
    main()
