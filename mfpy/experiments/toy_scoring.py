"""
Ground-truth parsing and change-point scoring for the toy model trajectories.

Data provenance
---------------
The toy trajectories are produced by ``data/Toy_Model_Trajectories/Trajectory_Generator.m``.
For each (sigma, tl) pair the generator builds a trajectory alternating between two
Laplace metastable states -- state A with mean ``baseint = 100`` and state B with mean
``baseint * intratio = 200``, both with scale ``sigma`` -- joined by *gradual* linear
ramps of length ``tl``.

The generator records boundaries in a vector ``changes`` of length 40, where for
segment ``i = 1..10`` (1-indexed, MATLAB):

    changes[4i-3] = end of the state-A block      (ramp A->B starts here)
    changes[4i-2] = end of the ramp A->B          (state-B block starts here)
    changes[4i-1] = end of the state-B block      (ramp B->A starts here)
    changes[4i]   = end of the ramp B->A          (next state-A block starts here)

It then writes ``[0; changes(1:39)]`` to ``*_GroundTruth.txt``. So the on-disk file is
40 numbers: a leading 0, followed by the first 39 entries of ``changes``. In 0-indexed
Python terms, ``gt[k] == changes[k]`` for ``k >= 1``.

The ramps therefore span the half-open intervals

    (gt[1], gt[2]), (gt[3], gt[4]), ..., (gt[37], gt[38])

There are 19 of them: the tenth state-B block is not followed by a ramp (the generator
guards that with ``if i < Nsegmentseach``), which is why a trajectory has length
``5000 + 19 * tl``. That identity is asserted in ``load_ground_truth``.

What counts as a change point
-----------------------------
A change point is an *instant at which the law of the process changes*: where stability
turns into transition, or transition turns into stability. Each ramp therefore
contributes **two** change points -- its start and its end -- giving

    38 true change points per trajectory

These are exactly the endpoints of the 19 ramp intervals, i.e. ``gt[1:39]``.

This is deliberately stricter than asking whether a prediction lands somewhere inside a
ramp. A prediction sitting in the middle of a long ramp has not identified either
boundary, and under a containment criterion it would still score as correct; here it
does not. For long ramps the distinction is large.

Scoring
-------
Predictions are matched to true change points by **one-to-one optimal assignment**
(Hungarian algorithm) restricted to pairs within a tolerance ``tau``:

    precision = (# matched predictions) / (# predictions)
    recall    = (# matched true change points) / 38
    F1        = harmonic mean

One-to-one matching means each true change point can be claimed by at most one
prediction, and each prediction can satisfy at most one true change point. Two
predictions clustered near the same boundary yield one true positive and one false
positive, which is the standard convention in the CPD benchmarking literature and
avoids rewarding a method for emitting redundant detections.

Because the notion of "close enough" is exactly what ``tau`` encodes, and because the
estimator studied here has a window-dependent localisation bias, results are reported
as a function of ``tau`` rather than at a single privileged value. ``localisation_stats``
reports the signed and absolute offsets directly, which characterise that bias without
reference to any threshold.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

N_RAMPS = 19
N_CHANGE_POINTS = 2 * N_RAMPS
N_POINTS_METASTABLE = 5000


def load_ground_truth(path, transition_length=None):
    """
    Read a ``*_GroundTruth.txt`` file and return its ramp intervals.

    Parameters
    ----------
    path : str
        Path to the ground-truth file.
    transition_length : int, optional
        The ``tl`` encoded in the filename. If given, used to sanity-check the
        parsed intervals.

        Note: 299 of the 300 shipped ground-truth files have every ramp width
        exactly equal to ``tl``. The single exception, ``sigma_5_transition_20``,
        has widths in {19, 20, 21}; its total transition mass (380 = 19*20) and
        trajectory length (5380 = 5000 + 19*20) are both exactly right, so this is
        a benign +/-1 bookkeeping artifact in the generator rather than malformed
        data. A tolerance of 1 is therefore allowed.

    Returns
    -------
    ramps : ndarray, shape (19, 2)
        Half-open intervals ``[start, end)`` spanning each gradual transition.
    """
    gt = np.loadtxt(path).astype(int)
    if gt.ndim == 0:
        gt = gt.reshape(1)

    starts = gt[1:-1:2]
    ends = gt[2::2]
    n = min(len(starts), len(ends))
    ramps = np.stack([starts[:n], ends[:n]], axis=1)

    if transition_length is not None:
        widths = ramps[:, 1] - ramps[:, 0]
        if not np.all(np.abs(widths - transition_length) <= 1):
            raise ValueError(
                f"{path}: parsed ramp widths {sorted(set(widths.tolist()))} "
                f"deviate from tl={transition_length} by more than 1"
            )
        if widths.sum() != N_RAMPS * transition_length:
            raise ValueError(
                f"{path}: total transition mass {widths.sum()} != "
                f"{N_RAMPS} * {transition_length}"
            )

    return ramps


def true_change_points(ramps):
    """
    The 38 true change points: the start and end of every ramp, sorted.

    These are the instants where stability becomes transition (ramp start) and where
    transition becomes stability (ramp end).
    """
    return np.sort(np.asarray(ramps, dtype=int).reshape(-1, 2).ravel())


def ground_truth_length(transition_length):
    """Expected trajectory length for a given transition length."""
    return N_POINTS_METASTABLE + N_RAMPS * transition_length


def _strip_boundary(predicted, trajectory_length):
    """
    Drop the trivial change points at 0 and ``T - 1``.

    ``identify_change_points`` always emits these by construction, so counting them as
    false positives would penalise every method by a fixed amount independent of
    detection quality.
    """
    predicted = np.asarray(predicted, dtype=int).ravel()
    if trajectory_length is None:
        return predicted
    boundary = {0, trajectory_length - 1}
    return np.array([p for p in predicted if p not in boundary], dtype=int)


def match_change_points(predicted, truth, tau):
    """
    One-to-one optimal assignment between predictions and true change points.

    Pairs further apart than ``tau`` are forbidden. Among feasible assignments, the one
    minimising total absolute displacement is chosen.

    Returns
    -------
    pairs : ndarray, shape (n_matched, 2)
        Columns are (index into ``predicted``, index into ``truth``).
    """
    predicted = np.asarray(predicted, dtype=int).ravel()
    truth = np.asarray(truth, dtype=int).ravel()

    if predicted.size == 0 or truth.size == 0:
        return np.empty((0, 2), dtype=int)

    cost = np.abs(predicted[:, None] - truth[None, :]).astype(float)
    feasible = cost <= tau
    if not feasible.any():
        return np.empty((0, 2), dtype=int)

    # forbid infeasible pairs with a cost that can never be worth paying
    big = float(cost.size) * (tau + 1.0) + 1.0
    cost_masked = np.where(feasible, cost, big)

    rows, cols = linear_sum_assignment(cost_masked)
    keep = feasible[rows, cols]
    return np.stack([rows[keep], cols[keep]], axis=1)


def score_change_points(predicted, ramps, tau=0, trajectory_length=None):
    """
    Score predicted change points against the 38 true ramp boundaries.

    Parameters
    ----------
    predicted : array-like of int
        Predicted change-point indices.
    ramps : ndarray, shape (n_ramps, 2)
        Ramp intervals as returned by ``load_ground_truth``.
    tau : int, default 0
        Matching tolerance in time steps.
    trajectory_length : int, optional
        If given, the trivial boundary predictions at 0 and ``T - 1`` are dropped.

    Returns
    -------
    dict with keys ``precision``, ``recall``, ``f1``, ``n_predicted``, ``n_true``,
    ``n_matched``.
    """
    predicted = _strip_boundary(predicted, trajectory_length)
    truth = true_change_points(ramps)

    n_predicted = int(predicted.size)
    n_true = int(truth.size)
    if n_true == 0:
        raise ValueError("no true change points supplied")

    pairs = match_change_points(predicted, truth, tau)
    n_matched = int(pairs.shape[0])

    precision = n_matched / n_predicted if n_predicted else 0.0
    recall = n_matched / n_true
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_predicted": n_predicted,
        "n_true": n_true,
        "n_matched": n_matched,
    }


def tolerance_curve(predicted, ramps, taus, trajectory_length=None):
    """Score at a range of tolerances. Returns one dict per tau, each with a ``tau`` key."""
    out = []
    for tau in taus:
        row = score_change_points(
            predicted, ramps, tau=tau, trajectory_length=trajectory_length
        )
        row["tau"] = int(tau)
        out.append(row)
    return out


def localisation_stats(predicted, ramps, trajectory_length=None):
    """
    Characterise *where* predictions fall relative to the true change points, with no
    tolerance threshold involved.

    Each prediction is assigned to its nearest true change point. The signed offset is
    oriented so that **positive means displaced into the interior of the ramp** and
    negative means displaced out into the adjacent stable region. This orientation is
    what makes a systematic inward bias visible: the estimator studied here reports each
    ramp as narrower than it truly is, by an amount that grows with the window size.

    Returns
    -------
    dict with ``median_abs_error``, ``mean_abs_error``, ``median_signed_error``,
    ``mean_signed_error``, ``p90_abs_error``, ``n``.
    """
    predicted = _strip_boundary(predicted, trajectory_length)
    ramps = np.asarray(ramps, dtype=int).reshape(-1, 2)
    truth = true_change_points(ramps)

    if predicted.size == 0:
        return {
            "median_abs_error": np.nan,
            "mean_abs_error": np.nan,
            "median_signed_error": np.nan,
            "mean_signed_error": np.nan,
            "p90_abs_error": np.nan,
            "n": 0,
        }

    # is each true change point a ramp start or a ramp end?
    # starts are ramps[:, 0]; interior lies to their right (+1 orientation)
    # ends are ramps[:, 1]; interior lies to their left  (-1 orientation)
    starts = set(ramps[:, 0].tolist())
    orientation = np.array([1 if t in starts else -1 for t in truth])

    nearest = np.argmin(np.abs(predicted[:, None] - truth[None, :]), axis=1)
    raw = predicted - truth[nearest]
    signed = raw * orientation[nearest]
    absolute = np.abs(raw)

    return {
        "median_abs_error": float(np.median(absolute)),
        "mean_abs_error": float(np.mean(absolute)),
        "median_signed_error": float(np.median(signed)),
        "mean_signed_error": float(np.mean(signed)),
        "p90_abs_error": float(np.percentile(absolute, 90)),
        "n": int(predicted.size),
    }
