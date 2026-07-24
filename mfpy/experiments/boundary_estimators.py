"""
Alternative rules for turning the Wasserstein test statistic into change points.

Background
----------
``mfpy.distances.identify_change_points`` works in two stages:

1. **Run detection.** Threshold the metric derivative at its own ``q``-quantile and
   collect maximal runs of consecutive supra-threshold indices. Each run corresponds to
   one gradual transition in the trajectory.
2. **Boundary extraction.** Within each run, report the argmin and argmax of the
   *gradient* of the test statistic as the two change points.

Stage 2 is what reviewer comment 1(a) asks us to justify. Empirically it produces
change points that are systematically *inset* from the true ramp boundaries -- the
steepest rise in the statistic occurs after the ramp has already begun, and the
steepest fall occurs before it ends -- by an amount that grows with the window size.

This module keeps stage 1 byte-identical to the library (including its quirks; see
``find_runs``) and parameterises only stage 2, so that differences between estimators
are attributable to boundary extraction alone.

Estimators
----------
``gradient``  argmin/argmax of the gradient within the run. Reproduces the library
              exactly; this is the incumbent.
``run_edge``  the first and last index of the run itself. The most direct reading of
              "the statistic became elevated here and stopped being elevated there".
``half_max``  the first and last index at which the statistic exceeds the midpoint of
              its within-run range. A standard edge-localisation rule, less sensitive
              than ``run_edge`` to exactly where the threshold happens to fall.
"""

from __future__ import annotations

import numpy as np

ESTIMATORS = ("gradient", "run_edge", "half_max")


def find_runs(statistic, q):
    """
    Maximal runs of consecutive supra-threshold indices.

    This is a faithful port of the run-detection half of
    ``mfpy.distances.identify_change_points``, preserving two behaviours that matter
    for reproducing its output exactly:

    * A run only opens when two *adjacent* candidate indices are seen, so isolated
      supra-threshold points are ignored rather than becoming length-1 runs.
    * The loop runs to ``N - 2``, so a run still open when the candidate list is
      exhausted is never closed and is silently dropped. (The library's ``n == N - 1``
      guard is unreachable inside ``range(N - 1)``.)

    Returns
    -------
    runs : list of (start, end)
        Half-open in the same sense the library uses: it slices ``[start:end]``.
    """
    statistic = np.asarray(statistic).ravel()
    cutoff = np.quantile(statistic, q)
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


def _extract(statistic, gradient, start, end, estimator):
    """Return the two boundary estimates for a single run."""
    if end <= start:
        return []

    if estimator == "gradient":
        segment = gradient[start:end]
        return [int(np.argmin(segment)) + start, int(np.argmax(segment)) + start]

    if estimator == "run_edge":
        return [int(start), int(end)]

    if estimator == "half_max":
        segment = statistic[start:end]
        level = 0.5 * (segment.min() + segment.max())
        above = np.where(segment >= level)[0]
        if above.size == 0:
            return [int(start), int(end)]
        return [int(above[0]) + start, int(above[-1]) + start]

    raise ValueError(f"unknown estimator: {estimator}")


def identify_change_points(statistic, q, estimator="gradient"):
    """
    Locate change points in a test statistic under the chosen boundary estimator.

    With ``estimator="gradient"`` this reproduces
    ``mfpy.distances.identify_change_points`` exactly, including the trivial endpoints
    at 0 and ``len(statistic) - 1`` that it always appends.
    """
    statistic = np.asarray(statistic).ravel()
    gradient = np.gradient(statistic)

    changes = [0]
    for start, end in find_runs(statistic, q):
        changes.extend(_extract(statistic, gradient, start, end, estimator))
    changes.append(statistic.shape[0] - 1)

    return np.sort(np.unique(changes))
