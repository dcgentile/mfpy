"""
Distributional fidelity: how close are the recovered clusters' *laws* to the true laws?

Motivation
----------
V-measure (`results/vmeasure/`) scores point assignment and nothing else: it is a function
of the label contingency table, so it cannot see the values, the geometry, or the shape of
the recovered distributions. A method that separates the states with a hard threshold in
`x` scores well even when the resulting clusters misrepresent each state's law -- which is
exactly the k-means behaviour the manuscript objects to qualitatively (figures
`toy-kmeans-distributions`, `langevin-kmeans-distributions`), and exactly what V-measure
cannot charge for.

This module measures the thing the manuscript actually claims, in the geometry the method
is built on: the 2-Wasserstein distance between the empirical law of each recovered
cluster and the empirical law of each true class.

Two directions, deliberately mirroring homogeneity and completeness
-------------------------------------------------------------------
Let the true classes have empirical laws ``mu_i`` with point masses ``p_i``, and the
recovered clusters have laws ``nu_j`` with masses ``w_j``.

``fidelity``   = sum_j w_j * min_i W2(nu_j, mu_i)
    "Does every cluster look like some true state?" The distributional analogue of
    homogeneity. A cluster that is a truncated slab of the value axis -- k-means' output --
    is far from every true law and is penalised here.

``coverage``   = sum_i p_i * min_j W2(mu_i, nu_j)
    "Is every true state represented by some cluster?" The distributional analogue of
    completeness. Collapsing two states into one cluster is penalised here.

Both are reported normalised by ``separation`` = W2 between the two metastable state laws,
which makes them dimensionless and comparable across data sets: a normalised error of 0.1
means the recovered law is off by a tenth of the A-to-B distance.

Caveat, stated up front
-----------------------
Neither direction alone is a score to optimise. Splitting the data into many tiny clusters
drives ``fidelity`` down while leaving ``coverage`` high; merging everything does the
reverse. They must be read together, and the cluster count is reported alongside. This is
the same failure mode homogeneity and completeness have individually, which is why
V-measure takes their harmonic mean; a harmonic mean is not meaningful for errors on an
unbounded scale, so both numbers are reported rather than combined.

Usage
-----
    python -m mfpy.experiments.distributional_fidelity --toy --langevin
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

from mfpy.experiments.toy_scoring import load_ground_truth  # noqa: E402
from mfpy.experiments.toy_sweep import (  # noqa: E402
    DATA_DIR,
    discover_trajectories,
)
from mfpy.experiments.vmeasure import (  # noqa: E402
    ADP_FAILURES,
    LANG_CORES,
    LANG_Q,
    LANG_W,
    LANGEVIN_PATH,
    STATE_A,
    STATE_B,
    TOY_Q,
    TOY_W,
    TRANSITION,
    _adp_labels,
    _pairwise_segment_distances,
    cluster_trajectory,
    kmeans_labels,
    langevin_point_labels,
    toy_point_labels,
)

RESULT_DIR = os.path.join(REPO_ROOT, "results", "distributional_fidelity")

N_QUANTILES = 512
MIN_CLUSTER_POINTS = 10  # laws estimated from fewer points than this are not meaningful


def quantile_profile(values, n=N_QUANTILES):
    """Inverse CDF of an empirical measure on a fixed grid of n quantile levels."""
    values = np.sort(np.asarray(values, dtype=float).ravel())
    if values.size == 0:
        return None
    grid = (np.arange(n) + 0.5) / n
    return np.interp(grid * values.size - 0.5, np.arange(values.size), values)


def w2(profile_a, profile_b):
    """2-Wasserstein distance between two measures given as quantile profiles."""
    return float(np.sqrt(np.mean((profile_a - profile_b) ** 2)))


def _profiles(values, labels, min_points=MIN_CLUSTER_POINTS):
    """Return {label: (quantile profile, mass)} for labels with enough support."""
    values = np.asarray(values, dtype=float).ravel()
    labels = np.asarray(labels)
    total = labels.size
    out = {}
    for lab in np.unique(labels):
        mask = labels == lab
        if mask.sum() < min_points:
            continue
        out[lab] = (quantile_profile(values[mask]), mask.sum() / total)
    return out


def distributional_error(values, truth, pred, prefix):
    """
    Size-weighted one-sided Wasserstein errors between cluster laws and class laws.

    Returns fidelity (clusters -> nearest class), coverage (classes -> nearest cluster),
    the A-to-B separation used for normalisation, and the normalised versions of each.
    """
    true_laws = _profiles(values, truth)
    pred_laws = _profiles(values, pred)

    out = {f"{prefix}_n_clusters_scored": len(pred_laws)}
    if not true_laws or not pred_laws:
        return out

    # Normalisation: the distance between the two metastable state laws. This is the
    # natural unit -- an error of 1.0 is as large as the whole A-to-B separation.
    if STATE_A in true_laws and STATE_B in true_laws:
        separation = w2(true_laws[STATE_A][0], true_laws[STATE_B][0])
    else:
        separation = np.nan

    fidelity = sum(
        mass * min(w2(prof, tp) for tp, _ in true_laws.values())
        for prof, mass in pred_laws.values()
    )
    total_pred_mass = sum(mass for _, mass in pred_laws.values())

    coverage = sum(
        mass * min(w2(prof, pp) for pp, _ in pred_laws.values())
        for prof, mass in true_laws.values()
    )
    total_true_mass = sum(mass for _, mass in true_laws.values())

    fidelity /= total_pred_mass
    coverage /= total_true_mass

    out.update(
        {
            f"{prefix}_fidelity": fidelity,
            f"{prefix}_coverage": coverage,
            f"{prefix}_fidelity_norm": fidelity / separation,
            f"{prefix}_coverage_norm": coverage / separation,
        }
    )
    out["separation"] = separation
    return out


def _score_all(values, truth3, pred, prefix):
    """Distributional error against the 3-class and the 2-class (cores only) targets."""
    row = distributional_error(values, truth3, pred, f"{prefix}3")
    keep = truth3 != TRANSITION
    row.update(
        distributional_error(values[keep], truth3[keep], pred[keep], f"{prefix}2")
    )
    return row


# --------------------------------------------------------------------------------------
# drivers
# --------------------------------------------------------------------------------------


def run_toy(
    limit=None, w=TOY_W, q=TOY_Q, threshold="quantile", R=199, start=0, stop=None
):
    jobs = discover_trajectories(DATA_DIR)
    if limit:
        jobs = jobs[:: max(1, len(jobs) // limit)][:limit]
    jobs = jobs[start:stop]

    rows, t0 = [], time.time()
    for n, (sigma, tl, traj_path, gt_path) in enumerate(jobs, 1):
        X = np.loadtxt(traj_path).ravel()
        ramps = load_ground_truth(gt_path, transition_length=tl)
        truth3 = toy_point_labels(ramps, X.shape[0])

        row = {
            "sigma": sigma,
            "transition_length": tl,
            "T": X.shape[0],
            "threshold": threshold,
            "q": q,
            "status": "ok",
        }
        try:
            pred, changes, _ = cluster_trajectory(X, w, q, threshold=threshold, R=R)
        except ADP_FAILURES as exc:
            row["status"] = f"adp failed: {type(exc).__name__}"
            rows.append(row)
            continue

        row["n_segments"] = len(changes) - 1
        row["adp_k"] = int(len(np.unique(pred)))
        row.update(_score_all(X, truth3, pred, "adp"))
        row.update(_score_all(X, truth3, kmeans_labels(X, 3), "kmeans"))

        try:
            oc = np.unique(np.concatenate([[0], ramps.ravel(), [X.shape[0]]]))
            seg, _ = _adp_labels(_pairwise_segment_distances(X, oc))
            op = np.empty(X.shape[0], dtype=int)
            for i in range(len(oc) - 1):
                op[oc[i] : oc[i + 1]] = seg[i]
            row.update(_score_all(X, truth3, op, "oracle"))
        except ADP_FAILURES:
            pass

        rows.append(row)
        if n % 50 == 0:
            print(f"[{n}/{len(jobs)}] {time.time() - t0:.0f}s", flush=True)

    return pd.DataFrame(rows)


def run_langevin(w=LANG_W, q=LANG_Q, cores=LANG_CORES, threshold="quantile", R=199):
    X = np.loadtxt(LANGEVIN_PATH).ravel()
    pred, changes, _ = cluster_trajectory(X, w, q, threshold=threshold, R=R)
    km = kmeans_labels(X, 3)

    rows = []
    for c in cores:
        truth3 = langevin_point_labels(X, c)
        row = {
            "core_halfwidth": c,
            "threshold": threshold,
            "q": q,
            "n_segments": len(changes) - 1,
            "adp_k": int(len(np.unique(pred))),
        }
        row.update(_score_all(X, truth3, pred, "adp"))
        row.update(_score_all(X, truth3, km, "kmeans"))
        rows.append(row)
    return pd.DataFrame(rows)


def langevin_per_cluster(c=0.0, w=LANG_W, q=LANG_Q, threshold="quantile", R=199):
    """
    Per-cluster breakdown for the Langevin trajectory: for each recovered cluster, its
    mass and its W2 distance to the nearest true state law. This is the quantitative
    version of the histogram figures in the manuscript.

    The default ``c = 0`` is the *basin* definition -- state A is ``x < 0``, state B is
    ``x > 0``, split at the barrier. This is the only non-arbitrary choice: any ``c > 0``
    makes the reference laws themselves truncated at a threshold in ``x``, which flatters
    k-means for reasons that have nothing to do with clustering quality. See FINDINGS.
    """
    X = np.loadtxt(LANGEVIN_PATH).ravel()
    truth3 = langevin_point_labels(X, c)
    true_laws = _profiles(X, truth3)
    names = {STATE_A: "state A", TRANSITION: "transition", STATE_B: "state B"}

    pred, _, _ = cluster_trajectory(X, w, q, threshold=threshold, R=R)
    km = kmeans_labels(X, 3)

    rows = []
    for method, labels in [("adp", pred), ("kmeans", km)]:
        for lab, (prof, mass) in sorted(
            _profiles(X, labels).items(), key=lambda kv: -kv[1][1]
        ):
            dists = {t: w2(prof, tp) for t, (tp, _) in true_laws.items()}
            nearest = min(dists, key=dists.get)
            rows.append(
                {
                    "method": method,
                    "cluster": int(lab),
                    "mass": mass,
                    "mean": float(np.mean(prof)),
                    "std": float(np.std(prof)),
                    "nearest_true_state": names.get(nearest, nearest),
                    "w2_to_nearest": dists[nearest],
                }
            )
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
    ap.add_argument("--q", type=float, default=None)
    ap.add_argument("--R", type=int, default=199)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stop", type=int, default=None)
    ap.add_argument("--suffix", default="")
    args = ap.parse_args()
    if not (args.toy or args.langevin):
        args.toy = args.langevin = True

    os.makedirs(args.out, exist_ok=True)
    tmp = "/tmp/distfid"
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

    kw = dict(threshold=args.threshold, R=args.R)
    if args.toy:
        save(
            run_toy(
                limit=args.limit,
                q=_q(TOY_Q),
                start=args.start,
                stop=args.stop,
                **kw,
            ),
            "distfid_toy",
        )
    if args.langevin:
        save(run_langevin(q=_q(LANG_Q), **kw), "distfid_langevin")
        save(langevin_per_cluster(q=_q(LANG_Q), **kw), "distfid_langevin_per_cluster")


if __name__ == "__main__":
    main()
