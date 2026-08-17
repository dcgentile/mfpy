"""
Sensitivity of the clustering to sigma in the similarity matrix (action item 7 /
reviewer minor comment 6).

What sigma is, and where it does *not* appear
---------------------------------------------
The manuscript forms the similarity matrix as

    A_ij = exp( - W2^2(nu_i, nu_j) / (2 sigma) )

and states "In this work, we always take sigma = 1". Two facts have to be stated before any
sweep is meaningful:

1. **sigma enters only the spectral path.** ADP consumes the pairwise *distance* matrix
   directly (`Data(distances=D)`) and never forms `A`. Every headline figure in the
   manuscript uses ADP, so those results carry no sigma dependence at all. The honest
   answer to minor comment 6 begins here.
2. **The code and the text disagree.** `mfpy/clustering.py` computes
   ``np.exp(-(distance_matrix**2))``, i.e. ``exp(-W2^2)``, which is ``sigma = 1/2`` in the
   manuscript's parameterisation -- not ``sigma = 1``. Both values are marked in the output
   so the size of the discrepancy can be read off directly.

Three notions of sensitivity are reported, because they can disagree
---------------------------------------------------------------------
``stability``
    Adjusted Rand index between the clustering at sigma and at the next sigma on the grid,
    and between each sigma and the reference sigma = 1/2 (what the code actually does).
    Needs no ground truth, so it applies to every data set uniformly.

``quality``
    V-measure and normalised distributional fidelity against ground truth, reusing
    `mfpy.experiments.vmeasure` and `mfpy.experiments.distributional_fidelity` so the
    numbers are comparable with those experiments.

``implied_k``
    The number of clusters an eigengap heuristic would select from the normalised graph
    Laplacian of `A`. Spectral clustering takes `K` as an input, so if sigma changes the
    eigengap it changes how many metastable states a practitioner would infer -- which is
    the question a referee familiar with spectral clustering will actually ask. Reported
    independently of the fixed-`K` clustering used for stability and quality.

Usage
-----
    python -m mfpy.experiments.sigma_sensitivity --langevin
    python -m mfpy.experiments.sigma_sensitivity --toy --limit 30
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
warnings.filterwarnings("ignore", message=".*Graph is not fully connected.*")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from mfpy.experiments.distributional_fidelity import (  # noqa: E402
    distributional_error,
)
from mfpy.experiments.toy_scoring import load_ground_truth  # noqa: E402
from mfpy.experiments.toy_sweep import (  # noqa: E402
    DATA_DIR,
    discover_trajectories,
    fast_metric_derivative_1d,
)
from mfpy.experiments.vmeasure import (  # noqa: E402
    LANG_Q,
    LANG_W,
    LANGEVIN_PATH,
    TOY_Q,
    TOY_W,
    TRANSITION,
    _pairwise_segment_distances,
    langevin_point_labels,
    score,
    toy_point_labels,
)

RESULT_DIR = os.path.join(REPO_ROOT, "results", "sigma_sensitivity")

# Six orders of magnitude, log-spaced, with the two values of interest inserted exactly:
# 0.5 is what the code does, 1.0 is what the manuscript says.
SIGMAS = sorted(
    set(np.round(np.logspace(-3, 3, 25), 6)).union({0.5, 1.0})
)

DEFAULT_K = 3  # the manuscript's choice for both the Langevin and toy spectral figures
LANGEVIN_BASIN_C = 0.0  # see results/distributional_fidelity/FINDINGS.md section 3


def similarity(D, sigma):
    """A_ij = exp(-W2^2 / (2 sigma)), the manuscript's rule."""
    return np.exp(-(np.asarray(D, dtype=float) ** 2) / (2.0 * sigma))


def spectral_labels(A, k, seed=0):
    from sklearn.cluster import SpectralClustering

    return SpectralClustering(
        n_clusters=k, affinity="precomputed", random_state=seed, n_init=10
    ).fit(A).labels_


def eigengap_k(A, k_max=10):
    """
    Number of clusters implied by the largest gap in the normalised Laplacian spectrum.

    Uses the symmetric normalised Laplacian L = I - D^-1/2 A D^-1/2 and returns the index
    of the largest gap among the smallest ``k_max`` eigenvalues -- the standard heuristic
    (von Luxburg 2007). Reported as a diagnostic only; it is not used to cluster.
    """
    A = np.asarray(A, dtype=float)
    d = A.sum(axis=1)
    d[d <= 0] = 1e-12
    dinv = 1.0 / np.sqrt(d)
    L = np.eye(A.shape[0]) - (A * dinv[:, None]) * dinv[None, :]
    vals = np.sort(np.linalg.eigvalsh(L))[: min(k_max + 1, A.shape[0])]
    gaps = np.diff(vals)
    return int(np.argmax(gaps) + 1)


def segments_for(X, w, q):
    """Change points and the segment distance matrix, fixed across the sigma sweep."""
    from mfpy.experiments.boundary_estimators import identify_change_points

    X = np.asarray(X, dtype=float).ravel()
    gammadot = fast_metric_derivative_1d(X, w)
    changes = np.unique(
        np.append(identify_change_points(gammadot, q, estimator="gradient"), X.shape[0])
    )
    return changes, _pairwise_segment_distances(X, changes)


def _broadcast(changes, segment_labels, T):
    point_labels = np.empty(T, dtype=int)
    for i in range(len(changes) - 1):
        point_labels[changes[i] : changes[i + 1]] = segment_labels[i]
    return point_labels


def median_heuristic_sigma(D):
    """
    Scale-free choice: sigma = median(W2^2) / 2, so the typical exponent is O(1).

    The manuscript's fixed sigma = 1 is *not* scale invariant -- see FINDINGS. This is the
    standard median heuristic for a Gaussian kernel, adapted to the manuscript's
    parameterisation ``exp(-W2^2 / (2 sigma))``.
    """
    off = np.asarray(D, dtype=float)[np.triu_indices(len(D), 1)]
    med = float(np.median(off**2))
    return max(med / 2.0, 1e-12)


def sweep_one(X, truth3, w, q, k=DEFAULT_K, sigmas=SIGMAS, meta=None):
    """
    Sweep sigma on a single trajectory with the segmentation held fixed.

    ``sigmas`` entries may be floats or the string ``"median"``, which is resolved per
    trajectory to ``median_heuristic_sigma(D)``.
    """
    from sklearn.metrics import adjusted_rand_score

    X = np.asarray(X, dtype=float).ravel()
    changes, D = segments_for(X, w, q)
    n_seg = len(changes) - 1
    if n_seg < k:
        return []

    resolved = [median_heuristic_sigma(D) if s == "median" else s for s in sigmas]

    per_sigma = []
    for s in resolved:
        A = similarity(D, s)
        per_sigma.append((spectral_labels(A, k), eigengap_k(A)))

    # Reference for the ARI column: sigma = 1/2, what mfpy/clustering.py actually computes.
    reference = spectral_labels(similarity(D, 0.5), k)
    keep = truth3 != TRANSITION

    rows = []
    for i, (s, label) in enumerate(zip(resolved, sigmas)):
        seg_labels, k_gap = per_sigma[i]
        pred = _broadcast(changes, seg_labels, X.shape[0])

        row = dict(meta or {})
        row.update(
            {
                "sigma": s,
                "sigma_rule": "median" if label == "median" else "fixed",
                "median_w2_sq": float(
                    np.median(np.asarray(D)[np.triu_indices(len(D), 1)] ** 2)
                ),
                "n_segments": n_seg,
                "k": k,
                "eigengap_k": k_gap,
                "ari_vs_code_default": adjusted_rand_score(reference, seg_labels),
                "ari_vs_next_sigma": (
                    adjusted_rand_score(seg_labels, per_sigma[i + 1][0])
                    if i + 1 < len(per_sigma)
                    else np.nan
                ),
                "n_clusters_used": int(len(np.unique(seg_labels))),
            }
        )
        row["v3"] = score(truth3, pred)[2]
        row["v2"] = score(truth3[keep], pred[keep])[2]
        row["fidelity_norm"] = distributional_error(X, truth3, pred, "d").get(
            "d_fidelity_norm", np.nan
        )
        rows.append(row)
    return rows


def run_langevin(w=LANG_W, q=LANG_Q, k=DEFAULT_K, sigmas=None):
    X = np.loadtxt(LANGEVIN_PATH).ravel()
    truth3 = langevin_point_labels(X, LANGEVIN_BASIN_C)
    return pd.DataFrame(
        sweep_one(
            X, truth3, w, q, k=k, sigmas=sigmas or SIGMAS, meta={"dataset": "langevin"}
        )
    )


def run_toy(limit=None, w=TOY_W, q=TOY_Q, k=DEFAULT_K, start=0, stop=None, sigmas=None):
    jobs = discover_trajectories(DATA_DIR)
    if limit:
        jobs = jobs[:: max(1, len(jobs) // limit)][:limit]
    jobs = jobs[start:stop]

    rows, t0 = [], time.time()
    for n, (sigma_noise, tl, traj_path, gt_path) in enumerate(jobs, 1):
        X = np.loadtxt(traj_path).ravel()
        ramps = load_ground_truth(gt_path, transition_length=tl)
        truth3 = toy_point_labels(ramps, X.shape[0])
        rows.extend(
            sweep_one(
                X,
                truth3,
                w,
                q,
                k=k,
                sigmas=sigmas or SIGMAS,
                meta={
                    "dataset": "toy",
                    "noise_sigma": sigma_noise,
                    "transition_length": tl,
                },
            )
        )
        if n % 10 == 0:
            print(f"[{n}/{len(jobs)}] {time.time() - t0:.0f}s", flush=True)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--toy", action="store_true")
    ap.add_argument("--langevin", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stop", type=int, default=None)
    ap.add_argument("--k", type=int, default=DEFAULT_K)
    ap.add_argument(
        "--heuristic-only",
        action="store_true",
        help="evaluate only the per-trajectory median heuristic sigma",
    )
    ap.add_argument("--suffix", default="")
    ap.add_argument("--out", default=RESULT_DIR)
    args = ap.parse_args()
    if not (args.toy or args.langevin):
        args.toy = args.langevin = True

    os.makedirs(args.out, exist_ok=True)
    tmp = "/tmp/sigmasens"
    os.makedirs(tmp, exist_ok=True)

    def save(df, name):
        name = f"{name}{args.suffix}.csv"
        local = os.path.join(tmp, name)
        df.to_csv(local, index=False)
        shutil.copy(local, os.path.join(args.out, name))
        print(f"{name}: {len(df)} rows")

    sigmas = ["median"] if args.heuristic_only else None
    if args.langevin:
        save(run_langevin(k=args.k, sigmas=sigmas), "sigma_langevin")
    if args.toy:
        save(
            run_toy(
                limit=args.limit,
                k=args.k,
                start=args.start,
                stop=args.stop,
                sigmas=sigmas,
            ),
            "sigma_toy",
        )


if __name__ == "__main__":
    main()
