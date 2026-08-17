"""Distance metrics for time series data."""

from tqdm import tqdm
import numpy as np
import ot
from scipy.spatial.distance import euclidean
from scipy.stats import pearsonr


def compute_metric_derivative_1d(X, w):
    """
    Compute change points for the time series X using windowsize w

    Parameters
    ----------
    X: array-like,
       time series data
    w: integer,
       windowsize for sliding window

    Returns
    -------
    gammadot
        floating point array of approximated metric derivative
        in the wasserstein space for the process
    """
    T = X.shape[0]
    gammadot = np.zeros(X.shape)
    for t in tqdm(range(w, T - w)):
        a = X[t - w : t]
        b = X[t : t + w]
        gammadot[t] = np.sqrt(ot.emd2_1d(a, b))
    return gammadot


def compute_metric_derivative_md(X, w, reg=0.05, numIterMax=200, **sinkhorn_kwargs):
    """
    Multivariate analogue of compute_metric_derivative_1d.

    Exact W2 between empirical measures in R^d (d > 1) is a d! or LP-scale problem and
    is not tractable at sliding-window scale, so this uses the entropic-regularized
    (Sinkhorn) OT distance instead, which is the standard fix for the curse of
    dimensionality in this setting. This is a genuinely different statistic from
    compute_metric_derivative_1d applied per-coordinate: it sees the joint sample in
    each half-window, so it is sensitive to changes in dependence structure (e.g. a
    change in correlation) that leave every marginal untouched.

    Parameters
    ----------
    X: array-like, shape (T, d)
       multivariate time series
    w: integer,
       windowsize for sliding window
    reg: float,
       entropic regularization strength passed to ot.bregman.empirical_sinkhorn2.
       Larger reg = more bias toward 0, smaller reg = slower/less stable Sinkhorn.
       Default 0.05 chosen empirically on results/correlation_change (best
       signal-to-noise of the near-tested grid {0.03, 0.05, 0.1, 0.2, 0.3}).
    numIterMax: integer,
       cap on Sinkhorn iterations per window. The POT default (10000) lets a
       non-converging window stall a whole sweep; 200 is enough to converge for
       reg >= 0.03 at window sizes used here and keeps runtime bounded regardless.
    sinkhorn_kwargs:
       forwarded to ot.bregman.empirical_sinkhorn2 (e.g. numIterMax, stopThr)

    Returns
    -------
    gammadot
        floating point array of shape (T,), the approximate metric derivative using
        the entropic OT cost between left- and right-half-window empirical measures.
        Comparable in scale to compute_metric_derivative_1d (both are sqrt of an OT
        cost with squared-Euclidean ground metric) but not identical, since Sinkhorn
        is a biased estimator of W2.
    """
    X = np.asarray(X)
    if X.ndim == 1:
        X = X[:, None]
    T = X.shape[0]
    gammadot = np.zeros(T)
    for t in tqdm(range(w, T - w)):
        a = X[t - w : t]
        b = X[t : t + w]
        cost = ot.bregman.empirical_sinkhorn2(a, b, reg, numIterMax=numIterMax, **sinkhorn_kwargs)
        gammadot[t] = np.sqrt(max(cost, 0.0))
    return gammadot


def compute_metric_derivative_periodic(X, w):
    """
    Compute change points for a periodic time series X using windowsize w
    N.B.: FUCTION ASSUMES TIME SERIES TAKES VALUES IN [0,1)

    Parameters
    ----------
    X: array-like,
       time series data
    w: integer,
       windowsize for sliding window

    Returns
    -------
    gammadot
        floating point array of approximated metric derivative
        in the wasserstein space for the process
    """
    T = X.shape[0]
    gammadot = np.zeros(X.shape)
    for t in tqdm(range(w, T - w)):
        a = X[t - w : t]
        b = X[t : t + w]
        gammadot[t] = np.sqrt(ot.wasserstein_circle(a, b, p=2))
    return gammadot


def identify_change_points(X, q):
    """
    Compute change points for the time series X using windowsize w

    Parameters
    ----------
    X: array-like,
       time series data
    w: integer,
       windowsize for sliding window

    Returns
    -------
    change_points
        boolean array, with entries corresponding to whether
        or not a change occured
    """
    grad = np.gradient(X)
    cutoff = np.quantile(X, q)
    candidates = np.where(X > cutoff)[0]

    in_sequence = False
    N = candidates.shape[0]
    seq_start = 0
    changes = [0]

    for n in range(N - 1):
        curr = candidates[n]
        nx = candidates[n + 1]

        if not in_sequence and nx - curr == 1:
            in_sequence = True
            seq_start = curr

        if (in_sequence and nx - curr > 1) or n == N - 1:
            in_sequence = False
            seq_end = curr
            sequence = grad[seq_start:seq_end]
            changes.append(np.argmin(sequence) + seq_start)
            changes.append(np.argmax(sequence) + seq_start)

    changes.append(X.shape[0] - 1)
    return np.sort(np.unique(changes))


def compute_change_points(X, w, q, periodic):
    gammadot = (
        compute_metric_derivative_1d(X, w)
        if not periodic
        else compute_metric_derivative_periodic(X, w)
    )
    changes = identify_change_points(gammadot, q)
    return changes


def compute_pairwise_segment_distances(X, changes):
    """
    assumes that changes[0] == 0 and changes[-1] == X.shape[0]
    """
    N = changes.shape[0]
    distance_matrix = np.zeros((N - 1, N - 1))
    indices = np.triu_indices_from(distance_matrix)

    for idx in tqdm(zip(indices[0], indices[1]), total=len(indices[0])):
        i, j = idx
        t0 = changes[i]
        t1 = changes[i + 1]
        a = X[t0:t1]

        # TODO: need to safety check against empty segments

        s0 = changes[j]
        s1 = changes[j + 1]
        b = X[s0:s1]

        d = np.sqrt(ot.emd2_1d(a, b))

        distance_matrix[i, j] = d
        distance_matrix[j, i] = d

    return distance_matrix


def compute_pairwise_segment_distances_periodic(X, changes):
    """
    assumes that changes[0] == 0 and changes[-1] == X.shape[0]
    """
    N = changes.shape[0]
    distance_matrix = np.zeros((N - 1, N - 1))
    indices = np.triu_indices_from(distance_matrix)

    for idx in tqdm(zip(indices[0], indices[1]), total=len(indices[0])):
        i, j = idx
        t0 = changes[i]
        t1 = changes[i + 1]
        a = X[t0:t1]

        # TODO: need to safety check against empty segments

        s0 = changes[j]
        s1 = changes[j + 1]
        b = X[s0:s1]

        d = np.sqrt(ot.wasserstein_circle(a, b, p=2))

        distance_matrix[i, j] = d
        distance_matrix[j, i] = d

    return distance_matrix
