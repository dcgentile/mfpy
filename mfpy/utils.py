"""Utility functions for time series clustering."""

import numpy as np
from scipy.linalg import svd, pinv
from scipy.sparse import csr_matrix
import warnings


def label_series(X, change_points, labels):
    """
    Let X be a Timeseries, with proposed change points
    change_points, and segments with cluser labels labels.
    This function creates a new, one-dimenstonal timeseries
    X_labels, so that the points of each inherit the label
    of their parent segment
    Parameters
    ----------
    X: array-like
       Timeseries data
    change_points: array-like
       Indices of change points in X
    labels: array-like
       Labels of segments corresponding to change_points of X
    Returns
    -------
    X_labels: array-like
              1D timeseries of labels with same length as X
    """
    T = len(change_points)
    point_labels = np.zeros(X.shape[0])

    for i in range(T - 1):
        t0 = change_points[i]
        t1 = change_points[i + 1]
        label = labels[i]
        point_labels[t0 : t1 - 1] = label

    return point_labels


def construct_counts_matrix(labels, lag_time):
    unique_labels = np.unique(labels)
    n = len(unique_labels)
    state_to_index = {l: i for i, l in enumerate(unique_labels)}
    C = np.zeros((n, n), dtype=int)
    for i in range(len(labels) - lag_time):
        current_idx = state_to_index[labels[i]]
        next_idx = state_to_index[labels[i + lag_time]]
        C[current_idx, next_idx] += 1
    return C


def fixed_point_iteration(C, tol=1e-5, max_iter=10000, verbose=False):
    n = C.shape[0]
    X = np.ones((n, n))
    X_new = X.copy()

    for iter in range(max_iter):
        c = C.sum(axis=1, keepdims=True)
        x = X.sum(axis=1, keepdims=True)

        mask = ~np.eye(n, dtype=bool)
        X_new[mask] = (C + C.T)[mask] / (c / x + (c / x).T)[mask]

        if np.linalg.norm(X_new - X) < tol:
            if verbose:
                print(f"Converged after {iter+1} iterations.")
            break
        X = X_new.copy()
    else:
        if verbose:
            print(f"Did not converge after {max_iter} iterations.")

    return X / X.sum(axis=1, keepdims=True)


def compute_vamp_scores_from_counts(count_matrix, lag_time=1, score_type="VAMP2"):
    """
    Compute VAMP scores directly from a count matrix for MSM discretization selection.

    Parameters:
    -----------
    count_matrix : array_like, shape (n_states, n_states)
        Transition count matrix C_ij = number of transitions from state i to j
    lag_time : int
        Lag time used to construct the count matrix (for interpretation)
    score_type : str
        Type of VAMP score to compute: 'VAMP1', 'VAMP2', or 'VAMPE'

    Returns:
    --------
    score : float
        VAMP score for this discretization
    singular_values : array
        Singular values of the Koopman matrix (useful for analysis)
    """

    C = np.array(count_matrix, dtype=float)
    n_states = C.shape[0]

    # Check if count matrix is square
    if C.shape[0] != C.shape[1]:
        raise ValueError("Count matrix must be square")

    # Compute stationary distribution (row sums for C00)
    row_sums = C.sum(axis=1)

    # Handle empty states (states with no outgoing transitions)
    active_states = row_sums > 0
    if not active_states.all():
        warnings.warn(
            f"Found {(~active_states).sum()} empty states. "
            "Consider removing them or using a different discretization."
        )
        # Keep only active states
        active_idx = np.where(active_states)[0]
        C = C[np.ix_(active_idx, active_idx)]
        row_sums = row_sums[active_states]
        n_states = len(active_idx)

    # Construct covariance matrices from count matrix
    # For classical MSM with indicator basis functions:

    # C00: covariance at time t (diagonal matrix of state populations)
    pi = row_sums / row_sums.sum()  # stationary probabilities
    C00 = np.diag(pi)

    # C01: cross-covariance between time t and t+tau
    # This is just the count matrix normalized by total counts
    total_counts = C.sum()
    C01 = C / total_counts

    # C11: covariance at time t+tau (same as C00 for stationary process)
    col_sums = C.sum(axis=0)
    pi_tau = col_sums / col_sums.sum()
    C11 = np.diag(pi_tau)

    return _compute_vamp_score_from_covariances(C00, C01, C11, score_type)


def compute_vamp_scores_transition_matrix(transition_matrix, score_type="VAMP2"):
    """
    Compute VAMP scores from a transition probability matrix.

    This is useful when you already have an estimated MSM transition matrix.
    """
    T = np.array(transition_matrix, dtype=float)

    # Check if it's a valid transition matrix
    if not np.allclose(T.sum(axis=1), 1.0):
        warnings.warn(
            "Rows don't sum to 1. This may not be a proper transition matrix."
        )

    # Compute stationary distribution
    eigenvals, eigenvecs = np.linalg.eig(T.T)
    stat_idx = np.argmax(np.real(eigenvals))
    pi = np.real(eigenvecs[:, stat_idx])
    pi = pi / pi.sum()
    pi = np.abs(pi)  # Ensure non-negative

    # Construct covariance matrices
    C00 = np.diag(pi)
    C01 = np.diag(pi) @ T  # pi_i * T_ij
    C11 = np.diag(pi)  # Same for stationary process

    return _compute_vamp_score_from_covariances(C00, C01, C11, score_type)


def _compute_vamp_score_from_covariances(C00, C01, C11, score_type="VAMP2"):
    """
    Core computation of VAMP scores from covariance matrices.
    """
    # Regularization for numerical stability
    reg = 1e-10
    C00_reg = C00 + reg * np.eye(C00.shape[0])
    C11_reg = C11 + reg * np.eye(C11.shape[0])

    try:
        # Compute whitened Koopman matrix: C00^(-1/2) * C01 * C11^(-1/2)
        C00_inv_sqrt = _matrix_power(C00_reg, -0.5)
        C11_inv_sqrt = _matrix_power(C11_reg, -0.5)

        K_whitened = C00_inv_sqrt @ C01 @ C11_inv_sqrt

        # SVD of whitened Koopman matrix
        U, s, Vt = svd(K_whitened, full_matrices=False)

        # Remove the trivial singular value (should be ~1 for equilibrium processes)
        # Keep only the non-trivial singular values
        if len(s) > 1:
            s = s[1:]  # Remove largest singular value

        # Compute VAMP score
        if score_type.upper() == "VAMP1":
            score = np.sum(s)
        elif score_type.upper() == "VAMP2":
            score = np.sum(s**2)
        elif score_type.upper() == "VAMPE":
            # VAMP-E is approximation error (lower is better)
            score = len(s) - np.sum(s**2)
        else:
            raise ValueError(f"Unknown score type: {score_type}")

        return score, s

    except np.linalg.LinAlgError:
        warnings.warn("Numerical issues encountered. Discretization may be too fine.")
        return np.nan, np.array([])


def _matrix_power(A, power):
    """Compute matrix power A^power using eigendecomposition."""
    eigenvals, eigenvecs = np.linalg.eigh(A)
    # Clip small eigenvalues to avoid numerical issues
    eigenvals = np.maximum(eigenvals, 1e-12)
    return eigenvecs @ np.diag(eigenvals**power) @ eigenvecs.T
