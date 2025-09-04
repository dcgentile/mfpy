"""Time series clustering algorithms."""

from dadapy.data import Data
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import SpectralClustering
from itertools import product

from mfpy.utils import (
    compute_vamp_scores_from_counts,
    label_series,
    construct_counts_matrix,
    fixed_point_iteration,
)
from .distances import (
    compute_change_points,
    compute_pairwise_segment_distances,
    compute_pairwise_segment_distances_periodic,
)


class SegmentBasedCluster(BaseEstimator, ClusterMixin):
    """
    Segment-based clustering for time series data with the Wasserstein distance

    Parameters
    ----------
    w : int, default=100
        number of samples per window
    q : float in (0,1), default=0.75
        quantile specifying cutoff for change points
    """

    def __init__(
        self,
        window_size=100,
        quantile=0.75,
        periodic=False,
        method="adp",
        num_clusters=0,
    ):
        self.window_size = window_size
        self.quantile = quantile
        self.periodic = periodic
        self.segment_labels_ = None
        self.point_labels_ = None
        self.method = method
        self.num_clusters = num_clusters
        self.change_points_ = None

    def fit(self, X, y=None):
        """
        Fit the clustering model. Functions on a componentwise basis,
        handling each dimension of the timeseries individually, and then
        merging labels together at the end of the process.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_timesteps)
            Time series data
        y : Ignored
            Not used, present here for API consistency

        Returns
        -------
        self : object
            Returns the instance itself
        """
        T, D = X.shape
        cw_labels = np.zeros((T, D))
        cw_changes = np.zeros((T, D))
        # dummy clusterer so that we don't mess with our own state haphazardly
        sbc = SegmentBasedCluster(
            self.window_size,
            self.quantile,
            periodic=self.periodic,
            method=self.method,
            num_clusters=self.num_clusters,
        )
        for d in range(D):
            cw_labels[:, d] = sbc.fit_1D(X[:, d]).point_labels_
            changes = sbc.change_points_
            for t in changes:
                cw_changes[t, d] = 1

        self.componentwise_labels_ = cw_labels
        self.change_points_ = cw_changes
        v = np.array([10**i for i in range(D)])
        labels = np.dot(cw_labels, v)
        y = (
            pd.Series(labels)
            .value_counts()
            .rank(method="dense", ascending=False)
            .astype(int)
        )
        self.point_labels_ = y[labels].values

        return self

    def fit_1D(self, X, y=None):
        """
        Fit the clustering model for a 1D timeseries.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_timesteps)
            Time series data
        y : Ignored
            Not used, present here for API consistency

        Returns
        -------
        self : object
            Returns the instance itself
        """

        # parse dimension of data
        # compute the change points
        # compute the pairwise distances
        # use ADP to assign the labels
        w = self.window_size
        q = self.quantile
        periodic = self.periodic
        changes = compute_change_points(X, w, q, periodic)
        self.change_points_ = changes
        distance_matrix = (
            compute_pairwise_segment_distances(X, changes)
            if not periodic
            else compute_pairwise_segment_distances_periodic(X, changes)
        )

        if self.method == "adp":
            data = Data(distances=distance_matrix)
            # data = Data(distances=distance_matrix**2)
            # data = Data(distances=np.exp(-(distance_matrix**2)))
            self.segment_labels_ = data.compute_clustering_ADP(Z=1.65)
        elif self.method == "spectral":
            sc = SpectralClustering(
                n_clusters=self.num_clusters, affinity="precomputed"
            )
            sc.fit(np.exp(-(distance_matrix**2)))
            self.segment_labels_ = sc.labels_
        elif self.method == "dpc":
            pass

        self.point_labels_ = label_series(X, changes, self.segment_labels_)
        return self

    def predict(self, X):
        """
        Predict cluster labels for new data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_timesteps)
            Time series data

        Returns
        -------
        labels : array, shape (n_samples,)
            Cluster labels
        """
        raise NotImplementedError

    def fit_predict(self, X, y=None):
        """
        Fit the model and predict cluster labels.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_timesteps)
            Time series data
        y : Ignored
            Not used, present here for API consistency

        Returns
        -------
        labels : array, shape (n_samples,)
            Cluster labels
        """
        return self.fit(X).point_labels_

    def score(self, X, lagtime=100):
        if self.point_labels_ is None:
            self.fit(X)

        labels = self.point_labels_
        counts = construct_counts_matrix(labels, lagtime)
        score, svals = compute_vamp_scores_from_counts(counts, lagtime)
        return score
        # msm = fixed_point_iteration(counts)
        # _, singular_values, _ = np.linalg.svd(msm)
        # return np.sum(singular_values**2)

    def gridsearch(
        self, X, w_values=None, q_values=None, fixed_w=None, fixed_q=None, lagtime=100
    ):
        """
        Perform grid search to find optimal parameters that maximize the score.

        Parameters
        ----------
        X : array-like, shape (n_timesteps, n_dims)
            Time series data
        w_values : array-like, optional
            Values of w (window_size) to search over. If None, uses default range.
        q_values : array-like, optional
            Values of q (quantile) to search over. If None, uses default range.
        fixed_w : int, optional
            If provided, w is fixed at this value and only q is optimized
        fixed_q : float, optional
            If provided, q is fixed at this value and only w is optimized
        lagtime : int, default=100
            Lagtime parameter for score calculation

        Returns
        -------
        best_params : dict
            Dictionary with 'window_size' and 'quantile' keys for best parameters
        best_score : float
            Best score achieved
        """

        if fixed_w is not None and fixed_q is not None:
            raise ValueError("Cannot fix both w and q simultaneously")

        if w_values is None:
            w_values = (
                [32, 64, 96, 128, 160, 192, 224, 256] if fixed_w is None else [fixed_w]
            )
        elif fixed_w is not None:
            w_values = [fixed_w]

        if q_values is None:
            q_values = [0.5, 0.6, 0.7, 0.8, 0.9] if fixed_q is None else [fixed_q]
        elif fixed_q is not None:
            q_values = [fixed_q]

        best_score = -np.inf
        best_params = {}

        for w, q in product(w_values, q_values):
            temp_model = SegmentBasedCluster(
                window_size=w,
                quantile=q,
                method=self.method,
                num_clusters=self.num_clusters,
                periodic=self.periodic,
            )

            try:
                score = temp_model.score(X, lagtime=lagtime)
                print(f"Score: {score:.4f}")
                if score > best_score:
                    best_score = score
                    best_params = {"window_size": w, "quantile": q}
            except Exception:
                continue

        if not best_params:
            raise RuntimeError(
                "Grid search failed - no valid parameter combinations found"
            )

        self.window_size = best_params["window_size"]
        self.quantile = best_params["quantile"]
        self.labels_ = None

        return best_params, best_score
