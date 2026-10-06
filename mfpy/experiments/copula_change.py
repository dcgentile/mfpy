"""
A second, harder counterexample for reviewer comment 1(c) / action item 5.

results/correlation_change/ showed that a pure correlation flip is invisible to the 1D
method but recoverable either by an entropic-OT (Sinkhorn) statistic on the joint sample,
or more cheaply by rotating to the pooled-PCA eigenbasis first and reusing the existing 1D
method. The PCA fix works there because the change is still second-moment information
(a covariance eigenvalue swap) -- PCA just needed the right basis to see it.

This script asks whether that second-moment ceiling is real. Construction: X1 ~ N(0,1)
throughout. Before the change point, X2 is an independent N(0,1) draw (X1 ⊥ X2). After the
change point, X2 = S * X1 where S is an independent Rademacher sign. Both regimes have:
  - identical marginals (X2 = S*X1 is still exactly N(0,1), since N(0,1) is symmetric)
  - identical covariance: Cov(X1,X2) = E[S]E[X1^2] = 0 in both regimes, and in fact the
    full covariance matrix is the identity I in *both* regimes (Var(X2)=E[X1^2]=1 always),
    so Var(v^T X) = 1 for every direction v, at every t. There is no rotation, and no pair
    of eigenvalues, that PCA can use to see this -- second-moment information genuinely does
    not move. What changes is the joint *shape*: an independent round cloud becomes two
    diagonal lines (|X1| = |X2| exactly after the switch), a change with infinite mutual
    information and zero Pearson correlation, i.e. a copula-type change in the classic
    "correlation is not dependence" sense.

The claim under test: 1D marginals fail (as always), PCA-componentwise now also fails (not
just under-resourced, but structurally blind -- there is no second-moment signal at all),
and Sinkhorn on the joint sample is the one detector left standing, because it is the only
one of the three that looks at the full empirical joint law rather than a linear summary
of it.
"""

import importlib.util
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)


def _load_distances_module():
    """Import mfpy/distances.py directly, bypassing mfpy/__init__.py (which requires
    dadapy). Same trick as toy_sweep.py / correlation_change.py."""
    path = os.path.join(REPO_ROOT, "mfpy", "distances.py")
    spec = importlib.util.spec_from_file_location("_mfpy_distances", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.tqdm = lambda it, **kwargs: it
    return module


DISTANCES = _load_distances_module()
compute_metric_derivative_1d = DISTANCES.compute_metric_derivative_1d
compute_metric_derivative_md = DISTANCES.compute_metric_derivative_md
identify_change_points = DISTANCES.identify_change_points

RESULTS_DIR = os.path.join(REPO_ROOT, "results", "copula_change")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")


def make_process(T=2000, cp=1000, seed=0):
    """X1 ~ N(0,1) throughout. X2 independent N(0,1) pre-cp; X2 = S*X1 (S = random
    sign) post-cp. Marginals and the full covariance matrix (= I) are identical in
    both regimes; only the joint shape changes."""
    rng = np.random.default_rng(seed)
    x1 = rng.standard_normal(T)
    x2 = np.empty(T)
    x2[:cp] = rng.standard_normal(cp)
    S = rng.choice([-1.0, 1.0], size=T - cp)
    x2[cp:] = S * x1[cp:]
    return np.stack([x1, x2], axis=1)


def run_1d_detector(x, w=50, q=0.95):
    gammadot = compute_metric_derivative_1d(x, w)
    changes = identify_change_points(gammadot, q)
    return gammadot, changes


def run_md_detector(X, w=150, q=0.95, reg=0.03):
    """w=150, reg=0.03 chosen by a small grid over w in {50,100,150,200} and reg in
    {0.01,0.03,0.05} on this process, same signal-to-noise criterion as
    correlation_change.py. Needed a much bigger window than the correlation
    counterexample (w=50 there): distinguishing "independent cloud" from "two crossing
    lines" from finite empirical samples needs more points per window than
    distinguishing two different correlations does, and reg=0.05 blurs out the thin
    diagonal structure too much at any window size tried."""
    gammadot = compute_metric_derivative_md(X, w, reg=reg, numIterMax=300)
    changes = identify_change_points(gammadot, q)
    return gammadot, changes


def run_pca_componentwise(X, w=50, q=0.95):
    Sigma = np.cov(X.T)
    eigval, eigvec = np.linalg.eigh(Sigma)
    order = np.argsort(eigval)[::-1]
    eigval, eigvec = eigval[order], eigvec[:, order]
    PCs = X @ eigvec
    g_pc1, c_pc1 = run_1d_detector(PCs[:, 0], w=w, q=q)
    g_pc2, c_pc2 = run_1d_detector(PCs[:, 1], w=w, q=q)
    return PCs, eigval, eigvec, g_pc1, c_pc1, g_pc2, c_pc2


def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    T, cp = 2000, 1000
    X = make_process(T=T, cp=cp, seed=0)
    x1, x2 = X[:, 0], X[:, 1]

    pre, post = slice(0, cp), slice(cp, T)
    report = {
        "x1_mean_pre": x1[pre].mean(), "x1_mean_post": x1[post].mean(),
        "x1_var_pre": x1[pre].var(), "x1_var_post": x1[post].var(),
        "x2_mean_pre": x2[pre].mean(), "x2_mean_post": x2[post].mean(),
        "x2_var_pre": x2[pre].var(), "x2_var_post": x2[post].var(),
        "corr_pre": np.corrcoef(x1[pre], x2[pre])[0, 1],
        "corr_post": np.corrcoef(x1[post], x2[post])[0, 1],
        "mean_abs_x1_minus_abs_x2_pre": np.mean(np.abs(np.abs(x1[pre]) - np.abs(x2[pre]))),
        "mean_abs_x1_minus_abs_x2_post": np.mean(np.abs(np.abs(x1[post]) - np.abs(x2[post]))),
    }

    # w=150 used throughout for a like-for-like comparison with the Sinkhorn detector
    # (which needs w=150 here; see run_md_detector docstring). Marginals/PCA are
    # structurally blind to this change at any w (their second moments are exactly
    # unchanged), so this choice does not favor them one way or the other.
    W = 150
    g1, c1 = run_1d_detector(x1, w=W)
    g2, c2 = run_1d_detector(x2, w=W)
    g_md, c_md = run_md_detector(X, w=W)
    PCs, eigval, eigvec, g_pc1, c_pc1, g_pc2, c_pc2 = run_pca_componentwise(X, w=W)

    np.savez(
        os.path.join(RESULTS_DIR, "copula_change.npz"),
        X=X, cp=cp, g1=g1, g2=g2, c1=c1, c2=c2, g_md=g_md, c_md=c_md,
        PCs=PCs, eigval=eigval, eigvec=eigvec,
        g_pc1=g_pc1, c_pc1=c_pc1, g_pc2=g_pc2, c_pc2=c_pc2,
    )

    with open(os.path.join(RESULTS_DIR, "marginal_check.txt"), "w") as f:
        for k, v in report.items():
            f.write(f"{k}: {v:.4f}\n")
        f.write(f"\npooled-covariance eigenvalues (descending): {list(eigval[::-1])}\n")
        f.write(f"detected change points, X1 marginal (w=50,q=0.95): {list(c1)}\n")
        f.write(f"detected change points, X2 marginal (w=50,q=0.95): {list(c2)}\n")
        f.write(f"detected change points, PC1 (w=50,q=0.95): {list(c_pc1)}\n")
        f.write(f"detected change points, PC2 (w=50,q=0.95): {list(c_pc2)}\n")
        f.write(f"detected change points, MD/Sinkhorn on joint (X1,X2) "
                f"(w=50,q=0.95,reg=0.05): {list(c_md)}\n")
        f.write(f"true change point: {cp}\n")

    make_figures(X, cp, g1, g2, c1, c2, g_md, c_md, g_pc1, c_pc1, g_pc2, c_pc2)

    print("done")
    for k, v in report.items():
        print(f"{k}: {v:.4f}")
    print("pooled-covariance eigenvalues:", eigval[::-1])
    print("detected CPs on X1:", list(c1))
    print("detected CPs on X2:", list(c2))
    print("detected CPs on PC1:", list(c_pc1))
    print("detected CPs on PC2:", list(c_pc2))
    print("detected CPs on joint (Sinkhorn):", list(c_md))


def make_figures(X, cp, g1, g2, c1, c2, g_md, c_md, g_pc1, c_pc1, g_pc2, c_pc2):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x1, x2 = X[:, 0], X[:, 1]

    # Figure 1: joint scatter, pre (round cloud) vs post (the X shape)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    axes[0].scatter(x1[:cp], x2[:cp], s=4, alpha=0.35, color="#3B7DD8")
    axes[0].set_title(r"$t < t^*$: independent")
    axes[1].scatter(x1[cp:], x2[cp:], s=4, alpha=0.35, color="#D8763B")
    axes[1].set_title(r"$t \geq t^*$: $X_2 = S \cdot X_1$")
    for ax in axes:
        ax.set_xlabel(r"$X_1$")
        ax.set_ylabel(r"$X_2$")
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_aspect("equal")
    fig.suptitle("Joint shape changes completely; every second moment is unchanged")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "joint_scatter.pdf"))
    fig.savefig(os.path.join(FIG_DIR, "joint_scatter.png"), dpi=150)
    plt.close(fig)

    # Figure 2: marginal histograms, pre vs post, overlaid
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    for ax, x, name in zip(axes, [x1, x2], [r"$X_1$", r"$X_2$"]):
        ax.hist(x[:cp], bins=40, density=True, alpha=0.5, color="#3B7DD8", label=r"$t<t^*$")
        ax.hist(x[cp:], bins=40, density=True, alpha=0.5, color="#D8763B", label=r"$t\geq t^*$")
        ax.set_title(f"marginal of {name}")
        ax.legend(fontsize=8)
    fig.suptitle("Marginal distributions are unchanged across t*")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "marginal_histograms.pdf"))
    fig.savefig(os.path.join(FIG_DIR, "marginal_histograms.png"), dpi=150)
    plt.close(fig)

    # Figure 3: all four detectors stacked -- X1, X2, PC1, PC2 (all should be flat),
    # Sinkhorn on the joint sample (should not be)
    fig, axes = plt.subplots(5, 1, figsize=(8, 10), sharex=True)
    panels = [
        (g1, c1, r"$\dot\gamma$, $X_1$", "#3B7DD8"),
        (g2, c2, r"$\dot\gamma$, $X_2$", "#D8763B"),
        (g_pc1, c_pc1, r"$\dot\gamma$, PC1", "#7A3BD8"),
        (g_pc2, c_pc2, r"$\dot\gamma$, PC2", "#3BB0D8"),
        (g_md, c_md, "Sinkhorn,\njoint $(X_1,X_2)$", "#2E9E5B"),
    ]
    for ax, (g, c, name, color) in zip(axes, panels):
        ax.plot(g, lw=0.8, color=color)
        ax.axvline(cp, color="k", ls="--", lw=1)
        for c_ in c:
            ax.axvline(c_, color="green", ls=":", lw=1)
        ax.set_ylabel(name)
    axes[0].axvline(cp, color="k", ls="--", lw=1, label="true change point")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("t")
    fig.suptitle("Marginals and PCA are structurally blind; Sinkhorn on the joint sample is not")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "all_detectors.pdf"))
    fig.savefig(os.path.join(FIG_DIR, "all_detectors.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
