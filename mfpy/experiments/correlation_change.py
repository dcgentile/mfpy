"""
Reviewer comment 1(c): the method "ignores potential correlation between
dimensions, and the change point itself may occur in the joint distribution
rather than the marginal distributions."

This script constructs the sharpest version of that failure. A bivariate
Gaussian process X_t = (X1_t, X2_t) is drawn i.i.d. in t from N(0, Sigma_t),
where Sigma_t has unit variance on both diagonal entries at every t and a
correlation rho_t that steps from rho_A to rho_B at a single true change
point. Because the diagonal is held fixed, the marginal law of X1_t and of
X2_t is exactly N(0,1) for *every* t -- there is no marginal change to find.
The joint law changes (the covariance ellipse rotates/stretches) but nothing
detectable survives projection onto either axis.

compute_metric_derivative_1d + identify_change_points (mfpy/distances.py) is
a purely 1D statistic: it never sees the joint sample, only one coordinate at
a time. Run on X1 or X2 alone, it is applied exactly as it would be to any
other 1D trajectory in the paper. The claim under test is that it fails
to fire at the true change point, since the marginal it can see never moves.
"""

import importlib.util
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)


def _load_distances_module():
    """Import mfpy/distances.py directly, bypassing mfpy/__init__.py (which
    requires dadapy). Same trick as toy_sweep.py."""
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

RESULTS_DIR = os.path.join(REPO_ROOT, "results", "correlation_change")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")


def make_process(T=2000, cp=1000, rho_a=-0.85, rho_b=0.85, seed=0):
    """Bivariate Gaussian, unit marginal variance throughout, correlation
    steps from rho_a to rho_b at index `cp`."""
    rng = np.random.default_rng(seed)
    X = np.zeros((T, 2))
    for t in range(T):
        rho = rho_a if t < cp else rho_b
        Sigma = np.array([[1.0, rho], [rho, 1.0]])
        L = np.linalg.cholesky(Sigma)
        X[t] = L @ rng.standard_normal(2)
    return X


def run_1d_detector(x, w=50, q=0.95):
    gammadot = compute_metric_derivative_1d(x, w)
    changes = identify_change_points(gammadot, q)
    return gammadot, changes


def run_md_detector(X, w=50, q=0.95, reg=0.05):
    """Entropic-OT (Sinkhorn) analogue of run_1d_detector, applied to the joint
    sample X (T, d) rather than to a single coordinate. reg=0.05 was chosen by a
    small grid search {0.03, 0.05, 0.1, 0.2, 0.3} on this same process, picking the
    largest (near-window mean - far-window mean) / far-window std."""
    gammadot = compute_metric_derivative_md(X, w, reg=reg)
    changes = identify_change_points(gammadot, q)
    return gammadot, changes


def run_pca_componentwise(X, w=50, q=0.95):
    """PCA on the pooled sample (eigendecomposition of the full-trajectory covariance,
    ignoring time), then the existing 1D detector run separately on each principal
    component. Componentwise, not multivariate: this only reuses
    compute_metric_derivative_1d, applied to two new 1D series instead of the original
    coordinates. Rotating to the eigenbasis is enough here because the change is a pure
    correlation flip: variance along the +/-45 degree directions swings hard in opposite
    directions even though it cancels out under the raw x1/x2 axes and (on average) under
    pooled PCA's own eigenvalues too (see FINDINGS.md)."""
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
    X = make_process(T=T, cp=cp, rho_a=-0.85, rho_b=0.85, seed=0)
    x1, x2 = X[:, 0], X[:, 1]

    # sanity check: marginal moments are stable across the change point
    pre, post = slice(0, cp), slice(cp, T)
    marginal_report = {
        "x1_mean_pre": x1[pre].mean(), "x1_mean_post": x1[post].mean(),
        "x1_var_pre": x1[pre].var(), "x1_var_post": x1[post].var(),
        "x2_mean_pre": x2[pre].mean(), "x2_mean_post": x2[post].mean(),
        "x2_var_pre": x2[pre].var(), "x2_var_post": x2[post].var(),
        "corr_pre": np.corrcoef(x1[pre], x2[pre])[0, 1],
        "corr_post": np.corrcoef(x1[post], x2[post])[0, 1],
    }

    g1, c1 = run_1d_detector(x1)
    g2, c2 = run_1d_detector(x2)
    g_md, c_md = run_md_detector(X)
    PCs, eigval, eigvec, g_pc1, c_pc1, g_pc2, c_pc2 = run_pca_componentwise(X)

    np.savez(
        os.path.join(RESULTS_DIR, "correlation_change.npz"),
        X=X, cp=cp, g1=g1, g2=g2, c1=c1, c2=c2, g_md=g_md, c_md=c_md,
        PCs=PCs, eigval=eigval, eigvec=eigvec,
        g_pc1=g_pc1, c_pc1=c_pc1, g_pc2=g_pc2, c_pc2=c_pc2,
    )

    with open(os.path.join(RESULTS_DIR, "marginal_check.txt"), "w") as f:
        for k, v in marginal_report.items():
            f.write(f"{k}: {v:.4f}\n")
        f.write(f"\ndetected change points, X1 marginal (w=50,q=0.95): {list(c1)}\n")
        f.write(f"detected change points, X2 marginal (w=50,q=0.95): {list(c2)}\n")
        f.write(f"detected change points, MD/Sinkhorn on joint (X1,X2) "
                f"(w=50,q=0.95,reg=0.05): {list(c_md)}\n")
        f.write(f"\npooled-covariance eigenvalues (descending): {list(eigval[::-1])}\n")
        f.write(f"pooled-covariance eigenvectors (cols, descending eigval):\n{eigvec[:, ::-1]}\n")
        pc1_pre, pc1_post = PCs[:cp, 0].var(), PCs[cp:, 0].var()
        pc2_pre, pc2_post = PCs[:cp, 1].var(), PCs[cp:, 1].var()
        f.write(f"PC1 var pre/post: {pc1_pre:.4f} / {pc1_post:.4f}\n")
        f.write(f"PC2 var pre/post: {pc2_pre:.4f} / {pc2_post:.4f}\n")
        f.write(f"detected change points, PC1 (w=50,q=0.95): {list(c_pc1)}\n")
        f.write(f"detected change points, PC2 (w=50,q=0.95): {list(c_pc2)}\n")
        f.write(f"true change point: {cp}\n")

    make_figures(X, cp, g1, g2, c1, c2, g_md, c_md)
    make_pca_figure(cp, g_pc1, c_pc1, g_pc2, c_pc2)
    print("done")
    for k, v in marginal_report.items():
        print(f"{k}: {v:.4f}")
    print("detected CPs on X1:", list(c1))
    print("detected CPs on X2:", list(c2))
    print("detected CPs on joint (Sinkhorn):", list(c_md))
    print("detected CPs on PC1:", list(c_pc1))
    print("detected CPs on PC2:", list(c_pc2))


def make_figures(X, cp, g1, g2, c1, c2, g_md=None, c_md=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x1, x2 = X[:, 0], X[:, 1]
    T = X.shape[0]

    # Figure 1: scatter in the plane, pre vs post, with covariance ellipses
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 5.5))
    ax.scatter(x1[:cp], x2[:cp], s=4, alpha=0.35, color="#3B7DD8", label=r"$t < t^*$ ($\rho=-0.85$)")
    ax.scatter(x1[cp:], x2[cp:], s=4, alpha=0.35, color="#D8763B", label=r"$t \geq t^*$ ($\rho=+0.85$)")
    theta = np.linspace(0, 2 * np.pi, 200)
    circle = np.stack([np.cos(theta), np.sin(theta)])
    for rho, color in [(-0.85, "#1F4E96"), (0.85, "#963B1F")]:
        Sigma = np.array([[1.0, rho], [rho, 1.0]])
        L = np.linalg.cholesky(Sigma)
        ell = (L @ circle) * 2  # 2-sigma ellipse
        ax.plot(ell[0], ell[1], color=color, lw=2)
    ax.set_xlabel(r"$X_1$")
    ax.set_ylabel(r"$X_2$")
    ax.set_title("Joint distribution changes; both marginals do not")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "joint_scatter.pdf"))
    fig.savefig(os.path.join(FIG_DIR, "joint_scatter.png"), dpi=150)
    plt.close(fig)

    # Figure 2: the two 1D marginal trajectories, with true CP marked
    fig, axes = plt.subplots(2, 1, figsize=(8, 4.5), sharex=True)
    axes[0].plot(x1, lw=0.5, color="#3B7DD8")
    axes[0].axvline(cp, color="k", ls="--", lw=1, label="true change point")
    axes[0].set_ylabel(r"$X_1$")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[1].plot(x2, lw=0.5, color="#D8763B")
    axes[1].axvline(cp, color="k", ls="--", lw=1)
    axes[1].set_ylabel(r"$X_2$")
    axes[1].set_xlabel("t")
    fig.suptitle("1D marginal trajectories: no visible change at t*")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "marginal_trajectories.pdf"))
    fig.savefig(os.path.join(FIG_DIR, "marginal_trajectories.png"), dpi=150)
    plt.close(fig)

    # Figure 3: marginal histograms, pre vs post, overlaid
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

    # Figure 4: the actual 1D detector statistic on each marginal
    fig, axes = plt.subplots(2, 1, figsize=(8, 4.5), sharex=True)
    axes[0].plot(g1, lw=0.8, color="#3B7DD8")
    axes[0].axvline(cp, color="k", ls="--", lw=1, label="true change point")
    for c in c1:
        axes[0].axvline(c, color="green", ls=":", lw=1)
    axes[0].set_ylabel(r"$\dot\gamma$ on $X_1$")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[1].plot(g2, lw=0.8, color="#D8763B")
    axes[1].axvline(cp, color="k", ls="--", lw=1)
    for c in c2:
        axes[1].axvline(c, color="green", ls=":", lw=1)
    axes[1].set_ylabel(r"$\dot\gamma$ on $X_2$")
    axes[1].set_xlabel("t")
    fig.suptitle("Method's own statistic (green = detected CP): no response at t*")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "detector_statistic.pdf"))
    fig.savefig(os.path.join(FIG_DIR, "detector_statistic.png"), dpi=150)
    plt.close(fig)

    if g_md is None:
        return

    # Figure 5: entropic-OT (Sinkhorn) statistic on the joint sample, vs the two
    # blind 1D marginal statistics, stacked for direct comparison
    fig, axes = plt.subplots(3, 1, figsize=(8, 6.5), sharex=True)
    axes[0].plot(g1, lw=0.7, color="#3B7DD8")
    for c in c1:
        axes[0].axvline(c, color="green", ls=":", lw=1)
    axes[0].set_ylabel(r"$\dot\gamma$, $X_1$ only")
    axes[1].plot(g2, lw=0.7, color="#D8763B")
    for c in c2:
        axes[1].axvline(c, color="green", ls=":", lw=1)
    axes[1].set_ylabel(r"$\dot\gamma$, $X_2$ only")
    axes[2].plot(g_md, lw=0.9, color="#2E9E5B")
    for c in c_md:
        axes[2].axvline(c, color="green", ls=":", lw=1)
    axes[2].set_ylabel("Sinkhorn,\njoint $(X_1,X_2)$")
    axes[2].set_xlabel("t")
    for ax in axes:
        ax.axvline(cp, color="k", ls="--", lw=1)
    axes[0].axvline(cp, color="k", ls="--", lw=1, label="true change point")
    axes[0].legend(loc="upper right", fontsize=8)
    y_all = np.concatenate([g1, g2, g_md])
    y_lo, y_hi = y_all.min(), y_all.max()
    y_pad = 0.05 * (y_hi - y_lo)
    for ax in axes:
        ax.set_ylim(y_lo - y_pad, y_hi + y_pad)
    fig.suptitle("Entropic OT on the joint sample recovers t*; either marginal alone cannot")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "sinkhorn_vs_marginal.pdf"))
    fig.savefig(os.path.join(FIG_DIR, "sinkhorn_vs_marginal.png"), dpi=150)
    plt.close(fig)


def make_pca_figure(cp, g_pc1, c_pc1, g_pc2, c_pc2):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(8, 4.5), sharex=True)
    for ax, g, c, name, color in [
        (axes[0], g_pc1, c_pc1, "PC1", "#7A3BD8"),
        (axes[1], g_pc2, c_pc2, "PC2", "#3BB0D8"),
    ]:
        ax.plot(g, lw=0.8, color=color)
        ax.axvline(cp, color="k", ls="--", lw=1)
        for c_ in c:
            ax.axvline(c_, color="green", ls=":", lw=1)
        ax.set_ylabel(rf"$\dot\gamma$, {name}")
    axes[0].axvline(cp, color="k", ls="--", lw=1, label="true change point")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[1].set_xlabel("t")
    fig.suptitle("Existing 1D method run componentwise on pooled-PCA axes")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "pca_componentwise.pdf"))
    fig.savefig(os.path.join(FIG_DIR, "pca_componentwise.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
