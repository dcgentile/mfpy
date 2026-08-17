"""
Illustrate the change-point pipeline on a toy trajectory: raw trajectory, its metric
derivative (the sliding-window Wasserstein statistic), and the gradient of that
derivative, with the resulting change points marked as dashed red lines on all three
panels.

Addresses reviewer major comment 1(a) ("justify using the gradient of the test
statistic"). The metric derivative gammadot rises into a hump over each ramp; the
change points reported by the method (mfpy.distances.identify_change_points) are the
argmin/argmax of gammadot's gradient within each supra-threshold run, i.e. the points
of steepest ascent/descent of the hump. Plotting all three series stacked makes this
visible directly: the dashed lines sit at the inflection points of gammadot, which is
the standard way to localise the steepest-change region of a smooth ramp (cf. Vogt &
Dette 2015's framing of a gradual change).

Uses w=25 (the best-performing fixed window in the toy sweep) and, by default, the
best empirically-achievable threshold q for each (sigma, tl) pair (read from the
toy_threshold_sweep.py results -- see BEST_EMPIRICAL_Q), rather than the
manuscript's incumbent fixed q=0.95. This is deliberate: these figures motivate the
approach (reviewer comment 1(a)) and should show the method's best foot forward,
not its known failure mode under a mistuned threshold -- that failure mode, and
what q=0.95 costs relative to a tuned q, is exactly the numerical-experiments
material for comment 1(b) instead. Pass ``q=0.95`` explicitly to reproduce the
earlier fixed-threshold figures.

Run across a small (sigma, transition length) grid to show the effect holds from
clean/abrupt to noisy/gradual: sigma in {0, 5, 20} (the three noise levels in the
toy set) crossed with transition length in {5, 20, 50} (short/medium/long ramp).

Each figure is cropped to a handful of consecutive metastable segments rather than
the full ~5000-step trajectory. The earlier all-19-ramps version was too crowded
horizontally to see any one gradient spike clearly; the statistic is computed on
the full trajectory first (so windowing and thresholding are unaffected) and only
the plot's x-limits are cropped afterward.

Each panel also shades the supra-threshold "run" -- the maximal stretch of
consecutive indices where gammadot exceeds its own q-quantile cutoff -- that
produced each pair of change points. This is the set of timestamps
``identify_change_points`` actually takes the argmin/argmax of the gradient over,
so the shading answers "why did the change points land here": they are the
steepest-ascent/steepest-descent points *within this shaded run*, not a global
search over the whole trajectory. The run-detection logic is duplicated locally
(``find_runs`` below) rather than imported, since ``mfpy.distances.identify_change_points``
returns only the final change points and not the intermediate runs.

Output: results/gradient_illustration/figures/gradient_illustration_sigma{S}_tl{L}.{pdf,png}
for each (S, L) in the grid.
"""

import importlib.util
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(REPO_ROOT, "data", "Toy_Model_Trajectories")
OUT_DIR = os.path.join(REPO_ROOT, "results", "gradient_illustration")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import toy_scoring as ts

SIGMAS = (0, 5, 20)
TLS = (5, 20, 50)

# Which ramps to crop around (0-indexed) and how many metastable segments to show.
# Ramp index 2 (not 0) is used so the crop sits well clear of the t=0 edge, where
# the sliding window itself is still filling up. N_SEGMENTS segments span
# N_SEGMENTS-1 ramps: segment 1 = stable region before ramps[RAMP_START], ...,
# last segment = stable region after ramps[RAMP_START + N_SEGMENTS - 2].
RAMP_START = 2
N_SEGMENTS = 6

# Toy-trajectory generator constants (mfpy/experiments/toy_threshold_sweep.py):
# 5000 metastable-state points plus 19 gradual ramps per trajectory.
N_RAMPS = 19
N_POINTS_METASTABLE = 5000


def predicted_optimal_q(tl):
    """
    Closed-form threshold q* = 1 - (fraction of the trajectory spent in
    transition), from results/toy_sweep/FINDINGS.md finding 3. Matches
    ``toy_threshold_sweep.predicted_optimal_q``. Depends only on tl, not sigma --
    used only as a fallback (see BEST_EMPIRICAL_Q below) for (sigma, tl) pairs
    outside the swept grid.
    """
    total = N_POINTS_METASTABLE + N_RAMPS * tl
    return 1.0 - (N_RAMPS * tl) / total


def _load_best_empirical_q():
    """
    The actual best q per (sigma, tl) in our grid, read from
    results/toy_sweep/toy_threshold_sweep.csv (the same 3300-fit sweep behind
    Section 2 of the progress memo) rather than the closed-form q* -- this is the
    true empirical optimum, not an approximation of it, so it is what "best
    possible change points" should mean. Selection: maximize F1(tau=10) first
    (catch all 19 ramps), then minimize median |error| among ties (localize them
    tightly). Falls back to the closed-form q* if the sweep CSV is unavailable.
    """
    csv_path = os.path.join(REPO_ROOT, "results", "toy_sweep", "toy_threshold_sweep.csv")
    if not os.path.exists(csv_path):
        return {}
    import pandas as pd

    df = pd.read_csv(csv_path)
    best = {}
    for (sigma, tl), sub in df.groupby(["sigma", "transition_length"]):
        fmax = sub["f1_tau10"].max()
        tied = sub[sub["f1_tau10"] >= fmax - 1e-9]
        row = tied.loc[tied["median_abs_error"].idxmin()]
        best[(int(sigma), int(tl))] = float(row["q"])
    return best


BEST_EMPIRICAL_Q = _load_best_empirical_q()


# Distinct, print-friendly, high-contrast palette.
COLOR_TRAJ = "#264653"  # dark slate blue
COLOR_GAMMADOT = "#2a9d8f"  # teal
COLOR_GRAD = "#e76f51"  # burnt orange
COLOR_CP = "#e63946"  # vivid crimson (change points)
COLOR_RUN = "#ffb703"  # amber (supra-threshold run shading)


def find_runs(gammadot, q):
    """
    Reproduce the run-detection stage of ``mfpy.distances.identify_change_points``
    (byte-for-byte the same candidate/run logic, including its quirk of dropping a
    run still open at the very end of the candidate list), returning the cutoff and
    the (start, end) index of every maximal supra-threshold run -- the set of
    timestamps each pair of change points is drawn from via argmin/argmax of the
    gradient. The library function only returns the resulting change points, not
    these intermediate runs, so this is kept in sync with it manually.
    """
    cutoff = np.quantile(gammadot, q)
    candidates = np.where(gammadot > cutoff)[0]
    runs = []
    in_sequence = False
    seq_start = 0
    N = candidates.shape[0]
    for n in range(N - 1):
        curr = candidates[n]
        nx = candidates[n + 1]
        if not in_sequence and nx - curr == 1:
            in_sequence = True
            seq_start = curr
        if (in_sequence and nx - curr > 1) or n == N - 1:
            in_sequence = False
            seq_end = curr
            runs.append((int(seq_start), int(seq_end)))
    return cutoff, runs


def _load_distances_module():
    """Import mfpy/distances.py directly, bypassing mfpy/__init__.py (avoids dadapy)."""
    path = os.path.join(REPO_ROOT, "mfpy", "distances.py")
    spec = importlib.util.spec_from_file_location("_mfpy_distances", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.tqdm = lambda it, **kwargs: it  # silence per-window progress bar
    return module


def crop_window(ramps, tl, ramp_start=RAMP_START, n_segments=N_SEGMENTS):
    """
    x-limits spanning exactly ``n_segments`` metastable segments: the stable block
    before ramps[ramp_start], the n_segments-1 ramps ramps[ramp_start .. ramp_start
    + n_segments - 2] with their intervening stable blocks, and the stable block
    after. A small margin (a quarter of the flanking stable segments, capped at
    2*tl) is added on each side so the flat portions read clearly as "stable"
    rather than being cut flush at the ramp edge.
    """
    n_ramps = n_segments - 1
    r0 = ramps[ramp_start]
    r1 = ramps[ramp_start + n_ramps - 1]
    prev_end = ramps[ramp_start - 1, 1] if ramp_start > 0 else 0
    last_idx = ramp_start + n_ramps
    next_start = ramps[last_idx, 0] if last_idx < len(ramps) else r1[1] + 4 * tl

    margin_lo = min(2 * tl, (r0[0] - prev_end) // 3)
    margin_hi = min(2 * tl, (next_start - r1[1]) // 3)
    return int(r0[0] - margin_lo), int(r1[1] + margin_hi)


def make_figure(dist, sigma, tl, w=25, q=None, fig_dir=None):
    """
    q=None (the default) uses the best empirically-achievable threshold for this
    (sigma, tl) -- from BEST_EMPIRICAL_Q if the pair was swept, else the
    closed-form q* -- i.e. the method's best foot forward, appropriate for figures
    that motivate the approach. Pass an explicit q (e.g. 0.95) to illustrate the
    manuscript's fixed incumbent threshold instead, e.g. for numerical-experiments
    figures about what a mistuned q costs.
    """
    tuned = q is None
    if tuned:
        q = BEST_EMPIRICAL_Q.get((sigma, tl), predicted_optimal_q(tl))

    traj_path = os.path.join(DATA_DIR, f"sigma_{sigma}_transition_{tl}.txt")
    gt_path = os.path.join(DATA_DIR, f"sigma_{sigma}_transition_{tl}_GroundTruth.txt")
    X = np.loadtxt(traj_path)
    T = X.shape[0]
    ramps = ts.load_ground_truth(gt_path, transition_length=tl)

    gammadot = dist.compute_metric_derivative_1d(X, w)
    grad = np.gradient(gammadot)
    changes = dist.identify_change_points(gammadot, q)
    interior = changes[(changes > 0) & (changes < T - 1)]  # drop fixed endpoints
    cutoff, runs = find_runs(gammadot, q)

    lo, hi = crop_window(ramps, tl)
    t = np.arange(lo, hi)
    cps_in_view = interior[(interior >= lo) & (interior <= hi)]
    runs_in_view = [(s, e) for s, e in runs if e >= lo and s <= hi]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(3, 1, figsize=(12, 8.5), sharex=True)
    panels = [
        (X, r"trajectory  $X_t$", COLOR_TRAJ),
        (gammadot, r"metric derivative  $\dot\gamma(t)$", COLOR_GAMMADOT),
        (grad, r"gradient of metric derivative  $\nabla\dot\gamma(t)$", COLOR_GRAD),
    ]
    for ax, (series, label, color) in zip(axes, panels):
        for j, (s, e) in enumerate(runs_in_view):
            ax.axvspan(
                max(s, lo), min(e, hi),
                color=COLOR_RUN, alpha=0.28, lw=0, zorder=0,
                label="supra-threshold run\n(argmin/argmax taken over this)"
                if (ax is axes[0] and j == 0) else None,
            )
        ax.plot(t, series[lo:hi], color=color, linewidth=1.4, zorder=3)
        for i, cp in enumerate(cps_in_view):
            ax.axvline(
                cp,
                color=COLOR_CP,
                linestyle="--",
                linewidth=1.4,
                alpha=0.85,
                zorder=2,
                label="change point" if (ax is axes[0] and i == 0) else None,
            )
        ax.set_ylabel(label)
        ax.set_xlim(lo, hi)
        ax.margins(x=0)
        ax.set_facecolor("#f7f7f5")

    q_label = r"best\ q" if tuned else "q"
    cutoff_line = axes[1].axhline(
        cutoff, color="#555555", linestyle=":", linewidth=1.6, zorder=1,
        label=rf"${q_label}={q:.3f}$ cutoff ({cutoff:.2f})",
    )

    # Single consolidated legend (rather than one per panel) for legibility, placed
    # above the axes so it never sits on top of the data.
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(cutoff_line)
    labels.append(cutoff_line.get_label())
    fig.legend(
        handles, labels,
        loc="upper center", ncol=3, frameon=True, fontsize=12,
        framealpha=0.95, facecolor="white", edgecolor="#888888",
        handlelength=1.8, columnspacing=1.6, bbox_to_anchor=(0.5, 1.06),
    )
    axes[-1].set_xlabel("t")
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    fig_dir = fig_dir or os.path.join(OUT_DIR, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    stem = f"gradient_illustration_sigma{sigma}_tl{tl}"
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(fig_dir, f"{stem}.{ext}"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"trajectory: {traj_path}")
    print(f"T={T}, w={w}, q={q:.3f} ({'tuned q*' if tuned else 'fixed'}), "
          f"cropped to [{lo}, {hi}]")
    print(f"{len(cps_in_view)} change points in view (of {len(interior)} total): "
          f"{cps_in_view.tolist()}")
    return interior


def main(sigmas=SIGMAS, tls=TLS, w=25, q=None):
    dist = _load_distances_module()
    fig_dir = os.path.join(OUT_DIR, "figures")
    results = {}
    for sigma in sigmas:
        for tl in tls:
            print(f"\n--- sigma={sigma}, tl={tl} ---")
            results[(sigma, tl)] = make_figure(dist, sigma, tl, w=w, q=q, fig_dir=fig_dir)
    return results


if __name__ == "__main__":
    main()
