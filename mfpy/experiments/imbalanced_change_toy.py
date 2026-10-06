"""
The reviewer's own motivating case for resampling (TODO item B4/4): imbalanced change
sizes.

Reviewer major comment 1(b), verbatim: "This will be particularly apparent when the
change sizes themselves are imbalanced: for example, if there are 2 (instantaneous)
changes in the mean, and one is much larger than the other, the quantile-based threshold
will be set too high to detect the smaller change."

This has never been tested directly (`results/resampling_threshold/TODO.md` item 4,
`paper-resources/CONTINUE-PROMPT.md` next task 1). Everything so far -- `toy_sweep`,
`resampling_threshold`, `qw_interaction`, `block_permutation_null` -- uses trajectories
where every change point has the same size. This module builds trajectories where it does
not, and asks the direct question: does the fixed quantile threshold miss the small change,
and does the resampling threshold recover it?

Design
------
Each trajectory returns to a common baseline Laplace state A repeatedly, taking two kinds
of instantaneous excursion in strict alternation: a *small* excursion to state S (mean
shifted by `delta`) and a *big* excursion to state B (mean shifted by `ratio * delta`),
same Laplace scale throughout. One (small, big) pair per cycle, `n_cycles` cycles, giving
`4 * n_cycles` segments and `4 * n_cycles - 1` true change points -- enough for ADP
clustering (needs >=5 distinct segments) while keeping the ground truth exactly the
reviewer's example, repeated for statistical power across `reps` independent draws.

`ratio = 1` is the calibration case (no imbalance -- both thresholds should look similar);
`ratio = 3, 10` are the reviewer's "much larger" case.

Two threshold rules are compared, both at the manuscript's `w = 25`:

``baseline``            fixed q = 0.95 quantile of the observed statistic (manuscript
                         default).
``resample_pointwise``  q = 0.5 quantile of the pointwise permutation null, R = 199
                         (`results/resampling_threshold/FINDINGS.md` recommendation).

Everything downstream -- run detection, gradient boundary extraction, ADP segment
clustering -- is `vmeasure.cluster_trajectory`, unmodified, so the only thing that differs
between rows is the source of the cutoff.

Usage
-----
    python mfpy/experiments/imbalanced_change_toy.py [--reps 20] [--R 199]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, HERE)

from toy_scoring import match_change_points  # noqa: E402
from vmeasure import cluster_trajectory, score  # noqa: E402

RESULT_DIR = os.path.join(REPO_ROOT, "results", "imbalanced_change_toy")

BASE_MU = 100.0
BASE_B = 15.0  # Laplace scale, shared by all three states
DELTA = 12.0  # size of the "small" excursion
RATIOS = (1, 3, 10)  # big excursion = ratio * DELTA
N_CYCLES = 8
BLOCK_LEN = 200  # samples per segment
W = 25
Q_BASELINE = 0.95
Q_NULL = 0.5
TAU = 10  # matching-tolerance for scoring, instantaneous changes

# point-label classes for V-measure
BASELINE, SMALL, BIG = 0, 1, 2


def generate(ratio, delta=DELTA, base_mu=BASE_MU, base_b=BASE_B,
             n_cycles=N_CYCLES, block_len=BLOCK_LEN, rng=None):
    """
    One trajectory: n_cycles repeats of (baseline, small excursion, baseline, big
    excursion), all instantaneous transitions, same Laplace scale throughout.

    Returns (x, change_points, point_labels, small_cps, big_cps). `small_cps` / `big_cps`
    are the change points that *open* each excursion (the ones a biased threshold is
    expected to selectively miss); every excursion also closes with a return to baseline,
    which is scored too but is not the reviewer's specific claim.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    big_delta = ratio * delta
    pieces, labels_pieces, cps = [], [], []
    small_open_cps, big_open_cps = [], []
    cursor = 0

    for _ in range(n_cycles):
        # baseline
        pieces.append(rng.laplace(base_mu, base_b, block_len))
        labels_pieces.append(np.full(block_len, BASELINE))
        cursor += block_len
        cps.append(cursor)
        small_open_cps.append(cursor)

        # small excursion
        pieces.append(rng.laplace(base_mu + delta, base_b, block_len))
        labels_pieces.append(np.full(block_len, SMALL))
        cursor += block_len
        cps.append(cursor)

        # back to baseline
        pieces.append(rng.laplace(base_mu, base_b, block_len))
        labels_pieces.append(np.full(block_len, BASELINE))
        cursor += block_len
        cps.append(cursor)
        big_open_cps.append(cursor)

        # big excursion
        pieces.append(rng.laplace(base_mu + big_delta, base_b, block_len))
        labels_pieces.append(np.full(block_len, BIG))
        cursor += block_len
        cps.append(cursor)

    # trim the trailing change point back to baseline (no more data follows it in the
    # cycle loop as written -- the loop always closes on an excursion) by appending one
    # final baseline block so every excursion both opens and closes inside the data.
    pieces.append(rng.laplace(base_mu, base_b, block_len))
    labels_pieces.append(np.full(block_len, BASELINE))
    cursor += block_len

    x = np.concatenate(pieces)
    point_labels = np.concatenate(labels_pieces)
    # cps already includes every transition boundary, including the final excursion's
    # return-to-baseline (added when that block was appended above).
    change_points = np.array(sorted(set(cps) | {cursor - block_len}), dtype=int)
    return x, change_points, point_labels, np.array(small_open_cps), np.array(big_open_cps)


def small_recall(predicted, small_cps, tau=TAU):
    predicted = np.asarray(predicted)
    pairs = match_change_points(predicted, small_cps, tau)
    return len(pairs) / len(small_cps) if len(small_cps) else np.nan


def overall_precision(predicted, truth, tau=TAU):
    """Fraction of predicted change points matched to *any* true change point.

    Reported alongside recall because a threshold that over-triggers can post a high
    recall while flooding the trajectory with false positives -- recall alone would
    misrepresent that as unqualified success.
    """
    predicted = np.asarray(predicted)
    if predicted.size == 0:
        return np.nan
    pairs = match_change_points(predicted, truth, tau)
    return len(pairs) / len(predicted)


def run(reps, R, seed=0):
    rows = []
    example_trajectories = {}  # (ratio, rep=0) kept for figures
    t0 = time.time()

    for ratio in RATIOS:
        for rep in range(reps):
            rng = np.random.default_rng(seed + 1000 * ratio + rep)
            x, truth_cps, point_labels, small_cps, big_cps = generate(ratio, rng=rng)
            T = x.shape[0]

            for method in ("baseline", "resample_pointwise"):
                threshold = "quantile" if method == "baseline" else "resample_pointwise"
                q = Q_BASELINE if method == "baseline" else Q_NULL
                pred, changes, n_distinct = cluster_trajectory(
                    x, W, q, threshold=threshold, R=R, seed=seed + rep
                )
                changes_interior = changes[(changes > 0) & (changes < T)]

                v_h, v_c, v_v = score(point_labels, pred)

                row = {
                    "ratio": ratio,
                    "rep": rep,
                    "method": method,
                    "n_true_cps": len(truth_cps),
                    "n_predicted_cps": len(changes_interior),
                    "n_distinct_segments": n_distinct,
                    "small_recall": small_recall(changes_interior, small_cps),
                    "big_recall": small_recall(changes_interior, big_cps),
                    "all_recall": small_recall(changes_interior, truth_cps),
                    "precision": overall_precision(changes_interior, truth_cps),
                    "v_homogeneity": v_h,
                    "v_completeness": v_c,
                    "v_measure": v_v,
                }
                rows.append(row)

                example_trajectories[(ratio, method, rep)] = (
                    x, changes, pred, truth_cps, small_cps, big_cps
                )

        print(f"ratio={ratio} done ({time.time() - t0:.0f}s)", flush=True)

    return pd.DataFrame(rows), example_trajectories


def pick_representative_rep(df, ratio):
    """
    The rep whose *baseline* small-change recall is closest to that ratio's mean.

    Picking rep 0 unconditionally risks illustrating a favourable or unfavourable outlier
    rather than the typical case (checked: rep 0 at ratio=3 detects the small excursions
    cleanly under the baseline, which is not what happens on average -- mean small_recall
    there is 0.081). Anchoring on the baseline's recall, and reusing the same rep for both
    panels, keeps the figure honest and keeps the two panels on the same trajectory.
    """
    sub = df[(df["ratio"] == ratio) & (df["method"] == "baseline")]
    target = sub["small_recall"].mean()
    return int(sub.iloc[(sub["small_recall"] - target).abs().argsort().iloc[0]]["rep"])


def _classify_predicted(predicted, small_cps, big_cps, tau=TAU):
    """
    Split predicted change points into three categories for plotting:
    matched to a true small-excursion onset, matched to a true big-excursion onset (both
    via the same one-to-one Hungarian matching used for scoring), or unmatched (a false
    positive against both). Every true onset is either a "small" or "big" open -- offsets
    (closes) are not distinguished here, only onsets, matching what `small_recall` scores.
    """
    predicted = np.asarray(predicted)
    hit_small = set(int(predicted[i]) for i, _ in match_change_points(predicted, small_cps, tau))
    hit_big = set(int(predicted[i]) for i, _ in match_change_points(predicted, big_cps, tau))
    is_small = np.array([p in hit_small for p in predicted])
    is_big = np.array([p in hit_big for p in predicted])
    is_miss = ~(is_small | is_big)
    return is_small, is_big, is_miss


def plot_examples(df, example_trajectories, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = np.array(["tab:blue", "tab:orange", "tab:red", "tab:green",
                       "tab:purple", "tab:brown", "tab:pink", "tab:olive"])

    for ratio in RATIOS:
        rep = pick_representative_rep(df, ratio)
        fig, axes = plt.subplots(2, 1, figsize=(13, 6.5), sharex=True, sharey=True)
        for ax, method in zip(axes, ("baseline", "resample_pointwise")):
            x, changes, pred, truth_cps, small_cps, big_cps = example_trajectories[
                (ratio, method, rep)
            ]
            predicted_interior = changes[(changes > 0) & (changes < x.shape[0])]
            is_small, is_big, is_miss = _classify_predicted(
                predicted_interior, small_cps, big_cps
            )

            # background rug: every *predicted* change point, colour-coded by whether it
            # matched a true small onset, a true big onset, or neither (false positive).
            # Thin + semi-transparent so density reads as shading rather than a solid wall
            # when there are hundreds of them (resample_pointwise).
            for cp in predicted_interior[is_miss]:
                ax.axvline(cp, color="0.6", lw=0.7, alpha=0.35, zorder=0)
            for cp in predicted_interior[is_big]:
                ax.axvline(cp, color="tab:red", lw=1.1, alpha=0.6, zorder=1)
            for cp in predicted_interior[is_small]:
                ax.axvline(cp, color="tab:green", lw=1.3, alpha=0.85, zorder=2)

            ax.scatter(np.arange(x.shape[0]), x, c=colors[pred % len(colors)],
                      s=3, linewidths=0, zorder=3)

            # true onsets on top, so they're always legible over the rug/scatter.
            for c in small_cps:
                ax.axvline(c, color="k", lw=1.0, alpha=0.8, ls="--", zorder=4)
            for c in big_cps:
                ax.axvline(c, color="k", lw=1.0, alpha=0.8, ls="-", zorder=4)

            k = len(np.unique(pred))
            n_pred = len(predicted_interior)
            small_r = small_recall(predicted_interior, small_cps)
            big_r = small_recall(predicted_interior, big_cps)
            prec = overall_precision(predicted_interior, truth_cps)
            label = "fixed q=0.95" if method == "baseline" else "pointwise null, q=0.5"
            ax.set_title(
                f"{label} — k={k} clusters, {n_pred} predicted CPs — "
                f"THIS rep: small recall={small_r:.2f}, big recall={big_r:.2f}, "
                f"precision={prec:.2f}",
                fontsize=9.5, loc="left",
            )
            ax.set_ylabel("x")
        axes[-1].set_xlabel("time step")
        fig.suptitle(
            f"Toy trajectory, big:small change ratio {ratio}:1, rep {rep} "
            f"(baseline recall on this rep closest to its {ratio}:1 mean across 20 reps) —\n"
            f"black dashed/solid = true small/big excursion onsets; green/red/grey rug = "
            f"predicted CPs that hit a small onset / hit a big onset / matched neither"
            " — generated by `imbalanced_change_toy.py`",
            fontsize=8.5,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        os.makedirs(outdir, exist_ok=True)
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(outdir, f"clustering_ratio{ratio}.{ext}"), dpi=150)
        plt.close(fig)


def plot_recall_summary(df, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = df.groupby(["ratio", "method"])[["small_recall", "big_recall"]].mean()

    fig, ax = plt.subplots(figsize=(6, 4))
    width = 0.35
    x_pos = np.arange(len(RATIOS))
    for i, method in enumerate(("baseline", "resample_pointwise")):
        vals = [summary.loc[(r, method), "small_recall"] for r in RATIOS]
        label = "fixed q=0.95" if method == "baseline" else "pointwise null, q=0.5"
        ax.bar(x_pos + (i - 0.5) * width, vals, width, label=label)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{r}:1" for r in RATIOS])
    ax.set_xlabel("big:small change-size ratio")
    ax.set_ylabel("recall on the small change's onset")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.set_title(
        "Detection of the smaller change, by imbalance ratio"
        "—generated by `imbalanced_change_toy.py`",
        fontsize=9,
    )
    fig.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"small_change_recall.{ext}"), dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--R", type=int, default=199)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=RESULT_DIR)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    figdir = os.path.join(args.out, "figures")

    df, examples = run(args.reps, args.R, args.seed)

    csv_path = os.path.join(args.out, "imbalanced_change_toy.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nwrote {csv_path} ({len(df)} rows)")

    plot_examples(df, examples, figdir)
    plot_recall_summary(df, figdir)

    summary = df.groupby(["ratio", "method"])[
        ["small_recall", "big_recall", "all_recall", "precision", "v_measure",
         "n_predicted_cps"]
    ].mean().round(3)
    print("\nmean recall / V-measure by ratio and method:")
    print(summary.to_string())
    summary.to_csv(os.path.join(args.out, "summary_by_ratio.csv"))


if __name__ == "__main__":
    main()
