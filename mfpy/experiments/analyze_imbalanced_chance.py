"""
Chance-level control for `imbalanced_change_toy.py`.

Motivation
----------
A method that predicts almost everywhere gets a high recall almost for free: predicting
~400 change points against 32 true ones (ratio=1, resample_pointwise) is not evidence of
localisation skill unless it beats what the *same number* of uniformly-random predictions
would score. Recall alone, without this control, overstates how well a flooding threshold
is actually doing -- this is exactly the concern raised about the original figures/table
(session note, 2026-09-04: "It seems like the pointwise null produces a huge number of
change points, the vast majority of which are irrelevant").

For each (ratio, method), draws the method's mean predicted-CP count uniformly at random
over the trajectory (400 trials) and scores recall against the small/big excursion onsets
with the same one-to-one Hungarian matching used everywhere else in this experiment. The
gap between observed and chance recall is the part of the score attributable to actual
localisation.

Usage
-----
    python mfpy/experiments/analyze_imbalanced_chance.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)

from imbalanced_change_toy import RATIOS, TAU, generate, RESULT_DIR  # noqa: E402
from toy_scoring import match_change_points  # noqa: E402

N_TRIALS = 400


def chance_recall(n_pred, true_cps, T, tau, trials, rng):
    vals = np.empty(trials)
    for i in range(trials):
        pred = rng.integers(1, T - 1, size=n_pred)
        pairs = match_change_points(pred, true_cps, tau)
        vals[i] = len(pairs) / len(true_cps)
    return vals.mean()


def main():
    csv_path = os.path.join(RESULT_DIR, "imbalanced_change_toy.csv")
    df = pd.read_csv(csv_path)

    rng = np.random.default_rng(2026)
    rows = []
    for ratio in RATIOS:
        # ground-truth geometry only depends on ratio, not on rep/method
        x, truth_cps, labels, small_cps, big_cps = generate(ratio, rng=np.random.default_rng(0))
        T = x.shape[0]
        for method in ("baseline", "resample_pointwise"):
            sub = df[(df["ratio"] == ratio) & (df["method"] == method)]
            n_pred = int(round(sub["n_predicted_cps"].mean()))
            chance_small = chance_recall(n_pred, small_cps, T, TAU, N_TRIALS, rng)
            chance_big = chance_recall(n_pred, big_cps, T, TAU, N_TRIALS, rng)
            rows.append({
                "ratio": ratio,
                "method": method,
                "mean_n_predicted": n_pred,
                "observed_small_recall": round(sub["small_recall"].mean(), 3),
                "chance_small_recall": round(chance_small, 3),
                "small_recall_above_chance": round(
                    sub["small_recall"].mean() - chance_small, 3
                ),
                "observed_big_recall": round(sub["big_recall"].mean(), 3),
                "chance_big_recall": round(chance_big, 3),
                "big_recall_above_chance": round(
                    sub["big_recall"].mean() - chance_big, 3
                ),
            })

    out = pd.DataFrame(rows)
    out_path = os.path.join(RESULT_DIR, "chance_level_control.csv")
    out.to_csv(out_path, index=False)
    print(out.to_string(index=False))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
