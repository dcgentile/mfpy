# q-w interaction — FINDINGS

Closes the open item flagged in `toy_threshold_sweep.py` and
`results/resampling_threshold/FINDINGS.md`: both threshold results were established at a
single window size, `w = 25`. Scripts: `mfpy/experiments/qw_interaction.py`,
`mfpy/experiments/analyze_qw_interaction.py`. Raw output:
`qw_interaction_baseline.csv` (300 trajectories x 4 windows x 11 quantiles, fixed-quantile
rule only) and `qw_interaction_resample.csv` (stratified 60 trajectories, both rules).
Metric throughout is mean `F1` at `tau = 10`; segmentation, boundary extraction
(`gradient`) and scoring are shared with the earlier experiments.

## Headline

**The fixed quantile has essentially no q-w interaction. The pointwise permutation null has
a severe one, and its recommended `q = 0.5` is only correct at the `w` it was tuned at.**

`tables/table3_best_q_per_w.md`:

| w | fixed quantile: best q | F1 | permutation null: best q | F1 |
|---|---|---|---|---|
| 10  | 0.80 | 0.486 | 0.50  | 0.320 |
| 25  | 0.75 | 0.626 | 0.50  | **0.666** |
| 50  | 0.75 | 0.539 | 0.925 | 0.535 |
| 100 | 0.80 | 0.581 | 0.99  | 0.105 |

The fixed quantile's optimum sits at `q in {0.75, 0.80}` at every window, and its worst
case across `w` at a single fixed `q` is 0.486. The permutation null's optimum sweeps from
0.5 to 0.99, and its worst case across `w` at any single `q` is **0.105** — worse than the
fixed quantile everywhere except at `w = 25`.

## Why: the two cutoffs move in opposite directions with w

`tables/table4_cutoff_alignment.md`, median over trajectories:

| w | null cutoff at q = 0.5 | where that sits on the *observed* statistic's quantile scale | fixed-quantile cutoff at q = 0.75 |
|---|---|---|---|
| 10  | 28.6 | 0.964 | 6.6 |
| 25  | 19.5 | **0.800** | 14.1 |
| 50  | 14.3 | 0.636 | 33.4 |
| 100 | 10.4 | 0.500 | 69.3 |

As `w` grows, the observed statistic grows (longer half-windows separate the two states
more sharply) while the permutation null *shrinks* (shuffling destroys structure, and with
more samples per window the two shuffled half-windows look more alike). The two curves
cross near `w = 25`.

Translated onto the observed statistic's own quantile scale, the null's `q = 0.5` cutoff
is equivalent to a fixed quantile of 0.964 at `w = 10`, 0.800 at `w = 25`, and 0.500 at
`w = 100`. Since the fixed rule's own optimum is ~0.75-0.80, **only `w = 25` places the
null cutoff in the right region — which is precisely the window at which the
`q = 0.5` recommendation was derived.** The earlier result was not wrong, but it was a
coincidence of the grid, and it does not generalise.

At `w = 100` the consequence is severe: the cutoff sits at the median of the observed
statistic, so most of the trajectory is supra-threshold, runs merge, and the extracted
boundaries land far from any true change point — median absolute localisation error 49.5
steps, against 8.5 at `w = 50`.

## Interaction with transition length

`tables/table5_resample_pointwise_f1_by_w_and_tl.md`, at `q = 0.5`:

| w | ell <= 25 | 25-50 | 50-75 | 75-100 |
|---|---|---|---|---|
| 10  | **0.899** | 0.273 | 0.067 | 0.043 |
| 25  | 0.374 | **0.855** | 0.772 | 0.662 |
| 50  | 0.010 | 0.343 | 0.772 | **0.716** |
| 100 | 0.000 | 0.005 | 0.021 | 0.292 |

Under the permutation null there is a strong `w`-`ell` matching effect: short ramps want a
short window, long ramps a longer one, and the penalty for mismatch is large (0.899 -> 0.010
for `ell <= 25` between `w = 10` and `w = 50`).

**This qualifies `results/toy_sweep/FINDINGS.md` finding 6.** That finding — an oracle
`w = 2*ell` is beaten by a flat `w = 25` — was established for the *fixed-quantile* rule,
and it still holds there. It does not carry over to the permutation null, where adapting
`w` to the transition length would help substantially. The manuscript's `w` advice should
be scoped to the thresholding rule it was measured under.

## Consequences for the manuscript

1. **For the method as published (fixed quantile), `q` and `w` can honestly be discussed as
   independent knobs.** `q in [0.75, 0.80]` is a defensible default at any window in this
   range. This is a clean, positive answer to reviewer 3(e) and strengthens the existing `w`
   discussion.
2. **The `resample_pointwise` recommendation must be stated as a `(w, q)` pair, not a `q`.**
   Combined with the exchangeability failure on the Langevin data
   (`results/vmeasure/FINDINGS.md` §6), the resampling threshold now has two documented
   limitations: it does not transfer across `w`, and it does not transfer to autocorrelated
   data. It remains the best configuration found anywhere in this repo at `(w = 25,
   q = 0.5)` — F1 0.666 against the fixed quantile's best-anywhere 0.626 — but it is a
   fragile optimum, not a robust improvement.
3. **A self-calibrating variant is the obvious repair** and is suggested directly by the
   alignment table: choose the null quantile so that the resulting cutoff lands at a target
   quantile (~0.8) of the *observed* statistic, rather than fixing the null quantile itself.
   Untested.

## Status / what is still shaky

- The fixed-quantile arm is all 300 trajectories; the resampling arm is a stratified 60
  (3 noise levels x 20 transition lengths spanning 1-100). The 60-trajectory subset
  reproduces the 300-trajectory baseline numbers to within 0.01-0.03 F1 at every `(w, q)`,
  which is the check that the subsample is representative (compare
  `table1_baseline_f1_by_w_and_q` with `table2_baseline_f1_by_w_and_q`).
- Only four windows, log-spaced-ish. The crossing point is located between `w = 10` and
  `w = 50` but not resolved finely.
- `R = 199` throughout, never swept — the same open caveat as the original resampling
  experiment.
- Scored at `tau = 10` in the tables above; `tau = 5` and `tau = 25` are in the CSVs and
  show the same ordering.
- The `w`-`ell` matching effect is read from the stratified 60 and from bins of 15-20
  trajectories each; the direction is unambiguous but the individual cell values are not
  precise.
