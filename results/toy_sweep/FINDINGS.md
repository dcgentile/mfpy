# Toy-model sweep: change-point detection across 300 1D trajectories

Prepared for the FODS revision. Addresses reviewer major comment 3(a) ("more simple 1D
test cases"), and turns out to bear directly on major comments 1(a) (justifying the
gradient step) and 1(b) (quantile thresholding), plus 3(e) ("discussion on picking `w`")
and 3(d) ("what did we mean by tolerance range?").

## What was run

The full `data/Toy_Model_Trajectories` dataset: a 3 x 100 grid over noise level
sigma in {0, 5, 20} (Laplace scale of the two metastable states) and transition length
`l` in {1, ..., 100}. Each trajectory holds 5000 metastable points in ten state-A blocks
(mean 100) and ten state-B blocks (mean 200), joined by 19 gradual linear ramps, giving
length `5000 + 19*l`. Block lengths are randomised, so change points sit at irregular
intervals.

Change-point detection only; segment clustering and V-measure are deferred.

- **Windows**: `w = 10, 25, 50, 100` (fixed, chosen without reference to `l`; `w = 25` is
  the existing notebook setting) plus `oracle` with `w = 2*l`. 1500 fits.
- **Boundary estimators**: three rules for reading change points out of the test
  statistic (below). Scored on every fit, so 4500 scored results.
- **Threshold sweep**: a separate diagnostic over `q` in [0.50, 0.99] at `w = 25`. 3300 fits.

```bash
python mfpy/experiments/toy_sweep.py --jobs 2
python mfpy/experiments/toy_threshold_sweep.py --jobs 2
python mfpy/experiments/analyze_toy_sweep.py
```

## What counts as a change point

A change point is the instant at which the law of the process changes: where stability
becomes transition, or transition becomes stability. **Each ramp therefore contributes
two change points, its start and its end, so every trajectory has 38.**

Predictions are matched to these by one-to-one optimal (Hungarian) assignment restricted
to pairs within a tolerance `tau`; precision is matched/predicted, recall is matched/38.
One-to-one matching means a cluster of predictions near one boundary yields one true
positive and the rest false positives.

This is stricter than asking whether a prediction lands *somewhere inside* a ramp. That
weaker criterion turns out to hide the main result: a prediction sitting in the middle of
a 100-step ramp has identified neither boundary, but under containment it scores as
correct.

**On the tolerance range.** No single `tau` is treated as privileged, because the
estimator has a localisation bias that scales with `l`; fixing one `tau` would confound
detection ability with that bias. The `tau` curve is the primary detection result, and
signed localisation error is reported alongside as a threshold-free measure of where
predictions actually land. Signed error is oriented so that **positive means displaced
into the interior of the ramp**.

## Finding 1: change points land deep inside the ramp, by ~0.4*l

This is the headline. Median localisation error grows almost perfectly linearly with
ramp length:

| fixed window | slope (steps per unit `l`) | r |
|---|---|---|
| `w = 10`  | 0.336 | 0.936 |
| `w = 25`  | 0.372 | 0.965 |
| `w = 50`  | 0.451 | 0.993 |
| `w = 100` | 0.444 | 0.992 |

The error is *signed inward*: the method reports the first change point well after the
ramp has begun and the second well before it ends, so the detected transition is a
shrunken image of the true one. At `w = 25`, median absolute error by ramp length:

| `l` | 1 | 10 | 25 | 50 | 75 | 100 |
|---|---|---|---|---|---|---|
| median abs error (steps) | 3.5 | 1.0 | 6.5 | 18.5 | 27.0 | 30.0 |

For short ramps (`l` < ~15) the error is small and slightly *outward* (negative), which
is the expected behaviour when the change is effectively instantaneous. See
`figures/localisation_bias.pdf`.

## Finding 2: the gradient step is not the cause

Reviewer comment 1(a) asks us to justify taking the argmin/argmax of the test-statistic
gradient. We implemented two alternatives that share the identical run-detection stage
and differ only in boundary extraction:

- `run_edge` — the first and last index of the supra-threshold run
- `half_max` — first and last crossing of the midpoint of the run's range

**All three agree to within a couple of time steps everywhere:**

| comparison | mean gap | median gap | p90 gap |
|---|---|---|---|
| gradient vs run_edge | 1.37 | 1.0 | 3.0 |
| gradient vs half_max | 1.29 | 1.0 | 2.5 |
| run_edge vs half_max | 2.44 | 2.5 | 4.0 |

Changing the extraction rule does not move the bias. It is inherited from stage 1: the
supra-threshold run is itself much narrower than the ramp, so *any* rule reading
boundaries off that run reproduces the same shrinkage. This is a stronger response to
comment 1(a) than a concession — the gradient step is defensible because the obvious
alternatives are equivalent, and the real problem lies upstream.

## Finding 3: the quantile threshold is the cause, and it is predictable

Upstream is the `q = 0.95` threshold — precisely reviewer comment 1(b). When ramps are
long, a large fraction of the trajectory is in transition, the test statistic is elevated
over a broad plateau, and a top-5% cut selects only the peak of that plateau.

If that account is right, the appropriate threshold should be

```
q* = 1 - (fraction of trajectory spent in transition) = 1 - 19*l / (5000 + 19*l)
```

Sweeping `q` over every trajectory confirms it. Empirically best `q` versus predicted `q*`:

| `l` | 1 | 10 | 20 | 40 | 60 | 80 | 100 |
|---|---|---|---|---|---|---|---|
| best `q` (empirical) | 0.990 | 0.950 | 0.925 | 0.850 | 0.800 | 0.750 | 0.700 |
| `q* = 1 - ramp fraction` | 0.996 | 0.963 | 0.929 | 0.868 | 0.814 | 0.767 | 0.725 |
| median abs error at best `q` | 0.0 | 1.0 | 1.0 | 1.0 | 2.0 | 4.0 | 3.0 |
| median abs error at `q = 0.95` | 3.5 | 1.0 | 4.0 | 14.0 | 22.0 | 27.0 | 30.0 |

Spearman(best `q`, `q*`) = 0.753, p = 4e-56. The agreement is limited mainly by the
coarseness of the `q` grid — the by-`l` medians above track `q*` to within one grid step
almost everywhere.

**At a well-chosen threshold the bias essentially vanishes**: median error stays at 0-4
steps across the entire range of `l`, versus growing to 30. See
`figures/threshold_diagnosis.pdf`.

This converts comment 1(b) from a concession into a result: we can state exactly *why*
the fixed quantile fails, exactly *how much* it costs, and give a closed-form starting
value for `q`. The catch is that `q*` depends on the transition fraction, which is not
known in advance — which is the argument for the resampling-based threshold the reviewer
points to (Matteson & James 2014), or for estimating the transition fraction iteratively.

## Finding 4: the priors, re-tested

**Abruptness: confirmed, strongly.** Spearman rho between F1 (`tau = 25`) and `l` is
negative in all 15 (window, sigma) cells, ranging from **-0.78 to -0.98**, all p < 1e-15.
Abrupt changes are unambiguously easier to localise.

Note this reverses what a containment-based criterion would have said. Under
"is the prediction inside a ramp?", the correlation at `w = 25` was indistinguishable from
zero — long ramps are big targets, so containment gets easier as `l` grows, exactly
cancelling the localisation degradation. The prior was right; the weaker metric was
hiding it.

**Noise: real but secondary.** Spearman rho between F1 and sigma is negative in every
window setting but weak: **-0.09 to -0.24** (`w = 10` not significant, p = 0.14). Under
boundary scoring, ramp length dominates noise level by a wide margin.

*Internal consistency check.* In `figures/f1_vs_transition_length.pdf`, F1 at `tau = 25`
holds near its ceiling and then falls off a cliff at `l` ~ 62 for sigma = 0 and sigma = 5.
That is exactly where Finding 1 predicts it should: the bias reaches the tolerance when
`0.4*l = 25`, i.e. `l = 62.5`. The detection curve and the localisation measurement are
independent computations, so their agreeing on that crossover is a useful check that both
are behaving as described.

## Finding 5: window choice

With boundary scoring the window matters much less than `l` or `q`. Mean F1 at
`tau = 25`, gradient estimator: `w = 25` is best at every noise level (0.645 / 0.685 /
0.639 for sigma = 0 / 5 / 20), with `oracle` (`w = 2l`) at 0.623 / 0.614 / 0.568.

The earlier conclusion stands and is if anything reinforced: **tying `w` to the transition
length does not help.** `oracle` is beaten by a flat `w = 25` at all three noise levels,
losing at both ends of the range (`w = 2` cannot estimate a distribution; `w = 200`
straddles neighbouring ramps). `w` should be set by how many samples are needed to
estimate the marginal stably, not by the expected transition duration.

## Caveats

- The method emits ~43.5 change points on average against 38 true ones, so precision is
  capped below 1 by redundant detections even when localisation is perfect.
- `q*` uses the true ramp fraction, so Finding 3 diagnoses the problem rather than
  supplying a deployable rule. Stated as such above.
- Only one realisation exists per (sigma, `l`) cell. The trends are read across 100 `l`
  values per row, not from individual cells — the same single-realisation design already
  defended in the response to comment 3.
- The threshold sweep was run at `w = 25` and with the gradient estimator only. Given
  Finding 2 the estimator choice should not matter, but the `w` interaction is untested.
- Under `q = 0.95`, 31 of the 1500 (trajectory, window) fits produce **zero** change points
  with the gradient estimator (93 of 4500 across all three estimators). All are sigma = 0
  with `w` in {10, 25} and `l` in [36, 99] — the degenerate zero-noise corner of the same
  thresholding failure described in Finding 3.

## Files

- `toy_sweep_per_trajectory.csv` — 4500 rows: (trajectory, window, estimator)
- `toy_sweep_tolerance_curve.csv` — 67500 rows, adding one per `tau`
- `toy_threshold_sweep.csv` — 3300 rows: (trajectory, `q`)
- `tables/` — detection by tolerance, localisation by `l`, estimator equivalence, window
  choice, threshold diagnosis, prior tests (csv + markdown)
- `figures/` — `tolerance_curves`, `localisation_bias`, `threshold_diagnosis` (pdf + png)

## Reproducibility and verification

`toy_sweep.py` replaces the per-timestep `ot.emd2_1d` loop with a vectorised equivalent:
for equal-size 1D empirical measures the squared 2-Wasserstein distance is the mean
squared difference of order statistics, so all windows can be sorted at once. 15-175x
faster; the whole sweep is ~16s of compute. Validated against
`mfpy.distances.compute_metric_derivative_1d` on 72 stratified cells (agreement to 4e-14,
change points bit-identical in 69/72, F1 identical in all 72) and on 30 randomly chosen
finished cells (0 mismatches). Pass `--exact` to use the library path.

`boundary_estimators.identify_change_points(..., estimator="gradient")` is a faithful port
of `mfpy.distances.identify_change_points`, preserving its quirks (isolated
supra-threshold points ignored; a run still open at the end of the candidate list is
dropped). Verified bit-identical on 120 cells spanning all sigma, `l`, and `w`.

The scorer was hand-checked on `sigma_20_transition_25`: at `tau` = 5 / 10 / 25 it matches
6 / 32 / 38 of 38 true change points, every matched pair was confirmed within `tau`, the
assignment confirmed injective on both sides, and recall confirmed equal to matched/38.
All 300 ground-truth files parse into 19 non-overlapping ramps of width `l` summing to the
correct trajectory length. One file, `sigma_5_transition_20`, has ramp widths in
{19, 20, 21} rather than exactly 20; its total transition mass and trajectory length are
both exactly right, so this is a benign +/-1 artifact in the MATLAB generator.

14 unit tests cover the scorer, including the case that motivated this revision — that a
prediction in the middle of a ramp is *not* a true positive
(`test_midramp_prediction_is_not_a_true_positive`).
