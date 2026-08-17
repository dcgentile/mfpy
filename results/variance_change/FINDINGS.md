# Changes in variance — FINDINGS

Answers reviewer comment 3(a): *"it would be worthwhile to include some further simulations
of simpler settings as a sanity check that the method is performing well, such as
(instantaneous) changes in: mean and/or covariance."*

Scripts: `mfpy/experiments/variance_change.py`, `mfpy/experiments/analyze_variance_change.py`.
Raw output: `variance_change.csv` (79,200 rows). Runtime ~65s.

## Why this was needed

`data/Toy_Model_Trajectories/Trajectory_Generator.m` draws both states with
`laprnd(n, 1, mean, sigma)` using the **same `sigma`**. Every change point in every
trajectory in every previous experiment is therefore a pure shift in the mean — the case a
mean-based statistic already handles. The paper's central claim is that a Wasserstein
statistic detects general distributional change without parametric assumptions, and until
now nothing tested that claim on a change the mean cannot see.

## Design

Both states Laplace; the transition is the McCann interpolant, which for a location-scale
family moves location and scale linearly and stays Laplace throughout. For Laplace the
standard quantile function has second moment 2, so

    W2^2( Laplace(mu1,b1), Laplace(mu2,b2) ) = (mu1-mu2)^2 + 2 (b1-b2)^2.

Each arm is parameterised by the **same** state separation `d = W2(A,B)`, split differently:

| arm | location | scale |
|---|---|---|
| `mean` | `mu_B - mu_A = d` | unchanged |
| `variance` | unchanged | `b_B - b_A = d/sqrt(2)` |
| `both` | `d/sqrt(2)` | `d/2` |

So a horizontal comparison at fixed `d` holds the distance between the two laws constant,
and any difference is attributable to *where* the change lives, not how big it is. The
identity is verified numerically against 400,000-sample empirical distances (agreement to
0.06 at `d = 40`).

Grid: 3 arms x 5 separations x 4 transition lengths (0, 10, 25, 50) x 3 windows
(25, 50, 100) x 11 quantiles x 2 detectors x 2 extractors, **10 independent replicates per
cell**. The replicates also supply the error bars missing elsewhere in the revision
(reviewer comment 3); typical standard deviation across reps is 0.02-0.10 F1.

**Comparator.** `mean` is `|mean(right half) - mean(left half)|` on the same sliding window
— a moving-window CUSUM-type statistic. Thresholding, run detection, boundary extraction and
Hungarian scoring are all shared, so the two detectors differ *only* in what they measure.
Each detector is given its own best `q`, so neither is handicapped by a threshold tuned for
the other.

## 1. The headline: the mean statistic is blind, the Wasserstein statistic is not

Instantaneous changes, `peak` extractor, mean F1 at `tau = 10` over 10 reps
(`tables/table1_instantaneous_f1.md`):

| arm | separation | mean stat, w=100 | W2, w=100 |
|---|---|---|---|
| variance | 10 | 0.120 | **0.248** |
| variance | 20 | 0.143 | **0.336** |
| variance | 40 | 0.148 | **0.551** |
| variance | 80 | 0.128 | **0.755** |

**On the variance arm the mean statistic does not improve with separation at all.** Across
all 120 variance-arm cells its F1 ranges from 0.076 to 0.255 with no trend in either `d` or
`w` — it is measuring a quantity that does not change. The Wasserstein statistic rises
monotonically in both, reaching 0.755. At `d = 80, w = 100` the ratio is **5.9x**.

This is the demonstration the manuscript has been missing: a setting where the distributional
statistic succeeds and the moment-based one cannot, by construction rather than by tuning.

## 2. The price of generality on the mean arm is small

`tables/table4_head_to_head_by_kind_and_window.md`, averaged over separations and
transition lengths:

| arm | W2 | mean statistic | difference |
|---|---|---|---|
| mean, w=25 | 0.349 | 0.386 | -0.037 |
| mean, w=100 | 0.415 | 0.436 | -0.021 |
| variance, w=25 | 0.201 | 0.151 | **+0.050** |
| variance, w=100 | 0.309 | 0.159 | **+0.151** |
| both, w=100 | 0.345 | 0.316 | +0.028 |

On pure mean changes the matched statistic is better, as it must be, but only by 0.02-0.04
F1 — and the gap narrows as the window grows. **The honest summary is that the Wasserstein
statistic gives up very little on the case a mean-based method is designed for, and gains a
great deal on the case it cannot handle.** That is a stronger and more defensible claim than
anything currently in the manuscript.

## 3. Variance changes need a larger window — material for 3(e)

`tables/table5_window_dependence.md`, mean F1 over separations, instantaneous:

| arm | detector | w=25 | w=50 | w=100 |
|---|---|---|---|---|
| mean | W2 | 0.480 | 0.534 | 0.556 |
| variance | W2 | 0.224 | 0.319 | **0.405** |
| variance | mean stat | 0.133 | 0.135 | 0.133 |

The Wasserstein statistic's performance on variance changes nearly doubles from `w = 25` to
`w = 100`, while on mean changes it moves only 0.08. This is the expected statistical fact —
estimating a scale takes more samples than estimating a location — and it sharpens the `w`
advice: **the window should be set by what has to be estimated, and a method that claims to
detect general distributional change needs a larger window than one that only tracks the
mean.** The existing guidance (`w` set by the samples needed to estimate the marginal
stably, not by the transition duration) survives and gains a concrete rationale.

## 4. A methodological point the instantaneous setting exposes

The manuscript's `gradient` rule emits **two** change points per supra-threshold run — the
start and the end of a ramp. That is right for a gradual transition and wrong for an
instantaneous one, where there is a single change point. Against 19 true change points it
emits ~38 predictions, capping precision near 0.5 and F1 near 0.67 *regardless of how well
the change is localised*.

Both extractors are recorded for every cell. At `d = 40, w = 25`, instantaneous, the `peak`
rule (one change point per run, at the run's maximum) lifts F1 from 0.532 to 0.797 for the
Wasserstein statistic and from 0.578 to 0.848 for the mean statistic.
**If the instantaneous experiments go into the paper, the extraction rule has to be stated
as conditional on the transition model**, or the numbers will look far worse than the
method's actual localisation warrants.

## 5. Gradual transitions

`tables/table3_gradual_f1_w50.md`. Same ordering, attenuated: on the variance arm at
`d = 80, w = 50`, W2 scores 0.506 / 0.411 / 0.238 at `tl = 10 / 25 / 50` against the mean
statistic's 0.183 / 0.167 / 0.160. The degradation with transition length reproduces the
finding from `results/toy_sweep/`, now on data where the mean is constant, which rules out
any explanation specific to the mean-shift geometry.

## Status / what is still shaky

- Only the Laplace family. A change in *shape* at fixed mean and variance — the case that
  would separate W2 from a variance-based statistic as cleanly as this separates it from a
  mean-based one — is untested. That is the natural next step if a referee presses.
- The comparator is a moving-window mean difference, not a true CUSUM. It is the right
  architectural match for our pipeline, but it is not literally the method in the CUSUM
  literature, and should not be described as such.
- Blocks are fixed at 250 samples; the shipped generator randomises block lengths. This
  makes the change points evenly spaced, which is slightly easier than the shipped data.
- Absolute F1 is low in the small-separation cells for both detectors. At `d = 5` with
  `b_A = 20` the two states overlap almost completely; those rows are near the noise floor
  and should not be over-read.
- 10 replicates gives a standard error of roughly 0.01-0.03 F1, adequate for the ordering
  claims made here but not for fine distinctions.
