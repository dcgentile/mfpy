# V-measure — FINDINGS

Answers action item 4 / reviewer major comment 3(b) ("Add V-measure of clustering? Look
into V-measure at least."). Scripts: `mfpy/experiments/vmeasure.py` (run),
`mfpy/experiments/analyze_vmeasure.py` (tables + figure). Raw output:
`vmeasure_toy.csv` (300 rows, one per toy trajectory) and `vmeasure_langevin.csv`
(5 rows, one per core-set half-width).

Parameters are the manuscript's: toy `w = 25, q = 0.95`; Langevin `w = 215, q = 0.86`;
ADP with `Z = 1.65`. Total runtime ~15s.

## Setup

V-measure (Rosenberg & Hirschberg 2007) is the right score here for two reasons the
rebuttal can state directly: it is invariant to label permutation, and it does not require
the predicted and true label sets to have the same cardinality. ADP chooses its own number
of clusters, so any score requiring a fixed `k` would have to be rigged.

Two targets are scored for every trajectory:

- **3-class** — ground truth `{state A, transition, state B}`. The pipeline emits ramp
  segments as segments in their own right, so it can in principle recover a transition
  cluster; the manuscript claims it does.
- **2-class** — `{state A, state B}` with transition points *deleted* from truth and
  prediction before scoring. Isolates metastable-state recovery from ramp handling.

Deletion rather than relabelling is deliberate: relabelling ramp points to a neighbouring
state would invent a ground truth the generator does not assert, and would make
completeness depend on ramp length.

Two references are scored alongside:

- **`kmeans_*`** — k-means on the raw values at `k = 3`. The manuscript already contrasts
  the two qualitatively; this makes the contrast a number.
- **`oracle_*`** — the *same* ADP clustering step run on the *true* ramp boundaries
  instead of detected change points. The gap between `oracle_*` and `adp_*` is the part of
  the shortfall caused by change-point localisation rather than by clustering.

## 1. The clustering step is not the bottleneck — localisation is

`results/vmeasure/tables/table3_localisation_gap.md`:

| sigma | n | V (detected CPs) | V (true CPs) | gap | gap as fraction of shortfall |
|---|---|---|---|---|---|
| 5  | 96  | 0.509 | 0.978 | 0.469 | 0.955 |
| 20 | 100 | 0.486 | 0.880 | 0.395 | 0.767 |

Given correct change points, ADP on Wasserstein distances between segments recovers the
three-class labelling essentially perfectly at sigma = 5 (V = 0.978) and very well at
sigma = 20 (V = 0.880). **96% (sigma = 5) and 77% (sigma = 20) of the V-measure shortfall
is attributable to change-point localisation, not to the clustering.**

This is the strongest single result in the revision so far, and it is the same story the
toy sweep and the resampling experiment tell from the other direction: the segment
geometry and the clustering are sound; the fixed-quantile threshold is what degrades the
output. It also means an improvement to the threshold (the pointwise permutation null in
`results/resampling_threshold/`) should propagate into clustering quality — **untested,
and the obvious next experiment.**

## 2. V-measure degrades with transition length, mirroring detection F1

`table2_vmeasure_by_transition_length.md`, 3-class, ADP on detected change points:

| sigma | ell <= 20 | 20-40 | 40-60 | 60-80 | 80-100 |
|---|---|---|---|---|---|
| 5  | 0.736 | 0.621 | 0.497 | 0.302 | 0.402 |
| 20 | 0.697 | 0.588 | 0.460 | 0.340 | 0.342 |

The oracle-segmentation curve stays flat over the same range (0.999 -> 0.947 at sigma = 5),
so this is the localisation bias propagating, not a failure of the clustering to handle
long ramps. See `figures/vmeasure_vs_tl.pdf`.

The 2-class scores are uniformly higher (0.68 and 0.65 on average) than the 3-class scores
(0.51, 0.49): the method recovers the two metastable states considerably better than it
recovers the transition class, which is what the localisation bias predicts — a
change point landing ~0.4·ell inside a ramp mislabels ramp points, but leaves the interiors
of the stable blocks intact.

## 3. Against k-means: a split decision, honestly reported

`table1_vmeasure_by_sigma.md`:

| sigma | V3 ADP | V3 k-means | V2 ADP | V2 k-means |
|---|---|---|---|---|
| 0  | 0.167 | 0.801 | 0.211 | 1.000 |
| 5  | 0.512 | 0.787 | 0.684 | 0.996 |
| 20 | 0.486 | 0.582 | 0.653 | 0.749 |

**k-means is not beaten on this benchmark, and the paper should not claim it is.** The toy
states are two Laplace distributions with well-separated means (100 and 200), so a
threshold on the value is close to sufficient, and k-means is exactly a threshold on the
value. The gap narrows sharply as noise rises — at sigma = 20 ADP is within 0.10 (3-class)
and the crossover in the figure happens around ell = 20 — which is the honest form of the
manuscript's claim: k-means degrades faster as the distributions overlap.

The defensible version of the existing argument is the one already in the manuscript's
figures (`toy-kmeans-distributions`), namely that k-means recovers point labels but not the
*distributions*, imposing hard dividing hyperplanes that misrepresent each state's
variance. V-measure on point labels is not the metric that shows this. **Recommend
reporting V-measure as an evaluation of our own method against ground truth, and keeping
the k-means comparison qualitative/distributional rather than promoting it to a V-measure
horse race we lose.**

## 4. sigma = 0 is degenerate for density-based clustering

At sigma = 0 the trajectory is piecewise constant (12 distinct values across the whole
series), so every state-A block is the *same measure* as every other state-A block and
their W2 distance is exactly 0. ADP's 2-NN intrinsic-dimension estimator takes logs of
nearest-neighbour distances and is undefined on these ties.

`vmeasure.py` collapses exact duplicate segments to one representative before clustering
and broadcasts labels back — exact, since identical measures must receive the same label
under any distance-based clustering, and a no-op in the noisy cells where no ties occur.
Even so, 11 of the 100 sigma = 0 trajectories fail outright (fewer than 5 distinct
segments), all 100 fail on the oracle segmentation, and where ADP does run it returns
`k = 1` on 65 of the remaining 89 cells, giving V = 0.

**The sigma = 0 row should be excluded from any headline number and reported as a
degenerate case**, with the one-line explanation above. It is not a defect of the
Wasserstein geometry — it is that a noiseless trajectory has no density for a
density-peaks method to find peaks in.

## 5. Langevin

`table4_langevin_core_sweep.md`. ADP finds **3 clusters**, confirming the manuscript's
claim ("three distinct states are identified, with one corresponding to each well, and the
third capturing the transition").

| core half-width c | transition frac | V3 ADP | V2 ADP | V3 k-means | V2 k-means |
|---|---|---|---|---|---|
| 0.0 | 0.000 | 0.685 | 0.685 | 0.658 | 0.658 |
| 0.2 | 0.064 | 0.688 | 0.783 | 0.718 | 0.729 |
| 0.4 | 0.142 | 0.667 | 0.866 | 0.869 | 0.868 |
| 0.6 | 0.242 | 0.588 | 0.913 | 0.813 | 1.000 |
| 0.8 | 0.410 | 0.459 | 0.943 | 0.564 | 1.000 |

Two-class V-measure rises monotonically to 0.943: on well cores, the method recovers the
two metastable states well. Three-class V-measure peaks near c = 0.2 and falls after, which
is expected — widening the core-set band relabels more and more genuinely-in-a-well points
as "transition", and no segmentation can match that.

### Caveat that must be stated in the paper

The Langevin trajectory ships no ground-truth labels, so truth here is a **position-based
proxy**: A is `x < -c`, B is `x > +c`, transition is `|x| <= c`. This is not a committor —
a point at `x = 0.1` heading back into the left well is called "transition" even though a
dynamical labelling would call it A.

Worse for the comparison: this proxy ground truth *is itself a thresholding of x*, and
k-means on 1-D values is also a thresholding of x. That is why k-means scores V2 = 1.000 at
c >= 0.6 — it is tautological, not a result. **The Langevin V-measure numbers are usable as
evidence that our method recovers the wells; they are not usable to compare against
k-means.** A committor- or MSM-based reference labelling (via `deeptime`, already a
dependency of the Langevin notebook) would fix this and is the honest way to make that
comparison, if we want it.

## 6. Under the pointwise resampling threshold: large gain on toy, regression on Langevin

Re-run with `--threshold resample_pointwise --q 0.5 --R 199`, the setting recommended by
`results/resampling_threshold/FINDINGS.md`. Outputs: `vmeasure_toy_resample.csv`,
`vmeasure_langevin_resample.csv`.

**Toy — a large improvement**, paired over the 200 trajectories with sigma > 0 that
succeed under both thresholds:

| | fixed q = 0.95 | pointwise null, q = 0.5 | oracle segmentation |
|---|---|---|---|
| V-measure, 3-class | 0.499 | **0.684** | 0.928 |
| V-measure, 2-class | 0.669 | **0.806** | 0.990 |
| mean clusters found (true: 3) | 2.39 / 2.37 | 3.01 / 2.78 | — |

The resampling threshold is better on **80% of trajectories** and closes roughly **43% of
the gap** between the incumbent and the oracle-segmentation ceiling. The gain is larger at
high noise (sigma = 20: V2 0.653 -> 0.861) than at low (sigma = 5: 0.684 -> 0.751), matching
the F1 result in the resampling experiment. This is the prediction from sections 1-2
confirmed: fix the threshold, and clustering quality follows without touching the
clustering.

**Langevin — worse, and the reason matters.** Segment count goes from 157 to 463, ADP drops
from 3 clusters to 2 (losing the transition state the manuscript highlights), and
V-measure at `c = 0` falls from 0.685 to 0.621.

Diagnostic:

| data set | lag-1 autocorrelation | median observed statistic | median permutation null | ratio |
|---|---|---|---|---|
| toy (sigma = 5, ell = 50) | 0.988 | 2.91 | 19.58 | 0.15 |
| Langevin | 0.999 | 0.269 | 0.107 | 2.5 |

The null sits *above* the observed statistic on the toy data and *below* it on Langevin, so
a single global `q` cannot mean the same thing on both. The cause is the exchangeability
assumption: `null_distributions` shuffles the raw values, which is a valid "no change" null
for the toy generator (independent Laplace draws within each block) but not for a Langevin
trajectory, where the within-state dynamics are strongly autocorrelated. Destroying that
autocorrelation makes half-window distributions far more similar than they really are, the
null collapses, the cutoff lands too low, and the trajectory is oversegmented.

**Consequence for the manuscript:** the recommendation in
`results/resampling_threshold/FINDINGS.md` ("adopt `resample_pointwise`") is now qualified.
It is well supported on iid-within-state data and demonstrably fails on the paper's own
Langevin trajectory. Either the claim is scoped to exchangeable-within-state processes, or
the null is replaced by a **block / circular permutation** that preserves short-range
dependence — the standard fix, untested here, and the obvious next experiment.

## Status / what is still shaky

- Section 1's headline (localisation is the bottleneck) is solid and rests on 196
  trajectories.
- Not tested: whether the pointwise resampling threshold from
  `results/resampling_threshold/` improves V-measure. It should, and it is cheap.
- Not tested: sensitivity to ADP's `Z` (fixed at 1.65 throughout, as in the manuscript).
- Langevin is a single trajectory at a single `(w, q)`, with a proxy ground truth — see
  the caveat above.
- The 3-class target is arguably harsh, since the generator's ramp is a Wasserstein
  geodesic and its midpoint is genuinely "between" states rather than a state of its own.
  The 2-class numbers are the more defensible headline.
