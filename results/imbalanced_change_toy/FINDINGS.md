# Imbalanced change sizes — FINDINGS

Answers `results/resampling_threshold/TODO.md` item 4 / `paper-resources/CONTINUE-PROMPT.md`
next-task 1: the reviewer's own worked example for major comment 1(b), never previously
tested. Reviewer, verbatim: "if there are 2 (instantaneous) changes in the mean, and one is
much larger than the other, the quantile-based threshold will be set too high to detect the
smaller change." Script: `mfpy/experiments/imbalanced_change_toy.py`. Output:
`imbalanced_change_toy.csv` (120 rows = 3 ratios x 2 methods x 20 reps),
`summary_by_ratio.csv`, `figures/`.

## Setup

Each trajectory is 8 cycles of (baseline, small excursion, baseline, big excursion), all
instantaneous, same Laplace scale (`b = 15`) throughout, base mean 100. The small excursion
shifts the mean by `delta = 12`; the big excursion shifts it by `ratio * delta` for
`ratio in {1, 3, 10}`. 32 true change points per trajectory (4 per cycle x 8 cycles), 20
independent reps per ratio.

Two threshold rules, both at the manuscript's `w = 25`, everything downstream
(`vmeasure.cluster_trajectory`: run detection, gradient boundary extraction, ADP segment
clustering) shared:

- **baseline** — fixed `q = 0.95` quantile of the observed statistic (manuscript default).
- **resample_pointwise** — `q = 0.5` quantile of the pointwise permutation null, `R = 199`
  (the rule recommended in `results/resampling_threshold/FINDINGS.md`).

Recall on the small change's onset and on the big change's onset are scored separately
(one-to-one Hungarian matching, `tau = 10`), alongside overall precision and 3-class
V-measure (baseline / small-excursion / big-excursion) against ground truth.

## 1. The reviewer's mechanism, reproduced cleanly

`summary_by_ratio.csv`, mean over 20 reps:

| ratio | method | small recall | big recall | precision | V-measure | mean predicted CPs |
|---|---|---|---|---|---|---|
| 1:1  | baseline | 0.356 | 0.444 | 0.163 | 0.001 | 75.5 |
| 1:1  | resample_pointwise | 0.888 | 0.881 | 0.063 | 0.040 | 443.2 |
| 3:1  | baseline | **0.081** | 0.975 | 0.387 | 0.497 | 42.3 |
| 3:1  | resample_pointwise | 0.875 | 0.775 | 0.068 | 0.408 | 393.6 |
| 10:1 | baseline | **0.000** | 1.000 | 0.482 | 0.555 | 31.0 |
| 10:1 | resample_pointwise | 0.112 | 0.094 | 0.078 | 0.522 | 46.0 |

At `ratio = 10`, the fixed quantile misses the small change **on every one of 20
replicates** (recall 0.000, std 0.000) while catching the big one perfectly (1.000). At
`ratio = 3` it is already down to 0.081. This is the reviewer's claimed mechanism, exactly
as described, and it had never been measured directly before this experiment. See
`figures/clustering_ratio10.pdf` — the small excursions (dashed lines) are invisible to the
`q = 0.95` clustering, folded entirely into the baseline cluster, while the big excursions
(solid lines) are cleanly separated.

## 2. Recall alone is the wrong way to read this — a chance-level control

Before crediting `resample_pointwise` with "recovering" the small change at `ratio = 1` or
`3`, ask what recall the *same number* of uniformly-random predicted change points would
score against the same ground truth. `chance_level_control.csv`
(`analyze_imbalanced_chance.py`, 400 random-placement trials per row, one-to-one Hungarian
matching, same `tau = 10` as everywhere else):

| ratio | method | mean predicted CPs | observed small recall | chance small recall | **above chance** | observed big recall | chance big recall | **above chance** |
|---|---|---|---|---|---|---|---|---|
| 1:1  | baseline | 76  | 0.356 | 0.215 | **+0.14** | 0.444 | 0.218 | **+0.23** |
| 1:1  | resample_pointwise | 443 | 0.888 | 0.762 | **+0.13** | 0.881 | 0.758 | **+0.12** |
| 3:1  | baseline | 42  | 0.081 | 0.118 | **-0.04** | 0.975 | 0.125 | **+0.85** |
| 3:1  | resample_pointwise | 394 | 0.875 | 0.711 | **+0.16** | 0.775 | 0.709 | **+0.07** |
| 10:1 | baseline | 31  | 0.000 | 0.095 | **-0.10** | 1.000 | 0.107 | **+0.89** |
| 10:1 | resample_pointwise | 46  | 0.112 | 0.146 | **-0.03** | 0.094 | 0.136 | **-0.04** |

Two things follow, and they cut against the framing in the previous draft of this file.

**`resample_pointwise`'s apparent recall gains at `ratio = 1` and `3` are mostly chance.**
Predicting 400+ change points into a ~6,600-8,600-sample series with 8 small and 8 big true
onsets scores 0.71-0.76 recall *by construction*, before the method has done anything —
that is what "above chance" is measuring away. The genuine edge over chance is small
(+0.12 to +0.16) and comparable in size to the baseline's own edge over chance at
`ratio = 1` (+0.14 to +0.23), despite the 5-10x difference in raw recall numbers that
motivated this being read as a win in the first draft of this file.

**At `ratio = 10` — the reviewer's own sharpest case — `resample_pointwise` is at or below
chance on both classes** (-0.03 small, -0.04 big). Its 46 predicted change points carry
*no measurable localisation signal at all* here: a practitioner would have done as well
throwing the same number of darts at the trajectory uniformly at random. This is a
stronger, more specific negative result than "does not fix the problem" — the pointwise
null is not merely unhelpful at extreme imbalance, its output is statistically
indistinguishable from noise there.

**The baseline's failure, by contrast, is not noise — it is a targeted, chance-beating
exclusion.** At `ratio = 10` baseline scores exactly 0 on the small change (worse than the
+0.095 chance rate: its 31 predictions are not merely failing to land on small onsets by
bad luck, they are being spent elsewhere with more-than-chance precision) while beating
chance by +0.89 on the big change. That asymmetry — a small budget of predictions, spent
with real (better-than-chance) accuracy, entirely on the large change and never on the
small one — is the reviewer's mechanism exactly as described, and it is what makes this
control necessary: it is what distinguishes "the threshold is measurably biased toward the
large change" from "the method is guessing."

## 3. Precision confirms the same story, independently of the chance control

`resample_pointwise` predicts 390-445 change points against 32 true ones at every ratio
(precision 0.06-0.08 throughout, against baseline's 0.16-0.48). It is not selectively
recovering the small excursions; it is oversegmenting the whole trajectory and picking up
the small excursions as a side effect of predicting almost everywhere. `V-measure` reflects
this: despite much higher small-change recall, `resample_pointwise`'s V-measure is *lower*
than baseline's at both `ratio = 3` (0.408 vs. 0.497) and `ratio = 10` (0.522 vs. 0.555) —
recall alone overstates how well the method is actually doing. See
`figures/clustering_ratio3.pdf`: the pointwise-null panel does not produce a clean
small-excursion cluster, it produces hundreds of small fragments.

## 4. Baseline is also poorly calibrated on this trajectory shape, independent of imbalance

At `ratio = 1` (no imbalance — the calibration case), baseline small/big recall is only
0.356/0.444, and one representative replicate (`figures/clustering_ratio1.pdf`) collapses
to a single cluster (`k = 1`) under the manuscript's `q = 0.95`. This trajectory shape
(sparse instantaneous excursions returning to a shared baseline, most of the series spent
at baseline) is qualitatively different from the shipped toy generator (dense alternation
between two states via gradual ramps), and `q = 0.95` was never tuned for it — the same
`q* = 1 - (fraction of trajectory elevated)` logic from `results/toy_sweep/FINDINGS.md`
predicts a much lower `q` should be used here, since only a small fraction of each
trajectory is in an excursion. **This confound must be kept separate from the imbalance
result**: the `ratio = 1` row measures general threshold miscalibration for this trajectory
shape, not imbalance sensitivity (there is nothing to be imbalanced yet). The imbalance
effect specifically is the *drop* from `ratio = 1` to `ratio = 10` within a method
(baseline: 0.356 -> 0.000; resample_pointwise: 0.888 -> 0.112) — both methods degrade with
imbalance, contrary to a reading where resampling would be imbalance-*invariant*.

## Consequences for the appendix

1. Item B4/4 is now closed: the reviewer's own sharp example is confirmed, sharply, on the
   fixed quantile, and confirmed as a *targeted* effect rather than noise (§2: baseline
   beats chance by +0.89 on the large change and sits *below* chance, at exactly 0, on the
   small one at `ratio = 10`). This is strong, previously-missing evidence for the
   appendix's motivation section (§A.1 in `appendix_resampling_outline.md`).
2. The pointwise resampling threshold is **not a fix, at any ratio tested, once recall is
   read against its chance level.** At `ratio = 1` and `3` its raw recall numbers look like
   a large improvement, but the genuine above-chance edge is modest (+0.12 to +0.16) and
   comparable to the baseline's own edge over chance — most of the apparent gain is the
   arithmetic consequence of predicting 400+ change points into a series with 16 true
   onsets. At `ratio = 10`, the reviewer's own sharpest case, it is at or below chance on
   both classes: its predictions carry no measurable localisation signal there at all. This
   supersedes the more charitable "helps at moderate imbalance" reading in an earlier draft
   of this file, and adds a **third** documented failure mode to the two already in
   `results/resampling_threshold/TODO.md` (no transfer across `w`; fails under
   autocorrelation).
3. Recommend citing this experiment in the appendix as: the reviewer's mechanism is real,
   targeted, and now directly measured against a chance baseline; the suggested resampling
   fix does not provide a real remedy for it in this test — its apparent gains are largely
   or entirely explained by loss of selectivity, not by recovered localisation. This is a
   *stronger* negative result for resampling than the appendix's existing "validated only
   at a stated operating point" framing (§A.6) allows for — worth tightening that language
   once this file is incorporated.

## Caveats

- One trajectory shape, one `(delta, b, w)` setting. `q = 0.95` / `q_null = 0.5` are the
  manuscript's defaults, not re-tuned for this generator — §4 explains why that matters for
  the `ratio = 1` row specifically.
- 20 reps per ratio (`std` on baseline small-recall: 0.173, 0.102, 0.000 for ratio 1, 3,
  10) — the `ratio = 10` result is exact across replicates; the `ratio = 1` and `3` numbers
  carry more spread and should be read as means.
- Only the pointwise permutation null was tested here, not the block bootstrap
  (`results/block_permutation_null/`) — the data is iid within state by construction (no
  autocorrelation to preserve), so the block null would reduce to the pointwise null at
  `L = 1`, per `results/block_permutation_null/FINDINGS.md` finding 4 (toy control).
- Example figures show one representative replicate per ratio (the rep whose baseline
  small-recall is closest to that ratio's mean, not rep 0 arbitrarily — an earlier draft
  used rep 0 unconditionally and it happened to be a favourable outlier at `ratio = 3`).
  Aggregate numbers are always from the full 20-rep CSV, not the plotted replicate.
- The chance-level control (§2) places random predictions uniformly across the whole
  trajectory. It does not control for a weaker (and more charitable to the methods) null in
  which random predictions are restricted to land only near *some* true change point rather
  than anywhere in the series — that would raise every chance estimate and shrink the
  above-chance margins further. The uniform control is the more standard and more
  conservative choice, and it is already enough to overturn the raw-recall reading; a
  location-restricted control was not run.
- 400 trials per chance estimate (`analyze_imbalanced_chance.py`); the resulting chance
  recalls carry a Monte Carlo standard error of roughly 0.01-0.02, small relative to the
  effects reported in §2, but the `ratio=10` above-chance margins (-0.03 to -0.04) are the
  same order of magnitude as that error and should be read as "not detectably above
  chance," not as a precisely estimated negative number.
