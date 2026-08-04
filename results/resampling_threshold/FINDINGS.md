# Resampling threshold — FINDINGS

Answers action item 2 / reviewer major comment 1(b) ("using the quantiles is not
appropriate ... simple work around ... use a resampling method; see e.g. Matteson and
James 2014"). Script: `mfpy/experiments/resampling_threshold.py`. Data: all 300
`data/Toy_Model_Trajectories` trajectories, `w=25`, `estimator="gradient"` (matching
`toy_threshold_sweep.py`), `R=199` permutations per trajectory. Raw output:
`results/resampling_threshold/resampling_threshold.csv` (9900 rows = 300 trajectories x
11 q values x 3 methods).

## 1. The paper, and why it's relevant

Matteson & James (2014, *JASA* 109(505):334-345, "A Nonparametric Approach for Multiple
Change Point Analysis of Multivariate Data") is not on arXiv as a standalone preprint; its
methodology — energy-distance statistic, hierarchical bisection ("E-Divisive"), and a
permutation significance test — is described in full, with worked examples, in the
companion open-access paper James & Matteson (2015, *JSS* 62(7), the `ecp` package paper,
arXiv:1309.3295), which is what this experiment is built from.

E-Divisive is architecturally unrelated to our pipeline (energy-distance bisection over
segments vs. a sliding-window Wasserstein derivative thresholded at a quantile), so the
method cannot be swapped in directly. But its threshold-calibration idea is exactly on
point. E-Divisive never reads a quantile off the observed statistic. For each candidate
change point it permutes the observations *within the current segmentation* — a null of
complete exchangeability, i.e. "no change here" — recomputes the statistic on the
permutation, repeats R times, and accepts the candidate only if its real statistic beats
enough of that null distribution (their Algorithm 3). The threshold is calibrated from
data manufactured to contain no change, so it is blind to how much of the real series is
elevated.

That is precisely the mechanism reviewer 1(b) is pointing at, and precisely what
`results/toy_sweep/FINDINGS.md` finding 4 diagnosed independently: our `q=0.95` cutoff is
a quantile of the *observed* statistic, so as more of the trajectory sits in transition
(longer ramps), the quantile is dragged upward and change points land further inside each
ramp. **Verdict: relevant, not a false alarm.** The fix is actionable without adopting
E-Divisive wholesale — replace "quantile of the observed statistic" with "quantile of a
resampled null" at the existing threshold step.

## 2. Two translations, tested head to head

Both reuse the exact run-detection / boundary-extraction code in `boundary_estimators.py`
(same `find_runs` / `gradient` extraction), so the three methods below differ *only* in
where the cutoff comes from:

- **`baseline`** — `cutoff = quantile(gammadot_real, q)`. The incumbent.
- **`resample_max`** — draw R permutations of the raw trajectory (full exchangeability),
  compute the windowed statistic on each, take its max over time, and set
  `cutoff = quantile(these R maxima, q)`. This is the direct single-shot analogue of a
  family-wise permutation significance test — the same logic as E-Divisive's per-candidate
  test, but applied once, globally, since our pipeline doesn't bisect.
- **`resample_pointwise`** — same R permutations, but pool *every* statistic value from
  every permutation (not just the max) and set `cutoff = quantile(pooled null values, q)`.
  This calibrates the marginal null distribution of the statistic rather than its extreme
  value.

### `resample_max` fails — and the reason is informative

Across essentially the whole tested range `q in [0.5, 0.99]`, `resample_max` produces
**zero detected change points on 60-72% of trajectories** (vs. 0-18% for baseline). Mean
predicted change points collapses to ~9-13 (true count: 38) regardless of q, and mean
`F1(tau=10)` never exceeds 0.28 at any q, well below baseline's 0.63 peak.

The reason: a max-over-time null implicitly corrects for ~5000-7000 simultaneous test
locations per trajectory (one per window position), so even its *median* (q=0.5) is a
severe Bonferroni-type bound, almost always above the real transition signal. E-Divisive
avoids this because it never takes a max over a whole series — it tests one candidate
split at a time within a shrinking segmentation, so each permutation test only has to
clear a single comparison. A literal "global max" port of their significance test is the
wrong translation for a sliding-window pipeline. This is worth stating explicitly in the
response to reviewers: we implemented the direct analogue of their test and it fails for
an identifiable, structural reason, rather than because resampling doesn't help.

### `resample_pointwise` works, and beats the baseline in the realistic comparison

The pointwise null sidesteps the multiple-testing problem — it asks "how large does the
statistic get, typically, absent change," not "how large does its maximum get." Results,
all at `tau=10`:

| comparison                                                         | baseline         | resample_pointwise |
|--------------------------------------------------------------------|------------------|--------------------|
| best **single global** q (no per-trajectory tuning)                | q=0.75, F1=0.626 | q=0.50, F1=0.681   |
| F1 at q=0.50 specifically                                          | 0.398            | 0.681              |
| localisation-bias slope vs. tl, at each method's own best global q | 0.466 / step     | 0.190 / step       |
| precision / recall at own best global q                            | 0.593 / 0.726    | 0.667 / 0.724      |
| mean predicted change points (truth = 38)                          | 63.4             | 48.4               |
| F1 at sigma=20 (noisiest cell), own best global q                  | 0.326            | 0.428              |
| F1 at sigma=20, q tuned per-sigma                                  | 0.326 (q=0.75)   | 0.517 (q=0.7)      |

The single-global-q comparison is the fair one: in deployment you do not know a
trajectory's transition length in advance, so you pick one q and apply it everywhere. At
that q, `resample_pointwise` beats the best the baseline can do at *its* best global q, on
its own turf, without per-trajectory tuning. The localisation-bias slope with transition
length is cut by more than half (0.190 vs. 0.466 per step), consistent with the mechanism:
the null-quantile threshold no longer inflates as more of the trajectory becomes
"elevated," because it is calibrated on data with no change structure at all. Precision
improves at matched recall (fewer spurious detections: 48 vs. 63 against a true count of
38). The gain is largest in the noisiest regime (sigma=20), where knowing the *marginal*
null level of the statistic — rather than assuming a fixed quantile works everywhere —
matters most.

One honest limitation: allowing **per-trajectory oracle** q-tuning (an undeployable upper
bound, included only as a ceiling) still favours the baseline, 0.852 vs. 0.799 mean
F1(tau=10). And even `resample_pointwise`'s oracle-best q still correlates with transition
length (r=0.612, vs. 0.612 for baseline — statistically indistinguishable) and with the
theoretical `q*` from `toy_threshold_sweep.py` (r=0.61-0.66 for both). So resampling does
not make the *ceiling* dependence on ramp fraction disappear in principle; what it buys is
a threshold that works well at one fixed setting in practice, which is the realistic axis
the reviewer is asking about.

## 3. Recommendation

- Report `resample_max` as a documented negative result with the multiple-testing
  explanation — it demonstrates engagement with the specific citation and explains why a
  literal port of E-Divisive's significance test doesn't fit a sliding-window statistic.
- Adopt `resample_pointwise` (or report it as a validated alternative) as the answer to
  1(b): a nonparametric, resampling-calibrated threshold that removes the need to guess q
  per trajectory, roughly halves the localisation-bias growth with transition length, and
  improves F1 at matched or better precision, with the largest gains at high noise.
- Compute cost: R=199 permutations x 300 trajectories x w=25 completed in well under two
  minutes total using the existing vectorised `fast_metric_derivative_1d`; this is cheap
  enough to run per-trajectory in the real pipeline, not just as a diagnostic.

## Caveats / what's still open

- Only tested at `w=25`; the `q`-`w` interaction (also flagged as open in
  `toy_threshold_sweep.py`) is untouched here.
- `R=199` was not itself swept; sensitivity of the pointwise null's quantile estimate to R
  is unverified (though `ecp`'s own default is comparable, R=199-499).
- The permutation null (full shuffle of raw values) assumes exchangeability under "no
  change," which holds for this generator (independent Laplace draws within each
  metastable state) but would need revisiting for autocorrelated noise.
- Single realisation per (sigma, tl) cell, same caveat as the rest of the toy-sweep work.
