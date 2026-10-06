# Block resampling for the CPD threshold: definitions, literature, and status

Companion to `results/block_permutation_null/FINDINGS.md` and
`results/auto_block_null/summary.csv`. Written for the FODS revision; the intended
destination is a subsection of the numerical-experiments section plus a paragraph in the
response letter.

---

## 0. What the reviewer actually said

Reviewer major comment 1(b), verbatim:

> Threshold selection: using the quantile of the test statistic itself as the threshold is
> not appropriate in the multiple change point setting, as the presence of multiple change
> points will upwardly bias the quantile. This will be particularly apparent when the
> change sizes themselves are imbalanced: for example, if there are 2 (instantaneous)
> changes in the mean, and one is much larger than the other, the quantile-based threshold
> will be set too high to detect the smaller change. A simple work-around (whilst
> computationally more expensive), is to use a resample method; see e.g. Matteson and
> James 2014.

The complaint is **upward bias of the quantile under multiple changes**, with a concrete
consequence (small changes missed when change sizes are imbalanced). It is not a complaint
about tuning `q` against the answer. `results/toy_sweep/FINDINGS.md` finding 4 confirms the
reviewer's mechanism independently: the empirically optimal quantile tracks
`q* = 1 − (fraction of the trajectory in transition)`, so as more of the series is elevated
the cutoff rises and change points land further inside each ramp.

Two consequences for how we write this up. The imbalanced-change-size case the reviewer
gives as the sharp version of the argument **has still not been tested** (TODO item B4);
it is the most direct evidence either way and should be run. And the argument for
resampling should be made in the reviewer's terms — a threshold calibrated on data
containing no change cannot be inflated by the changes present in the real data — rather
than in terms of parameter tuning.

---

## 1. Definitions

Let `x = (x_1, …, x_T)` be the observed series and `γ̇_t(w)` the windowed 2-Wasserstein
statistic: the `W₂` distance between the empirical measures of `x_{t−w:t}` and
`x_{t:t+w}`. The pipeline thresholds `γ̇` at a cutoff `c`, takes maximal supra-threshold
runs, and extracts two change points per run.

**Incumbent (fixed quantile).** `c = Q_q(γ̇)`, the `q`-quantile of the observed statistic.
This is what the reviewer objects to.

**Permutation null (published).** Draw `R` permutations `x^(r)` of `x`, compute `γ̇^(r)`
on each, pool all values over `r` and `t`, and set `c = Q_{q_null}(pooled)`. The null is
complete exchangeability — "no change anywhere" — so the cutoff cannot be inflated by the
changes in the real data. Adopted at `q_null = 0.5` in
`results/resampling_threshold/FINDINGS.md`.

**Block bootstrap null (this work).** Identical, except the replicate is built from
contiguous blocks rather than individual points. Fix a block length `L` and let
`k = ⌈T/L⌉`.

- *Moving-block* (Künsch 1989): draw `i_1, …, i_k` iid uniform on `{1, …, T−L+1}` and
  concatenate `x_{i_j}, …, x_{i_j+L−1}`, truncating to length `T`.
- *Circular-block* (Politis & Romano 1992): the same, but indices are taken mod `T` on the
  series wrapped end to end, so every observation appears in exactly `L` candidate blocks
  and there is no edge deficit.

`L = 1` recovers an iid resample of the values, i.e. the published null up to
sampling with rather than without replacement. Both schemes agree to three decimal places
on everything we measured for `L ≤ 400` (`block_permutation_null/tables/table2`); we use
circular by default.

**Figure `fig_construction.{pdf,png}`** shows all three constructions on the Langevin
trajectory with their autocorrelation functions: identical marginal distributions, but the
full shuffle's ACF drops to zero at lag 1 while the block replicate's decays to zero at
roughly lag `L`. That is the whole mechanism.

---

## 2. Why exchangeability is the wrong null here

A full shuffle is a valid "no change" null only if the process is iid within each state.
That holds for the toy generator (independent Laplace draws inside each block) and fails
for any SDE trajectory, where consecutive points within a well are strongly correlated.
Destroying that correlation makes the two half-window empirical measures far more alike
than they ever are in the real data, so the null statistic collapses and the cutoff lands
far too low.

Measured on the Langevin at `w = 215`:

| | cutoff | as a quantile of the observed statistic | segments |
|---|---|---|---|
| fixed `q = 0.86` | 0.660 | 0.859 | 157 |
| full shuffle, `q_null = 0.5` | 0.107 | **0.140** | 466 |
| block `L = 51` (automatic) | 0.623 | 0.842 | 172 |

Six sevenths of the trajectory sits above the full-shuffle cutoff; runs merge and the
segmentation shatters. See `fig_null_shift.{pdf,png}`.

---

## 3. Choosing `L`, and the two failure modes

`L` is bounded on both sides (`fig_block_tradeoff.{pdf,png}`):

- **`L` too small** — within-state dependence is destroyed, the null collapses, and we are
  back to the full-shuffle failure.
- **`L` too large** — blocks carry whole metastable states, so the replicate contains real
  change structure and stops being a no-change null at all. On the Langevin this shows up
  as the cutoff falling again past `L ≈ 200` and the segment count climbing back to 320 at
  `L = 800`.

So the target is: longer than the **within-state** correlation time, shorter than the
dwell time. On the Langevin those are ~50–75 and ~1141 respectively.

**The estimator (`auto_block_length.py`), fully automatic:**

1. Segment the series with the incumbent fixed-quantile rule at the manuscript's own
   `(w, q)`. The pilot need not be good, only frequent enough that most segments lie
   inside one state — which is the fixed rule's behaviour, since its documented failure is
   over-cutting transitions, not merging states.
2. Trim `w` points from each end of every pilot segment. **This step is essential and is
   not a tuning knob:** the pipeline's known boundary bias (`toy_sweep` finding 2) puts
   detected change points *inside* transitions, so both ends of every pilot segment carry
   ramp material, and a ramp is smooth and therefore strongly autocorrelated. Untrimmed,
   the estimator returns `τ_int` up to 10 on toy data whose within-state process is iid.
   `w` is the natural trim because it is the scale the statistic is computed on and hence
   the resolution limit of a detected boundary.
3. Estimate `τ_int` on each remaining interior of length `≥ max(50, 8·τ̂)` by Sokal
   windowing, and take the median.
4. Set `L = round(τ_int)`, build the block null, re-threshold.

Estimates produced, against the whole-series value:

| dataset | `τ_int` whole series | `τ_int` within segments | `L` | correct? |
|---|---|---|---|---|
| toy, σ=5, ℓ=6 | 66 | 0.78 | **1** | yes (iid within state) |
| toy, σ=5, ℓ=48 | 143 | 0.92 | **1** | yes |
| toy, σ=20, ℓ=48 | 67 | 0.91 | **1** | yes |
| toy, σ=20, ℓ=95 | 72 | 1.46 | **1** | yes |
| Langevin | **1676** | 50.6 | **51** | in the good plateau [40, 75] |
| Prinz | 115 | 2.12 | **2** | plausible (sampled every 500 integrator steps) |

The Langevin row is the point of the whole section: the whole-trajectory `τ_int` of 1676
is a *metastability* time dominated by the well-to-well switching the method exists to
detect. Any global stationary-series block-length selector returns that number, and it is
in the degenerate region. Segmenting first is what makes the estimate mean the right thing.

---

## 4. Results, fully automatic

`results/auto_block_null/summary.csv`, `cluster_composition.csv`, and
`figures/clustering_*.{pdf,png}`. Three rules, same `(w, q)` per dataset, no hand-set `L`.

**Toy** (`w = 25`, `q = 0.95`, V₂ against the two-class target):

| case | fixed | full shuffle | auto (`L`) |
|---|---|---|---|
| σ=5, ℓ=6 | 0.774, k=3 | 0.629, k=3 | 0.629, k=3 (`L`=1) |
| σ=5, ℓ=48 | 0.854, k=2 | 0.816, k=3 | 0.816, k=3 (`L`=1) |
| σ=20, ℓ=48 | 0.539, k=2 | 0.816, k=4 | 0.816, k=4 (`L`=1) |
| σ=20, ℓ=95 | 1.000, k=3 | 0.995, k=2 | 0.995, k=2 (`L`=1) |

`L = 1` in every toy case, so `auto` and `shuffle` coincide — the correct behaviour on
iid-within-state data, and the check that the automatic machinery does not invent
dependence that is not there. Note that per-trajectory the fixed quantile wins on three of
four here; the resampling advantage reported in `resampling_threshold/FINDINGS.md` is a
mean over 300 trajectories at matched global `q`, not a guarantee per case.

**Langevin** (`w = 215`, `q = 0.86`):

| rule | `L` | segments | k | V₂ | cluster sizes | frac_core |
|---|---|---|---|---|---|---|
| fixed | — | 157 | **3** | 0.685 | 48373 / 42051 / 9576 | 0.10 / 0.07 / **0.68** |
| full shuffle | 1 | 466 | 2 | 0.621 | 50060 / 49940 | 0.14 / 0.14 |
| **auto** | **51** | 172 | **3** | **0.687** | 48997 / 41990 / 9013 | 0.10 / 0.07 / **0.69** |

`frac_core` is the fraction of a cluster's points with |x| < 0.4, i.e. between the wells;
it identifies the transition state. The automatic block null recovers the incumbent's
partition — two wells and a transition state of comparable size and composition — where
the full shuffle loses the transition state entirely.

**Prinz** (`w = 16`, `q = 0.5`, four-well potential, no ground truth). Cluster means
against the true minima at −0.739, −0.224, 0.269, 0.673:

| rule | `L` | segments | k | cluster means |
|---|---|---|---|---|
| fixed | — | 532 | 6 | −0.63, −0.26, −0.00, **0.36 (7 points)**, 0.36, 0.69 |
| full shuffle | 1 | 390 | 5 | −0.61, −0.26, 0.05, 0.36, 0.69 |
| **auto** | **2** | 329 | 5 | −0.64, −0.25, 0.10, 0.37, 0.70 |

All three find the four wells. `auto` and `shuffle` return four wells plus one small
transition cluster; the fixed quantile additionally emits a degenerate 7-point cluster.

---

## 5. The block-join bias — the honest caveat

The block null is **not** an unbiased "no change" null, and the mechanism is not as clean
as §2 implies. Concatenating blocks drawn from unrelated parts of the trajectory creates an
artificial discontinuity at every join. A window of `2w = 430` points spans about eight
joins at `L = 51`, so the half-window measures differ more than they would under any real
no-change process. Measured on the Langevin:

| | mean | median | 90th pct | max |
|---|---|---|---|---|
| observed statistic | 0.353 | 0.269 | 0.758 | 1.644 |
| full-shuffle null | 0.122 | 0.106 | 0.206 | 0.561 |
| block null, `L = 51` | 0.681 | **0.624** | 1.145 | 2.173 |

The block null's median sits *above* the observed statistic's 86th percentile. So `L ≈ 51`
does not work because the null is correctly calibrated; it works because two errors of
opposite sign — destroyed within-state dependence pushing the null down, block-join
discontinuities pushing it up — happen to cancel near the right cutoff. That also explains
the sharp sensitivity to `L` (`k` flips between `L = 50` and `L = 60`) which a
well-calibrated null would not show.

This must be stated in the paper. It is a real weakness, and a referee who knows the
bootstrap literature will spot it immediately. Two standard remedies exist and neither has
been tried: the **matched-block bootstrap** (Carlstein et al. 1998), which chooses each
successive block so that its start is a plausible successor of the previous block's end,
and the **tapered block bootstrap** (Paparoditis & Politis 2001), which down-weights block
edges. Either would attack the join bias directly. The **stationary bootstrap** (Politis &
Romano 1994), with geometric random block lengths, does not fix joins but removes the
sensitivity to a single `L`, and is cheap to test.

---

## 6. Why not Politis & White's automatic selector

Politis & White (2004) is the standard data-driven answer to "how long should the blocks
be" and should be cited, but it is not directly usable here. It targets the block length
minimising the MSE of a *variance estimator for the sample mean under stationarity*,
whereas what this pipeline needs is the shape of the null distribution of a windowed
two-sample statistic. More importantly it is a global estimator: applied to a series
containing change points it inherits exactly the contamination in §3 — it would see the
Langevin's `τ_int = 1676`. The segment-first construction is the adaptation that makes the
estimate mean what we need. Worth one sentence in the paper and a citation, not an
implementation.

---

## 7. References

Verify page ranges against the publishers before these go into the `.bib`; the volume,
issue and year fields below are confirmed, the page ranges for Carlstein et al. and
Politis & White are from secondary sources.

1. Künsch, H. R. (1989). The jackknife and the bootstrap for general stationary
   observations. *Annals of Statistics* **17**(3), 1217–1241. — origin of the moving-block
   bootstrap.
2. Politis, D. N. & Romano, J. P. (1992). A circular block-resampling procedure for
   stationary data. In LePage & Billard (eds.), *Exploring the Limits of Bootstrap*, Wiley,
   263–270. — the circular scheme used here.
3. Politis, D. N. & Romano, J. P. (1994). The stationary bootstrap. *JASA* **89**(428),
   1303–1313. — random geometric block lengths; removes sensitivity to a single `L`.
4. Carlstein, E., Do, K.-A., Hall, P., Hesterberg, T. & Künsch, H. R. (1998).
   Matched-block bootstrap for dependent data. *Bernoulli* **4**(3), 305–328. — the
   standard fix for the join bias in §5.
5. Paparoditis, E. & Politis, D. N. (2001). Tapered block bootstrap. *Biometrika*
   **88**(4), 1105–1119. — down-weights block edges; the other fix for §5.
6. Politis, D. N. & White, H. (2004). Automatic block-length selection for the dependent
   bootstrap. *Econometric Reviews* **23**(1), 53–70. Correction: Patton, Politis & White
   (2009), *Econometric Reviews* **28**(4), 372–375. — the standard automatic selector;
   see §6.
7. Lahiri, S. N. (2003). *Resampling Methods for Dependent Data*. Springer Series in
   Statistics. — textbook treatment of all of the above.
8. Matteson, D. S. & James, N. A. (2014). A nonparametric approach for multiple change
   point analysis of multivariate data. *JASA* **109**(505), 334–345. — the reviewer's
   citation. Methodology also in James & Matteson (2015), *JSS* **62**(7),
   arXiv:1309.3295.
9. Prinz, J.-H. et al. (2011). Markov models of molecular kinetics: generation and
   validation. *J. Chem. Phys.* **134**(17), 174105. — the Prinz potential.

---

## 8. Status and what to do next

Done: block null implemented and validated; `L` selected automatically with no
per-dataset hand tuning; end-to-end clusterings on toy, Langevin and Prinz.

Open, in priority order:

1. **The reviewer's imbalanced-change-size case** (TODO B4) — still untested, and it is the
   sharp form of their argument.
2. **Transfer across `w`** — everything here is at one `w` per dataset. This is what killed
   the full-shuffle recommendation (`qw_interaction/FINDINGS.md`) and there is no reason to
   assume the block null is exempt.
3. **The join bias** (§5) — try the tapered or matched-block variant, or at minimum report
   the bias honestly and show the stationary bootstrap's `L`-insensitivity.
4. `data/prinz/prinz.txt` was missing from the repo and has been **regenerated**
   (`make_prinz_data.py`, deeptime defaults, seed 42). Numbers on it will not match
   anything computed from the original file; the old notebook figures should be recomputed
   rather than compared.
