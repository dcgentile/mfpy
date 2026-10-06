# Block permutation null — FINDINGS

Item A1 of `results/resampling_threshold/TODO.md`: the candidate repair for the Langevin
failure in `results/vmeasure/FINDINGS.md` §6. Scripts: `block_permutation_null.py`,
`block_null_vmeasure.py`, `langevin_clustering_output.py`,
`analyze_block_permutation_null.py`. Output: `*.csv`, `tables/`, `clusterings/`,
`figures/`.

## Headline

Replacing the full shuffle with a circular-block bootstrap at `L ≈ 60` recovers the
Langevin result: cutoff at the 0.861 quantile of the observed statistic (fixed rule:
0.859), 156 segments (157), ADP `k = 3`, V₂ = 0.685 (0.685). `L = 1` reduces to the
published full shuffle and reproduces its toy F₁. **Parity on the Langevin, not
improvement** — the value is that the cutoff is calibrated on data manufactured to
contain no change, so it cannot be inflated by the changes present in the real series.
That is the reviewer's actual objection in 1(b): multiple change points bias the quantile
of the observed statistic upward, and small changes are then missed.

`L` is the new parameter, but unlike `q` it is estimable: the correct value is the
**within-state** integrated autocorrelation time — ~1 for the toy generator, 52–75 within
the Langevin wells — and those are the values that work.

## 1. Langevin alignment and segmentation

`tables/table1_langevin_alignment.md`, `w = 215`:

| threshold | cutoff | as quantile of observed statistic | segments |
|---|---|---|---|
| fixed `q = 0.86` (incumbent) | 0.660 | 0.859 | **157** |
| full shuffle `q = 0.5` (published) | 0.107 | **0.140** | 466 |
| block `L = 25` | 0.472 | 0.754 | 263 |
| block `L = 50` | 0.619 | 0.840 | 172 |
| block **`L = 60`** | 0.664 | **0.861** | **156** |
| block `L = 75` | 0.712 | 0.881 | 134 |
| block `L = 200` | 0.823 | 0.920 | 98 |
| block `L = 800` | 0.384 | 0.668 | 320 |

The failure is mechanical: the full shuffle's cutoff sits at the 14th percentile of the
observed statistic where the fixed rule sits at the 86th, so most of the trajectory is
supra-threshold and runs merge. Restoring within-well dependence raises the null and the
cutoff climbs with `L`.

**Non-monotone past `L ≈ 200`:** once blocks exceed the median dwell time (~1141), the
null replicate carries whole metastable segments and stops being a no-change null. The
usable range is bounded below by the within-state correlation time and above by the dwell
time — wide here (60 vs. 1141), not guaranteed in general. Circular and moving schemes
agree to 3 d.p. for `L ≤ 400` (`tables/table2_scheme_robustness.md`).

## 2. Choosing `L`

| series | τ_int, whole trajectory | τ_int **within a state** |
|---|---|---|
| Langevin | 1676 | **52–75** (median over 24 within-well runs ≥ 1000) |
| toy σ = 0 / 5 / 20 | — | **1.00 / 1.07 / 0.83** |

The Langevin's whole-trajectory τ_int is a *metastability* time dominated by the
switching the method is meant to detect; using it as `L` lands in the degenerate region.
The within-state rule selects `L = 1` on toy and `L ≈ 60` on Langevin — both the
empirically best values. This is what makes the choice defensible rather than fitted.

## 3. Langevin clustering

`tables/table4_langevin_vmeasure.md` and `clusterings/langevin_cluster_composition.csv`.
`frac_core` = fraction of a cluster's points with |x| < 0.4, which identifies the
transition state (~0.1 for a well, ~0.65+ for the transition):

| threshold | segments | k | V₂ | cluster sizes | frac_core |
|---|---|---|---|---|---|
| fixed `q = 0.86` | 157 | **3** | 0.685 | 48373 / 42051 / 9576 | 0.10 / 0.07 / **0.68** |
| full shuffle | 466 | 2 | 0.621 | 50060 / 49940 | 0.14 / 0.14 |
| block `L = 25` | 263 | 3 | 0.590 | — | — |
| block `L = 50` | 172 | 2 | 0.711 | 48929 / 51071 | 0.10 / **0.18** |
| block **`L = 60`** | 156 | **3** | 0.685 | 49212 / 42050 / 8738 | 0.11 / 0.07 / **0.68** |
| block `L = 75` | 134 | **3** | 0.693 | 50307 / 42423 / 7270 | 0.12 / 0.08 / **0.65** |
| block `L = 100` | 112 | 2 | 0.494 | — | — |

At `L = 60` the partition is structurally the incumbent's, not merely score-equivalent.

**`L = 50`'s higher V₂ (0.711) must not be quoted as the headline:** it absorbs the
transition into the lower well (frac_core 0.18 vs. 0.10), and the two-class target does
not penalise losing a state the manuscript is about. The full shuffle fails differently,
scattering transition points into both clusters (0.14 / 0.14).

`L` matters — 0.590 at `L = 25`, 0.494 at `L = 100`. The plateau is roughly `[40, 75]`.

## 4. Toy control

`tables/table3_toy_by_block_length.md`, stratified 60, `w = 25`, mean F₁ at `τ = 10`:

| threshold | F₁ | precision | recall | predicted CPs (truth 38) |
|---|---|---|---|---|
| fixed `q = 0.95` (manuscript setting) | 0.328 | 0.334 | 0.323 | 48.2 |
| full shuffle `q = 0.5` (published) | 0.664 | 0.650 | 0.710 | 48.9 |
| block `L = 1` | **0.665** | 0.651 | 0.709 | 48.8 |
| block `L = 5` | 0.408 | 0.412 | 0.409 | 36.4 |
| block `L = 25` | 0.321 | 0.329 | 0.314 | 22.6 |
| block `L = 50` | 0.504 | 0.507 | 0.508 | 43.4 |

`L = 1` reproduces the full shuffle to 0.001 F₁ — the block harness is a strict
generalisation of the published code (0.664 also matches the 0.666 for this same
stratified 60 in `results/qw_interaction/FINDINGS.md`). Blocking *hurts* on toy data,
as it should: the generator is iid within each state, so `L > 1` is a misspecified null.
The fixed-quantile row is at the manuscript's `q = 0.95`, not its best global `q = 0.75`
(F₁ 0.626 in `results/resampling_threshold/FINDINGS.md`).

## 5. `R` is not a live parameter

At `L = 60`, two seeds: cutoff 0.669→0.664 and 156 segments across `R ∈ {10, 30, 60,
120}`. The pooled null aggregates ~99,600 values per replicate, so its median is
converged by `R = 10`. Closes TODO item C7 for the pointwise null (not for the max-type
null, which is a genuine extreme-value estimate).

## Consequences

1. Reviewer 1(b) can be answered affirmatively with a caveat: mechanism confirmed
   independently, literal E-Divisive port fails structurally, pointwise full shuffle works
   only under within-state exchangeability, block null fixes that with `L` read off the
   data — but on the Langevin it matches rather than beats the fixed quantile. Note the
   imbalanced-change-size case the reviewer gives as the sharp form of the argument is
   still untested (TODO B4).
2. **The `(w, q)` fragility from `results/qw_interaction/FINDINGS.md` is untouched.**
   One `w` per dataset here (215 / 25). Transfer across `w` is the next check.
3. Item A2 (self-calibrating variant) is now less attractive: it would set the cutoff from
   a quantile of the observed statistic, which is the quantity the reviewer showed to be
   upward-biased under multiple changes.

## Still shaky

- Langevin is **one trajectory** with a proxy ground truth. TODO item B6 matters more now.
- One `w` per dataset.
- ~~`L` needs a state assignment to estimate~~ **CLOSED.** `auto_block_length.py` now
  estimates `L` automatically from a fixed-quantile pilot segmentation; see
  `results/auto_block_null/MEMO-block-resampling.md` §3. It selects `L = 1` on every toy
  case and `L = 51` on the Langevin, both correct.
- **The block null overshoots.** Its median (0.624) sits above the observed statistic's
  86th percentile: block joins create artificial discontinuities that inflate it. `L ≈ 51`
  works because that inflation cancels the dependence loss, not because the null is well
  calibrated. See MEMO §5 — this is the most serious open weakness.
- No process tried with a short dwell time relative to its within-state correlation.
- `k` flips between `L = 50` and `L = 60`; ADP is near a boundary and sensitivity to `Z`
  (fixed at 1.65) was not probed.
- Block bootstrap resamples with replacement, the published null permutes without; at
  `L = 1` they agree to 0.001 F₁ and ~1 segment.
