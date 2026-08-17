# Distributional fidelity — FINDINGS

A metric that measures what the manuscript actually claims about k-means, which V-measure
structurally cannot. Scripts: `mfpy/experiments/distributional_fidelity.py`,
`mfpy/experiments/analyze_distributional_fidelity.py`. Raw output: `distfid_toy.csv` (300
rows), `distfid_langevin.csv` (5 rows), `distfid_langevin_per_cluster.csv`. Runtime ~15s.

## Why this exists

V-measure (`results/vmeasure/FINDINGS.md`) is a function of the label contingency table
and nothing else — it cannot see the values, the geometry, or the shape of the recovered
distributions. A method that separates the states with a hard threshold in `x` scores well
even when the resulting clusters misrepresent each state's law. That is precisely the
k-means behaviour the manuscript objects to (figures `toy-kmeans-distributions`,
`langevin-kmeans-distributions`), and precisely what V-measure cannot charge for.

This measures the claim in the geometry the method is built on: the W2 distance between
each recovered cluster's empirical law and each true class's empirical law.

## The metric

With true class laws `mu_i` (mass `p_i`) and recovered cluster laws `nu_j` (mass `w_j`):

- **fidelity** = `sum_j w_j min_i W2(nu_j, mu_i)` — "does every cluster look like some true
  state?" The distributional analogue of homogeneity. A truncated slab of the value axis is
  far from every true law and is penalised here.
- **coverage** = `sum_i p_i min_j W2(mu_i, nu_j)` — "is every true state represented by
  some cluster?" The distributional analogue of completeness.

Both are normalised by the W2 distance between the two metastable state laws, so a reported
value of 0.10 means "off by a tenth of the A-to-B separation". Neither direction is a score
to optimise alone — many tiny clusters drive fidelity down while leaving coverage high, and
merging does the reverse — so both are reported, with cluster counts alongside. Clusters
with fewer than 10 points are excluded, since their empirical law is not estimable.

## 1. Langevin: the manuscript's qualitative claim, confirmed quantitatively

Ground truth is the **basin** definition: state A is `x < 0`, state B is `x > 0`, split at
the barrier. This is the only non-arbitrary choice, and it matters — see section 3.

`tables/table5_langevin_per_cluster.md`:

| method | cluster | mass | mean | std | nearest state | W2 to nearest |
|---|---|---|---|---|---|---|
| ADP | 0 | 0.484 | 0.861 | 0.339 | B | **0.061** |
| ADP | 1 | 0.421 | -0.904 | 0.323 | A | **0.083** |
| ADP | 2 | 0.096 | -0.087 | 0.434 | A | 0.767 |
| k-means | 0 | 0.417 | 0.962 | 0.235 | B | 0.184 |
| k-means | 1 | 0.403 | -0.959 | 0.252 | A | 0.160 |
| k-means | 2 | 0.180 | 0.071 | 0.291 | B | 0.759 |

The two well clusters recovered by segment clustering are **2-3x closer to the true basin
laws** than k-means' (0.061 / 0.083 vs 0.184 / 0.160). Aggregated
(`table4_langevin_core_sweep.md`, `c = 0` row): fidelity 0.082 vs 0.166, coverage 0.043 vs
0.103 — a factor of 2 on both directions.

The mechanism is visible in the standard deviations. The true basin laws have mass running
all the way to the barrier; ADP's clusters reproduce that (sd 0.32-0.34), while k-means'
are truncated (sd 0.24-0.25) because a hard threshold at `x = +/-0.5` cuts each well's
inner tail off and hands it to the middle cluster. `figures/langevin_cluster_laws.pdf`
shows this directly: the k-means clusters have sharp vertical cut edges, the segment
clusters overlap and carry their tails across the barrier.

**This is the figure and the table to put in the paper.** It is the quantitative version of
the histogram figures already there, and it makes the "hard dividing hyperplanes" argument
without relying on the reader's eye.

## 2. Toy: k-means wins on detected change points, loses on true ones

`tables/table3_win_rate_vs_kmeans.md`, fraction of trajectories where each variant is
distributionally closer to the true laws than k-means:

| sigma | n | ADP (detected CPs) | ADP (true CPs) |
|---|---|---|---|
| 5  | 100 | 0.00 | **0.93** |
| 20 | 100 | 0.45 | **0.64** |

Mean normalised fidelity error (`table1_fidelity_by_sigma.md`):

| sigma | ADP detected | ADP true CPs | k-means |
|---|---|---|---|
| 5  | 0.128 | **0.012** | 0.030 |
| 20 | 0.111 | **0.058** | 0.092 |

Same conclusion as the V-measure experiment, reached independently: **with correct change
points the Wasserstein clustering is distributionally better than k-means (93% / 64% of
trajectories); with the current fixed-quantile threshold it is not.** Localisation error is
what costs us the comparison, not the clustering.

Two secondary points, both consistent with the mechanism:

- ADP-on-detected does beat k-means at sigma = 20 with short ramps (0.064 vs 0.099 for
  `ell <= 40`), and loses as ramps lengthen (`table2`). Short ramps are where localisation
  is accurate.
- k-means' error is nearly flat in `ell` but rises sharply with noise (0.030 -> 0.092),
  because truncation only loses mass where the two laws overlap. At sigma = 5 the Laplace
  states at 100 and 200 barely overlap, so truncating at the midpoint is almost lossless
  and k-means is near-exact by construction. **The toy data is close to a best case for
  k-means and should not be used to argue against it.**

## 3. The Langevin core-set sweep is a trap — read only the c = 0 row

`table4_langevin_core_sweep.md` sweeps the core half-width `c` used to define the reference
labels:

| c | ADP fidelity | k-means fidelity |
|---|---|---|
| 0.0 | **0.082** | 0.166 |
| 0.2 | 0.048 | 0.066 |
| 0.4 | 0.065 | **0.020** |
| 0.6 | 0.088 | **0.026** |
| 0.8 | 0.120 | **0.072** |

The apparent crossover at `c >= 0.4` is an artefact, not a result. For `c > 0` the
reference laws are *themselves* defined by thresholding `x` (state A is `x < -c`), so they
are truncated distributions — and k-means produces truncated distributions. The comparison
becomes partly tautological, exactly as it did for V-measure on the same data.

Only `c = 0` avoids this: the barrier at the origin is a property of the potential, not a
tuning knob, and the resulting basin laws are un-truncated. **Report the `c = 0` row; the
sweep belongs in the record as the reason for that choice, not as a result.**

A committor- or MSM-based reference labelling (via `deeptime`) would remove the issue
entirely and is the right fix if a reviewer presses on it.

## 4. Under the pointwise resampling threshold: the toy comparison flips

Re-run with `--threshold resample_pointwise --q 0.5 --R 199`. Outputs:
`distfid_toy_resample.csv`, `distfid_langevin_resample.csv`,
`distfid_langevin_per_cluster_resample.csv`.

**Toy**, paired over the 200 trajectories with sigma > 0 that succeed under both
thresholds (normalised error, lower is better):

| | fixed q = 0.95 | pointwise null, q = 0.5 | oracle segmentation |
|---|---|---|---|
| fidelity, 3-class | 0.120 | **0.058** | 0.035 |
| coverage, 3-class | 0.165 | **0.061** | 0.035 |
| fidelity, 2-class | 0.165 | **0.060** | 0.035 |

The error roughly halves and lands close to the oracle-segmentation ceiling; the
resampling threshold is better on 72% of trajectories. Coverage improves most (0.165 ->
0.061), consistent with the V-measure result that the method starts finding the third
(transition) cluster rather than collapsing to two.

**The head-to-head with k-means flips.** Fraction of trajectories on which segment
clustering is distributionally closer to the true laws than k-means:

| sigma | fixed q = 0.95 | pointwise null |
|---|---|---|
| 5  | 0.00 | **0.53** |
| 20 | 0.45 | **0.52** |

At sigma = 5 this goes from never winning to winning on a majority. Taken with section 2,
the picture is consistent: the gap to k-means was a segmentation artefact, and roughly
half of it closes with a better threshold and all of it closes with perfect change points.
This is still not a decisive win over k-means on the toy data, and should not be reported
as one — see section 2 on why the toy set flatters k-means.

**Langevin**, at the basin definition `c = 0`: fidelity 0.082 -> 0.099 and coverage
0.043 -> 0.097, i.e. slightly *worse*, and ADP drops from 3 clusters to 2 (the transition
cluster is lost). This is the same oversegmentation regression documented in
`results/vmeasure/FINDINGS.md` §6, with the same cause — the permutation null assumes
exchangeability, which the autocorrelated Langevin dynamics violate.

Note that even in the regressed state, the two recovered well clusters remain closer to
the true basin laws than k-means' (0.218 / 0.112 vs 0.160 / 0.184 per cluster; aggregate
fidelity 0.099 vs 0.166). The distributional advantage over k-means survives a
segmentation that V-measure calls a clear regression, which is itself a useful
illustration of the difference between the two metrics.

## Status / what is still shaky

- Section 1 (Langevin, `c = 0`) is a single trajectory at a single `(w, q)`. The effect is
  large (2x) and mechanistically explained, but it is n = 1.
- Section 2 rests on 200 toy trajectories and is solid.
- Not tested: whether the pointwise resampling threshold closes the toy gap. Given that
  the oracle-segmentation variant wins 93% / 64%, it should move a long way.
- The `min_i` in both directions is a nearest-law assignment, not a one-to-one matching.
  With `k = 3` and 3 classes the two coincide in every case inspected, but a method that
  emitted many clusters could in principle exploit this. Cluster counts are reported so the
  reader can check.
- sigma = 0 is degenerate for ADP for the reasons given in `results/vmeasure/FINDINGS.md`;
  11 of 100 trajectories fail outright and the rest should not be read.
