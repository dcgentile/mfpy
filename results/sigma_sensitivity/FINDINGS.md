# Similarity-matrix sigma — FINDINGS

> **Scope note added after the experiment was run.** The submitted results use ADPC only;
> spectral clustering was dropped before submission and its figures are commented out in
> the manuscript source. The similarity-matrix kernel in §2.3 is therefore a holdover that
> should not have been submitted, and the answer to minor comment 6 is a correction rather
> than a sensitivity study — see `paper-resources/rebuttal_notes.md`. **Sections 1–3 below
> describe the spectral path and are out of scope for the paper.** They are kept as due
> diligence, and section 0 and the underflow mechanism in section 2 remain directly
> relevant because they justify deleting the line rather than tuning it.

Answers action item 7 / reviewer minor comment 6 ("How sensitive is the method to tuning
`sigma` in constructing the similarity matrix?"). Scripts:
`mfpy/experiments/sigma_sensitivity.py`, `mfpy/experiments/analyze_sigma_sensitivity.py`.
Raw output: `sigma_toy.csv` (289 trajectories x 26 sigma values), `sigma_langevin.csv`,
plus `*_median_heuristic.csv`. Segmentation is held fixed across the sweep, so every
difference below is attributable to `sigma` alone.

**NB the letter clash:** this `sigma` is the similarity-matrix bandwidth. The toy data's
noise level is also called sigma and appears here as `noise_sigma`.

## 0. Two facts that have to come first

**ADP does not use `sigma` at all.** It consumes the pairwise *distance* matrix directly
(`Data(distances=D)`) and never forms `A`. Every headline figure in the manuscript uses
ADP, so **those results carry no sigma dependence whatsoever**. This is the first sentence
of the answer to minor comment 6, and it is a complete answer for most of the paper.

**The manuscript and the code disagree about the value.** The text says
`A = exp(-W2^2 / (2 sigma))` with "we always take sigma = 1"; `mfpy/clustering.py` computes
`np.exp(-(distance_matrix**2))`, which is `sigma = 1/2`. The algorithm listing (line 430 of
`cpd-paper-revised.tex`) also writes `exp(-W2^2/2)`, matching the code. **Recommend
correcting the sentence in section 2.3 to `sigma = 1/2`, or equivalently restating the rule
as `exp(-W2^2)`.** On the Langevin data the discrepancy is immaterial (V-measure 0.660 vs
0.645); on the toy data both values are in the failure regime described below.

## 1. Langevin: insensitive

`tables/table4_langevin_sigma_sweep.md`, sigma over six orders of magnitude:

| | sigma = 1/2 (code) | sigma = 1 (manuscript) | median heuristic | best on grid |
|---|---|---|---|---|
| V-measure (3-class) | 0.660 | 0.645 | 0.674 | 0.691 |
| distributional fidelity | 0.122 | 0.132 | 0.108 | 0.099 |

V-measure stays within **[0.645, 0.691] across the entire grid** — a range of 0.046 on a
scale where the difference between our method and k-means is 0.03. For the data set the
action item names, the answer to the reviewer is simply: not sensitive.

One caveat worth reporting rather than hiding: the *labelling* does move even though the
*quality* does not. ARI against the code's sigma = 1/2 falls to 0.33 at large sigma. The
segments that get relabelled are the ambiguous ones near the barrier, which is exactly the
population where — per the argument for reviewer 3(b) — a point label is not well posed in
the first place.

## 2. Toy: highly sensitive, and the manuscript's value is in the failure regime

This is the finding that matters. `tables/table1_toy_sigma_rules.md`, V-measure (3-class):

| noise_sigma | sigma = 1/2 (code) | sigma = 1 (manuscript) | median heuristic | best per trajectory |
|---|---|---|---|---|
| 0  | 0.502 | 0.365 | **0.607** | 0.613 |
| 5  | 0.216 | 0.076 | **0.580** | 0.596 |
| 20 | 0.008 | 0.009 | **0.544** | 0.546 |

At `noise_sigma = 20` the fixed value gives V-measure **0.008** — the spectral clustering is
returning essentially nothing — against 0.544 for a scale-free choice.

### Why: `sigma = 1` is not scale invariant

The toy trajectory's two states have means 100 and 200, so segment distances are large:
median `W2 = 44.8`, median `W2^2 = 2004`. At `sigma = 1/2` the exponent is
`-W2^2 / 1 ~ -2000`, so

    83% of the off-diagonal entries of A underflow to below 1e-12

and the similarity matrix is numerically the identity. Spectral clustering on an identity
affinity has no structure to find. On the Langevin data, median `W2^2 = 0.56`, the exponent
is O(1), and `A` is perfectly well conditioned — which is the only reason `sigma = 1` has
worked so far.

**So the reviewer's concern is correct, and sharper than they put it: the issue is not that
`sigma` needs tuning, it is that a fixed `sigma` is dimensionally wrong.** `W2^2` carries
the squared units of the data, so any fixed bandwidth silently encodes an assumption that
the trajectory takes O(1) values. The Langevin, Prinz and MD trajectories happen to satisfy
that; the toy trajectory does not.

### The fix: median heuristic

Setting `sigma = median(W2^2) / 2` per trajectory (`median_heuristic_sigma`) is scale free
and standard for Gaussian kernels. It:

- **matches the per-trajectory oracle** on the toy set (0.580 vs 0.596 at noise 5; 0.544 vs
  0.546 at noise 20) — i.e. it captures essentially all of the achievable performance
  without tuning;
- is **neutral-to-slightly-better on Langevin** (0.674 vs 0.660), so adopting it costs
  nothing on the data already in the paper;
- also improves distributional fidelity everywhere (toy noise 20: 0.202 -> 0.058).

Recommend adopting it, reporting the sweep as the justification, and noting that it reduces
to something close to the current behaviour on O(1) data.

## 3. The eigengap is sensitive even where the clustering is not

`tables/table3_toy_eigengap_by_sigma.md`. The number of clusters an eigengap heuristic would
select from `A` moves with sigma across the whole grid — on the toy set, `k = 1` for 87% of
trajectories at `sigma = 0.001` and for 25% at `sigma = 0.5`, peaking at `k = 3` only in a
narrow band; on Langevin it runs 9 -> 3 -> 2 -> 1 as sigma increases, and at both `sigma =
1/2` and `sigma = 1` it reads `k = 1`.

This matters only because `K` is currently supplied by hand (`K = 3`), which insulates the
reported results. It should be stated: **the spectral path's cluster count is not being
inferred from the data, and if it were, it would be sigma-dependent.** ADP, which does infer
its own cluster count, is unaffected — another reason it is the right default.

## Status / what is still shaky

- The toy failure mechanism (underflow to a near-identity `A`) is verified directly by
  inspecting `A`, not inferred from the scores.
- The median heuristic is evaluated at `K = 3` only. Its interaction with `K`, and with the
  eigengap-selected `K`, is untested.
- Langevin remains n = 1 at a single `(w, q)`.
- 11 of 300 toy trajectories are dropped because they yield fewer than `K = 3` segments
  (all at `noise_sigma = 0`).
- Not tested: whether the median heuristic changes any conclusion in the Prinz, MD or
  underwater-acoustics sections. Those all use ADP in the manuscript, so it should not, but
  it has not been checked.
