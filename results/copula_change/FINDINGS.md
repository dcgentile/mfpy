# Copula-type dependence change — a harder counterexample

Follow-up to `results/correlation_change/`. That experiment showed a pure correlation flip
is invisible to the paper's 1D marginal method but recoverable by (a) Sinkhorn on the joint
sample, or (b) rotating to the pooled-PCA eigenbasis and reusing the unmodified 1D method.
The PCA fix worked there because the change was still second-moment information (a
covariance eigenvalue swap). This experiment asks whether that second-moment ceiling is
real, using a construction where it provably is.

Script: `mfpy/experiments/copula_change.py`. Runtime ~15s (Sinkhorn now uses w=150 rather
than w=50; see below). Output: `copula_change.npz`, `marginal_check.txt`, `figures/`.

## Construction

X1 ~ N(0,1) throughout, T=2000, true change point t*=1000. Before t*, X2 is an independent
N(0,1) draw. After t*, X2 = S·X1 where S is an independent random sign (Rademacher). This
keeps, in both regimes:

- **identical marginals** — X2 = S·X1 is still exactly N(0,1), since N(0,1) is symmetric
  under sign flips.
- **identical full covariance matrix.** Cov(X1,X2) = E[S]·E[X1²] = 0 in both regimes (E[S]=0
  after the switch; independence before), and Var(X2) = E[X1²] = 1 always, so the covariance
  is the identity matrix I in *both* regimes. `marginal_check.txt` confirms this
  numerically: pooled eigenvalues 0.947/1.059, corr -0.05 pre vs -0.06 post — noise around
  zero, not a real signal either way.

Because the covariance is I at every t, Var(vᵀX) = vᵀIv = 1 for *every* unit direction v,
always. There is no rotation and no eigenvalue gap for PCA to exploit — unlike the
correlation-flip case, this isn't PCA being under-resourced, it structurally cannot see this
change. What does change is the joint shape: an independent round cloud (`figures/
joint_scatter.pdf`, left) becomes two crossing diagonal lines, |X1| = |X2| exactly
(`mean_abs_x1_minus_abs_x2` drops from 0.625 to exactly 0.0000 in `marginal_check.txt`) — a
copula-type change with (numerically) unbounded mutual information and exactly zero Pearson
correlation, the classic "correlation is not dependence" setup.

## Result

Four detectors, all at a shared window w=150 for a fair comparison (chosen for the Sinkhorn
detector; marginals/PCA are blind to this change at any w since their signal is exactly
zero, not just small — see `run_md_detector` docstring for the w/reg grid):

| detector | near-t* peak vs background | global argmax | localizes t*=1000? |
|---|---|---|---|
| X1 marginal | 3.1 sigma | 1083 | no — global max is a different, unrelated peak |
| X2 marginal | 0.8 sigma | 1465 | no |
| PC1 | 2.7 sigma | 1083 | no — same spurious peak as X1 |
| PC2 | 4.4 sigma | 1023 | partial — elevated near t*, but not the true peak location, and no more separated from background than several other peaks along the trajectory |
| **Sinkhorn, joint (X1,X2)** | **8.4 sigma** | **999** | **yes** — one step from t*, and the unambiguous global maximum of the whole 1900-point sweep |

`figures/all_detectors.pdf` stacks all five; only the Sinkhorn panel has a single peak that
is both centered on t* and clearly the tallest feature in the trace. `identify_change_points`
(q=0.95) reports the same tight run, (946, 948, 953, 954), on Sinkhorn and (by a numerical
coincidence — the two statistics are correlated at r=0.87 but not identical, see script
history) on PC2; X1/X2/PC1 report a scattered mix of comparably-sized false positives
elsewhere in the trajectory.

**This needed a bigger window than the correlation counterexample.** At w=50 (the setting
that worked cleanly for the correlation flip), Sinkhorn's signal here is much weaker
(near/far gap ~1.2-1.4 sigma, not a usable detector); it only becomes clean at w>=150. This
makes sense: telling "independent cloud" from "two crossing lines" apart from finite
empirical samples needs more points per window than telling two different correlations
apart does, and reg=0.05 (fine for the correlation case) over-blurs the thin diagonal
structure — reg=0.03 helped too. Grid checked: w in {50,100,150,200}, reg in
{0.01,0.03,0.05}.

## Takeaway for the manuscript

This is the sharper version of action item 5: a change that is provably invisible to any
second-moment-based method (raw marginals, or PCA on top of them) because the covariance is
literally unchanged, not just poorly summarized. Entropic OT on the joint sample recovers it
cleanly, but only once the window is large enough to resolve dependence structure rather
than just central moments — a caveat worth stating explicitly if this goes in the paper,
since it means the window-size guidance from the toy-sweep work ("set w by what's needed to
estimate the marginal stably", `results/toy_sweep/FINDINGS.md` point 6) does not transfer
as-is to the multivariate case: here w also has to be large enough to estimate *dependence*
stably, which is a higher bar. As before, this is one synthetic trajectory, not a validated
recipe — no false-positive-rate study, no sweep over noise levels or transition types.
