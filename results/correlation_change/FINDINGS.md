# Correlation-only change — a constructed failure case

Answers reviewer comment 1(c): the method "ignores potential correlation between
dimensions, and the change point itself may occur in the joint distribution rather than
the marginal distributions." This is a worked counterexample for the discussion section,
not a fix — the method is 1D by construction (`ot.emd2_1d` in
`compute_metric_derivative_1d`), so there is no version of it that sees this change.

Script: `mfpy/experiments/correlation_change.py`. Runtime a few seconds. Output:
`correlation_change.npz`, `marginal_check.txt`, `figures/`.

## Construction

X_t = (X1_t, X2_t) drawn i.i.d. across t from N(0, Sigma_t), T=2000, true change point
t*=1000. Sigma_t has unit diagonal at every t (Var(X1_t) = Var(X2_t) = 1 always) and
off-diagonal rho_t that steps from -0.85 to +0.85 at t*. This is the sharpest version of
the reviewer's concern: the marginal law of each coordinate is exactly N(0,1) for every t,
so there is nothing for a 1D statistic to see — the change lives entirely in the
correlation, i.e. in the joint distribution.

`marginal_check.txt` confirms the construction: pre/post marginal means and variances on
each coordinate agree to within Monte Carlo noise (e.g. Var(X1) = 1.039 pre vs 0.970 post),
while the sample correlation moves from -0.86 to +0.84 as designed.

## Result

`compute_metric_derivative_1d` + `identify_change_points` (w=50, q=0.95), run on X1 and X2
separately exactly as the method is used elsewhere in the paper:

- Neither statistic shows any elevation at t*=1000 (`figures/detector_statistic.pdf`) — it
  sits at the same noise level throughout the trajectory.
- The change points each marginal *does* report (e.g. X1: 507, 825, 868–870, 919–928,
  1469–1527; X2: similar) are false positives from ordinary sampling fluctuation in the
  1D marginal, scattered with no concentration near t*.
- `figures/joint_scatter.pdf` shows the covariance ellipse rotating from a
  negative-correlation orientation to a positive one; `figures/marginal_histograms.pdf` and
  `figures/marginal_trajectories.pdf` show the two 1D projections are indistinguishable
  before and after.

## A multivariate fix: entropic-regularized (Sinkhorn) OT

The method as stated operates coordinate-by-coordinate and is blind to any change that is
purely in the dependence structure between coordinates — not a bug fixable by retuning `w`
or `q`, since marginal W2(N(0,1), N(0,1)) is exactly zero regardless of correlation. Exact
W2 on the joint sample would need the LP-scale OT solver, which is the "dimensionally
cursed" cost flagged in `revision_summary.md` action item 5. Entropic regularization
(Sinkhorn) is the standard way around that cost, so the natural next question was whether it
recovers this specific change.

Added `compute_metric_derivative_md(X, w, reg, numIterMax)` to `mfpy/distances.py`,
alongside `compute_metric_derivative_1d`. Same interface and sliding-window design — left
and right half-windows, `sqrt` of the OT cost — but it takes the full joint sample
`X[t-w:t]`, `X[t:t+w]` in R^d and calls `ot.bregman.empirical_sinkhorn2` instead of
`ot.emd2_1d`. `identify_change_points` (threshold + run extraction) is reused unchanged,
since it only ever needed a scalar statistic per t. `reg=0.05` was picked by a small grid
{0.03, 0.05, 0.1, 0.2, 0.3} maximizing (near-window mean − far-window mean)/far-window std
on this process (see script for the sweep); `numIterMax=200` caps runtime since low-reg
Sinkhorn can fail to converge on some windows.

Run on the joint `(X1, X2)` sample from the counterexample above (`w=50, q=0.95`):

- The statistic's **global maximum over the full 1900-point sweep lands at t=1001**, one
  step from the true change point at t*=1000, at 1.57 vs a background mean/std of 0.54/0.12
  (~8.8 sigma). `identify_change_points` fires a run bracketing it (t=985, 1005).
  `figures/sinkhorn_vs_marginal.pdf` stacks this against the two blind marginal statistics
  from before for direct comparison — same data, same window, only the statistic differs.
- It is not a clean detector: two smaller runs still fire elsewhere (~920, ~1470–1525) at
  roughly the same rate as the marginal detectors' false positives, consistent with a
  q=0.95 threshold on ~1900 correlated samples. The true change point is the standout peak,
  not the only one.

## A cheaper fix: PCA + the existing componentwise method

Sinkhorn needs a new statistic. A simpler question: does rotating to the right basis let
the *existing* 1D method (`compute_metric_derivative_1d`, unchanged) see this change?

`run_pca_componentwise` in the script computes the eigendecomposition of the pooled
(whole-trajectory, time-blind) covariance, projects onto the two eigenvectors, and runs the
existing 1D detector on each resulting series exactly as on X1/X2. This works here for a
structural reason: the pre/post covariances are `[[1,rho],[rho,1]]` with rho = -0.85 then
+0.85, whose eigenvectors are the fixed +-45 degree directions in both regimes — only the
eigenvalues swap (0.15 <-> 1.85). So although the *raw* x1/x2 axes see no variance change
(by construction) and the *pooled* covariance is nearly isotropic (rho_a = -rho_b makes the
time-averaged off-diagonal ~ -0.05, eigenvalues ~0.95/1.04 — see `marginal_check.txt`), the
pooled eigenvectors still land within 0.996 cosine similarity of the true +-45 degree axes
here. Projecting onto them turns the correlation flip into a pure, large variance swap
(PC1: var 1.92 -> 0.17; PC2: var 0.16 -> 1.74) that the mean/variance-sensitive 1D
Wasserstein statistic reads easily. `figures/pca_componentwise.pdf`.

- PC2's statistic rises sharply exactly at t*=1000 and stays elevated; `identify_change_points`
  reports a run at (994, 1005) — a clean hit, no worse than the Sinkhorn result above and
  far cheaper (no Sinkhorn iterations, just an eigendecomposition of a 2x2 matrix).
- PC1's statistic peaks even more sharply *at* t=1000 (the global maximum of its entire
  trace) then permanently drops to a lower baseline — visually the least ambiguous signal
  in this whole experiment. But `identify_change_points` reports nothing there. Digging in:
  this run is the *last* above-threshold run in PC1's candidate list (nothing crosses
  threshold again after the collapse), and `identify_change_points`'s loop is
  `for n in range(N-1)` with a closing check `n == N-1` that can never be true (`n` only
  reaches `N-2`) — so a run ending at the very last candidate is silently dropped. This is a
  preexisting off-by-one in the boundary-run handling of `identify_change_points`
  (`mfpy/distances.py`), not something introduced by this experiment, and it happens to eat
  exactly the true change point here because the post-change regime never triggers the
  threshold again. **Flagging, not fixing** — this function is shared with every scored
  experiment in the paper (toy sweep, resampling threshold, etc.), so a fix needs to be
  checked against all of them before it goes in; out of scope for this counterexample.

## Takeaway for the manuscript

Correlation-only changes are invisible to the paper's 1D statistic in the raw coordinate
system no matter how `w`/`q` are tuned, but not invisible to OT in general, and not even
invisible to the *existing* statistic once the basis is fixed. Two routes recover it here:

1. **Sinkhorn on the joint sample** — general, no assumption about where the change lives,
   but a new statistic, only a proof of concept (`reg` tuned on this one process), and
   reintroduces the runtime/dimensionality cost the 1D method was built to avoid.
2. **Pooled PCA + the existing 1D method, componentwise** — cheap (2x2 eigendecomposition,
   zero new code in `distances.py`), reuses the exact scored pipeline. But it is a linear,
   second-moment method: it worked because this particular change is a sign flip in a
   Gaussian correlation, which is exactly a variance swap once rotated into eigen-
   coordinates. It would not work on a copula-type or higher-moment dependence change with
   zero covariance change (or on rho_a, rho_b of the same sign, where the pooled-time
   eigenvectors would not track the true diagonal directions as they happen to here) — PCA
   only sees second moments, so it inherits the same blind spot as the 1D method one level
   up.

Both are worth a paragraph in the discussion for action item 5: the reviewer's concern is
real and the raw method is blind to it, but "OT is dimensionally cursed" is not the end of
the story — cheap linear tricks and entropic relaxations both give partial answers, each
with their own scope limits, and neither has been validated beyond this one synthetic case.
