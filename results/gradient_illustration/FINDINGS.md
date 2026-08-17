# Gradient illustration — FINDINGS

Script: `mfpy/experiments/gradient_illustration.py`. Figures:
`results/gradient_illustration/figures/gradient_illustration_sigma{S}_tl{L}.{pdf,png}`
for the grid S in {0, 5, 20} (the toy set's three noise levels) x L in {5, 20, 50}
(short/medium/long ramp), 9 figures total, all at w=25, q=0.95 (toy-sweep defaults).
Two stray files `gradient_illustration.{pdf,png}` (no sigma/tl suffix) are leftover
from an earlier single-trajectory version of this script and are superseded — the
sandbox can't delete files on the mounted repo, so they need a manual `rm`.

Answers reviewer major comment 1(a) ("justify using the gradient of the test
statistic") with a picture rather than an argument: three stacked panels per
trajectory showing the raw trajectory, its metric derivative gammadot, and gammadot's
gradient, with the reported change points (`mfpy.distances.identify_change_points`)
marked as dashed red lines on all three. The pattern is the same across the whole
grid: it holds at zero noise and abrupt (tl=5) as well as at high noise and gradual
(tl=50) ramps, so the mechanism isn't an artifact of one parameter setting.

The middle panel makes the mechanism visible: gammadot forms an isolated hump over
each ramp because the sliding window only picks up a distributional shift when it
straddles a transition. The bottom panel shows that hump's gradient is a clean
antisymmetric spike — sharply positive on the rise, sharply negative on the fall — so
argmin/argmax of it is a natural (and in the toy data, robust) way to read off "where
the hump was steepest," which is what stage 2 of the method does. This is consistent
with `results/toy_sweep/FINDINGS.md` point 3: alternative boundary rules (run edge,
half max) agree with the gradient rule to within 1-2.5 steps, i.e. the gradient step
itself is not what drives the localisation bias — the top panel shows the marked
points landing a bit inside each ramp, matching that finding.

One of the 19 ramps (~t=5000) does not clear the q=0.95 threshold and gets no marked
change points — an example of finding 4 (detection loss from the fixed quantile) in a
single trajectory, not a bug in this script.

Not yet done: picking/justifying the specific (sigma, tl) example for the manuscript
figure, and folding this into the write-up for comment 1(a).
