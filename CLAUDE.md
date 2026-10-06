# MFPy — working context

Research code for **"Wasserstein-Based Identification of Metastable States in Time Series
Data via Change Point Detection and Segment Clustering"** (Gentile, Huang, Murphy),
submitted to an AIMS journal. **Status: revise-and-resubmit.** The current work is
addressing the reviewer report.

- Manuscript: `paper-resources/cpd-paper.pdf`
- Reviewer report: `paper-resources/report_X_on_v1.pdf`
- Action items and status: `paper-resources/revision_summary.md` ← **start here**

David Gentile is first author; James M. Murphy is the advisor.

## The method in one paragraph

Slide a window over a 1D time series and compute the 2-Wasserstein distance between the
empirical distributions of the left and right half-windows — this approximates the metric
derivative of the trajectory in Wasserstein space. Threshold that statistic at its own
`q`-quantile, take maximal runs of supra-threshold indices, and read two change points out
of each run. The resulting segments are then treated as measures, compared pairwise with
the Wasserstein distance, and clustered (ADP or spectral) to recover metastable states.

Core code: `mfpy/distances.py` (statistic + change points), `mfpy/clustering.py`
(`SegmentBasedCluster`), `mfpy/utils.py`.

## Environment gotchas

- **`import mfpy` requires `dadapy`**, because `mfpy/__init__.py` imports `clustering`,
  which imports it. `dadapy` is slow to build. If you only need change-point detection,
  load `mfpy/distances.py` directly by path — see `_load_distances_module()` in
  `mfpy/experiments/toy_sweep.py`.
- `.venv/` in this repo is a bare Python 3.14 venv with only pip. The sandbox has its own
  Python 3.10; install with `pip install --break-system-packages numpy scipy pandas pot
  tqdm matplotlib tabulate`.
- There is **no `pyproject.toml` or `setup.py`**, so `pip install -e .` from the README
  does not work. Put the repo root on `sys.path`.
- **Writing many small flushes to the mounted repo is very slow.** Long sweeps should
  write their progress log to `/tmp` and copy the final CSVs back. This is the difference
  between the toy sweep taking 11 minutes and taking 40 seconds.
- Background processes do not survive between shell calls in the sandbox. Long jobs need
  `--budget` style resumability, or must fit in a single call.

## Toy-trajectory experiments — DONE, do not re-run

Everything in `results/toy_sweep/` is finished and verified. **Read
`results/toy_sweep/FINDINGS.md` rather than recomputing.** Total compute is only ~20s if
you do need to regenerate, but the analysis and its caveats are the expensive part.

Scripts (`mfpy/experiments/`):

| file | purpose |
|---|---|
| `toy_scoring.py` | ground-truth parsing + change-point scoring |
| `boundary_estimators.py` | three rules for extracting change points from the statistic |
| `toy_sweep.py` | main sweep: 300 trajectories x 5 windows x 3 estimators |
| `toy_threshold_sweep.py` | quantile-threshold diagnostic |
| `analyze_toy_sweep.py` | tables + figures |
| `test_toy_scoring.py` | 14 unit tests; run directly, no pytest needed |

### What they established

1. **A change point is a ramp boundary, not "somewhere inside a ramp."** Each of the 19
   ramps contributes two change points (start and end), so 38 per trajectory. Scoring uses
   one-to-one Hungarian matching within a tolerance `tau`. An earlier containment-based
   criterion inverted several conclusions — do not go back to it.
2. **Detected change points land ~0.4·ℓ inside each ramp** (slope 0.34–0.45 per unit ramp
   length, r up to 0.99). The detected transition is a shrunken image of the true one.
3. **The gradient argmin/argmax step is not the cause.** Run-edge and half-max extraction
   agree with it to within 1–2.5 steps everywhere. Useful *positive* material for reviewer
   comment 1(a).
4. **The `q = 0.95` threshold is the cause.** The right threshold is
   `q* = 1 − (fraction of trajectory in transition)`; empirical optima track this closely
   (Spearman 0.75), and at a tuned `q` the localisation error collapses from 30 steps to
   0–4. This is reviewer comment 1(b), confirmed quantitatively.
5. **Abrupt changes are easier** (rho −0.78 to −0.98, all 15 cells). Noise level matters
   much less (rho −0.09 to −0.24).
6. **Tying `w` to the transition length does not help.** An oracle `w = 2ℓ` is beaten by a
   flat `w = 25` at every noise level. Material for reviewer comment 3(e).

### Known caveats

- `q*` uses the true ramp fraction, so finding 4 diagnoses rather than fixes.
- The threshold sweep only ran at `w = 25`; the `q`–`w` interaction is untested.
- The method emits ~43.5 change points against 38 true, capping precision.
- One realisation per (sigma, ℓ) cell; trends are read across ℓ, not from single cells.

## Open work

Roughly in dependency order. `paper-resources/revision_summary.md` has the full list with
reviewer wording.

1. **Resampling threshold (Matteson & James 2014) — done, see
   `results/resampling_threshold/FINDINGS.md`.** A pointwise permutation-null threshold
   (`resample_pointwise` in `mfpy/experiments/resampling_threshold.py`) beats the fixed
   quantile at a single global q=0.5, no per-trajectory tuning needed, and roughly halves
   the localisation-bias growth with transition length. A naive max-type/family-wise
   translation of their permutation test was tried first and fails (too conservative);
   that negative result is also written up and worth citing in the rebuttal as evidence of
   engagement. Still open: the `q`–`w` interaction, and folding this into the manuscript.
2. **Literature.** Vogt & Dette 2015 (gradual changes — relevant to the ramp structure
   here); Matteson & James 2014; Eichinger & Kirch 2018 and McGonigle & Cho 2025 (to be
   added to the MWCPD literature review); Jula Vanegas et al. 2022 (possible dynamic-sort
   speedup, reviewer minor comment 8).
3. **V-measure and clustering evaluation.** Deferred from the toy work. Ground-truth point
   labels are reconstructable from the same `*_GroundTruth.txt` files: state A, ramp,
   state B, ramp, ... Decide between a 3-class target {A, transition, B} and a 2-class
   target excluding transition points. Needs `dadapy` for ADP.
4. **Similarity-matrix sigma sensitivity** (action item 7). Vary sigma in the similarity
   matrix on the Langevin data at fixed `w`, `q`, and compare clusterings. **Note this
   sigma is unrelated to the toy-data noise sigma** — different parameter, same letter.
5. **Multi-dimensional correlation** (action item 5, reviewer comment 1(c)). Likely a
   discussion rather than an implementation, since OT is dimensionally cursed.
6. **Writing.** Beef up the numerical-experiments section using `FINDINGS.md`; the
   tolerance-range definition (3(d)) and the `w` discussion (3(e)) are now answerable from
   data. Minor comments 2, 3, 4, 7 still open.

## Conventions

- Experiment scripts go in `mfpy/experiments/`, outputs in `results/<experiment>/`, with a
  `FINDINGS.md` per experiment stating what was established *and* what is still shaky.
- Figures are written as both `.pdf` (for the manuscript) and `.png` (for quick viewing).
- **Figure captions in memos must include the name of the script that generated them**
  (format: "Figure title—generated by `script_name.py`"), so David can track how each figure
  was made; for hand-drawn figures, say so. **Manuscript captions must not** mention scripts or files.
- Quote numbers from the CSVs, not from memory — the earlier draft of `FINDINGS.md`
  carried two numbers from an obsolete metric and it was not obvious until re-checked.
- `data/Toy_Model_Trajectories/AllTrajs.zip.part*` is gitignored (118MB, needs LFS, and
  git-lfs is not installed in the sandbox). The 600 `.txt` files are the real inputs and
  are tracked.

## Housekeeping left over from the first session

The sandbox cannot delete files on the mounted repo, so a few things need a manual `rm`:

```bash
rm -f .git/index.lock .git/HEAD.lock .git/objects/maintenance.lock
rm -f .git/objects/*/tmp_obj_*
rm -f results/toy_sweep/tables/table1_by_sigma.* \
      results/toy_sweep/tables/table2_by_tl_bin.* \
      results/toy_sweep/tables/table3_window_choice.* \
      results/toy_sweep/tables/table4_tolerance.*
git status   # index will look stale until this runs; that is expected
```

Those four table files are **stubs marked SUPERSEDED** — they came from the earlier
containment-based scoring and their real content was wrong. The current tables are
`table1_detection_by_tolerance`, `table2_localisation_by_tl`,
`table3_estimator_equivalence`, `table4_window_choice`, `table5_threshold_diagnosis`.

There is also one uncommitted change: `analyze_toy_sweep.py` regained
`fig_f1_vs_transition_length` and `fig_window_heatmap` (regenerated under boundary
scoring), plus a consistency-check paragraph in `FINDINGS.md`. Commit after clearing the
locks.
