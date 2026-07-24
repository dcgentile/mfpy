"""
Tests for toy_scoring. Run with: python mfpy/experiments/test_toy_scoring.py
"""

import numpy as np

from toy_scoring import (
    load_ground_truth,
    true_change_points,
    match_change_points,
    score_change_points,
    tolerance_curve,
    localisation_stats,
    ground_truth_length,
)


# three ramps -> six true change points at 100, 110, 300, 310, 500, 510
RAMPS = np.array([[100, 110], [300, 310], [500, 510]])


def test_true_change_points_are_ramp_endpoints():
    t = true_change_points(RAMPS)
    assert t.tolist() == [100, 110, 300, 310, 500, 510]


def test_perfect_detection():
    pred = [100, 110, 300, 310, 500, 510]
    r = score_change_points(pred, RAMPS, tau=0)
    assert r["precision"] == 1.0 and r["recall"] == 1.0 and r["f1"] == 1.0
    assert r["n_true"] == 6


def test_midramp_prediction_is_not_a_true_positive():
    """The key behavioural change: sitting inside a ramp is not enough."""
    pred = [105, 305, 505]  # dead centre of each ramp, 5 from either boundary
    strict = score_change_points(pred, RAMPS, tau=2)
    assert strict["n_matched"] == 0, "should not match at tau=2"
    loose = score_change_points(pred, RAMPS, tau=5)
    assert loose["n_matched"] == 3, "should match one boundary each at tau=5"
    # only 3 of 6 true change points found, so recall is capped at 0.5
    assert abs(loose["recall"] - 0.5) < 1e-12


def test_one_to_one_penalises_duplicates():
    """Two predictions on one boundary: one TP, one FP -- not two TPs."""
    pred = [100, 101]
    r = score_change_points(pred, RAMPS, tau=5)
    assert r["n_matched"] == 1
    assert abs(r["precision"] - 0.5) < 1e-12
    assert abs(r["recall"] - 1 / 6) < 1e-12


def test_one_to_one_allows_both_ends_of_a_narrow_ramp():
    """A bracketing pair should claim both boundaries even when they are close."""
    narrow = np.array([[100, 101]])  # tl = 1
    pred = [96, 105]
    r = score_change_points(pred, narrow, tau=10)
    assert r["n_matched"] == 2
    assert r["precision"] == 1.0 and r["recall"] == 1.0


def test_assignment_minimises_displacement():
    pred = [99, 111]
    pairs = match_change_points(pred, np.array([100, 110]), tau=5)
    mapping = {int(p): int(t) for p, t in pairs}
    assert mapping == {0: 0, 1: 1}, "99->100 and 111->110, not crossed"


def test_tolerance_forbids_distant_pairs():
    pred = [80, 130]
    assert score_change_points(pred, RAMPS, tau=5)["n_matched"] == 0
    assert score_change_points(pred, RAMPS, tau=25)["n_matched"] == 2


def test_spurious_predictions_hurt_precision():
    pred = [100, 110, 300, 310, 500, 510, 700, 800]
    r = score_change_points(pred, RAMPS, tau=2)
    assert abs(r["precision"] - 6 / 8) < 1e-12
    assert r["recall"] == 1.0


def test_boundary_points_dropped():
    pred = [0, 100, 110, 999]
    r = score_change_points(pred, RAMPS, tau=2, trajectory_length=1000)
    assert r["n_predicted"] == 2 and r["precision"] == 1.0


def test_empty_prediction():
    r = score_change_points([], RAMPS, tau=10)
    assert r["precision"] == 0.0 and r["recall"] == 0.0 and r["f1"] == 0.0


def test_recall_monotone_in_tolerance():
    pred = [96, 113, 305, 500, 511]
    curve = tolerance_curve(pred, RAMPS, taus=range(0, 40, 2))
    recalls = [c["recall"] for c in curve]
    assert all(b >= a for a, b in zip(recalls, recalls[1:]))


def test_localisation_sign_convention():
    """Positive signed error = displaced into the ramp interior."""
    inward = localisation_stats([104, 106], np.array([[100, 110]]))
    assert inward["mean_signed_error"] > 0

    outward = localisation_stats([96, 114], np.array([[100, 110]]))
    assert outward["mean_signed_error"] < 0

    exact = localisation_stats([100, 110], np.array([[100, 110]]))
    assert exact["mean_signed_error"] == 0
    assert exact["mean_abs_error"] == 0


def test_localisation_matches_known_inset():
    """A +6 / -7 inset pattern should report as a positive (inward) signed error."""
    stats = localisation_stats([106, 103], np.array([[100, 110]]))
    # 106 is 6 past the start; 103 is 3 past the start (nearest is start, not end)
    assert stats["mean_signed_error"] > 0
    assert stats["n"] == 2


def test_real_ground_truth_files(data_dir):
    """Parse every GT file on disk and check it agrees with the generator."""
    import glob
    import os
    import re

    paths = sorted(glob.glob(os.path.join(data_dir, "*_GroundTruth.txt")))
    assert len(paths) == 300, f"expected 300 GT files, found {len(paths)}"

    for p in paths:
        m = re.search(r"sigma_(\d+)_transition_(\d+)_GroundTruth", os.path.basename(p))
        tl = int(m.group(2))
        ramps = load_ground_truth(p, transition_length=tl)
        assert ramps.shape == (19, 2), f"{p}: got {ramps.shape}"
        assert np.all(ramps[:, 1] > ramps[:, 0])
        assert np.all(ramps[1:, 0] > ramps[:-1, 1]), f"{p}: ramps overlap"

        cps = true_change_points(ramps)
        assert cps.size == 38, f"{p}: expected 38 change points, got {cps.size}"
        assert np.all(np.diff(cps) > 0), f"{p}: change points not strictly increasing"

        traj = np.loadtxt(p.replace("_GroundTruth", ""))
        assert traj.shape[0] == ground_truth_length(tl), (
            f"{p}: trajectory length {traj.shape[0]} != 5000 + 19*{tl}"
        )
    return len(paths)


if __name__ == "__main__":
    import sys
    import os

    here = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(here, "..", "..", "data", "Toy_Model_Trajectories")

    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            if name == "test_real_ground_truth_files":
                n = fn(data_dir)
                print(f"PASS {name} ({n} files)")
            else:
                fn()
                print(f"PASS {name}")
        except Exception as e:
            failures += 1
            print(f"FAIL {name}: {e}")

    print("\n" + ("all tests passed" if failures == 0 else f"{failures} failure(s)"))
    sys.exit(1 if failures else 0)
