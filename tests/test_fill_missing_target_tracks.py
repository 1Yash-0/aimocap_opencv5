# tests/test_fill_missing_target_tracks.py
"""WP2AC: NaN-gap fill must be endpoint-clamped interpolation, no invented
NaN, no changed finite values."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aimocap.retarget.temporal import fill_missing_target_tracks


def test_gap_filled_by_linear_interp():
    t = np.array([[0.0, 0.0, 0.0], [np.nan, np.nan, np.nan], [2.0, 4.0, 8.0]])
    out = fill_missing_target_tracks({"pelvis": t})["pelvis"]
    assert np.allclose(out[1], [1.0, 2.0, 4.0])


def test_leading_trailing_edges_clamped_not_extrapolated():
    t = np.full((5, 3), np.nan)
    t[2] = [10.0, 20.0, 30.0]
    out = fill_missing_target_tracks({"n": t})["n"]
    assert np.allclose(out[0], [10.0, 20.0, 30.0])
    assert np.allclose(out[4], [10.0, 20.0, 30.0])


def test_never_observed_track_stays_nan():
    t = np.full((4, 3), np.nan)
    out = fill_missing_target_tracks({"x": t})["x"]
    assert np.isnan(out).all()


def test_finite_values_unchanged_and_other_shapes_passthrough():
    t = np.arange(15, dtype=float).reshape(5, 3)
    out = fill_missing_target_tracks({"ok": t})["ok"]
    assert np.array_equal(out, t)
    weird = np.zeros((3, 2))
    out2 = fill_missing_target_tracks({"w": weird})["w"]
    assert out2.shape == weird.shape
