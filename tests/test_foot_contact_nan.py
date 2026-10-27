# tests/test_foot_contact_nan.py
"""WP2.A: NaN-tolerant source contact detection (CPU-only, deterministic).

Locks the WP0.3 mechanism fix: a NaN heel track must not poison the whole
toe cluster into NaN ground / zero stance. Fully-unknown frames stay
invalid (never stance); partially observed frames use finite members.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aimocap.retarget.foot_lock import _interpolate_missing_rows, detect_foot_contacts


def _planted_track(n=60, nan_ranges=()):
    """(n, 133, 3) Z-up cm: left toes (17, 18, 19) planted at z≈0.5."""
    pts = np.full((n, 133, 3), np.nan)
    for j in (17, 18, 19, 20, 21, 22):
        pts[:, j, 0] = np.linspace(0, 5, n)
        pts[:, j, 1] = 0.0
        pts[:, j, 2] = 0.5
    for a, b in nan_ranges:
        pts[a:b + 1, 19, :] = np.nan  # heel gaps like WP0.3 (19% missing)
    return pts


def test_nan_heel_does_not_kill_stance():
    contacts = detect_foot_contacts(_planted_track(nan_ranges=[(10, 12), (40, 47)]), fps=30.0)
    left = contacts["left"]
    assert np.isfinite(left["ground_z"]), "ground must not be NaN"
    assert left["segments"], "planted foot with gappy heel must yield stance"
    assert float(np.mean(left["mask"])) > 0.5


def test_fully_unknown_frames_stay_invalid():
    pts = _planted_track()
    pts[20:30, 17:20, :] = np.nan  # whole left cluster unknown
    contacts = detect_foot_contacts(pts, fps=30.0)
    left = contacts["left"]
    assert not left["mask"][20:30].any(), "fully-unknown frames must never be stance"
    assert left["invalid_frames"] >= 10


def test_all_nan_input_yields_no_stance_no_crash():
    pts = np.full((30, 133, 3), np.nan)
    contacts = detect_foot_contacts(pts, fps=30.0)
    for side in ("left", "right"):
        assert not contacts[side]["mask"].any()
        assert contacts[side]["segments"] == []


def test_interpolate_missing_rows_known_values():
    pos = np.array([[0.0, 0.0], [np.nan, 2.0], [np.nan, 4.0], [3.0, 6.0]])
    out = _interpolate_missing_rows(pos)
    assert out[1, 0] == 1.0 and out[2, 0] == 2.0
    assert out[1, 1] == 2.0 and out[2, 1] == 4.0
    allnan = np.full((4, 2), np.nan)
    assert np.isnan(_interpolate_missing_rows(allnan)).all()
