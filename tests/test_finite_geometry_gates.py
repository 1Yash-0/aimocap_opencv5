from __future__ import annotations

import numpy as np
import pytest

from aimocap.math.kinematics import compute_median_bone_lengths, fit_skeleton_sequence


def test_skeleton_fit_rejects_bones_without_finite_support():
    points = np.full((5, 17, 3), np.nan, dtype=np.float64)
    points[:, 11] = [0.0, 0.0, 1.0]
    points[:, 12] = [1.0, 0.0, 1.0]
    with pytest.raises(ValueError, match="insufficient finite geometry"):
        compute_median_bone_lengths(points)


def test_skeleton_fit_rejects_nonfinite_initial_geometry():
    bone_lengths = np.ones(18, dtype=np.float64)
    points = np.full((1, 17, 3), np.nan, dtype=np.float64)
    with pytest.raises(ValueError, match="positive bone lengths|finite geometry"):
        fit_skeleton_sequence(points, bone_lengths)
