"""Foot-lock leg-IK NaN hardening (CPU-only, deterministic).

Locks the crash fix: a non-finite (NaN) chain vector must degrade to
"leave the frame's pose unchanged" (identity rotation), never raise
"Found zero norm quaternions". NaN slips past `norm < 1e-8` guards
because NaN comparisons are False, so the guard must be positive-form.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aimocap.retarget.foot_lock import (
    _apply_leg_ik_frame,
    _rotation_between_vectors,
)


from tests._assets import requires_manny_fbx
def test_identity_on_finite_known_rotation():
    src = np.array([1.0, 0.0, 0.0])
    dst = np.array([0.0, 1.0, 0.0])
    rot = _rotation_between_vectors(src, dst)
    assert np.allclose(rot.apply(src), dst, atol=1e-9)
    assert float(rot.magnitude()) > 0.0


def test_identity_on_degenerate_and_nan_vectors():
    assert _rotation_between_vectors(np.zeros(3), np.array([0.0, 1.0, 0.0])).approx_equal(
        Rotation.identity(), atol=1e-9)
    # The regression: NaN must not reach align_vectors.
    nan = np.full(3, np.nan)
    r1 = _rotation_between_vectors(nan, np.array([0.0, 1.0, 0.0]))
    r2 = _rotation_between_vectors(np.array([1.0, 0.0, 0.0]), nan)
    r3 = _rotation_between_vectors(nan, nan)
    for r in (r1, r2, r3):
        assert np.isfinite(r.as_quat()).all()
        assert float(np.linalg.norm(r.as_quat())) > 0.0
        assert r.approx_equal(Rotation.identity(), atol=1e-9)
    # Partially-NaN input is still non-finite.
    assert np.isfinite(
        _rotation_between_vectors(np.array([np.nan, 1.0, 0.0]), np.array([0.0, 1.0, 0.0])).as_quat()
    ).all()


@requires_manny_fbx
def test_apply_leg_ik_frame_nan_root_does_not_raise(tmp_path: Path):
    """End-to-end: a NaN root translation degrades, it does not crash."""
    from aimocap.retarget.fbx_rig import Skeleton

    skel = Skeleton("Manny.FBX")
    n = skel.num_joints
    local = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1))
    out = _apply_leg_ik_frame(
        skel, local, np.full(3, np.nan), "left",
        np.full(3, np.nan), np.full(3, np.nan),
    )
    assert out.shape == local.shape
    assert np.isfinite(out).all()


@requires_manny_fbx
def test_apply_leg_ik_frame_preserves_finite_pose_shape(tmp_path: Path):
    """Finite inputs keep producing a valid (normalized) quaternion set."""
    from aimocap.retarget.fbx_rig import Skeleton

    skel = Skeleton("Manny.FBX")
    n = skel.num_joints
    local = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1))
    out = _apply_leg_ik_frame(
        skel, local, np.array([10.0, 20.0, 90.0]), "left",
        np.array([10.0, 20.0, 88.0]), np.array([10.0, 21.0, 88.1]),
        np.array([10.0, 21.5, 88.2]),
    )
    assert np.isfinite(out).all()
    norms = np.linalg.norm(out, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)