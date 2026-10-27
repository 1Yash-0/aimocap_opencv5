"""Regression tests for the opt-in fixed head target frame."""

import numpy as np
from scipy.spatial.transform import Rotation

from aimocap.retarget.mocap_ik import MocapIKSolver
from aimocap.retarget.mocap_skeleton import MocapSkeleton


from tests._assets import requires_manny_fbx
def _solver():
    points = np.zeros((1, 133, 3), dtype=np.float64)
    points[0, :, 1] = np.linspace(0.0, 150.0, 133)
    weights = np.ones((1, 133), dtype=np.float64)
    skel = MocapSkeleton(points, weights)
    return MocapIKSolver(skel, head_target_mode="fixed")


def _measured_for_frame(solver, delta):
    # Start from a complete finite body fixture for analytic_init.
    measured_by_name = {
        name: np.asarray(solver.skel.rest_t[idx], dtype=np.float64) + np.array([0.0, 0.0, 100.0])
        for idx, name in solver.skel.coco_anchor.items()
    }
    center = measured_by_name["nose"]
    # The body fixture needs finite core measurements for analytic_init.
    for name in ("pelvis", "neck", "hip_l", "hip_r", "shoulder_l", "shoulder_r",
                 "elbow_l", "elbow_r", "wrist_l", "wrist_r", "knee_l", "knee_r",
                 "ankle_l", "ankle_r"):
        measured_by_name.setdefault(name, center.copy())
    F = delta.as_matrix() @ solver._F_rest_head_fixed
    ear_mid = center.copy()
    left_ear = ear_mid + 7.2 * F[:, 0]
    right_ear = ear_mid - 7.2 * F[:, 0]
    face_mid = ear_mid + 5.0 * F[:, 1]
    measured_by_name.update({
        "left_ear": left_ear,
        "right_ear": right_ear,
        "nose": face_mid + 1.0 * F[:, 1],
        "left_eye": face_mid,
        "right_eye": face_mid,
    })
    # Heel/toe keys keep the normal precompute path finite.
    for name in ("heel_l", "heel_r", "big_toe_l", "big_toe_r"):
        measured_by_name.setdefault(name, center.copy())
    return measured_by_name


@requires_manny_fbx
def test_fixed_head_target_preserves_each_world_axis():
    solver = _solver()
    for axis in np.eye(3):
        delta = Rotation.from_rotvec(axis * np.deg2rad(25.0))
        measured = _measured_for_frame(solver, delta)
        init_x = solver.analytic_init(measured)
        ctx = solver._precompute_frame(measured, init_x=init_x)
        target = Rotation.from_quat(ctx["head_target_quat"])
        error = (target.inv() * delta).magnitude()
        assert error < 1e-7, (axis, np.degrees(error))
        assert ctx["head_target_mode"] == "fixed"


@requires_manny_fbx
def test_legacy_head_target_mode_remains_default():
    points = np.zeros((1, 133, 3), dtype=np.float64)
    points[0, :, 1] = np.linspace(0.0, 150.0, 133)
    solver = MocapIKSolver(MocapSkeleton(points, np.ones((1, 133))))
    assert solver.head_target_mode == "legacy"
    assert solver._F_rest_head_fixed.shape == (3, 3)
