"""Regression tests for opt-in clip-calibrated head targets."""

import numpy as np
from scipy.spatial.transform import Rotation

from aimocap.retarget.mocap_ik import MocapIKSolver
from aimocap.retarget.mocap_skeleton import MocapSkeleton


from tests._assets import requires_manny_fbx
def _seed_solver(mode="fixed", target_sequence=None):
    points = np.zeros((1, 133, 3), dtype=np.float64)
    points[0, :, 1] = np.linspace(0.0, 150.0, 133)
    skel = MocapSkeleton(points, np.ones((1, 133), dtype=np.float64))
    return MocapIKSolver(skel, target_sequence=target_sequence,
                         head_target_mode=mode)


def _body(solver, frame, face_points):
    """Complete finite frame with torso basis matching solver rest convention."""
    center = np.array([0.0, 0.0, 100.0], dtype=np.float64)
    F = solver._F_rest_head
    # _torso_frame_from_points applies the same chirality correction as the
    # face frame.  Its raw up vector is the pre-correction opposite of F[:, 2].
    torso_up_raw = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    measured = {
        "pelvis": center - 30.0 * torso_up_raw,
        "neck": center + 10.0 * torso_up_raw,
        "shoulder_l": center + 10.0 * F[:, 0],
        "shoulder_r": center - 10.0 * F[:, 0],
        "hip_l": center + 8.0 * F[:, 0] - 30.0 * torso_up_raw,
        "hip_r": center - 8.0 * F[:, 0] - 30.0 * torso_up_raw,
        "knee_l": center + 8.0 * F[:, 0] - 60.0 * torso_up_raw,
        "knee_r": center - 8.0 * F[:, 0] - 60.0 * torso_up_raw,
        "ankle_l": center + 8.0 * F[:, 0] - 90.0 * torso_up_raw,
        "ankle_r": center - 8.0 * F[:, 0] - 90.0 * torso_up_raw,
        "elbow_l": center + 25.0 * F[:, 0] + 5.0 * F[:, 1],
        "elbow_r": center - 25.0 * F[:, 0] + 5.0 * F[:, 1],
        "wrist_l": center + 40.0 * F[:, 0] + 5.0 * F[:, 1],
        "wrist_r": center - 40.0 * F[:, 0] + 5.0 * F[:, 1],
        "heel_l": center.copy(), "heel_r": center.copy(),
        "big_toe_l": center.copy(), "big_toe_r": center.copy(),
    }
    measured.update(face_points)
    return measured


def _face_points(F_obs, layout="gt"):
    if layout == "gt":
        local = {
            "left_ear": np.array([7.2, 5.0, 3.0]),
            "right_ear": np.array([-7.2, 5.0, 3.0]),
            "nose": np.array([0.0, 10.0, 5.0]),
            "left_eye": np.array([3.2, 9.2, 6.0]),
            "right_eye": np.array([-3.2, 9.2, 6.0]),
        }
    else:
        # A measured-human-like layout with wider ears and asymmetric depth.
        local = {
            "left_ear": np.array([8.4, 0.7, 0.0]),
            "right_ear": np.array([-8.1, -0.4, 0.2]),
            "nose": np.array([0.4, 10.7, 1.6]),
            "left_eye": np.array([3.3, 8.2, 3.9]),
            "right_eye": np.array([-3.1, 8.0, 3.7]),
        }
    origin = np.array([0.0, -1.0, 120.0])
    return {name: origin + F_obs @ value for name, value in local.items()}


def _sequence(mode, deltas, layout="gt"):
    seed = _seed_solver("fixed")
    F_rest = seed._F_rest_head
    # The fixture torso is world-Z-up; retain the rig's lateral axis while
    # converting its historical handedness into the physical torso basis.
    torso_rest = np.column_stack([F_rest[:, 0], -F_rest[:, 1], -F_rest[:, 2]])
    offset = Rotation.from_euler("zyx", [0.0, -7.0, 11.0], degrees=True)
    rows = []
    for delta in deltas:
        F_obs = delta.as_matrix() @ torso_rest @ offset.as_matrix()
        rows.append(_body(seed, len(rows), _face_points(F_obs, layout)))
    target_sequence = {
        key: np.asarray([row[key] for row in rows], dtype=np.float64)
        for key in rows[0]
    }
    solver = _seed_solver("calibrated", target_sequence=target_sequence)
    return solver, rows, offset


def _target_rotation(solver, measured):
    F_obs = solver._face_frame_from_points(measured)
    assert F_obs is not None
    return solver._head_target_rotation(F_obs)


@requires_manny_fbx
def test_calibrated_neutral_is_upright_for_gt_and_realistic_layouts():
    for layout in ("gt", "real"):
        neutral = [Rotation.identity()] * 24
        solver, rows, _ = _sequence("calibrated", neutral, layout)
        target = _target_rotation(solver, rows[-1])
        F_torso = solver._torso_frame_from_points(rows[-1])
        head_up = target.apply(solver._F_rest_head[:, 2])
        angle = np.degrees(np.arccos(np.clip(np.dot(head_up, F_torso[:, 2]), -1.0, 1.0)))
        assert angle < 10.0, (layout, angle)
        assert solver.head_calibration_diagnostics()["status"] == "calibrated"


@requires_manny_fbx
def test_calibrated_preserves_single_axis_motion():
    for axis in np.eye(3):
        delta = Rotation.from_rotvec(axis * np.deg2rad(25.0))
        solver, rows, _ = _sequence("calibrated", [Rotation.identity()] * 12 + [delta] * 12)
        neutral_target = _target_rotation(solver, rows[0])
        target = _target_rotation(solver, rows[-1])
        motion = target * neutral_target.inv()
        error = (motion.inv() * delta).magnitude()
        assert error < 1e-7, (axis, np.degrees(error))


@requires_manny_fbx
def test_calibrated_offset_is_deterministic_and_axis_independently_verifiable():
    deltas = [Rotation.identity()] * 20
    solver_a, rows_a, _ = _sequence("calibrated", deltas, "real")
    solver_b, rows_b, _ = _sequence("calibrated", deltas, "real")
    assert np.array_equal(
        np.asarray(solver_a.head_calibration_diagnostics()["offset_matrix"]),
        np.asarray(solver_b.head_calibration_diagnostics()["offset_matrix"]),
    )
    assert np.array_equal(
        _target_rotation(solver_a, rows_a[0]).as_quat(),
        _target_rotation(solver_b, rows_b[0]).as_quat(),
    )


@requires_manny_fbx
def test_default_legacy_mode_is_unchanged_and_calibrated_is_opt_in():
    solver = _seed_solver()
    assert solver.head_target_mode == "fixed"
    legacy = _seed_solver("legacy")
    assert legacy.head_target_mode == "legacy"
    assert np.allclose(legacy._head_offset.as_matrix(), np.eye(3))


@requires_manny_fbx
def test_fixed_mode_reproduces_realistic_neutral_tilt_before_fix():
    calibrated_seed, rows, _ = _sequence("calibrated", [Rotation.identity()] * 24, "real")
    target_sequence = {
        key: np.asarray([row[key] for row in rows], dtype=np.float64)
        for key in rows[0]
    }
    fixed = _seed_solver("fixed", target_sequence=target_sequence)
    target = _target_rotation(fixed, rows[-1])
    torso = fixed._torso_frame_from_points(rows[-1])
    head_up = target.apply(fixed._F_rest_head[:, 2])
    angle = np.degrees(np.arccos(np.clip(np.dot(head_up, torso[:, 2]), -1.0, 1.0)))
    assert angle > 10.0, angle
    assert calibrated_seed.head_calibration_diagnostics()["status"] == "calibrated"


@requires_manny_fbx
def test_legacy_target_formula_is_unchanged():
    solver = _seed_solver("legacy")
    F_obs = Rotation.from_euler("zyx", [13.0, -9.0, 21.0], degrees=True).as_matrix()
    expected = Rotation.from_matrix(F_obs @ solver._F_rest_head.T).as_matrix()
    actual = solver._head_target_rotation(F_obs).as_matrix()
    assert np.array_equal(actual, expected)

