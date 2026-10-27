"""Focused regression tests for the opt-in FK foot landmark model."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from aimocap.data.wholebody_layout import COCO_ANCHORS
from aimocap.retarget.fbx_rig import Skeleton
from aimocap.retarget.mocap_ik import MocapIKSolver
from aimocap.retarget.mocap_skeleton import MocapSkeleton, extract_mocap_points

ROOT = Path(__file__).resolve().parents[1]


from tests._assets import requires_manny_fbx
def _base_points(frames: int = 4) -> tuple[np.ndarray, np.ndarray]:
    points = np.tile(np.array([0.0, 0.0, 80.0]), (frames, 133, 1)).astype(np.float64)
    anchors = {
        "pelvis_l": (-10.0, 0.0, 100.0), "pelvis_r": (10.0, 0.0, 100.0),
        "shoulder_l": (-20.0, 0.0, 140.0), "shoulder_r": (20.0, 0.0, 140.0),
        "elbow_l": (-42.0, 0.0, 105.0), "elbow_r": (42.0, 0.0, 105.0),
        "wrist_l": (-64.0, 0.0, 90.0), "wrist_r": (64.0, 0.0, 90.0),
        "knee_l": (-12.0, 0.0, 55.0), "knee_r": (12.0, 0.0, 55.0),
        "ankle_l": (-12.0, 0.0, 10.0), "ankle_r": (12.0, 0.0, 10.0),
        "big_toe_l": (-12.0, 15.0, 9.0), "big_toe_r": (12.0, 15.0, 9.0),
        "heel_l": (-12.0, -3.0, 9.0), "heel_r": (12.0, -3.0, 9.0),
        "small_toe_l": (-14.0, 14.0, 9.0), "small_toe_r": (14.0, 14.0, 9.0),
        "left_ear": (9.0, 2.0, 145.0), "right_ear": (-9.0, 2.0, 145.0),
        "left_eye": (2.0, 8.0, 146.0), "right_eye": (-2.0, 8.0, 146.0),
        "nose": (0.0, 12.0, 144.0),
    }
    for name, xyz in anchors.items():
        if name in COCO_ANCHORS:
            points[:, COCO_ANCHORS[name], :] = xyz
    confidence = np.ones((frames, 133), dtype=np.float64)
    return points, confidence


def _fixture(frames: int = 4):
    points, confidence = _base_points(frames)
    rig = Skeleton(str(ROOT / "Manny.FBX"))
    skel = MocapSkeleton(points, confidence, fbx_skel=rig)
    targets = extract_mocap_points(points)
    targets["small_toe_l"] = points[:, COCO_ANCHORS["small_toe_l"], :].copy()
    targets["small_toe_r"] = points[:, COCO_ANCHORS["small_toe_r"], :].copy()
    return rig, skel, targets, confidence


def _landmark_targets(solver: MocapIKSolver, targets: dict[str, np.ndarray], offsets: dict[str, np.ndarray], frames: int):
    state = np.zeros(solver.num_vars, dtype=np.float64)
    root_t, local_q = solver._state_to_local_rotations(state)
    positions, rotations = solver.forward_kinematics(root_t, local_q)
    output = {key: value.copy() for key, value in targets.items()}
    for side in ("l", "r"):
        for name in ("heel", "big_toe", "small_toe"):
            key = f"{name}_{side}"
            proxy = solver._foot_landmark_bone_index(name, side)
            observed = positions[proxy] + Rotation.from_quat(rotations[proxy]).apply(
                np.asarray(offsets[key], dtype=np.float64)
            )
            output[key] = np.tile(observed, (frames, 1))
    return state, output


@requires_manny_fbx
def test_proxy_default_and_explicit_proxy_are_residual_and_solve_identical():
    _, skel, targets, _ = _fixture(frames=1)
    measured = {key: value[0].copy() for key, value in targets.items()}
    default = MocapIKSolver(skel, target_sequence=targets)
    explicit = MocapIKSolver(skel, target_sequence=targets, foot_landmark_model="proxy")
    x = np.linspace(-0.1, 0.1, default.num_vars)
    ctx_default = default._precompute_frame(measured, frame_idx=0)
    ctx_explicit = explicit._precompute_frame(measured, frame_idx=0)
    residual_default = default._residuals_with_ctx(x, ctx_default)
    residual_explicit = explicit._residuals_with_ctx(x, ctx_explicit)
    assert np.array_equal(residual_default, residual_explicit)
    assert ctx_default["foot_landmark_packet"].size == 0
    assert ctx_explicit["foot_landmark_packet"].size == 0
    solved_default = default.solve_frame(measured, temporal_weight=0.0)
    solved_explicit = explicit.solve_frame(measured, temporal_weight=0.0)
    assert np.array_equal(solved_default, solved_explicit)


@requires_manny_fbx
def test_calibration_recovers_known_offsets_within_one_cm():
    rig, skel, targets, confidence = _fixture()
    solver = MocapIKSolver(skel, target_sequence=targets, foot_landmark_model="fk_offsets")
    known = {
        f"{name}_{side}": np.asarray(value, dtype=np.float64)
        for side in ("l", "r")
        for name, value in (
            ("heel", [0.0, 1.0, -1.0]),
            ("big_toe", [1.0, -1.0, 0.0]),
            ("small_toe", [-1.0, -1.0, 0.0]),
        )
    }
    state, synthetic_targets = _landmark_targets(solver, targets, known, len(next(iter(targets.values()))))
    solver.solve_frame = lambda measured, **kwargs: state.copy()
    diagnostics = solver.calibrate_foot_landmark_offsets(
        synthetic_targets, confidence=confidence, frame_count=4, stride=1, iterations=1
    )
    assert diagnostics["status"] == "calibrated"
    for key, expected in known.items():
        recovered = np.asarray(diagnostics["offsets_cm"][key])
        assert np.linalg.norm(recovered - expected) < 1.0, key
    assert diagnostics["loss"]["robust"] == "Huber IRLS"
    assert diagnostics["loss"]["huber_delta_cm"] == 2.0


@requires_manny_fbx
def test_invalid_calibration_falls_back_to_proxy():
    _, skel, targets, confidence = _fixture()
    solver = MocapIKSolver(skel, target_sequence=targets, foot_landmark_model="fk_offsets")
    bad = {key: value.copy() for key, value in targets.items()}
    for side in ("l", "r"):
        for name in ("heel", "big_toe", "small_toe"):
            bad[f"{name}_{side}"] = np.full((4, 3), 1000.0)
    state = np.zeros(solver.num_vars, dtype=np.float64)
    solver.solve_frame = lambda measured, **kwargs: state.copy()
    diagnostics = solver.calibrate_foot_landmark_offsets(
        bad, confidence=confidence, frame_count=4, stride=1, iterations=1
    )
    assert diagnostics["status"] == "fallback_proxy"
    assert diagnostics["effective_model"] == "proxy"
    assert solver.foot_landmark_model == "proxy"
    assert solver._foot_landmark_offsets == {}


if __name__ == "__main__":
    test_proxy_default_and_explicit_proxy_are_residual_and_solve_identical()
    test_calibration_recovers_known_offsets_within_one_cm()
    test_invalid_calibration_falls_back_to_proxy()
    print("foot landmark model tests passed")
