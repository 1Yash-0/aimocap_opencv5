"""Left/right swap repair in triangulation (P3.4).

``GeometryConfig.lr_swap_fix`` tests, per frame and camera, whether swapping
the labels of an arm, leg or whole-hand group lowers the group's robust
reprojection cost by ``lr_swap_ratio``, and applies the swap only when the
decision persists for two frames (or continues the previous decision).

These tests cover recovery of a swapped arm and hand, silence on clean data,
the temporal persistence rule on alternating labels, the legacy policy path
and unchanged output when the feature is off.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from aimocap.math.coords import opencv_to_internal
from aimocap.math.triangulate import project_points
from aimocap.triangulate.config import GeometryConfig
from aimocap.triangulate.engine import (
    LR_SWAP_GROUP_NAMES,
    triangulate_sequence_with_diagnostics,
)

FIXTURE = Path(__file__).parent / "fixtures" / "triangulate_soft_legacy.npz"

BASE_CONFIG = dict(
    min_detection_conf=0.5,
    max_reprojection_error_px=5.0,
    hard_reprojection_error_px=12.0,
    min_ray_angle_deg=0.0,
    min_depth_m=0.0,
    f_scale=10.0,
    min_inliers=2,
    hysteresis_margin_px=5.0,
    w_temporal=0.35,
    w_geom=2.0,
    w_anat=2.0,
)

ARM_LEFT = [5, 7, 9]
ARM_RIGHT = [6, 8, 10]
LEG_LEFT = [11, 13, 15]
LEG_RIGHT = [12, 14, 16]
HANDS_LEFT = list(range(91, 112))
HANDS_RIGHT = list(range(112, 133))


def _camera_at(position, target=np.array([0.0, -1.0, 0.0]), focal=1400.0):
    position = np.asarray(position, dtype=np.float64)
    forward = np.asarray(target, dtype=np.float64) - position
    forward /= np.linalg.norm(forward)
    up = np.array([0.0, -1.0, 0.0])
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)
    rotation = np.stack([right, -true_up, forward])
    translation = (-rotation @ position).reshape(3, 1)
    intrinsic = np.array(
        [[focal, 0.0, 960.0], [0.0, focal, 540.0], [0.0, 0.0, 1.0]]
    )
    return intrinsic, rotation, translation


def _cameras():
    return [
        _camera_at([4.0, -1.0, 0.0]),
        _camera_at([0.0, -1.0, 4.0]),
        _camera_at([-4.0, -1.0, 0.0]),
    ]


def _base_skeleton() -> np.ndarray:
    base = np.zeros((133, 3), dtype=np.float64)
    base[0] = (0.0, -1.62, 0.0)
    base[1] = (-0.03, -1.63, -0.04)
    base[2] = (0.03, -1.63, -0.04)
    base[3] = (-0.08, -1.62, 0.0)
    base[4] = (0.08, -1.62, 0.0)
    base[5] = (-0.18, -1.42, 0.0)
    base[6] = (0.18, -1.42, 0.0)
    base[7] = (-0.20, -1.13, 0.04)
    base[8] = (0.20, -1.13, 0.04)
    base[9] = (-0.20, -0.87, 0.04)
    base[10] = (0.20, -0.87, 0.04)
    base[11] = (-0.09, -1.0, 0.0)
    base[12] = (0.09, -1.0, 0.0)
    base[13] = (-0.09, -0.58, 0.0)
    base[14] = (0.09, -0.58, 0.0)
    base[15] = (-0.09, -0.17, 0.0)
    base[16] = (0.09, -0.17, 0.0)
    base[17] = (-0.09, -0.09, 0.07)
    base[18] = (-0.12, -0.11, 0.0)
    base[19] = (-0.06, -0.11, 0.03)
    base[20] = (0.09, -0.09, 0.07)
    base[21] = (0.12, -0.11, 0.0)
    base[22] = (0.06, -0.11, 0.03)
    base[23:91] = (0.0, -1.62, 0.0)
    for index in range(21):
        base[91 + index] = base[9] + np.array([0.01 * (index % 5), -0.01 * (index // 5), 0.0])
        base[112 + index] = base[10] + np.array([0.01 * (index % 5), -0.01 * (index // 5), 0.0])
    return base


def _sequence(num_frames: int = 6, noise: float = 0.4):
    rng = np.random.RandomState(7)
    base = _base_skeleton()
    cameras = _cameras()
    keypoints = np.zeros((num_frames, 3, 133, 2), dtype=np.float64)
    truths = np.zeros((num_frames, 133, 3), dtype=np.float64)
    for frame in range(num_frames):
        points = base.copy()
        phase = 2.0 * np.pi * frame / num_frames
        points[:, 1] += 0.01 * np.sin(phase)
        points[[7, 9], 0] += 0.02 * np.sin(phase)
        points[[8, 10], 0] -= 0.02 * np.sin(phase)
        truths[frame] = points
        for camera, (intrinsic, rotation, translation) in enumerate(cameras):
            projection = intrinsic @ np.hstack((rotation, translation.reshape(3, 1)))
            keypoints[frame, camera] = project_points(points, projection)
            keypoints[frame, camera] += rng.randn(133, 2) * noise
    scores = np.ones((num_frames, 3, 133), dtype=np.float64)
    intrinsics = [item[0] for item in cameras]
    extrinsics = [(item[1], item[2]) for item in cameras]
    return keypoints, scores, intrinsics, extrinsics, truths


def _swap_camera(keypoints, scores, camera: int, left, right, frames):
    for frame in frames:
        kp_left = keypoints[frame, camera, left].copy()
        keypoints[frame, camera, left] = keypoints[frame, camera, right]
        keypoints[frame, camera, right] = kp_left
        score_left = scores[frame, camera, left].copy()
        scores[frame, camera, left] = scores[frame, camera, right]
        scores[frame, camera, right] = score_left
    return keypoints, scores


def _soft_config(**overrides) -> GeometryConfig:
    values = dict(BASE_CONFIG)
    values.update(
        camera_policy="soft",
        soft_scale_px=6.0,
        soft_loo_ratio=2.0,
        soft_ema_alpha=0.5,
        soft_iterations=3,
    )
    values.update(overrides)
    return GeometryConfig(**values)


def _arm_error(diag, truths, joints):
    predicted = diag.points3d[:, joints]
    truth = opencv_to_internal(truths[:, joints])
    finite = np.isfinite(predicted).all(axis=2)
    return float(np.median(np.linalg.norm(predicted - truth, axis=2)[finite]))


def test_swapped_camera_arm_is_repaired():
    keypoints, scores, intrinsics, extrinsics, truths = _sequence()
    keypoints, scores = _swap_camera(
        keypoints, scores, 1, ARM_LEFT, ARM_RIGHT, range(len(keypoints))
    )
    diag = triangulate_sequence_with_diagnostics(
        keypoints,
        scores,
        intrinsics,
        extrinsics,
        geometry_config=_soft_config(lr_swap_fix=True, lr_swap_ratio=2.0),
    )
    arm_group = LR_SWAP_GROUP_NAMES.index("arm")
    accepted = diag.lr_swap_mask[:, 1, arm_group]
    # The first frame has no history, so the two-frame rule starts at frame 1.
    np.testing.assert_array_equal(
        accepted, np.array([False] + [True] * (len(keypoints) - 1))
    )
    assert not diag.lr_swap_mask[:, 0, arm_group].any()
    assert not diag.lr_swap_mask[:, 2, arm_group].any()
    error = _arm_error(diag, truths, ARM_LEFT + ARM_RIGHT)
    assert error < 0.02, f"arm error after repair {error * 1000:.1f} mm"

    unfixed = triangulate_sequence_with_diagnostics(
        keypoints,
        scores,
        intrinsics,
        extrinsics,
        geometry_config=_soft_config(lr_swap_fix=False),
    )
    assert _arm_error(unfixed, truths, ARM_LEFT + ARM_RIGHT) > 5.0 * error


def test_clean_data_accepts_no_swaps():
    keypoints, scores, intrinsics, extrinsics, _ = _sequence()
    diag = triangulate_sequence_with_diagnostics(
        keypoints,
        scores,
        intrinsics,
        extrinsics,
        geometry_config=_soft_config(lr_swap_fix=True, lr_swap_ratio=2.0),
    )
    assert diag.lr_swap_mask.shape == (len(keypoints), 3, len(LR_SWAP_GROUP_NAMES))
    assert not diag.lr_swap_mask.any()


def test_alternating_labels_do_not_toggle():
    keypoints, scores, intrinsics, extrinsics, _ = _sequence()
    keypoints, scores = _swap_camera(
        keypoints, scores, 2, ARM_LEFT, ARM_RIGHT, range(0, len(keypoints), 2)
    )
    diag = triangulate_sequence_with_diagnostics(
        keypoints,
        scores,
        intrinsics,
        extrinsics,
        geometry_config=_soft_config(lr_swap_fix=True, lr_swap_ratio=2.0),
    )
    arm_group = LR_SWAP_GROUP_NAMES.index("arm")
    assert not diag.lr_swap_mask[:, 2, arm_group].any()


def test_hand_group_swap_is_repaired():
    keypoints, scores, intrinsics, extrinsics, truths = _sequence()
    keypoints, scores = _swap_camera(
        keypoints, scores, 2, HANDS_LEFT, HANDS_RIGHT, range(len(keypoints))
    )
    diag = triangulate_sequence_with_diagnostics(
        keypoints,
        scores,
        intrinsics,
        extrinsics,
        geometry_config=_soft_config(lr_swap_fix=True, lr_swap_ratio=2.0),
    )
    hands_group = LR_SWAP_GROUP_NAMES.index("hands")
    accepted = diag.lr_swap_mask[:, 2, hands_group]
    np.testing.assert_array_equal(
        accepted, np.array([False] + [True] * (len(keypoints) - 1))
    )
    predicted = diag.points3d[:, HANDS_LEFT + HANDS_RIGHT]
    truth = opencv_to_internal(truths[:, HANDS_LEFT + HANDS_RIGHT])
    finite = np.isfinite(predicted).all(axis=2)
    error = float(np.median(np.linalg.norm(predicted - truth, axis=2)[finite]))
    assert error < 0.02


def test_legacy_policy_also_repairs_swap():
    keypoints, scores, intrinsics, extrinsics, truths = _sequence()
    keypoints, scores = _swap_camera(
        keypoints, scores, 0, ARM_LEFT, ARM_RIGHT, range(len(keypoints))
    )
    diag = triangulate_sequence_with_diagnostics(
        keypoints,
        scores,
        intrinsics,
        extrinsics,
        geometry_config=GeometryConfig(
            **BASE_CONFIG, camera_policy="legacy", lr_swap_fix=True, lr_swap_ratio=2.0
        ),
    )
    arm_group = LR_SWAP_GROUP_NAMES.index("arm")
    assert diag.lr_swap_mask[1:, 0, arm_group].all()
    # Legacy emits only confident points; compare the finite arm joints.
    predicted = diag.points3d[1:, ARM_LEFT + ARM_RIGHT]
    truth = opencv_to_internal(truths[1:, ARM_LEFT + ARM_RIGHT])
    finite = np.isfinite(predicted).all(axis=2)
    assert finite.any()
    error = float(np.median(np.linalg.norm(predicted - truth, axis=2)[finite]))
    assert error < 0.03


def test_legacy_output_is_bit_identical_with_feature_off():
    expected = np.load(FIXTURE)
    config = GeometryConfig(**BASE_CONFIG)
    assert config.lr_swap_fix is False
    assert config.lr_swap_ratio == 2.0
    # The shared legacy fixture is also asserted by test_triangulate_soft; here
    # we pin that the new diagnostics field is empty and does not alter arrays.
    from tests.test_triangulate_soft import _run as run_soft_test_scene

    diag = run_soft_test_scene(config)
    np.testing.assert_array_equal(diag.points3d, expected["points3d"])
    np.testing.assert_array_equal(diag.candidate_points3d, expected["candidate_points3d"])
    np.testing.assert_array_equal(diag.confidence, expected["confidence"])
    np.testing.assert_array_equal(diag.inlier_mask, expected["inlier_mask"])
    np.testing.assert_array_equal(
        diag.reprojection_error_px, expected["reprojection_error_px"]
    )
    np.testing.assert_array_equal(diag.camera_trust, expected["camera_trust"])
    assert not diag.lr_swap_mask.any()
