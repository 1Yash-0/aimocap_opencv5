"""Per-part confidence gating and weighting for triangulation.

The calibrated 2D scores are probabilities with different meanings per body
part, so triangulation gates and weights body, feet, face and hands
independently.  ``GeometryConfig.legacy_confidence()`` reproduces the P3.1
behaviour (every group at 0.5, gamma 1) bit-exactly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aimocap.math.triangulate import project_points
from aimocap.triangulate.config import GeometryConfig, PartConfidence
from aimocap.triangulate.engine import triangulate_sequence_with_diagnostics

FIXTURE = Path(__file__).parent / "fixtures" / "triangulate_confidence_p31.npz"

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


def _camera_at(position, target=np.array([0.0, -1.0, 0.0]), focal=1400.0):
    """OpenCV world-to-camera model (x right, y down, z forward)."""
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


def _synthetic_sequence():
    """Deterministic 4-frame, 3-camera COCO-WholeBody sequence."""
    rng = np.random.RandomState(7)
    base = np.zeros((133, 3), dtype=np.float64)
    base[0] = (0.0, -1.62, 0.0)
    base[5] = (-0.18, -1.42, 0.0)
    base[6] = (0.18, -1.42, 0.0)
    base[7] = (-0.18, -1.126, 0.0)
    base[8] = (0.18, -1.12, 0.0)
    base[9] = (-0.18, -0.867, 0.0)
    base[10] = (0.18, -0.856, 0.0)
    base[11] = (-0.09, -1.0, 0.0)
    base[12] = (0.09, -1.0, 0.0)
    base[13] = (-0.09, -0.584, 0.0)
    base[14] = (0.09, -0.605, 0.0)
    base[15] = (-0.09, -0.168, 0.0)
    base[16] = (0.09, -0.230, 0.0)
    base[17:23] = np.array([0.0, -0.12, 0.02]) + rng.randn(6, 3) * 0.03
    base[23:] = np.array([0.0, -1.62, 0.0]) + rng.randn(110, 3) * 0.12
    base[91:112] = base[9] + rng.randn(21, 3) * 0.03
    base[112:133] = base[10] + rng.randn(21, 3) * 0.03

    cameras = [
        _camera_at([4.0, -1.0, 0.0]),
        _camera_at([0.0, -1.0, 4.0]),
        _camera_at([-4.0, -1.0, 0.0]),
    ]
    num_frames = 4
    num_joints = 133
    keypoints = np.zeros((num_frames, 3, num_joints, 2), dtype=np.float64)
    for frame in range(num_frames):
        phase = 2.0 * np.pi * frame / num_frames
        points = base.copy()
        points[:, 1] += 0.015 * np.sin(phase)
        points[[13, 15], 0] += 0.05 * np.sin(phase)
        points[[14, 16], 0] -= 0.05 * np.sin(phase)
        for camera, (intrinsic, rotation, translation) in enumerate(cameras):
            projection = intrinsic @ np.hstack((rotation, translation.reshape(3, 1)))
            keypoints[frame, camera] = project_points(points, projection)
            keypoints[frame, camera] += rng.randn(num_joints, 2) * 1.0

    scores = np.ones((num_frames, 3, num_joints), dtype=np.float64)
    # Low but valid values across groups and one missing score.
    scores[1, 0, 5] = 0.3
    scores[1, 0, 18] = 0.3
    scores[2, 1, 30] = 0.3
    scores[2, 1, 100] = 0.3
    scores[3, 2, 20] = np.nan

    intrinsics = [item[0] for item in cameras]
    extrinsics = [(item[1], item[2]) for item in cameras]
    return keypoints, scores, intrinsics, extrinsics


def _run(config: GeometryConfig):
    keypoints, scores, intrinsics, extrinsics = _synthetic_sequence()
    return triangulate_sequence_with_diagnostics(
        keypoints, scores, intrinsics, extrinsics, geometry_config=config
    )


def test_legacy_preset_matches_p31_fixture_bit_exactly():
    expected = np.load(FIXTURE)
    diag = _run(GeometryConfig.legacy_confidence(**BASE_CONFIG))
    np.testing.assert_array_equal(diag.points3d, expected["points3d"])
    np.testing.assert_array_equal(diag.candidate_points3d, expected["candidate_points3d"])
    np.testing.assert_array_equal(diag.confidence, expected["confidence"])
    np.testing.assert_array_equal(diag.inlier_mask, expected["inlier_mask"])
    np.testing.assert_array_equal(
        diag.reprojection_error_px, expected["reprojection_error_px"]
    )


def test_per_group_min_conf_applies_to_the_right_joint_indices():
    keypoints, scores, intrinsics, extrinsics = _synthetic_sequence()
    for joint in (5, 18, 30, 100):
        scores[1, 0, joint] = 0.2
    config = GeometryConfig(
        **BASE_CONFIG,
        body_confidence=PartConfidence(0.5, 1.0),
        feet_confidence=PartConfidence(0.1, 1.0),
        face_confidence=PartConfidence(0.05, 1.0),
        hands_confidence=PartConfidence(0.5, 1.0),
    )
    diag = triangulate_sequence_with_diagnostics(
        keypoints, scores, intrinsics, extrinsics, geometry_config=config
    )
    # body (5) and hands (100) are gated; feet (18) and face (30) survive.
    assert diag.camera_rejection_reason[1, 5, 0] == "low_confidence"
    assert diag.camera_rejection_reason[1, 100, 0] == "low_confidence"
    assert diag.observed_camera_count[1, 5] == 2
    assert diag.observed_camera_count[1, 100] == 2
    assert diag.observed_camera_count[1, 18] == 3
    assert diag.observed_camera_count[1, 30] == 3
    assert np.isfinite(diag.points3d[1, 18]).all()
    assert np.isfinite(diag.points3d[1, 30]).all()


def test_nan_scores_and_gamma_are_safe():
    keypoints, scores, intrinsics, extrinsics = _synthetic_sequence()
    scores[0, 0, 7] = np.nan
    scores[0, 1, 95] = np.nan
    keypoints[1, 0, 40] = np.nan
    config = GeometryConfig(
        **BASE_CONFIG,
        body_confidence=PartConfidence(0.1, 2.0),
        feet_confidence=PartConfidence(0.1, 0.5),
        face_confidence=PartConfidence(0.02, 0.5),
        hands_confidence=PartConfidence(0.01, 2.0),
    )
    diag = triangulate_sequence_with_diagnostics(
        keypoints, scores, intrinsics, extrinsics, geometry_config=config
    )
    assert diag.points3d.shape == (4, 133, 3)
    assert np.isfinite(diag.points3d[0, 7]).all()
    assert np.isfinite(diag.points3d[0, 95]).all()
    assert np.isfinite(diag.points3d[1, 40]).all()
    assert np.isfinite(diag.confidence[:, [7, 40, 95]]).all()


def test_weight_transform_is_power_of_score():
    assert PartConfidence(min_conf=0.1, gamma=1.0).weight(0.37) == 0.37
    assert PartConfidence(min_conf=0.1, gamma=2.0).weight(0.5) == pytest.approx(0.25)
    assert PartConfidence(min_conf=0.1, gamma=0.5).weight(0.25) == pytest.approx(0.5)


def test_confidence_policy_validation_and_joint_groups():
    with pytest.raises(ValueError):
        PartConfidence(min_conf=1.5)
    with pytest.raises(ValueError):
        PartConfidence(gamma=0.0)
    config = GeometryConfig(
        min_detection_conf=0.3,
        body_confidence=PartConfidence(0.1, 2.0),
        feet_confidence=PartConfidence(0.2, 1.0),
        face_confidence=PartConfidence(0.02, 0.5),
        hands_confidence=PartConfidence(0.05, 1.5),
    )
    assert config.confidence_for_joint(0).min_conf == 0.1
    assert config.confidence_for_joint(17).min_conf == 0.2
    assert config.confidence_for_joint(23).min_conf == 0.02
    assert config.confidence_for_joint(132).min_conf == 0.05
    assert GeometryConfig(min_detection_conf=0.3).confidence_for_joint(50).min_conf == 0.3


def _write_fixture() -> None:
    """Regenerate the P3.1 expected output (run before changing the engine)."""
    diag = _run(GeometryConfig(**BASE_CONFIG))
    np.savez(
        FIXTURE,
        points3d=diag.points3d,
        candidate_points3d=diag.candidate_points3d,
        confidence=diag.confidence,
        inlier_mask=diag.inlier_mask,
        reprojection_error_px=diag.reprojection_error_px,
    )
    print(f"wrote {FIXTURE}")


if __name__ == "__main__":
    _write_fixture()
