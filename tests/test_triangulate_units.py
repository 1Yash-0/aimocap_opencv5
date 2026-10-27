"""Unit/unit-test coverage for the metre/centimetre contract in triangulation.

The engine triangulates in metres.  The temporal weight was tuned for
centimetre displacements and the anatomical table is authored in centimetres.
These tests pin the corrected conversion and the ``legacy_unit_behaviour``
escape hatch that reproduces the historical (unit-mixed) numbers bit-exactly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aimocap.math.triangulate import project_points
from aimocap.triangulate.config import GeometryConfig
from aimocap.triangulate.engine import triangulate_sequence_with_diagnostics

FIXTURE = Path(__file__).parent / "fixtures" / "triangulate_units_legacy.npz"

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
    """Deterministic 6-frame, 3-camera COCO-WholeBody sequence in OpenCV world.

    Limb lengths follow the production table (values in cm) so the anatomical
    term is meaningful, and the limbs swing several centimetres per frame so
    the temporal term is meaningful too.  A few large but sub-threshold pixel
    offsets force competing camera subsets.
    """
    rng = np.random.RandomState(0)
    base = np.zeros((133, 3), dtype=np.float64)
    base[0] = (0.0, -1.62, 0.0)
    base[1] = (-0.03, -1.63, -0.04)
    base[2] = (0.03, -1.63, -0.04)
    base[3] = (-0.08, -1.62, 0.0)
    base[4] = (0.08, -1.62, 0.0)
    base[5] = (-0.18, -1.42, 0.0)
    base[6] = (0.18, -1.42, 0.0)
    base[7] = (-0.18, -1.126, 0.0)   # 29.4 cm upper arm
    base[8] = (0.18, -1.12, 0.0)     # 30.0 cm upper arm
    base[9] = (-0.18, -0.867, 0.0)   # 25.9 cm forearm
    base[10] = (0.18, -0.856, 0.0)   # 26.4 cm forearm
    base[11] = (-0.09, -1.0, 0.0)
    base[12] = (0.09, -1.0, 0.0)
    base[13] = (-0.09, -0.584, 0.0)  # 41.6 cm thigh
    base[14] = (0.09, -0.605, 0.0)   # 39.5 cm thigh
    base[15] = (-0.09, -0.168, 0.0)  # 41.6 cm shin
    base[16] = (0.09, -0.230, 0.0)   # 37.5 cm shin
    base[17] = (-0.09, -0.10, 0.05)
    base[18] = (-0.12, -0.12, 0.0)
    base[19] = (-0.06, -0.12, 0.02)
    base[20] = (0.09, -0.16, 0.05)
    base[21] = (0.12, -0.18, 0.0)
    base[22] = (0.06, -0.18, 0.02)
    base[23:] = np.array([0.0, -1.62, 0.0]) + rng.randn(110, 3) * 0.12
    base[91:112] = base[9] + rng.randn(21, 3) * 0.03
    base[112:133] = base[10] + rng.randn(21, 3) * 0.03

    num_frames = 6
    num_cameras = 3
    num_joints = 133
    cameras = [
        _camera_at([4.0, -1.0, 0.0]),
        _camera_at([0.0, -1.0, 4.0]),
        _camera_at([-4.0, -1.0, 0.0]),
    ]
    keypoints = np.zeros((num_frames, num_cameras, num_joints, 2), dtype=np.float64)
    for frame in range(num_frames):
        phase = 2.0 * np.pi * frame / num_frames
        points = base.copy()
        points[:, 1] += 0.015 * np.sin(phase)
        points[[13, 15], 0] += 0.05 * np.sin(phase)
        points[[14, 16], 0] -= 0.05 * np.sin(phase)
        points[[7, 9], 0] += 0.04 * np.sin(phase + np.pi / 3.0)
        points[[8, 10], 0] -= 0.04 * np.sin(phase + np.pi / 3.0)
        for camera, (intrinsic, rotation, translation) in enumerate(cameras):
            projection = intrinsic @ np.hstack((rotation, translation.reshape(3, 1)))
            keypoints[frame, camera] = project_points(points, projection)
            keypoints[frame, camera] += rng.randn(num_joints, 2) * 1.2
    # Large but sub-hard-threshold offsets create genuinely different candidate
    # points for different camera pairs.
    keypoints[2, 2, 12] += np.array([6.0, -4.0])
    keypoints[3, 1, 14] += np.array([-7.0, 5.0])
    keypoints[4, 2, 7] += np.array([5.0, 6.0])
    scores = np.ones((num_frames, num_cameras, num_joints), dtype=np.float64)
    scores[4, 0, 20] = 0.2
    scores[3, 1, 25] = 0.0

    intrinsics = [item[0] for item in cameras]
    extrinsics = [(item[1], item[2]) for item in cameras]
    return keypoints, scores, intrinsics, extrinsics


def _run(legacy_units: bool | None = None):
    keypoints, scores, intrinsics, extrinsics = _synthetic_sequence()
    kwargs = dict(BASE_CONFIG)
    if legacy_units is not None:
        kwargs["legacy_unit_behaviour"] = legacy_units
    return triangulate_sequence_with_diagnostics(
        keypoints,
        scores,
        intrinsics,
        extrinsics,
        geometry_config=GeometryConfig(**kwargs),
    )


def test_legacy_unit_behaviour_reproduces_previous_output_bit_exactly():
    expected = np.load(FIXTURE)
    diag = _run(legacy_units=True)
    np.testing.assert_array_equal(diag.points3d, expected["points3d"])
    np.testing.assert_array_equal(diag.candidate_points3d, expected["candidate_points3d"])
    np.testing.assert_array_equal(diag.confidence, expected["confidence"])
    np.testing.assert_array_equal(diag.inlier_mask, expected["inlier_mask"])
    np.testing.assert_array_equal(
        diag.reprojection_error_px, expected["reprojection_error_px"]
    )


def test_bone_penalty_distinguishes_correct_and_wrong_length_candidates():
    from aimocap.triangulate.engine import _temporal_anatomy_penalties

    connected = np.array([0.0, 0.0, 0.0])
    nominal_cm = 39.5
    correct = np.array([nominal_cm / 100.0, 0.0, 0.0])
    wrong = np.array([0.20, 0.0, 0.0])  # 20 cm: 49% too short
    _, correct_fixed = _temporal_anatomy_penalties(
        correct,
        prev_point3d=None,
        prev_velocity3d=None,
        connected_point3d=connected,
        ref_bone_length_cm=nominal_cm,
        legacy_units=False,
    )
    _, wrong_fixed = _temporal_anatomy_penalties(
        wrong,
        prev_point3d=None,
        prev_velocity3d=None,
        connected_point3d=connected,
        ref_bone_length_cm=nominal_cm,
        legacy_units=False,
    )
    assert correct_fixed == pytest.approx(0.0, abs=1e-12)
    assert wrong_fixed > 0.5

    _, correct_legacy = _temporal_anatomy_penalties(
        correct,
        prev_point3d=None,
        prev_velocity3d=None,
        connected_point3d=connected,
        ref_bone_length_cm=nominal_cm,
        legacy_units=True,
    )
    _, wrong_legacy = _temporal_anatomy_penalties(
        wrong,
        prev_point3d=None,
        prev_velocity3d=None,
        connected_point3d=connected,
        ref_bone_length_cm=nominal_cm,
        legacy_units=True,
    )
    assert abs(wrong_legacy - correct_legacy) < 0.1


def test_temporal_penalty_scales_with_displacement_in_metres():
    from aimocap.triangulate.engine import _temporal_anatomy_penalties

    previous = np.zeros(3)
    near = np.array([0.01, 0.0, 0.0])  # 1 cm
    far = np.array([0.10, 0.0, 0.0])   # 10 cm
    near_penalty, _ = _temporal_anatomy_penalties(
        near, prev_point3d=previous, prev_velocity3d=None,
        connected_point3d=None, ref_bone_length_cm=None, legacy_units=False,
    )
    far_penalty, _ = _temporal_anatomy_penalties(
        far, prev_point3d=previous, prev_velocity3d=None,
        connected_point3d=None, ref_bone_length_cm=None, legacy_units=False,
    )
    assert near_penalty == pytest.approx(1.0)
    assert far_penalty == pytest.approx(10.0)
    assert far_penalty == pytest.approx(10.0 * near_penalty)
    legacy_near, _ = _temporal_anatomy_penalties(
        near, prev_point3d=previous, prev_velocity3d=None,
        connected_point3d=None, ref_bone_length_cm=None, legacy_units=True,
    )
    legacy_far, _ = _temporal_anatomy_penalties(
        far, prev_point3d=previous, prev_velocity3d=None,
        connected_point3d=None, ref_bone_length_cm=None, legacy_units=True,
    )
    assert legacy_near == pytest.approx(0.01)
    assert legacy_far == pytest.approx(0.10)


def test_fixed_behaviour_changes_scoring_on_synthetic_sequence():
    legacy = _run(legacy_units=True)
    fixed = _run(legacy_units=False)
    assert not np.array_equal(legacy.points3d, fixed.points3d)


def _write_fixture() -> None:
    """Regenerate the pre-fix expected output (run once against the old engine)."""
    diag = _run()
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
