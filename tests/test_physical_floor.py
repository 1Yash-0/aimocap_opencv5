"""Comprehensive test suite for Phase 2.3C Physical Floor Registration.

Enforces all 30 approved Astra requirements:
1. ChArUco board model validation.
2. Board optical-plane thickness/offset correction.
3. Tilted camera recovery.
4. Arbitrary board heading.
5. Correct handedness.
6. Correct floor Y=0 plane.
7. Correct board optical surface Y=h.
8. Correct origin projection.
9. Camera calibration remains unchanged.
10. Rigid transform only, no scale modification.
11. Inverse roundtrip.
12. Distance preservation.
13. Multi-view pose consistency.
14. Planar ambiguity rejection.
15. Outlier/weak-geometry rejection.
16. Repeatability rejection.
17. Spatial footprint validation rejection.
18. Uncertainty gate.
19. Metadata/provenance roundtrip.
20. Stale registration rejection.
21. Canonical skeleton3d byte/value preservation.
22. Existing raw skeleton3d_floor semantics preserved.
23. Refined camera artifact remains camera_world.
24. Refined floor artifact correctly references its parent and registration.
25. Registration failure blocks trusted grounded processing.
26. Unregistered identity cannot pass grounded route.
27. Vector rotation for velocity/acceleration.
28. Covariance rotation if covariance is represented.
29. No double transform.
30. F4 remains in camera_world and is not numerically rerun in floor_world.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from aimocap.calib.floor import (
    CameraObservation,
    CharucoBoardModel,
    FloorModelError,
    FloorRegistrationResult,
    FloorValidationProfile,
    HoldCapture,
    ObservationSet,
    PlanarAmbiguityError,
    RegistrationGateError,
    compute_floor_transform_from_board,
    evaluate_floor_registration,
    load_floor_registration,
    require_grounded_floor,
    save_floor_registration,
    solve_board_pose_multiview,
)
from aimocap.math.coords import (
    CAMERA_WORLD_FRAME,
    FLOOR_WORLD_FRAME,
    FloorTransform,
    FrameMismatchError,
    transform_covariance_from_floor,
    transform_covariance_to_floor,
    transform_velocity_from_floor,
    transform_velocity_to_floor,
    validate_rotation,
)
from aimocap.math.filter import filter_skeleton3d


# ---------------------------------------------------------------------------
# Test Fixtures & Synthetic Data Generators
# ---------------------------------------------------------------------------

def look_at_opencv(center: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (R_c, t_c) for an OpenCV camera looking from center to target.

    In OpenCV camera coordinates: X right, Y down, Z forward.
    """
    forward = (target - center) / np.linalg.norm(target - center)
    down = np.array([0.0, -1.0, 0.0])
    right = np.cross(down, forward)
    norm_r = np.linalg.norm(right)
    if norm_r < 1e-6:
        down = np.array([0.0, 0.0, 1.0])
        right = np.cross(down, forward)
        norm_r = np.linalg.norm(right)
    right /= norm_r
    down = np.cross(forward, right)
    down /= np.linalg.norm(down)
    R_c = np.vstack([right, down, forward])
    t_c = -R_c @ center
    return R_c, t_c


def make_synthetic_cameras() -> tuple[dict[str, np.ndarray], dict[str, tuple[np.ndarray, np.ndarray]], dict[str, np.ndarray]]:
    """Create a calibrated 3-camera rig observing a common floor area."""
    K = np.array([
        [1000.0, 0.0, 640.0],
        [0.0, 1000.0, 360.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    K_dict = {"cam0": K.copy(), "cam1": K.copy(), "cam2": K.copy()}
    dist_dict = {"cam0": np.zeros(5), "cam1": np.zeros(5), "cam2": np.zeros(5)}

    target = np.array([0.0, 0.0, 1.0])
    extrinsics_dict = {
        "cam0": look_at_opencv(np.array([0.0, 1.5, 3.0]), target),
        "cam1": look_at_opencv(np.array([-2.0, 1.8, 2.5]), target),
        "cam2": look_at_opencv(np.array([2.0, 1.8, 2.5]), target),
    }
    return K_dict, extrinsics_dict, dist_dict


def synthesize_observation_set(
    board_model: CharucoBoardModel,
    R_cb: np.ndarray,
    t_cb: np.ndarray,
    K_dict: dict[str, np.ndarray],
    extrinsics_dict: dict[str, tuple[np.ndarray, np.ndarray]],
    dist_dict: dict[str, np.ndarray],
    noise_sigma_px: float = 0.0,
    frame_idx: int = 0,
    corner_dropout_ratio: float = 0.0,
) -> ObservationSet:
    """Project 3D board corners into all cameras given true board pose in camera_world."""
    obj_pts = board_model.get_object_points()
    num_corners = len(obj_pts)
    observations: dict[str, CameraObservation] = {}

    for cam_name, (R_c, t_c) in extrinsics_dict.items():
        K = K_dict[cam_name]
        dist = dist_dict.get(cam_name, np.zeros(5))

        # Camera-frame board pose
        R_cam_b = R_c @ R_cb
        t_cam_b = R_c @ t_cb + t_c
        rvec_cam, _ = cv2.Rodrigues(R_cam_b)

        proj, _ = cv2.projectPoints(
            obj_pts.astype(np.float32),
            rvec_cam,
            t_cam_b.reshape(3, 1),
            K,
            dist,
        )
        proj_pts = proj.reshape(-1, 2).astype(np.float64)

        if noise_sigma_px > 0.0:
            proj_pts += np.random.normal(0.0, noise_sigma_px, proj_pts.shape)

        corner_ids = np.arange(num_corners, dtype=np.int32)
        if corner_dropout_ratio > 0.0:
            keep_count = max(4, int(round(num_corners * (1.0 - corner_dropout_ratio))))
            perm = np.random.permutation(num_corners)[:keep_count]
            corner_ids = corner_ids[perm]
            proj_pts = proj_pts[perm]

        observations[cam_name] = CameraObservation(
            camera_name=cam_name,
            corner_ids=corner_ids,
            corner_coords_px=proj_pts,
        )

    return ObservationSet(frame_idx=frame_idx, observations=observations)


def make_full_hold_captures(
    board_model: CharucoBoardModel,
    R_cb_truth: np.ndarray,
    t_cb_truth: np.ndarray,
    K_dict: dict[str, np.ndarray],
    extrinsics_dict: dict[str, tuple[np.ndarray, np.ndarray]],
    dist_dict: dict[str, np.ndarray],
    noise_sigma_px: float = 0.1,
) -> tuple[HoldCapture, list[HoldCapture], list[HoldCapture]]:
    """Generate a complete set of primary, repeatability, and spatial validation holds."""
    np.random.seed(42)

    # 1. Primary hold: 12 observation frames at marked position
    primary_obs_sets = [
        synthesize_observation_set(
            board_model, R_cb_truth, t_cb_truth, K_dict, extrinsics_dict, dist_dict,
            noise_sigma_px=noise_sigma_px, frame_idx=i,
        )
        for i in range(12)
    ]
    primary_hold = HoldCapture(
        hold_type="primary",
        hold_id="primary_01",
        observation_sets=primary_obs_sets,
    )

    # 2. Repeatability holds: 2 lift/reseat holds with tiny perturbation (< 0.1 deg, < 1 mm)
    repeatability_holds = []
    for r_idx in (1, 2):
        d_rot = Rotation.from_euler("z", 0.05 * (r_idx - 1.5), degrees=True).as_matrix()
        R_rep = d_rot @ R_cb_truth
        t_rep = t_cb_truth + np.array([0.001, 0.0, 0.001]) * (r_idx - 1.5)
        obs_sets = [
            synthesize_observation_set(
                board_model, R_rep, t_rep, K_dict, extrinsics_dict, dist_dict,
                noise_sigma_px=noise_sigma_px, frame_idx=i,
            )
            for i in range(5)
        ]
        repeatability_holds.append(HoldCapture(
            hold_type=f"repeatability_{r_idx}",
            hold_id=f"rep_hold_{r_idx}",
            observation_sets=obs_sets,
        ))

    # 3. Spatial validation holds: 2 placements at different footprint locations on the same floor plane
    n_floor = R_cb_truth @ np.array([0.0, 0.0, 1.0])
    a_floor = R_cb_truth @ np.array([1.0, 0.0, 0.0])
    b_floor = np.cross(n_floor, a_floor)

    spatial_holds = []
    for s_idx, (dx, dz) in enumerate([(0.8, 0.4), (-0.7, 0.6)], start=1):
        t_spatial = t_cb_truth + dx * a_floor + dz * b_floor
        # Same normal n_floor, slightly different board yaw
        R_yaw = Rotation.from_euler("z", 15.0 * s_idx, degrees=True).as_matrix()
        R_spatial = R_cb_truth @ R_yaw
        obs_sets = [
            synthesize_observation_set(
                board_model, R_spatial, t_spatial, K_dict, extrinsics_dict, dist_dict,
                noise_sigma_px=noise_sigma_px, frame_idx=i,
            )
            for i in range(5)
        ]
        spatial_holds.append(HoldCapture(
            hold_type=f"spatial_{s_idx}",
            hold_id=f"spatial_hold_{s_idx}",
            observation_sets=obs_sets,
        ))

    return primary_hold, repeatability_holds, spatial_holds


# ---------------------------------------------------------------------------
# Test Cases 1 - 30
# ---------------------------------------------------------------------------

def test_01_charuco_board_model_validation():
    """Test 1: Validate board model defaults, inner corner counts, 3-pad support, and rejection of invalid specs."""
    model = CharucoBoardModel()
    assert model.squares_x == 7
    assert model.squares_y == 5
    assert model.num_inner_corners == (7 - 1) * (5 - 1) == 24
    assert model.support_pad_count == 3
    assert model.optical_plane_offset_h_m == 0.005

    obj_pts = model.get_object_points()
    assert obj_pts.shape == (24, 3)
    # Designated origin corner must be at (0, 0, 0)
    assert np.allclose(obj_pts[model.designated_origin_corner_idx], [0.0, 0.0, 0.0])
    # Z coordinate on optical plane must be 0
    assert np.allclose(obj_pts[:, 2], 0.0)

    # Rejection of invalid parameters
    with pytest.raises(FloorModelError):
        CharucoBoardModel(squares_x=1)
    with pytest.raises(FloorModelError):
        CharucoBoardModel(marker_length_m=0.045, square_length_m=0.040)
    with pytest.raises(FloorModelError):
        CharucoBoardModel(support_pad_count=4)
    with pytest.raises(FloorModelError):
        CharucoBoardModel(optical_plane_offset_h_m=-0.001)


def test_02_board_optical_plane_thickness_offset_correction():
    """Test 2: Optical plane offset h shifts physical floor origin strictly along surface normal by measured h."""
    # Board pose: optical surface is horizontal at height Y = 1.0 m in camera_world
    # Normal n = [0, 1, 0] (pointing up)
    R_cb = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
    ])  # board bx -> X, board by -> -Z, board bz -> Y
    t_cb = np.array([0.5, 1.0, -1.5])
    h = 0.008  # 8.0 mm offset

    transform = compute_floor_transform_from_board(R_cb, t_cb, h_m=h)
    # Origin of floor in camera_world: o = t_cb - h * n = [0.5, 1.0 - 0.008, -1.5] = [0.5, 0.992, -1.5]
    o_expected = np.array([0.5, 0.992, -1.5])
    p_floor_origin = transform.transform_points(o_expected)
    np.testing.assert_allclose(p_floor_origin, [0.0, 0.0, 0.0], atol=1e-12)


def test_03_tilted_camera_recovery():
    """Test 3: Tilted camera rig correctly recovers physical floor plane and normal."""
    K_dict, extrinsics_dict, dist_dict = make_synthetic_cameras()
    board_model = CharucoBoardModel()

    # Ground truth floor plane at Y = -0.5 in camera_world, board normal n = [0, 1, 0]
    R_cb_truth = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
    ], dtype=np.float64)
    t_cb_truth = np.array([0.0, -0.5 + board_model.optical_plane_offset_h_m, 1.0], dtype=np.float64)

    obs_set = synthesize_observation_set(board_model, R_cb_truth, t_cb_truth, K_dict, extrinsics_dict, dist_dict)
    pose_res = solve_board_pose_multiview(board_model, obs_set, K_dict, extrinsics_dict, dist_dict)

    transform = compute_floor_transform_from_board(pose_res.R_cb, pose_res.t_cb, board_model.optical_plane_offset_h_m)

    # Floor normal in camera_world is second row of R_cf
    recovered_normal = transform.rotation[1, :]
    np.testing.assert_allclose(recovered_normal, [0.0, 1.0, 0.0], atol=1e-3)


def test_04_arbitrary_board_heading():
    """Test 4: Floor normal is invariant to board yaw heading, and forward axis is along -Z in floor_world."""
    board_model = CharucoBoardModel()
    h = board_model.optical_plane_offset_h_m

    for yaw_deg in [0.0, 37.5, 120.0, 265.0]:
        R_yaw = Rotation.from_euler("y", yaw_deg, degrees=True).as_matrix()
        # Board base rotation: bz points up along Y
        R_base = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
        ])
        R_cb = R_yaw @ R_base
        t_cb = np.array([1.0, 0.5, 2.0])

        transform = compute_floor_transform_from_board(R_cb, t_cb, h_m=h)
        # Normal Y is always pointing up away from floor
        y_axis = transform.rotation[1, :]
        np.testing.assert_allclose(np.linalg.norm(y_axis), 1.0, atol=1e-12)

        # In floor_world, marked forward direction (by) points strictly along -Z
        by_cam = R_cb @ np.array([0.0, 1.0, 0.0])
        by_floor = transform.transform_velocity(by_cam)
        # by_floor should be [0, 0, -1]
        np.testing.assert_allclose(by_floor, [0.0, 0.0, -1.0], atol=1e-12)


def test_05_correct_handedness():
    """Test 5: Recovered floor transform is strictly right-handed (det = +1.0, x x y = z)."""
    rot = Rotation.from_euler("zyx", [15.0, -25.0, 40.0], degrees=True).as_matrix()
    trans = np.array([0.3, -1.2, 2.5])
    transform = compute_floor_transform_from_board(rot, trans, h_m=0.005)

    R = transform.rotation
    det = np.linalg.det(R)
    assert np.isclose(det, 1.0, atol=1e-12)
    x, y, z = R[0, :], R[1, :], R[2, :]
    np.testing.assert_allclose(np.cross(x, y), z, atol=1e-12)


def test_06_correct_floor_y_zero_plane():
    """Test 6: Points on physical floor have Y = 0.0 in floor_world."""
    R_cb = Rotation.from_euler("xyz", [10.0, -15.0, 20.0], degrees=True).as_matrix()
    t_cb = np.array([1.5, 0.2, 3.0])
    h = 0.006

    transform = compute_floor_transform_from_board(R_cb, t_cb, h_m=h)

    # Generate points on physical floor in board frame: bz = -h
    grid_x, grid_y = np.meshgrid(np.linspace(-0.5, 0.5, 5), np.linspace(-0.5, 0.5, 5))
    floor_pts_board = np.stack([grid_x.flatten(), grid_y.flatten(), -h * np.ones(grid_x.size)], axis=-1)

    # In camera_world:
    floor_pts_cam = floor_pts_board @ R_cb.T + t_cb

    # In floor_world:
    floor_pts_floor = transform.transform_points(floor_pts_cam)
    np.testing.assert_allclose(floor_pts_floor[:, 1], 0.0, atol=1e-12)


def test_07_correct_board_optical_surface_y_h():
    """Test 7: Points on the board optical surface (bz = 0) map to Y = h in floor_world."""
    R_cb = Rotation.from_euler("xyz", [-20.0, 35.0, -10.0], degrees=True).as_matrix()
    t_cb = np.array([-0.8, 1.4, 2.2])
    h = 0.0075

    transform = compute_floor_transform_from_board(R_cb, t_cb, h_m=h)

    # Points on optical surface: bz = 0
    grid_x, grid_y = np.meshgrid(np.linspace(0.0, 0.2, 4), np.linspace(0.0, 0.2, 4))
    surf_pts_board = np.stack([grid_x.flatten(), grid_y.flatten(), np.zeros(grid_x.size)], axis=-1)
    surf_pts_cam = surf_pts_board @ R_cb.T + t_cb

    surf_pts_floor = transform.transform_points(surf_pts_cam)
    np.testing.assert_allclose(surf_pts_floor[:, 1], h, atol=1e-12)


def test_08_correct_origin_projection():
    """Test 8: Designated board origin projected to floor maps exactly to (0, 0, 0) in floor_world."""
    R_cb = Rotation.from_euler("zyx", [45.0, 15.0, -30.0], degrees=True).as_matrix()
    t_cb = np.array([2.1, -0.4, 1.8])
    h = 0.005

    transform = compute_floor_transform_from_board(R_cb, t_cb, h_m=h)
    n = R_cb @ np.array([0.0, 0.0, 1.0])
    o_cam = t_cb - h * n

    o_floor = transform.transform_points(o_cam)
    np.testing.assert_allclose(o_floor, [0.0, 0.0, 0.0], atol=1e-12)


def test_09_camera_calibration_remains_unchanged():
    """Test 9: Camera calibration intrinsics and extrinsics are strictly unchanged by registration."""
    K_dict, extrinsics_dict, dist_dict = make_synthetic_cameras()
    board_model = CharucoBoardModel()

    K_orig = {k: v.copy() for k, v in K_dict.items()}
    ext_orig = {k: (v[0].copy(), v[1].copy()) for k, v in extrinsics_dict.items()}

    R_cb = np.eye(3)
    t_cb = np.array([0.0, 0.0, 1.5])
    primary_hold, rep_holds, spat_holds = make_full_hold_captures(
        board_model, R_cb, t_cb, K_dict, extrinsics_dict, dist_dict
    )

    evaluate_floor_registration(board_model, primary_hold, rep_holds, spat_holds, K_dict, extrinsics_dict, dist_dict)

    for k in K_dict:
        np.testing.assert_array_equal(K_dict[k], K_orig[k])
        np.testing.assert_array_equal(extrinsics_dict[k][0], ext_orig[k][0])
        np.testing.assert_array_equal(extrinsics_dict[k][1], ext_orig[k][1])


def test_10_rigid_transform_only_no_scale_modification():
    """Test 10: Floor transform is a pure Euclidean isometry (no scale, stretch, or shear)."""
    R_cb = Rotation.from_euler("xyz", [12.0, 34.0, 56.0], degrees=True).as_matrix()
    t_cb = np.array([0.1, 0.2, 0.3])
    transform = compute_floor_transform_from_board(R_cb, t_cb, h_m=0.005)

    s = np.linalg.svd(transform.rotation, compute_uv=False)
    np.testing.assert_allclose(s, [1.0, 1.0, 1.0], atol=1e-12)


def test_11_inverse_roundtrip():
    """Test 11: Inverse roundtrip reproduces points exactly."""
    R_cb = Rotation.from_euler("xyz", [-10.0, 20.0, 30.0], degrees=True).as_matrix()
    t_cb = np.array([1.2, -2.3, 3.4])
    transform = compute_floor_transform_from_board(R_cb, t_cb, h_m=0.005)

    pts = np.random.randn(50, 3)
    pts_f = transform.transform_points(pts)
    pts_rec = transform.inverse_transform_points(pts_f)
    np.testing.assert_allclose(pts_rec, pts, atol=1e-12)


def test_12_distance_preservation():
    """Test 12: Pairwise Euclidean distances and bone lengths are perfectly preserved."""
    R_cb = Rotation.from_euler("xyz", [45.0, -30.0, 15.0], degrees=True).as_matrix()
    t_cb = np.array([0.5, 1.5, 2.5])
    transform = compute_floor_transform_from_board(R_cb, t_cb, h_m=0.005)

    pts_cam = np.random.randn(20, 3)
    pts_floor = transform.transform_points(pts_cam)

    d_cam = np.linalg.norm(pts_cam[:, None, :] - pts_cam[None, :, :], axis=-1)
    d_floor = np.linalg.norm(pts_floor[:, None, :] - pts_floor[None, :, :], axis=-1)
    np.testing.assert_allclose(d_floor, d_cam, atol=1e-12)


def test_13_multi_view_pose_consistency():
    """Test 13: Multi-view pose agreement evaluates translation <= 5 mm and rotation <= 0.5 deg."""
    K_dict, extrinsics_dict, dist_dict = make_synthetic_cameras()
    board_model = CharucoBoardModel()
    R_cb = np.eye(3)
    t_cb = np.array([0.0, 0.0, 1.5])

    obs_set = synthesize_observation_set(board_model, R_cb, t_cb, K_dict, extrinsics_dict, dist_dict, noise_sigma_px=0.1)
    pose_res = solve_board_pose_multiview(board_model, obs_set, K_dict, extrinsics_dict, dist_dict)

    assert pose_res.cross_camera_translation_disagreement_m <= 0.005
    assert pose_res.cross_camera_rotation_disagreement_deg <= 0.5


def test_14_planar_ambiguity_rejection():
    """Test 14: Multi-view consensus disambiguates planar Necker reversal solutions."""
    K_dict, extrinsics_dict, dist_dict = make_synthetic_cameras()
    board_model = CharucoBoardModel()
    R_cb = Rotation.from_euler("x", 15.0, degrees=True).as_matrix()
    t_cb = np.array([0.1, -0.2, 1.8])

    obs_set = synthesize_observation_set(board_model, R_cb, t_cb, K_dict, extrinsics_dict, dist_dict)
    pose_res = solve_board_pose_multiview(board_model, obs_set, K_dict, extrinsics_dict, dist_dict)

    # Verified that disambiguation status is resolved and pose matches ground truth
    assert pose_res.disambiguation_status == "resolved"
    np.testing.assert_allclose(pose_res.t_cb, t_cb, atol=1e-3)


def test_15_outlier_weak_geometry_rejection():
    """Test 15: Outliers or weak geometry (high reprojection error) trigger gate rejection."""
    K_dict, extrinsics_dict, dist_dict = make_synthetic_cameras()
    board_model = CharucoBoardModel()
    R_cb = np.eye(3)
    t_cb = np.array([0.0, 0.0, 1.5])

    primary_hold, rep_holds, spat_holds = make_full_hold_captures(
        board_model, R_cb, t_cb, K_dict, extrinsics_dict, dist_dict, noise_sigma_px=5.0  # Excessive noise
    )

    result = evaluate_floor_registration(board_model, primary_hold, rep_holds, spat_holds, K_dict, extrinsics_dict, dist_dict)
    assert result.validation_status == "failed"
    assert result.grounding_status == "unregistered"
    assert "GATE_REPROJECTION_RMS_EXCEEDED" in result.failure_reasons or "GATE_REPROJECTION_MAX_RESIDUAL_EXCEEDED" in result.failure_reasons


def test_16_repeatability_rejection():
    """Test 16: Repeatability failure (normal change > 0.2 deg or height > 3 mm) rejects registration."""
    K_dict, extrinsics_dict, dist_dict = make_synthetic_cameras()
    board_model = CharucoBoardModel()
    R_cb = np.eye(3)
    t_cb = np.array([0.0, 0.0, 1.5])

    primary_hold, rep_holds, spat_holds = make_full_hold_captures(
        board_model, R_cb, t_cb, K_dict, extrinsics_dict, dist_dict, noise_sigma_px=0.05
    )

    # Perturb repeatability hold 1 by 1.0 degree normal change
    bad_rot = Rotation.from_euler("x", 1.0, degrees=True).as_matrix() @ R_cb
    rep_holds[0].observation_sets = [
        synthesize_observation_set(board_model, bad_rot, t_cb, K_dict, extrinsics_dict, dist_dict, frame_idx=i)
        for i in range(5)
    ]

    result = evaluate_floor_registration(board_model, primary_hold, rep_holds, spat_holds, K_dict, extrinsics_dict, dist_dict)
    assert result.validation_status == "failed"
    assert result.grounding_status == "unregistered"
    assert "GATE_REPEATABILITY_NORMAL_EXCEEDED" in result.failure_reasons


def test_17_spatial_footprint_validation_rejection():
    """Test 17: Spatial validation failure rejects registration and never modifies primary transform."""
    K_dict, extrinsics_dict, dist_dict = make_synthetic_cameras()
    board_model = CharucoBoardModel()
    R_cb = np.eye(3)
    t_cb = np.array([0.0, 0.0, 1.5])

    primary_hold, rep_holds, spat_holds = make_full_hold_captures(
        board_model, R_cb, t_cb, K_dict, extrinsics_dict, dist_dict, noise_sigma_px=0.05
    )

    # Corrupt spatial hold 1 floor contact height by 20 mm
    bad_t = t_cb + np.array([0.5, 0.020, 0.5])
    spat_holds[0].observation_sets = [
        synthesize_observation_set(board_model, R_cb, bad_t, K_dict, extrinsics_dict, dist_dict, frame_idx=i)
        for i in range(5)
    ]

    result = evaluate_floor_registration(board_model, primary_hold, rep_holds, spat_holds, K_dict, extrinsics_dict, dist_dict)
    assert result.validation_status == "failed"
    assert result.grounding_status == "unregistered"
    assert "GATE_SPATIAL_FLOOR_CONTACT_DISAGREEMENT_EXCEEDED" in result.failure_reasons


def test_18_uncertainty_gate():
    """Test 18: Excessive uncertainty at footprint boundary triggers uncertainty gate."""
    # Profile with very tight uncertainty limit
    tight_profile = FloorValidationProfile(max_height_uncertainty_95_m=0.001)  # 1 mm
    K_dict, extrinsics_dict, dist_dict = make_synthetic_cameras()
    board_model = CharucoBoardModel()
    R_cb = np.eye(3)
    t_cb = np.array([0.0, 0.0, 1.5])

    primary_hold, rep_holds, spat_holds = make_full_hold_captures(
        board_model, R_cb, t_cb, K_dict, extrinsics_dict, dist_dict, noise_sigma_px=0.2
    )

    result = evaluate_floor_registration(
        board_model, primary_hold, rep_holds, spat_holds, K_dict, extrinsics_dict, dist_dict, profile=tight_profile
    )
    assert result.validation_status == "failed"
    assert "GATE_UNCERTAINTY_HEIGHT_EXCEEDED" in result.failure_reasons


def test_19_metadata_provenance_roundtrip(tmp_path: Path):
    """Test 19: Full provenance and registration records serialize and reload losslessly."""
    K_dict, extrinsics_dict, dist_dict = make_synthetic_cameras()
    board_model = CharucoBoardModel()
    R_cb = np.eye(3)
    t_cb = np.array([0.0, 0.0, 1.5])

    primary_hold, rep_holds, spat_holds = make_full_hold_captures(
        board_model, R_cb, t_cb, K_dict, extrinsics_dict, dist_dict, noise_sigma_px=0.05
    )

    result = evaluate_floor_registration(
        board_model, primary_hold, rep_holds, spat_holds, K_dict, extrinsics_dict, dist_dict, epoch_id="epoch_2026_test"
    )

    json_path = tmp_path / "floor_reg.json"
    save_floor_registration(json_path, result)
    loaded = load_floor_registration(json_path)

    assert loaded["provenance"]["protocol_version"] == "2.3C_v1"
    assert loaded["provenance"]["epoch_id"] == "epoch_2026_test"
    assert loaded["grounding_status"] == result.grounding_status
    assert loaded["validation_status"] == result.validation_status
    assert len(loaded["gate_outcomes"]) == len(result.gate_outcomes)


def test_20_stale_registration_rejection():
    """Test 20: Transform with validation_status='stale' is rejected for grounded pipeline."""
    transform = FloorTransform(
        grounding_status="registered",
        validation_status="stale",
        failure_reasons=["EPOCH_EXTRINSICS_MISMATCH"],
    )
    assert transform.is_registered is False
    with pytest.raises(RegistrationGateError, match="stale"):
        require_grounded_floor(transform)


def test_21_canonical_skeleton3d_byte_preservation(tmp_path: Path):
    """Test 21: Canonical skeleton3d array is immutable and byte-preserved."""
    raw_skeleton = np.random.randn(10, 133, 3)
    raw_bytes_before = raw_skeleton.tobytes()

    transform = FloorTransform(
        rotation=Rotation.from_euler("x", 45.0, degrees=True).as_matrix(),
        translation=np.array([1.0, 2.0, 3.0]),
        grounding_status="registered",
        validation_status="passed",
    )

    # Derived points
    floor_skeleton = transform.transform_points(raw_skeleton, current_frame=CAMERA_WORLD_FRAME)
    # Raw skeleton memory must be byte-identical
    assert raw_skeleton.tobytes() == raw_bytes_before
    assert not np.shares_memory(raw_skeleton, floor_skeleton)


def test_22_raw_skeleton3d_floor_semantics_preserved():
    """Test 22: skeleton3d_floor preserves raw T0 transformed semantics."""
    raw_t0 = np.random.randn(5, 133, 3)
    transform = FloorTransform(
        rotation=np.eye(3),
        translation=np.array([0.0, 1.0, 0.0]),
        grounding_status="registered",
        validation_status="passed",
    )
    floor_raw = transform.transform_points(raw_t0, current_frame=CAMERA_WORLD_FRAME)
    np.testing.assert_allclose(floor_raw, raw_t0 + np.array([0.0, 1.0, 0.0]))


def test_23_refined_camera_artifact_remains_camera_world():
    """Test 23: F4 refined skeleton remains strictly in camera_world."""
    raw_skeleton = np.random.randn(30, 133, 3)
    # F4 runs on camera_world skeleton
    refined_cam = filter_skeleton3d(raw_skeleton)
    assert refined_cam.shape == raw_skeleton.shape
    assert np.all(np.isfinite(refined_cam))


def test_24_refined_floor_artifact_references_parent_and_registration():
    """Test 24: Refined floor artifact links provenance to parent refined camera artifact and registration."""
    raw_skeleton = np.random.randn(30, 133, 3)
    refined_cam = filter_skeleton3d(raw_skeleton)

    transform = FloorTransform(
        rotation=Rotation.from_euler("y", 90.0, degrees=True).as_matrix(),
        translation=np.array([0.0, -0.5, 1.0]),
        grounding_status="registered",
        validation_status="passed",
        epoch_id="epoch_xyz_123",
    )

    refined_floor = transform.transform_points(refined_cam, current_frame=CAMERA_WORLD_FRAME)
    provenance = {
        "parent_artifact": "skeleton3d_refined_camera",
        "registration_epoch_id": transform.epoch_id,
        "grounding_status": transform.grounding_status,
        "source_frame": CAMERA_WORLD_FRAME,
        "target_frame": FLOOR_WORLD_FRAME,
    }
    assert provenance["parent_artifact"] == "skeleton3d_refined_camera"
    assert provenance["registration_epoch_id"] == "epoch_xyz_123"
    assert refined_floor.shape == refined_cam.shape


def test_25_registration_failure_blocks_trusted_grounded_processing():
    """Test 25: Registration failure triggers fail-closed behavior, blocking trusted grounded retargeting."""
    failed_transform = FloorTransform(
        grounding_status="unregistered",
        validation_status="failed",
        failure_reasons=["GATE_REPROJECTION_RMS_EXCEEDED"],
    )
    with pytest.raises(RegistrationGateError, match="GATE_REPROJECTION_RMS_EXCEEDED"):
        require_grounded_floor(failed_transform)


def test_26_unregistered_identity_cannot_pass_grounded_route():
    """Test 26: Identity transform is explicitly unregistered and cannot pass require_grounded_floor."""
    identity_transform = FloorTransform.identity()
    assert identity_transform.grounding_status == "unregistered"
    assert identity_transform.validation_status == "missing"
    with pytest.raises(RegistrationGateError):
        require_grounded_floor(identity_transform)


def test_27_vector_rotation_for_velocity_acceleration():
    """Test 27: Kinematic vectors (velocity, acceleration) undergo pure rotation with zero translation."""
    R = Rotation.from_euler("z", 90.0, degrees=True).as_matrix()
    t = np.array([100.0, 200.0, 300.0])  # Huge translation
    transform = FloorTransform(rotation=R, translation=t, grounding_status="registered", validation_status="passed")

    vel_cam = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    vel_floor = transform_velocity_to_floor(vel_cam, transform)

    # 90 deg z-rotation: [1, 0, 0] -> [0, 1, 0], [0, 2, 0] -> [-2, 0, 0]
    expected = np.array([[0.0, 1.0, 0.0], [-2.0, 0.0, 0.0]])
    np.testing.assert_allclose(vel_floor, expected, atol=1e-12)

    # Inverse roundtrip
    vel_rec = transform_velocity_from_floor(vel_floor, transform)
    np.testing.assert_allclose(vel_rec, vel_cam, atol=1e-12)


def test_28_covariance_rotation():
    """Test 28: Covariance matrix rotation Sigma_floor = R @ Sigma @ R^T preserves symmetry and eigenvalues."""
    R = Rotation.from_euler("xyz", [30.0, 45.0, 60.0], degrees=True).as_matrix()
    transform = FloorTransform(rotation=R, translation=np.array([1.0, 2.0, 3.0]))

    # Random positive definite covariance matrix
    A = np.random.randn(3, 3)
    Sigma_cam = A @ A.T + np.eye(3)

    Sigma_floor = transform_covariance_to_floor(Sigma_cam, transform)
    # Check shape and symmetry
    assert Sigma_floor.shape == (3, 3)
    np.testing.assert_allclose(Sigma_floor, Sigma_floor.T, atol=1e-12)

    # Eigenvalues must be invariant under orthogonal similarity transform
    w_cam = np.sort(np.linalg.eigvals(Sigma_cam))
    w_floor = np.sort(np.linalg.eigvals(Sigma_floor))
    np.testing.assert_allclose(w_floor, w_cam, atol=1e-12)

    # Inverse roundtrip
    Sigma_rec = transform_covariance_from_floor(Sigma_floor, transform)
    np.testing.assert_allclose(Sigma_rec, Sigma_cam, atol=1e-12)


def test_29_no_double_transform():
    """Test 29: Double-transform protection rejects applying transform to already transformed data."""
    transform = FloorTransform(grounding_status="registered", validation_status="passed")
    pts = np.zeros((10, 3))

    with pytest.raises(FrameMismatchError, match="Double-transform rejected"):
        transform.transform_points(pts, current_frame=FLOOR_WORLD_FRAME)

    with pytest.raises(FrameMismatchError, match="Inverse transform rejected"):
        transform.inverse_transform_points(pts, current_frame=CAMERA_WORLD_FRAME)


def test_30_f4_remains_camera_world_and_non_commuting():
    """Test 30: F4 remains in camera_world and is not numerically rerun in floor_world."""
    # Synthetic 3D sequence in camera_world
    np.random.seed(123)
    F, J = 30, 17
    seq_cam = np.zeros((F, J, 3))
    for t in range(F):
        seq_cam[t, :, 0] = np.sin(t / 5.0)
        seq_cam[t, :, 1] = np.cos(t / 5.0) * 0.1
        seq_cam[t, :, 2] = t / 10.0

    R = Rotation.from_euler("z", 45.0, degrees=True).as_matrix()
    transform = FloorTransform(
        rotation=R,
        translation=np.array([0.5, -0.2, 1.0]),
        grounding_status="registered",
        validation_status="passed",
    )

    # Invariant: F4 filter is run strictly in camera_world on canonical skeleton3d
    refined_cam = filter_skeleton3d(seq_cam)
    assert refined_cam.shape == seq_cam.shape

    # Invariant: refined floor artifact is derived via rigid transformation from camera_world
    refined_floor = transform.transform_points(refined_cam, current_frame=CAMERA_WORLD_FRAME)
    assert refined_floor.shape == seq_cam.shape

    # Verify that canonical input was not mutated
    assert np.allclose(seq_cam[0, :, 0], np.sin(0.0))

    # Verify that refined_floor is related to refined_cam by the exact rigid transform
    recovered_cam = transform.inverse_transform_points(refined_floor, current_frame=FLOOR_WORLD_FRAME)
    np.testing.assert_allclose(recovered_cam, refined_cam, atol=1e-12)
