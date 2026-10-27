"""Regression tests for explicit world/camera geometry contracts."""

from __future__ import annotations

import cv2
import numpy as np

from aimocap.calib.io import load_calibration_bundle, save_calibration
from aimocap.math.coords import (
    DisplayTransform,
    camera_extrinsics_to_display,
    camera_center,
    inverse_transform_world_points,
    transform_world_points,
)


def test_display_world_transform_preserves_projection(tmp_path):
    angle = np.deg2rad(23.0)
    R_display = np.array(
        [[np.cos(angle), 0.0, np.sin(angle)], [0.0, 1.0, 0.0], [-np.sin(angle), 0.0, np.cos(angle)]]
    )
    display = DisplayTransform(R_display, np.array([0.4, -0.2, 0.7]))
    R = np.eye(3)
    t = np.array([[0.1], [-0.3], [2.0]])
    points = np.array([[0.2, -0.1, 4.0], [-0.5, 0.4, 3.0]])

    R_out, t_out = camera_extrinsics_to_display(R, t, display)
    old_camera_points = (R @ points.T + t).T
    display_points = transform_world_points(points, display)
    new_camera_points = (R_out @ display_points.T + t_out).T

    np.testing.assert_allclose(new_camera_points, old_camera_points, atol=1e-10)
    np.testing.assert_allclose(inverse_transform_world_points(display_points, display), points, atol=1e-10)
    expected_center_display = transform_world_points(camera_center(R, t), display)
    np.testing.assert_allclose(expected_center_display, -R_out.T @ t_out.reshape(3), atol=1e-10)


def test_calibration_bundle_roundtrip_includes_distortion_and_aliases(tmp_path):
    K = np.array([[1000.0, 0.0, 640.0], [0.0, 1000.0, 360.0], [0.0, 0.0, 1.0]])
    R = np.eye(3)
    t = np.array([[0.0], [0.0], [0.0]])
    dist = np.array([0.12, -0.03, 0.001, -0.002, 0.0])
    path = tmp_path / "calibration.json"

    save_calibration(
        path,
        [K],
        [(R, t)],
        image_sizes=[(1280, 720)],
        dist_coeffs=[dist],
        camera_names=["00_00"],
        units="meters",
        calibration_source="checkerboard",
        accuracy_class="production",
    )

    bundle = load_calibration_bundle(path, camera_names=["hd_00_00.mp4"], require_metadata=True)
    np.testing.assert_allclose(bundle.K_list[0], K)
    np.testing.assert_allclose(bundle.extrinsics[0][1], t)
    np.testing.assert_allclose(bundle.dist_coeffs[0], dist)
    assert bundle.camera_names == ["00_00"]
    assert bundle.metadata["projection_convention"] == "world_to_camera"
    assert bundle.metadata["units"] == "meters"
    assert bundle.metadata["accuracy_class"] == "production"


def test_camera_model_distortion_is_used_for_residuals():
    from aimocap.triangulate.config import GeometryConfig
    from aimocap.triangulate.engine import triangulate_sequence_with_diagnostics

    K = np.array([[900.0, 0.0, 640.0], [0.0, 900.0, 360.0], [0.0, 0.0, 1.0]])
    dist = np.array([0.16, -0.04, 0.001, -0.001, 0.0])
    extrinsics = [
        (np.eye(3), np.zeros((3, 1))),
        (np.eye(3), np.array([[-0.8], [0.0], [0.0]])),
        (np.eye(3), np.array([[0.8], [0.0], [0.0]])),
    ]
    point_cv = np.array([[0.2, -0.1, 4.0]], dtype=np.float64)
    keypoints = np.zeros((1, 3, 1, 2), dtype=np.float64)
    for camera_index, (R, t) in enumerate(extrinsics):
        rvec, _ = cv2.Rodrigues(R)
        projected, _ = cv2.projectPoints(point_cv, rvec, t, K, dist)
        keypoints[0, camera_index, 0] = projected.reshape(2)
    scores = np.ones((1, 3, 1), dtype=np.float64)

    diag = triangulate_sequence_with_diagnostics(
        keypoints,
        scores,
        [K, K, K],
        extrinsics,
        dist_coeffs=[dist, dist, dist],
        geometry_config=GeometryConfig(
            max_reprojection_error_px=2.0,
            hard_reprojection_error_px=5.0,
            min_ray_angle_deg=0.0,
            min_depth_m=0.01,
            f_scale=3.0,
        ),
    )

    expected_internal = np.array([0.2, 0.1, -4.0])
    np.testing.assert_allclose(diag.points3d[0, 0], expected_internal, atol=1e-4)
    np.testing.assert_allclose(diag.candidate_points3d[0, 0], expected_internal, atol=1e-4)
    assert diag.candidate_num_inliers[0, 0] == 3
    assert diag.num_inliers[0, 0] == 3
    assert np.nanmax(diag.reprojection_error_px[0, 0]) < 1e-3


def test_shared_calibrated_triangulation_service_uses_bundle_contract():
    from aimocap.calib.io import CalibrationBundle
    from aimocap.triangulate.config import GeometryConfig
    from aimocap.triangulate.service import triangulate_calibrated

    K = np.array([[900.0, 0.0, 640.0], [0.0, 900.0, 360.0], [0.0, 0.0, 1.0]])
    extrinsics = [
        (np.eye(3), np.zeros((3, 1))),
        (np.eye(3), np.array([[-0.8], [0.0], [0.0]])),
    ]
    point_cv = np.array([[0.2, -0.1, 4.0]], dtype=np.float64)
    keypoints = np.zeros((1, 2, 1, 2), dtype=np.float64)
    for camera_index, (R, t) in enumerate(extrinsics):
        rvec, _ = cv2.Rodrigues(R)
        projected, _ = cv2.projectPoints(point_cv, rvec, t, K, np.zeros(5))
        keypoints[0, camera_index, 0] = projected.reshape(2)
    scores = np.ones((1, 2, 1), dtype=np.float64)
    bundle = CalibrationBundle(
        [K, K],
        extrinsics,
        [np.zeros(5), np.zeros(5)],
        {
            "camera_names": ["00_00", "00_01"],
            "units": "meters",
            "geometry_units": "meters",
            "projection_convention": "world_to_camera",
            "camera_coordinate_system": "opencv_right_handed_y_down_z_forward",
            "distortion_treatment": "distorted_pixels",
        },
    )

    diagnostics = triangulate_calibrated(
        keypoints,
        scores,
        bundle,
        geometry_config=GeometryConfig(
            max_reprojection_error_px=2.0,
            hard_reprojection_error_px=5.0,
            min_ray_angle_deg=0.0,
            min_depth_m=0.01,
        ),
    )

    assert diagnostics.num_inliers[0, 0] == 2
    assert np.nanmax(diagnostics.reprojection_error_px[0, 0]) < 1e-3


def test_pairwise_3d_disagreement_is_small_for_consistent_views():
    from aimocap.calib.io import CalibrationBundle
    from aimocap.triangulate.service import pairwise_3d_disagreement

    K = np.array([[900.0, 0.0, 640.0], [0.0, 900.0, 360.0], [0.0, 0.0, 1.0]])
    extrinsics = [
        (np.eye(3), np.zeros((3, 1))),
        (np.eye(3), np.array([[-0.8], [0.0], [0.0]])),
        (np.eye(3), np.array([[0.8], [0.0], [0.0]])),
    ]
    point_cv = np.array([[0.2, -0.1, 4.0]], dtype=np.float64)
    keypoints = np.zeros((1, 3, 1, 2), dtype=np.float64)
    for camera_index, (R, t) in enumerate(extrinsics):
        rvec, _ = cv2.Rodrigues(R)
        projected, _ = cv2.projectPoints(point_cv, rvec, t, K, np.zeros(5))
        keypoints[0, camera_index, 0] = projected.reshape(2)
    scores = np.ones((1, 3, 1), dtype=np.float64)
    bundle = CalibrationBundle(
        [K, K, K],
        extrinsics,
        [np.zeros(5), np.zeros(5), np.zeros(5)],
        {
            "camera_names": ["00_00", "00_01", "00_02"],
            "units": "meters",
            "geometry_units": "meters",
            "projection_convention": "world_to_camera",
            "camera_coordinate_system": "opencv_right_handed_y_down_z_forward",
            "distortion_treatment": "undistorted_pixels",
        },
    )

    disagreement = pairwise_3d_disagreement(keypoints, scores, bundle, min_conf=0.5)

    np.testing.assert_allclose(disagreement[0, 0], 0.0, atol=1e-8)


def test_calibration_mapping_and_intrinsics_scale_are_explicit(tmp_path):
    path = tmp_path / "calibration.json"
    K = np.array([[1000.0, 0.0, 640.0], [0.0, 900.0, 360.0], [0.0, 0.0, 1.0]])
    save_calibration(
        path,
        [K],
        [(np.eye(3), np.zeros((3, 1)))],
        image_sizes=[(1280, 720)],
        camera_names=["00_00"],
        units="meters",
        calibration_source="checkerboard",
        accuracy_class="production",
    )
    bundle = load_calibration_bundle(
        path,
        camera_names=["capture_hd_00_00.mp4"],
        camera_map={"capture_hd_00_00.mp4": "00_00"},
        observation_image_sizes=[(640, 360)],
        require_metadata=True,
    )
    np.testing.assert_allclose(
        bundle.K_list[0],
        [[500.0, 0.0, 320.0], [0.0, 450.0, 180.0], [0.0, 0.0, 1.0]],
    )
    assert bundle.metadata["resolved_camera_map"] == {"capture_hd_00_00.mp4": "00_00"}


def test_calibration_mapping_rejects_ambiguous_or_duplicate_names(tmp_path):
    path = tmp_path / "calibration.json"
    save_calibration(
        path,
        [np.eye(3), np.eye(3)],
        [(np.eye(3), np.zeros((3, 1))), (np.eye(3), np.zeros((3, 1)))],
        image_sizes=[(10, 10), (10, 10)],
        camera_names=["cam_a", "cam_b"],
        units="meters",
        calibration_source="checkerboard",
        accuracy_class="production",
    )
    import pytest
    with pytest.raises(ValueError, match="one-to-one"):
        load_calibration_bundle(
            path,
            camera_names=["a", "b"],
            camera_map={"a": "cam_a", "b": "cam_a"},
            require_metadata=True,
        )


def test_calibration_bundle_normalizes_centimetres_to_meters(tmp_path):
    path = tmp_path / "centimetre_calibration.json"
    save_calibration(
        path,
        [np.eye(3)],
        [(np.eye(3), np.array([[0.0], [0.0], [250.0]]))],
        image_sizes=[(1920, 1080)],
        camera_names=["00_00"],
        units="cm",
        calibration_source="panoptic",
        accuracy_class="production",
    )

    bundle = load_calibration_bundle(path, require_metadata=True)

    np.testing.assert_allclose(bundle.extrinsics[0][1], [[0.0], [0.0], [2.5]])
    assert bundle.metadata["source_units"] == "cm"
    assert bundle.metadata["units"] == "meters"
    assert bundle.metadata["geometry_units"] == "meters"
