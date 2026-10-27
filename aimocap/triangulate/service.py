"""Strict calibrated triangulation authority shared by CLI and diagnostics."""

from __future__ import annotations

import numpy as np
import cv2

from aimocap.calib.io import CalibrationBundle
from aimocap.math.triangulate import triangulate_robust
from aimocap.triangulate.config import GeometryConfig
from aimocap.triangulate.engine import TriangulationDiagnostics, triangulate_sequence_with_diagnostics


def triangulate_calibrated(
    keypoints: np.ndarray,
    scores: np.ndarray,
    bundle: CalibrationBundle,
    *,
    geometry_config: GeometryConfig,
    image_sizes: np.ndarray | None = None,
    allow_exploratory: bool = False,
    fps: float | None = None,
) -> TriangulationDiagnostics:
    """Triangulate observations under one validated camera/geometry contract."""
    keypoints = np.asarray(keypoints, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    if keypoints.ndim != 4 or keypoints.shape[-1] != 2:
        raise ValueError("keypoints must have shape (frames, cameras, joints, 2)")
    if scores.shape != keypoints.shape[:3]:
        raise ValueError("scores must have shape (frames, cameras, joints)")
    if keypoints.shape[1] != len(bundle.K_list):
        raise ValueError("keypoint camera axis does not match calibration bundle")
    if len(bundle.extrinsics) != len(bundle.K_list) or len(bundle.dist_coeffs) != len(bundle.K_list):
        raise ValueError("calibration bundle camera arrays must have equal lengths")

    metadata = bundle.metadata
    if metadata.get("geometry_units") != "meters":
        if not (allow_exploratory and metadata.get("accuracy_class") == "exploratory"):
            raise ValueError("calibrated triangulation requires meter-normalized geometry")
    if metadata.get("projection_convention") != "world_to_camera":
        raise ValueError("calibration projection convention must be world_to_camera")
    if metadata.get("camera_coordinate_system") != "opencv_right_handed_y_down_z_forward":
        raise ValueError("calibration camera coordinate system must be OpenCV")

    distortion_treatment = metadata.get("distortion_treatment")
    if distortion_treatment == "distorted_pixels":
        dist_coeffs = bundle.dist_coeffs
    elif distortion_treatment == "undistorted_pixels":
        dist_coeffs = None
    elif allow_exploratory and distortion_treatment in ("unknown", None):
        dist_coeffs = None
    else:
        raise ValueError("calibration distortion treatment must be explicitly declared")

    return triangulate_sequence_with_diagnostics(
        keypoints,
        scores,
        bundle.K_list,
        bundle.extrinsics,
        min_conf=geometry_config.min_detection_conf,
        reproj_threshold_px=geometry_config.max_reprojection_error_px,
        dist_coeffs=dist_coeffs,
        geometry_config=geometry_config,
        image_sizes=image_sizes,
        fps=fps,
    )


def pairwise_3d_disagreement(
    keypoints: np.ndarray,
    scores: np.ndarray,
    bundle: CalibrationBundle,
    *,
    min_conf: float = 0.5,
    image_sizes: np.ndarray | None = None,
) -> np.ndarray:
    """Return the max distance between valid two-camera reconstructions per point."""
    keypoints = np.asarray(keypoints, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    if keypoints.ndim != 4 or keypoints.shape[-1] != 2:
        raise ValueError("keypoints must have shape (frames, cameras, joints, 2)")
    if scores.shape != keypoints.shape[:3]:
        raise ValueError("scores must have shape (frames, cameras, joints)")
    if keypoints.shape[1] != len(bundle.K_list):
        raise ValueError("keypoint camera axis does not match calibration bundle")
    if bundle.metadata.get("geometry_units") != "meters":
        raise ValueError("pairwise disagreement requires meter-normalized geometry")
    if bundle.metadata.get("projection_convention") != "world_to_camera":
        raise ValueError("calibration projection convention must be world_to_camera")
    if bundle.metadata.get("camera_coordinate_system") != "opencv_right_handed_y_down_z_forward":
        raise ValueError("calibration camera coordinate system must be OpenCV")

    distorted = bundle.metadata.get("distortion_treatment") == "distorted_pixels"
    if bundle.metadata.get("distortion_treatment") not in {"distorted_pixels", "undistorted_pixels"}:
        raise ValueError("calibration distortion treatment must be explicitly declared")
    P_list = [K @ np.hstack(ext) for K, ext in zip(bundle.K_list, bundle.extrinsics)]
    result = np.full(keypoints.shape[:1] + keypoints.shape[2:3], np.nan, dtype=np.float64)
    num_frames, num_cameras, num_joints, _ = keypoints.shape
    for frame in range(num_frames):
        for joint in range(num_joints):
            solutions = []
            for left in range(num_cameras):
                for right in range(left + 1, num_cameras):
                    if scores[frame, left, joint] < min_conf or scores[frame, right, joint] < min_conf:
                        continue
                    observations = keypoints[frame, [left, right], joint]
                    if not np.isfinite(observations).all():
                        continue
                    if image_sizes is not None:
                        valid_image = True
                        for camera, point in zip((left, right), observations):
                            width, height = image_sizes[camera]
                            valid_image &= 0.0 <= point[0] < width and 0.0 <= point[1] < height
                        if not valid_image:
                            continue
                    triangulation_points = observations.copy()
                    if distorted:
                        for local, camera in enumerate((left, right)):
                            triangulation_points[local] = cv2.undistortPoints(
                                triangulation_points[local].reshape(1, 1, 2),
                                bundle.K_list[camera],
                                bundle.dist_coeffs[camera],
                                P=bundle.K_list[camera],
                            ).reshape(2)
                    point = triangulate_robust(
                        triangulation_points,
                        [P_list[left], P_list[right]],
                        scores[frame, [left, right], joint],
                    )
                    if np.isfinite(point).all():
                        solutions.append(point)
            if solutions:
                result[frame, joint] = max(
                    float(np.linalg.norm(left - right))
                    for index, left in enumerate(solutions)
                    for right in solutions[index + 1 :]
                ) if len(solutions) > 1 else 0.0
    return result


__all__ = ["pairwise_3d_disagreement", "triangulate_calibrated"]
