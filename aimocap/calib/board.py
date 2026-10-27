"""Checkerboard calibration for real camera intrinsics and extrinsics."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def _image_paths(directory: str | Path) -> list[Path]:
    path = Path(directory)
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    paths = [p for pattern in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff") for p in path.glob(pattern)]
    return sorted(set(paths))


def _find_corners(image: np.ndarray, board_size: tuple[int, int]) -> np.ndarray | None:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    try:
        found, corners = cv2.findChessboardCornersSB(
            gray,
            board_size,
            flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY,
        )
    except AttributeError:
        found, corners = cv2.findChessboardCorners(
            gray,
            board_size,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
        if found:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-3)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return corners.reshape(-1, 2).astype(np.float32) if found else None


def calibrate_checkerboard(
    image_dirs: list[str | Path],
    *,
    board_size: tuple[int, int] = (9, 6),
    square_size: float = 0.025,
    camera_names: list[str] | None = None,
    min_frames: int = 8,
) -> dict:
    """Calibrate a synchronized checkerboard capture.

    Each directory contains one camera's board images. Matching filenames are
    treated as simultaneous board poses. Camera 0 is the world reference and
    the returned extrinsics satisfy ``X_camera = R @ X_world + t``.
    """
    if len(image_dirs) < 2:
        raise ValueError("at least two camera image directories are required")
    if square_size <= 0:
        raise ValueError("square_size must be positive")
    if min_frames < 3:
        raise ValueError("min_frames must be at least 3")
    if camera_names is None:
        camera_names = [Path(directory).stem for directory in image_dirs]
    if len(camera_names) != len(image_dirs):
        raise ValueError("camera_names length must match image_dirs")

    paths_per_camera = [_image_paths(directory) for directory in image_dirs]
    by_name = [{path.name: path for path in paths} for paths in paths_per_camera]
    frame_names = sorted(set.intersection(*(set(items) for items in by_name)))
    if len(frame_names) < min_frames:
        # A sorted positional fallback is useful for image exports that rename
        # every frame, but is recorded so the caller can audit the pairing.
        count = min(len(paths) for paths in paths_per_camera)
        if count < min_frames:
            raise ValueError(f"only {count} common board frames; need at least {min_frames}")
        frame_names = [f"__position_{i:06d}" for i in range(count)]
        by_name = [{name: paths[i] for i, name in enumerate(frame_names)} for paths in paths_per_camera]

    cols, rows = board_size
    object_template = np.zeros((cols * rows, 3), dtype=np.float32)
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    object_template[:, :2] = grid * float(square_size)

    detections: list[dict[str, np.ndarray]] = [dict() for _ in image_dirs]
    image_sizes: list[tuple[int, int] | None] = [None] * len(image_dirs)
    for ci, mapping in enumerate(by_name):
        for frame_name in frame_names:
            image = cv2.imread(str(mapping[frame_name]), cv2.IMREAD_COLOR)
            if image is None:
                continue
            image_sizes[ci] = (int(image.shape[1]), int(image.shape[0]))
            corners = _find_corners(image, board_size)
            if corners is not None:
                detections[ci][frame_name] = corners

    object_points_by_camera = []
    K_list = []
    dist_coeffs = []
    rms_by_camera = []
    for ci, detected in enumerate(detections):
        if len(detected) < min_frames:
            raise ValueError(
                f"camera {camera_names[ci]} has {len(detected)} detected board frames; "
                f"need at least {min_frames}"
            )
        size = image_sizes[ci]
        assert size is not None
        object_points = [object_template.copy() for _ in detected]
        image_points = [detected[name] for name in detected]
        rms, K, dist, _, _ = cv2.calibrateCamera(object_points, image_points, size, None, None)
        object_points_by_camera.append(object_points)
        K_list.append(np.asarray(K, dtype=np.float64))
        dist_coeffs.append(np.asarray(dist, dtype=np.float64).reshape(-1))
        rms_by_camera.append(float(rms))

    extrinsics: list[tuple[np.ndarray, np.ndarray]] = [(np.eye(3), np.zeros((3, 1)))]
    stereo_rms = []
    for ci in range(1, len(image_dirs)):
        common = [name for name in frame_names if name in detections[0] and name in detections[ci]]
        if len(common) < min_frames:
            raise ValueError(
                f"camera pair 0/{ci} has only {len(common)} common board detections; "
                f"need at least {min_frames}"
            )
        object_points = [object_template.copy() for _ in common]
        points0 = [detections[0][name] for name in common]
        points_i = [detections[ci][name] for name in common]
        size = image_sizes[ci]
        assert size is not None
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-7)
        rms, _, _, _, _, R, t, _, _ = cv2.stereoCalibrate(
            object_points,
            points0,
            points_i,
            K_list[0],
            dist_coeffs[0],
            K_list[ci],
            dist_coeffs[ci],
            size,
            criteria=criteria,
            flags=cv2.CALIB_FIX_INTRINSIC,
        )
        extrinsics.append((np.asarray(R, dtype=np.float64), np.asarray(t, dtype=np.float64).reshape(3, 1)))
        stereo_rms.append(float(rms))

    return {
        "K_list": K_list,
        "extrinsics": extrinsics,
        "dist_coeffs": dist_coeffs,
        "camera_names": list(camera_names),
        "image_sizes": [tuple(size) for size in image_sizes],
        "board_size": list(board_size),
        "square_size": float(square_size),
        "frame_names": frame_names,
        "rms_by_camera": rms_by_camera,
        "stereo_rms": stereo_rms,
    }


__all__ = ["calibrate_checkerboard"]
