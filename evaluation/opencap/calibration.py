"""Checkerboard extrinsics with the shipped intrinsics.

The OpenCap archive ships one intrinsic and one extrinsic pose per camera.
This module re-solves the extrinsics from the calibration video of each
camera, keeps the shipped intrinsics and distortion, and compares the result
with the shipped pose.

Method
------
* The board is 11 x 8 inner corners with 60 mm squares.
* The frame is sampled at ``vidLength - 0.3 s`` and stepped back on failure.
* Corners are detected on the frame upscaled by 2x (then 4x, then native),
  refined with ``cornerSubPix``; ``findChessboardCornersSB`` is the fallback,
  with its corner order reversed.  Corners are divided back by the scale, so
  the pose is solved in native pixels.
* ``solvePnPGeneric`` with ``SOLVEPNP_IPPE`` (solution 0) gives the pose.

The output ``calibration.json`` uses the schema that ``aimocap triangulate``
reads.  ``calibration_check.json`` holds the comparison with the shipped pose
and the acceptance flags.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from aimocap.calib.io import save_calibration
from aimocap.math.coords import OPENCV_COORDINATE_SYSTEM
from evaluation.opencap.dataset import BOARD_INNER_CORNERS, BOARD_SQUARE_MM

ACCEPT_MAX_DR_DEG = 0.35
ACCEPT_MAX_CENTRE_MM = 30.0
UPSCALES: tuple[float, ...] = (2.0, 4.0, 1.0)


@dataclass(frozen=True)
class CameraPose:
    """Checkerboard pose of one camera and its comparison with the shipped pose."""

    camera: str
    K: np.ndarray
    dist: np.ndarray
    R: np.ndarray
    t: np.ndarray
    R_shipped: np.ndarray
    t_shipped: np.ndarray
    image_size: tuple[int, int]
    frame: int
    frames_total: int
    method: str
    reproj_rms_px: float
    dR_deg: float
    centre_diff_mm: float
    dt_mm: float

    def summary(self) -> dict[str, Any]:
        """JSON-safe summary without the matrices."""
        return {
            "frame": self.frame,
            "frames_total": self.frames_total,
            "method": self.method,
            "reproj_rms_px": round(self.reproj_rms_px, 4),
            "dR_deg": round(self.dR_deg, 4),
            "centre_diff_mm": round(self.centre_diff_mm, 3),
            "dt_mm": round(self.dt_mm, 3),
        }


def board_object_points(
    inner: tuple[int, int] = BOARD_INNER_CORNERS, square_mm: float = BOARD_SQUARE_MM
) -> np.ndarray:
    """Board corner coordinates in millimetres, row-major as OpenCV reports them."""
    cols, rows = inner
    obj = np.zeros((cols * rows, 3), np.float32)
    obj[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_mm
    return obj


def rotation_angle_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    """Angle of the relative rotation between two rotation matrices, in degrees."""
    return float(np.degrees(np.arccos(np.clip((np.trace(Ra @ Rb.T) - 1.0) / 2.0, -1.0, 1.0))))


def detect_board(gray: np.ndarray) -> tuple[np.ndarray | None, str]:
    """Detect the inner corners; return ``(corners (N, 2), method)`` or ``(None, "none")``."""
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    for scale in UPSCALES:
        g = gray if scale == 1.0 else cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        ok, corners = cv2.findChessboardCorners(
            g, BOARD_INNER_CORNERS, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        if ok:
            corners = cv2.cornerSubPix(g, corners, (11, 11), (-1, -1), criteria)
            return corners.reshape(-1, 2).astype(np.float64) / scale, f"classic_x{scale:g}"
    for scale in UPSCALES:
        g = gray if scale == 1.0 else cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        ok, corners = cv2.findChessboardCornersSB(g, BOARD_INNER_CORNERS)
        if ok:
            reversed_corners = corners.reshape(-1, 2).astype(np.float64)[::-1] / scale
            return reversed_corners.copy(), f"sb_x{scale:g}_reversed"
    return None, "none"


def frame_candidates(n_frames: int, fps: float) -> list[int]:
    """Frames to try, in order: ``vidLength - 0.3 s``, then steps back, then spread samples."""
    rate = max(fps, 1.0)
    base = int(round(n_frames - 0.3 * rate))
    candidates = [base]
    step = max(1, int(round(0.2 * rate)))
    candidates += [base - k * step for k in range(1, 12)]
    candidates += [n_frames - 5, n_frames // 2, n_frames // 3]
    return [int(np.clip(c, 0, n_frames - 1)) for c in candidates]


def estimate_board_pose(
    corners: np.ndarray, K: np.ndarray, dist: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """Pose (R, t) of the board from its detected corners, and the RMS reprojection error in px."""
    obj = board_object_points()
    ret = cv2.solvePnPGeneric(obj, corners.astype(np.float32), K, dist, flags=cv2.SOLVEPNP_IPPE)
    rvec = ret[1][0]
    tvec = np.asarray(ret[2][0], dtype=np.float64).reshape(3, 1)
    R, _ = cv2.Rodrigues(rvec)
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    rms = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - corners) ** 2, axis=1))))
    return R, tvec, rms


def _read_shipped(trial_dir: Path, cam: str) -> dict[str, Any]:
    with (trial_dir / "intrinsics" / f"{cam}.pickle").open("rb") as handle:
        return pickle.load(handle)


def solve_camera(trial_dir: str | Path, cam: str) -> CameraPose:
    """Solve the checkerboard pose of one camera from its extrinsics video."""
    trial_dir = Path(trial_dir)
    shipped = _read_shipped(trial_dir, cam)
    K = np.asarray(shipped["intrinsicMat"], dtype=np.float64)
    dist = np.asarray(shipped["distortion"], dtype=np.float64).reshape(-1)
    R_shipped = np.asarray(shipped["rotation"], dtype=np.float64)
    t_shipped = np.asarray(shipped["translation"], dtype=np.float64).reshape(3, 1)

    cap = cv2.VideoCapture(str(trial_dir / "extrinsics" / f"{cam}.avi"))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 60.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    corners: np.ndarray | None = None
    method = "none"
    used = -1
    for idx in frame_candidates(n_frames, fps):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        corners, method = detect_board(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        if corners is not None:
            used = idx
            break
    cap.release()
    if corners is None:
        raise RuntimeError(f"checkerboard not detected for {cam} in {trial_dir}")

    R, t, rms = estimate_board_pose(corners, K, dist)
    centre = (-R.T @ t).ravel()
    centre_shipped = (-R_shipped.T @ t_shipped).ravel()
    return CameraPose(
        camera=cam,
        K=K,
        dist=dist,
        R=R,
        t=t,
        R_shipped=R_shipped,
        t_shipped=t_shipped,
        image_size=(width, height),
        frame=used,
        frames_total=n_frames,
        method=method,
        reproj_rms_px=rms,
        dR_deg=rotation_angle_deg(R, R_shipped),
        centre_diff_mm=float(np.linalg.norm(centre - centre_shipped)),
        dt_mm=float(np.linalg.norm(t.ravel() - t_shipped.ravel())),
    )


def calibrate_trial(
    trial_dir: str | Path,
    cams: list[str],
    out_json: str | Path,
    check_json: str | Path,
) -> dict[str, Any]:
    """Solve every camera, write the calibration and the comparison, return the summary."""
    poses = [solve_camera(trial_dir, cam) for cam in cams]
    save_calibration(
        out_json,
        [p.K for p in poses],
        [(p.R, p.t) for p in poses],
        image_sizes=[p.image_size for p in poses],
        dist_coeffs=[p.dist for p in poses],
        camera_names=[p.camera for p in poses],
        units="mm",
        calibration_source="checkerboard",
        accuracy_class="production",
        scale_source="checkerboard",
        projection_convention="world_to_camera",
        coordinate_system=OPENCV_COORDINATE_SYSTEM,
        distortion_model="opencv_radtan",
        distortion_treatment="distorted_pixels",
    )
    max_dr = max(p.dR_deg for p in poses)
    max_centre = max(p.centre_diff_mm for p in poses)
    check = {
        "intrinsics": "shipped (iPhone calibration pickle, shared by the cameras)",
        "board": {"inner_corners": list(BOARD_INNER_CORNERS), "square_mm": BOARD_SQUARE_MM},
        "solver": "SOLVEPNP_IPPE solution 0",
        "cameras": {p.camera: p.summary() for p in poses},
        "max_dR_deg": round(max_dr, 4),
        "max_centre_diff_mm": round(max_centre, 3),
        "acceptance": {"max_dR_deg": ACCEPT_MAX_DR_DEG, "max_centre_diff_mm": ACCEPT_MAX_CENTRE_MM},
        "pass": bool(max_dr <= ACCEPT_MAX_DR_DEG and max_centre <= ACCEPT_MAX_CENTRE_MM),
    }
    Path(check_json).write_text(json.dumps(check, indent=2), encoding="utf-8")
    return check
