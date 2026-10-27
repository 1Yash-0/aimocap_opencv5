"""Mocap-derived per-camera truth: evaluation only, never a reconstruction input.

For each camera the shipped ``*_syncdWithMocap`` clip gives a coarse start frame
(mean absolute difference of downscaled grayscale frames).  A rigid PnP fit of
the twelve mocap virtual joints to the camera's 2D keypoints is then scanned
over raw-frame lags of -150..+150 frames.  The lag with the smallest median
reprojection error is accepted only when it is an interior minimum and is
sharp (positive curvature, second-best at least 1 % above the minimum, and a
neighbour contrast above 0.02 px); otherwise the camera is UNRESOLVED.

The true start of a camera is ``coarse start + best lag``.  The true start of
the reference camera anchors the mocap window that scoring uses at lag zero.

Gait aliasing: a lag can be off by a whole gait cycle when the body repeats
its pose.  The rule above does not consider that; trials with no sharp
interior minimum are reported as UNRESOLVED rather than guessed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from evaluation.opencap.mocap import JOINT_COCO_IDX, interp_rows, load_trc, mocap_joints

LAG_LIMIT = 150
LAGS = np.arange(-LAG_LIMIT, LAG_LIMIT + 1, dtype=int)
SCORE_THRESHOLD = 0.5
MIN_INTERVAL_POINTS = 30


def read_gray_stack(path: str | Path, scale: float = 0.125, limit: int | None = None) -> np.ndarray:
    """Downscaled grayscale frames of a video, shape (frames, h, w), float32."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"cannot open {path}")
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        frames.append(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
        if limit is not None and len(frames) >= limit:
            break
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames in {path}")
    return np.stack(frames).astype(np.float32)


def coarse_start(raw_path: str | Path, syncd_path: str | Path) -> dict[str, Any]:
    """Frame of the raw video where the synchronised clip starts (minimum mean absolute difference)."""
    raw = read_gray_stack(raw_path)
    probe = read_gray_stack(syncd_path, limit=10)
    n = len(raw) - len(probe) + 1
    curve = np.asarray([np.mean(np.abs(raw[s:s + len(probe)] - probe)) for s in range(max(1, n))])
    best = int(np.argmin(curve))
    mask = np.ones(len(curve), dtype=bool)
    mask[max(0, best - 5):min(len(curve), best + 6)] = False
    second = float(np.min(curve[mask])) if mask.any() else None
    cap = cv2.VideoCapture(str(syncd_path))
    syncd_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return {
        "start_frame": best,
        "mad": float(curve[best]),
        "second_best_mad": second,
        "raw_frames": int(len(raw)),
        "syncd_frames": syncd_frames,
        "probe_frames": int(len(probe)),
    }


def _fit_pnp(obj: np.ndarray, img: np.ndarray, K: np.ndarray, dist: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    if len(obj) < 6:
        return None
    ok, rvec, tvec = cv2.solvePnP(obj.astype(float), img.astype(float), K, dist, flags=cv2.SOLVEPNP_EPNP)
    if not ok:
        return None
    ok, rvec, tvec = cv2.solvePnP(
        obj.astype(float), img.astype(float), K, dist, rvec, tvec, True, flags=cv2.SOLVEPNP_ITERATIVE
    )
    return (rvec, tvec) if ok else None


def truth_curve(
    mocap_frames: np.ndarray,
    kp: np.ndarray,
    sc: np.ndarray,
    start: int,
    nwin: int,
    cam_index: int,
    K: np.ndarray,
    dist: np.ndarray,
    lags: np.ndarray = LAGS,
) -> dict[str, Any]:
    """Median reprojection error of the mocap PnP fit for every lag, and the accepted minimum."""
    values: list[float] = []
    counts: list[int] = []
    for lag in lags:
        obj_parts: list[np.ndarray] = []
        img_parts: list[np.ndarray] = []
        for k in range(nwin):
            raw = start + k + int(lag)
            if raw < 0 or raw >= len(kp):
                continue
            img_all = kp[raw, cam_index, JOINT_COCO_IDX]
            good = (sc[raw, cam_index, JOINT_COCO_IDX] >= SCORE_THRESHOLD) & np.isfinite(img_all).all(axis=1)
            if good.any():
                obj_parts.append(mocap_frames[k, good])
                img_parts.append(img_all[good])
        if not obj_parts:
            values.append(np.nan)
            counts.append(0)
            continue
        obj = np.concatenate(obj_parts)
        img = np.concatenate(img_parts)
        fit = _fit_pnp(obj, img, K, dist)
        if fit is None:
            values.append(np.nan)
            counts.append(len(obj))
            continue
        proj, _ = cv2.projectPoints(obj.astype(float).reshape(-1, 1, 3), fit[0], fit[1], K, dist)
        err = np.linalg.norm(proj.reshape(-1, 2) - img, axis=1)
        values.append(float(np.median(err)))
        counts.append(int(len(err)))

    curve = np.asarray(values, dtype=float)
    finite = np.isfinite(curve)
    if not finite.any():
        raise RuntimeError("no finite PnP reprojection curve")
    bi = int(np.flatnonzero(finite)[np.argmin(curve[finite])])
    lag = int(lags[bi])
    interior = bool(0 < bi < len(curve) - 1 and np.isfinite(curve[bi - 1:bi + 2]).all())
    if interior:
        ym, y0, yp = curve[bi - 1:bi + 2]
        den = ym - 2 * y0 + yp
        sub = float(lag + np.clip(0.5 * (ym - yp) / den, -1, 1)) if abs(den) > 1e-12 else float(lag)
        curvature: float | None = float(den)
        neighbor: float | None = float(min(ym, yp))
        excluded = np.arange(max(0, bi - 5), min(len(curve), bi + 6))
        second = float(np.nanmin(np.delete(curve, excluded)))
    else:
        sub = float(lag)
        curvature = None
        neighbor = None
        mask = np.ones(len(curve), dtype=bool)
        mask[max(0, bi - 5):min(len(curve), bi + 6)] = False
        second = float(np.nanmin(curve[mask])) if mask.any() else None
    sharp = bool(
        interior and curvature is not None and curvature > 0
        and second is not None and second > curve[bi] * 1.01
        and neighbor is not None and (neighbor - curve[bi]) > 0.02
    )
    return {
        "lags": [int(x) for x in lags],
        "median_px": [None if not np.isfinite(v) else float(v) for v in curve],
        "points": counts,
        "best_lag": lag,
        "subframe_lag": sub,
        "median_px_best": float(curve[bi]),
        "curvature_second_difference": curvature,
        "neighbor_best_px": neighbor,
        "second_best_px": second,
        "interior": interior,
        "sharp": sharp,
        "status": "RESOLVED" if interior and sharp else "UNRESOLVED",
    }


def _load_calibration(path: str | Path) -> dict[str, dict[str, np.ndarray]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out: dict[str, dict[str, np.ndarray]] = {}
    for cam in data["cameras"]:
        out[str(cam["name"])] = {
            "K": np.asarray(cam["K"], dtype=float),
            "dist": np.asarray(cam["dist"], dtype=float).reshape(-1, 1),
        }
    return out


def resolve_truth(
    trial_dir: str | Path,
    multipose_npz: str | Path,
    calibration_json: str | Path,
    mocap_trc: str | Path,
) -> dict[str, Any]:
    """Resolve each camera's true start frame and the reference-camera window anchor.

    ``multipose_npz`` holds the 2D keypoints in source pixels for the cameras in
    ``camera_names`` order; the clips under ``trial_dir`` provide the coarse starts.
    """
    trial_dir = Path(trial_dir)
    trc = load_trc(mocap_trc)
    mocap = mocap_joints(trc)
    step = float(trc.rate) / 60.0
    data = np.load(multipose_npz, allow_pickle=True)
    kp = np.asarray(data["keypoints"], dtype=float)
    sc = np.asarray(data["scores"], dtype=float)
    names = [str(x) for x in data["camera_names"]]
    calib = _load_calibration(calibration_json)

    n = min(len(mocap), len(kp))
    mocap_frames = interp_rows(mocap, np.arange(n, dtype=float) * step)

    cameras: dict[str, Any] = {}
    for ci, name in enumerate(names):
        coarse = coarse_start(trial_dir / "videos" / f"{name}.avi", trial_dir / "syncd" / f"{name}.avi")
        start = int(coarse["start_frame"])
        nwin = min(len(mocap_frames), int(coarse["syncd_frames"]), len(kp) - start)
        if nwin < MIN_INTERVAL_POINTS:
            cameras[name] = {"status": "UNRESOLVED", "reason": "insufficient overlap", "coarse": coarse}
            continue
        result = truth_curve(
            mocap_frames, kp, sc, start, nwin, ci, calib[name]["K"], calib[name]["dist"]
        )
        result["coarse"] = coarse
        result["true_start_frame"] = start + result["best_lag"] if result["status"] == "RESOLVED" else None
        cameras[name] = result

    ref = names[0]
    ref_start = cameras[ref].get("true_start_frame")
    for name in names:
        start = cameras[name].get("true_start_frame")
        cameras[name]["true_offset_vs_cam0"] = None if start is None or ref_start is None else int(start - ref_start)

    if ref_start is not None:
        anchor = {"cam0_start_frame": int(ref_start), "source": "mocap_truth", "resolved": True}
    else:
        fallback = cameras[ref]["coarse"]["start_frame"]
        anchor = {"cam0_start_frame": int(fallback), "source": "video_coarse_fallback", "resolved": False}
    return {
        "method": "mocap_pnp_lag_scan",
        "lag_range_frames": [-LAG_LIMIT, LAG_LIMIT],
        "score_threshold": SCORE_THRESHOLD,
        "reference": ref,
        "cameras": cameras,
        "anchor": anchor,
    }
