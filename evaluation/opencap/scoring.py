"""Accuracy and smoothness of a triangulated skeleton against marker mocap.

Definitions (identical to the evaluation reports):

* Virtual joints: the twelve joints of ``mocap.JOINT_NAMES``, in the OpenCV
  camera frame and millimetres.
* Lag-0 accuracy: window frame ``i`` is compared with the mocap at time
  ``i / 60`` s, i.e. mocap row ``i * rate / 60``.  Each frame is rigidly
  aligned (Kabsch, no scale) and the frame error is the RMS joint distance.
  Reported: median and 95th percentile over frames.  Frames whose mocap time
  falls outside the recorded mocap are not scored.
* Whole-trial rigid alignment: one Kabsch fit over every scored joint-frame;
  mean, median and 95th percentile of the residual distance.
* Smoothness on the 60 Hz grid:
  - acceleration error: median of ``|acc_ours - acc_mocap|`` (mm / frame^2);
  - slow-frame jitter: median ``|acc_ours|`` on the slowest 20 % of mocap frames
    (ranked by the median joint speed of the mocap);
  - speed ratio: 95th percentile of the ours joint speed divided by the 95th
    percentile of the mocap joint speed.
* OpenCap HRNet baselines: the same metrics on the ``*_videoAndMocap.trc``
  twelve joints (accuracy at the file's own rows, smoothness on a 60 Hz
  resample).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from aimocap.math.coords import internal_to_opencv
from evaluation.opencap.mocap import (
    JOINT_COCO_IDX,
    JOINT_NAMES,
    interp_rows,
    kabsch,
    load_trc,
    mocap_joints,
    opencap_joints,
    resample_rate,
)

WINDOW_FPS = 60.0


def skeleton_mm(npz_path: str | Path) -> np.ndarray:
    """Twelve virtual joints of a triangulated skeleton, (frames, 12, 3), mm, OpenCV frame."""
    data = np.load(npz_path, allow_pickle=True)
    points = np.asarray(data["skeleton3d"], dtype=float)
    return internal_to_opencv(points)[:, JOINT_COCO_IDX] * 1000.0


def _scored_frames(count: int, step: float, mocap_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Window frame indices whose mocap time lies inside the recording, and their mocap rows."""
    frame_ids = np.arange(count)
    rows = frame_ids * step
    keep = (rows >= 0) & (rows <= mocap_len - 1)
    return frame_ids[keep], rows[keep]


def lag0_accuracy(ours: np.ndarray, mocap: np.ndarray, step: float) -> dict[str, Any] | None:
    """Per-frame rigid-aligned RMS error at lag zero: median, p95 and frame count."""
    frame_ids, rows = _scored_frames(len(ours), step, len(mocap))
    targets = interp_rows(mocap, rows)
    errors: list[float] = []
    for i, target in zip(frame_ids, targets):
        a = ours[i]
        ok = np.isfinite(a).all(axis=1) & np.isfinite(target).all(axis=1)
        if ok.sum() < 6:
            continue
        R, t = kabsch(a[ok], target[ok])
        residual = (R @ a[ok].T).T + t - target[ok]
        errors.append(float(np.sqrt(np.mean(np.sum(residual**2, axis=1)))))
    if not errors:
        return None
    return {
        "median_mm": float(np.median(errors)),
        "p95_mm": float(np.percentile(errors, 95)),
        "frames": len(errors),
    }


def global_rigid_error(ours: np.ndarray, mocap: np.ndarray, step: float) -> dict[str, Any] | None:
    """One rigid fit over all scored joint-frames; residual distance statistics."""
    frame_ids, rows = _scored_frames(len(ours), step, len(mocap))
    targets = interp_rows(mocap, rows)
    A_parts: list[np.ndarray] = []
    B_parts: list[np.ndarray] = []
    for i, target in zip(frame_ids, targets):
        a = ours[i]
        ok = np.isfinite(a).all(axis=1) & np.isfinite(target).all(axis=1)
        if ok.sum() >= 6:
            A_parts.append(a[ok])
            B_parts.append(target[ok])
    if not A_parts:
        return None
    A = np.concatenate(A_parts)
    B = np.concatenate(B_parts)
    R, t = kabsch(A, B)
    residual = np.linalg.norm((R @ A.T).T + t - B, axis=1)
    return {
        "mean_mm": float(residual.mean()),
        "median_mm": float(np.median(residual)),
        "p95_mm": float(np.percentile(residual, 95)),
        "points": int(len(residual)),
    }


def smoothness(ours: np.ndarray, mocap: np.ndarray, step: float) -> dict[str, Any]:
    """Acceleration error, slow-frame jitter and p95 speed ratio on the 60 Hz grid."""
    n = min(len(ours), int(np.floor((len(mocap) - 1) / step)) + 1)
    times = np.arange(n, dtype=float) * step
    reference = interp_rows(mocap, times)
    A = ours[:n]

    def acc(X: np.ndarray) -> np.ndarray:
        return X[2:] - 2 * X[1:-1] + X[:-2]

    def speed(X: np.ndarray) -> np.ndarray:
        return np.linalg.norm(X[1:] - X[:-1], axis=2)

    acc_ours = acc(A)
    acc_ref = acc(reference)
    ok = np.isfinite(acc_ours).all(axis=2) & np.isfinite(acc_ref).all(axis=2)
    accel_error = np.linalg.norm(acc_ours - acc_ref, axis=2)[ok]

    ref_speed = np.nanmedian(speed(reference), axis=1)
    slow = np.argsort(ref_speed)[: max(1, int(0.2 * len(ref_speed)))]
    slow = slow[(slow >= 1) & (slow < len(ref_speed) - 1)] - 1
    jitter = np.linalg.norm(acc_ours, axis=2)[slow]
    jitter = jitter[np.isfinite(jitter)]

    ratio = float(np.nanpercentile(speed(A), 95) / max(float(np.nanpercentile(speed(reference), 95)), 1e-9))
    return {
        "acceleration_error_median_mm_per_frame2": float(np.median(accel_error)) if accel_error.size else None,
        "jitter_slowest20_median_mm_per_frame2": float(np.median(jitter)) if jitter.size else None,
        "peak_speed_ratio_p95": ratio,
        "frames": int(n),
    }


def score_skeleton(ours: np.ndarray, mocap: np.ndarray, rate: float) -> dict[str, Any]:
    """All accuracy and smoothness blocks for one skeleton against one mocap."""
    step = rate / WINDOW_FPS
    return {
        "frames": int(len(ours)),
        "lag0_procrustes": lag0_accuracy(ours, mocap, step),
        "global_rigid": global_rigid_error(ours, mocap, step),
        "smoothness": smoothness(ours, mocap, step),
    }


def score_window(
    skeleton_npz: str | Path,
    mocap_trc: str | Path,
    opencap_trcs: dict[str, str | Path] | None = None,
) -> dict[str, Any]:
    """Score the triangulated window and, when given, the OpenCap HRNet baselines.

    ``opencap_trcs`` maps a label (for example ``"opencap_hrnet_5cam"``) to a
    ``*_videoAndMocap.trc`` path.
    """
    mocap_file = load_trc(mocap_trc)
    mocap = mocap_joints(mocap_file)
    rate = float(mocap_file.rate)
    result: dict[str, Any] = {
        "mocap": {
            "file": str(mocap_trc),
            "rate_hz": rate,
            "rows": int(len(mocap)),
            "duration_s": round(len(mocap) / rate, 3),
            "joints": list(JOINT_NAMES),
        },
        "ours": score_skeleton(skeleton_mm(skeleton_npz), mocap, rate),
    }
    for label, path in (opencap_trcs or {}).items():
        oc_file = load_trc(path)
        if float(oc_file.rate) != rate:
            raise ValueError(f"{label} rate {oc_file.rate} differs from mocap rate {rate}")
        oc = opencap_joints(oc_file)
        result[label] = {
            "file": str(path),
            "lag0_procrustes": lag0_accuracy(oc, mocap, 1.0),
            "global_rigid": global_rigid_error(oc, mocap, 1.0),
            "smoothness": smoothness(resample_rate(oc, rate, WINDOW_FPS), mocap, rate / WINDOW_FPS),
        }
    return result
