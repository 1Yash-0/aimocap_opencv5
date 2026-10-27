"""Marker-mocap helpers: TRC parsing, virtual joints, interpolation and rigid fits.

The twelve virtual joints (hips, knees, ankles, shoulders, elbows and wrists)
are the common scoring set.  Each one is derived from the OpenCap marker set
exactly as the evaluation reports define it.  The COCO index of each joint
is kept so that 2D and 3D arrays produced by ``aimocap`` can be indexed with
the same ordering.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# COCO-WholeBody body indices of the twelve scored joints.  The order of the
# keys defines the joint order used by every array in this package.
COCO_INDEX: dict[str, int] = {
    "L_hip": 11,
    "R_hip": 12,
    "L_knee": 13,
    "R_knee": 14,
    "L_ankle": 15,
    "R_ankle": 16,
    "L_sh": 5,
    "R_sh": 6,
    "L_elb": 7,
    "R_elb": 8,
    "L_wr": 9,
    "R_wr": 10,
}
JOINT_NAMES: tuple[str, ...] = tuple(COCO_INDEX)
JOINT_COCO_IDX: np.ndarray = np.asarray([COCO_INDEX[k] for k in JOINT_NAMES], dtype=int)

# Marker-mocap block at the start of every TRC file; the OpenSim markers follow.
MOCAP_MARKER_COUNT = 51

# OpenCap HRNet TRC marker names for the same twelve joints.
OPENCAP_MARKERS: dict[str, str] = {
    "R_hip": "RHip",
    "L_hip": "LHip",
    "R_knee": "RKnee",
    "L_knee": "LKnee",
    "R_ankle": "RAnkle",
    "L_ankle": "LAnkle",
    "R_sh": "RShoulder",
    "L_sh": "LShoulder",
    "R_elb": "RElbow",
    "L_elb": "LElbow",
    "R_wr": "RWrist",
    "L_wr": "LWrist",
}


@dataclass(frozen=True)
class Trc:
    """Parsed TRC file.

    ``data`` has shape (frames, markers, 3) in the file's units (millimetres for
    the OpenCap archive); ``rate`` is the marker sampling rate in hertz.
    """

    names: tuple[str, ...]
    data: np.ndarray
    rate: float


def load_trc(path: str | Path) -> Trc:
    """Parse a tab-separated TRC file."""
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    names = tuple(x.strip() for x in lines[3].split("\t")[2:] if x.strip())
    rows: list[list[float]] = []
    for line in lines[5:]:
        if not line.strip():
            continue
        rows.append([float(v) for v in line.split("\t")[2:] if v.strip()])
    data = np.asarray(rows, dtype=float).reshape(len(rows), len(names), 3)
    rate = float(lines[2].split("\t")[1])
    return Trc(names=names, data=data, rate=rate)


def virtual_joints(names: tuple[str, ...] | list[str], data: np.ndarray) -> dict[str, np.ndarray]:
    """Return the twelve virtual joints (each (frames, 3)) from marker data.

    Ankles and knees use single skin markers; hips use the regression joint
    centres ``R_HJC``/``L_HJC``; elbows and wrists average their two markers.
    Missing markers raise ``KeyError`` so an incomplete TRC cannot be scored.
    """
    index = {name: i for i, name in enumerate(names)}

    def marker_mean(*candidates: str) -> np.ndarray:
        present = [index[c] for c in candidates if c in index]
        if not present:
            raise KeyError(f"none of the markers {candidates} are present")
        return data[:, present].mean(axis=1)

    return {
        "R_hip": marker_mean("R_HJC"),
        "L_hip": marker_mean("L_HJC"),
        "R_knee": marker_mean("r_knee"),
        "L_knee": marker_mean("L_knee"),
        "R_ankle": marker_mean("r_ankle"),
        "L_ankle": marker_mean("L_ankle"),
        "R_sh": marker_mean("R_Shoulder"),
        "L_sh": marker_mean("L_Shoulder"),
        "R_elb": marker_mean("R_elbow_med", "R_elbow_lat"),
        "L_elb": marker_mean("L_elbow_med", "L_elbow_lat"),
        "R_wr": marker_mean("R_wrist_radius", "R_wrist_ulna"),
        "L_wr": marker_mean("L_wrist_radius", "L_wrist_ulna"),
    }


def mocap_joints(trc: Trc) -> np.ndarray:
    """Twelve virtual joints of the marker-mocap block, shape (frames, 12, 3), mm."""
    vj = virtual_joints(trc.names[:MOCAP_MARKER_COUNT], trc.data[:, :MOCAP_MARKER_COUNT])
    return np.stack([vj[k] for k in JOINT_NAMES], axis=1)


def opencap_joints(trc: Trc) -> np.ndarray:
    """Twelve OpenCap HRNet joints from a ``*_videoAndMocap.trc`` file, (frames, 12, 3)."""
    index = {name: i for i, name in enumerate(trc.names)}
    cols = [index[OPENCAP_MARKERS[k]] for k in JOINT_NAMES]
    return trc.data[:, cols]


def interp_rows(X: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Linear interpolation of (frames, J, 3) at fractional row positions.

    Positions outside the sampled range are clamped to the end values, as
    ``numpy.interp`` does.  Callers that must not use clamped values restrict
    the rows themselves.
    """
    base = np.arange(len(X), dtype=float)
    out = np.empty((len(rows), X.shape[1], 3))
    for j in range(X.shape[1]):
        for a in range(3):
            out[:, j, a] = np.interp(rows, base, X[:, j, a])
    return out


def resample_rate(X: np.ndarray, rate: float, target: float = 60.0) -> np.ndarray:
    """Resample (frames, J, 3) recorded at ``rate`` Hz onto a ``target`` Hz grid."""
    count = int(np.floor((len(X) - 1) * target / rate)) + 1
    times = np.arange(count, dtype=float) * rate / target
    return interp_rows(X, times)


def kabsch(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rigid transform (R, t) minimising ``|R a + t - b|`` (no scale, proper rotation)."""
    ca, cb = a.mean(axis=0), b.mean(axis=0)
    H = (a - ca).T @ (b - cb)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    t = cb - R @ ca
    return R, t
