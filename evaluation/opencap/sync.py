"""Visibility-aware geometric synchronisation of multi-camera 2D keypoints.

The estimator needs no audio, no speed signal and no mocap:

1. Epipolar initialisation: for every camera, the frame offset that minimises
   the median squared Sampson distance of the body joints to the epipolar
   geometry of the reference camera.
2. Triangulation refinement: with the other cameras at their current offsets,
   the offset of each camera is the lag at which its 2D detections agree best
   with the 3D points triangulated from the rest.  Two coordinate-descent rounds
   refine all offsets jointly.

Frozen rules (from the evaluation reports): a joint counts when its score is at
least the threshold (0.3 for the epipolar step, 0.5 for triangulation); a person
box must lie at least 2 px inside the image; a candidate lag needs at least 30
overlapping frames; each overlapping frame contributes one median, so frames are
weighted equally; the minimum is refined with a three-point parabola.  The
search range is +-120 frames for the epipolar step and +-150 frames for
triangulation.

``cut_window`` applies the offsets to build the synchronised window that is
triangulated.  Camera ``c`` contributes frame ``anchor + offset_c + k`` for
window index ``k``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

BORDER_PX = 2.0
MIN_OVERLAP_FRAMES = 30
BODY_JOINTS = np.arange(23)
EPIPOLAR_MAX_LAG = 120
EPIPOLAR_MIN_SCORE = 0.3
TRIANGULATION_MAX_LAG = 150
TRIANGULATION_MIN_SCORE = 0.5
TRIANGULATION_ROUNDS = 2


class Geometry:
    """Keypoints and per-camera geometry loaded from a multipose npz and a calibration."""

    def __init__(self, npz_path: str | Path, calib_path: str | Path) -> None:
        data = np.load(npz_path, allow_pickle=True)
        self.keypoints = np.asarray(data["keypoints"], dtype=np.float64)
        self.scores = np.asarray(data["scores"], dtype=np.float64)
        self.names = [Path(str(n)).stem for n in data["camera_names"]]
        self.image_sizes = np.asarray(data["image_sizes"]) if "image_sizes" in data.files else None
        calib = json.loads(Path(calib_path).read_text(encoding="utf-8"))
        by_name = {str(c["name"]): c for c in calib["cameras"]}
        self.K: list[np.ndarray] = []
        self.R: list[np.ndarray] = []
        self.t: list[np.ndarray] = []
        norms = []
        for i, name in enumerate(self.names):
            cam = by_name[name]
            K = np.asarray(cam["K"], dtype=np.float64)
            self.K.append(K)
            self.R.append(np.asarray(cam["R"], dtype=np.float64))
            self.t.append(np.asarray(cam["t"], dtype=np.float64).reshape(3))
            pts = self.keypoints[:, i].reshape(-1, 1, 2)
            undistorted = cv2.undistortPoints(pts, K, np.asarray(cam["dist"], dtype=np.float64))
            norms.append(undistorted.reshape(self.keypoints.shape[0], -1, 2))
        self.norm = np.stack(norms, axis=1)  # (frames, cams, joints, 2), normalised coordinates

    @property
    def frames(self) -> int:
        """Number of frames in the multipose arrays."""
        return int(self.keypoints.shape[0])

    def valid_mask(self, joints: np.ndarray, min_score: float) -> np.ndarray:
        """Joint validity (frames, cams, joints) combined with the person-box border rule."""
        valid = (self.scores[:, :, joints] >= min_score) & np.isfinite(self.keypoints[:, :, joints]).all(axis=-1)
        return valid & visibility_mask(self.keypoints, self.scores, self.image_sizes, joints, min_score)[:, :, None]


def visibility_mask(
    kp: np.ndarray,
    scores: np.ndarray,
    image_sizes: np.ndarray | None,
    joints: np.ndarray,
    min_score: float,
    border_px: float = BORDER_PX,
) -> np.ndarray:
    """(frames, cams) mask: at least four confident joints, all inside the image border."""
    J = np.asarray(joints, dtype=int)
    valid = (scores[:, :, J] >= min_score) & np.isfinite(kp[:, :, J]).all(axis=-1)
    out = np.zeros(valid.shape[:2], dtype=bool)
    for f in range(kp.shape[0]):
        for c in range(kp.shape[1]):
            pts = kp[f, c, J][valid[f, c]]
            if len(pts) < 4:
                continue
            if image_sizes is None:
                out[f, c] = True
                continue
            width, height = np.asarray(image_sizes[c], dtype=float).reshape(-1)[:2]
            lo, hi = pts.min(axis=0), pts.max(axis=0)
            out[f, c] = bool(
                lo[0] >= border_px and lo[1] >= border_px
                and hi[0] <= width - border_px and hi[1] <= height - border_px
            )
    return out


def triangulate_stack(A: np.ndarray) -> np.ndarray:
    """Batched DLT: ``A`` (M, 2N, 4) -> points (M, 3)."""
    _, _, vh = np.linalg.svd(A, full_matrices=False)
    h = vh[:, 3, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        return h[:, :3] / h[:, 3:4]


def _parabola_offset(y0: float, y1: float, y2: float) -> float | None:
    denom = y0 - 2.0 * y1 + y2
    if abs(denom) <= 1e-12:
        return None
    return 0.5 * (y0 - y2) / denom


def epipolar_offsets(
    geom: Geometry,
    max_lag: int = EPIPOLAR_MAX_LAG,
    min_score: float = EPIPOLAR_MIN_SCORE,
) -> dict[str, Any]:
    """Per-camera offset (frames, relative to the reference camera) from epipolar consistency."""
    joints = BODY_JOINTS
    valid = geom.valid_mask(joints, min_score)
    F, C = geom.frames, len(geom.names)
    ref = 0
    out: dict[str, Any] = {}
    for c in range(1, C):
        Rc = geom.R[c] @ geom.R[ref].T
        tc = geom.t[c] - geom.R[c] @ geom.R[ref].T @ geom.t[ref]
        tx = np.array([[0, -tc[2], tc[1]], [tc[2], 0, -tc[0]], [-tc[1], tc[0], 0]])
        E = tx @ Rc
        best_lag, best_val = 0, np.inf
        curve: dict[int, float] = {}
        overlap: dict[int, int] = {}
        for lag in range(-max_lag, max_lag + 1):
            t0 = max(0, -lag)
            t1 = min(F, F - lag)
            if t1 - t0 < 20:
                continue
            x1 = geom.norm[t0:t1, ref][:, joints]
            x2 = geom.norm[t0 + lag:t1 + lag, c][:, joints]
            m = valid[t0:t1, ref][:, joints] & valid[t0 + lag:t1 + lag, c][:, joints]
            frame_overlap = int(np.count_nonzero(np.any(m, axis=1)))
            if frame_overlap < MIN_OVERLAP_FRAMES or m.sum() < 60:
                continue
            a = np.concatenate([x1, np.ones(x1.shape[:2] + (1,))], axis=2)
            b = np.concatenate([x2, np.ones(x2.shape[:2] + (1,))], axis=2)
            Ea = a @ E.T
            Eb = b @ E
            num = np.sum(b * Ea, axis=2) ** 2
            den = Ea[..., 0] ** 2 + Ea[..., 1] ** 2 + Eb[..., 0] ** 2 + Eb[..., 1] ** 2
            frame_scores = [
                float(np.median(drow[mrow]))
                for drow, mrow in zip(num / np.maximum(den, 1e-12), m)
                if mrow.any()
            ]
            value = float(np.median(frame_scores)) * float(geom.K[c][0, 0]) ** 2
            curve[lag] = value
            overlap[lag] = frame_overlap
            if value < best_val:
                best_val, best_lag = value, lag
        sub = None
        if best_lag - 1 in curve and best_lag + 1 in curve:
            sub = _parabola_offset(curve[best_lag - 1], curve[best_lag], curve[best_lag + 1])
        out[geom.names[c]] = {
            "offset": int(best_lag),
            "subframe_offset": None if sub is None else round(best_lag + sub, 3),
            "median_sampson_px": round(float(np.sqrt(best_val)), 3) if np.isfinite(best_val) else None,
            "overlap_frames_at_best": int(overlap.get(best_lag, 0)),
        }
    out[geom.names[ref]] = {"offset": 0, "subframe_offset": 0.0, "median_sampson_px": 0.0,
                            "overlap_frames_at_best": F}
    return {"cameras": out, "reference": geom.names[ref], "max_lag": int(max_lag)}


def triangulation_offsets(
    geom: Geometry,
    max_lag: int = TRIANGULATION_MAX_LAG,
    min_score: float = TRIANGULATION_MIN_SCORE,
    init: dict[str, int] | None = None,
    rounds: int = TRIANGULATION_ROUNDS,
) -> dict[str, Any]:
    """Per-camera offsets from triangulation-reprojection consistency (coordinate descent)."""
    joints = BODY_JOINTS
    F, C = geom.frames, len(geom.names)
    valid = geom.valid_mask(joints, min_score)
    offsets = {n: 0 for n in geom.names}
    for name, value in (init or {}).items():
        if name in offsets:
            offsets[name] = int(value)

    history: list[dict[str, int]] = []
    curves: dict[str, dict[int, float]] = {}
    for _ in range(rounds):
        new_offsets: dict[str, int] = {}
        curves = {}
        for c in range(C):
            others = [i for i in range(C) if i != c]
            best_lag, best_val = offsets[geom.names[c]], np.inf
            curve: dict[int, float] = {}
            for lag in range(-max_lag, max_lag + 1):
                t0 = max(0, -lag, -min(offsets[geom.names[i]] for i in others))
                t1 = min(F, F - lag, F - max(offsets[geom.names[i]] for i in others))
                if t1 - t0 < 20:
                    continue
                frames = np.arange(t0, t1)
                A_parts, ok_parts = [], []
                for i in others:
                    p = geom.norm[frames + offsets[geom.names[i]], i][:, joints]
                    m = valid[frames + offsets[geom.names[i]], i][:, joints]
                    P = np.concatenate([geom.R[i], geom.t[i].reshape(3, 1)], axis=1)
                    x, y = p[..., 0], p[..., 1]
                    A_parts.append(np.stack([x[..., None] * P[2] - P[0], y[..., None] * P[2] - P[1]], axis=2))
                    ok_parts.append(m)
                A = np.concatenate(A_parts, axis=2)
                m_all = np.all(np.stack(ok_parts), axis=0)
                meas = geom.norm[frames + lag, c][:, joints]
                m = m_all & valid[frames + lag, c][:, joints]
                if m.sum() < 50:
                    continue
                X = triangulate_stack(A[m])
                xc = X @ geom.R[c].T + geom.t[c]
                with np.errstate(divide="ignore", invalid="ignore"):
                    proj = xc[:, :2] / xc[:, 2:3]
                d = np.linalg.norm(proj - meas[m], axis=1) * float(geom.K[c][0, 0])
                d = d[np.isfinite(d)]
                if d.size < 50:
                    continue
                value = float(np.median(d))
                curve[lag] = value
                if value < best_val:
                    best_val, best_lag = value, lag
            new_offsets[geom.names[c]] = int(best_lag)
            curves[geom.names[c]] = curve
        offsets = new_offsets
        history.append(dict(offsets))

    out: dict[str, Any] = {}
    for name in geom.names:
        curve = curves[name]
        lag = offsets[name]
        sub = None
        if lag - 1 in curve and lag + 1 in curve:
            sub = _parabola_offset(curve[lag - 1], curve[lag], curve[lag + 1])
        out[name] = {
            "offset": int(lag),
            "subframe_offset": None if sub is None else round(lag + sub, 3),
            "median_reproj_px": round(float(curve.get(lag, np.nan)), 3),
        }
    return {"cameras": out, "reference": geom.names[0], "max_lag": int(max_lag),
            "rounds": int(rounds), "history": history}


def estimate_sync(
    npz_path: str | Path,
    calib_path: str | Path,
    *,
    epipolar_max_lag: int = EPIPOLAR_MAX_LAG,
    epipolar_min_score: float = EPIPOLAR_MIN_SCORE,
    triangulation_max_lag: int = TRIANGULATION_MAX_LAG,
    triangulation_min_score: float = TRIANGULATION_MIN_SCORE,
    rounds: int = TRIANGULATION_ROUNDS,
) -> dict[str, Any]:
    """Run the frozen estimator and return offsets relative to the reference camera."""
    geom = Geometry(npz_path, calib_path)
    epi = epipolar_offsets(geom, max_lag=epipolar_max_lag, min_score=epipolar_min_score)
    init = {name: int(v["offset"]) for name, v in epi["cameras"].items()}
    tri = triangulation_offsets(
        geom,
        max_lag=triangulation_max_lag,
        min_score=triangulation_min_score,
        init=init,
        rounds=rounds,
    )
    ref = geom.names[0]
    ref_tri = tri["cameras"][ref]
    cameras: dict[str, Any] = {}
    for name in geom.names:
        entry = tri["cameras"][name]
        sub = entry["subframe_offset"]
        ref_sub = ref_tri["subframe_offset"]
        cameras[name] = {
            "offset_vs_cam0": int(entry["offset"]) - int(ref_tri["offset"]),
            "subframe_offset_vs_cam0": (
                None if sub is None or ref_sub is None else round(float(sub) - float(ref_sub), 3)
            ),
            "epipolar_offset_vs_cam0": int(epi["cameras"][name]["offset"]) - int(epi["cameras"][ref]["offset"]),
            "median_reproj_px": entry["median_reproj_px"],
        }
    return {
        "method": "geometric_epipolar_then_triangulation_reprojection",
        "reference": ref,
        "frames": geom.frames,
        "cameras": cameras,
        "parameters": {
            "epipolar_max_lag": epipolar_max_lag,
            "epipolar_min_score": epipolar_min_score,
            "triangulation_max_lag": triangulation_max_lag,
            "triangulation_min_score": triangulation_min_score,
            "rounds": rounds,
            "border_px": BORDER_PX,
            "min_overlap_frames": MIN_OVERLAP_FRAMES,
        },
        "uses_mocap": False,
        "uses_speed_signal": False,
    }


def write_sync_file(path: str | Path, sync: dict[str, Any]) -> Path:
    """Write a sync result (relative offsets and window anchor) as JSON."""
    dst = Path(path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(sync, indent=2), encoding="utf-8")
    return dst


def read_sync_file(path: str | Path) -> dict[str, Any]:
    """Read a sync result written by ``write_sync_file``."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def relative_offsets(sync: dict[str, Any]) -> dict[str, int]:
    """Camera name -> integer offset relative to the reference camera."""
    return {name: int(v["offset_vs_cam0"]) for name, v in sync["cameras"].items()}


def cut_window(
    npz_path: str | Path,
    out_npz: str | Path,
    offsets: dict[str, int],
    anchor_frame: int,
    n_frames: int,
    *,
    sync_status: str,
) -> dict[str, Any]:
    """Cut the synchronised window from a multipose npz.

    Window index ``k`` of camera ``c`` is source frame ``anchor + offsets[c] + k``.
    ``source_frame_indices`` of the input gives the absolute frame of each row, so
    inputs that were already cropped to a window are handled correctly.
    """
    src = np.load(npz_path, allow_pickle=True)
    names = [str(x) for x in src["camera_names"]]
    kp = src["keypoints"]
    sc = src["scores"]
    frames = kp.shape[0]
    if "source_frame_indices" in src.files:
        sfi = np.asarray(src["source_frame_indices"], dtype=np.int64)
    else:
        sfi = np.tile(np.arange(frames, dtype=np.int64)[:, None], (1, len(names)))
    starts = {n: int(anchor_frame) + int(offsets[n]) for n in names}
    rows: list[int] = []
    for c, name in enumerate(names):
        row = starts[name] - int(sfi[0, c])
        if row < 0:
            raise ValueError(
                f"window for {name} starts at source frame {starts[name]}, before the first decoded frame "
                f"{int(sfi[0, c])}"
            )
        rows.append(row)
    n = min([int(n_frames)] + [frames - r for r in rows])
    if n < 2:
        raise ValueError(f"window has {n} frames; the decoded range is too short")
    keypoints = np.stack([kp[r:r + n, c] for c, r in enumerate(rows)], axis=1)
    scores = np.stack([sc[r:r + n, c] for c, r in enumerate(rows)], axis=1)
    indices = np.stack([starts[name] + np.arange(n) for name in names], axis=1)
    fields: dict[str, Any] = {
        "keypoints": keypoints,
        "scores": scores,
        "camera_names": np.asarray(names),
        "image_sizes": src["image_sizes"],
        "source_frame_indices": indices,
        "sync_offsets": np.asarray([offsets[n] for n in names], dtype=np.float64),
        "sync_status": np.asarray(sync_status),
        "sync_mechanism": np.asarray("geometric_estimate_or_sync_file"),
        "sync_reference_camera": np.asarray(names[0]),
        "window_anchor_frame": np.asarray(int(anchor_frame)),
        "window_frames": np.asarray(n),
    }
    for key in ("fps", "coordinate_space", "units", "crop_provenance"):
        if key in src.files:
            fields[key] = src[key]
    dst = Path(out_npz)
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst, **fields)
    return {
        "frames": int(n),
        "anchor_frame": int(anchor_frame),
        "start_frame": {name: starts[name] for name in names},
        "offsets_vs_cam0": {name: int(offsets[name]) for name in names},
        "path": str(dst),
    }
