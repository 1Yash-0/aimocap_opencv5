"""Person-centred crop videos and the mapping of 2D keypoints to source pixels.

The production 2D detector works on a fixed 416 x 416 input.  On the portrait
720 x 1280 OpenCap clips the performer covers only a few hundred pixels, so the
detector misses him.  The crop path therefore:

1. locates the person every third frame (``rtmlib`` pose tracker, or a
   frame-differencing blob tracker with ``tracker="motion"``),
2. smooths the box over time, fixes its aspect ratio (width / height = 0.55),
   enforces a 240 px minimum height and adds a 16 % margin,
3. writes a fixed-size crop video (422 x 768) with one output frame per input
   frame, and a per-row affine table ``(x0, y0, scale)``.

Keypoints produced on the crops are mapped back with ``x = x_crop * sx + x0``
and ``y = x_crop * sy + y0`` where ``sy = sx * out_w / (ASPECT * out_h)``.

A frame range can be given per camera.  Only frames inside that range are
written, which keeps short windowed runs cheap.  Tracking extends a fixed
margin beyond the range so that the temporal smoothing matches a full-clip run
near the window edges.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np
from scipy.ndimage import median_filter

ASPECT = 0.55
OUT_HEIGHT = 768
MIN_CROP_HEIGHT_PX = 240.0
MARGIN_FACTOR = 1.16
WORK_WIDTH = 240
STEP = 3
SMOOTH = 9
QUALITY = 78
TRACK_MARGIN_FRAMES = 30


class PoseEstimatorLike(Protocol):
    """The part of ``RTMPoseEstimator`` the tracker uses."""

    def estimate(self, frame: np.ndarray, pick: str = "largest") -> list[Any]:
        """Return detected poses; each exposes ``keypoints`` and ``scores``."""


@dataclass(frozen=True)
class TrackResult:
    """Per-row crop boxes ``(x0, y0, x1, y1)`` in source pixels.

    Row ``r`` of ``table`` is the box of source frame ``(row0 + r) * STEP``.
    """

    table: np.ndarray
    row0: int
    stats: dict[str, Any]


def video_info(path: str | Path) -> dict[str, Any]:
    """Frame count, frame rate and size of a video."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"cannot open {path}")
    info = {
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": float(cap.get(cv2.CAP_PROP_FPS)) or 60.0,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return info


def largest_blob_box(diff: np.ndarray, scale: float) -> tuple[float, float, float, float] | None:
    """Bounding box (source pixels) of the largest motion blob, merged with nearby blobs."""
    mad = float(np.median(diff)) + 1e-6
    thr = max(7.0, 3.0 * 1.4826 * mad)
    mask = (diff > thr).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    mask = cv2.dilate(mask, np.ones((7, 7), np.uint8))
    n, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return None
    comp = stats[1:]
    order = np.argsort(comp[:, 4])[::-1]
    main = comp[order[0]]
    if main[4] < 400:
        return None
    mx0, my0, mw, mh = main[0], main[1], main[2], main[3]
    mx1, my1 = mx0 + mw, my0 + mh
    px, py = 0.35 * mw, 0.25 * mh
    bx0, by0, bx1, by1 = mx0, my0, mx1, my1
    for k in order[1:]:
        c = comp[k]
        if c[4] < 0.03 * main[4]:
            continue
        cx0, cy0 = c[0], c[1]
        cx1, cy1 = c[0] + c[2], c[1] + c[3]
        if cx1 >= mx0 - px and cx0 <= mx1 + px and cy1 >= my0 - py and cy0 <= my1 + py:
            bx0, by0 = min(bx0, cx0), min(by0, cy0)
            bx1, by1 = max(bx1, cx1), max(by1, cy1)
    return bx0 / scale, by0 / scale, bx1 / scale, by1 / scale


def _pose_box(poses: list[Any]) -> tuple[float, float, float, float] | None:
    """Box around the body keypoints of the first pose (COCO body joints 0..22)."""
    if not poses:
        return None
    keypoints = poses[0].keypoints[:23]
    scores = poses[0].scores[:23]
    good = np.isfinite(keypoints).all(axis=1) & (scores >= 0.25)
    if good.sum() < 6:
        return None
    pts = keypoints[good]
    return float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max())


def track_boxes(
    path: str | Path,
    *,
    start: int = 0,
    stop: int | None = None,
    estimator: PoseEstimatorLike | None = None,
    margin: int = 0,
) -> TrackResult:
    """Track the subject for frames ``[start, stop)`` (plus ``margin`` on both sides).

    ``estimator`` locates the person with a pose model; when it is ``None`` a
    frame-differencing blob tracker is used instead.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"cannot open {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    stop = total if stop is None else min(int(stop), total)
    lo = max(0, int(start) - margin)
    hi = min(total, stop + margin)
    scale = WORK_WIDTH / float(width)
    ws, hs = int(round(width * scale)), int(round(height * scale))

    frame_ids: list[int] = []
    raw_rows: list[tuple[float, float, float, float] | None] = []
    grays: list[np.ndarray] = []
    index = 0
    while index < hi:
        ok, frame = cap.read()
        if not ok:
            break
        if index >= lo and index % STEP == 0:
            frame_ids.append(index)
            if estimator is None:
                small = cv2.resize(frame, (ws, hs), interpolation=cv2.INTER_AREA)
                grays.append(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
            else:
                raw_rows.append(_pose_box(estimator.estimate(frame, pick="largest")))
        index += 1
    cap.release()

    n = len(frame_ids)
    if n < 10:
        raise RuntimeError(f"too few sampled frames in {path}")
    if estimator is None:
        raw_rows = []
        for t in range(n):
            ref = grays[max(0, t - 3)]
            diff = np.abs(grays[t].astype(np.float32) - ref.astype(np.float32)).astype(np.uint8)
            raw_rows.append(largest_blob_box(diff, scale))

    raw = np.full((n, 4), np.nan, dtype=np.float64)
    for t, box in enumerate(raw_rows):
        if box is not None:
            raw[t] = box
    valid = np.isfinite(raw[:, 0])
    if valid.sum() < 10:
        raise RuntimeError(f"subject tracking failed in {path}")

    grid = np.arange(n, dtype=np.float64)
    for c in range(4):
        raw[:, c] = np.interp(grid, grid[valid], raw[valid, c])

    pad = SMOOTH // 2
    boxes = raw.copy()
    for t in range(n):
        lo_t, hi_t = max(0, t - pad), min(n, t + pad + 1)
        boxes[t, 0] = raw[lo_t:hi_t, 0].min()
        boxes[t, 1] = raw[lo_t:hi_t, 1].min()
        boxes[t, 2] = raw[lo_t:hi_t, 2].max()
        boxes[t, 3] = raw[lo_t:hi_t, 3].max()
    boxes = median_filter(boxes, size=(max(3, SMOOTH), 1), mode="nearest")

    cx = 0.5 * (boxes[:, 0] + boxes[:, 2])
    cy = 0.5 * (boxes[:, 1] + boxes[:, 3])
    bh = np.maximum(boxes[:, 3] - boxes[:, 1], MIN_CROP_HEIGHT_PX) * MARGIN_FACTOR
    bw = bh * ASPECT
    bh = np.minimum(bh, height)
    bw = np.minimum(bw, width)
    x0 = np.clip(cx - 0.5 * bw, 0, width - bw)
    y0 = np.clip(cy - 0.5 * bh, 0, height - bh)
    x1 = np.minimum(x0 + bw, width)
    y1 = np.minimum(y0 + bh, height)
    table = np.stack([x0, y0, x1, y1], axis=1)
    stats = {
        "frames_tracked": n,
        "step": STEP,
        "tracker": "motion" if estimator is None else "pose_estimator",
        "valid_fraction": float(valid.mean()),
        "crop_h_median": float(np.median(bh)),
        "crop_h_p5": float(np.percentile(bh, 5)),
        "crop_h_p95": float(np.percentile(bh, 95)),
    }
    return TrackResult(table=table, row0=frame_ids[0] // STEP, stats=stats)


def write_crop(
    src: str | Path,
    dst: str | Path,
    track: TrackResult,
    start: int,
    stop: int,
    out_w: int,
    out_h: int,
    quality: int = QUALITY,
) -> tuple[int, np.ndarray]:
    """Write crops of frames ``[start, stop)``; return the frame count and per-row ``(x0, y0, sx)``."""
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"MJPG"), fps, (out_w, out_h))
    if not writer.isOpened():
        raise OSError(f"cannot open crop writer {dst}")
    try:
        writer.set(cv2.VIDEOWRITER_PROP_QUALITY, float(quality))
    except cv2.error:
        pass
    n_rows = track.table.shape[0]
    params = np.zeros((n_rows, 3), dtype=np.float64)
    written = 0
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok or index >= stop:
            break
        if index >= start:
            t = min(max(index // STEP - track.row0, 0), n_rows - 1)
            x0, y0, x1, y1 = track.table[t]
            view = frame[int(round(y0)):int(round(y1)), int(round(x0)):int(round(x1))]
            if view.size == 0:
                view = frame
                x0, y0, x1, y1 = 0, 0, frame.shape[1], frame.shape[0]
            view = cv2.resize(view, (out_w, out_h), interpolation=cv2.INTER_CUBIC)
            writer.write(view)
            params[t] = (x0, y0, (x1 - x0) / float(out_w))
            written += 1
        index += 1
    cap.release()
    writer.release()
    return written, params


def prepare_crops(
    trial_dir: str | Path,
    out_dir: str | Path,
    cams: list[str],
    *,
    ranges: dict[str, tuple[int, int]] | None = None,
    estimator: PoseEstimatorLike | None = None,
    out_height: int = OUT_HEIGHT,
    quality: int = QUALITY,
) -> dict[str, Any]:
    """Write crop videos, per-row crop maps and ``crop_params.json`` for each camera.

    ``ranges`` maps a camera to ``(start, stop)`` source frames; a camera without
    an entry uses the whole clip.
    """
    trial_dir = Path(trial_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    out_h = out_height // 2 * 2
    out_w = int(round(ASPECT * out_h)) // 2 * 2
    params: dict[str, Any] = {}
    for cam in cams:
        src = trial_dir / "videos" / f"{cam}.avi"
        info = video_info(src)
        if ranges and cam in ranges:
            start, stop = ranges[cam]
            margin = TRACK_MARGIN_FRAMES
        else:
            start, stop = 0, info["frames"]
            margin = 0
        track = track_boxes(src, start=start, stop=stop, estimator=estimator, margin=margin)
        dst = out / f"{cam}.avi"
        written, per_row = write_crop(src, dst, track, start, stop, out_w, out_h, quality)
        if written == 0:
            raise RuntimeError(f"no crop frames written for {cam} in range [{start}, {stop})")
        map_path = out / f"{cam}_crop_map.npy"
        np.save(map_path, per_row)
        params[cam] = {
            "source": str(src),
            "crop_video": str(dst),
            "crop_map": str(map_path),
            "start_frame": int(start),
            "stop_frame": int(stop),
            "frames_written": int(written),
            "row0": int(track.row0),
            "step": STEP,
            "fps": info["fps"],
            "src_w": info["width"],
            "src_h": info["height"],
            "out_w": out_w,
            "out_h": out_h,
            "tracking": track.stats,
        }
    (out / "crop_params.json").write_text(json.dumps(params, indent=2), encoding="utf-8")
    return params


def map_keypoints(
    keypoints: np.ndarray,
    cam_params: dict[str, Any],
    cam_map: np.ndarray,
) -> np.ndarray:
    """Map one camera's crop keypoints (frames, K, 2) to source pixels."""
    frames = keypoints.shape[0]
    step = int(cam_params["step"])
    rows = np.arange(frames) + int(cam_params["start_frame"])
    rows = np.clip(rows // step - int(cam_params["row0"]), 0, len(cam_map) - 1)
    x0 = cam_map[rows, 0][:, None]
    y0 = cam_map[rows, 1][:, None]
    sx = cam_map[rows, 2][:, None]
    sy = sx * cam_params["out_w"] / (ASPECT * cam_params["out_h"])
    out = np.empty_like(keypoints, dtype=np.float64)
    out[..., 0] = keypoints[..., 0] * sx + x0
    out[..., 1] = keypoints[..., 1] * sy + y0
    return out


def map_to_source(
    crop_npz: str | Path,
    crop_params: dict[str, Any],
    out_npz: str | Path,
    *,
    fps: float,
) -> Path:
    """Write a multipose-schema npz whose keypoints are in source pixels.

    The output adds ``source_frame_indices`` (frames, cameras), the absolute
    source frame of every row, so that later stages can cut windows exactly.
    """
    src = np.load(crop_npz, allow_pickle=True)
    keypoints = np.asarray(src["keypoints"], dtype=np.float64)
    names = [Path(str(x)).stem for x in src["camera_names"]]
    frames = keypoints.shape[0]
    mapped = keypoints.copy()
    indices = np.empty((frames, len(names)), dtype=np.int64)
    sizes = []
    for c, name in enumerate(names):
        p = crop_params[name]
        cam_map = np.load(p["crop_map"])
        mapped[:, c] = map_keypoints(keypoints[:, c], p, cam_map)
        indices[:, c] = int(p["start_frame"]) + np.arange(frames)
        sizes.append([int(p["src_w"]), int(p["src_h"])])
    fields = {k: src[k] for k in src.files if k not in ("keypoints", "image_sizes", "source_frame_indices")}
    fields["keypoints"] = mapped
    fields["image_sizes"] = np.asarray(sizes, dtype=np.int32)
    fields["camera_names"] = np.asarray(names)
    fields["source_frame_indices"] = indices
    fields["coordinate_space"] = np.asarray("image_pixels")
    fields["units"] = np.asarray("pixels")
    fields.setdefault("fps", np.asarray(float(fps)))
    fields["crop_provenance"] = np.asarray(json.dumps({"source": str(crop_npz), "cameras": names}))
    dst = Path(out_npz)
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez(dst, **fields)
    return dst
