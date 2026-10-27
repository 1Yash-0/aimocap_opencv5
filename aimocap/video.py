"""Frame-by-frame video pose runner.

Reads a video (any OpenCV-readable container: mp4/avi/mov/gif...), runs the
PoseEstimator on each frame, draws the overlay, and writes an annotated video.
Also yields per-frame stats so callers (CLI, eval scripts) can log progress or
detect flicker/dropouts.

Scope (current milestone): single-camera, single-subject. Each frame is
processed independently; we pick the largest-area person. This is deliberately
not multi-frame tracking — identity can swap if two people overlap, which is a
later stage's problem.

Usage:
    from aimocap.video import run_video
    stats = run_video("clip.mp4", "out.mp4")
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from aimocap.pose.keypoints import BODY_17

if TYPE_CHECKING:
    from aimocap.pose.infer import PoseEstimator


@dataclass(slots=True)
class FrameStat:
    """Per-frame summary, collected over a run for stability analysis."""

    frame: int
    n_persons: int
    body_mean_conf: float       # mean confidence of the 17 body keypoints
    n_visible_body: int         # how many of 17 body keypoints pass threshold
    ms: float                   # wall-clock for this frame's inference+draw
    source: str                 # 'detect' | 'carry' | 'fallback' | 'none'


def _pick_subject(poses, threshold: float):
    """Return the single pose to draw (largest area). None if no detection."""
    if not poses:
        return None
    visible = [
        p for p in poses
        if int((p.scores[np.asarray(BODY_17)] >= threshold).sum()) >= 4
    ]
    pool = visible or poses
    return max(pool, key=lambda p: p.area)


def _expand_bbox(bbox: np.ndarray, frame_shape, margin: float = 0.20) -> np.ndarray:
    """Expand a bbox outward by ``margin`` (fraction of each side), clamped to frame."""
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    dx, dy = bw * margin, bh * margin
    return np.array([
        max(0, x1 - dx), max(0, y1 - dy),
        min(w, x2 + dx), min(h, y2 + dy),
    ], dtype=np.float32)


def run_video(
    src: str | Path,
    dst: str | Path,
    *,
    estimator: PoseEstimator | None = None,
    threshold: float = 0.3,
    max_frames: int | None = None,
    pick: str = "largest",
    progress_every: int = 10,
    carry_margin: float = 0.20,
    carry_max_gap: int = 10,
) -> list[FrameStat]:
    """Run pose estimation over a video and write the annotated output.

    Parameters
    ----------
    dst         : output video path. Parent dir is created.
    max_frames  : stop after this many frames (None = whole video).
    pick        : person selection each frame ('largest' | 'all').
    progress_every : print a one-line progress summary every N frames.
    carry_margin   : when detection fails, reuse the last bbox expanded by this
                     fraction on each side (temporal carry-over). This is the
                     minimal single-subject tracking needed to avoid dropouts
                     when the lightweight YOLOX-Nano detector loses lock on
                     motion-blurred or unusual-pose frames.
    carry_max_gap  : stop carrying after this many consecutive failed frames
                     (the subject may have actually left). Falls back to whole-
                     frame pose after that.

    Returns a list of FrameStat for stability analysis.
    """
    src = Path(src)
    dst = Path(dst)
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)

    if estimator is None:
        from aimocap.pose.infer import PoseEstimator
        est = PoseEstimator()
    else:
        est = estimator
    from aimocap.viz.overlay import draw_poses

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise OSError(f"could not open video: {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*("mp4v" if dst.suffix.lower() == ".mp4" else "MJPG"))
    writer = cv2.VideoWriter(str(dst), fourcc, fps, (w, h))
    if not writer.isOpened():
        cap.release()
        raise OSError(f"could not open VideoWriter for {dst}")

    body = np.asarray(BODY_17)
    stats: list[FrameStat] = []
    frame_idx = 0
    last_bbox: np.ndarray | None = None   # for carry-over
    carry_streak = 0
    import time

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if max_frames is not None and frame_idx >= max_frames:
                break

            t0 = time.perf_counter()
            poses = est.estimate(frame, pick="all")
            subject = _pick_subject(poses, threshold) if poses else None
            source = "none"

            if subject is not None:
                last_bbox = subject.bbox.copy()
                carry_streak = 0
                source = "detect"
            elif last_bbox is not None and carry_streak < carry_max_gap:
                # Temporal carry-over: re-run pose on the expanded last bbox.
                expanded = _expand_bbox(last_bbox, frame.shape, carry_margin)
                carried = est.estimate(frame, pick="all")  # re-detect first
                subject = _pick_subject(carried, threshold) if carried else None
                if subject is None:
                    # Force pose on the carried bbox directly.
                    from cigpose import preprocess_person, decode_simcc, remap_to_frame
                    from aimocap.pose.infer import Pose2D
                    tensor, crop_region = preprocess_person(
                        frame, expanded.tolist(), est.input_w, est.input_h
                    )
                    sx, sy = est.pose_session.run(None, {"input": tensor})
                    kpts, scores = decode_simcc(
                        sx, sy, est.input_w, est.input_h, est.split_ratio
                    )
                    kpts = remap_to_frame(kpts, crop_region, est.input_w, est.input_h)
                    subject = Pose2D(
                        keypoints=kpts.astype(np.float32),
                        scores=scores.astype(np.float32),
                        bbox=expanded,
                    )
                carry_streak += 1
                source = "carry"
            else:
                # Whole-frame fallback as a last resort.
                fallback = est.estimate(frame, pick="all")
                subject = _pick_subject(fallback, threshold) if fallback else None
                source = "fallback" if subject is not None else "none"

            to_draw = [subject] if (subject is not None and pick == "largest") else (
                poses if pick == "all" else []
            )
            annotated = draw_poses(frame, to_draw, threshold=threshold) if to_draw else frame.copy()
            dt = (time.perf_counter() - t0) * 1000.0

            if subject is not None:
                bmc = float(subject.scores[body].mean())
                nv = int((subject.scores[body] >= threshold).sum())
            else:
                bmc = 0.0
                nv = 0

            writer.write(annotated)
            stats.append(FrameStat(frame_idx, len(poses), bmc, nv, dt, source))

            if progress_every and (frame_idx % progress_every == 0):
                pct = f"{100*frame_idx/total:.0f}%" if total > 0 else "?"
                print(
                    f"  frame {frame_idx:>4}/{total or '?'} ({pct})  "
                    f"{len(poses)} det  src={source:8s}  body_conf={bmc:.2f}  {dt:.1f}ms",
                    flush=True,
                )
            frame_idx += 1
    finally:
        cap.release()
        writer.release()

    print(f"\nDone: {frame_idx} frames -> {dst}  ({fps:.1f} fps, {w}x{h})")
    return stats


def summarize(stats: list[FrameStat]) -> dict:
    """Aggregate stats for the stability check (flicker/dropout detection)."""
    if not stats:
        return {"n_frames": 0}
    body_conf = np.array([s.body_mean_conf for s in stats])
    n_vis = np.array([s.n_visible_body for s in stats])
    ms = np.array([s.ms for s in stats])
    n_det = sum(1 for s in stats if s.n_persons > 0)
    sources = [s.source for s in stats]
    from collections import Counter
    src_counts = Counter(sources)
    # "tracked" = the subject had a pose this frame (any non-'none' source).
    n_tracked = sum(1 for s in stats if s.source != "none")
    return {
        "n_frames": len(stats),
        "detection_rate": n_det / len(stats),
        "tracked_rate": n_tracked / len(stats),
        "body_conf_mean": float(body_conf.mean()),
        "body_conf_std": float(body_conf.std()),
        "body_conf_min": float(body_conf.min()),
        "body_conf_max": float(body_conf.max()),
        "visible_body_mean": float(n_vis.mean()),
        "visible_body_min": int(n_vis.min()),
        "ms_mean": float(ms.mean()),
        "fps_effective": float(1000.0 / ms.mean()) if ms.mean() > 0 else 0.0,
        "sources": dict(src_counts),
    }


__all__ = ["run_video", "summarize", "FrameStat"]
