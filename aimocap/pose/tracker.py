"""Temporally consistent single-subject selection for multi-camera capture.

The naive selection policies in this repo are independent per frame:

* ``_pick_subject`` (largest-area) can flip between two people when occlusion
  or motion changes which bounding box is momentarily largest.
* ``_select_single_subject`` refuses to run when two people are visible.

For a *single-performer* mocap capture we instead want one persistent identity:
the intended subject, followed continuously across the sequence.  This module
implements a lightweight, dependency-free tracker that:

1. **Initializes** on the strongest detection (most high-confidence body
   keypoints, then area) in the first frame a lock is acquired.
2. **Associates** each subsequent frame's detections to the tracked bbox by a
   combined score of bounding-box IoU and center-distance, preferring temporal
   continuity over instantaneous size.
3. **Persists through gaps**: when the subject is momentarily not detected, the
   tracker coasts (marks the frame as a carried/held state) for up to
   ``max_gap`` frames before releasing the lock and re-acquiring.
4. Is robust to a **real second person** entering the frame: as long as the
   tracked subject remains visible, a stronger-looking newcomer with poor IoU
   to the track is not stolen.

The tracker is deliberately decoupled from the estimator output type; it only
needs objects exposing ``.bbox`` (x1, y1, x2, y2), ``.area`` and ``.scores``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from aimocap.pose.keypoints import BODY_17


def _as_bbox(pose) -> np.ndarray:
    """Best-effort bbox for a pose: use .bbox if present else derive from keypoints."""
    bbox = getattr(pose, "bbox", None)
    if bbox is not None and np.asarray(bbox).size == 4:
        return np.asarray(bbox, dtype=np.float64)
    kpts = np.asarray(getattr(pose, "keypoints"))
    scores = np.asarray(getattr(pose, "scores"))
    body = kpts[np.asarray(BODY_17)]
    bs = scores[np.asarray(BODY_17)]
    valid = bs > 0.0
    if not valid.any():
        return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    pts = body[valid]
    return np.array(
        [pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()],
        dtype=np.float64,
    )


def _area(bbox: np.ndarray) -> float:
    w = max(0.0, float(bbox[2] - bbox[0]))
    h = max(0.0, float(bbox[3] - bbox[1]))
    return w * h


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    ua = _area(a) + _area(b) - inter
    return inter / ua if ua > 0 else 0.0


def _center(bbox: np.ndarray) -> np.ndarray:
    return np.array([(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0])


def _strong_confidence(pose, threshold: float) -> int:
    return int((np.asarray(pose.scores)[np.asarray(BODY_17)] >= threshold).sum())


@dataclass
class TrackState:
    """Mutable lock state for the tracked subject."""

    bbox: np.ndarray
    misses: int = 0
    frames_tracked: int = 0


class SingleSubjectTracker:
    """Follows one performer across frames using IoU + center continuity."""

    def __init__(
        self,
        *,
        threshold: float = 0.3,
        iou_weight: float = 1.0,
        center_weight: float = 1.5,
        max_center_jump_frac: float = 0.5,
        min_visibility: int = 4,
        max_gap: int = 10,
        min_iou_to_keep: float = 0.02,
    ) -> None:
        self.threshold = float(threshold)
        self.iou_weight = float(iou_weight)
        self.center_weight = float(center_weight)
        self.max_center_jump_frac = float(max_center_jump_frac)
        self.min_visibility = int(min_visibility)
        self.max_gap = int(max_gap)
        self.min_iou_to_keep = float(min_iou_to_keep)
        self._track: TrackState | None = None
        # Tunable image diagonal used to normalize center distance; set per-frame.
        self._diag: float | None = None

    # -- internal helpers -------------------------------------------------
    def _visible(self, pose) -> bool:
        return _strong_confidence(pose, self.threshold) >= self.min_visibility

    def _init_candidates(self, poses):
        cand = [p for p in poses if self._visible(p)]
        pool = cand or list(poses)
        if not pool:
            return None
        # Strongest subject: most visible keypoints, then area.
        def keyf(p):
            return (_strong_confidence(p, self.threshold), _area(_as_bbox(p)))
        return max(pool, key=keyf)

    def _score(self, pose, track: TrackState) -> float:
        bbox = _as_bbox(pose)
        iou = _iou(bbox, track.bbox)
        d = float(np.linalg.norm(_center(bbox) - _center(track.bbox)))
        diag = self._diag or 1.0
        # Scale the hard jump gate by the tracked box size (roughly a full
        # body-width of motion per frame is implausible at 30fps).
        gate = self.max_center_jump_frac * max(_area(track.bbox) ** 0.5 * 2.0, 100.0)
        if d > gate and iou < self.min_iou_to_keep:
            return -1.0
        return self.iou_weight * iou - self.center_weight * (d / max(diag, 1.0))

    # -- public API -------------------------------------------------------
    def select(self, poses, frame_shape=None) -> tuple[object | None, str]:
        """Return ``(pose, status)`` for the tracked subject this frame.

        status is one of: ``"selected"`` (new lock), ``"tracked"`` (associated to
        an existing lock), ``"carried"`` (no detection this frame; lock coasted),
        or ``"none"`` (no lock available / released).
        """
        if frame_shape is not None:
            h, w = frame_shape[:2]
            self._diag = float(math.hypot(w, h)) or self._diag

        poses = poses or []
        if self._track is None:
            subject = self._init_candidates(poses)
            if subject is None:
                return None, "none"
            self._track = TrackState(bbox=_as_bbox(subject))
            self._track.frames_tracked = 1
            return subject, "selected"

        # Associate to the existing track.
        best = None
        best_score = -1e9
        for pose in poses:
            s = self._score(pose, self._track)
            if s > best_score:
                best_score = s
                best = pose

        if best is not None and best_score >= 0.0 and self._visible(best):
            new_bbox = _as_bbox(best)
            # EMA update of the tracked bbox to absorb scale drift without
            # snapping to a same-frame distractor.
            alpha = 0.7
            self._track.bbox = alpha * new_bbox + (1.0 - alpha) * self._track.bbox
            self._track.misses = 0
            self._track.frames_tracked += 1
            return best, "tracked"

        # No acceptable association: coast the lock.
        self._track.misses += 1
        if self._track.misses > self.max_gap:
            self._track = None
            return None, "none"
        return None, "carried"

    def reset(self) -> None:
        self._track = None


__all__ = ["SingleSubjectTracker", "TrackState"]
