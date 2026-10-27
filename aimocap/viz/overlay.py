"""Skeleton overlay visualization for 2D poses.

Draws the 133-keypoint skeleton on a frame, color-coded by region
(body=green, feet=teal, face=orange, left hand=cyan, right hand=magenta),
with per-keypoint radius scaled by confidence. The bounding box is drawn
in yellow. Low-confidence keypoints (below threshold) are omitted entirely
so a noisy detection doesn't produce a misleading skeleton.

This is the Checkpoint 2 visual — used to eyeball whether the pose actually
lines up with the body, hands tracked, no wild points.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np

from aimocap.pose.infer import Pose2D
from aimocap.pose.keypoints import (
    REGIONS_133, SKELETON_133,
)

# BGR colors per region.
REGION_COLORS: dict[str, tuple[int, int, int]] = {
    "body":       ( 60, 200,  60),   # green
    "feet":       (180, 180,  40),   # teal
    "face":       ( 60, 140, 240),   # orange
    "left_hand":  (220, 110,  40),   # cyan
    "right_hand": (220,  40, 200),   # magenta
}
BBOX_COLOR = (0, 220, 255)   # yellow
TEXT_COLOR = (255, 255, 255)


def _region_of_index(idx: int) -> str:
    for name, region in REGIONS_133.items():
        if region.start <= idx < region.end:
            return name
    raise IndexError(f"keypoint {idx} out of range")


def _keypoint_color(idx: int) -> tuple[int, int, int]:
    return REGION_COLORS[_region_of_index(idx)]


def draw_pose(
    frame: np.ndarray,
    pose: Pose2D,
    *,
    threshold: float = 0.3,
    draw_skeleton: bool = True,
    draw_points: bool = True,
    draw_bbox: bool = True,
    draw_face_mesh: bool = False,
    point_radius: int = 3,
    line_thickness: int = 2,
    alpha: float = 1.0,
) -> np.ndarray:
    """Draw a single pose onto a copy of the frame.

    Parameters
    ----------
    threshold : keypoints with score < threshold are skipped (and edges that
        touch them are skipped too).
    draw_face_mesh : if True, connect the 68 face landmarks with their contour
        edges (dense); otherwise only the canonical skeleton edges are drawn
        (which already includes face contour edges from cigpose).
    alpha : blend factor for the overlay (1.0 = opaque, 0.5 = half).
    """
    out = deepcopy(frame)
    overlay = deepcopy(frame)

    kpts = pose.keypoints
    scores = pose.scores
    visible = scores >= threshold

    if draw_skeleton:
        for i, j in SKELETON_133:
            if not (visible[i] and visible[j]):
                continue
            # skip face mesh edges unless explicitly requested (too dense by default)
            if not draw_face_mesh:
                if _region_of_index(i) == "face" and _region_of_index(j) == "face":
                    continue
            color = _keypoint_color(i)
            cv2.line(
                overlay,
                (int(kpts[i, 0]), int(kpts[i, 1])),
                (int(kpts[j, 0]), int(kpts[j, 1])),
                color, line_thickness, cv2.LINE_AA,
            )

    if draw_points:
        for k in range(len(kpts)):
            if not visible[k]:
                continue
            x, y = int(kpts[k, 0]), int(kpts[k, 1])
            # scale radius with confidence, clamped to [2, 2*radius]
            r = max(2, int(point_radius * min(1.0, scores[k] / max(scores.max(), 1e-6))))
            color = _keypoint_color(k)
            cv2.circle(overlay, (x, y), r, color, -1, cv2.LINE_AA)
            cv2.circle(overlay, (x, y), r, (0, 0, 0), 1, cv2.LINE_AA)

    if draw_bbox:
        x1, y1, x2, y2 = [int(v) for v in pose.bbox]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), BBOX_COLOR, 1, cv2.LINE_AA)
        label = f"conf {pose.mean_score():.2f}"
        cv2.rectangle(
            overlay,
            (x1, max(0, y1 - 18)),
            (x1 + 12 * len(label) + 8, max(0, y1 - 2)),
            BBOX_COLOR, -1,
        )
        cv2.putText(overlay, label, (x1 + 4, max(11, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

    if alpha < 1.0:
        out = cv2.addWeighted(overlay, alpha, out, 1.0 - alpha, 0)
    else:
        out = overlay
    return out


def draw_poses(
    frame: np.ndarray,
    poses: list[Pose2D],
    **kwargs,
) -> np.ndarray:
    """Draw multiple poses sequentially onto the frame."""
    out = frame
    for p in poses:
        out = draw_pose(out, p, **kwargs)
    return out


def save_overlay(frame: np.ndarray, poses: list[Pose2D], out_path: str | Path,
                 **kwargs) -> Path:
    """Draw poses onto frame and write the result to ``out_path``.

    Creates the destination directory if needed and asserts the write
    succeeded — OpenCV's imwrite silently returns False on a missing dir,
    which is easy to miss. Always use this instead of a bare cv2.imwrite.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    img = draw_poses(frame, poses, **kwargs)
    ok = cv2.imwrite(str(out), img)
    if not ok:
        raise OSError(f"cv2.imwrite failed to write {out}")
    return out


__all__ = ["draw_pose", "draw_poses", "save_overlay", "REGION_COLORS"]
