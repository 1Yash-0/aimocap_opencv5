"""COCO-WholeBody 133-keypoint layout — the canonical reference for the pipeline.

Single source of truth for keypoint indices, names, regions, and skeleton edges.
Every downstream stage (triangulation, retargeting, export) keys off these
indices, so they must be stable and correct.

Layout (COCO-WholeBody spec, https://github.com/jin-s13/COCO-WholeBody):

    0–16    body       (17)  nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles
    17–22   feet       ( 6)  L/R big-toe, small-toe, heel (3 left, 3 right)
    23–90   face       (68)  standard 68-point face contour
    91–111  left hand  (21)  wrist -> thumb/index/middle/ring/pinky (4 joints each)
    112–132 right hand (21)  mirrored

Skeleton edges (SKELETON_133) are sourced from cigpose.inference.COCO133_SKELETON
so they are guaranteed consistent with the inference output of our CIGPose model.
"""

from __future__ import annotations

from typing import NamedTuple
from dataclasses import dataclass
import numpy as np


@dataclass(slots=True)
class Pose2D:
    """One person's 2D pose in a single frame.

    keypoints: (133, 2) float32, pixel coords in the original frame.
    scores:    (133,)   float32, per-keypoint confidence in [0, ~1].
    bbox:      (4,)     float32, [x1, y1, x2, y2] person box in pixels.
    """

    keypoints: np.ndarray
    scores: np.ndarray
    bbox: np.ndarray

    def __post_init__(self) -> None:
        if self.keypoints.shape != (133, 2):
            raise ValueError(
                f"keypoints shape {self.keypoints.shape}, expected (133, 2)"
            )
        if self.scores.shape != (133,):
            raise ValueError(
                f"scores shape {self.scores.shape}, expected (133,)"
            )
        if self.bbox.shape != (4,):
            raise ValueError(f"bbox shape {self.bbox.shape}, expected (4,)")

    @property
    def area(self) -> float:
        """Bounding-box area in pixels^2 (used to pick the largest person)."""
        x1, y1, x2, y2 = self.bbox
        return float(max(0.0, x2 - x1) * max(0.0, y2 - y1))


class Region(NamedTuple):
    """A contiguous slice of the 133-vector belonging to one body region."""

    name: str
    start: int  # inclusive
    end: int    # exclusive


# Region index ranges — must sum to 133.
REGIONS_133: dict[str, Region] = {
    "body":       Region("body",       0, 17),
    "feet":       Region("feet",      17, 23),
    "face":       Region("face",      23, 91),
    "left_hand":  Region("left_hand",  91, 112),
    "right_hand": Region("right_hand", 112, 133),
}

BODY_17 = tuple(range(0, 17))
FEET_6 = tuple(range(17, 23))
FACE_68 = tuple(range(23, 91))
LEFT_HAND_21 = tuple(range(91, 112))
RIGHT_HAND_21 = tuple(range(112, 133))


# ---------------------------------------------------------------------------
# Canonical names — built region by region, total 133.
# ---------------------------------------------------------------------------

_BODY = [
    "nose",                                  # 0
    "left_eye", "right_eye",                 # 1, 2
    "left_ear", "right_ear",                 # 3, 4
    "left_shoulder", "right_shoulder",       # 5, 6
    "left_elbow", "right_elbow",             # 7, 8
    "left_wrist", "right_wrist",             # 9, 10
    "left_hip", "right_hip",                 # 11, 12
    "left_knee", "right_knee",               # 13, 14
    "left_ankle", "right_ankle",             # 15, 16
]

# COCO-WholeBody foot keypoints (17–22): three per foot.
# Official order used by COCO-WholeBody/MMPose:
# left big toe, left small toe, left heel, right big toe, right small toe,
# right heel.  Older comments in this project called the middle point
# "eye_toe"; keep the indices but correct the canonical names so downstream
# leg/foot orientation code can build real foot frames.
_FEET = [
    "left_big_toe", "left_small_toe", "left_heel",       # 17–19
    "right_big_toe", "right_small_toe", "right_heel",    # 20–22
]

# 68-point face landmarks (23–90), following the standard iBUG 68-point layout
# that COCO-WholeBody adopts:
#   jaw 17, brows 10 (5 each), nose 9 (4 bridge + 5 tip/ala), eyes 12 (6 each),
#   mouth 20 (12 outer + 8 inner). 17+10+9+12+20 = 68.
_FACE = (
    [f"face_jaw_{i}" for i in range(17)]               # 23–39  jaw outline (17)
    + [f"face_rbrow_{i}" for i in range(5)]            # 40–44  right eyebrow (5)
    + [f"face_lbrow_{i}" for i in range(5)]            # 45–49  left eyebrow (5)
    + [f"face_nose_bridge_{i}" for i in range(4)]      # 50–53  nose bridge (4)
    + [f"face_nose_tip_{i}" for i in range(5)]         # 54–58  nose tip + alae (5)
    + [f"face_reye_{i}" for i in range(6)]             # 59–64  right eye (6)
    + [f"face_leye_{i}" for i in range(6)]             # 65–70  left eye (6)
    + [f"face_mouth_outer_{i}" for i in range(12)]     # 71–82  outer mouth (12)
    + [f"face_mouth_inner_{i}" for i in range(8)]      # 83–90  inner mouth (8)
)
assert len(_FACE) == 68, f"face layout != 68: got {len(_FACE)}"

# 21-point hand per side (left 91–111, right 112–132).
# Layout: wrist, then 4 joints per finger (cmc/mcp/ip/tip for thumb,
# mcp/pip/dip/tip for the others).
# NOTE: the hand "wrist" keypoint (idx 91/112) is physically the same joint as
# the body wrist (idx 9/10), but we give it a distinct name to keep all 133
# names unique (downstream code keys off names for retargeting).
def _hand(side: str) -> list[str]:
    return (
        [f"{side}_hand_wrist"]
        + [f"{side}_thumb_cmc", f"{side}_thumb_mcp", f"{side}_thumb_ip", f"{side}_thumb_tip"]
        + [f"{side}_{f}_{j}"
           for f in ("index", "middle", "ring", "pinky")
           for j in ("mcp", "pip", "dip", "tip")]
    )

_LEFT_HAND = _hand("left")    # 91–111
_RIGHT_HAND = _hand("right")  # 112–132


def _build_names() -> tuple[str, ...]:
    names = _BODY + _FEET + _FACE + _LEFT_HAND + _RIGHT_HAND
    if len(names) != 133:
        raise AssertionError(
            f"keypoint name count != 133: body={len(_BODY)} feet={len(_FEET)} "
            f"face={len(_FACE)} lh={len(_LEFT_HAND)} rh={len(_RIGHT_HAND)} "
            f"total={len(names)}"
        )
    return tuple(names)


KEYPOINT_NAMES_133: tuple[str, ...] = _build_names()


# ---------------------------------------------------------------------------
# Skeleton connectivity — sourced from cigpose so it matches inference output.
# ---------------------------------------------------------------------------

# Hand-maintained fallback. SKELETON_133 below is normally loaded from cigpose;
# this stays in sync with cigpose.inference.COCO133_SKELETON.
_SKELETON_133_FALLBACK: tuple[tuple[int, int], ...] = (
    # body
    (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    # feet
    (15, 17), (15, 18), (15, 19), (16, 20), (16, 21), (16, 22),
    # wrists -> hand wrists
    (9, 91), (10, 112),
    # left hand
    (91, 92), (92, 93), (93, 94), (94, 95),
    (91, 96), (96, 97), (97, 98), (98, 99),
    (91, 100), (100, 101), (101, 102), (102, 103),
    (91, 104), (104, 105), (105, 106), (106, 107),
    (91, 108), (108, 109), (109, 110), (110, 111),
    # right hand
    (112, 113), (113, 114), (114, 115), (115, 116),
    (112, 117), (117, 118), (118, 119), (119, 120),
    (112, 121), (121, 122), (122, 123), (123, 124),
    (112, 125), (125, 126), (126, 127), (127, 128),
    (112, 129), (129, 130), (130, 131), (131, 132),
)


def _import_cigpose_skeleton() -> tuple[tuple[int, int], ...]:
    """Load COCO133_SKELETON from cigpose; fall back to the bundled copy."""
    try:
        from cigpose.inference import COCO133_SKELETON  # type: ignore
        return tuple(COCO133_SKELETON)
    except ImportError:
        return _SKELETON_133_FALLBACK


SKELETON_133: tuple[tuple[int, int], ...] = _import_cigpose_skeleton()


# ---------------------------------------------------------------------------
# Named indices for the most-used body keypoints.
# ---------------------------------------------------------------------------

NOSE = 0
LEFT_EYE, RIGHT_EYE = 1, 2
LEFT_EAR, RIGHT_EAR = 3, 4
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_ELBOW, RIGHT_ELBOW = 7, 8
LEFT_WRIST, RIGHT_WRIST = 9, 10
LEFT_HIP, RIGHT_HIP = 11, 12
LEFT_KNEE, RIGHT_KNEE = 13, 14
LEFT_ANKLE, RIGHT_ANKLE = 15, 16
LEFT_BIG_TOE, LEFT_SMALL_TOE, LEFT_HEEL = 17, 18, 19
RIGHT_BIG_TOE, RIGHT_SMALL_TOE, RIGHT_HEEL = 20, 21, 22
LEFT_HAND_WRIST = 91
RIGHT_HAND_WRIST = 112
# COCO-17 has no pelvis keypoint; the 3D stage synthesizes one as the midpoint
# of the hips (indices 11, 12).
PELVIS_MID = None

N_KEYPOINTS = 133


__all__ = [
    "KEYPOINT_NAMES_133",
    "SKELETON_133",
    "REGIONS_133",
    "Region",
    "BODY_17", "FEET_6", "FACE_68", "LEFT_HAND_21", "RIGHT_HAND_21",
    "N_KEYPOINTS",
    "NOSE", "LEFT_EYE", "RIGHT_EYE", "LEFT_EAR", "RIGHT_EAR",
    "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW",
    "LEFT_WRIST", "RIGHT_WRIST", "LEFT_HIP", "RIGHT_HIP",
    "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE",
    "LEFT_BIG_TOE", "LEFT_SMALL_TOE", "LEFT_HEEL",
    "RIGHT_BIG_TOE", "RIGHT_SMALL_TOE", "RIGHT_HEEL",
    "PELVIS_MID", "LEFT_HAND_WRIST", "RIGHT_HAND_WRIST",
]
