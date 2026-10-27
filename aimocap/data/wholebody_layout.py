"""Canonical COCO-WholeBody 133-keypoint layout and region definitions.

This module defines the authoritative keypoint layout for COCO-WholeBody format
used throughout the AIMoCap pipeline. All other modules should import from here
instead of hardcoding indices.
"""

import numpy as np

# ── Total count ──────────────────────────────────────────────────────────────
N_KEYPOINTS = 133

# ── Region slices ────────────────────────────────────────────────────────────
REGIONS = {
    "body": slice(0, 17),           # 0-16: nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles
    "feet": slice(17, 23),          # 17-22: big_toe_l, small_toe_l, heel_l, big_toe_r, small_toe_r, heel_r
    "face": slice(23, 91),          # 23-90: 68 face landmarks
    "left_hand": slice(91, 112),    # 91-111: wrist + 20 hand joints (4 per finger × 5)
    "right_hand": slice(112, 133),  # 112-132: wrist + 20 hand joints
}

# ── Semantic names for body keypoints (0-16) ─────────────────────────────────
BODY_NAMES = [
    "nose",           # 0
    "left_eye",       # 1
    "right_eye",      # 2
    "left_ear",       # 3
    "right_ear",      # 4
    "left_shoulder",  # 5
    "right_shoulder", # 6
    "left_elbow",     # 7
    "right_elbow",    # 8
    "left_wrist",     # 9
    "right_wrist",    # 10
    "left_hip",       # 11
    "right_hip",      # 12
    "left_knee",      # 13
    "right_knee",     # 14
    "left_ankle",     # 15
    "right_ankle",    # 16
]

# ── Foot keypoints (17-22) ───────────────────────────────────────────────────
FOOT_NAMES = [
    "big_toe_l",    # 17
    "small_toe_l",  # 18
    "heel_l",       # 19
    "big_toe_r",    # 20
    "small_toe_r",  # 21
    "heel_r",       # 22
]

# ── Face keypoints (23-90): 68 landmarks ─────────────────────────────────────
# Ordered as: face contour (0-16), eyebrows (17-26), eyes (27-36), nose (37-44), lips (45-67)
FACE_NAMES = [
    # Face contour (17 points)
    "face_0", "face_1", "face_2", "face_3", "face_4", "face_5", "face_6", "face_7",
    "face_8", "face_9", "face_10", "face_11", "face_12", "face_13", "face_14", "face_15", "face_16",
    # Left eyebrow (5 points)
    "left_eyebrow_0", "left_eyebrow_1", "left_eyebrow_2", "left_eyebrow_3", "left_eyebrow_4",
    # Right eyebrow (5 points)
    "right_eyebrow_0", "right_eyebrow_1", "right_eyebrow_2", "right_eyebrow_3", "right_eyebrow_4",
    # Left eye (6 points)
    "left_eye_0", "left_eye_1", "left_eye_2", "left_eye_3", "left_eye_4", "left_eye_5",
    # Right eye (6 points)
    "right_eye_0", "right_eye_1", "right_eye_2", "right_eye_3", "right_eye_4", "right_eye_5",
    # Nose (9 points)
    "nose_0", "nose_1", "nose_2", "nose_3", "nose_4", "nose_5", "nose_6", "nose_7", "nose_8",
    # Mouth outer (12 points)
    "mouth_0", "mouth_1", "mouth_2", "mouth_3", "mouth_4", "mouth_5", "mouth_6", "mouth_7", "mouth_8", "mouth_9", "mouth_10", "mouth_11",
    # Mouth inner (8 points)
    "mouth_12", "mouth_13", "mouth_14", "mouth_15", "mouth_16", "mouth_17", "mouth_18", "mouth_19",
]

assert len(FACE_NAMES) == 68, f"FACE_NAMES has {len(FACE_NAMES)} entries, expected 68"

# ── Hand keypoints (91-111 left, 112-132 right): 21 per hand ──────────────────
# Each hand: wrist (0) + thumb(1-4) + index(5-8) + middle(9-12) + ring(13-16) + pinky(17-20)
HAND_FINGER_NAMES = [
    "wrist",
    # Thumb: CMC, MCP, IP, TIP
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    # Index: MCP, PIP, DIP, TIP
    "index_mcp", "index_pip", "index_dip", "index_tip",
    # Middle: MCP, PIP, DIP, TIP
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    # Ring: MCP, PIP, DIP, TIP
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    # Pinky: MCP, PIP, DIP, TIP
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
]

assert len(HAND_FINGER_NAMES) == 21, f"HAND_FINGER_NAMES has {len(HAND_FINGER_NAMES)} entries, expected 21"

# Build full hand names with side prefix
LEFT_HAND_NAMES = [f"left_hand_{n}" for n in HAND_FINGER_NAMES]
RIGHT_HAND_NAMES = [f"right_hand_{n}" for n in HAND_FINGER_NAMES]

# ── Complete name list ───────────────────────────────────────────────────────
ALL_NAMES = BODY_NAMES + FOOT_NAMES + FACE_NAMES + LEFT_HAND_NAMES + RIGHT_HAND_NAMES
assert len(ALL_NAMES) == N_KEYPOINTS, f"Total names: {len(ALL_NAMES)}, expected {N_KEYPOINTS}"

# ── Index lookup ─────────────────────────────────────────────────────────────
NAME_TO_IDX = {name: i for i, name in enumerate(ALL_NAMES)}

# ── Convenience functions ────────────────────────────────────────────────────

def body_index(name: str) -> int:
    """Get index for body keypoint (0-16)."""
    return NAME_TO_IDX[name]

def foot_index(name: str) -> int:
    """Get index for foot keypoint (17-22)."""
    return NAME_TO_IDX[name]


def hand_index(side: str, name: str) -> int:
    """Get index for hand keypoint (91-111 left, 112-132 right).
    
    Args:
        side: "left" or "right"
        name: one of HAND_FINGER_NAMES (e.g., "index_mcp", "wrist")
    """
    prefix = "left_hand_" if side == "left" else "right_hand_"
    return NAME_TO_IDX[prefix + name]

# ── COCO anchor dict for mocap_skeleton.py ───────────────────────────────────
# Maps semantic names used by mocap_skeleton to COCO-WholeBody indices.
# These are the ONLY hardcoded indices in the pipeline — everything else derives from here.
COCO_ANCHORS = {
    # Body (0-16)
    "nose": body_index("nose"),
    "left_eye": body_index("left_eye"),
    "right_eye": body_index("right_eye"),
    "left_ear": body_index("left_ear"),
    "right_ear": body_index("right_ear"),
    "shoulder_l": body_index("left_shoulder"),
    "shoulder_r": body_index("right_shoulder"),
    "elbow_l": body_index("left_elbow"),
    "elbow_r": body_index("right_elbow"),
    "wrist_l": body_index("left_wrist"),
    "wrist_r": body_index("right_wrist"),
    "hip_l": body_index("left_hip"),
    "hip_r": body_index("right_hip"),
    "knee_l": body_index("left_knee"),
    "knee_r": body_index("right_knee"),
    "ankle_l": body_index("left_ankle"),
    "ankle_r": body_index("right_ankle"),
    # Derived
    "pelvis_l": body_index("left_hip"),
    "pelvis_r": body_index("right_hip"),
    "pelvis": body_index("left_hip"),  # alias for root
    "neck": body_index("nose"),        # alias (mid-shoulder computed at runtime)
    # Feet (17-22)
    "big_toe_l": foot_index("big_toe_l"),
    "big_toe_r": foot_index("big_toe_r"),
    "small_toe_l": foot_index("small_toe_l"),
    "small_toe_r": foot_index("small_toe_r"),
    "heel_l": foot_index("heel_l"),
    "heel_r": foot_index("heel_r"),
    # Hand MCPs — available in 133-point data but NOT used for palm frame
    # (Gate 2 verdict: UNUSABLE). Kept here for completeness/future use.
    "hand_wrist_l": hand_index("left", "wrist"),
    "hand_index_mcp_l": hand_index("left", "index_mcp"),
    "hand_middle_mcp_l": hand_index("left", "middle_mcp"),
    "hand_ring_mcp_l": hand_index("left", "ring_mcp"),
    "hand_pinky_mcp_l": hand_index("left", "pinky_mcp"),
    "hand_wrist_r": hand_index("right", "wrist"),
    "hand_index_mcp_r": hand_index("right", "index_mcp"),
    "hand_middle_mcp_r": hand_index("right", "middle_mcp"),
    "hand_ring_mcp_r": hand_index("right", "ring_mcp"),
    "hand_pinky_mcp_r": hand_index("right", "pinky_mcp"),
}

# ── Skeleton edges for visualization (body only, 17 joints) ──────────────────
SKELETON_EDGES_BODY = [
    (0, 1), (0, 2), (1, 3), (2, 4),       # face
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # arms
    (11, 12), (5, 11), (6, 12),           # torso
    (11, 13), (13, 15), (12, 14), (14, 16),  # legs
]

# ── Skeleton edges for hands (21 joints each) ────────────────────────────────
HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # index
    (0, 9), (9, 10), (10, 11), (11, 12),  # middle
    (0, 13), (13, 14), (14, 15), (15, 16), # ring
    (0, 17), (17, 18), (18, 19), (19, 20), # pinky
]

# ── Validation ────────────────────────────────────────────────────────────────



if __name__ == "__main__":
    # Sanity checks
    print(f"Total keypoints: {N_KEYPOINTS}")
    print(f"Regions: {REGIONS}")
    print(f"Body names: {len(BODY_NAMES)}")
    print(f"Foot names: {len(FOOT_NAMES)}")
    print(f"Face names: {len(FACE_NAMES)}")
    print(f"Hand names (per side): {len(HAND_FINGER_NAMES)}")
    print(f"Total names: {len(ALL_NAMES)}")
    print(f"COCO anchors: {len(COCO_ANCHORS)}")
    print("\nSample indices:")
    for name in ["nose", "left_shoulder", "right_wrist", "left_hip", "right_ankle",
                 "big_toe_l", "heel_r", "left_index_mcp", "right_middle_mcp"]:
        print(f"  {name:20s} -> {COCO_ANCHORS.get(name, 'N/A')}")
    print("\nAll checks passed.")