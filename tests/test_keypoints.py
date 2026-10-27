"""Sanity tests for the COCO-WholeBody 133-keypoint layout.

These lock in the invariants the rest of the pipeline depends on:
- exactly 133 keypoints
- region ranges are contiguous, non-overlapping, and sum to 133
- names align with region boundaries
- every skeleton edge references valid indices
- the cigpose-sourced skeleton matches our fallback (source of truth in sync)
"""

from __future__ import annotations

import pytest

from aimocap.pose import keypoints as K


def test_total_keypoint_count():
    assert K.N_KEYPOINTS == 133
    assert len(K.KEYPOINT_NAMES_133) == 133
    assert len(set(K.KEYPOINT_NAMES_133)) == 133, "duplicate keypoint names"


def test_region_counts_sum_to_133():
    total = sum(r.end - r.start for r in K.REGIONS_133.values())
    assert total == 133


def test_region_counts_match_constants():
    body, feet, face, lh, rh = K.REGIONS_133.values()
    assert body.start == 0 and body.end == 17
    assert feet.start == 17 and feet.end == 23
    assert face.start == 23 and face.end == 91
    assert lh.start == 91 and lh.end == 112
    assert rh.start == 112 and rh.end == 133

    assert len(K.BODY_17) == 17
    assert len(K.FEET_6) == 6
    assert len(K.FACE_68) == 68
    assert len(K.LEFT_HAND_21) == 21
    assert len(K.RIGHT_HAND_21) == 21


def test_named_indices_match_names():
    """The named constants must point at the right names."""
    assert K.KEYPOINT_NAMES_133[K.NOSE] == "nose"
    assert K.KEYPOINT_NAMES_133[K.LEFT_EYE] == "left_eye"
    assert K.KEYPOINT_NAMES_133[K.RIGHT_SHOULDER] == "right_shoulder"
    assert K.KEYPOINT_NAMES_133[K.LEFT_WRIST] == "left_wrist"
    assert K.KEYPOINT_NAMES_133[K.RIGHT_ANKLE] == "right_ankle"
    assert K.KEYPOINT_NAMES_133[K.LEFT_BIG_TOE] == "left_big_toe"
    assert K.KEYPOINT_NAMES_133[K.LEFT_SMALL_TOE] == "left_small_toe"
    assert K.KEYPOINT_NAMES_133[K.LEFT_HEEL] == "left_heel"
    assert K.KEYPOINT_NAMES_133[K.RIGHT_BIG_TOE] == "right_big_toe"
    assert K.KEYPOINT_NAMES_133[K.RIGHT_SMALL_TOE] == "right_small_toe"
    assert K.KEYPOINT_NAMES_133[K.RIGHT_HEEL] == "right_heel"
    assert K.KEYPOINT_NAMES_133[K.LEFT_HAND_WRIST] == "left_hand_wrist"
    assert K.KEYPOINT_NAMES_133[K.RIGHT_HAND_WRIST] == "right_hand_wrist"


def test_skeleton_edges_in_range():
    assert len(K.SKELETON_133) > 0
    for i, j in K.SKELETON_133:
        assert 0 <= i < 133, f"edge {i}-{j}: i out of range"
        assert 0 <= j < 133, f"edge {i}-{j}: j out of range"
        assert i != j, f"self-loop at {i}"


def test_skeleton_has_body_core_edges():
    """A handful of anatomically-essential edges must be present."""
    edges = set(K.SKELETON_133)
    essential = [
        (K.LEFT_SHOULDER, K.LEFT_ELBOW),    # 5-7
        (K.LEFT_ELBOW, K.LEFT_WRIST),       # 7-9
        (K.LEFT_HIP, K.LEFT_KNEE),          # 11-13
        (K.LEFT_KNEE, K.LEFT_ANKLE),        # 13-15
        (K.LEFT_HIP, K.RIGHT_HIP),          # 11-12
    ]
    for e in essential:
        assert e in edges, f"missing essential edge {e}"


def test_skeleton_matches_cigpose_source_of_truth():
    """Our SKELETON_133 must equal cigpose's COCO133_SKELETON (the inference lib).

    This is the guard that keeps our drawing in sync with model output. If it
    ever drifts, either update _SKELETON_133_FALLBACK or the import path.
    """
    cigpose_sk = K._import_cigpose_skeleton()
    assert tuple(cigpose_sk) == tuple(K.SKELETON_133)


def test_face_region_has_68_names():
    face_names = [K.KEYPOINT_NAMES_133[i] for i in K.FACE_68]
    assert len(face_names) == 68
    # All face names should be prefixed "face_"
    assert all(n.startswith("face_") for n in face_names)


def test_hand_layout_per_side():
    lh = [K.KEYPOINT_NAMES_133[i] for i in K.LEFT_HAND_21]
    rh = [K.KEYPOINT_NAMES_133[i] for i in K.RIGHT_HAND_21]
    assert lh[0] == "left_hand_wrist"
    assert rh[0] == "right_hand_wrist"
    # thumb: wrist(0), cmc(1), mcp(2), ip(3), tip(4)
    assert lh[3] == "left_thumb_ip"
    assert lh[4] == "left_thumb_tip"
    assert rh[3] == "right_thumb_ip"
    assert rh[4] == "right_thumb_tip"
    # pinky tip is the last hand keypoint
    assert lh[-1] == "left_pinky_tip"
    assert rh[-1] == "right_pinky_tip"
