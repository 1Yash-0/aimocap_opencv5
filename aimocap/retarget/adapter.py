"""Explicit adapter between COCO-WholeBody (133 keypoints) and Manny Proxy (17 body keypoints).

This module establishes a formal, verified boundary between:
1. Canonical 133-joint WholeBody 3D reconstruction (produced by triangulation and research models).
2. 17-joint Manny body proxy (consumed by reduced kinematic models, baseline solvers, or legacy retargeting).

Source Joint Order (COCO-WholeBody 133):
    0..16:   Body (0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear,
                   5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,
                   9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,
                   13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle)
    17..22:  Feet (17: big_toe_l, 18: small_toe_l, 19: heel_l,
                   20: big_toe_r, 21: small_toe_r, 22: heel_r)
    23..90:  Face (68 facial landmarks)
    91..111: Left Hand (91: wrist + 20 finger joints: thumb, index, middle, ring, pinky)
    112..132: Right Hand (112: wrist + 20 finger joints: thumb, index, middle, ring, pinky)

Target Joint Order (Manny Proxy 17):
    0: nose
    1: left_eye
    2: right_eye
    3: left_ear
    4: right_ear
    5: left_shoulder
    6: right_shoulder
    7: left_elbow
    8: right_elbow
    9: left_wrist
    10: right_wrist
    11: left_hip
    12: right_hip
    13: left_knee
    14: right_knee
    15: left_ankle
    16: right_ankle
"""

from __future__ import annotations

from typing import Tuple
import numpy as np

from aimocap.data.wholebody_layout import BODY_NAMES, N_KEYPOINTS

WHOLEBODY_133_NUM_JOINTS = N_KEYPOINTS  # 133
MANNY_17_NUM_JOINTS = 17
MANNY_17_JOINT_NAMES = tuple(BODY_NAMES)


class WholeBody133ToManny17Adapter:
    """Explicit adapter from COCO-WholeBody 133 joints to Manny 17-joint body proxy.

    Guarantees:
    - Never silently drops joints: fails loudly if input is not 133 joints (unless
      allow_passthrough_17=True for explicit legacy support).
    - Preserves coordinates and dtypes exactly without numerical distortion.
    - Preserves confidence alignment when confidence is provided.
    """

    def __init__(self, *, allow_passthrough_17: bool = False):
        self.allow_passthrough_17 = allow_passthrough_17

    def adapt(
        self,
        skeleton3d: np.ndarray,
        confidence: np.ndarray | None = None,
    ) -> Tuple[np.ndarray, np.ndarray | None]:
        """Convert a 133-joint WholeBody skeleton to the 17-joint Manny body proxy.

        Args:
            skeleton3d: Array of shape (..., J, 3) where J must be 133 (or 17 if
                allow_passthrough_17=True).
            confidence: Optional confidence array of shape (..., J).

        Returns:
            tuple: (adapted_skeleton3d of shape (..., 17, 3),
                    adapted_confidence of shape (..., 17) or None)

        Raises:
            ValueError: If skeleton3d shape, joint dimension, or confidence shape
                violates the contract.
        """
        arr = np.asarray(skeleton3d)
        if arr.ndim < 2:
            raise ValueError(
                f"WholeBody133ToManny17Adapter expected array with ndim >= 2, got shape {arr.shape}"
            )
        if arr.shape[-1] != 3:
            raise ValueError(
                f"WholeBody133ToManny17Adapter expected 3D coordinates (last dimension 3), got shape {arr.shape}"
            )

        num_joints = arr.shape[-2]
        if num_joints == WHOLEBODY_133_NUM_JOINTS:
            adapted_pts = arr[..., :MANNY_17_NUM_JOINTS, :].copy()
        elif num_joints == MANNY_17_NUM_JOINTS:
            if not self.allow_passthrough_17:
                raise ValueError(
                    "WholeBody133ToManny17Adapter received 17 joints but allow_passthrough_17=False. "
                    "Expected authoritative 133-joint WholeBody input."
                )
            adapted_pts = arr.copy()
        else:
            raise ValueError(
                f"WholeBody133ToManny17Adapter expected joint dimension {WHOLEBODY_133_NUM_JOINTS} "
                f"(COCO-WholeBody layout), but received {num_joints} joints in shape {arr.shape}."
            )

        adapted_conf: np.ndarray | None = None
        if confidence is not None:
            conf_arr = np.asarray(confidence)
            if conf_arr.shape != arr.shape[:-1]:
                raise ValueError(
                    f"Confidence shape {conf_arr.shape} does not match skeleton shape {arr.shape[:-1]}"
                )
            if num_joints == WHOLEBODY_133_NUM_JOINTS:
                adapted_conf = conf_arr[..., :MANNY_17_NUM_JOINTS].copy()
            else:
                adapted_conf = conf_arr.copy()

        return adapted_pts, adapted_conf


def wholebody133_to_manny17(
    skeleton3d: np.ndarray,
    confidence: np.ndarray | None = None,
    *,
    allow_passthrough_17: bool = False,
) -> Tuple[np.ndarray, np.ndarray | None]:
    """Convenience function for WholeBody133ToManny17Adapter.adapt."""
    adapter = WholeBody133ToManny17Adapter(allow_passthrough_17=allow_passthrough_17)
    return adapter.adapt(skeleton3d, confidence)


__all__ = [
    "MANNY_17_JOINT_NAMES",
    "MANNY_17_NUM_JOINTS",
    "WHOLEBODY_133_NUM_JOINTS",
    "WholeBody133ToManny17Adapter",
    "wholebody133_to_manny17",
]
