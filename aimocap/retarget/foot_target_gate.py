"""Observability and plausibility checks for ankle-to-big-toe IK targets."""
from __future__ import annotations

import numpy as np


PLAUSIBLE_FOOT_LENGTH_CM = (5.0, 40.0)
FOOT_LENGTH_RATIO_BOUNDS = (0.4, 1.3)


def estimate_foot_gate_references(skel, target_sequence: dict | None = None) -> tuple[dict, dict, dict]:
    """Return per-side reference length/mode and raw-observation masks."""
    references: dict[str, float | None] = {}
    modes: dict[str, str] = {}
    observed: dict[str, dict[str, np.ndarray] | None] = {}
    for side in ("l", "r"):
        ankle = None if target_sequence is None else target_sequence.get(f"ankle_{side}")
        toe = None if target_sequence is None else target_sequence.get(f"big_toe_{side}")
        plausible = np.empty(0, dtype=np.float64)
        mask = None
        if ankle is not None and toe is not None:
            ankle = np.asarray(ankle, dtype=np.float64)
            toe = np.asarray(toe, dtype=np.float64)
            n = min(len(ankle), len(toe))
            ankle_finite = np.isfinite(ankle[:n]).all(axis=1)
            toe_finite = np.isfinite(toe[:n]).all(axis=1)
            finite = ankle_finite & toe_finite
            mask = {"ankle": ankle_finite, "big_toe": toe_finite}
            distance = np.full(n, np.nan, dtype=np.float64)
            distance[finite] = np.linalg.norm(toe[:n][finite] - ankle[:n][finite], axis=1)
            low, high = PLAUSIBLE_FOOT_LENGTH_CM
            plausible = distance[finite & (distance >= low) & (distance <= high)]

        toe_idx = skel.name_to_idx.get(f"toe_{side}")
        if len(plausible) >= 10:
            references[side] = float(np.median(plausible))
            modes[side] = "sequence_median_plausible"
        else:
            rest_length = None
            if toe_idx is not None and hasattr(skel, "bone_lengths"):
                lengths = np.asarray(skel.bone_lengths, dtype=np.float64)
                if toe_idx < len(lengths) and np.isfinite(lengths[toe_idx]) and lengths[toe_idx] > 0.0:
                    rest_length = float(lengths[toe_idx])
            if rest_length is None and toe_idx is not None:
                length = float(np.linalg.norm(skel.rest_offsets[toe_idx]))
                if np.isfinite(length) and length > 0.0:
                    rest_length = length
            references[side] = rest_length
            modes[side] = "mocap_skeleton_rest_length" if rest_length is not None else "direction_only"
        observed[side] = mask
    return references, modes, observed


def assess_foot_target(
    ankle: np.ndarray,
    toe: np.ndarray,
    longitudinal_world: np.ndarray,
    reference_length_cm: float | None,
    reference_mode: str,
    *,
    ankle_observed: bool | None = None,
    toe_observed: bool | None = None,
    foot_gate_reference: str = "solved",
    heel_target: np.ndarray | None = None,
    fps: float | None = None,
    last_accepted_frame: int | None = None,
    last_accepted_direction: np.ndarray | None = None,
    current_frame_idx: int | None = None,
) -> dict:
    """Assess one ankle/toe observation; reports every failed health condition."""
    ankle = np.asarray(ankle, dtype=np.float64)
    toe = np.asarray(toe, dtype=np.float64)
    axis = np.asarray(longitudinal_world, dtype=np.float64)
    finite_ankle = ankle.shape == (3,) and bool(np.isfinite(ankle).all())
    finite_toe = toe.shape == (3,) and bool(np.isfinite(toe).all())
    axis_norm = float(np.linalg.norm(axis)) if axis.shape == (3,) and np.isfinite(axis).all() else 0.0
    failures = []
    if not finite_ankle:
        failures.append("ankle_nonfinite")
    if not finite_toe:
        failures.append("big_toe_nonfinite")
    if ankle_observed is False:
        failures.append("ankle_unobserved")
    if toe_observed is False:
        failures.append("big_toe_unobserved")

    distance = angle = cosine = None
    if finite_ankle and finite_toe:
        direction = toe - ankle
        distance = float(np.linalg.norm(direction))
        if reference_length_cm is not None and reference_mode != "direction_only":
            low = FOOT_LENGTH_RATIO_BOUNDS[0] * reference_length_cm
            high = FOOT_LENGTH_RATIO_BOUNDS[1] * reference_length_cm
            if distance < low:
                failures.append("toe_distance_below_reference_band")
            elif distance > high:
                failures.append("toe_distance_above_reference_band")
        if distance <= 0.0:
            failures.append("toe_direction_degenerate")
        elif axis_norm <= 1e-12 and foot_gate_reference == "solved":
            failures.append("transported_foot_axis_degenerate")
        
        # For observed mode, apply additional checks
        if foot_gate_reference == "observed" and finite_ankle and finite_toe:
            # 2. Heel consistency check
            if heel_target is not None and np.isfinite(heel_target).all():
                heel_direction = toe - heel_target
                heel_norm = float(np.linalg.norm(heel_direction))
                if heel_norm > 1e-12 and distance > 1e-12:
                    heel_dir_norm = heel_direction / heel_norm
                    dir_norm_vec = direction / distance
                    heel_cosine = float(np.clip(np.dot(heel_dir_norm, dir_norm_vec), -1.0, 1.0))
                    heel_angle = float(np.degrees(np.arccos(heel_cosine)))
                    if heel_angle >= 60.0:
                        failures.append("toe_heel_inconsistent")
            
            # 3. Temporal consistency with last accepted observation
            if (last_accepted_frame is not None and last_accepted_direction is not None and 
                fps is not None and current_frame_idx is not None and distance > 1e-12):
                frame_diff = abs(current_frame_idx - last_accepted_frame)
                max_frames = int(round(0.5 * fps)) if fps is not None else 0
                if frame_diff <= max_frames:
                    direction_norm = direction / distance
                    last_dir_norm = last_accepted_direction / float(np.linalg.norm(last_accepted_direction))
                    cosine = float(np.clip(np.dot(direction_norm, last_dir_norm), -1.0, 1.0))
                    angle = float(np.degrees(np.arccos(cosine)))
                    if angle >= 90.0:
                        failures.append("toe_jump_vs_last_accepted")
        
        # Direction check for solved mode only
        if foot_gate_reference == "solved" and axis_norm > 1e-12 and distance > 1e-12:
            cosine = float(np.clip(np.dot(direction / distance, axis / axis_norm), -1.0, 1.0))
            angle = float(np.degrees(np.arccos(cosine)))
            if angle >= 90.0:
                failures.append("toe_direction_antiparallel")
    
    return {
        "healthy": not failures,
        "failed_conditions": failures,
        "distance_cm": distance,
        "reference_length_cm": reference_length_cm,
        "reference_mode": reference_mode,
        "direction_angle_deg": angle,
        "direction_cosine": cosine,
        "ankle_observed": ankle_observed,
        "big_toe_observed": toe_observed,
        "foot_gate_reference": foot_gate_reference,
    }
