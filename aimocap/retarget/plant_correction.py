"""Minimal deterministic planted-foot residual correction.

The correction is deliberately opt-in.  It operates in the existing solver
coordinate system (centimetres, Z-up), changes only local leg rotations, and
never changes root translation.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from aimocap.retarget import foot_lock


_SIDE_NAMES = {"left": "l", "right": "r"}


def _segments(mask: np.ndarray) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(np.asarray(mask, dtype=bool)):
        if value and start is None:
            start = index
        if start is not None and (not value or index == len(mask) - 1):
            result.append((start, index - 1 if not value else index))
            start = None
    return result


def _mad(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if not len(values):
        return 0.0
    median = float(np.median(values))
    return float(np.median(np.abs(values - median)))


def _smoothstep(value: float) -> float:
    x = float(np.clip(value, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def _track(source_targets: Mapping[str, Any], name: str, n: int) -> np.ndarray | None:
    value = source_targets.get(name)
    if value is None:
        return None
    array = np.asarray(value, dtype=float)
    if array.ndim != 2 or array.shape[1] != 3:
        return None
    if len(array) < n:
        padded = np.full((n, 3), np.nan, dtype=float)
        padded[: len(array)] = array
        return padded
    return array[:n].copy()


def _noise_level(tracks: list[np.ndarray]) -> float:
    values: list[np.ndarray] = []
    for track in tracks:
        finite = np.isfinite(track).all(axis=1)
        clean = track[finite]
        if len(clean) >= 3:
            values.append(np.linalg.norm(np.diff(clean, n=2, axis=0), axis=1))
    # 0.05 cm/frame is only a numerical floor.  It prevents exact or nearly
    # exact tracks from making a zero-width threshold while remaining far
    # below ordinary mocap residuals.
    return max((_mad(np.concatenate(values)) if values else 0.0), 0.05)


def _plant_mask(tracks: list[np.ndarray], noise: float, k: float) -> np.ndarray:
    n = len(tracks[0]) if tracks else 0
    if not tracks or n == 0:
        return np.zeros(n, dtype=bool)
    valid = np.ones(n, dtype=bool)
    speed = np.zeros(n, dtype=float)
    for track in tracks:
        finite = np.isfinite(track).all(axis=1)
        valid &= finite
        if n > 1:
            step = np.linalg.norm(np.diff(track, axis=0), axis=1)
            speed[1:] = np.maximum(speed[1:], np.where(np.isfinite(step), step, np.inf))
    return valid & (speed <= float(k) * noise)


def _split_by_drift(mask: np.ndarray, tracks: list[np.ndarray], tolerance: float,
                    min_frames: int) -> list[tuple[int, int]]:
    """Split a low-speed run when conditioned targets leave its anchor."""
    candidates = _segments(mask)
    out: list[tuple[int, int]] = []
    for start, end in candidates:
        sub_start = start
        anchor = np.mean([track[start] for track in tracks], axis=0)
        for frame in range(start + 1, end + 1):
            point = np.mean([track[frame] for track in tracks], axis=0)
            if float(np.linalg.norm(point - anchor)) > tolerance:
                if frame - sub_start >= min_frames:
                    out.append((sub_start, frame - 1))
                sub_start = frame
                anchor = point
        if end - sub_start + 1 >= min_frames:
            out.append((sub_start, end))
    return out


def _fade_weight(frame: int, start: int, end: int, fade_frames: int) -> float:
    if fade_frames <= 0:
        return 1.0
    span = end - start + 1
    edge = min(fade_frames, max(1, span // 2))
    if frame < start + edge:
        return _smoothstep((frame - start + 1) / edge)
    if frame > end - edge:
        return _smoothstep((end - frame + 1) / edge)
    return 1.0


def _bounded_target(current: np.ndarray, anchor: np.ndarray, maximum: float,
                    weight: float) -> np.ndarray:
    delta = np.asarray(anchor, dtype=float) - np.asarray(current, dtype=float)
    distance = float(np.linalg.norm(delta))
    if not np.isfinite(distance) or distance <= 1e-12 or maximum <= 0.0:
        return np.asarray(current, dtype=float).copy()
    delta *= min(1.0, maximum / distance) * float(np.clip(weight, 0.0, 1.0))
    return np.asarray(current, dtype=float) + delta


def _robust_quaternion_mean(quaternions: list[np.ndarray]) -> Rotation | None:
    """Return a sign-invariant, MAD-trimmed Markley quaternion average."""
    values = np.asarray(quaternions, dtype=float).reshape(-1, 4)
    finite = np.isfinite(values).all(axis=1)
    values = values[finite]
    norms = np.linalg.norm(values, axis=1)
    values = values[norms > 1e-12] / norms[norms > 1e-12, None]
    if not len(values):
        return None
    reference = values[0]
    values *= np.where((values @ reference)[:, None] < 0.0, -1.0, 1.0)

    def markley(rows: np.ndarray) -> Rotation:
        scatter = rows.T @ rows
        eigenvalues, eigenvectors = np.linalg.eigh(scatter)
        mean = eigenvectors[:, int(np.argmax(eigenvalues))]
        if float(np.dot(mean, reference)) < 0.0:
            mean = -mean
        return Rotation.from_quat(mean / np.linalg.norm(mean))

    initial = markley(values)
    distances = (initial.inv() * Rotation.from_quat(values)).magnitude()
    median = float(np.median(distances))
    mad = _mad(distances)
    cutoff = median + 3.0 * mad
    keep = distances <= cutoff + 1e-12
    return markley(values[keep])


def _blend_world_rotation(
    skeleton: Any,
    local_quaternions: np.ndarray,
    root_translation: np.ndarray,
    joint_index: int,
    target: Rotation,
    weight: float,
) -> np.ndarray:
    """Blend one joint toward a world-space target without moving its parent."""
    if weight <= 0.0:
        return local_quaternions
    _, world_quaternions = skeleton.get_forward_kinematics(
        local_quaternions, root_translation=root_translation)
    current = Rotation.from_quat(world_quaternions[joint_index])
    delta = current.inv() * target
    blended = current * Rotation.from_rotvec(float(np.clip(weight, 0.0, 1.0)) * delta.as_rotvec())
    parent = skeleton.parents[joint_index]
    if parent < 0:
        local_quaternions[joint_index] = blended.as_quat()
    else:
        parent_world = Rotation.from_quat(world_quaternions[parent])
        local_quaternions[joint_index] = (parent_world.inv() * blended).as_quat()
    return local_quaternions


def correct_planted_feet(
    skeleton: Any,
    animation: np.ndarray,
    source_targets: Mapping[str, Any],
    fps: float,
    *,
    k: float = 3.0,
    min_duration_s: float = 0.1,
    fade_duration_s: float = 0.1,
    max_correction_cm: float = 3.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply a small residual plant correction after IK.

    ``source_targets`` contains conditioned solver targets named
    ``ankle_l/r``, ``heel_l/r`` and ``big_toe_l/r``.  The target tracks detect
    plants; anchors are means of the solved FK ``foot`` and ``ball`` points.
    During each detected plant, foot and ball world rotations are also held at
    their robust (MAD-trimmed Markley) plant means.  This is an orientation
    regularizer only: it uses the captured plant orientation and never assumes
    a flat floor.  Both position and orientation terms use the same edge fade.
    """
    data = np.asarray(animation, dtype=float)
    if data.ndim != 2 or data.shape[1] < 3 or not np.isfinite(data).all():
        return data.copy(), {"enabled": False, "reason": "nonfinite_or_invalid_animation"}
    if not np.isfinite(float(fps)) or fps <= 0.0:
        raise ValueError("fps must be positive and finite")
    n = len(data)
    if n == 0:
        return data.copy(), {"enabled": True, "frames": 0, "plants_detected": 0,
                             "frames_corrected": 0, "skipped_unreachable": 0,
                             "finite": True}
    if k <= 0.0:
        raise ValueError("k must be positive")

    fade_frames = max(1, int(round(float(fade_duration_s) * fps)))
    min_frames = max(2, int(round(float(min_duration_s) * fps)))
    local = Rotation.from_euler("xyz", data[:, 3:].reshape(-1, 3), degrees=False).as_quat()
    local = local.reshape(n, skeleton.num_joints, 4)
    roots = data[:, :3].copy()
    output = data.copy()
    total_plants = 0
    corrected = 0
    skipped_unreachable = 0
    side_reports: dict[str, Any] = {}

    for side, short in _SIDE_NAMES.items():
        tracks = [
            _track(source_targets, f"ankle_{short}", n),
            _track(source_targets, f"heel_{short}", n),
            _track(source_targets, f"big_toe_{short}", n),
        ]
        if any(track is None for track in tracks):
            side_reports[side] = {"noise_level_cm": None, "plants": [], "skipped": "missing_targets"}
            continue
        valid_tracks = [track for track in tracks if track is not None]
        noise = _noise_level(valid_tracks)
        mask = _plant_mask(valid_tracks, noise, k)
        tolerance = max(4.0 * noise, 0.05)
        plants = _split_by_drift(mask, valid_tracks, tolerance, min_frames)
        total_plants += len(plants)
        side_reports[side] = {
            "noise_level_cm_per_frame2": float(noise),
            "speed_threshold_cm_per_frame": float(k * noise),
            "drift_tolerance_cm": float(tolerance),
            "minimum_frames": int(min_frames),
            "plants": [[int(a), int(b)] for a, b in plants],
        }

        foot_idx = skeleton.name_to_idx[f"foot_{short}"]
        ball_idx = skeleton.name_to_idx[f"ball_{short}"]
        thigh_idx = skeleton.name_to_idx[f"thigh_{short}"]
        calf_idx = skeleton.name_to_idx[f"calf_{short}"]
        residuals: list[float] = []
        fk_cache: list[np.ndarray] = []
        world_q_cache: list[np.ndarray] = []
        for frame in range(n):
            positions, world_quaternions = skeleton.get_forward_kinematics(
                local[frame], roots[frame])
            fk_cache.append(positions)
            world_q_cache.append(world_quaternions)
        for start, end in plants:
            foot_mean = _robust_quaternion_mean([
                world_q_cache[frame][foot_idx] for frame in range(start, end + 1)
            ])
            ball_mean = _robust_quaternion_mean([
                world_q_cache[frame][ball_idx] for frame in range(start, end + 1)
            ])
            side_reports[side].setdefault("orientation_hold", []).append({
                "interval": [int(start), int(end)],
                "foot_mean_available": foot_mean is not None,
                "ball_mean_available": ball_mean is not None,
            })
            for frame in range(start, end + 1):
                for index, target_name in ((foot_idx, f"ankle_{short}"),
                                           (ball_idx, f"big_toe_{short}")):
                    target = _track(source_targets, target_name, n)
                    assert target is not None
                    if np.isfinite(target[frame]).all():
                        residuals.append(float(np.linalg.norm(fk_cache[frame][index] - target[frame])))
        residual_p95 = float(np.percentile(residuals, 95)) if residuals else 0.0
        c_max = min(float(max_correction_cm), residual_p95)
        side_reports[side]["residual_p95_cm"] = residual_p95
        side_reports[side]["c_max_cm"] = c_max
        side_reports[side]["cap_limited_samples"] = int(sum(
            value > c_max for value in residuals if np.isfinite(value)
        ))
        side_corrected = 0
        side_skipped = 0
        for start, end in plants:
            foot_anchor = np.mean([fk_cache[frame][foot_idx] for frame in range(start, end + 1)], axis=0)
            ball_anchor = np.mean([fk_cache[frame][ball_idx] for frame in range(start, end + 1)], axis=0)
            for frame in range(start, end + 1):
                weight = _fade_weight(frame, start, end, fade_frames)
                positions, _ = skeleton.get_forward_kinematics(local[frame], roots[frame])
                current_foot = positions[foot_idx]
                current_ball = positions[ball_idx]
                foot_target = _bounded_target(current_foot, foot_anchor, c_max, weight)
                ball_target = _bounded_target(current_ball, ball_anchor, c_max, weight)
                hip = positions[thigh_idx]
                upper = float(np.linalg.norm(positions[calf_idx] - hip))
                lower = float(np.linalg.norm(current_foot - positions[calf_idx]))
                distance = float(np.linalg.norm(foot_target - hip))
                min_reach = abs(upper - lower) + 1e-5
                max_reach = upper + lower - 1e-5
                if (not np.isfinite(distance) or upper <= 1e-8 or lower <= 1e-8
                        or distance < min_reach or distance > max_reach):
                    skipped_unreachable += 1
                    side_skipped += 1
                    continue
                pole = foot_lock._leg_pole_vector(positions, thigh_idx, calf_idx, foot_idx)
                local[frame] = foot_lock._apply_leg_ik_frame(
                    skeleton, local[frame], roots[frame], side,
                    foot_target, ball_target, pole_hint=pole,
                    sole_flatten_weight=0.0,
                )
                if foot_mean is not None:
                    local[frame] = _blend_world_rotation(
                        skeleton, local[frame], roots[frame], foot_idx, foot_mean, weight)
                if ball_mean is not None:
                    local[frame] = _blend_world_rotation(
                        skeleton, local[frame], roots[frame], ball_idx, ball_mean, weight)
                corrected += 1
                side_corrected += 1
        side_reports[side]["frames_corrected"] = int(side_corrected)
        side_reports[side]["skipped_unreachable"] = int(side_skipped)

    output[:, 3:] = Rotation.from_quat(local.reshape(-1, 4)).as_euler(
        "xyz", degrees=False).reshape(n, -1)
    output[:, :3] = data[:, :3]
    if not np.isfinite(output).all():
        return data.copy(), {
            "enabled": True, "frames": int(n), "plants_detected": int(total_plants),
            "frames_corrected": 0, "skipped_unreachable": int(skipped_unreachable),
            "finite": False, "finite_fallback": True, "sides": side_reports,
        }
    return output, {
        "enabled": True, "frames": int(n), "plants_detected": int(total_plants),
        "frames_corrected": int(corrected), "skipped_unreachable": int(skipped_unreachable),
        "finite": True, "finite_fallback": False, "fade_frames": int(fade_frames),
        "minimum_frames": int(min_frames), "k": float(k), "sides": side_reports,
    }
