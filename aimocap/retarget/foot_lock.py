from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.spatial.transform import Rotation

from aimocap.retarget.fbx_rig import Skeleton


FOOT_CHAINS = {
    "left": {
        "thigh": "thigh_l",
        "calf": "calf_l",
        "foot": "foot_l",
        "ball": "ball_l",
        "source_idxs": (17, 18, 19),
    },
    "right": {
        "thigh": "thigh_r",
        "calf": "calf_r",
        "foot": "foot_r",
        "ball": "ball_r",
        "source_idxs": (20, 21, 22),
    },
}

IMAGE_FOOT_IDXS = {
    "left": (15, 17, 18, 19),
    "right": (16, 20, 21, 22),
}


def _zero_phase_filter_positions(
    positions: np.ndarray,
    fps: float,
    cutoff_hz: float = 5.0,
    order: int = 2,
) -> np.ndarray:
    if len(positions) < 9:
        return positions.copy()

    cutoff = min(cutoff_hz / (0.5 * fps), 0.99)
    sos = butter(order, cutoff, output="sos")
    out = positions.copy()
    padlen = min(15, len(positions) - 1)
    for dim in range(positions.shape[1]):
        try:
            out[:, dim] = sosfiltfilt(sos, positions[:, dim], padlen=padlen)
        except ValueError:
            out[:, dim] = positions[:, dim]
    return out


def _interpolate_missing_rows(pos: np.ndarray) -> np.ndarray:
    """Linearly interpolate NaN rows per dimension (in place-safe copy).

    Columns with no finite values are left NaN. Used so a few unknown
    frames do not poison zero-phase filtering of the whole track, while
    callers still treat fully-unknown frames as invalid.
    """
    out = np.asarray(pos, dtype=float).copy()
    n = len(out)
    for dim in range(out.shape[1]):
        col = out[:, dim]
        ok = np.isfinite(col)
        if ok.all() or not ok.any():
            continue
        out[:, dim] = np.interp(np.arange(n), np.flatnonzero(ok), col[ok])
    return out


def _segments(mask: np.ndarray) -> list[tuple[int, int]]:
    found: list[tuple[int, int]] = []
    start = None
    for i, value in enumerate(mask):
        if value and start is None:
            start = i
        if start is not None and ((not value) or i == len(mask) - 1):
            end = i - 1 if not value else i
            found.append((start, end))
            start = None
    return found


def _fill_short_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    out = mask.copy()
    false_segments = _segments(~out)
    for start, end in false_segments:
        if start == 0 or end == len(out) - 1:
            continue
        if end - start + 1 <= max_gap:
            out[start:end + 1] = True
    return out


def _remove_short_segments(mask: np.ndarray, min_len: int) -> np.ndarray:
    out = mask.copy()
    for start, end in _segments(out):
        if end - start + 1 < min_len:
            out[start:end + 1] = False
    return out


def _smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _build_contact_strength(
    mask: np.ndarray,
    blend_frames: int,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    segments = _segments(mask)
    strength = np.zeros(len(mask), dtype=float)
    for start, end in segments:
        strength[start:end + 1] = 1.0
        span = end - start + 1
        edge = min(blend_frames, max(2, int(round(span / 3.0))))
        ramp = _smoothstep(np.linspace(0.0, 1.0, edge))
        strength[start:start + edge] = np.minimum(strength[start:start + edge], ramp)
        strength[end - edge + 1:end + 1] = np.minimum(
            strength[end - edge + 1:end + 1],
            ramp[::-1],
        )
    return strength, segments


def _limit_xy_delta(sequence: np.ndarray, max_delta: float) -> np.ndarray:
    if len(sequence) < 2 or max_delta <= 0.0:
        return sequence.copy()
    out = sequence.copy()
    for frame in range(1, len(out)):
        delta = out[frame, :2] - out[frame - 1, :2]
        norm = np.linalg.norm(delta)
        if norm > max_delta:
            out[frame, :2] = out[frame - 1, :2] + delta * (max_delta / norm)
    for frame in range(len(out) - 2, -1, -1):
        delta = out[frame, :2] - out[frame + 1, :2]
        norm = np.linalg.norm(delta)
        if norm > max_delta:
            out[frame, :2] = out[frame + 1, :2] + delta * (max_delta / norm)
    return out


def detect_foot_contacts(
    source_pts3d: np.ndarray,
    fps: float = 30.0,
    min_segment_frames: int = 6,
    max_gap_frames: int = 4,
    blend_frames: int = 5,
) -> dict[str, dict[str, np.ndarray | list[tuple[int, int]] | float]]:
    """Detect planted intervals from source toe clusters in Z-up centimeters."""
    contacts: dict[str, dict[str, np.ndarray | list[tuple[int, int]] | float]] = {}

    for side, spec in FOOT_CHAINS.items():
        idxs = spec["source_idxs"]
        if source_pts3d.shape[1] <= max(idxs):
            # Fallback to ankle if toes are missing (e.g. COCO-17)
            ankle_idx = 15 if side == "left" else 16
            toe_pos = source_pts3d[:, ankle_idx, :]
        else:
            # WP2.A: mean over finite cluster members per frame (a NaN heel
            # no longer poisons the whole cluster into NaN ground).
            cluster = np.asarray(source_pts3d[:, idxs, :], dtype=float)
            summed = np.nansum(cluster, axis=1)
            counts = np.sum(np.isfinite(cluster), axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                toe_pos = summed / np.maximum(counts, 1)
            toe_pos[counts == 0] = np.nan
        # Frames with no observation at all stay invalid: interpolate them
        # for filter stability, but never claim stance on them below.
        invalid_frames = ~np.isfinite(toe_pos).all(axis=1)
        toe_pos = _interpolate_missing_rows(toe_pos)
        toe_smooth = _zero_phase_filter_positions(toe_pos, fps, cutoff_hz=5.0)

        height = toe_smooth[:, 2]
        horiz_speed = np.zeros(len(toe_smooth))
        if len(toe_smooth) > 1:
            horiz_speed[1:] = np.linalg.norm(np.diff(toe_smooth[:, :2], axis=0), axis=1) * fps

        ground = float(np.percentile(height, 8))
        height_range = float(np.percentile(height, 80) - ground)
        height_band = max(3.0, 0.25 * height_range)
        speed_threshold = max(12.0, float(np.percentile(horiz_speed, 40)))

        mask = (height <= ground + height_band) & (horiz_speed <= speed_threshold)
        mask = _fill_short_gaps(mask, max_gap_frames)
        mask = mask & ~invalid_frames
        mask = _remove_short_segments(mask, min_segment_frames)
        strength, segments = _build_contact_strength(mask, blend_frames)

        contacts[side] = {
            "mask": mask,
            "strength": strength,
            "segments": segments,
            "ground_z": ground,
            "height_band": float(height_band),
            "speed_threshold": float(speed_threshold),
            "invalid_frames": int(np.sum(invalid_frames)),
        }

    return contacts


def _interpolate_missing_points(points: np.ndarray, valid: np.ndarray) -> np.ndarray | None:
    valid_idx = np.flatnonzero(valid)
    if len(valid_idx) < 3:
        return None

    frames = np.arange(len(points))
    out = points.copy()
    for dim in range(points.shape[1]):
        out[:, dim] = np.interp(frames, valid_idx, points[valid_idx, dim])
    return out


def _camera_body_scales(
    keypoints: np.ndarray,
    scores: np.ndarray,
    image_sizes: np.ndarray | None,
) -> np.ndarray:
    scales = np.zeros(keypoints.shape[1], dtype=float)
    body_idxs = np.arange(min(17, keypoints.shape[2]))
    for cam in range(keypoints.shape[1]):
        shoulder = 0.5 * (keypoints[:, cam, 5] + keypoints[:, cam, 6])
        hip = 0.5 * (keypoints[:, cam, 11] + keypoints[:, cam, 12])
        torso_ok = np.all(scores[:, cam, [5, 6, 11, 12]] > 0.3, axis=1)
        torso = np.linalg.norm(shoulder - hip, axis=1)
        scale = float(np.median(torso[torso_ok & np.isfinite(torso) & (torso > 1.0)]))
        if not np.isfinite(scale) or scale <= 1.0:
            body_scores = scores[:, cam, body_idxs]
            body_points = keypoints[:, cam, body_idxs]
            frame_scales = []
            for frame in range(len(keypoints)):
                ok = body_scores[frame] > 0.3
                if np.count_nonzero(ok) < 4:
                    continue
                span = np.ptp(body_points[frame, ok], axis=0)
                frame_scales.append(float(np.linalg.norm(span)))
            scale = float(np.median(frame_scales)) if frame_scales else 0.0
        if (not np.isfinite(scale) or scale <= 1.0) and image_sizes is not None:
            width, height = image_sizes[cam]
            scale = float(np.hypot(width, height) * 0.12)
        scales[cam] = max(scale, 1.0)
    return scales


def _weighted_percentile(values: np.ndarray, weights: np.ndarray, percentile: float) -> float:
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(valid):
        return float("nan")
    vals = values[valid]
    wts = weights[valid]
    order = np.argsort(vals)
    vals = vals[order]
    wts = wts[order]
    cutoff = np.sum(wts) * percentile / 100.0
    return float(vals[min(np.searchsorted(np.cumsum(wts), cutoff), len(vals) - 1)])


def detect_image_foot_contacts(
    pose2d_npz: str | Path,
    num_frames: int,
    fps: float = 30.0,
    min_segment_frames: int = 8,
    max_gap_frames: int = 14,
    blend_frames: int = 5,
    min_camera_confidence: float = 0.35,
) -> dict[str, dict[str, np.ndarray | list[tuple[int, int]] | float]]:
    """Detect stationary feet from multi-camera 2D observations.

    This is intentionally used only as a contact hint. The final lock is still
    solved in target-skeleton space, but 2D contact is less vulnerable to the
    depth wobble that makes planted 3D feet appear to slide.
    """
    data = np.load(pose2d_npz)
    keypoints = data["keypoints"][:num_frames].astype(float)
    scores = data["scores"][:num_frames].astype(float)
    image_sizes = data["image_sizes"] if "image_sizes" in data else None

    frames, cameras = keypoints.shape[:2]
    scales = _camera_body_scales(keypoints, scores, image_sizes)
    contacts: dict[str, dict[str, np.ndarray | list[tuple[int, int]] | float]] = {}

    for side, idxs in IMAGE_FOOT_IDXS.items():
        per_camera_speed = np.full((frames, cameras), np.nan, dtype=float)
        per_camera_conf = np.zeros((frames, cameras), dtype=float)
        per_camera_ground = np.zeros((frames, cameras), dtype=bool)
        for cam in range(cameras):
            foot_scores = scores[:, cam, idxs]
            foot_points = keypoints[:, cam, idxs]
            visible = foot_scores > 0.2
            valid = np.count_nonzero(visible, axis=1) >= 2
            weighted = np.zeros((frames, 2), dtype=float)
            conf = np.zeros(frames, dtype=float)
            for frame in range(frames):
                ok = visible[frame]
                if not valid[frame]:
                    continue
                weights = np.clip(foot_scores[frame, ok], 1e-3, None)
                weighted[frame] = np.average(foot_points[frame, ok], axis=0, weights=weights)
                conf[frame] = float(np.mean(foot_scores[frame, ok]))

            interp = _interpolate_missing_points(weighted, valid)
            if interp is None:
                continue
            smooth = _zero_phase_filter_positions(interp, fps=fps, cutoff_hz=2.0)
            speed = np.zeros(frames, dtype=float)
            if frames > 1:
                speed[1:] = (
                    np.linalg.norm(np.diff(smooth, axis=0), axis=1)
                    * fps
                    / scales[cam]
                )
            y = smooth[:, 1]
            y_valid = y[valid]
            if len(y_valid) > 0:
                image_ground_y = float(np.percentile(y_valid, 90))
                lift_range = max(0.0, image_ground_y - float(np.percentile(y_valid, 10)))
                image_height_band = max(0.08 * scales[cam], 0.35 * lift_range)
                per_camera_ground[:, cam] = y >= image_ground_y - image_height_band
            per_camera_speed[:, cam] = speed
            per_camera_conf[:, cam] = conf

        image_speed = np.full(frames, np.inf, dtype=float)
        image_conf = np.zeros(frames, dtype=float)
        image_ground = np.zeros(frames, dtype=bool)
        for frame in range(frames):
            cam_ok = (
                np.isfinite(per_camera_speed[frame])
                & (per_camera_conf[frame] >= min_camera_confidence)
            )
            if not np.any(cam_ok):
                continue
            ground_ok = cam_ok & per_camera_ground[frame]
            speed_ok = ground_ok if np.any(ground_ok) else cam_ok
            weights = np.square(per_camera_conf[frame, speed_ok])
            image_speed[frame] = _weighted_percentile(
                per_camera_speed[frame, speed_ok],
                weights,
                35.0,
            )
            image_conf[frame] = float(np.max(per_camera_conf[frame, speed_ok]))
            image_ground[frame] = bool(np.any(ground_ok))

        finite_speed = image_speed[np.isfinite(image_speed)]
        if len(finite_speed) == 0:
            mask = np.zeros(frames, dtype=bool)
            threshold = float("inf")
        else:
            threshold = float(np.clip(np.percentile(finite_speed, 55), 0.12, 0.35))
            enter_threshold = threshold
            stay_threshold = min(0.60, threshold * 1.9)
            mask = np.zeros(frames, dtype=bool)
            active = False
            for frame in range(frames):
                if not np.isfinite(image_speed[frame]):
                    active = False
                elif active:
                    active = image_speed[frame] <= stay_threshold
                else:
                    active = image_speed[frame] <= enter_threshold
                mask[frame] = active

        mask = _fill_short_gaps(mask, max_gap_frames)
        mask = _remove_short_segments(mask, min_segment_frames)
        strength, segments = _build_contact_strength(mask, blend_frames)

        contacts[side] = {
            "mask": mask,
            "strength": strength,
            "segments": segments,
            "speed_threshold": threshold,
            "image_speed": image_speed,
            "image_confidence": image_conf,
            "image_ground": image_ground,
        }

    return contacts


def detect_target_foot_contacts(
    skel: Skeleton,
    target_positions: np.ndarray,
    fps: float = 30.0,
    source_contacts: dict[str, dict[str, np.ndarray | list[tuple[int, int]] | float]] | None = None,
    image_contacts: dict[str, dict[str, np.ndarray | list[tuple[int, int]] | float]] | None = None,
    min_segment_frames: int = 6,
    max_gap_frames: int = 4,
    blend_frames: int = 5,
) -> dict[str, dict[str, np.ndarray | list[tuple[int, int]] | float]]:
    """Detect stance from the target rig's sole height and vertical stability."""
    contacts: dict[str, dict[str, np.ndarray | list[tuple[int, int]] | float]] = {}

    for side, spec in FOOT_CHAINS.items():
        foot_idx = skel.name_to_idx[spec["foot"]]
        ball_idx = skel.name_to_idx[spec["ball"]]

        foot_pos = target_positions[:, foot_idx]
        ball_pos = target_positions[:, ball_idx]
        sole_z = np.minimum(foot_pos[:, 2], ball_pos[:, 2])

        foot_vz = np.zeros(len(target_positions))
        ball_vz = np.zeros(len(target_positions))
        if len(target_positions) > 1:
            foot_vz[1:] = np.abs(np.diff(foot_pos[:, 2])) * fps
            ball_vz[1:] = np.abs(np.diff(ball_pos[:, 2])) * fps
        vertical_speed = np.minimum(foot_vz, ball_vz)

        ground = float(np.percentile(sole_z, 5))
        height_band = max(3.0, float(np.percentile(sole_z, 25) - ground))
        if image_contacts is not None:
            height_band = max(height_band, 5.0, float(np.percentile(sole_z, 40) - ground))
        vertical_threshold = max(12.0, float(np.percentile(vertical_speed, 45)))

        near_ground = sole_z <= ground + height_band
        target_mask = near_ground & (vertical_speed <= vertical_threshold)
        if image_contacts is not None:
            image_mask = image_contacts[side]["mask"]
            assert isinstance(image_mask, np.ndarray)
            target_mask = image_mask[:len(target_positions)]
            max_gap_frames = max(max_gap_frames, 14)
            min_segment_frames = max(min_segment_frames, 8)
            blend_frames = max(blend_frames, int(round(0.4 * fps)))
        elif source_contacts is not None:
            source_mask = source_contacts[side]["mask"]
            assert isinstance(source_mask, np.ndarray)
            target_mask = target_mask | source_mask

        target_mask = _fill_short_gaps(target_mask, max_gap_frames)
        target_mask = _remove_short_segments(target_mask, min_segment_frames)
        strength, segments = _build_contact_strength(target_mask, blend_frames)

        contacts[side] = {
            "mask": target_mask,
            "strength": strength,
            "segments": segments,
            "ground_z": ground,
            "height_band": float(height_band),
            "speed_threshold": float(vertical_threshold),
        }

    return contacts


def _fk_sequence(
    skel: Skeleton,
    root_seq: np.ndarray,
    local_quats: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    all_pos = np.zeros((len(root_seq), skel.num_joints, 3))
    all_rot = np.zeros((len(root_seq), skel.num_joints, 4))
    for frame in range(len(root_seq)):
        pos, rot = skel.get_forward_kinematics(local_quats[frame], root_translation=root_seq[frame])
        all_pos[frame] = pos
        all_rot[frame] = rot
    return all_pos, all_rot


def _rotation_between_vectors(src: np.ndarray, dst: np.ndarray) -> Rotation:
    """Shortest-arc rotation src -> dst, identity when undefined.

    Undefined means degenerate (near-zero) OR non-finite input. The finite
    test must be positive-form: NaN comparisons are False, so a `norm <
    1e-8` guard alone lets NaN through to align_vectors, which then returns
    zero-norm quaternions and raises. Callers degrade to "leave this frame's
    pose unchanged", which is the fail-safe for an unusable target.
    """
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    if not (np.isfinite(src).all() and np.isfinite(dst).all()):
        return Rotation.identity()
    src_norm = float(np.linalg.norm(src))
    dst_norm = float(np.linalg.norm(dst))
    if not (src_norm >= 1e-8 and dst_norm >= 1e-8):
        return Rotation.identity()
    try:
        rot, _ = Rotation.align_vectors(
            (dst / dst_norm).reshape(1, 3),
            (src / src_norm).reshape(1, 3),
        )
    except Exception:
        rot = Rotation.identity()
    quat = np.asarray(rot.as_quat(), dtype=float)
    if not np.isfinite(quat).all() or float(np.linalg.norm(quat)) < 1e-8:
        return Rotation.identity()
    return Rotation.from_quat(quat / np.linalg.norm(quat))


def _solve_knee_position(
    hip: np.ndarray,
    knee: np.ndarray,
    ankle_target: np.ndarray,
    upper_len: float,
    lower_len: float,
    pole_hint: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    to_target = ankle_target - hip
    dist = np.linalg.norm(to_target)
    if dist < 1e-8:
        return knee, ankle_target

    max_reach = max(upper_len + lower_len - 1e-5, 1e-5)
    min_reach = abs(upper_len - lower_len) + 1e-5
    clamped_dist = float(np.clip(dist, min_reach, max_reach))
    direction = to_target / dist
    ankle_eff = hip + direction * clamped_dist

    pole = None
    if pole_hint is not None:
        hinted = pole_hint - direction * np.dot(pole_hint, direction)
        if np.linalg.norm(hinted) > 1e-6:
            pole = hinted

    if pole is None:
        pole = knee - (hip + direction * np.dot(knee - hip, direction))
    pole_norm = np.linalg.norm(pole)
    if pole_norm < 1e-6:
        up = np.array([0.0, 0.0, 1.0])
        pole = np.cross(direction, up)
        pole_norm = np.linalg.norm(pole)
        if pole_norm < 1e-6:
            pole = np.array([0.0, 1.0, 0.0])
            pole_norm = 1.0
    pole /= pole_norm

    x = (upper_len * upper_len - lower_len * lower_len + clamped_dist * clamped_dist) / (2.0 * clamped_dist)
    h2 = max(upper_len * upper_len - x * x, 0.0)
    knee_eff = hip + direction * x + pole * np.sqrt(h2)
    return knee_eff, ankle_eff


def _apply_leg_ik_frame(
    skel: Skeleton,
    local_quats: np.ndarray,
    root_t: np.ndarray,
    side: str,
    foot_target: np.ndarray,
    ball_target: np.ndarray,
    pole_hint: np.ndarray | None = None,
    sole_flatten_weight: float = 0.0,
) -> np.ndarray:
    spec = FOOT_CHAINS[side]
    thigh = skel.name_to_idx[spec["thigh"]]
    calf = skel.name_to_idx[spec["calf"]]
    foot = skel.name_to_idx[spec["foot"]]
    ball = skel.name_to_idx[spec["ball"]]

    pos, global_quats = skel.get_forward_kinematics(local_quats, root_translation=root_t)
    global_rots = Rotation.from_quat(global_quats)
    out_quats = local_quats.copy()

    hip = pos[thigh]
    knee = pos[calf]
    ankle = pos[foot]
    upper_len = np.linalg.norm(knee - hip)
    lower_len = np.linalg.norm(ankle - knee)
    knee_target, ankle_target = _solve_knee_position(
        hip,
        knee,
        foot_target,
        upper_len,
        lower_len,
        pole_hint=pole_hint,
    )

    thigh_vec_current = knee - hip
    thigh_vec_target = knee_target - hip
    thigh_global = _rotation_between_vectors(thigh_vec_current, thigh_vec_target) * global_rots[thigh]
    thigh_parent = skel.parents[thigh]
    if thigh_parent == -1:
        out_quats[thigh] = thigh_global.as_quat()
    else:
        out_quats[thigh] = (global_rots[thigh_parent].inv() * thigh_global).as_quat()

    calf_global_current = thigh_global * Rotation.from_quat(out_quats[calf])
    calf_vec_current = calf_global_current.apply(skel.rest_translations[foot])
    calf_vec_target = ankle_target - knee_target
    calf_global = _rotation_between_vectors(calf_vec_current, calf_vec_target) * calf_global_current
    out_quats[calf] = (thigh_global.inv() * calf_global).as_quat()

    foot_global_current = calf_global * Rotation.from_quat(out_quats[foot])
    ball_vec_current = foot_global_current.apply(skel.rest_translations[ball])
    ball_vec_target = ball_target - ankle_target
    if np.linalg.norm(ball_vec_target) > 1e-6:
        foot_global = _rotation_between_vectors(ball_vec_current, ball_vec_target) * foot_global_current
    else:
        foot_global = foot_global_current

    # VQ-FIX3: stance sole flattening. With ankle AND ball positions locked,
    # the foot can still roll about the ankle->ball axis without moving
    # either contact point — the "foot rotates in place" artifact. Both
    # contact points lie ON that axis, so a twist about it is a pure
    # in-place roll: pin the rest-relative sole normal toward vertical
    # (flat ground), weighted by contact strength for smooth ramps.
    if sole_flatten_weight > 0.0 and np.linalg.norm(ball_vec_target) > 1e-6:
        ball_dir = ball_vec_target / np.linalg.norm(ball_vec_target)
        rest_gr = skel.get_forward_kinematics()[1][foot]
        sole_local = Rotation.from_quat(rest_gr).inv().apply(np.array([0.0, 0.0, 1.0]))
        sole_now = foot_global.apply(sole_local)
        z_world = np.array([0.0, 0.0, 1.0])
        s_perp = sole_now - ball_dir * float(np.dot(sole_now, ball_dir))
        z_perp = z_world - ball_dir * float(np.dot(z_world, ball_dir))
        s_n = float(np.linalg.norm(s_perp))
        z_n = float(np.linalg.norm(z_perp))
        if s_n > 1e-6 and z_n > 1e-6:
            s_u = s_perp / s_n
            z_u = z_perp / z_n
            cos_a = float(np.clip(np.dot(s_u, z_u), -1.0, 1.0))
            sin_a = float(np.dot(np.cross(s_u, z_u), ball_dir))
            angle = np.arctan2(sin_a, cos_a)
            if np.isfinite(angle):
                twist = Rotation.from_rotvec(sole_flatten_weight * angle * ball_dir)
                foot_global = twist * foot_global

    out_quats[foot] = (calf_global.inv() * foot_global).as_quat()

    return out_quats


def _leg_pole_vector(
    positions: np.ndarray,
    thigh_idx: int,
    calf_idx: int,
    foot_idx: int,
) -> np.ndarray:
    hip = positions[thigh_idx]
    knee = positions[calf_idx]
    ankle = positions[foot_idx]
    axis = ankle - hip
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-8:
        return np.zeros(3)
    axis /= axis_norm
    pole = knee - (hip + axis * np.dot(knee - hip, axis))
    pole_norm = np.linalg.norm(pole)
    if pole_norm < 1e-8:
        return np.zeros(3)
    return pole / pole_norm


def _frontal_valgus_point(
    hip_l: np.ndarray,
    hip_r: np.ndarray,
    hip: np.ndarray,
    knee: np.ndarray,
    ankle: np.ndarray,
    side: str,
) -> float | None:
    """Signed frontal-plane knee angle from absolute joint positions."""
    up = np.array([0.0, 0.0, 1.0])
    horizontal = np.asarray(hip_r, dtype=float) - np.asarray(hip_l, dtype=float)
    horizontal[2] = 0.0
    hn = float(np.linalg.norm(horizontal))
    if not np.isfinite(horizontal).all() or hn < 1e-9:
        return None
    forward = np.cross(up, horizontal)
    fn = float(np.linalg.norm(forward))
    if fn < 1e-9:
        return None
    forward /= fn
    outward = (-1.0 if side == "l" else 1.0) * horizontal / hn
    thigh = np.asarray(knee, dtype=float) - np.asarray(hip, dtype=float)
    shin = np.asarray(ankle, dtype=float) - np.asarray(knee, dtype=float)
    thigh_plane = thigh - float(np.dot(thigh, forward)) * forward
    shin_plane = shin - float(np.dot(shin, forward)) * forward
    if np.linalg.norm(thigh_plane) < 1e-9 or np.linalg.norm(shin_plane) < 1e-9:
        return None
    thigh_angle = np.degrees(np.arctan2(
        float(np.dot(thigh_plane, up)), float(np.dot(thigh_plane, outward))))
    shin_angle = np.degrees(np.arctan2(
        float(np.dot(shin_plane, up)), float(np.dot(shin_plane, outward))))
    angle = (shin_angle - thigh_angle + 180.0) % 360.0 - 180.0
    return float(angle) if np.isfinite(angle) else None


def _frontal_match_pole(
    hip: np.ndarray,
    knee_cur: np.ndarray,
    foot_target: np.ndarray,
    upper: float,
    lower: float,
    pole_current: np.ndarray,
    hip_l: np.ndarray,
    hip_r: np.ndarray,
    target_valgus: float,
    side: str,
    prev_phi: float | None,
) -> tuple[np.ndarray, float | None]:
    """Grid-search the IK circle for a pole matching the frontal knee angle."""
    to_target = foot_target - hip
    dist = float(np.linalg.norm(to_target))
    if dist < 1e-8 or not np.isfinite(target_valgus):
        return pole_current, prev_phi
    max_reach = max(upper + lower - 1e-5, 1e-5)
    min_reach = abs(upper - lower) + 1e-5
    clamped_dist = float(np.clip(dist, min_reach, max_reach))
    axis = to_target / dist
    along = (upper * upper - lower * lower + clamped_dist * clamped_dist) / (2.0 * clamped_dist)
    radius = float(np.sqrt(max(upper * upper - along * along, 0.0)))
    if radius < 1e-6:
        return pole_current, prev_phi

    e1 = knee_cur - (hip + axis * float(np.dot(knee_cur - hip, axis)))
    e1_norm = float(np.linalg.norm(e1))
    if e1_norm < 1e-9:
        e1 = pole_current - axis * float(np.dot(pole_current, axis))
        e1_norm = float(np.linalg.norm(e1))
    if e1_norm < 1e-9:
        return pole_current, prev_phi
    e1 /= e1_norm
    e2 = np.cross(axis, e1)
    center = hip + axis * along
    phis = np.linspace(0.0, 2.0 * np.pi, 181, endpoint=False)
    errors = np.full(len(phis), np.inf)
    for i, phi in enumerate(phis):
        knee = center + radius * (np.cos(phi) * e1 + np.sin(phi) * e2)
        valgus = _frontal_valgus_point(
            hip_l, hip_r, hip, knee, foot_target, side)
        if valgus is not None:
            errors[i] = abs(valgus - target_valgus)
    if not np.isfinite(errors).any():
        return pole_current, prev_phi
    best = float(np.min(errors))
    candidates = np.flatnonzero(errors <= best + 1.0)
    if prev_phi is not None and np.isfinite(prev_phi):
        dphi = np.abs((phis[candidates] - prev_phi + np.pi) % (2.0 * np.pi) - np.pi)
        chosen = int(candidates[int(np.argmin(dphi))])
    else:
        chosen = int(candidates[int(np.argmin(errors[candidates]))])
    phi = float(phis[chosen])
    return np.cos(phi) * e1 + np.sin(phi) * e2, phi


def _smooth_valgus_target(series: np.ndarray) -> np.ndarray:
    """Centered five-frame median with edge-clamped, NaN-safe padding."""
    values = np.asarray(series, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return values.copy()
    filled = values.copy()
    filled[~finite] = float(np.median(values[finite]))
    padded = np.pad(filled, (2, 2), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, 5)
    return np.median(windows, axis=-1)


def _lock_segments(
    positions: np.ndarray,
    segments: list[tuple[int, int]],
    fps: float,
    strength: np.ndarray | None = None,
    anchor_frames: int = 5,
) -> dict[tuple[int, int], np.ndarray]:
    smooth = _zero_phase_filter_positions(positions, fps=fps, cutoff_hz=4.0)
    anchors: dict[tuple[int, int], np.ndarray] = {}
    for start, end in segments:
        segment_frames = np.arange(start, end + 1)
        use_count = min(anchor_frames, len(segment_frames))
        anchor_idx = segment_frames[:use_count]
        anchors[(start, end)] = np.median(smooth[anchor_idx], axis=0)
    return anchors


def _replant_anchor_tracks(
    foot_positions: np.ndarray,
    ball_positions: np.ndarray,
    hip_positions: np.ndarray,
    knee_positions: np.ndarray,
    segments: list[tuple[int, int]],
    strength: np.ndarray,
    skel: Skeleton,
    side: str,
    fps: float = 30.0,
    blend_frames: int = 5,
    anchor_frames: int = 5,
) -> tuple[dict[tuple[int, int], np.ndarray], dict[tuple[int, int], np.ndarray], np.ndarray, int, float]:
    """Build causal planted-anchor tracks for a replant policy.

    The existing five-frame median is retained for each initial plant.  A
    plant is considered stale when either contact point leaves a tolerance of
    the current anchor.  The tolerance is derived from the rig's measured
    leg reach and observed frame-to-frame contact noise: three percent of
    reach is the geometric allowance, three noise MAD-like steps absorb
    measured jitter, and the existing 3 cm contact-height floor is the upper
    bound.  Thus this policy has no clip-specific distance threshold.
    """
    n = len(foot_positions)
    out_foot: dict[tuple[int, int], np.ndarray] = {}
    out_ball: dict[tuple[int, int], np.ndarray] = {}
    out_strength = np.asarray(strength, dtype=float).copy()
    replant_count = 0
    max_drift = 0.0
    spec = FOOT_CHAINS[side]
    calf = skel.name_to_idx[spec["calf"]]
    foot = skel.name_to_idx[spec["foot"]]
    rest = np.asarray(skel.rest_translations, dtype=float)
    limb_scale = float(np.nanmedian(
        np.linalg.norm(knee_positions - hip_positions, axis=1)
        + np.linalg.norm(foot_positions - knee_positions, axis=1)
    ))
    if not np.isfinite(limb_scale) or limb_scale <= 1e-6:
        limb_scale = float(np.linalg.norm(rest[calf]) + np.linalg.norm(rest[foot]))
    if not np.isfinite(limb_scale) or limb_scale <= 1e-6:
        limb_scale = float(np.nanmedian(np.linalg.norm(
            foot_positions - ball_positions, axis=1)))
    limb_scale = max(limb_scale, 1.0)
    smooth_foot_positions = _zero_phase_filter_positions(
        np.asarray(foot_positions, dtype=float), fps=fps, cutoff_hz=4.0)
    smooth_ball_positions = _zero_phase_filter_positions(
        np.asarray(ball_positions, dtype=float), fps=fps, cutoff_hz=4.0)
    for start, end in segments:
        frames = np.arange(start, end + 1)
        if len(frames) == 0:
            continue
        use_count = min(anchor_frames, len(frames))
        first = frames[:use_count]
        foot_anchor = np.nanmedian(smooth_foot_positions[first], axis=0)
        ball_anchor = np.nanmedian(smooth_ball_positions[first], axis=0)
        if not (np.isfinite(foot_anchor).all() and np.isfinite(ball_anchor).all()):
            continue
        # Contact noise is measured from frame-to-frame support motion. The
        # cap reuses the existing 3 cm stance-height floor, preventing a
        # slowly moving pivot from accumulating a stale multi-decimetre lock.
        support = 0.35 * foot_positions[frames] + 0.65 * ball_positions[frames]
        steps = np.linalg.norm(np.diff(support, axis=0), axis=1)
        finite_steps = steps[np.isfinite(steps)]
        noise_step = float(np.median(finite_steps)) if len(finite_steps) else 0.0
        tolerance = min(3.0, max(0.03 * limb_scale, 3.0 * noise_step))
        foot_track = np.full((n, 3), np.nan, dtype=float)
        ball_track = np.full((n, 3), np.nan, dtype=float)
        current_foot = foot_anchor.copy()
        current_ball = ball_anchor.copy()
        transitions: list[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        for frame in frames:
            foot_track[frame] = current_foot
            ball_track[frame] = current_ball
            foot_drift = float(np.linalg.norm(foot_positions[frame] - current_foot))
            ball_drift = float(np.linalg.norm(ball_positions[frame] - current_ball))
            if np.isfinite(foot_drift) and np.isfinite(ball_drift):
                max_drift = max(max_drift, foot_drift, ball_drift)
            if max(foot_drift, ball_drift) > tolerance:
                replant_count += 1
                old_foot = current_foot.copy()
                old_ball = current_ball.copy()
                current_foot = np.asarray(smooth_foot_positions[frame], dtype=float).copy()
                current_ball = np.asarray(smooth_ball_positions[frame], dtype=float).copy()
                # Reuse contact's smoothstep edge machinery: transition
                # from the stale anchor to the new plant over the existing
                # blend window, so no target jump is introduced.
                transitions.append((frame, old_foot, old_ball,
                                    current_foot.copy(), current_ball.copy()))
        for frame, old_foot, old_ball, new_foot, new_ball in transitions:
            width = min(end - frame + 1, max(1, blend_frames))
            ramp = _smoothstep(np.linspace(0.0, 1.0, width))
            foot_track[frame:frame + width] = (
                (1.0 - ramp[:, None]) * old_foot + ramp[:, None] * new_foot)
            ball_track[frame:frame + width] = (
                (1.0 - ramp[:, None]) * old_ball + ramp[:, None] * new_ball)
        out_foot[(start, end)] = foot_track
        out_ball[(start, end)] = ball_track
    return out_foot, out_ball, out_strength, replant_count, max_drift


def _compute_root_contact_offsets(

    before_pos: np.ndarray,
    contacts: dict[str, dict[str, np.ndarray | list[tuple[int, int]] | float]],
    foot_anchors: dict[str, dict[tuple[int, int], np.ndarray]],
    ball_anchors: dict[str, dict[tuple[int, int], np.ndarray]],
    skel: Skeleton,
    fps: float,
    anchor_policy: str = "fixed",
) -> np.ndarray:
    if anchor_policy == "replant":
        raw_offsets = np.zeros((len(before_pos), 3), dtype=float)
        confidence = np.zeros(len(before_pos), dtype=float)
        bounds = np.zeros(len(before_pos), dtype=float)
        for frame in range(len(before_pos)):
            total_weight = 0.0
            offset = np.zeros(3, dtype=float)
            max_drift = 0.0
            for side, spec in FOOT_CHAINS.items():

                strength = contacts[side]["strength"]
                assert isinstance(strength, np.ndarray)
                weight = float(strength[frame])
                if weight <= 0.0:
                    continue
                foot_anchor = _segment_anchor(frame, foot_anchors[side])
                ball_anchor = _segment_anchor(frame, ball_anchors[side])
                if foot_anchor is None or ball_anchor is None:
                    continue
                foot_idx = skel.name_to_idx[spec["foot"]]
                ball_idx = skel.name_to_idx[spec["ball"]]
                support_now = 0.35 * before_pos[frame, foot_idx] + 0.65 * before_pos[frame, ball_idx]
                support_anchor = 0.35 * foot_anchor + 0.65 * ball_anchor
                drift = support_anchor - support_now
                if not np.isfinite(drift).all():
                    continue
                drift[2] = 0.0
                offset += weight * drift
                total_weight += weight
                max_drift = max(max_drift, float(np.linalg.norm(drift[:2])))
            if total_weight > 0.0:
                raw_offsets[frame] = offset / total_weight
                confidence[frame] = 1.0
                bounds[frame] = max_drift

        # Keep the established zero-phase smoothing and per-frame step limit,
        # but do not carry correction through frames without a valid plant.
        smoothed = _zero_phase_filter_positions(raw_offsets, fps=fps, cutoff_hz=3.0)
        smoothed[confidence <= 0.0] = 0.0
        xy_norm = np.linalg.norm(smoothed[:, :2], axis=1)
        valid = (confidence > 0.0) & np.isfinite(xy_norm)
        scale = np.ones(len(smoothed), dtype=float)
        nonzero = valid & (xy_norm > 0.0)
        scale[nonzero] = np.minimum(1.0, bounds[nonzero] / xy_norm[nonzero])
        smoothed[:, :2] *= scale[:, None]
        limited = _limit_xy_delta(smoothed, max_delta=120.0 / fps)
        limited[confidence <= 0.0] = 0.0
        limited_norm = np.linalg.norm(limited[:, :2], axis=1)
        limited_valid = (confidence > 0.0) & np.isfinite(limited_norm)
        limited_nonzero = limited_valid & (limited_norm > 0.0)
        final_scale = np.ones(len(limited), dtype=float)
        final_scale[limited_nonzero] = np.minimum(
            1.0, bounds[limited_nonzero] / limited_norm[limited_nonzero])
        limited[:, :2] *= final_scale[:, None]
        limited[~np.isfinite(limited)] = 0.0
        return limited

    raw_offsets = np.zeros((len(before_pos), 3), dtype=float)
    confidence = np.zeros(len(before_pos), dtype=float)
    for frame in range(len(before_pos)):
        total_weight = 0.0
        offset = np.zeros(3, dtype=float)
        inactive_probability = 1.0
        for side, spec in FOOT_CHAINS.items():
            strength = contacts[side]["strength"]
            assert isinstance(strength, np.ndarray)
            weight = float(strength[frame])
            if weight <= 0.0:
                continue

            foot_anchor = _segment_anchor(frame, foot_anchors[side])
            ball_anchor = _segment_anchor(frame, ball_anchors[side])
            if foot_anchor is None or ball_anchor is None:
                continue

            foot_idx = skel.name_to_idx[spec["foot"]]
            ball_idx = skel.name_to_idx[spec["ball"]]
            support_now = 0.35 * before_pos[frame, foot_idx] + 0.65 * before_pos[frame, ball_idx]
            support_anchor = 0.35 * foot_anchor + 0.65 * ball_anchor
            correction = support_anchor - support_now
            correction[2] = 0.0
            offset += weight * correction
            total_weight += weight
            inactive_probability *= 1.0 - weight

        if total_weight > 0.0:
            raw_offsets[frame] = offset / total_weight
            confidence[frame] = 1.0 - inactive_probability

    constrained = np.flatnonzero(confidence >= 0.99)
    if len(constrained) == 0:
        constrained = np.flatnonzero(confidence > 0.0)
    if len(constrained) == 0:
        return raw_offsets

    # Carry root correction continuously through swing intervals. Resetting it
    # to zero at every contact boundary teleports the whole target skeleton.
    continuous = np.zeros_like(raw_offsets)
    frames = np.arange(len(before_pos))
    for dim in range(2):
        continuous[:, dim] = np.interp(
            frames,
            constrained,
            raw_offsets[constrained, dim],
        )
    continuous = _zero_phase_filter_positions(
        continuous,
        fps=fps,
        cutoff_hz=3.0,
    )

    # Exact support compensation is retained in full contact. At contact
    # boundaries, confidence blends into the continuous trajectory instead of
    # cancelling the ease-in/ease-out ramp through weight normalization.
    correction = (
        confidence[:, None] * raw_offsets
        + (1.0 - confidence[:, None]) * continuous
    )
    return _limit_xy_delta(correction, max_delta=120.0 / fps)




def _reachable_lock_target(
    hip: np.ndarray,
    prelock_foot: np.ndarray,
    anchor_foot: np.ndarray,
    requested_weight: float,
    upper_len: float,
    lower_len: float,
) -> tuple[np.ndarray, float, bool]:
    """Fade an anchored target back toward pre-lock until it is reachable."""
    base = np.asarray(prelock_foot, dtype=float)
    anchor = np.asarray(anchor_foot, dtype=float)
    if not (np.isfinite(hip).all() and np.isfinite(base).all() and np.isfinite(anchor).all()):
        return base, 0.0, True
    maximum = max(float(upper_len + lower_len) - 1e-5, 1e-5)
    minimum = abs(float(upper_len - lower_len)) + 1e-5

    def feasible(weight: float) -> bool:
        target = base + weight * (anchor - base)
        distance = float(np.linalg.norm(target - hip))
        return minimum <= distance <= maximum

    desired = float(np.clip(requested_weight, 0.0, 1.0))
    if feasible(desired):
        return base + desired * (anchor - base), desired, False
    if not feasible(0.0):
        # The pre-lock FK itself is unusable; fail closed rather than passing
        # an unbounded target into the existing IK reach clamp.
        return base, 0.0, True
    low, high = 0.0, desired
    for _ in range(32):
        midpoint = 0.5 * (low + high)
        if feasible(midpoint):
            low = midpoint
        else:
            high = midpoint
    return base + low * (anchor - base), low, True

def _lock_pole_segments(
    positions: np.ndarray,
    segments: list[tuple[int, int]],
    skel: Skeleton,
    side: str,
) -> dict[tuple[int, int], np.ndarray]:
    spec = FOOT_CHAINS[side]
    thigh_idx = skel.name_to_idx[spec["thigh"]]
    calf_idx = skel.name_to_idx[spec["calf"]]
    foot_idx = skel.name_to_idx[spec["foot"]]
    anchors: dict[tuple[int, int], np.ndarray] = {}
    for segment in segments:
        start, end = segment
        poles = np.array([
            _leg_pole_vector(positions[frame], thigh_idx, calf_idx, foot_idx)
            for frame in range(start, end + 1)
        ])
        valid = np.linalg.norm(poles, axis=1) > 1e-8
        if not np.any(valid):
            continue
        pole = np.median(poles[valid], axis=0)
        pole_norm = np.linalg.norm(pole)
        if pole_norm > 1e-8:
            anchors[segment] = pole / pole_norm
    return anchors


def _segment_anchor(
    frame: int,
    anchors: dict[tuple[int, int], np.ndarray],
) -> np.ndarray | None:
    for (start, end), anchor in anchors.items():
        if start <= frame <= end:
            anchor_array = np.asarray(anchor, dtype=float)
            if anchor_array.ndim == 2:
                if frame < len(anchor_array):
                    value = anchor_array[frame]
                else:
                    return None
            else:
                value = anchor_array
            return value if np.isfinite(value).all() else None
    return None


def _contact_drift_metrics(
    positions: np.ndarray,
    contacts: dict[str, dict[str, np.ndarray | list[tuple[int, int]] | float]],
    skel: Skeleton,
) -> dict[str, dict[str, float | int]]:
    metrics: dict[str, dict[str, float | int]] = {}
    for side, spec in FOOT_CHAINS.items():
        foot_idx = skel.name_to_idx[spec["foot"]]
        ball_idx = skel.name_to_idx[spec["ball"]]
        segments = contacts[side]["segments"]
        strength = contacts[side]["strength"]
        assert isinstance(segments, list)
        assert isinstance(strength, np.ndarray)

        foot_drifts = []
        ball_drifts = []
        foot_accels = []
        for start, end in segments:
            segment_frames = np.arange(start, end + 1)
            locked_frames = segment_frames[strength[start:end + 1] >= 0.99]
            if len(locked_frames) >= 3:
                frame_idx = locked_frames
            else:
                frame_idx = segment_frames

            if len(frame_idx) < 3:
                continue
            foot_xy = positions[frame_idx, foot_idx, :2]
            ball_xy = positions[frame_idx, ball_idx, :2]
            foot_anchor = np.median(foot_xy, axis=0)
            ball_anchor = np.median(ball_xy, axis=0)
            foot_drifts.append(float(np.max(np.linalg.norm(foot_xy - foot_anchor, axis=1))))
            ball_drifts.append(float(np.max(np.linalg.norm(ball_xy - ball_anchor, axis=1))))
            foot_accels.append(float(np.percentile(
                np.linalg.norm(np.diff(positions[frame_idx, foot_idx], n=2, axis=0), axis=1),
                95,
            )))

        metrics[side] = {
            "segments": len(segments),
            "foot_drift_cm": float(max(foot_drifts) if foot_drifts else 0.0),
            "ball_drift_cm": float(max(ball_drifts) if ball_drifts else 0.0),
            "foot_accel_p95_cm_per_frame2": float(max(foot_accels) if foot_accels else 0.0),
        }
    return metrics


def apply_foot_lock(
    skel: Skeleton,
    animation_data: np.ndarray,
    source_pts3d: np.ndarray,
    fps: float = 30.0,
    debug_dir: str | Path | None = None,
    pose2d_npz: str | Path | None = None,
    anchor_policy: str = "fixed",
) -> tuple[np.ndarray, dict]:
    """Apply soft target-space foot locking to full local-rotation animation data.

    ``anchor_policy='fixed'`` is the historical, bit-identical path.  The
    opt-in ``'replant'`` policy splits stale contacts into causal plants.
    """
    if anchor_policy not in {"fixed", "replant"}:
        raise ValueError(f"anchor_policy must be 'fixed' or 'replant', got {anchor_policy!r}")
    if animation_data.size == 0:
        return animation_data, {}

    required = {name for spec in FOOT_CHAINS.values() for name in (spec["thigh"], spec["calf"], spec["foot"], spec["ball"])}
    if not required.issubset(skel.name_to_idx):
        return animation_data, {"enabled": False, "reason": "missing foot chain bones"}

    root_seq = animation_data[:, :3].copy()
    eulers = animation_data[:, 3:].reshape(len(animation_data), skel.num_joints, 3)
    local_quats = Rotation.from_euler("xyz", eulers.reshape(-1, 3), degrees=False).as_quat()
    local_quats = local_quats.reshape(len(animation_data), skel.num_joints, 4)

    before_pos, _ = _fk_sequence(skel, root_seq, local_quats)
    source_valgus = {"l": np.full(len(animation_data), np.nan),
                     "r": np.full(len(animation_data), np.nan)}
    source_hip_l = skel.name_to_idx["thigh_l"]
    source_hip_r = skel.name_to_idx["thigh_r"]
    for frame in range(len(animation_data)):
        for side, hip_idx, knee_idx, ankle_idx in (
            ("l", source_hip_l, skel.name_to_idx["calf_l"], skel.name_to_idx["foot_l"]),
            ("r", source_hip_r, skel.name_to_idx["calf_r"], skel.name_to_idx["foot_r"]),
        ):
            value = _frontal_valgus_point(
                before_pos[frame, source_hip_l], before_pos[frame, source_hip_r],
                before_pos[frame, hip_idx], before_pos[frame, knee_idx],
                before_pos[frame, ankle_idx], side,
            )
            if value is not None:
                source_valgus[side][frame] = value
    smoothed_source_valgus = {
        side: _smooth_valgus_target(values) for side, values in source_valgus.items()
    }
    source_contacts = detect_foot_contacts(source_pts3d[:len(animation_data)], fps=fps)
    image_contacts = None
    image_contact_error = None
    if pose2d_npz is not None:
        try:
            image_contacts = detect_image_foot_contacts(
                pose2d_npz,
                num_frames=len(animation_data),
                fps=fps,
            )
        except Exception as exc:
            image_contact_error = str(exc)
    contacts = detect_target_foot_contacts(
        skel,
        before_pos,
        fps=fps,
        source_contacts=source_contacts,
        image_contacts=image_contacts,
    )

    foot_anchors = {}
    ball_anchors = {}
    replant_count = 0
    max_anchor_drift = 0.0
    for side, spec in FOOT_CHAINS.items():
        foot_idx = skel.name_to_idx[spec["foot"]]
        ball_idx = skel.name_to_idx[spec["ball"]]
        thigh = skel.name_to_idx[spec["thigh"]]
        calf = skel.name_to_idx[spec["calf"]]
        segments = contacts[side]["segments"]
        strength = contacts[side]["strength"]
        assert isinstance(segments, list)
        assert isinstance(strength, np.ndarray)
        if anchor_policy == "replant":
            (
                foot_anchors[side],
                ball_anchors[side],
                replanted_strength,
                side_replants,
                side_max_drift,
            ) = _replant_anchor_tracks(
                before_pos[:, foot_idx],
                before_pos[:, ball_idx],
                before_pos[:, thigh],
                before_pos[:, calf],
                segments,
                strength,
                skel,
                side,
                fps=fps,
            )
            contacts[side]["strength"] = replanted_strength
            replant_count += side_replants
            max_anchor_drift = max(max_anchor_drift, side_max_drift)
        else:
            foot_anchors[side] = _lock_segments(
                before_pos[:, foot_idx],
                segments,
                fps=fps,
                strength=strength,
            )
            ball_anchors[side] = _lock_segments(
                before_pos[:, ball_idx],
                segments,
                fps=fps,
                strength=strength,
            )
    root_offsets = _compute_root_contact_offsets(
        before_pos,
        contacts,
        foot_anchors,
        ball_anchors,
        skel,
        fps=fps,
        anchor_policy=anchor_policy,
    )
    root_seq_locked = root_seq + root_offsets

    unreachable_frames = 0

    def _solve_locked_legs(root_sequence: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        nonlocal unreachable_frames
        shifted_pos, _ = _fk_sequence(skel, root_sequence, local_quats)
        solved_quats = local_quats.copy()
        prev_phi: dict[str, float | None] = {"left": None, "right": None}
        for frame in range(len(animation_data)):
            for side in FOOT_CHAINS:
                strength = contacts[side]["strength"]
                assert isinstance(strength, np.ndarray)
                weight = float(strength[frame])
                foot_anchor = _segment_anchor(frame, foot_anchors[side])
                ball_anchor = _segment_anchor(frame, ball_anchors[side])
                if weight <= 0.0 or foot_anchor is None or ball_anchor is None:
                    prev_phi[side] = None
                    continue

                spec = FOOT_CHAINS[side]
                thigh_idx = skel.name_to_idx[spec["thigh"]]
                calf_idx = skel.name_to_idx[spec["calf"]]
                foot_idx = skel.name_to_idx[spec["foot"]]
                ball_idx = skel.name_to_idx[spec["ball"]]
                foot_target = (1.0 - weight) * shifted_pos[frame, foot_idx] + weight * foot_anchor
                effective_weight = weight
                if anchor_policy == "replant":
                    upper_len = float(np.linalg.norm(
                        shifted_pos[frame, calf_idx] - shifted_pos[frame, thigh_idx]))
                    lower_len = float(np.linalg.norm(
                        shifted_pos[frame, foot_idx] - shifted_pos[frame, calf_idx]))
                    foot_target, effective_weight, unreachable = _reachable_lock_target(
                        shifted_pos[frame, thigh_idx],
                        shifted_pos[frame, foot_idx],
                        foot_anchor,
                        weight,
                        upper_len,
                        lower_len,
                    )
                    if unreachable:
                        unreachable_frames += 1
                ball_target = ((1.0 - effective_weight) * shifted_pos[frame, ball_idx]
                               + effective_weight * ball_anchor)
                current_pole = _leg_pole_vector(
                    shifted_pos[frame], thigh_idx, calf_idx, foot_idx)
                v_cur = _frontal_valgus_point(
                    shifted_pos[frame, source_hip_l], shifted_pos[frame, source_hip_r],
                    shifted_pos[frame, thigh_idx], shifted_pos[frame, calf_idx],
                    before_pos[frame, foot_idx], "l" if side == "left" else "r",
                )
                v_src = smoothed_source_valgus["l" if side == "left" else "r"][frame]
                if v_cur is None:
                    v_cur = float(v_src) if np.isfinite(v_src) else float("nan")
                if anchor_policy == "replant":
                    v_goal = (1.0 - effective_weight) * v_cur + effective_weight * v_src
                else:
                    v_goal = (1.0 - weight) * v_cur + weight * v_src
                # VQ-FIX4 (frontal knee-angle matching): select the IK-circle
                # pole against the source frontal target. Validated by
                # outputs/vq_investigation/audit_knee_policy.json and
                # docs/phase_reports/VQ_INVESTIGATION.md + VQ_FIXES.md.
                pole_hint, prev_phi[side] = _frontal_match_pole(
                    shifted_pos[frame, thigh_idx], shifted_pos[frame, calf_idx],
                    foot_target,
                    float(np.linalg.norm(shifted_pos[frame, calf_idx] - shifted_pos[frame, thigh_idx])),
                    float(np.linalg.norm(shifted_pos[frame, foot_idx] - shifted_pos[frame, calf_idx])),
                    current_pole,
                    shifted_pos[frame, source_hip_l], shifted_pos[frame, source_hip_r],
                    float(v_goal), "l" if side == "left" else "r", prev_phi[side],
                )
                solved_quats[frame] = _apply_leg_ik_frame(
                    skel,
                    solved_quats[frame],
                    root_sequence[frame],
                    side,
                    foot_target,
                    ball_target,
                    pole_hint=pole_hint,
                    sole_flatten_weight=(
                        effective_weight if anchor_policy == "replant" else weight
                    ),
                )
        solved_pos, _ = _fk_sequence(skel, root_sequence, solved_quats)
        return solved_quats, solved_pos

    locked_quats, after_pos = _solve_locked_legs(root_seq_locked)
    locked_eulers = Rotation.from_quat(locked_quats.reshape(-1, 4)).as_euler("xyz", degrees=False)
    out = animation_data.copy()
    out[:, :3] = root_seq_locked
    out[:, 3:] = locked_eulers.reshape(len(animation_data), -1)
    finite_fallback = False
    if anchor_policy == "replant" and not np.isfinite(out).all():
        # Fail closed: never emit a partially non-finite animation from the
        # optional policy. The untouched input is the deterministic fallback.
        out = animation_data.copy()
        after_pos = before_pos.copy()
        finite_fallback = True

    diagnostics = {
        "enabled": True,
        "anchor_policy": anchor_policy,
        "replant_count": int(replant_count),
        "unreachable_frames": int(unreachable_frames),
        "max_anchor_drift_cm": float(max_anchor_drift),
        "finite_fallback": finite_fallback,
        "fps": float(fps),
        "root_correction_xy_p95_cm": float(np.percentile(np.linalg.norm(root_offsets[:, :2], axis=1), 95)),
        "root_correction_xy_max_cm": float(np.max(np.linalg.norm(root_offsets[:, :2], axis=1))),
        "root_correction_speed_p95_cm_per_sec": float(np.percentile(
            np.linalg.norm(np.diff(root_offsets[:, :2], axis=0), axis=1) * fps,
            95,
        )),
        "root_correction_speed_max_cm_per_sec": float(np.max(
            np.linalg.norm(np.diff(root_offsets[:, :2], axis=0), axis=1) * fps,
        )),
        "before": _contact_drift_metrics(before_pos, contacts, skel),
        "after": _contact_drift_metrics(after_pos, contacts, skel),
        "image_contacts_used": image_contacts is not None,
        "image_contact_error": image_contact_error,
        "contacts": {
            side: {
                "segments": contacts[side]["segments"],
                "ground_z": contacts[side]["ground_z"],
                "height_band": contacts[side]["height_band"],
                "speed_threshold": contacts[side]["speed_threshold"],
                "source_segments": source_contacts[side]["segments"],
                "image_segments": (
                    image_contacts[side]["segments"] if image_contacts is not None else []
                ),
                "image_speed_threshold": (
                    image_contacts[side]["speed_threshold"] if image_contacts is not None else None
                ),
            }
            for side in FOOT_CHAINS
        },
    }

    if debug_dir is not None:
        debug_path = Path(debug_dir)
        debug_path.mkdir(parents=True, exist_ok=True)
        with (debug_path / "foot_lock_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(diagnostics, f, indent=2)
        with (debug_path / "foot_contacts.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["frame", "left_strength", "right_strength"])
            left_strength = contacts["left"]["strength"]
            right_strength = contacts["right"]["strength"]
            assert isinstance(left_strength, np.ndarray)
            assert isinstance(right_strength, np.ndarray)
            for frame in range(len(animation_data)):
                writer.writerow([frame, float(left_strength[frame]), float(right_strength[frame])])

    return out, diagnostics
