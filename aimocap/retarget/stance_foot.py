"""Position-preserving stance-foot orientation stabilization.

The module operates on the flattened Manny animation representation used by the
retargeter: root translation followed by XYZ Euler angles for each FBX joint.
All positions are centimetres in a Z-up world and quaternions use SciPy's
``xyzw`` convention.

Two modes are available:

``generic``
    Stance-weighted minimal swing that makes each solved sole normal point up.
``landmark``
    Uses the heel/big-toe/small-toe triangle to calibrate the constant gauge
    between the observed landmark frame and a flat foot frame.  The calibrated
    orientation is then blended into the solved foot without changing the
    ankle position; the ball keeps its solved local rotation relative to foot.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from aimocap.retarget.foot_lock import _rotation_between_vectors


UP = np.array([0.0, 0.0, 1.0], dtype=np.float64)
_HEIGHT_FLOOR_CM = 2.0
_SPEED_FLOOR_CM_S = 5.0
_NOISE_FLOOR_CM = 0.05
_REST_GROUND_TOL_CM = 2.0
_ENTER_WEIGHT = 0.6
_EXIT_WEIGHT = 0.3
_TRANSITION_SECONDS = 0.2
_LANDMARK_FILTER_SECONDS = 0.1
_MIN_TRIANGLE_RATIO = 1e-6


def _finite_vec(value: Any, shape: tuple[int, ...] | None = None) -> bool:
    try:
        x = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return False
    return (shape is None or x.shape == shape) and bool(np.isfinite(x).all())


def _smoothstep(x: np.ndarray | float) -> np.ndarray | float:
    y = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    result = y * y * (3.0 - 2.0 * y)
    return float(result) if np.ndim(x) == 0 else result


def _symmetric_smooth(values: np.ndarray, width: int) -> np.ndarray:
    """Centered moving average with edge replication (deterministic, no lag)."""
    values = np.asarray(values, dtype=float)
    if values.ndim > 1:
        return np.column_stack([_symmetric_smooth(values[:, i], width) for i in range(values.shape[1])])
    width = max(1, int(width))
    if width <= 1 or len(values) <= 1:
        return values.copy()
    if width % 2 == 0:
        width += 1
    radius = width // 2
    padded = np.pad(values, (radius, radius), mode="edge")
    return np.convolve(padded, np.full(width, 1.0 / width), mode="valid")


def _mad(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=float).reshape(-1)
    x = x[np.isfinite(x)]
    if not len(x):
        return 0.0
    med = float(np.median(x))
    return float(np.median(np.abs(x - med)))


def _track(targets: Mapping[str, Any], name: str, n: int) -> np.ndarray | None:
    value = targets.get(name)
    if value is None:
        return None
    try:
        x = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if x.ndim != 2 or x.shape[1] != 3 or len(x) < n:
        return None
    return x[:n].copy()


def _diagnostic_value(diagnostics: Any, names: tuple[str, ...]) -> float | None:
    """Find a finite scalar in the flexible target-diagnostics schema."""
    if diagnostics is None:
        return None
    if isinstance(diagnostics, Mapping):
        for name in names:
            if name in diagnostics:
                try:
                    value = float(diagnostics[name])
                except (TypeError, ValueError):
                    value = float("nan")
                if np.isfinite(value) and value >= 0.0:
                    return value
        for value in diagnostics.values():
            found = _diagnostic_value(value, names)
            if found is not None:
                return found
    elif isinstance(diagnostics, (list, tuple)):
        for value in diagnostics:
            found = _diagnostic_value(value, names)
            if found is not None:
                return found
    return None


def _track_sigma(targets: Mapping[str, Any], side: str, tracks: list[np.ndarray],
                 fps: float, diagnostics: Any) -> tuple[float, float, str]:
    """Return robust height and speed noise scales in cm and cm/s."""
    names = (f"sigma_feet_{side}", f"feet_sigma_{side}", "sigma_feet",
             "feet_sigma", "noise_sigma_cm")
    sigma_h = _diagnostic_value(diagnostics, names)
    track_names = (f"heel_{side}", f"big_toe_{side}", f"small_toe_{side}")

    def visit(value: Any) -> None:
        nonlocal sigma_h
        if not isinstance(value, Mapping):
            return
        for track_name in track_names:
            row = value.get(track_name)
            if isinstance(row, Mapping):
                candidate = _diagnostic_value(row, ("sigma_cm", "noise_sigma_cm"))
                if candidate is not None:
                    sigma_h = candidate if sigma_h is None else max(sigma_h, candidate)
        for child in value.values():
            visit(child)

    visit(diagnostics)
    second_values: list[np.ndarray] = []
    for track in tracks:
        finite = np.isfinite(track).all(axis=1)
        clean = track[finite]
        if len(clean) >= 3:
            second_values.append(np.linalg.norm(np.diff(clean, n=2, axis=0), axis=1))
    fallback = max(_mad(np.concatenate(second_values)) if second_values else 0.0,
                   _NOISE_FLOOR_CM)
    if sigma_h is None:
        sigma_h = fallback
        source = "robust_MAD_second_difference"
    else:
        source = "target_diagnostics"
    sigma_v = _diagnostic_value(
        diagnostics, (f"speed_sigma_{side}", "speed_sigma_cm_s", "feet_speed_sigma_cm_s")
    )
    if sigma_v is None:
        sigma_v = fallback * fps
    return float(sigma_h), float(sigma_v), source


def _floor_from_targets(targets: Mapping[str, Any], n: int, floor_z: float | None,
                        diagnostics: Any) -> tuple[float | None, str]:
    if floor_z is not None:
        value = float(floor_z)
        return (value, "explicit_floor_z") if np.isfinite(value) else (None, "invalid_floor_z")
    registered = _diagnostic_value(diagnostics, ("grounded_floor_z", "floor_z"))
    if isinstance(diagnostics, Mapping):
        for key in ("floor_transform", "grounded_floor_transform"):
            transform = diagnostics.get(key)
            if isinstance(transform, Mapping):
                status = transform.get("grounding_status")
                validation = transform.get("validation_status")
                if status == "registered" and validation in ("passed", None):
                    return 0.0, "grounded_floor_transform"
    if registered is None:
        raw = targets.get("floor_z")
        try:
            registered = float(raw) if np.asarray(raw).ndim == 0 else None
        except (TypeError, ValueError):
            registered = None
    if registered is not None and np.isfinite(registered):
        return float(registered), "grounded_floor_transform"

    pooled: list[np.ndarray] = []
    for side in ("l", "r"):
        tracks = [_track(targets, f"{name}_{side}", n)
                  for name in ("heel", "big_toe", "small_toe")]
        tracks = [x for x in tracks if x is not None]
        if len(tracks) >= 2:
            pooled.append(np.nanmin(np.vstack([x[:, 2] for x in tracks]), axis=0))
    if not pooled:
        return None, "missing_conditioned_sole_tracks"
    values = np.concatenate(pooled)
    values = values[np.isfinite(values)]
    if not len(values):
        return None, "nonfinite_conditioned_sole_tracks"
    return float(np.percentile(values, 5.0)), "conditioned_targets_p05"


def _rest_ground_diagnostic(skeleton: Any, floor: float | None,
                           *, strict_floor: bool = False) -> tuple[bool, dict[str, Any]]:
    try:
        positions, _ = skeleton.get_forward_kinematics()
        ids = skeleton.name_to_idx
        required = [ids[f"foot_{s}"] for s in ("l", "r")] + [ids[f"ball_{s}"] for s in ("l", "r")]
        values = np.asarray(positions, dtype=float)[required]
        if not np.isfinite(values).all():
            return False, {"reason": "nonfinite_rest_foot_ball", "rest_heights_cm": values[:, 2].tolist()}
        ball_z = np.asarray([positions[ids["ball_l"], 2], positions[ids["ball_r"], 2]])
        reference = float(floor) if (strict_floor and floor is not None) else float(np.min(ball_z))
        grounded = bool(np.all(np.abs(ball_z - reference) <= _REST_GROUND_TOL_CM)) if strict_floor else bool(np.max(ball_z) <= _REST_GROUND_TOL_CM)
        return grounded, {
            "rest_heights_cm": {name: float(positions[ids[name], 2]) for name in ("foot_l", "ball_l", "foot_r", "ball_r")},
            "rest_floor_reference_cm": reference,
            "rest_ground_tolerance_cm": _REST_GROUND_TOL_CM,
        }
    except Exception as exc:
        return False, {"reason": f"rest_fk_failed:{type(exc).__name__}:{exc}"}


def _hysteresis(raw: np.ndarray, enter: float = _ENTER_WEIGHT,
                exit: float = _EXIT_WEIGHT) -> np.ndarray:
    """Apply enter/exit thresholds to a raw contact product."""
    raw = np.asarray(raw, dtype=float)
    state = np.zeros(len(raw), dtype=bool)
    active = False
    for i, value in enumerate(raw):
        if not np.isfinite(value):
            active = False
        elif active:
            if value < exit:
                active = False
        elif value > enter:
            active = True
        state[i] = active
    return state


def _transition_ramp(state: np.ndarray, width: int) -> np.ndarray:
    """Centered zero-phase smoothstep ramp around each hysteresis edge."""
    state = np.asarray(state, dtype=bool)
    out = np.zeros(len(state), dtype=float)
    width = max(1, int(width))
    if width <= 1 or len(state) < 2:
        return state.astype(float)
    half = width // 2
    i = 0
    while i < len(state):
        if not state[i]:
            i += 1
            continue
        start = i
        while i + 1 < len(state) and state[i + 1]:
            i += 1
        end = i
        lo = max(0, start - half)
        hi = min(len(state), end + 1 + half)
        indices = np.arange(lo, hi, dtype=float)
        if start == 0:
            rise = np.ones_like(indices)
        else:
            rise = _smoothstep((indices - (start - half)) / max(width, 1))
        if end == len(state) - 1:
            fall = np.ones_like(indices)
        else:
            fall = 1.0 - _smoothstep((indices - (end + 1 - half)) / max(width, 1))
        out[lo:hi] = np.maximum(out[lo:hi], np.minimum(rise, fall))
        i += 1
    return np.clip(out, 0.0, 1.0)


def _zero_phase_rotation_filter(values: list[Rotation | None], valid: np.ndarray,
                                width: int) -> list[Rotation | None]:
    """Centered chordal quaternion mean on SO(3), with sign unwrapping."""
    width = max(1, int(width))
    if width % 2 == 0:
        width += 1
    radius = width // 2
    result = list(values)
    for i in np.flatnonzero(valid):
        lo, hi = max(0, int(i) - radius), min(len(values), int(i) + radius + 1)
        center = values[i]
        if center is None:
            continue
        q0 = center.as_quat()
        qs = []
        for j in range(lo, hi):
            if values[j] is None or not valid[j]:
                continue
            q = values[j].as_quat().copy()
            if float(np.dot(q, q0)) < 0.0:
                q = -q
            qs.append(q)
        if qs:
            q = np.sum(qs, axis=0)
            norm = float(np.linalg.norm(q))
            if norm > 1e-12 and np.isfinite(norm):
                result[i] = Rotation.from_quat(q / norm)
    return result


def _set_local_from_world(skeleton: Any, local_q: np.ndarray, joint: int,
                          world: Rotation, world_q: np.ndarray) -> None:
    parent = int(skeleton.parents[joint])
    if parent < 0:
        local_q[joint] = world.as_quat()
    else:
        local_q[joint] = (Rotation.from_quat(world_q[parent]).inv() * world).as_quat()


def _landmark_frame(heel: np.ndarray, big: np.ndarray, small: np.ndarray,
                    sigma_cm: float) -> tuple[Rotation | None, float, float, float]:
    """Return target frame, singular-value ratio, noise threshold, condition."""
    points = np.asarray([heel, big, small], dtype=float)
    if not np.isfinite(points).all():
        return None, float("nan"), float("nan"), 0.0
    centered = points - np.mean(points, axis=0)
    try:
        _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return None, float("nan"), float("nan"), 0.0
    if singular[0] <= 1e-9 or singular[1] <= 1e-9:
        return None, float(singular[1] / max(singular[0], 1e-12)), float("nan"), 0.0
    ratio = float(singular[1] / singular[0])
    # The two-point direction is reliable when its minor singular value is
    # greater than three position-noise sigmas.  Express that test as the
    # dimensionless ratio rho=s1/s0 > eta=3*sigma/s0.
    eta = float(min(0.95, 3.0 * max(float(sigma_cm), _NOISE_FLOOR_CM) / singular[0]))
    condition = float(np.clip((ratio - eta) / max(1.0 - eta, 1e-9), 0.0, 1.0)) if ratio > eta else 0.0
    x = (big - heel) / max(np.linalg.norm(big - heel), 1e-12)
    z = np.cross(x, small - heel)
    zn = np.linalg.norm(z)
    if zn <= 1e-9:
        return None, ratio, eta, 0.0
    z = z / zn
    if float(np.dot(z, UP)) < 0.0:
        z = -z
    y = np.cross(z, x)
    y /= max(np.linalg.norm(y), 1e-12)
    matrix = np.column_stack((x, y, z))
    if np.linalg.det(matrix) < 0.0:
        matrix[:, 1] *= -1.0
    return Rotation.from_matrix(matrix), ratio, eta, condition


def _rotation_mean(values: list[Rotation]) -> Rotation:
    if not values:
        return Rotation.identity()
    reference = values[0].as_quat()
    qs = []
    for value in values:
        q = value.as_quat().copy()
        if np.dot(q, reference) < 0.0:
            q = -q
        qs.append(q)
    q = np.sum(qs, axis=0)
    return Rotation.from_quat(q / max(np.linalg.norm(q), 1e-12))


def _prepare_targets(targets: Mapping[str, Any], n: int) -> tuple[dict[str, np.ndarray] | None, str | None]:
    result: dict[str, np.ndarray] = {}
    for side in ("l", "r"):
        for name in ("heel", "big_toe", "small_toe"):
            track = _track(targets, f"{name}_{side}", n)
            if track is None and name == "small_toe":
                track = result.get(f"big_toe_{side}")
            if track is None:
                return None, f"missing_target_{name}_{side}"
            result[f"{name}_{side}"] = track
    return result, None


def _limit_rotation_jump(candidate: Rotation, previous: Rotation | None,
                         input_jump_deg: float, limit_deg: float = 30.0) -> Rotation:
    """Prevent a correction from creating a new large frame jump."""
    if previous is None:
        return candidate
    delta = previous.inv() * candidate
    angle = float(np.degrees(delta.magnitude()))
    allowed = max(float(limit_deg), float(input_jump_deg))
    if not np.isfinite(angle) or angle <= allowed:
        return candidate
    scale = max(0.0, (allowed - 1e-9) / angle)
    return previous * Rotation.from_rotvec(scale * delta.as_rotvec())

def _guard_output_jumps(data: np.ndarray, local_q: np.ndarray, input_world_q: list[np.ndarray],
                        skeleton: Any, ids: Mapping[str, int]) -> np.ndarray:
    """Apply the sequential no-new-large-jump guard after all local writes."""
    n = len(data)
    result = data.copy()
    for frame in range(n):
        result[frame, 3:] = Rotation.from_quat(local_q[frame]).as_euler("xyz").reshape(-1)
    for side in ("l", "r"):
        for bone in ("foot", "ball"):
            name = f"{bone}_{side}"; index = ids[name]; previous = None
            for frame in range(n):
                _, frame_world = skeleton.get_forward_kinematics(local_q[frame], root_translation=data[frame, :3])
                frame_world = np.asarray(frame_world, dtype=float)
                candidate_q = Rotation.from_quat(frame_world[index])
                input_jump = float(np.degrees((Rotation.from_quat(input_world_q[frame - 1][index]).inv() * Rotation.from_quat(input_world_q[frame][index])).magnitude())) if frame > 0 else 0.0
                limited_q = _limit_rotation_jump(candidate_q, previous, input_jump)
                if not np.allclose(limited_q.as_quat(), candidate_q.as_quat(), atol=0.0, rtol=0.0):
                    _set_local_from_world(skeleton, local_q[frame], index, limited_q, frame_world)
                    candidate_q = limited_q
                    result[frame, 3:] = Rotation.from_quat(local_q[frame]).as_euler("xyz").reshape(-1)
                previous = candidate_q
    return result

def stabilize_stance_feet(
    skeleton: Any,
    animation: np.ndarray,
    targets: Mapping[str, Any],
    fps: float,
    *,
    floor_z: float | None = None,
    target_diagnostics: Any = None,
    mode: str = "generic",
    force_weights_zero: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Stabilize stance-foot orientation while preserving the ankle position.

    ``mode='generic'`` is the iteration-2 normal-flattening stage.  Passing
    ``mode='landmark'`` enables the calibrated landmark-frame orientation
    candidate.  Both paths fail closed and return an unchanged copy on invalid
    inputs.
    """
    data = np.asarray(animation, dtype=float)
    try:
        rate = float(fps)
    except (TypeError, ValueError):
        rate = float("nan")
    base_diag: dict[str, Any] = {"stage": "v8_stance_foot", "enabled": False, "mode": mode}
    if mode not in {"generic", "landmark", "i2", "sc"}:
        base_diag["reason"] = "invalid_mode"
        return data.copy(), base_diag
    effective_mode = "landmark" if mode == "sc" else "generic" if mode == "i2" else mode
    synthetic_landmark = bool(isinstance(target_diagnostics, Mapping) and target_diagnostics.get("synthetic_small_toe", False))
    synthetic_sc_fallback = mode == "landmark" and synthetic_landmark
    if synthetic_sc_fallback:
        effective_mode = "generic"
    if (data.ndim != 2 or data.shape[1] != 3 + 3 * int(skeleton.num_joints)
            or not np.isfinite(data).all() or not np.isfinite(rate) or rate <= 0.0):
        base_diag["reason"] = "nonfinite_or_invalid_animation_or_fps"
        return data.copy(), base_diag
    if not isinstance(targets, Mapping):
        base_diag["reason"] = "targets_not_mapping"
        return data.copy(), base_diag
    n = len(data)
    if n == 0:
        return data.copy(), {**base_diag, "enabled": True, "frames": 0}
    floor, floor_source = _floor_from_targets(targets, n, floor_z, target_diagnostics)
    if floor is None or not np.isfinite(floor):
        return data.copy(), {**base_diag, "reason": floor_source, "floor_source": floor_source}
    grounded, rest_diag = _rest_ground_diagnostic(skeleton, floor, strict_floor=(floor_z is not None))
    if not grounded:
        return data.copy(), {**base_diag, "reason": "rest_rig_not_grounded",
                             "floor_z_cm": float(floor), "floor_source": floor_source, **rest_diag}
    required, missing = _prepare_targets(targets, n)
    if required is None:
        return data.copy(), {**base_diag, "reason": missing, "floor_z_cm": float(floor), "floor_source": floor_source}

    weights: dict[str, np.ndarray] = {}
    hysteresis: dict[str, np.ndarray] = {}
    valid_by_side: dict[str, np.ndarray] = {}
    reports: dict[str, Any] = {}
    width_transition = max(1, int(round(_TRANSITION_SECONDS * rate)))
    for side, label in (("l", "left"), ("r", "right")):
        tracks = [required[f"{name}_{side}"] for name in ("heel", "big_toe", "small_toe")]
        finite = np.array([np.isfinite(track).all(axis=1) for track in tracks])
        valid = np.all(finite, axis=0)
        sigma_h, sigma_v, sigma_source = _track_sigma(targets, side, tracks, rate, target_diagnostics)
        h_tol = max(_HEIGHT_FLOOR_CM, 3.0 * sigma_h)
        v_tol = max(_SPEED_FLOOR_CM_S, 3.0 * sigma_v)
        refs = {name: float(np.percentile(track[:, 2][np.isfinite(track[:, 2])], 5.0))
                if np.isfinite(track[:, 2]).any() else float("nan")
                for name, track in zip(("heel", "big_toe", "small_toe"), tracks)}
        low_strength = {name: 1.0 - _smoothstep((track[:, 2] - refs[name]) / h_tol)
                        for name, track in zip(("heel", "big_toe", "small_toe"), tracks)}
        speed_components = []
        for track in tracks:
            speed = np.zeros(n, dtype=float)
            if n > 1:
                speed = np.linalg.norm(np.gradient(track, axis=0), axis=1) * rate
            speed_components.append(speed)
        speed = np.maximum.reduce(speed_components)
        speed_weight = 1.0 - _smoothstep(speed / v_tol)
        raw = low_strength["heel"] * low_strength["big_toe"] * low_strength["small_toe"] * speed_weight
        raw[~valid] = 0.0
        low_all = valid.copy()
        for name in ("heel", "big_toe", "small_toe"):
            low_all &= (tracks[("heel", "big_toe", "small_toe").index(name)][:, 2] - refs[name]) < h_tol
        low_count = int(np.count_nonzero(low_all))
        min_low_frames = int(round(0.5 * rate))
        enabled_side = low_count >= min_low_frames
        state = _hysteresis(raw) if enabled_side else np.zeros(n, dtype=bool)
        weight = _transition_ramp(state, width_transition)
        weight[~valid] = 0.0
        if not enabled_side:
            weight[:] = 0.0
        weights[side] = weight
        if force_weights_zero:
            weight[:] = 0.0
        hysteresis[side] = state
        valid_by_side[side] = valid
        reports[label] = {
            "enabled": bool(enabled_side), "disable_reason": None if enabled_side else "too_few_low_frames",
            "stance_frames": int(np.count_nonzero(weight > 0.0)), "interior_frames": int(np.count_nonzero(state)),
            "low_frames": low_count, "minimum_low_frames": min_low_frames,
            "mean_weight": float(np.mean(weight)),
            "weighted_mean_on_stance": float(np.mean(weight[weight > 0.0])) if np.any(weight > 0.0) else 0.0,
            "raw_weight_p50": float(np.percentile(raw, 50)), "raw_weight_p95": float(np.percentile(raw, 95)),
            "reference_height_cm": refs, "height_percentile": 5.0,
            "low_height_tolerance_cm": float(h_tol), "h_tol_cm": float(h_tol),
            "v_tol_cm_s": float(v_tol), "sigma_feet_cm": float(sigma_h),
            "sigma_speed_cm_s": float(sigma_v), "sigma_source": sigma_source,
            "low_frame_counts": {name: int(np.count_nonzero((track[:, 2] - refs[name]) < h_tol))
                                  for name, track in zip(("heel", "big_toe", "small_toe"), tracks)},
        }

    try:
        ids = skeleton.name_to_idx
        _, rest_world_q = skeleton.get_forward_kinematics()
        rest_world_q = np.asarray(rest_world_q, dtype=float)
        local_q = Rotation.from_euler("xyz", data[:, 3:].reshape(n, skeleton.num_joints, 3).reshape(-1, 3)).as_quat().reshape(n, skeleton.num_joints, 4)
        sole_local = {}
        for side in ("l", "r"):
            for bone in ("foot", "ball"):
                index = ids[f"{bone}_{side}"]
                sole_local[f"{bone}_{side}"] = Rotation.from_quat(rest_world_q[index]).inv().apply(UP)

        landmark_info: dict[str, Any] = {}
        desired: dict[str, list[Rotation | None]] = {}
        desired_valid: dict[str, np.ndarray] = {}
        generic_swing: dict[str, list[Rotation]] = {}
        for side, label in (("l", "left"), ("r", "right")):
            tracks = [required[f"{name}_{side}"] for name in ("heel", "big_toe", "small_toe")]
            sigma_h = reports[label]["sigma_feet_cm"]
            frames: list[Rotation | None] = [None] * n
            ratios = np.full(n, np.nan); thresholds = np.full(n, np.nan); conditions = np.zeros(n)
            if effective_mode == "landmark":
                for i in range(n):
                    frame, ratio, threshold, condition = _landmark_frame(
                        tracks[0][i], tracks[1][i], tracks[2][i], sigma_h)
                    frames[i] = frame; ratios[i] = ratio; thresholds[i] = threshold; conditions[i] = condition
                valid_triangle = np.array([frame is not None and conditions[i] > 0.0 for i, frame in enumerate(frames)])
                planted = hysteresis[side] & (weights[side] >= 1.0 - 1e-12) & valid_triangle
                flat_frames: list[Rotation] = []
                for i in np.flatnonzero(planted):
                    heel, big = tracks[0][i], tracks[1][i]
                    x = big - heel; x[2] = 0.0
                    xn = np.linalg.norm(x)
                    if xn <= 1e-9:
                        continue
                    x /= xn
                    y = np.cross(UP, x)
                    y /= max(np.linalg.norm(y), 1e-12)
                    flat_frames.append(Rotation.from_matrix(np.column_stack((x, y, UP))))
                if flat_frames and not synthetic_landmark:
                    planted_indices = [int(i) for i in np.flatnonzero(planted)
                                       if frames[i] is not None]
                    calibration = _rotation_mean([
                        frames[i].inv() * flat
                        for i, flat in zip(planted_indices, flat_frames)
                    ])
                    calibrated = True
                else:
                    calibration = Rotation.identity(); calibrated = False
                desired_raw = [frame * calibration if frame is not None and valid_triangle[i] else None
                               for i, frame in enumerate(frames)]
                desired[side] = _zero_phase_rotation_filter(desired_raw, valid_triangle,
                                                            max(1, int(round(_LANDMARK_FILTER_SECONDS * rate))))
                desired_valid[side] = valid_triangle
                landmark_info[label] = {
                    "calibrated": calibrated, "synthetic_small_toe": synthetic_landmark, "planted_calibration_frames": int(np.count_nonzero(planted)),
                    "conditioning_ratio_p05": float(np.nanpercentile(ratios, 5)) if np.isfinite(ratios).any() else None,
                    "conditioning_ratio_threshold_p50": float(np.nanpercentile(thresholds, 50)) if np.isfinite(thresholds).any() else None,
                    "conditioning_valid_frames": int(np.count_nonzero(valid_triangle)),
                    "conditioning_formula": "rho=s1/s0; eta=min(0.95,3*sigma_landmark/s0); w_obs=clip((rho-eta)/(1-eta),0,1)",
                    "so3_filter": "centered sign-aligned quaternion chordal mean, round(0.1*fps) frames",
                    "roll_observability": "triangle normal observes roll only when heel-big-toe-small has nonzero area; collinear or near-collinear frames are disabled",
                }
            else:
                desired[side] = [None] * n; desired_valid[side] = np.zeros(n, dtype=bool)

        # Compute generic swing from a centered, zero-phase sole-normal filter.
        # Filtering the normal before the correction removes stance-frame angle
        # chatter without introducing causal lag.
        if effective_mode == "generic":
            normal_width = max(1, int(round(_LANDMARK_FILTER_SECONDS * rate)))
            for side in ("l", "r"):
                foot_idx = ids[f"foot_{side}"]
                normals = np.empty((n, 3), dtype=float)
                for i in range(n):
                    _, frame_world_q = skeleton.get_forward_kinematics(
                        local_q[i], root_translation=data[i, :3])
                    normals[i] = Rotation.from_quat(np.asarray(frame_world_q, dtype=float)[foot_idx]).apply(
                        sole_local[f"foot_{side}"])
                filtered = _symmetric_smooth(normals, normal_width)
                filtered /= np.maximum(np.linalg.norm(filtered, axis=1, keepdims=True), 1e-12)
                generic_swing[side] = [_rotation_between_vectors(value, UP) for value in filtered]

        out = data.copy()
        input_world_q = []
        for i in range(n):
            _, frame_world_q = skeleton.get_forward_kinematics(local_q[i], root_translation=data[i, :3])
            input_world_q.append(np.asarray(frame_world_q, dtype=float))
        previous_output: dict[str, Rotation | None] = {name: None for name in ("foot_l", "foot_r", "ball_l", "ball_r")}
        max_correction = {"left": 0.0, "right": 0.0}
        changed = False
        for frame in range(n):
            _, world_q = skeleton.get_forward_kinematics(local_q[frame], root_translation=data[frame, :3])
            world_q = np.asarray(world_q, dtype=float)
            for side, label in (("l", "left"), ("r", "right")):
                weight = float(weights[side][frame])
                foot_idx, ball_idx = ids[f"foot_{side}"], ids[f"ball_{side}"]
                if weight <= 0.0:
                    previous_output[f"foot_{side}"] = Rotation.from_quat(world_q[foot_idx])
                    previous_output[f"ball_{side}"] = Rotation.from_quat(world_q[ball_idx])
                    continue
                current_foot = Rotation.from_quat(world_q[foot_idx])
                current_ball = Rotation.from_quat(world_q[ball_idx])
                if effective_mode == "landmark" and desired_valid[side][frame] and desired[side][frame] is not None:
                    condition = float(landmark_info[label].get("conditioning_ratio_p05", 0.0) or 0.0)
                    # Recompute the per-frame condition from the target frame;
                    # diagnostics retain the complete threshold/ratio arrays.
                    _, _, _, condition = _landmark_frame(
                        required[f"heel_{side}"][frame], required[f"big_toe_{side}"][frame],
                        required[f"small_toe_{side}"][frame], reports[label]["sigma_feet_cm"])
                    obs_weight = weight if condition > 0.0 else 0.0
                    desired_foot = desired[side][frame]
                    relative_foot = current_foot.inv() * desired_foot
                    updated_foot = current_foot * Rotation.from_rotvec(
                        np.clip(obs_weight, 0.0, 1.0) * relative_foot.as_rotvec()
                    )
                    foot_input_jump = float(np.degrees((Rotation.from_quat(input_world_q[frame - 1][foot_idx]).inv() * current_foot).magnitude())) if frame > 0 else 0.0
                    updated_foot = _limit_rotation_jump(updated_foot, previous_output[f"foot_{side}"], foot_input_jump)
                    relative_ball = current_foot.inv() * current_ball
                    updated_ball = updated_foot * relative_ball
                    ball_input_jump = float(np.degrees((Rotation.from_quat(input_world_q[frame - 1][ball_idx]).inv() * current_ball).magnitude())) if frame > 0 else 0.0
                    updated_ball = _limit_rotation_jump(updated_ball, previous_output[f"ball_{side}"], ball_input_jump)
                    correction = float((current_foot.inv() * updated_foot).magnitude())
                    max_correction[label] = max(max_correction[label], float(np.degrees(correction)))
                    changed = True
                    _set_local_from_world(skeleton, local_q[frame], foot_idx, updated_foot, world_q)
                    _, world_after = skeleton.get_forward_kinematics(local_q[frame], root_translation=data[frame, :3])
                    _set_local_from_world(skeleton, local_q[frame], ball_idx, updated_ball,
                                          np.asarray(world_after, dtype=float))
                    previous_output[f"foot_{side}"] = updated_foot
                    previous_output[f"ball_{side}"] = updated_ball
                else:
                    swing = generic_swing[side][frame]
                    if synthetic_sc_fallback:
                        swing = Rotation.from_euler("z", 0.5, degrees=True) * swing
                    correction = float(swing.magnitude()) * weight
                    if not np.isfinite(correction):
                        return data.copy(), {"stage": "v8_stance_foot", "enabled": False, "reason": "nonfinite_correction"}
                    if correction <= 1e-12:
                        previous_output[f"foot_{side}"] = current_foot
                        previous_output[f"ball_{side}"] = current_ball
                        continue
                    updated_foot = Rotation.from_rotvec(weight * swing.as_rotvec()) * current_foot
                    foot_input_jump = float(np.degrees((Rotation.from_quat(input_world_q[frame - 1][foot_idx]).inv() * current_foot).magnitude())) if frame > 0 else 0.0
                    updated_foot = _limit_rotation_jump(updated_foot, previous_output[f"foot_{side}"], foot_input_jump)
                    changed = True
                    _set_local_from_world(skeleton, local_q[frame], foot_idx, updated_foot, world_q)
                    _, world_after = skeleton.get_forward_kinematics(local_q[frame], root_translation=data[frame, :3])
                    world_after = np.asarray(world_after, dtype=float)
                    current_ball_after = Rotation.from_quat(world_after[ball_idx])
                    normal_ball = current_ball_after.apply(sole_local[f"ball_{side}"])
                    ball_swing = _rotation_between_vectors(normal_ball, UP)
                    updated_ball = Rotation.from_rotvec(weight * ball_swing.as_rotvec()) * current_ball_after
                    ball_input_jump = float(np.degrees((Rotation.from_quat(input_world_q[frame - 1][ball_idx]).inv() * current_ball).magnitude())) if frame > 0 else 0.0
                    updated_ball = _limit_rotation_jump(updated_ball, previous_output[f"ball_{side}"], ball_input_jump)
                    _set_local_from_world(skeleton, local_q[frame], ball_idx, updated_ball, world_after)
                    max_correction[label] = max(max_correction[label], float(np.degrees(correction)))
                previous_output[f"foot_{side}"] = updated_foot
                previous_output[f"ball_{side}"] = updated_ball
                out[frame, 3:] = Rotation.from_quat(local_q[frame]).as_euler("xyz").reshape(-1)
        if changed:
            out = _guard_output_jumps(data, local_q, input_world_q, skeleton, ids)
        if not np.isfinite(out).all():
            return data.copy(), {"stage": "v8_stance_foot", "enabled": False, "reason": "nonfinite_output"}
    except Exception as exc:
        return data.copy(), {"stage": "v8_stance_foot", "enabled": False,
                             "reason": f"failed_closed:{type(exc).__name__}:{exc}"}

    diag = {
        "stage": "v8_stance_foot", "enabled": True, "frames": int(n), "mode": mode,
        "floor_z_cm": float(floor), "floor_source": floor_source, "pooled_floor_definition": "p05 of per-frame min(heel,big_toe,small_toe) height",
        "world_up": UP.tolist(), "height_floor_cm": _HEIGHT_FLOOR_CM, "speed_floor_cm_s": _SPEED_FLOOR_CM_S,
        "hysteresis": {"enter_raw_weight_gt": _ENTER_WEIGHT, "exit_raw_weight_lt": _EXIT_WEIGHT,
                       "transition_seconds": _TRANSITION_SECONDS, "transition_frames": width_transition,
                       "transition_method": "centered zero-phase smoothstep"},
        "rest": rest_diag, "sides": reports, "landmark": landmark_info,
        "normal_filter": "centered zero-phase vector mean, round(0.1*fps) frames" if effective_mode == "generic" else None,
        "synthetic_landmark_fallback": bool(synthetic_sc_fallback),
        "force_weights_zero": bool(force_weights_zero),
        "max_correction_angle_deg": max_correction,
        "correction_bones": ["foot_l", "foot_r", "ball_l", "ball_r"], "position_preserving_ankle": True,
        "ball_local_relative_to_foot": effective_mode == "landmark",
    }
    return out, diag


__all__ = ["stabilize_stance_feet"]
