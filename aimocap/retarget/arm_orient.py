"""Side-agnostic arm orientation from world-space observations.

This module implements the root-cause fix for the right-arm axial roll error.
The key insight: a rotation about a bone's own long axis leaves its child joint
position invariant, so rewriting ONLY the twist component in GLOBAL space is
provably position-preserving.

SPACE SUFFIXES (mandatory):
  _w    world space
  _loc  bone-local
  _rest rig rest pose
  _obs  observed / triangulated

CORE CONVENTION:
  G(t)[b] = D[b](t) @ G_rest[b]     # D is a WORLD delta from rig rest global
  L(t)[b] = G(t)[parent].inv() * G(t)[b]

INVARIANT: this module contains zero `if side ==` branches. If you need one,
the rest basis is wrong. Fix it in load, not here.
"""

import numpy as np
from scipy.spatial.transform import Rotation


# ── Public configuration (exposed for tuning) ────────────────────────────────
BEND_GATE_LO_DEG = 20.0       # below: humeral roll unobservable from arm triangle
BEND_GATE_HI_DEG = 40.0       # above: fully trusted. Linear ramp between.
HUMERAL_AXIAL_LIMIT_DEG = 90.0    # gleno-humeral internal/external ROM, hanging arm
ROLL_MAX_STEP_DEG = 12.0     # continuity clamp, 30 fps human shoulder
PROJ_DEGENERACY_MIN = 0.15   # reject signed-angle when projection collapses
ARM_MIN_SEGMENT_CM = 5.0
ARM_MAX_DIRECTION_STEP_DEG = 12.0
ARM_HOLD_FRAMES = 3


def unit(v: np.ndarray, name: str) -> np.ndarray:
    """Normalize vector, raising on degeneracy."""
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if not np.isfinite(n) or n < 1e-10:
        raise ValueError(f"{name}: degenerate vector (norm={n})")
    return v / n


def resolve_bone_axis_parent_local(
    rest_positions_world: np.ndarray,
    parent_global_rest: Rotation,
    parent_idx: int,
    child_idx: int,
) -> np.ndarray:
    """Bone long axis in PARENT LOCAL frame at rest.
    
    Transforms the world-space parent→child offset into the parent's
    local frame. This is the correct input to solve_upperarm_global.
    """
    delta_w = rest_positions_world[child_idx] - rest_positions_world[parent_idx]
    delta_parent_local = parent_global_rest.inv().apply(delta_w)
    return unit(delta_parent_local, "bone axis parent-local")


def _orthonormalize_rows(M: np.ndarray) -> np.ndarray:
    """Return nearest rotation matrix via SVD with det=+1."""
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1.0
        R = U @ Vt
    assert np.linalg.det(R) > 0.999, f"improper rotation, det={np.linalg.det(R)}"
    return R




def shortest_arc(u_w: np.ndarray, v_w: np.ndarray) -> Rotation:
    """Minimal-twist rotation aligning u_w to v_w in WORLD space.
    
    Handles antiparallel case deterministically (cross with +X, fallback +Y).
    Returns Rotation object.
    """
    u = u_w / (np.linalg.norm(u_w) + 1e-12)
    v = v_w / (np.linalg.norm(v_w) + 1e-12)
    d = np.dot(u, v)
    if d > 1.0 - 1e-12:
        return Rotation.identity()
    if d < -1.0 + 1e-9:
        axis = np.cross(u, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(u, np.array([0.0, 1.0, 0.0]))
        axis = axis / np.linalg.norm(axis)
        return Rotation.from_rotvec(np.pi * axis)
    axis = np.cross(u, v)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-12:
        return Rotation.identity()
    axis = axis / axis_norm
    angle = np.arccos(np.clip(d, -1.0, 1.0))
    return Rotation.from_rotvec(angle * axis)


def signed_angle_about(u_w: np.ndarray, v_w: np.ndarray, axis_w: np.ndarray
                        ) -> tuple[float, bool, str]:
    """Signed angle from u_w to v_w about axis_w, with degeneracy check.
    
    Both u_w and v_w are projected onto the plane normal to axis_w.
    Returns: (angle_rad, valid, reason_string)
    """
    axis = axis_w / (np.linalg.norm(axis_w) + 1e-12)
    u_proj = u_w - np.dot(u_w, axis) * axis
    v_proj = v_w - np.dot(v_w, axis) * axis
    u_norm = np.linalg.norm(u_proj)
    v_norm = np.linalg.norm(v_proj)
    u_orig = np.linalg.norm(u_w)
    v_orig = np.linalg.norm(v_w)
    
    if u_norm < PROJ_DEGENERACY_MIN * u_orig or v_norm < PROJ_DEGENERACY_MIN * v_orig:
        return 0.0, False, "degenerate_projection"
    
    u_proj = u_proj / u_norm
    v_proj = v_proj / v_norm
    cross = np.cross(u_proj, v_proj)
    sin_a = np.dot(cross, axis)
    cos_a = np.dot(u_proj, v_proj)
    angle = np.arctan2(sin_a, cos_a)
    return float(angle), True, "elbow_plane"


def rotation_angle_deg(a: Rotation, b: Rotation) -> float:
    """Shortest angular distance between two rotations, in degrees."""
    return float(np.degrees((a * b.inv()).magnitude()))


def project_perpendicular(v: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Normalize the component of *v* perpendicular to *axis*."""
    axis = unit(axis, "projection axis")
    v = np.asarray(v, dtype=np.float64)
    p = v - np.dot(v, axis) * axis
    n = np.linalg.norm(p)
    if n < PROJ_DEGENERACY_MIN * max(np.linalg.norm(v), 1e-12):
        raise ValueError("degenerate perpendicular projection")
    return p / n


def signed_twist_about(
    rotation_w: Rotation,
    axis_w: np.ndarray,
    reference_w: np.ndarray,
) -> float:
    """Extract the signed axial twist of ``rotation_w`` about ``axis_w``."""
    axis_w = unit(axis_w, "twist axis")
    ref = project_perpendicular(reference_w, axis_w)
    moved = project_perpendicular(rotation_w.apply(ref), axis_w)
    return float(np.arctan2(
        np.dot(np.cross(ref, moved), axis_w),
        np.dot(ref, moved),
    ))


def replace_twist(
    rotation_w: Rotation,
    axis_w: np.ndarray,
    reference_w: np.ndarray,
    target_twist_rad: float,
) -> Rotation:
    """Preserve swing while replacing, rather than adding, axial twist."""
    axis_w = unit(axis_w, "replacement axis")
    ref = project_perpendicular(reference_w, axis_w)
    current_twist = signed_twist_about(rotation_w, axis_w, ref)
    delta = target_twist_rad - current_twist
    return Rotation.from_rotvec(delta * axis_w) * rotation_w


def unwrap_against(phi: float, phi_prev: float) -> float:
    """Add k·2π to minimize |phi - phi_prev|."""
    diff = phi - phi_prev
    k_raw = diff / (2 * np.pi)
    k = int(np.round(k_raw))
    if abs(k_raw - k) > 0.5 - 1e-12:
        return phi - k * 2 * np.pi
    candidates = [k]
    if diff != 0:
        candidates.append(k - int(np.sign(diff)))
    best = min(candidates, key=lambda kk: abs(phi - kk * 2 * np.pi - phi_prev))
    return phi - best * 2 * np.pi


def limit_step(phi: float, phi_prev: float, max_step_deg: float) -> float:
    """Clamp frame-to-frame change to max_step_deg."""
    max_step = np.deg2rad(max_step_deg)
    diff = phi - phi_prev
    if abs(diff) > max_step:
        return phi_prev + np.sign(diff) * max_step
    return phi


def clamp_deg(phi_rad: float, limit_deg: float) -> float:
    """Clamp angle to [-limit, +limit] in radians."""
    limit = np.deg2rad(limit_deg)
    return np.clip(phi_rad, -limit, +limit)


class Observation:
    """Structured return for estimators — never None."""
    def __init__(self, value: float, valid: bool, weight: float,
                 source: str, reason: str = ""):
        self.value = value
        self.valid = valid
        self.weight = weight
        self.source = source
        self.reason = reason
    
    def __repr__(self):
        return (f"Observation(val={np.rad2deg(self.value):.1f}°, "
                f"valid={self.valid}, w={self.weight:.2f}, src={self.source})")


def validate_arm_observation(
    sh_w: np.ndarray,
    el_w: np.ndarray,
    wr_w: np.ndarray,
    state: dict,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray] | None, str]:
    """Reject implausible one-frame direction spikes and hold briefly."""
    pts = [np.asarray(v, dtype=np.float64) for v in (sh_w, el_w, wr_w)]
    if not all(v.shape == (3,) and np.all(np.isfinite(v)) for v in pts):
        state["invalid_run"] += 1
        return (None, "ik_only" if state["invalid_run"] > ARM_HOLD_FRAMES else "none")
    sh, el, wr = pts
    upper_len = np.linalg.norm(el - sh)
    fore_len = np.linalg.norm(wr - el)
    if min(upper_len, fore_len) < ARM_MIN_SEGMENT_CM:
        state["invalid_run"] += 1
        return (None, "ik_only" if state["invalid_run"] > ARM_HOLD_FRAMES else "none")

    upper_dir = (el - sh) / upper_len
    fore_dir = (wr - el) / fore_len
    def step_exceeded(new, old):
        return old is not None and np.degrees(np.arccos(np.clip(np.dot(new, old), -1.0, 1.0))) > ARM_MAX_DIRECTION_STEP_DEG

    spike = step_exceeded(upper_dir, state.get("prev_upper_dir")) or step_exceeded(
        fore_dir, state.get("prev_fore_dir")
    )
    if spike:
        state["invalid_run"] += 1
        if state["invalid_run"] <= ARM_HOLD_FRAMES and state.get("prev_upper_dir") is not None:
            held_el = sh + upper_len * state["prev_upper_dir"]
            held_wr = held_el + fore_len * state["prev_fore_dir"]
            return ((sh, held_el, held_wr), "held")
        return (None, "ik_only")

    state["invalid_run"] = 0
    state["prev_upper_dir"] = upper_dir
    state["prev_fore_dir"] = fore_dir
    return ((sh, el, wr), "accepted")


def solve_upperarm_global(
    *,
    D_parent: Rotation,            # world delta of clavicle from proxy rest
    G_parent_rest: Rotation,       # proxy rest GLOBAL of clavicle (parent)
    e_upper_parent: np.ndarray,    # upperarm axis in clavicle LOCAL frame
    e_fore_upper_local: np.ndarray,# forearm axis in upperarm LOCAL frame
    G_rest_upper: Rotation,        # proxy rest GLOBAL of upperarm
    sh_w: np.ndarray,
    el_w: np.ndarray,
    wr_w: np.ndarray,
    phi_prev: float,
    G_upper_ik: Rotation = None,   # IK solved upperarm global rotation
) -> tuple[Rotation, Observation]:
    """Compute upperarm global rotation: pure swing + conditional roll.
    
    SWING: uses G_upper_ik when provided (position-optimal from IK solve),
           otherwise falls back to shortest_arc alignment onto (el_w - sh_w).
    
    ROLL:  only if elbow is bent enough to define the arm plane.
           Rotates ONLY about the humerus long axis, preserving elbow position.
    
    Returns: (G_upper_w, Observation)
    """
    axis_obs_w = unit(el_w - sh_w, "measured humerus")
    axis_apply_w = axis_obs_w
    if G_upper_ik is not None:
        G_swing = G_upper_ik
        axis_ik_w = unit(
            G_upper_ik.apply(G_parent_rest.apply(unit(e_upper_parent, "upper rest axis"))),
            "IK humerus",
        )
        # The measured axis drives observability and target estimation, but
        # the actual twist rewrite must use the IK bone axis.  That axis is
        # exactly the parent-to-elbow proxy axis, so rotating around it cannot
        # move the elbow proxy even when measured geometry is noisy.
        axis_apply_w = axis_ik_w
        axis_disagreement = rotation_angle_deg(
            shortest_arc(axis_ik_w, axis_obs_w), Rotation.identity()
        )
        if axis_disagreement >= 25.0:
            return G_swing, Observation(
                0.0, False, 0.0, "none", "ik_axis_disagrees_with_observation"
            )
    else:
        a_rest_w = D_parent.apply(G_parent_rest.apply(unit(e_upper_parent, "upper rest axis")))
        S = shortest_arc(a_rest_w, axis_obs_w)
        G_swing = S * D_parent * G_rest_upper
    
    # ROLL — conditional on elbow bend
    f_obs_w = unit(wr_w - el_w, "measured forearm")
    bend = np.degrees(np.arccos(np.clip(np.dot(axis_obs_w, f_obs_w), -1.0, 1.0)))
    f_carried_w = G_swing.apply(unit(e_fore_upper_local, "forearm rest axis"))

    if bend < BEND_GATE_LO_DEG:
        target_phi = 0.0
        source = "swing_only"
        valid = True
        reason = f"bend={bend:.1f}°"
    else:
        obs = signed_angle_about(f_carried_w, f_obs_w, axis_obs_w)
        if not obs[1]:
            return G_swing, Observation(0.0, False, 0.0, "none", obs[2])
        measured_phi = clamp_deg(obs[0], HUMERAL_AXIAL_LIMIT_DEG)
        if bend < BEND_GATE_HI_DEG:
            # The transition is deliberately held, not linearly blended.
            target_phi = phi_prev
            source = "transition_rejected"
            valid = False
        else:
            # A small scalar low-pass plus a hard per-frame gate suppresses
            # triangulation spikes without touching the global temporal filter.
            filtered_phi = 0.5 * phi_prev + 0.5 * measured_phi
            target_phi = limit_step(filtered_phi, phi_prev, ROLL_MAX_STEP_DEG)
            source = "elbow_plane"
            valid = True
        reason = f"bend={bend:.1f}°"

    target_phi = limit_step(target_phi, phi_prev, ROLL_MAX_STEP_DEG)
    target_phi = clamp_deg(target_phi, HUMERAL_AXIAL_LIMIT_DEG)
    try:
        G = replace_twist(G_swing, axis_apply_w, f_carried_w, target_phi)
    except ValueError as exc:
        return G_swing, Observation(0.0, False, 0.0, "none", str(exc))
    return G, Observation(target_phi, valid, 1.0 if valid else 0.0, source, reason)


def solve_forearm_global(
    *,
    G_upper_w: Rotation,            # already solved upperarm global
    G_rest_fore: Rotation,          # rig rest global of lowerarm
    e_fore_loc: np.ndarray,         # forearm long axis in upperarm local (unit)
    el_w: np.ndarray,               # observed elbow position (world)
    wr_w: np.ndarray,               # observed wrist position (world)
    phi_prev: float                 # previous frame's forearm roll (rad)
) -> tuple[Rotation, Observation]:
    """Forearm = pure swing onto observed direction. No pronation from 23 pts."""
    f_rest_w = G_upper_w.apply(e_fore_loc)
    f_obs_w = wr_w - el_w
    f_obs_w = f_obs_w / (np.linalg.norm(f_obs_w) + 1e-12)
    
    S = shortest_arc(f_rest_w, f_obs_w)
    G_fore = S * G_upper_w  # No additional twist
    
    return G_fore, Observation(0.0, True, 1.0, "swing_only", "no_pronation_data")


def solve_hand_global_v2(
    *,
    G_fore_w: Rotation,
    G_rest_fore: Rotation,
    G_rest_hand: Rotation
) -> Rotation:
    """Hand global = forearm delta applied to hand rest."""
    D_fore = G_fore_w * G_rest_fore.inv()
    return D_fore * G_rest_hand
