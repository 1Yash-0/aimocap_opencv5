"""Constrained swing-twist rotation solver.

Replaces free-roll ``Rotation.align_vectors`` in the IK chain walk. Given a
bone's rest direction and its desired direction, plus an optional roll-pinning
child direction, returns the rotation that maps the bone direction AND fixes
the roll from the child.

Pure numpy/scipy. No Blender, no ufbx.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


# Shared guard used by analytic initialization and the optional limb-twist
# pin fade. Keeping the threshold in one module prevents the fade interval
# from drifting from the straight-limb fallback branch.
STRAIGHT_FALLBACK_THRESHOLD_DEG = 35.0


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n > 1e-12:
        return v / n
    # Return unit vector along +X as fallback for zero vectors
    return np.array([1.0, 0.0, 0.0])


def arm_frame(E_minus_S: np.ndarray, W_minus_E: np.ndarray,
              sin_lo: float = 0.05, sin_hi: float = 0.9) -> tuple[np.ndarray | None, np.ndarray | None, float | None]:
    """
    Build an orthonormal frame from the arm triangle (shoulder->elbow->wrist).

    Parameters
    ----------
    E_minus_S : elbow - shoulder  (upper arm vector)
    W_minus_E : wrist - elbow     (forearm vector)
    sin_lo, sin_hi : blend thresholds on sin(angle between upper and forearm)

    Returns
    -------
    F : (3,3) orthonormal frame with columns [x_forward, y_up, z_lateral]
        x = normalized(W-E) × normalized(E-S)  -> forward (palm normal)
        y = normalized(E-S)                    -> up (along upper arm)
        z = cross(y, x)                        -> lateral
    sin_blend : float in [0,1] or None if degenerate/straight arm
    angle_deg : twist angle in degrees (signed, + = pronation for right arm)
    """
    u = E_minus_S
    f = W_minus_E
    nu = np.linalg.norm(u)
    nf = np.linalg.norm(f)
    if nu < 1e-6 or nf < 1e-6:
        return None, None, None
    u = u / nu
    f = f / nf
    # cross = u × f  (forward when arm is straight)
    cross = np.cross(u, f)
    n_cross = np.linalg.norm(cross)
    if n_cross < 1e-6:
        return None, None, None
    x = cross / n_cross          # forward (palm normal)
    z = u                        # up along upper arm
    y = np.cross(z, x)           # lateral
    y = y / (np.linalg.norm(y) + 1e-12)
    x = np.cross(y, z)           # re-orthogonalize
    x = x / (np.linalg.norm(x) + 1e-12)
    z = np.cross(x, y)
    z = z / (np.linalg.norm(z) + 1e-12)
    F = np.column_stack([x, y, z])  # columns = [x, y, z]
    # twist angle: angle between rest lateral axis and current lateral axis projected onto forearm axis
    # This is a simplified version; the actual twist is computed in mocap_ik.py
    sin_blend = float(np.clip(n_cross, sin_lo, sin_hi))
    return F, None, sin_blend




def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n > 1e-12:
        return v / n
    # Return unit vector along +X as fallback for zero vectors
    return np.array([1.0, 0.0, 0.0])


def constrained_rotation(
    rest_dir: np.ndarray,
    desired_dir: np.ndarray,
    roll_child_rest: np.ndarray | None = None,
    roll_child_desired: np.ndarray | None = None,
) -> Rotation:
    """Rotation mapping rest_dir -> desired_dir, with roll fixed by child vectors.

    Algorithm (swing-twist decomposition in direction space):
      1. Swing = shortest rotation rest_dir -> desired_dir.
      2. If roll children given, apply swing to roll_child_rest, then compute the
         residual rotation ABOUT desired_dir that aligns the swung child onto
         roll_child_desired (project both into the plane perpendicular to
         desired_dir, take the in-plane angle). That residual is the twist.
      3. Result = twist * swing.
    If no roll children, return swing only (minimal roll = free, same as before).
    """
    rest = _normalize(np.asarray(rest_dir, dtype=np.float64))
    desired = _normalize(np.asarray(desired_dir, dtype=np.float64))
    
    # If desired direction is zero/near-zero, no swing needed - return identity
    if np.linalg.norm(desired) < 1e-12:
        return Rotation.identity()

    # Swing: shortest rotation rest -> desired. align_vectors on a single vector
    # is well-defined and picks the minimal rotation (zero twist component).
    swing, _ = Rotation.align_vectors(desired.reshape(1, 3), rest.reshape(1, 3))

    if roll_child_rest is None or roll_child_desired is None:
        return swing

    cr = _normalize(np.asarray(roll_child_rest, dtype=np.float64))
    cd = _normalize(np.asarray(roll_child_desired, dtype=np.float64))

    # If the roll child is nearly parallel to the bone direction, it carries
    # no useful roll information (the projection into the perpendicular plane
    # is noise-dominated).  This happens when a limb is nearly straight: the
    # hand/foot is almost collinear with the shin/forearm, so projecting it
    # perpendicular to the bone gives a near-zero vector with an arbitrary
    # direction.  The resulting twist is garbage -- often 40-100 deg of
    # arbitrary roll that makes the limb "cave in" or the hand spin.  Skip
    # the twist and return swing-only when the roll child is within ~35 deg
    # of the bone (covers near-straight arms and legs; a bent limb at >35deg
    # has enough perpendicular signal for a reliable twist).
    if abs(np.dot(cd, desired)) > np.cos(np.deg2rad(STRAIGHT_FALLBACK_THRESHOLD_DEG)):
        return swing

    swung_child = swing.apply(cr)
    # project both into plane perp to desired
    sc = swung_child - np.dot(swung_child, desired) * desired
    cd_p = cd - np.dot(cd, desired) * desired
    sc_n, cd_n = np.linalg.norm(sc), np.linalg.norm(cd_p)
    if sc_n < 0.05 or cd_n < 1e-9:
        return swing
    sc_u, cd_u = sc / sc_n, cd_p / cd_n
    # in-plane angle about desired axis
    cos_a = np.clip(np.dot(sc_u, cd_u), -1.0, 1.0)
    axis_angle = np.arctan2(np.dot(np.cross(sc_u, cd_u), desired), cos_a)
    twist = Rotation.from_rotvec(axis_angle * desired)
    return twist * swing


def bend_plane_rotation(
    rest_dir: np.ndarray,
    desired_dir: np.ndarray,
    roll_child_rest: np.ndarray | None = None,
    roll_child_desired: np.ndarray | None = None,
) -> Rotation | None:
    """Build the bend-plane rotation without the collinearity guard.

    This is the continuous-anchor companion to :func:`constrained_rotation`.
    It deliberately skips only the approximately-35-degree observability
    switch; the projection degeneracy fallback remains and returns ``None``.
    """
    rest = _normalize(np.asarray(rest_dir, dtype=np.float64))
    desired_raw = np.asarray(desired_dir, dtype=np.float64)
    if not np.all(np.isfinite(desired_raw)) or np.linalg.norm(desired_raw) < 1e-12:
        return None
    desired = _normalize(desired_raw)
    swing, _ = Rotation.align_vectors(desired.reshape(1, 3), rest.reshape(1, 3))
    if roll_child_rest is None or roll_child_desired is None:
        return swing
    cr = _normalize(np.asarray(roll_child_rest, dtype=np.float64))
    cd_raw = np.asarray(roll_child_desired, dtype=np.float64)
    if not np.all(np.isfinite(cd_raw)) or np.linalg.norm(cd_raw) < 1e-12:
        return None
    cd = _normalize(cd_raw)
    swung_child = swing.apply(cr)
    sc = swung_child - np.dot(swung_child, desired) * desired
    cd_p = cd - np.dot(cd, desired) * desired
    sc_n, cd_n = np.linalg.norm(sc), np.linalg.norm(cd_p)
    if sc_n < 0.05 or cd_n < 1e-9:
        return None
    sc_u, cd_u = sc / sc_n, cd_p / cd_n
    cos_a = np.clip(np.dot(sc_u, cd_u), -1.0, 1.0)
    axis_angle = np.arctan2(np.dot(np.cross(sc_u, cd_u), desired), cos_a)
    return Rotation.from_rotvec(axis_angle * desired) * swing


# ── Swing-twist decomposition and clamping ───────────────────────────────

def swing_twist_decompose(
    q: Rotation, bone_axis: np.ndarray,
) -> tuple[Rotation, Rotation]:
    """Decompose rotation ``q`` into (swing, twist) around ``bone_axis``.

    The twist component is the rotation about ``bone_axis``; the swing is
    everything else.  ``q = swing * twist`` (swing applied first, then twist).

    Args:
        q:          the rotation to decompose.
        bone_axis:  unit vector defining the bone's longitudinal axis
                    (e.g. the rest direction of the bone in parent space).

    Returns:
        ``(swing, twist)`` — two ``Rotation`` objects whose product equals ``q``.
    """
    axis = _normalize(np.asarray(bone_axis, dtype=np.float64))
    # scipy convention: as_quat() returns [x, y, z, w] (scalar last)
    x, y, z, w = q.as_quat()
    # Project the vector part onto the bone axis
    proj = np.dot([x, y, z], axis) * axis
    # Twist quaternion: [proj_x, proj_y, proj_z, w] (scalar last)
    twist_quat = np.array([proj[0], proj[1], proj[2], w])
    n = np.linalg.norm(twist_quat)
    if n < 1e-12:
        twist = Rotation.identity()
    else:
        twist_quat = twist_quat / n
        twist = Rotation.from_quat(twist_quat)
    swing = q * twist.inv()
    return swing, twist


def clamp_twist(
    q: Rotation, bone_axis: np.ndarray, max_twist_deg: float,
) -> Rotation:
    """Clamp the twist component of ``q`` around ``bone_axis``.

    Decomposes ``q`` into swing + twist, clamps the twist angle to
    ``±max_twist_deg``, and reconstructs: ``q_clamped = swing * twist_clamped``.

    Args:
        q:              the rotation to clamp.
        bone_axis:      unit vector defining the bone's longitudinal axis.
        max_twist_deg:  maximum allowed twist angle in degrees.

    Returns:
        ``Rotation`` with twist clamped to ``±max_twist_deg``.
    """
    swing, twist = swing_twist_decompose(q, bone_axis)
    # Twist is a rotation about bone_axis — extract its angle
    twist_rotvec = twist.as_rotvec()
    twist_angle = np.dot(twist_rotvec, bone_axis)  # signed angle
    max_rad = np.deg2rad(max_twist_deg)
    if abs(twist_angle) > max_rad:
        twist_angle = np.clip(twist_angle, -max_rad, max_rad)
        twist = Rotation.from_rotvec(twist_angle * bone_axis)
    return swing * twist


# ── Helpers for forearm roll recovery ───────────────────────────────────

def signed_twist_angle(q: Rotation, axis: np.ndarray) -> float:
    """Extract the signed twist angle (radians) of rotation ``q`` about ``axis``.

    Uses the same swing-twist decomposition as ``clamp_twist``.
    """
    axis = _normalize(np.asarray(axis, dtype=np.float64))
    swing, twist = swing_twist_decompose(q, axis)
    # Twist is a pure rotation about axis; its rotvec is parallel to axis
    twist_rv = twist.as_rotvec()
    return float(np.dot(twist_rv, axis))




# ── Twist-constrained bones and their limits ────────────────────────────
# Joint name → max twist angle in degrees.
# Hinge joints (knees, ankles) have near-zero twist; wrists allow some.
# Shoulders and hips are ball joints — no twist constraint.
# IMPORTANT: elbow_l/r are NOT constrained here because in the proxy skeleton
# the elbow joint carries the forearm axial roll (there is no separate radius/
# ulna roll bone). Constraining elbow twist to 5 deg would zero out the valid
# forearm roll signal from the arm triangle.
# Wrist twist limits are MOVED to AXIAL_LIMITS below to be enforced via the
# forearm roll residual instead of a per-frame clamp, which allows temporal
# continuity (no per-frame 90° hard clamp).
TWIST_LIMITS: dict[str, float] = {
    "knee_l": 5.0, "knee_r": 5.0,
    "ankle_l": 10.0, "ankle_r": 10.0,
    "clavicle_l": 15.0, "clavicle_r": 15.0,
}


# ── Axial limits for bones with dedicated twist DOFs (forearm pronation) ──
# Each entry: bone_name -> (max_degrees, axis_source)
# axis_source can be:
#   "self"       = bone's own long axis (e.g., wrist)
#   "child:X"    = child bone X's long axis (e.g., elbow uses wrist axis)
# This allows the forearm axial roll to be clamped while keeping temporal
# continuity via the residual (not a hard per-frame clamp).
AXIAL_LIMITS: dict[str, tuple[float, str]] = {
    "elbow_l": (100.0, "child:wrist_l"),  # forearm axis
    "elbow_r": (100.0, "child:wrist_r"),
    "wrist_l": (25.0, "self"),
    "wrist_r": (25.0, "self"),
}

