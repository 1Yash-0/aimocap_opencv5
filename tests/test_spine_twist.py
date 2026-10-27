"""Tests for the spine twist distribution fix.

The spine init must distribute the *torso twist* (R_upper . R_root^-1) evenly
across spine joints via SLERP, NOT the full root rotation R_root.  The old
code multiplied R_root^(1/n) on top of an already-R_root pelvis, over-rotating
the upper spine by up to R_root^(5/6) (~83 deg).  The fix interpolates from
R_root at the pelvis to R_upper at the neck, distributing only the difference.

These tests verify:
  - Spine globals follow SLERP(R_root, R_upper), not R_root^(1+k/n)
  - No over-rotation at the chest (spine_05)
  - neck_01 local rotation is a small fraction of R_diff, not the whole thing
  - The solver's orientation residual pins the twist (no drift)
"""
import numpy as np
from scipy.spatial.transform import Rotation

from aimocap.retarget.fbx_rig import Skeleton
from aimocap.retarget.mocap_skeleton import MocapSkeleton
from aimocap.retarget.mocap_ik import MocapIKSolver
from aimocap.retarget.root_frame import root_rotation


from tests._assets import requires_manny_fbx
def _build_twisted_measured(twist_deg: float, body_yaw_deg: float = 0.0) -> dict:
    """Build a single-frame measured dict from Manny rest, with the whole
    body rotated by ``body_yaw_deg`` around vertical (z) and the shoulders
    additionally twisted by ``twist_deg`` relative to the hips.

    ``body_yaw_deg`` makes R_root non-identity so the over-rotation bug is
    detectable.  ``twist_deg`` creates the torso twist (R_diff).  All joints
    (including limbs) are yawed consistently so the solver sees a physically
    realizable pose.
    """
    skel = Skeleton("Manny.FBX")
    rest, _ = skel.get_forward_kinematics()
    ni = skel.name_to_idx

    # Key rest positions
    hip_l = rest[ni["thigh_l"]]
    hip_r = rest[ni["thigh_r"]]
    sho_l = rest[ni["upperarm_l"]]
    sho_r = rest[ni["upperarm_r"]]
    pelvis_rest = (hip_l + hip_r) / 2.0
    neck_rest = (sho_l + sho_r) / 2.0

    Rz_body = Rotation.from_euler("z", body_yaw_deg, degrees=True)
    Rz_twist = Rotation.from_euler("z", twist_deg, degrees=True)

    def yaw_all(p):
        """Apply body yaw around pelvis center."""
        return pelvis_rest + Rz_body.apply(p - pelvis_rest)

    def twist_upper(p):
        """Apply shoulder twist around neck center (on top of body yaw)."""
        return neck_rest + Rz_twist.apply(p - neck_rest)

    # All joints get body yaw; upper body also gets the twist
    hip_l_w = yaw_all(hip_l)
    hip_r_w = yaw_all(hip_r)
    sho_l_w = twist_upper(yaw_all(sho_l))
    sho_r_w = twist_upper(yaw_all(sho_r))

    measured = {
        "pelvis": (hip_l_w + hip_r_w) / 2.0,
        "neck": (sho_l_w + sho_r_w) / 2.0,
        "hip_l": hip_l_w,
        "hip_r": hip_r_w,
        "shoulder_l": sho_l_w,
        "shoulder_r": sho_r_w,
    }

    # Remaining joints: yaw everything, twist upper body (above neck)
    upper_body = {"elbow_l", "elbow_r", "wrist_l", "wrist_r", "nose"}
    rig_map = {
        "knee_l": "calf_l", "knee_r": "calf_r",
        "ankle_l": "foot_l", "ankle_r": "foot_r",
        "elbow_l": "lowerarm_l", "elbow_r": "lowerarm_r",
        "wrist_l": "hand_l", "wrist_r": "hand_r",
        "nose": "head", "big_toe_l": "ball_l", "big_toe_r": "ball_r",
    }
    for name, rig_name in rig_map.items():
        if rig_name in ni:
            p = rest[ni[rig_name]]
            p = yaw_all(p)
            if name in upper_body:
                p = twist_upper(p)
            measured[name] = p

    return measured


def _spine_globals_from_state(solver: MocapIKSolver, x: np.ndarray):
    """Extract spine global rotations from an IK state via FK."""
    root_t, local_quats = solver._state_to_local_rotations(x)
    _, global_rot_quat = solver.forward_kinematics(root_t, local_quats)
    spine_names = solver.skel.topo.spine_chain("pelvis", "neck_01")
    spine_idx = [solver.skel.name_to_idx[nm] for nm in spine_names]
    return Rotation.from_quat(global_rot_quat[spine_idx]), spine_names


def _compute_root_frames(solver: MocapIKSolver, measured: dict):
    """Replicate the R_root / R_upper computation from analytic_init."""
    skel = solver.skel
    spine_dir = measured["neck"] - measured["pelvis"]
    hip_line = measured["hip_r"] - measured["hip_l"]
    shoulder_line = measured["shoulder_r"] - measured["shoulder_l"]

    idx_hl = skel.name_to_idx["hip_l"]
    idx_hr = skel.name_to_idx["hip_r"]
    idx_sl = skel.name_to_idx["shoulder_l"]
    idx_sr = skel.name_to_idx["shoulder_r"]
    rest_hip = skel.rest_t[idx_hr] - skel.rest_t[idx_hl]
    rest_shoulder = skel.rest_t[idx_sr] - skel.rest_t[idx_sl]
    rest_pelvis_mid = (skel.rest_t[idx_hl] + skel.rest_t[idx_hr]) / 2.0
    rest_neck_mid = (skel.rest_t[idx_sl] + skel.rest_t[idx_sr]) / 2.0
    rest_spine = rest_neck_mid - rest_pelvis_mid

    R_root = root_rotation(spine_dir, hip_line, rest_spine, rest_hip)
    R_upper = root_rotation(spine_dir, shoulder_line, rest_spine, rest_shoulder)
    return R_root, R_upper


def _compute_rdiff(R_root, R_upper):
    """Torso twist in the hip local frame (left-SLERP convention)."""
    return R_root.inv() * R_upper


def _make_solver() -> MocapIKSolver:
    fbx = Skeleton("Manny.FBX")
    rest, _ = fbx.get_forward_kinematics()
    ni = fbx.name_to_idx
    # Build a 133-joint COCO-ordered point cloud from Manny rest so the
    # proxy gets correct bone lengths and spine_scale ≈ 1.0.  COCO index
    # -> Manny joint mapping (see mocap_skeleton.COCO).
    coco_to_rig = {
        0: "head",                    # nose
        5: "upperarm_l", 6: "upperarm_r",
        7: "lowerarm_l", 8: "lowerarm_r",
        9: "hand_l",     10: "hand_r",
        11: "thigh_l",   12: "thigh_r",
        13: "calf_l",    14: "calf_r",
        15: "foot_l",    16: "foot_r",
        17: "ball_l",    20: "ball_r",
    }
    pts = np.zeros((10, 133, 3))
    for coco_i, rig_name in coco_to_rig.items():
        if rig_name in ni:
            pts[:, coco_i, :] = rest[ni[rig_name]]
    # Hips and shoulders as midpoints (matching extract_mocap_points)
    pts[:, 11, :] = rest[ni["thigh_l"]]   # left hip (pelvis_l)
    pts[:, 12, :] = rest[ni["thigh_r"]]   # right hip (pelvis_r)
    pts[:, 5, :] = rest[ni["upperarm_l"]]
    pts[:, 6, :] = rest[ni["upperarm_r"]]
    skel = MocapSkeleton(pts, fbx_skel=fbx)
    return MocapIKSolver(skel)


# ── Init tests ───────────────────────────────────────────────────────────────

@requires_manny_fbx
def test_spine_globals_follow_slerp_not_root_power():
    """Spine globals must be R_root * R_diff^(k/n), not R_root^(1+k/n).

    With a 60-deg body yaw and 30-deg shoulder twist, the old code would
    over-rotate spine_05 by ~25 deg.  The SLERP fix should give spine_05 =
    R_root * R_diff^(5/6) exactly.
    """
    solver = _make_solver()
    measured = _build_twisted_measured(twist_deg=30, body_yaw_deg=60)

    x = solver.analytic_init(measured)
    spine_globals, spine_names = _spine_globals_from_state(solver, x)

    R_root, R_upper = _compute_root_frames(solver, measured)
    R_diff = _compute_rdiff(R_root, R_upper)
    n = len(spine_names)  # 7

    for k in range(n):
        frac = k / (n - 1)
        R_expected = R_root * (R_diff ** frac)
        R_actual = spine_globals[k]
        angle = (R_expected.inv() * R_actual).magnitude()
        assert angle < np.deg2rad(0.5), (
            f"spine joint {k} ({spine_names[k]}): SLERP target mismatch, "
            f"angle={np.rad2deg(angle):.2f} deg"
        )


@requires_manny_fbx
def test_no_over_rotation_at_chest():
    """spine_05 global must NOT be R_root^(11/6).  The over-rotation (angle
    between the actual and the old buggy formula) must be large, confirming
    the fix changes the behavior.  And the actual must be close to the
    correct SLERP target."""
    solver = _make_solver()
    measured = _build_twisted_measured(twist_deg=30, body_yaw_deg=60)

    x = solver.analytic_init(measured)
    spine_globals, _ = _spine_globals_from_state(solver, x)

    R_root, R_upper = _compute_root_frames(solver, measured)
    R_diff = _compute_rdiff(R_root, R_upper)
    n = 7

    # Correct target
    R_correct = R_root * (R_diff ** (5.0 / 6.0))
    # Old buggy formula
    R_buggy = R_root ** (11.0 / 6.0)

    actual = spine_globals[5]  # spine_05

    assert (R_correct.inv() * actual).magnitude() < np.deg2rad(0.5), (
        "spine_05 should match SLERP target"
    )
    assert (R_buggy.inv() * actual).magnitude() > np.deg2rad(5.0), (
        f"spine_05 should NOT match old buggy R_root^(11/6); "
        f"deviation={np.rad2deg((R_buggy.inv() * actual).magnitude()):.1f} deg"
    )


@requires_manny_fbx
def test_per_segment_delta_is_constant():
    """Each spine segment's relative rotation should be R_diff^(1/(n-1)),
    i.e. the twist is distributed evenly.  All per-segment deltas must be
    equal (within tolerance)."""
    solver = _make_solver()
    measured = _build_twisted_measured(twist_deg=45, body_yaw_deg=30)

    x = solver.analytic_init(measured)
    spine_globals, spine_names = _spine_globals_from_state(solver, x)

    R_root, R_upper = _compute_root_frames(solver, measured)
    R_diff = _compute_rdiff(R_root, R_upper)
    n = len(spine_names)

    # With left-SLERP, all per-segment deltas are R_root * D^(1/(n-1)) *
    # R_root.inv() (the same conjugated fraction).  Check they're all equal.
    delta_0 = spine_globals[1] * spine_globals[0].inv()
    for k in range(1, n):
        delta = spine_globals[k] * spine_globals[k - 1].inv()
        angle = (delta_0.inv() * delta).magnitude()
        assert angle < np.deg2rad(0.5), (
            f"segment {k - 1}->{k} ({spine_names[k - 1]}->{spine_names[k]}): "
            f"delta not constant, angle={np.rad2deg(angle):.2f} deg"
        )
    # The total from pelvis to neck must equal R_upper * R_root.inv()
    total = spine_globals[-1] * spine_globals[0].inv()
    expected_total = R_upper * R_root.inv()
    assert (expected_total.inv() * total).magnitude() < np.deg2rad(0.5)


@requires_manny_fbx
def test_neck_local_rotation_is_small_fraction():
    """neck_01's local rotation should be R_diff^(1/(n-1)) (a small fraction
    of the twist), not the entire R_diff dumped at once."""
    solver = _make_solver()
    measured = _build_twisted_measured(twist_deg=45, body_yaw_deg=30)

    x = solver.analytic_init(measured)
    root_t, local_quats = solver._state_to_local_rotations(x)

    R_root, R_upper = _compute_root_frames(solver, measured)
    R_diff = _compute_rdiff(R_root, R_upper)
    n = 7

    neck_i = solver.skel.name_to_idx["neck_01"]
    R_neck_local = Rotation.from_quat(local_quats[neck_i])

    expected_local = R_diff ** (1.0 / (n - 1))
    angle_to_expected = (expected_local.inv() * R_neck_local).magnitude()
    angle_to_full = (R_diff.inv() * R_neck_local).magnitude()

    # Should match the small fraction, not the full R_diff
    assert angle_to_expected < np.deg2rad(0.5), (
        f"neck_01 local should be R_diff^(1/6)={np.rad2deg(expected_local.magnitude()):.1f} deg, "
        f"got {np.rad2deg(R_neck_local.magnitude()):.1f} deg"
    )
    assert angle_to_full > np.deg2rad(5.0), (
        "neck_01 local should NOT be the full R_diff"
    )


@requires_manny_fbx
def test_spine_init_with_zero_twist_gives_identity_spine():
    """When shoulders and hips are aligned (R_diff = identity), all spine
    joints should get R_root (no twist).  This is the rest-facing case."""
    solver = _make_solver()
    measured = _build_twisted_measured(twist_deg=0, body_yaw_deg=0)

    x = solver.analytic_init(measured)
    spine_globals, spine_names = _spine_globals_from_state(solver, x)

    R_root, R_upper = _compute_root_frames(solver, measured)
    R_diff = _compute_rdiff(R_root, R_upper)

    # R_diff should be near-identity
    assert R_diff.magnitude() < np.deg2rad(1.0), "R_diff should be ~identity"

    # All spine globals should equal R_root
    for k in range(len(spine_names)):
        angle = (R_root.inv() * spine_globals[k]).magnitude()
        assert angle < np.deg2rad(0.5), (
            f"spine joint {k}: should be R_root when no twist, "
            f"deviation={np.rad2deg(angle):.2f} deg"
        )


# ── Solver tests ─────────────────────────────────────────────────────────────

@requires_manny_fbx
def test_solver_preserves_spine_twist_orientation():
    """After solving a frame, the spine globals should stay close to the
    SLERP targets.  The orientation residual must prevent the solver from
    drifting the unobservable twist DOF to the old ~83 deg over-rotation.

    The solver inevitably trades some orientation for position fit (the proxy
    spine length doesn't exactly match the measured distance), so we allow
    moderate drift.  The key invariant: the drift must be MUCH less than the
    old over-rotation (~83 deg).  We check < 35 deg, which is well below the
    old bug but allows the position fit to proceed.
    """
    solver = _make_solver()
    measured = _build_twisted_measured(twist_deg=30, body_yaw_deg=60)

    x_solved = solver.solve_frame(measured, prev_x=None, temporal_weight=0.0)
    spine_globals, spine_names = _spine_globals_from_state(solver, x_solved)

    R_root, R_upper = _compute_root_frames(solver, measured)
    R_diff = _compute_rdiff(R_root, R_upper)
    n = len(spine_names)

    max_dev = 0.0
    for k in range(n):
        frac = k / (n - 1)
        R_expected = R_root * (R_diff ** frac)
        dev = (R_expected.inv() * spine_globals[k]).magnitude()
        max_dev = max(max_dev, dev)

    # The old buggy code over-rotated by ~83 deg.  The fix must keep drift
    # well below that.  35 deg allows for position-fitting trade-off while
    # proving the orientation residual is active.
    assert max_dev < np.deg2rad(35.0), (
        f"Spine twist drifted {np.rad2deg(max_dev):.1f} deg from SLERP target "
        f"(should be < 35 deg; old bug was ~83 deg)"
    )


@requires_manny_fbx
def test_solver_neck_position_matches_measurement():
    """The solver must still fit the measured neck position (the orientation
    residual must not dominate the position fit).  The proxy spine is longer
    than the measured distance (~57cm vs ~50cm), so some neck error is
    unavoidable.  We check the error is within the proxy/measurement mismatch.
    """
    solver = _make_solver()
    measured = _build_twisted_measured(twist_deg=30, body_yaw_deg=60)

    x_solved = solver.solve_frame(measured, prev_x=None, temporal_weight=0.0)
    root_t, local_quats = solver._state_to_local_rotations(x_solved)
    global_pos, _ = solver.forward_kinematics(root_t, local_quats)

    neck_i = solver.skel.name_to_idx["neck_01"]
    pelvis_i = 0
    tgt_neck = measured["neck"]
    actual_neck = global_pos[neck_i]
    # The proxy spine is ~57cm but measured distance is ~50cm, so the neck
    # can't reach the target exactly.  Allow up to 20cm (the solver trades
    # position for orientation with the new residual, but should still get
    # reasonably close).
    assert np.linalg.norm(actual_neck - tgt_neck) < 20.0, (
        f"neck_01 position {actual_neck} too far from target {tgt_neck}, "
        f"error={np.linalg.norm(actual_neck - tgt_neck):.2f} cm"
    )
    assert np.linalg.norm(global_pos[pelvis_i] - measured["pelvis"]) < 20.0, (
        f"pelvis position {global_pos[pelvis_i]} too far from target "
        f"{measured['pelvis']}"
    )


@requires_manny_fbx
def test_solver_does_not_overrotate_like_old_bug():
    """The solved spine must NOT match the old buggy formula R_root^(1+k/n).
    This is the regression guard: even with position fitting, the spine
    must stay far from the old over-rotation pattern."""
    solver = _make_solver()
    measured = _build_twisted_measured(twist_deg=30, body_yaw_deg=60)

    x_solved = solver.solve_frame(measured, prev_x=None, temporal_weight=0.0)
    spine_globals, _ = _spine_globals_from_state(solver, x_solved)

    R_root, R_upper = _compute_root_frames(solver, measured)
    n = 7

    # Check spine_05 specifically — the old bug over-rotated it to R_root^(11/6)
    R_buggy = R_root ** (11.0 / 6.0)
    actual = spine_globals[5]
    dev_from_buggy = (R_buggy.inv() * actual).magnitude()

    # Must be far from the buggy formula (the old code matched it exactly)
    assert dev_from_buggy > np.deg2rad(20.0), (
        f"spine_05 matches old buggy R_root^(11/6) too closely: "
        f"deviation={np.rad2deg(dev_from_buggy):.1f} deg (should be > 20 deg)"
    )
