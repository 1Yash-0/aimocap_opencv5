import numpy as np
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from aimocap.retarget.fbx_rig import Skeleton
from aimocap.retarget.mocap_skeleton import MocapSkeleton
from aimocap.retarget.mocap_ik import MocapIKSolver
from aimocap.retarget.swing_twist import signed_twist_angle


from tests._assets import requires_manny_fbx
def _solver(anchor="init", two_pass_limbs="all", smoothed_limbs="all"):
    fbx = Skeleton("Manny.FBX")
    rest, _ = fbx.get_forward_kinematics()
    ni = fbx.name_to_idx
    pts = np.zeros((2, 133, 3), dtype=float)
    mapping = {
        0: "head", 5: "upperarm_l", 6: "upperarm_r",
        7: "lowerarm_l", 8: "lowerarm_r", 9: "hand_l", 10: "hand_r",
        11: "thigh_l", 12: "thigh_r", 13: "calf_l", 14: "calf_r",
        15: "foot_l", 16: "foot_r", 17: "ball_l", 20: "ball_r",
    }
    for ci, name in mapping.items():
        pts[:, ci] = rest[ni[name]]
    pts[:, 11] = rest[ni["thigh_l"]]
    pts[:, 12] = rest[ni["thigh_r"]]
    pts[:, 5] = rest[ni["upperarm_l"]]
    pts[:, 6] = rest[ni["upperarm_r"]]
    skel = MocapSkeleton(pts, fbx_skel=fbx)
    return MocapIKSolver(skel, limb_twist_anchor=anchor, two_pass_limbs=two_pass_limbs,
                         smoothed_limbs=smoothed_limbs)


def _measured(solver):
    measured = {
        name: solver.skel.rest_t[i].copy()
        for i, name in solver.skel.coco_anchor.items()
    }
    ni = solver.skel.name_to_idx
    measured["pelvis"] = (solver.skel.rest_t[ni["hip_l"]] + solver.skel.rest_t[ni["hip_r"]]) / 2
    measured["neck"] = (solver.skel.rest_t[ni["shoulder_l"]] + solver.skel.rest_t[ni["shoulder_r"]]) / 2
    return measured


def _pin_context(solver, measured, prev_x=None):
    init = solver.analytic_init(measured)
    return solver._precompute_frame(measured, prev_x=prev_x, init_x=init)


@requires_manny_fbx
def test_default_init_matches_frozen_old_pin_logic():
    solver = _solver()
    rng = np.random.default_rng(123)
    for _ in range(5):
        measured = _measured(solver)
        for key in ("hip_l", "knee_l", "ankle_l", "shoulder_l", "elbow_l", "wrist_l"):
            measured[key] = measured[key] + rng.normal(0, 0.5, 3)
        ctx = _pin_context(solver, measured)
        for info in ctx["limb_pinned_info"]:
            p = info["p"]
            bone = info["bone"]
            init = ctx["init_x"][3:].reshape(-1, 3)[p]
            if np.linalg.norm(init) > 1e-12:
                expected = signed_twist_angle(Rotation.from_rotvec(init), bone)
            else:
                expected = 0.0
            assert info["twist_target"] == expected


@requires_manny_fbx
def test_continuous_anchor_is_smooth_through_guard():
    solver = _solver("continuous")
    base = _measured(solver)
    p = solver.skel.name_to_idx["hip_l"]
    c = solver.skel.name_to_idx["knee_l"]
    p0 = base["hip_l"].copy()
    axis = np.array([1.0, 0.0, 0.0])
    perp = np.array([0.0, 1.0, 0.0])
    prev = solver.analytic_init(base)
    values = []
    old_values = []
    for theta in np.arange(10.0, 60.01, 0.5):
        m = {k: np.asarray(v).copy() for k, v in base.items()}
        m["knee_l"] = p0 + 30.0 * axis
        direction = np.cos(np.deg2rad(theta)) * axis + np.sin(np.deg2rad(theta)) * perp
        m["ankle_l"] = m["knee_l"] + 25.0 * direction
        ctx = _pin_context(solver, m, prev_x=prev)
        info = next(x for x in ctx["limb_pinned_info"] if x["p"] == p and x["c"] == c)
        values.append(info["twist_target"])
        old_values.append(signed_twist_angle(
            Rotation.from_rotvec(ctx["init_x"][3:].reshape(-1, 3)[p]), info["bone"]
        ))
    assert np.max(np.abs(np.diff(values))) < np.deg2rad(3.0)
    assert np.max(np.abs(np.diff(old_values))) > np.deg2rad(3.0)


@requires_manny_fbx
def test_continuous_anchor_saturates_at_thresholds():
    solver = _solver("continuous")
    base = _measured(solver)
    p = solver.skel.name_to_idx["shoulder_l"]
    c = solver.skel.name_to_idx["elbow_l"]
    prev = solver.analytic_init(base)
    for theta in (10.0, 20.0, 50.0, 60.0):
        m = {k: np.asarray(v).copy() for k, v in base.items()}
        m["elbow_l"] = base["shoulder_l"] + np.array([30.0, 0.0, 0.0])
        d = np.array([np.cos(np.deg2rad(theta)), np.sin(np.deg2rad(theta)), 0.0])
        m["wrist_l"] = m["elbow_l"] + 25.0 * d
        ctx = _pin_context(solver, m, prev_x=prev)
        info = next(x for x in ctx["limb_pinned_info"] if x["p"] == p and x["c"] == c)
        expected_s = 0.0 if theta <= 20.0 else 1.0 if theta >= 50.0 else info["twist_observability"]
        assert info["twist_observability"] == pytest.approx(expected_s)
        if theta <= 20.0:
            assert info["twist_target"] == pytest.approx(info["twist_tau_ref"])
        if theta >= 50.0:
            wrapped = (info["twist_target"] - info["twist_tau_b"] + np.pi) % (2 * np.pi) - np.pi
            assert wrapped == pytest.approx(0.0, abs=1e-10)


def test_invalid_limb_twist_anchor_fails_closed():
    with pytest.raises(ValueError, match="limb_twist_anchor"):
        MocapIKSolver(None, limb_twist_anchor="bad")



def test_smoothed_anchor_removes_branch_switch_step():
    from aimocap.retarget.mocap_ik import _smooth_twist_sequence, _unwrap_observable

    # A 35-degree analytic-init branch switch is intentionally abrupt in the
    # source sequence; the offline target remains continuous.
    raw = np.deg2rad(np.r_[np.zeros(50), np.full(50, 35.0)])
    weights = np.ones(raw.size)
    target = _smooth_twist_sequence(_unwrap_observable(raw, weights), weights, 30.0)
    assert np.max(np.abs(np.diff(target))) < np.deg2rad(10.0)


@requires_manny_fbx
def test_two_pass_targets_are_zero_phase_lowpassed_pass1_term5_and_deterministic():
    solver = _solver("two_pass")
    base = solver.analytic_init(_measured(solver))
    side, p, c, _g, bone = solver._limb_pin_specs()[0]
    states = []
    for angle in np.deg2rad(np.sin(np.linspace(0.0, 16.0 * np.pi, 32)) * 20.0):
        state = base.copy()
        state[3 + p * 3:3 + p * 3 + 3] = angle * bone
        states.append(state)
    first = solver.prepare_two_pass_anchor(states, fps=30.0)
    expected_raw = np.asarray([
        solver._two_pass_term5_twist(state, solver.parents,
                                     solver.forward_kinematics, bone, p)
        for state in states
    ])
    expected = solver._zero_phase_lowpass(np.unwrap(expected_raw), 30.0, 3.0)
    actual = np.asarray([solver._two_pass_targets[(i, p, c)] for i in range(len(states))])
    assert np.allclose(actual, expected, rtol=0.0, atol=1e-12)
    second = solver.prepare_two_pass_anchor(states, fps=30.0)
    assert first["limbs"][f"{side}:{p}->{c}"]["target_deg"] == second["limbs"][f"{side}:{p}->{c}"]["target_deg"]
    solver.begin_two_pass(2)
    ctx = solver._precompute_frame(_measured(solver), frame_idx=0,
                                   init_x=base)
    info = next(item for item in ctx["limb_pinned_info"]
                if item["p"] == p and item["c"] == c)
    assert info["twist_target"] == pytest.approx(actual[0])


@requires_manny_fbx
def test_smoothed_arms_keeps_leg_pin_targets_at_init_targets():
    solver = _solver("smoothed", smoothed_limbs="arms")
    base = _measured(solver)
    target_sequence = {
        name: np.stack([np.asarray(value, dtype=float)] * 16)
        for name, value in base.items()
    }
    # Give the arm path a changing bend-plane observation so this exercises
    # the opt-in smoothed path rather than only its validation/defaults.
    for frame in range(16):
        target_sequence["wrist_l"][frame] += np.array(
            [0.0, 2.0 * np.sin(frame * 0.7), 0.0])
    solver.prepare_limb_twist_anchor(target_sequence, fps=30.0, frame_count=16)
    measured = {name: values[0].copy() for name, values in target_sequence.items()}
    init = solver.analytic_init(measured)
    ctx = solver._precompute_frame(measured, init_x=init, frame_idx=0)
    init_rotvecs = init[3:].reshape(-1, 3)
    leg_names = {"hip_l", "knee_l", "hip_r", "knee_r"}
    for info in ctx["limb_pinned_info"]:
        if solver.skel.joint_names[info["p"]] not in leg_names:
            continue
        rv = init_rotvecs[info["p"]]
        expected = (signed_twist_angle(Rotation.from_rotvec(rv), info["bone"])
                    if np.linalg.norm(rv) > 1e-12 else 0.0)
        assert info["twist_target"] == pytest.approx(expected, abs=0.0)


@requires_manny_fbx
def test_two_pass_arms_keeps_leg_pin_targets_at_init_targets():
    solver = _solver("two_pass", two_pass_limbs="arms")
    measured = _measured(solver)
    init = solver.analytic_init(measured)
    # Prepare a nontrivial pass-1 sequence so the arm-only policy has real
    # filtered targets, while leg targets must remain the exact init values.
    states = []
    _side, arm_p, _arm_c, _arm_g, arm_bone = solver._limb_pin_specs()[2]
    for angle in np.deg2rad(np.sin(np.linspace(0.0, 8.0 * np.pi, 16)) * 20.0):
        state = init.copy()
        state[3 + arm_p * 3:3 + arm_p * 3 + 3] = angle * arm_bone
        states.append(state)
    solver.prepare_two_pass_anchor(states, fps=30.0)
    solver.begin_two_pass(2)
    ctx = solver._precompute_frame(measured, frame_idx=0, init_x=init)
    init_rotvecs = init[3:].reshape(-1, 3)
    leg_names = {"hip_l", "knee_l", "hip_r", "knee_r"}
    for info in ctx["limb_pinned_info"]:
        p_name = solver.skel.joint_names[info["p"]]
        if p_name not in leg_names:
            continue
        expected = signed_twist_angle(
            Rotation.from_rotvec(init_rotvecs[info["p"]]), info["bone"]
        ) if np.linalg.norm(init_rotvecs[info["p"]]) > 1e-12 else 0.0
        assert info["twist_target"] == pytest.approx(expected, abs=0.0)
        assert (0, info["p"], info["c"]) not in solver._two_pass_targets


def test_smoothed_anchor_unwraps_plus_minus_pi():
    from aimocap.retarget.mocap_ik import _unwrap_observable

    raw = np.deg2rad([179.0, -179.0, 178.0, -178.0])
    unwrapped = _unwrap_observable(raw, np.ones(raw.size))
    assert np.allclose(np.degrees(unwrapped), [179.0, 181.0, 178.0, 182.0])


def test_smoothed_anchor_is_deterministic():
    from aimocap.retarget.mocap_ik import _smooth_twist_sequence

    values = np.linspace(-1.2, 2.4, 31)
    weights = np.linspace(0.0, 1.0, 31)
    a = _smooth_twist_sequence(values, weights, 29.97)
    b = _smooth_twist_sequence(values, weights, 29.97)
    assert np.array_equal(a, b)


@requires_manny_fbx
def test_two_pass_stride_one_is_bit_identical_to_historical_anchor():
    solver = _solver("two_pass", two_pass_limbs="arms")
    base = solver.analytic_init(_measured(solver))
    _side, p, c, _g, bone = solver._limb_pin_specs()[2]
    states = []
    for angle in np.deg2rad(np.sin(np.linspace(0.0, 8.0 * np.pi, 16)) * 20.0):
        state = base.copy()
        state[3 + p * 3:3 + p * 3 + 3] = angle * bone
        states.append(state)
    historical = solver.prepare_two_pass_anchor(states, fps=30.0, frame_count=16)
    explicit = solver.prepare_two_pass_anchor(
        states, fps=30.0, frame_count=16,
        pass1_frame_indices=np.arange(16, dtype=np.int64),
    )
    key = f"l:{p}->{c}"
    assert historical["limbs"][key]["target_deg"] == explicit["limbs"][key]["target_deg"]
    assert historical["limbs"][key]["filtered_true_twist_deg"] == explicit["limbs"][key]["filtered_true_twist_deg"]


@requires_manny_fbx
def test_two_pass_stride_interpolates_unwrapped_twist_before_filter():
    solver = _solver("two_pass", two_pass_limbs="arms")
    base = solver.analytic_init(_measured(solver))
    _side, p, c, _g, bone = solver._limb_pin_specs()[2]
    full_angles = np.deg2rad(np.array([170.0, 179.0, -179.0, -170.0]))
    indices = np.array([0, 3, 6, 9], dtype=np.int64)
    states = []
    for angle in full_angles:
        state = base.copy()
        state[3 + p * 3:3 + p * 3 + 3] = angle * bone
        states.append(state)
    diag = solver.prepare_two_pass_anchor(
        states, fps=30.0, frame_count=10, pass1_frame_indices=indices,
    )
    key = f"l:{p}->{c}"
    filtered = np.asarray(diag["limbs"][key]["filtered_true_twist_deg"])
    assert filtered.shape == (10,)
    assert np.isfinite(filtered).all()
    # The interpolated signal follows the short +/−pi crossing branch.
    raw = np.asarray(diag["limbs"][key]["raw_true_twist_deg"])
    assert np.max(np.abs(np.diff(raw))) < 15.0
