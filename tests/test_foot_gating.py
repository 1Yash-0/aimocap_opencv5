"""CPU-only VQ-FIX6 target observability and transport-pin regression."""
import numpy as np
from scipy.spatial.transform import Rotation

from aimocap.data.wholebody_layout import COCO_ANCHORS
from aimocap.retarget import mocap_ik
from aimocap.retarget.fbx_rig import Skeleton
from aimocap.retarget.mocap_ik import MocapIKSolver
from aimocap.retarget.mocap_skeleton import MocapSkeleton, extract_mocap_points
from aimocap.retarget.temporal import stabilize_lower_body_animation


from tests._assets import requires_manny_fbx
def _make_scenario():
    frames = 12
    points = np.tile(np.array([0.0, 0.0, 80.0]), (frames, 133, 1)).astype(np.float64)
    anchors = {
        "pelvis_l": (-10.0, 0.0, 100.0), "pelvis_r": (10.0, 0.0, 100.0),
        "shoulder_l": (-20.0, 0.0, 140.0), "shoulder_r": (20.0, 0.0, 140.0),
        "elbow_l": (-42.0, 0.0, 105.0), "elbow_r": (42.0, 0.0, 105.0),
        "wrist_l": (-64.0, 0.0, 90.0), "wrist_r": (64.0, 0.0, 90.0),
        "knee_l": (-12.0, 0.0, 55.0), "knee_r": (12.0, 0.0, 55.0),
        "ankle_l": (-12.0, 0.0, 10.0), "ankle_r": (12.0, 0.0, 10.0),
        "big_toe_l": (-12.0, 15.0, 9.0), "big_toe_r": (12.0, 15.0, 9.0),
        "heel_l": (-12.0, -3.0, 9.0), "heel_r": (12.0, -3.0, 9.0),
        "left_ear": (9.0, 2.0, 145.0), "right_ear": (-9.0, 2.0, 145.0),
        "left_eye": (2.0, 8.0, 146.0), "right_eye": (-2.0, 8.0, 146.0),
        "nose": (0.0, 12.0, 144.0),
    }
    for name, xyz in anchors.items():
        points[:, COCO_ANCHORS[name], :] = xyz
    confidence = np.ones((frames, 133), dtype=np.float32)
    skel = MocapSkeleton(points, confidence, fbx_skel=Skeleton("Manny.FBX"))
    targets = extract_mocap_points(points)
    solver = MocapIKSolver(skel, target_sequence=targets)
    measured0 = {name: track[0].copy() for name, track in targets.items()}
    init0 = solver.analytic_init(measured0)
    source, axes, _ = solver._foot_gate_transport_state(None, init0, measured0, 0)
    assert source == "analytic_init"
    for side in ("l", "r"):
        ref = solver._foot_gate_reference_lengths[side]
        assert solver._foot_gate_reference_modes[side] == "sequence_median_plausible"
        ankle = measured0[f"ankle_{side}"]
        measured0[f"big_toe_{side}"] = ankle + ref * axes[side]
        targets[f"big_toe_{side}"] = np.tile(measured0[f"big_toe_{side}"], (frames, 1))
    # Rebuild against the exact synthetic target tracks whose reference was
    # estimated, then make a fixed previous-state sequence for the unit fixture.
    solver = MocapIKSolver(skel, target_sequence=targets)
    return solver, targets, solver.analytic_init(measured0)


def _frame(targets, frame, edits=None):
    measured = {name: np.asarray(track[frame], dtype=np.float64).copy()
                for name, track in targets.items()}
    if edits:
        for key, value in edits.items():
            measured[key] = np.asarray(value, dtype=np.float64).copy()
    return measured


def _gate_context(solver, targets, frame, prev_x, edits=None):
    measured = _frame(targets, frame, edits)
    init = solver.analytic_init(measured)
    ctx = solver._precompute_frame(measured, prev_x=prev_x, init_x=init,
                                   frame_idx=frame)
    return measured, ctx


@requires_manny_fbx
def test_foot_target_gate_and_transport_pin_are_symmetric_deterministic(monkeypatch):
    solver, targets, init0 = _make_scenario()
    axes = {}
    _, axes, _ = solver._foot_gate_transport_state(None, init0,
                                                   _frame(targets, 0), 0)
    base = _frame(targets, 0)
    ref_l = solver._foot_gate_reference_lengths["l"]
    ref_r = solver._foot_gate_reference_lengths["r"]

    rows = []
    # Frame 0 healthy (analytic-init axis); frame 1 left distance failure;
    # frame 2 left direction failure + right distance failure; frame 3 healthy.
    for f in range(4):
        edits = {}
        if f == 1:
            edits["big_toe_l"] = base["ankle_l"] + 3.0 * ref_l * axes["l"]
        elif f == 2:
            edits["big_toe_l"] = base["ankle_l"] - ref_l * axes["l"]
            edits["big_toe_r"] = base["ankle_r"] + 3.0 * ref_r * axes["r"]
        elif f == 3:
            edits["big_toe_l"] = base["ankle_l"] + ref_l * axes["l"]
            edits["big_toe_r"] = base["ankle_r"] + ref_r * axes["r"]
        measured, ctx = _gate_context(solver, targets, f,
                                      prev_x=None if f == 0 else init0,
                                      edits=edits)
        rows.append((measured, ctx))

    # Healthy contexts preserve the pre-gate positions/weights exactly and add no pin.
    for f in (0, 3):
        measured, ctx = rows[f]
        for side in ("l", "r"):
            toe_i = solver.skel.name_to_idx[f"toe_{side}"]
            assert ctx["foot_target_gates"][side]["healthy"] is True
            assert ctx["w"][toe_i] == 1.0
            assert np.array_equal(ctx["tgt_pos"][toe_i], measured[f"big_toe_{side}"])
        assert ctx["foot_transport_pins"] == []

    # Force the health predicate true to model the old/ungated path. Healthy
    # frames must leave the target positions and all weights bit-identical.
    healthy_result = lambda *args, **kwargs: {"healthy": True, "failed_conditions": []}
    monkeypatch.setattr(mocap_ik, "assess_foot_target", healthy_result)
    healthy_measured = _frame(targets, 3, {
        "big_toe_l": base["ankle_l"] + ref_l * axes["l"],
        "big_toe_r": base["ankle_r"] + ref_r * axes["r"],
    })
    healthy_init = solver.analytic_init(healthy_measured)
    ungated_ctx = solver._precompute_frame(healthy_measured, prev_x=init0,
                                           init_x=healthy_init, frame_idx=3)
    assert np.array_equal(rows[3][1]["tgt_pos"], ungated_ctx["tgt_pos"])
    assert np.array_equal(rows[3][1]["w"], ungated_ctx["w"])
    monkeypatch.undo()

    l_toe = solver.skel.name_to_idx["toe_l"]
    r_toe = solver.skel.name_to_idx["toe_r"]
    frame1 = rows[1][1]
    assert frame1["foot_target_gates"]["l"]["failed_conditions"] == [
        "toe_distance_above_reference_band"]
    assert frame1["w"][l_toe] == 0.0
    assert [p["side"] for p in frame1["foot_transport_pins"]] == ["l"]
    assert frame1["foot_target_gates"]["l"]["transport_target_source"] == "previous_solved_frame"

    frame2 = rows[2][1]
    assert frame2["foot_target_gates"]["l"]["failed_conditions"] == [
        "toe_direction_antiparallel"]
    assert frame2["foot_target_gates"]["r"]["failed_conditions"] == [
        "toe_distance_above_reference_band"]
    assert frame2["w"][l_toe] == frame2["w"][r_toe] == 0.0
    assert {p["side"] for p in frame2["foot_transport_pins"]} == {"l", "r"}

    # Rebuild once more: reference modes, gate conditions, and pin targets repeat exactly.
    again, targets2, init_again = _make_scenario()
    assert solver._foot_gate_reference_lengths == again._foot_gate_reference_lengths
    assert solver._foot_gate_reference_modes == again._foot_gate_reference_modes
    for f in range(4):
        edits = {}
        if f == 1:
            edits["big_toe_l"] = base["ankle_l"] + 3.0 * ref_l * axes["l"]
        elif f == 2:
            edits["big_toe_l"] = base["ankle_l"] - ref_l * axes["l"]
            edits["big_toe_r"] = base["ankle_r"] + 3.0 * ref_r * axes["r"]
        elif f == 3:
            edits["big_toe_l"] = base["ankle_l"] + ref_l * axes["l"]
            edits["big_toe_r"] = base["ankle_r"] + ref_r * axes["r"]
        _, ctx2 = _gate_context(again, targets2, f,
                                prev_x=None if f == 0 else init_again,
                                edits=edits)
        ctx1 = rows[f][1]
        assert np.array_equal(ctx1["w"], ctx2["w"])
        assert ctx1["foot_target_gates"] == ctx2["foot_target_gates"]
        assert len(ctx1["foot_transport_pins"]) == len(ctx2["foot_transport_pins"])
        for a, b in zip(ctx1["foot_transport_pins"], ctx2["foot_transport_pins"]):
            assert a["side"] == b["side"]
            assert np.array_equal(a["target_quaternion"], b["target_quaternion"])

    provenance = solver.foot_gate_diagnostics()
    assert provenance["gated_frame_count_by_side"] == {"l": 2, "r": 1}
    assert provenance["failed_condition_counts"] == {
        "toe_distance_above_reference_band": 2,
        "toe_direction_antiparallel": 1,
    }


@requires_manny_fbx
def test_scipy_and_torch_transport_residuals_match_on_cpu():
    import torch
    from aimocap.retarget.ik_torch import _residuals_torch

    solver, targets, prev = _make_scenario()
    measured = _frame(targets, 1, {
        "big_toe_l": _frame(targets, 1)["ankle_l"]
        + 3.0 * solver._foot_gate_reference_lengths["l"]
        * solver._foot_gate_transport_state(prev, solver.analytic_init(_frame(targets, 1)),
                                            _frame(targets, 1), 1)[1]["l"]
    })
    init = solver.analytic_init(measured)
    ctx = solver._precompute_frame(measured, prev_x=prev, init_x=init, frame_idx=1)
    assert len(ctx["foot_transport_pins"]) == 1

    # Compare only position + transport-pin terms to isolate the new residual
    # in the two production backends while keeping this test CPU-only.
    scipy_ctx = dict(ctx)
    scipy_ctx.update({"ori_weight": 0.0, "head_ori_active": False,
                      "prev_x": None, "prev_prev_x": None,
                      "temporal_weight": 0.0, "init_x": None})
    x = init.copy()
    ankle_i = solver.skel.name_to_idx["ankle_l"]
    x[3 + 3 * ankle_i:3 + 3 * ankle_i + 3] += np.array([0.17, -0.11, 0.08])
    numpy_res = solver._residuals_with_ctx(x, scipy_ctx)

    device, dtype = torch.device("cpu"), torch.float64
    to_t = lambda value: torch.as_tensor(np.asarray(value), dtype=dtype, device=device)
    levels_pre = []
    for level in solver.levels:
        children = torch.as_tensor(level, dtype=torch.long, device=device)
        parents = torch.as_tensor(solver.parents[level], dtype=torch.long, device=device)
        offsets = torch.as_tensor(solver.rest_offsets[level], dtype=dtype, device=device)
        levels_pre.append((children, parents, offsets))
    torch_ctx = {
        "num_joints": solver.num_joints,
        "rest_t0": to_t(solver.skel.rest_t[0]),
        "rest_offsets": to_t(solver.rest_offsets),
        "parents": torch.as_tensor(solver.parents, dtype=torch.long, device=device),
        "levels_flat": torch.empty(0, dtype=torch.long, device=device),
        "level_starts": torch.tensor([0], dtype=torch.long, device=device),
        "levels_pre": levels_pre,
        "tgt_pos": to_t(scipy_ctx["tgt_pos"]),
        "w": to_t(scipy_ctx["w"]),
        "ori_weight": 0.0,
        "prev_x": None,
        "prev_prev_x": None,
        "accel_weight": 0.0,
        "temporal_weight": 0.0,
        "init_x": None,
        "limb_pinned_info": [],
        "foot_transport_pin_scale": 30.0,
        "foot_transport_pins": [{
            "joint_idx": pin["joint_idx"],
            "target_quaternion": to_t(pin["target_quaternion"]),
            "weight": pin["weight"],
        } for pin in scipy_ctx["foot_transport_pins"]],
    }
    torch_res = _residuals_torch(to_t(x).unsqueeze(0), torch_ctx)[0].detach().numpy()
    assert numpy_res.shape == torch_res.shape
    assert np.allclose(numpy_res, torch_res, rtol=1e-7, atol=1e-7)


@requires_manny_fbx
def test_stage3_velocity_cap_alarm_counts_preclip_exceedances():
    skel = Skeleton("Manny.FBX")
    frames = 60
    rest_euler = Rotation.from_quat(np.asarray(skel.rest_rotations)).as_euler("xyz")
    eulers = np.tile(rest_euler, (frames, 1, 1))
    foot_i = skel.name_to_idx["foot_l"]
    eulers[30:, foot_i, 0] += np.deg2rad(120.0)
    animation = np.zeros((frames, 3 + 3 * skel.num_joints), dtype=np.float64)
    animation[:, 3:] = eulers.reshape(frames, -1)

    out, diagnostics = stabilize_lower_body_animation(
        skel, animation, fps=30.0, cutoff_hz=5.0, max_deg_per_frame=30.0)
    alarm = diagnostics["velocity_cap"]
    assert alarm["per_joint_exceedance_counts"]["foot_l"] == 3
    assert alarm["total_exceedances"] >= 3
    assert alarm["worst_input_delta_deg"] > 30.0
    assert all(set(item) == {"joint", "frame", "input_delta_deg"}
               for item in alarm["worst_offenders"])

    out_eulers = out[:, 3:].reshape(frames, skel.num_joints, 3)[:, foot_i]
    out_q = Rotation.from_euler("xyz", out_eulers)
    local_deltas = np.degrees((out_q[:-1].inv() * out_q[1:]).magnitude())
    assert float(local_deltas.max()) <= 30.0 + 1e-8


@requires_manny_fbx
def test_batched_torch_solver_reports_same_foot_gate_conditions_cpu():
    import torch
    from aimocap.retarget.torch_mocap_ik import TorchMocapIKSolver

    solver, targets, _ = _make_scenario()
    skel = solver.skel
    batch_solver = TorchMocapIKSolver(skel, device=torch.device("cpu"))
    T = len(targets["pelvis"])
    init_rotvecs = np.zeros((T, skel.num_joints, 3), dtype=np.float32)
    root_t, root_q = batch_solver.compute_analytic_root_sequence(targets)
    init_rotvecs[:, 0, :] = Rotation.from_quat(root_q).as_rotvec()
    refs = solver._foot_gate_reference_lengths
    ankle_i, toe_i = skel.name_to_idx["ankle_l"], skel.name_to_idx["toe_l"]
    axis_local = skel.rest_offsets[toe_i] / np.linalg.norm(skel.rest_offsets[toe_i])
    for frame in range(T):
        axis_world = Rotation.from_quat(root_q[frame]).apply(axis_local)
        ankle = targets["ankle_l"][frame]
        targets["big_toe_l"][frame] = ankle + refs["l"] * axis_world
    # Two deliberately unhealthy observations: one is 3x the reference length,
    # the other points opposite the transported long axis.
    axis_0 = Rotation.from_quat(root_q[0]).apply(axis_local)
    targets["big_toe_l"][1] = targets["ankle_l"][1] + 3.0 * refs["l"] * axis_0
    axis_1 = Rotation.from_quat(root_q[1]).apply(axis_local)
    targets["big_toe_l"][2] = targets["ankle_l"][2] - refs["l"] * axis_1

    tgt_pos = np.zeros((T, skel.num_joints, 3), dtype=np.float32)
    weights = np.ones((T, skel.num_joints), dtype=np.float32)
    for idx, name in skel.coco_anchor.items():
        if name in targets:
            tgt_pos[:, idx] = targets[name]
    contacts = {
        side: {"anchors": np.zeros((T, 3), dtype=np.float32),
               "c_effective": np.zeros(T, dtype=np.float32)}
        for side in ("left", "right")
    }
    _rt, _rot, _pos, info = batch_solver.solve_sequence_gpu(
        targets, tgt_pos, weights, contacts, init_rotvecs,
        w_contact_max=0.0, w_temporal=0.0, w_accel=0.0, max_iters=1,
    )
    events = [row for row in info["foot_target_gate"]["gated_frames"] if row["side"] == "l"]
    assert [row["frame"] for row in events] == [1, 2]
    assert events[0]["failed_conditions"] == ["toe_distance_above_reference_band"]
    assert events[1]["failed_conditions"] == ["toe_direction_antiparallel"]
