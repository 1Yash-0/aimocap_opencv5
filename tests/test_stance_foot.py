from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from aimocap.eval.gt_harness import animation_fk, keypoints_from_animation
from aimocap.retarget.fbx_rig import Skeleton
from aimocap.retarget.stance_foot import (
    _hysteresis,
    _transition_ramp,
    stabilize_stance_feet,
)


ROOT = Path(__file__).resolve().parents[1]


from tests._assets import requires_manny_fbx
def _rig():
    return Skeleton(str(ROOT / "Manny.FBX"))


def _base(rig, frames=60):
    rest = Rotation.from_quat(np.asarray(rig.rest_rotations)).as_euler("xyz")
    animation = np.zeros((frames, 3 + 3 * rig.num_joints), dtype=float)
    animation[:, 3:] = rest.reshape(-1)
    points = keypoints_from_animation(animation, rig)
    ids = {"l": 17, "r": 20}
    heel_ids = {"l": 19, "r": 22}
    targets = {}
    for side in ("l", "r"):
        targets[f"big_toe_{side}"] = points[:, ids[side]].copy()
        targets[f"heel_{side}"] = points[:, heel_ids[side]].copy()
        targets[f"small_toe_{side}"] = points[:, 18 if side == "l" else 21].copy()
    return animation, targets


def _tilted(animation, rig, angle_deg=20.0):
    out = animation.copy()
    idx = rig.name_to_idx["foot_l"]
    q = Rotation.from_euler("xyz", out[0, 3 + 3 * idx:3 + 3 * idx + 3])
    q = Rotation.from_euler("x", angle_deg, degrees=True) * q
    out[:, 3 + 3 * idx:3 + 3 * idx + 3] = q.as_euler("xyz")
    return out


@requires_manny_fbx
def test_tilted_planted_foot_flattened_without_ankle_motion():
    rig = _rig()
    base, targets = _base(rig)
    tilted = _tilted(base, rig)
    before = animation_fk(tilted, rig)
    out, diag = stabilize_stance_feet(rig, tilted, targets, 30.0)
    after = animation_fk(out, rig)
    foot = rig.name_to_idx["foot_l"]
    assert diag["enabled"]
    assert diag["sides"]["left"]["interior_frames"] == len(base)
    assert np.max(np.linalg.norm(after["positions"][:, foot] - before["positions"][:, foot], axis=1)) < 1e-9
    rest_q = Rotation.from_quat(rig.get_forward_kinematics()[1][foot])
    local_up = rest_q.inv().apply([0.0, 0.0, 1.0])
    normal = Rotation.from_quat(after["world_q"][:, foot]).apply(local_up)
    tilt = np.degrees(np.arccos(np.clip(normal[:, 2], -1.0, 1.0)))
    assert float(np.max(tilt)) < 1.0


@requires_manny_fbx
def test_per_landmark_references_are_independent():
    rig = _rig()
    animation, targets = _base(rig)
    targets["heel_l"][:, 2] = 0.0
    targets["big_toe_l"][:, 2] = 10.0
    targets["small_toe_l"][:, 2] = 20.0
    out, diag = stabilize_stance_feet(rig, animation, targets, 30.0)
    assert np.array_equal(out, animation)
    refs = diag["sides"]["left"]["reference_height_cm"]
    assert refs["heel"] == 0.0
    assert refs["big_toe"] == 10.0
    assert refs["small_toe"] == 20.0
    assert diag["sides"]["left"]["low_frame_counts"] == {
        "heel": 60, "big_toe": 60, "small_toe": 60,
    }


def test_hysteresis_and_zero_phase_transition_ramp():
    raw = np.array([0.0, 0.0, 0.7, 0.5, 0.4, 0.4, 0.5, 0.7, 0.2, 0.4, 0.2, 0.0])
    state = _hysteresis(raw)
    assert np.array_equal(state, [False, False, True, True, True, True, True, True, False, False, False, False])
    ramp = _transition_ramp(state, 6)
    assert 0.0 <= ramp[0] <= 1.0 and 0.0 <= ramp[-1] <= 1.0
    assert 0.0 < ramp[2] < 1.0
    assert 0.0 < ramp[7] < 1.0
    assert np.max(ramp) == 1.0


@requires_manny_fbx
def test_short_low_set_disables_side():
    rig = _rig()
    animation, targets = _base(rig)
    targets["heel_l"][20:, 2] += 20.0
    targets["big_toe_l"][20:, 2] += 20.0
    targets["small_toe_l"][20:, 2] += 20.0
    out, diag = stabilize_stance_feet(rig, animation, targets, 60.0)
    assert np.array_equal(out, animation)
    assert not diag["sides"]["left"]["enabled"]
    assert diag["sides"]["left"]["disable_reason"] == "too_few_low_frames"


@requires_manny_fbx
def test_heel_off_is_untouched_after_valid_stance():
    rig = _rig()
    animation, targets = _base(rig)
    targets["heel_l"][30:, 2] += 20.0
    out, diag = stabilize_stance_feet(rig, animation, targets, 30.0)
    assert np.array_equal(out, animation)
    assert diag["sides"]["left"]["interior_frames"] >= 29


@requires_manny_fbx
def test_fast_swing_is_untouched():
    rig = _rig()
    animation, targets = _base(rig)
    ramp = np.arange(len(animation), dtype=float) * 20.0
    for name in ("big_toe_l", "heel_l", "small_toe_l"):
        targets[name][:, 0] += ramp
    out, diag = stabilize_stance_feet(rig, animation, targets, 30.0)
    assert np.array_equal(out, animation)
    assert diag["sides"]["left"]["mean_weight"] == 0.0


@requires_manny_fbx
def test_landmark_mode_recovers_known_calibration_rotation():
    rig = _rig()
    animation, targets = _base(rig)
    frames = len(animation)
    flat = Rotation.from_euler("z", 25.0, degrees=True)
    known_c = Rotation.from_euler("xyz", [8.0, -12.0, 17.0], degrees=True)
    target_frame = flat * known_c.inv()
    origin = np.array([2.0, -3.0, 0.0])
    heel = np.broadcast_to(origin, (frames, 3)).copy()
    big = np.broadcast_to(origin + target_frame.apply([6.0, 0.0, 0.0]), (frames, 3)).copy()
    small = np.broadcast_to(origin + target_frame.apply([2.0, 3.0, 0.0]), (frames, 3)).copy()
    targets["heel_l"], targets["big_toe_l"], targets["small_toe_l"] = heel, big, small
    out, diag = stabilize_stance_feet(rig, animation, targets, 30.0, mode="landmark")
    assert diag["enabled"]
    assert diag["landmark"]["left"]["calibrated"]
    fk = animation_fk(out, rig)
    foot = rig.name_to_idx["foot_l"]
    actual = Rotation.from_quat(fk["world_q"][-1, foot])
    horizontal = big[0] - heel[0]
    horizontal[2] = 0.0
    horizontal /= np.linalg.norm(horizontal)
    expected_flat = Rotation.from_matrix(np.column_stack((horizontal, np.cross(np.array([0.0, 0.0, 1.0]), horizontal), np.array([0.0, 0.0, 1.0]))))
    assert np.degrees((actual.inv() * expected_flat).magnitude()) < 1.0
    assert diag["ball_local_relative_to_foot"]


@requires_manny_fbx
def test_deterministic():
    rig = _rig()
    animation, targets = _base(rig)
    tilted = _tilted(animation, rig)
    a, da = stabilize_stance_feet(rig, tilted, targets, 30.0)
    b, db = stabilize_stance_feet(rig, tilted, targets, 30.0)
    assert np.array_equal(a, b)
    assert da == db


@requires_manny_fbx
def test_non_grounded_rest_rig_fails_closed():
    rig = _rig()
    animation, targets = _base(rig)
    rig.rest_translations = [np.asarray(x, dtype=float).copy() for x in rig.rest_translations]
    rig.rest_translations[0][2] += 10.0
    out, diag = stabilize_stance_feet(rig, animation, targets, 30.0)
    assert np.array_equal(out, animation)
    assert not diag["enabled"]
    assert diag["reason"] == "rest_rig_not_grounded"


@requires_manny_fbx
def test_fk_positions_preserved_outside_foot_ball_descendants():
    rig = _rig()
    base, targets = _base(rig)
    inp = _tilted(base, rig, 20.0)
    out, diag = stabilize_stance_feet(rig, inp, targets, 30.0, mode="generic")
    before, after = animation_fk(inp, rig), animation_fk(out, rig)
    excluded = set()
    for name, idx in rig.name_to_idx.items():
        if name.startswith(("foot_", "ball_")):
            excluded.add(idx)
    changed = True
    while changed:
        changed = False
        for idx, parent in enumerate(rig.parents):
            if int(parent) in excluded and idx not in excluded:
                excluded.add(idx)
                changed = True
    for idx in range(rig.num_joints):
        if idx not in excluded:
            assert np.max(np.linalg.norm(after["positions"][:, idx] - before["positions"][:, idx], axis=1)) <= 1e-6


@requires_manny_fbx
def test_landmark_mode_is_not_generic_mode():
    rig = _rig()
    base, targets = _base(rig)
    inp = _tilted(base, rig, 18.0)
    generic, _ = stabilize_stance_feet(rig, inp, targets, 30.0, mode="generic")
    landmark, diag = stabilize_stance_feet(rig, inp, targets, 30.0, mode="landmark")
    assert diag["enabled"]
    assert not np.array_equal(generic, landmark)


@requires_manny_fbx
def test_forced_zero_weight_identity_is_byte_for_byte():
    rig = _rig()
    animation, targets = _base(rig)
    out, diag = stabilize_stance_feet(rig, animation, targets, 30.0, force_weights_zero=True)
    assert diag["force_weights_zero"]
    assert np.array_equal(out, animation)


@requires_manny_fbx
def test_correction_does_not_add_over_30_degree_frame_jumps():
    rig = _rig()
    base, targets = _base(rig)
    inp = _tilted(base, rig, 20.0)
    out, _ = stabilize_stance_feet(rig, inp, targets, 30.0, mode="generic")
    before, after = animation_fk(inp, rig), animation_fk(out, rig)
    for name in ("foot_l", "foot_r", "ball_l", "ball_r"):
        idx = rig.name_to_idx[name]
        q0 = Rotation.from_quat(before["world_q"][:, idx])
        q1 = Rotation.from_quat(after["world_q"][:, idx])
        in_jumps = np.degrees((q0[:-1].inv() * q0[1:]).magnitude())
        out_jumps = np.degrees((q1[:-1].inv() * q1[1:]).magnitude())
        assert np.count_nonzero(out_jumps > 30.0) <= np.count_nonzero(in_jumps > 30.0)

