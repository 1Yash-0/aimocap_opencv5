import numpy as np
from scipy.spatial.transform import Rotation

from aimocap.retarget.bvh import read_bvh, write_bvh
from aimocap.retarget.fbx_rig import Skeleton
from aimocap.retarget.foot_lock import (
    _solve_knee_position,
    apply_foot_lock,
    detect_foot_contacts,
    detect_image_foot_contacts,
    detect_target_foot_contacts,
)
from aimocap.retarget.mocap_skeleton import MocapSkeleton, extract_mocap_points, COCO
from aimocap.retarget.temporal import (
    stabilize_lower_body_animation,
    stabilize_mocap_fit_targets,
)


from tests._assets import requires_manny_fbx
@requires_manny_fbx
def test_manny_hierarchy_is_single_acyclic_tree():
    skel = Skeleton("Manny.FBX")

    roots = [i for i, p in enumerate(skel.parents) if p == -1]
    assert roots == [skel.name_to_idx["root"]]

    for i, p in enumerate(skel.parents):
        assert p < i

    for i in range(skel.num_joints):
        seen = set()
        cur = i
        while cur != -1:
            assert cur not in seen
            seen.add(cur)
            cur = skel.parents[cur]


@requires_manny_fbx
def test_proxy_rest_directions_match_manny_fk_directions():
    fbx_skel = Skeleton("Manny.FBX")
    fbx_global_pos, _ = fbx_skel.get_forward_kinematics()

    # Use random points to ensure all estimated bone lengths are non-zero, 
    # avoiding NaN directions.
    rng = np.random.default_rng(42)
    sequence = rng.normal(0.0, 10.0, (1, 133, 3))
    
    weights = np.ones((1, 133), dtype=float)
    mocap_skel = MocapSkeleton(sequence, weights, fbx_skel=fbx_skel)

    for i in range(1, mocap_skel.num_joints):
        p = mocap_skel.parents[i]
        fbx_i = fbx_skel.name_to_idx[mocap_skel.fbx_mapping[i]]
        fbx_p = fbx_skel.name_to_idx[mocap_skel.fbx_mapping[p]]

        expected = fbx_global_pos[fbx_i] - fbx_global_pos[fbx_p]
        expected /= np.linalg.norm(expected)
        actual = mocap_skel.rest_offsets[i] / np.linalg.norm(mocap_skel.rest_offsets[i])

        assert np.dot(actual, expected) > 0.999999


def _fk_animation_positions(skel: Skeleton, animation: np.ndarray) -> np.ndarray:
    positions = np.zeros((len(animation), skel.num_joints, 3))
    for frame in range(len(animation)):
        eulers = animation[frame, 3:].reshape(skel.num_joints, 3)
        local_quats = Rotation.from_euler("xyz", eulers, degrees=False).as_quat()
        positions[frame], _ = skel.get_forward_kinematics(
            local_quats,
            root_translation=animation[frame, :3],
        )
    return positions


def _fk_bvh_positions(bvh_path):
    names, parents, offsets, channel_map, frames, _ = read_bvh(str(bvh_path))
    positions = np.zeros((len(frames), len(names), 3), dtype=float)
    for frame_idx, row in enumerate(frames):
        cursor = 0
        local_rots = [Rotation.identity() for _ in names]
        root_translation = np.zeros(3)
        for joint_idx, channels in channel_map:
            values = row[cursor:cursor + len(channels)]
            cursor += len(channels)
            rot_axes = []
            rot_values = []
            translation = np.zeros(3)
            for channel, value in zip(channels, values):
                if channel.endswith("position"):
                    translation["XYZ".index(channel[0])] = value
                elif channel.endswith("rotation"):
                    rot_axes.append(channel[0])
                    rot_values.append(value)
            if rot_axes:
                local_rots[joint_idx] = Rotation.from_euler(
                    "".join(rot_axes).upper(),
                    rot_values,
                    degrees=True,
                )
            if parents[joint_idx] == -1:
                root_translation = translation

        global_rots = [Rotation.identity() for _ in names]
        for joint_idx in range(len(names)):
            parent = parents[joint_idx]
            if parent == -1:
                positions[frame_idx, joint_idx] = root_translation + offsets[joint_idx]
                global_rots[joint_idx] = local_rots[joint_idx]
            else:
                positions[frame_idx, joint_idx] = (
                    positions[frame_idx, parent]
                    + global_rots[parent].apply(offsets[joint_idx])
                )
                global_rots[joint_idx] = global_rots[parent] * local_rots[joint_idx]

    return names, positions


def test_two_bone_knee_solver_preserves_lengths():
    hip = np.array([0.0, 0.0, 0.0])
    knee = np.array([0.0, 3.0, 4.0])
    ankle_target = np.array([0.0, 0.0, 8.0])

    knee_eff, ankle_eff = _solve_knee_position(
        hip,
        knee,
        ankle_target,
        upper_len=5.0,
        lower_len=5.0,
        pole_hint=np.array([0.0, 1.0, 0.0]),
    )

    assert np.allclose(ankle_eff, ankle_target)
    assert np.isclose(np.linalg.norm(knee_eff - hip), 5.0)
    assert np.isclose(np.linalg.norm(ankle_eff - knee_eff), 5.0)


@requires_manny_fbx
def test_bvh_export_roundtrips_to_internal_fk(tmp_path):
    skel = Skeleton("Manny.FBX")
    frames = 6
    rest_eulers = Rotation.from_quat(np.array(skel.rest_rotations)).as_euler(
        "xyz",
        degrees=False,
    )
    animation = np.zeros((frames, 3 + skel.num_joints * 3), dtype=float)
    animation[:, 3:] = rest_eulers.flatten()
    animation[:, 0] = np.linspace(0.0, 5.0, frames)

    eulers = animation[:, 3:].reshape(frames, skel.num_joints, 3)
    eulers[:, skel.name_to_idx["spine_03"], 2] += np.deg2rad(np.linspace(0.0, 20.0, frames))
    eulers[:, skel.name_to_idx["upperarm_l"], 1] += np.deg2rad(np.linspace(0.0, -35.0, frames))
    eulers[:, skel.name_to_idx["thigh_r"], 0] += np.deg2rad(np.linspace(0.0, 25.0, frames))
    animation[:, 3:] = eulers.reshape(frames, -1)

    out = tmp_path / "roundtrip.bvh"
    write_bvh(str(out), skel, animation, fps=30.0)

    # BVH is now Y-up and omits the 'root' bone.
    R_conv = Rotation.from_euler("x", -90, degrees=True)
    names, bvh_pos = _fk_bvh_positions(out)
    internal_pos = _fk_animation_positions(skel, animation)

    # Compare only joints that appear in BVH, converting internal FK to Y-up.
    for i, name in enumerate(names):
        skel_idx = skel.name_to_idx[name]
        ref_yup = np.array([R_conv.apply(internal_pos[f, skel_idx]) for f in range(frames)])
        max_err = np.max(np.linalg.norm(bvh_pos[:, i] - ref_yup, axis=1))
        assert max_err < 2.0, f"Joint '{name}' max error {max_err:.3f} cm"


def test_contact_detection_finds_static_low_toes():
    source = np.zeros((60, 133, 3), dtype=float)
    source[:, [17, 18, 19], :] = np.array([0.0, 0.0, 1.0])
    source[:, [20, 21, 22], :] = np.array([20.0, 0.0, 1.0])
    source[30:, [17, 18, 19], 0] += np.linspace(0.0, 60.0, 30)[:, None]
    source[30:, [17, 18, 19], 2] += 30.0

    contacts = detect_foot_contacts(source, fps=30.0)
    left_mask = contacts["left"]["mask"]

    assert np.all(left_mask[:20])
    assert not np.any(left_mask[40:])


@requires_manny_fbx
def test_target_contact_detection_recovers_near_ground_stance():
    skel = Skeleton("Manny.FBX")
    frames = 50
    positions = np.zeros((frames, skel.num_joints, 3), dtype=float)
    foot = skel.name_to_idx["foot_l"]
    ball = skel.name_to_idx["ball_l"]
    positions[:, foot, 2] = 2.0
    positions[:, ball, 2] = 1.0
    positions[:, foot, 0] = np.linspace(0.0, 20.0, frames)
    positions[:, ball, 0] = np.linspace(0.0, 20.0, frames)
    positions[25:, foot, 2] = 25.0
    positions[25:, ball, 2] = 25.0

    contacts = detect_target_foot_contacts(skel, positions, fps=30.0)

    assert np.any(contacts["left"]["mask"][:20])
    assert not np.any(contacts["left"]["mask"][35:])


def test_image_contact_detection_uses_stationary_2d_feet(tmp_path):
    frames = 45
    cameras = 2
    keypoints = np.zeros((frames, cameras, 133, 2), dtype=np.float32)
    scores = np.ones((frames, cameras, 133), dtype=np.float32) * 0.9

    for cam in range(cameras):
        keypoints[:, cam, 5] = np.array([500.0, 350.0])
        keypoints[:, cam, 6] = np.array([620.0, 350.0])
        keypoints[:, cam, 11] = np.array([510.0, 520.0])
        keypoints[:, cam, 12] = np.array([610.0, 520.0])
        keypoints[:, cam, [15, 17, 18, 19]] = np.array([560.0, 850.0])
        keypoints[:, cam, [16, 20, 21, 22]] = np.array([660.0, 850.0])

    motion = np.linspace(0.0, 220.0, frames - 25)
    keypoints[25:, :, [15, 17, 18, 19], 0] += motion[:, None, None]

    npz_path = tmp_path / "pose2d.npz"
    np.savez(
        npz_path,
        keypoints=keypoints,
        scores=scores,
        image_sizes=np.array([[1920, 1080], [1920, 1080]], dtype=np.int32),
    )

    contacts = detect_image_foot_contacts(npz_path, num_frames=frames, fps=30.0)

    assert np.all(contacts["left"]["mask"][:15])
    assert not np.any(contacts["left"]["mask"][34:])


@requires_manny_fbx
def test_target_space_foot_lock_reduces_planted_foot_drift():
    skel = Skeleton("Manny.FBX")
    frames = 60
    rest_eulers = Rotation.from_quat(np.array(skel.rest_rotations)).as_euler(
        "xyz",
        degrees=False,
    )
    animation = np.zeros((frames, 3 + skel.num_joints * 3), dtype=float)
    animation[:, 3:] = rest_eulers.flatten()
    animation[:, 0] = np.linspace(0.0, 6.0, frames)

    source = np.zeros((frames, 133, 3), dtype=float)
    source[:, [17, 18, 19], :] = np.array([0.0, 0.0, 0.0])
    source[:, [20, 21, 22], :] = np.array([10.0, 0.0, 0.0])

    locked, diagnostics = apply_foot_lock(skel, animation, source, fps=30.0)

    assert diagnostics["enabled"] is True
    before = _fk_animation_positions(skel, animation)
    after = _fk_animation_positions(skel, locked)
    foot_idx = skel.name_to_idx["foot_l"]
    mid = slice(10, 50)
    before_drift = np.ptp(before[mid, foot_idx, 0])
    after_drift = np.ptp(after[mid, foot_idx, 0])

    assert after_drift < before_drift * 0.25
    assert np.ptp(locked[mid, 0]) < np.ptp(animation[mid, 0])
    assert np.max(np.abs(np.diff(locked[:, 0]))) < 1.0


@requires_manny_fbx
def test_lower_body_stabilization_reduces_rotation_spike():
    skel = Skeleton("Manny.FBX")
    frames = 45
    rest_eulers = Rotation.from_quat(np.array(skel.rest_rotations)).as_euler(
        "xyz",
        degrees=False,
    )
    animation = np.zeros((frames, 3 + skel.num_joints * 3), dtype=float)
    animation[:, 3:] = rest_eulers.flatten()

    calf = skel.name_to_idx["calf_l"]
    eulers = animation[:, 3:].reshape(frames, skel.num_joints, 3)
    eulers[22, calf, 0] += np.deg2rad(80.0)
    animation[:, 3:] = eulers.reshape(frames, -1)

    stabilized, diagnostics = stabilize_lower_body_animation(
        skel,
        animation,
        fps=30.0,
        cutoff_hz=4.0,
    )

    assert diagnostics["enabled"] is True
    assert (
        diagnostics["after"]["calf_l"]["max_deg_per_frame"]
        < diagnostics["before"]["calf_l"]["max_deg_per_frame"] * 0.5
    )
    assert stabilized.shape == animation.shape


def test_mocap_fit_target_stabilization_filters_arm_endpoints_and_preserves_fast_motion():
    # V6 evidence shows that target-side arm filtering removes held-arm input
    # noise without worsening fast-frame position fidelity: see
    # outputs/vq_investigation/arm_stab/audit_v6_battery.json and
    # audit_v6_fastcheck.json, with the frozen-arm floor control in
    # outputs/vq_investigation/arm_frozen/.
    frames = 120
    fps = 29.97002997002997
    targets = {
        name: np.zeros((frames, 3), dtype=float)
        for name in ("pelvis", "hip_l", "knee_l", "ankle_l", "hip_r", "knee_r", "ankle_r")
    }
    targets["ankle_l"][22, 0] = 40.0

    arm_names = ("shoulder_l", "shoulder_r", "elbow_l", "elbow_r", "wrist_l", "wrist_r")
    rng = np.random.default_rng(147)
    start, frequency = 40, 0.5
    amplitude = 150.0 / (2.0 * np.pi * frequency)
    t = (np.arange(frames) - start) / fps
    for name in arm_names:
        track = np.zeros((frames, 3), dtype=float)
        track[start:, 0] = amplitude * np.sin(2.0 * np.pi * frequency * t[start:])
        track[10:40, 1] = 0.6 * rng.choice(np.array([-1.0, 1.0]), size=30)
        targets[name] = track

    stabilized, diagnostics = stabilize_mocap_fit_targets(
        targets,
        fps=fps,
        cutoff_hz=1.5,
    )

    assert diagnostics["enabled"] is True
    assert np.ptp(np.diff(stabilized["ankle_l"][:, 0])) < np.ptp(np.diff(targets["ankle_l"][:, 0]))
    held = slice(12, 38)
    for name in arm_names:
        before = targets[name]
        after = stabilized[name]
        before_pad = np.pad(before, ((2, 2), (0, 0)), mode="edge")
        after_pad = np.pad(after, ((2, 2), (0, 0)), mode="edge")
        before_med = np.median(np.lib.stride_tricks.sliding_window_view(before_pad, 5, axis=0), axis=-1)
        after_med = np.median(np.lib.stride_tricks.sliding_window_view(after_pad, 5, axis=0), axis=-1)
        before_hf = np.linalg.norm(before - before_med, axis=1)
        after_hf = np.linalg.norm(after - after_med, axis=1)
        assert np.percentile(after_hf[held], 95) < 0.5 * np.percentile(before_hf[held], 95), name

    fast_pairs = slice(45, 114)
    before_speed = np.linalg.norm(np.diff(targets["wrist_l"], axis=0), axis=1) * fps
    after_speed = np.linalg.norm(np.diff(stabilized["wrist_l"], axis=0), axis=1) * fps
    before_p95 = np.percentile(before_speed[fast_pairs], 95)
    after_p95 = np.percentile(after_speed[fast_pairs], 95)
    assert abs(after_p95 - before_p95) / before_p95 < 0.15

    # The lower-body adaptive cutoff and outputs remain unchanged when arm
    # targets are present: leg-only input reproduces the original behavior.
    lower_only, _ = stabilize_mocap_fit_targets(
        {name: targets[name].copy() for name in targets if name not in arm_names},
        fps=fps,
        cutoff_hz=1.5,
    )
    for name in lower_only:
        assert np.array_equal(stabilized[name], lower_only[name]), name


@requires_manny_fbx
def test_bvh_round_trip_channel_order(tmp_path):
    """
    Export a short animation to BVH, parse it back, reconstruct FK, and verify
    that the recovered joint positions match the original FK within 1 cm.

    The BVH exporter:
    - Skips the 'root' helper bone (pelvis is BVH ROOT)
    - Converts from Z-up to Y-up

    So BVH FK positions should equal R_conv.apply(ref_positions) for all
    joints except 'root' (which is not in the BVH).
    """
    from aimocap.retarget.bvh import write_bvh, read_bvh

    skel = Skeleton("Manny.FBX")
    fps = 30.0
    frames = 5
    rng = np.random.default_rng(42)

    # Build a small random animation (small angles so FK stays near rest).
    animation = np.zeros((frames, 3 + skel.num_joints * 3), dtype=float)
    animation[:, :3] = rng.uniform(-5.0, 5.0, (frames, 3))   # root translation
    animation[:, 3:] = rng.uniform(-0.15, 0.15, (frames, skel.num_joints * 3))

    # Reference FK positions from the NPY-level data (Z-up).
    ref_positions = _fk_animation_positions(skel, animation)

    # Z-up → Y-up conversion (same as the exporter uses).
    R_conv = Rotation.from_euler("x", -90, degrees=True)

    # Export to BVH.
    bvh_path = str(tmp_path / "roundtrip.bvh")
    write_bvh(bvh_path, skel, animation, fps=fps)

    # Parse BVH back.
    joint_names, parents, offsets, channel_map, motion_rows, parsed_fps = read_bvh(bvh_path)

    # BVH has one fewer joint (root is omitted).
    assert len(joint_names) == skel.num_joints - 1, (
        f"BVH has {len(joint_names)} joints, expected {skel.num_joints - 1}"
    )
    assert abs(parsed_fps - fps) < 0.1, f"FPS mismatch: {parsed_fps} vs {fps}"
    assert len(motion_rows) == frames, f"Frame count mismatch: {len(motion_rows)} vs {frames}"

    # Reconstruct joint-index → name map from parsed BVH.
    bvh_name_to_idx = {name: i for i, name in enumerate(joint_names)}

    # Build a children list for BVH FK.
    bvh_children: list[list[int]] = [[] for _ in range(len(joint_names))]
    for i, p in enumerate(parents):
        if p != -1:
            bvh_children[p].append(i)

    def bvh_fk(frame_row: np.ndarray) -> np.ndarray:
        """Forward kinematics from a single parsed BVH motion row."""
        # Parse channels in declaration order.
        col = 0
        root_t = np.zeros(3)
        local_rots: dict[int, Rotation] = {}

        for j_idx, ch_names in channel_map:
            n = len(ch_names)
            vals = frame_row[col: col + n]
            col += n

            rot_x = rot_y = rot_z = 0.0
            tx = ty = tz = 0.0
            for ch, v in zip(ch_names, vals):
                if ch == "Xposition":
                    tx = v
                elif ch == "Yposition":
                    ty = v
                elif ch == "Zposition":
                    tz = v
                elif ch == "Xrotation":
                    rot_x = v
                elif ch == "Yrotation":
                    rot_y = v
                elif ch == "Zrotation":
                    rot_z = v

            if j_idx == 0:
                root_t = np.array([tx, ty, tz])
            local_rots[j_idx] = Rotation.from_euler(
                "XYZ",
                [rot_x, rot_y, rot_z],
                degrees=True,
            )

        # Propagate FK.
        J = len(joint_names)
        global_pos = np.zeros((J, 3))
        global_rot: list[Rotation | None] = [None] * J

        def _fk_node(idx: int, parent_pos: np.ndarray, parent_rot: Rotation) -> None:
            local_r = local_rots.get(idx, Rotation.identity())
            g_rot = parent_rot * local_r
            local_offset = offsets[idx]
            g_pos = parent_pos + parent_rot.apply(local_offset)
            global_pos[idx] = g_pos
            global_rot[idx] = g_rot
            for ch in bvh_children[idx]:
                _fk_node(ch, g_pos, g_rot)

        # Root: position comes from motion row (already absolute), rotation from local_rots.
        root_rot = local_rots.get(0, Rotation.identity())
        global_pos[0] = root_t
        global_rot[0] = root_rot
        for ch in bvh_children[0]:
            _fk_node(ch, root_t, root_rot)

        return global_pos

    # Compare per-frame FK positions.
    # BVH is in Y-up; reference FK is in Z-up. Convert reference to Y-up.
    for f in range(frames):
        bvh_pos = bvh_fk(motion_rows[f])

        # Map BVH joint indices → skeleton indices using names.
        # 'root' is not in BVH, so it will be skipped automatically.
        for skel_i, skel_name in enumerate(skel.node_names):
            if skel_name not in bvh_name_to_idx:
                continue
            bvh_i = bvh_name_to_idx[skel_name]
            ref_yup = R_conv.apply(ref_positions[f, skel_i])
            err = np.linalg.norm(bvh_pos[bvh_i] - ref_yup)
            assert err < 2.0, (
                f"Frame {f} joint '{skel_name}': BVH FK pos {bvh_pos[bvh_i]} "
                f"vs ref(Y-up) {ref_yup}, error={err:.3f} cm"
            )


# ── Foot keypoint tests ─────────────────────────────────────────────────────

def test_extract_mocap_points_includes_big_toe():
    """extract_mocap_points must return big_toe_l/big_toe_r from a 133-joint input."""
    pts = np.zeros((2, 133, 3))
    pts[:, 17, :] = [1.0, 2.0, 3.0]   # left big toe
    pts[:, 20, :] = [4.0, 5.0, 6.0]   # right big toe
    d = extract_mocap_points(pts)
    assert "big_toe_l" in d and "big_toe_r" in d
    assert np.allclose(d["big_toe_l"], [[1, 2, 3], [1, 2, 3]])
    assert np.allclose(d["big_toe_r"], [[4, 5, 6], [4, 5, 6]])


def test_extract_mocap_points_omits_big_toe_on_17_joint_input():
    """When input has only 17 joints, big_toe keys must be absent (no crash)."""
    pts = np.zeros((2, 17, 3))
    d = extract_mocap_points(pts)
    assert "big_toe_l" not in d and "big_toe_r" not in d
    assert "ankle_l" in d  # body joints still present


@requires_manny_fbx
def test_toe_joints_are_coco_anchored():
    """toe_l/toe_r must be in coco_anchor so they get weight 1.0 (measured)."""
    fbx = Skeleton("Manny.FBX")
    pts = np.zeros((10, 133, 3))
    # Put valid ankle+hip positions so MocapSkeleton can build bone lengths.
    pts[:, 11, :] = [10, 0, 0]   # left hip
    pts[:, 12, :] = [-10, 0, 0]  # right hip
    pts[:, 15, :] = [10, -40, 0]  # left ankle
    pts[:, 16, :] = [-10, -40, 0] # right ankle
    pts[:, 17, :] = [10, -50, 0]  # left big toe
    pts[:, 20, :] = [-10, -50, 0] # right big toe
    skel = MocapSkeleton(pts, fbx_skel=fbx)
    toe_l_i = skel.name_to_idx["toe_l"]
    toe_r_i = skel.name_to_idx["toe_r"]
    assert toe_l_i in skel.coco_anchor, "toe_l must be coco-anchored"
    assert toe_r_i in skel.coco_anchor, "toe_r must be coco-anchored"
    assert skel.coco_anchor[toe_l_i] == "big_toe_l"
    assert skel.coco_anchor[toe_r_i] == "big_toe_r"


@requires_manny_fbx
def test_toe_bone_length_is_measured():
    """Toe bone length must come from median_dist(ankle, big_toe), not rest*spine_scale."""
    fbx = Skeleton("Manny.FBX")
    pts = np.zeros((20, 133, 3))
    # Set ankle-to-big-toe distance to 15 cm (measured).
    pts[:, 11, :] = [10, 0, 0]
    pts[:, 12, :] = [-10, 0, 0]
    pts[:, 15, :] = [10, -40, 0]   # left ankle
    pts[:, 16, :] = [-10, -40, 0]
    pts[:, 17, :] = [10, -55, 0]   # left big toe: 15 cm from ankle
    pts[:, 20, :] = [-10, -55, 0]
    skel = MocapSkeleton(pts, fbx_skel=fbx)
    toe_l_i = skel.name_to_idx["toe_l"]
    ankle_l_i = skel.name_to_idx["ankle_l"]
    toe_bl = skel.bone_lengths[toe_l_i]
    assert abs(toe_bl - 15.0) < 1.0, (
        f"toe bone length should be ~15cm (measured), got {toe_bl}"
    )

