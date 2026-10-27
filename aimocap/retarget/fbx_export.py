"""
FBX exporter: bakes a solved .npy animation onto a source FBX rig via Blender.

Design (what every prior attempt got wrong, and how this is fixed):
  - The .npy stores ABSOLUTE local rotations per joint. Blender's pose_bone
    rotation is a DELTA from the bone's REST pose, so:
        pose_quat = rest_local^-1 @ absolute_local_quat
    (verified: Blender's bone rest rotations match ufbx's exactly.)
  - We author QUATERNION F-curves, never Euler — the pelvis rest Euler
    [-90, -86, +90] is a gimbal singularity; quaternions are singularity-free.
  - We do NOT use pose_bone.matrix / world-matrix transfer: that path lets
    Blender re-derive each bone's ROLL (rotation about its length axis), which
    differs from Manny's authored roll and accumulates down the chain into a
    visible lateral twist. Driving rotation_quaternion directly preserves roll.
  - The FBX ``root`` helper node becomes Blender's armature OBJECT on import,
    not a bone. Root translation therefore goes on the armature object's
    location, not on any pose bone.
  - We export Z-up (axis_up='Z') to match the source rig's native frame.
  - REST-POSE FRAME AT FRAME 1 (the Unreal-twist fix): Blender's FBX exporter
    with bake_anim=True bakes the FIRST keyed frame's pose as the FBX "Bind
    Pose" — the very pose Unreal reads as the rest skeleton and retargets
    against. If frame 1 carries animation (the old behavior), the Bind Pose
    becomes R_abs[0] instead of R_src, and UE's per-bone delta retargeting
    (R_src * R_bind^-1 * animated_local) introduces a twist of
    R_src * R_abs[0]^-1 (up to ~134deg on upperarms, 92deg on pelvis). Fix:
    keyframe the identity-delta rest pose on every bone + rest root translation
    at frame 1, and shift the actual animation to frames 2..F+1. Then Bind Pose
    == R_src, UE's retargeting recovers R_abs exactly, and the parity gate
    still passes on frames 2..F+1.

Correctness is verified independently by ``aimocap.retarget.parity``, which
reads the exported FBX back with ufbx and checks joint world positions against
the npy FK to <0.5 cm. This module does not self-verify; the parity gate does.
"""

from __future__ import annotations

import os

import numpy as np
from scipy.spatial.transform import Rotation


def _pose_quats_from_absolute(abs_quats: np.ndarray, rest_quats: np.ndarray) -> np.ndarray:
    """Absolute local quats -> rest-relative pose quats.

    abs_quats  : (F, J, 4) xyzw, absolute local rotation per joint per frame.
    rest_quats : (J, 4)    xyzw, absolute rest local rotation per joint.

    Returns (F, J, 4) pose = rest^-1 @ absolute (xyzw). This is the single
    load-bearing transform; writing absolutes straight into Blender would
    pre-rotate every bone by its own rest twist ("Zamasu").
    """
    F, J, _ = abs_quats.shape
    R_abs = Rotation.from_quat(abs_quats.reshape(F * J, 4))
    R_rest_tiled = Rotation.from_quat(np.tile(rest_quats, (F, 1)))   # (F*J,)
    pose = (R_rest_tiled.inv() * R_abs).as_quat().reshape(F, J, 4)
    return pose


def write_fbx(
    npy_path: str,
    fbx_in_path: str,
    fbx_out_path: str,
    fps: float = 30.0,
    bone_names: list[str] | None = None,
) -> None:
    """Bake the .npy animation onto ``fbx_in_path`` and export to ``fbx_out_path``.

    Parameters
    ----------
    npy_path : solved animation, shape (F, 3 + J*3) XYZ Euler radians, Z-up cm.
    fbx_in_path : source rig (e.g. Manny.FBX).
    fbx_out_path : destination animated FBX.
    fps : frame rate.
    bone_names : optional whitelist; ignored (API compat). The npy defines coverage.
    """
    import bpy

    from aimocap.retarget.anim_state import load_anim_state
    from aimocap.retarget.fbx_rig import Skeleton

    # ── 1. Load animation + rig conventions ────────────────────────────────────
    state = load_anim_state(npy_path)
    abs_quats = state["quats"]                       # (F, J, 4)
    root_t = state["root_t"]                         # (F, 3) cm, Z-up
    F, J, _ = abs_quats.shape

    rig = Skeleton(fbx_in_path)
    if J != rig.num_joints:
        raise ValueError(
            f"npy has {J} joints but rig '{fbx_in_path}' has {rig.num_joints}."
        )
    rest_quats = np.array(rig.rest_rotations, dtype=np.float64)   # (J,4) xyzw

    # Absolute local -> rest-relative pose quaternion (the load-bearing step).
    pose_quats = _pose_quats_from_absolute(abs_quats, rest_quats)  # (F,J,4)

    # ── 2. Reset Blender, import the source rig ────────────────────────────────
    print(f"[FBX Export] Importing rig: {fbx_in_path}")
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=fbx_in_path, automatic_bone_orientation=False)

    arm = next((o for o in bpy.context.scene.objects if o.type == "ARMATURE"), None)
    if arm is None:
        raise RuntimeError(f"No armature in {fbx_in_path}")
    print(f"[FBX Export] Armature '{arm.name}': {len(arm.pose.bones)} pose bones")

    # The FBX 'root' helper node is imported as the armature OBJECT. Root motion
    # therefore rides on the object, keyed per frame below.
    pb_by_name = {pb.name: pb for pb in arm.pose.bones}
    pairs = []  # (skel_idx, pose_bone) in skeleton (topological) order
    for skel_idx, nm in enumerate(rig.node_names):
        pb = pb_by_name.get(nm)
        if pb is not None:
            pairs.append((skel_idx, pb))
    print(f"[FBX Export] Mapped {len(pairs)}/{J} bones to Blender pose bones")

    # ── 3. Create the animation action + slot (Blender 5 API) ──────────────────
    if arm.animation_data is None:
        arm.animation_data_create()
    action = bpy.data.actions.new(name="AIMocapRetarget")
    slot = action.slots.new(id_type="OBJECT", name=arm.name)
    arm.animation_data.action = action
    arm.animation_data.action_slot = slot

    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = F + 1  # +1 for the rest-pose frame at frame 1
    bpy.context.scene.render.fps = int(round(fps))

    # ── 4. Switch every mapped bone to quaternion mode ─────────────────────────
    for _skel_idx, pb in pairs:
        pb.rotation_mode = "QUATERNION"

    # ── 5. Keyframe the REST POSE at frame 1 ───────────────────────────────────
    # Blender's FBX exporter (bake_anim=True) bakes the FIRST keyed frame's pose
    # as the FBX "Bind Pose" (what importers like UE read as the rest skeleton).
    # If frame 1 carries animation, that animated pose becomes the rest, and
    # every consumer that retargets against its own Manny rest sees a
    # per-bone twist equal to R_src * R_abs[0]^-1 (up to ~134deg on upperarms).
    # Fix: keyframe the identity-delta (rest) pose on every bone at frame 1, and
    # also the rest root translation, so Blender bakes the true rest as Bind
    # Pose. The actual animation is then keyed on frames 2..F+1.
    for _skel_idx, pb in pairs:
        pb.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)  # identity delta == rest
        pb.keyframe_insert(data_path="rotation_quaternion", frame=1)
    arm.location = (0.0, 0.0, 0.0)  # Manny root rest translation
    arm.keyframe_insert(data_path="location", frame=1)

    # ── 6. Key every animation frame on frames 2..F+1 ──────────────────────────
    # Blender quaternion order is (w, x, y, z); scipy is (x, y, z, w).
    for f in range(F):
        frame_number = f + 2  # shifted by 1 to follow the rest-pose frame

        # Bone rotations (rest-relative deltas). Driving rotation_quaternion
        # directly — not pose_bone.matrix — preserves Manny's authored bone roll.
        for skel_idx, pb in pairs:
            x, y, z, w = pose_quats[f, skel_idx]
            pb.rotation_quaternion = (w, x, y, z)
            pb.keyframe_insert(data_path="rotation_quaternion", frame=frame_number)

        # Root translation on the armature object (the FBX 'root' node).
        arm.location = tuple(root_t[f])
        arm.keyframe_insert(data_path="location", frame=frame_number)

        if f % 50 == 0:
            print(f"[FBX Export] keyed frame {frame_number}/{F + 1}")

    # ── 6. Export a clean single skeleton, Z-up to match the source frame ──────
    bpy.ops.object.select_all(action="DESELECT")
    arm.select_set(True)
    bpy.context.view_layer.objects.active = arm

    print(f"[FBX Export] Exporting to {fbx_out_path}")
    out_dir = os.path.dirname(os.path.abspath(fbx_out_path))
    os.makedirs(out_dir or ".", exist_ok=True)
    bpy.ops.export_scene.fbx(
        filepath=fbx_out_path,
        use_selection=True,
        add_leaf_bones=False,
        bake_anim=True,
        bake_anim_use_all_bones=True,
        bake_anim_use_nla_strips=False,
        bake_anim_use_all_actions=False,
        bake_anim_force_startend_keying=True,
        bake_anim_step=1.0,
        bake_anim_simplify_factor=0.0,
        object_types={"ARMATURE"},
        axis_up="Z",
        axis_forward="Y",
        apply_unit_scale=False,
        use_space_transform=False,
    )
    print(f"[FBX Export] Done -> {fbx_out_path}")
