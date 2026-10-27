"""
ufbx-based ground-truth FBX evaluator.

Loads an FBX, evaluates its animation at each frame, and returns per-joint
world translations. This is the **parity oracle**: whatever we export, this
module reads it back and tells us where the bones actually ended up —
independently of Blender, of Euler conventions, of anything.

If the npy FK (what the gif shows) and this FBX evaluation agree to <0.5 cm,
the export is provably correct. No eyeballing in Unreal.

Coordinate handling
-------------------
- The FBX (Manny) is authored Z-up, cm. ufbx returns transforms in that space.
- Our npy FK is also Z-up, cm. So world translations compare directly.
- We key on **node names** to pair joints across the two evaluations.

Duplicate-bone robustness
-------------------------
Some exporters (notably Blender's ``export_scene.fbx``) emit each bone twice:
once as the real deformer node and once as a leaf/helper. We disambiguate by
preferring the node that **has children** (real bones do; leaf duplicates don't),
falling back to the node whose rest world-translation matches the source rig.
"""

from __future__ import annotations

import numpy as np
import ufbx




def build_node_lookup(scene_eval) -> dict[str, object]:
    """Map bone name -> the *real* bone node, disambiguating duplicates.

    Among nodes sharing a name, keep the one with the most incoming connections
    (a parent bone is pointed to by its children). This survives the Blender
    duplicate-node export artifact.
    """
    grouped: dict[str, list] = {}
    for n in scene_eval.nodes:
        grouped.setdefault(n.name, []).append(n)

    lookup: dict[str, object] = {}
    for nm, nodes in grouped.items():
        if len(nodes) == 1:
            lookup[nm] = nodes[0]
            continue
        # pick the node with the most dst connections (most "parent-like")
        best = max(nodes, key=lambda n: len(n.connections_dst))
        lookup[nm] = best
    return lookup


def world_translations_at(scene_eval, bone_names: list[str]) -> dict[str, np.ndarray]:
    """From an evaluated ufbx scene, pull each named bone's world translation (cm).

    ufbx Matrix is column-major; translation lives in column ``c3``.
    """
    lookup = build_node_lookup(scene_eval)
    out: dict[str, np.ndarray] = {}
    for nm in bone_names:
        n = lookup.get(nm)
        if n is None:
            continue
        c = n.node_to_world.c3
        out[nm] = np.array([c.x, c.y, c.z], dtype=np.float64)
    return out




def fbx_world_positions(
    fbx_path: str,
    bone_names: list[str],
    num_frames: int,
    fps: float = 30.0,
    rest_frame_offset: int = 0,
) -> np.ndarray:
    """Evaluate an animated FBX frame-by-frame. Returns ``(F, len(bone_names), 3)``.

    Frames are 1-based in FBX; time = frame_index / fps seconds. We sample the
    same frame numbers our pipeline exported (1..num_frames), shifted by
    ``rest_frame_offset`` to skip a leading rest-pose frame.

    ``rest_frame_offset``: number of leading FBX frames to skip. Our FBX
    exporter (``fbx_export.write_fbx``) keys the rest pose at FBX frame 1 and
    the animation at frames 2..F+1, so the Bind Pose is preserved for UE
    retargeting. Callers verifying such an FBX against an F-frame npy must pass
    ``rest_frame_offset=1`` so FBX frames 2..F+1 are compared to npy frames
    0..F-1. Default 0 for backward compatibility (BVH, legacy FBXs).
    """
    s = ufbx.load_file(fbx_path)
    if len(s.anim_stacks) == 0:
        raise ValueError(f"{fbx_path} has no AnimStack — not animated.")

    out = np.zeros((num_frames, len(bone_names), 3), dtype=np.float64)
    for f in range(num_frames):
        time = (f + 1 + rest_frame_offset) / fps
        ev = s.evaluate(s.anim, time)
        w = world_translations_at(ev, bone_names)
        for k, nm in enumerate(bone_names):
            if nm in w:
                out[f, k] = w[nm]
    return out


def fbx_world_positions_at_frames(
    fbx_path: str,
    bone_names: list[str],
    frame_indices: list[int] | np.ndarray,
    fps: float = 30.0,
    rest_frame_offset: int = 0,
) -> np.ndarray:
    """Evaluate specific zero-based FBX frame indices.

    This is used by the parity gate to sample/chunk long clips.  Keeping chunks
    small avoids the native ufbx access violation observed when evaluating some
    exported FBX files frame-by-frame in one long process.

    ``rest_frame_offset``: see ``fbx_world_positions``. Shifts the evaluation
    time to skip a leading rest-pose frame baked by our exporter.
    """
    frames = np.asarray(frame_indices, dtype=np.int64)
    s = ufbx.load_file(fbx_path)
    if len(s.anim_stacks) == 0:
        raise ValueError(f"{fbx_path} has no AnimStack — not animated.")

    out = np.zeros((len(frames), len(bone_names), 3), dtype=np.float64)
    for i, f in enumerate(frames):
        time = (int(f) + 1 + rest_frame_offset) / fps
        ev = s.evaluate(s.anim, time)
        w = world_translations_at(ev, bone_names)
        for k, nm in enumerate(bone_names):
            if nm in w:
                out[i, k] = w[nm]
    return out


# ---------- npy-side ground truth (what the gif shows) -----------------------

def npy_world_positions(
    npy_path: str,
    source_rig_path: str,
    bone_names: list[str],
    num_frames: int | None = None,
) -> np.ndarray:
    """FK the npy with our Skeleton (built from the **source** rig) to get the
    ground-truth joint world positions.

    This is exactly what ``viz_bvh.py`` renders, so it is the pose the user has
    already confirmed looks correct. ``source_rig_path`` must be the *original*
    rig (e.g. Manny.FBX), NOT an exported FBX (which may have duplicated bones).
    """
    from aimocap.retarget.anim_state import load_anim_state
    from aimocap.retarget.fbx_rig import Skeleton

    state = load_anim_state(npy_path)
    skel = Skeleton(source_rig_path)
    F = state["num_frames"]
    if num_frames is not None:
        F = min(F, num_frames)

    idx_map = {nm: skel.name_to_idx[nm] for nm in bone_names if nm in skel.name_to_idx}
    out = np.zeros((F, len(bone_names), 3), dtype=np.float64)
    J_skel = skel.num_joints
    for f in range(F):
        local_q = state["quats"][f]
        if local_q.shape[0] != J_skel:
            raise ValueError(
                f"npy joint count ({local_q.shape[0]}) != skeleton joints ({J_skel}). "
                f"The npy was produced by a different Skeleton bone-set."
            )
        pos, _ = skel.get_forward_kinematics(local_q, root_translation=state["root_t"][f])
        for k, nm in enumerate(bone_names):
            if nm in idx_map:
                out[f, k] = pos[idx_map[nm]]
    return out


