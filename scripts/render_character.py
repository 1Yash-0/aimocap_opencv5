"""Render a solved animation on its FBX character with Blender.

The animation is applied exactly as in the FBX bake (``write_fbx``); the skinned
mesh that is already in the scene is then rendered, so what you see is what the
exported FBX plays in a game engine.

Usage::

    blender --background --python scripts/render_character.py -- \
        <solved.npy> <rig.fbx> <frames_dir> [--fps 60] [--step 2] [--seconds 10]

Frames are written as PNG; encode them with scripts/encode_frames.py.
"""

from __future__ import annotations

import argparse
import site
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
_user_site = site.getusersitepackages()
if _user_site and _user_site not in sys.path:
    sys.path.append(_user_site)


def _bounds(arm, frames):
    """World-space bounding box of the pelvis and feet over the given frames."""
    import bpy
    from mathutils import Vector

    lo = Vector((1e9, 1e9, 1e9))
    hi = Vector((-1e9, -1e9, -1e9))
    names = [n for n in ("pelvis", "head", "foot_l", "foot_r", "hand_l", "hand_r") if n in arm.pose.bones]
    for f in frames:
        bpy.context.scene.frame_set(f)
        for n in names:
            p = arm.matrix_world @ arm.pose.bones[n].head
            lo = Vector(map(min, lo, p))
            hi = Vector(map(max, hi, p))
    return lo, hi


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("npy")
    ap.add_argument("rig")
    ap.add_argument("out")
    ap.add_argument("--fps", type=float, default=60.0)
    ap.add_argument("--step", type=int, default=2, help="render every Nth frame")
    ap.add_argument("--seconds", type=float, default=10.0)
    args = ap.parse_args(argv)

    import bpy
    from mathutils import Vector

    from aimocap.retarget.fbx_export import write_fbx

    with tempfile.TemporaryDirectory() as tmp:
        write_fbx(args.npy, args.rig, str(Path(tmp) / "baked.fbx"), fps=args.fps)

    scn = bpy.context.scene
    arm = next(o for o in scn.objects if o.type == "ARMATURE")
    meshes = [o for o in scn.objects if o.type == "MESH"]
    print(f"[render] armature {arm.name}, meshes {[m.name for m in meshes]}")

    start = scn.frame_start + 1  # frame 1 of the bake is the bind-pose guard
    end = min(scn.frame_end, start + int(args.seconds * args.fps))
    lo, hi = _bounds(arm, range(start, end, 10))
    centre = (lo + hi) / 2
    height = max(hi.z - lo.z, 1.0)

    # Ground plane at the lowest foot position.
    bpy.ops.mesh.primitive_plane_add(size=40, location=(centre.x, centre.y, lo.z - 0.02))
    floor = bpy.context.active_object
    mat = bpy.data.materials.new("floor")
    mat.diffuse_color = (0.55, 0.57, 0.6, 1)
    floor.data.materials.append(mat)

    body = bpy.data.materials.new("body")
    body.diffuse_color = (0.78, 0.8, 0.84, 1)
    for m in meshes:
        # Only the highest level of detail is rendered.
        m.hide_render = not m.name.endswith("LOD0") and len(meshes) > 1
        m.data.materials.clear()
        m.data.materials.append(body)

    cam = bpy.data.objects.new("camera", bpy.data.cameras.new("camera"))
    scn.collection.objects.link(cam)
    dist = 2.6 * height + 1.5
    cam.location = centre + Vector((0.55 * dist, -dist, 0.25 * height))
    cam.rotation_euler = (centre + Vector((0, 0, 0.05 * height)) - cam.location).to_track_quat("-Z", "Y").to_euler()
    cam.data.lens = 50
    scn.camera = cam

    for name, energy, rot in (("key", 3.5, (0.9, 0.0, 0.6)), ("fill", 1.2, (1.1, 0.0, -1.9)), ("rim", 2.0, (0.6, 0.0, 3.0))):
        light = bpy.data.objects.new(name, bpy.data.lights.new(name, "SUN"))
        light.data.energy = energy
        light.rotation_euler = rot
        scn.collection.objects.link(light)
    world = bpy.data.worlds.new("world")
    world.color = (0.82, 0.84, 0.87)
    scn.world = world

    scn.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in {
        e.identifier for e in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items} else "BLENDER_EEVEE"
    scn.render.resolution_x, scn.render.resolution_y = 1280, 720
    scn.frame_start, scn.frame_end, scn.frame_step = start, end, args.step
    scn.render.image_settings.file_format = "PNG"
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    scn.render.filepath = str(out_dir / "frame_")
    bpy.ops.render.render(animation=True)
    print(f"[render] wrote {args.out}")
    return 0


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    raise SystemExit(main(argv))
