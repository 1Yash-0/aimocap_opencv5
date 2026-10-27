"""Blender driver: run the production FBX bake with Blender's embedded Python.

Invocation:
  blender --background --python scripts/blender_bake_fbx.py -- <npy> <rig.fbx> <out.fbx> [--fps F]

This calls ``aimocap.retarget.fbx_export.write_fbx`` EXACTLY as the CLI/scripts
path do; the only difference is which Python interpreter provides ``bpy``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy  # noqa: F401  (verify Blender's numpy is present before bpy)

# Blender's embedded Python is portable, so user-site packages are not on
# its sys.path. The pip target for its bin/python lands there; bolt it on
# so ufbx/scipy (FBX rig loader, rotation math) import in-session.
import site

_user_site = site.getusersitepackages()
if _user_site and _user_site not in sys.path:
    sys.path.append(_user_site)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("npy")
    ap.add_argument("rig")
    ap.add_argument("out")
    ap.add_argument("--fps", type=float, default=30.0)
    args = ap.parse_args(argv)

    from aimocap.retarget.fbx_export import write_fbx

    print(f"[Blender driver] bake {args.npy} onto {args.rig} -> {args.out}")
    write_fbx(args.npy, args.rig, args.out, fps=args.fps)
    print("[Blender driver] write_fbx returned OK")
    return 0


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:]
    raise SystemExit(main(argv))
