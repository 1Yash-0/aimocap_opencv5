"""Subprocess worker for crash-isolated ufbx world-orientation samples."""
from __future__ import annotations

import argparse
import json

import numpy as np
import ufbx

from aimocap.retarget.fbx_eval import build_node_lookup


def _rotation_xyzw(node) -> list[float]:
    m = node.node_to_world
    matrix = np.column_stack((
        np.array([m.c0.x, m.c0.y, m.c0.z], dtype=np.float64),
        np.array([m.c1.x, m.c1.y, m.c1.z], dtype=np.float64),
        np.array([m.c2.x, m.c2.y, m.c2.z], dtype=np.float64),
    ))
    u, _, vh = np.linalg.svd(matrix)
    rot = u @ vh
    if np.linalg.det(rot) < 0.0:
        u[:, -1] *= -1.0
        rot = u @ vh
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(rot).as_quat().tolist()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("fbx_path")
    parser.add_argument("--times", required=True, help="JSON list of FBX sample times")
    parser.add_argument("--bones", required=True, help="JSON list of node names")
    args = parser.parse_args()
    times = json.loads(args.times)
    bones = json.loads(args.bones)
    scene = ufbx.load_file(args.fbx_path)
    rows = []
    for time in times:
        evaluated = scene.evaluate(scene.anim, float(time))
        lookup = build_node_lookup(evaluated)
        rows.append({
            "time_sec": float(time),
            "world_quaternion_xyzw": {
                name: _rotation_xyzw(lookup[name]) for name in bones if name in lookup
            },
        })
    print(json.dumps(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
