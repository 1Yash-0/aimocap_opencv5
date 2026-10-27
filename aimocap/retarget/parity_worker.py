"""Subprocess worker for crash-isolated FBX parity evaluation."""

from __future__ import annotations

import argparse
import json

import numpy as np

from aimocap.retarget.fbx_eval import fbx_world_positions_at_frames


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("fbx_path")
    ap.add_argument("--bones", required=True, help="JSON list of bone names")
    ap.add_argument("--frames", required=True, help="JSON list of zero-based frame indices")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--rest-frame-offset", type=int, default=0,
                    help="Leading FBX rest-pose frames to skip (1 for our exporter's output)")
    args = ap.parse_args()

    bones = json.loads(args.bones)
    frames = json.loads(args.frames)
    pos = fbx_world_positions_at_frames(
        args.fbx_path, bones, np.array(frames, dtype=np.int64),
        fps=args.fps, rest_frame_offset=args.rest_frame_offset,
    )
    print(json.dumps({"frames": frames, "positions": pos.tolist()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
