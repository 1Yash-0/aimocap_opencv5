"""
FBX parity gate.

Compares the joint world positions produced by our npy FK (ground truth — what
the gif/BVH show and the user has confirmed) against the joint world positions
an exported FBX actually evaluates to (read back via ufbx).

PASS bar: max abs position error < 0.5 cm over all body bones, all frames.

Usage:
    python scripts/parity_test.py <npy> <fbx> [--fps 30] [--bones pelvis,head,...]
    python -m pytest tests/test_fbx_parity.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

import numpy as np

from aimocap.retarget.fbx_eval import fbx_world_positions, npy_world_positions

# Bones that actually move and matter for "does it look right". We exclude
# twist/finger/helper bones that our animation doesn't drive.
BODY_BONES = [
    "pelvis", "spine_01", "spine_02", "spine_03", "spine_04", "spine_05",
    "neck_01", "neck_02", "head",
    "clavicle_l", "upperarm_l", "lowerarm_l", "hand_l",
    "clavicle_r", "upperarm_r", "lowerarm_r", "hand_r",
    "thigh_l", "calf_l", "foot_l", "ball_l",
    "thigh_r", "calf_r", "foot_r", "ball_r",
]

PASS_BAR_CM = 0.5


def _fbx_world_positions_subprocess(
    fbx_path: str,
    bones: list[str],
    frame_indices: np.ndarray,
    fps: float,
    chunk_size: int,
    rest_frame_offset: int = 0,
) -> np.ndarray:
    """Evaluate FBX frames in small child processes.

    ufbx can native-crash on some long in-process evaluation loops.  A child
    process turns that into a normal non-zero return code and lets the gate
    fail safely without taking the parent process with it.
    """
    out = np.zeros((len(frame_indices), len(bones), 3), dtype=np.float64)
    index_to_out = {int(f): i for i, f in enumerate(frame_indices.tolist())}
    for start in range(0, len(frame_indices), chunk_size):
        chunk = frame_indices[start:start + chunk_size].astype(int).tolist()
        cmd = [
            sys.executable,
            "-m",
            "aimocap.retarget.parity_worker",
            fbx_path,
            "--bones",
            json.dumps(bones),
            "--frames",
            json.dumps(chunk),
            "--fps",
            str(fps),
            "--rest-frame-offset",
            str(rest_frame_offset),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                "FBX parity worker failed "
                f"(frames {chunk[0]}..{chunk[-1]}, code {proc.returncode}): "
                f"{proc.stderr.strip() or proc.stdout.strip()}"
            )
        payload = json.loads(proc.stdout)
        positions = np.asarray(payload["positions"], dtype=np.float64)
        for local_i, frame in enumerate(payload["frames"]):
            out[index_to_out[int(frame)]] = positions[local_i]
    return out


def _detect_rest_frame_offset(fbx_path: str, npy_frames: int, fps: float) -> int:
    """Detect whether ``fbx_path`` has a leading rest-pose frame baked by our
    exporter (``fbx_export.write_fbx`` keys the rest pose at FBX frame 1 and
    the animation at frames 2..F+1 so the Bind Pose is preserved for UE
    retargeting).

    Returns 1 if the FBX AnimStack ``time_end`` matches ``npy_frames + 1``
    frames (within half a frame), else 0. We use ``time_end`` (not duration)
    because the stack spans frames 1..F+1, giving duration F frames but
    ``time_end = (F+1)/fps``. Loaded in a child process to avoid the ufbx
    two-FBX-in-one-process segfault.
    """
    import json, subprocess, sys
    cmd = [
        sys.executable, "-c",
        "import ufbx,json,sys; s=ufbx.load_file(sys.argv[1]); "
        "st=s.anim_stacks[0]; "
        "print(json.dumps({'time_end': st.time_end}))",
        fbx_path,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if proc.returncode != 0:
            return 0
        time_end = json.loads(proc.stdout)["time_end"]
        fbx_end_frame = time_end * fps
        if abs(fbx_end_frame - (npy_frames + 1)) < 0.5:
            return 1
        return 0
    except Exception:
        return 0


def run_parity(npy_path: str, fbx_path: str, source_rig_path: str,
               fps: float = 30.0,
               bone_names: list[str] | None = None,
               verbose: bool = True,
               sample_stride: int = 1,
               chunk_size: int = 30,
               safe_fbx_eval: bool = True) -> dict:
    """Compare npy-FK ground truth (built from ``source_rig_path``) against the
    joint world positions the exported ``fbx_path`` actually evaluates to.

    ``source_rig_path`` must be the *original* rig (e.g. Manny.FBX); the npy was
    generated against it. ``fbx_path`` is the exported animation under test.
    """
    bones = bone_names or BODY_BONES
    npy_pos = npy_world_positions(npy_path, source_rig_path, bones)
    F = npy_pos.shape[0]
    stride = max(1, int(sample_stride))
    frame_indices = np.arange(0, F, stride, dtype=np.int64)
    if frame_indices[-1] != F - 1:
        frame_indices = np.append(frame_indices, F - 1)
    npy_pos_cmp = npy_pos[frame_indices]

    # Detect our exporter's leading rest-pose frame and skip it if present.
    rest_offset = _detect_rest_frame_offset(fbx_path, F, fps)
    if verbose and rest_offset:
        print(f"  [parity] FBX has a leading rest-pose frame; skipping FBX frame 1 (rest_offset=1)")

    if safe_fbx_eval:
        fbx_pos = _fbx_world_positions_subprocess(
            fbx_path,
            bones,
            frame_indices,
            fps=fps,
            chunk_size=max(1, int(chunk_size)),
            rest_frame_offset=rest_offset,
        )
    else:
        fbx_full = fbx_world_positions(fbx_path, bones, num_frames=F, fps=fps,
                                       rest_frame_offset=rest_offset)
        fbx_pos = fbx_full[frame_indices]

    diff = npy_pos_cmp - fbx_pos                   # (S, B, 3)
    per_bone_err = np.linalg.norm(diff, axis=-1)   # (F, B) cm
    max_err = float(per_bone_err.max())
    mean_err = float(per_bone_err.mean())
    p95_err = float(np.percentile(per_bone_err, 95))

    # worst bone
    per_bone_max = per_bone_err.max(axis=0)        # (B,)
    worst_i = int(per_bone_max.argmax())
    worst_bone = bones[worst_i] if worst_i < len(bones) else "?"

    result = {
        "pass": max_err < PASS_BAR_CM,
        "max_err_cm": max_err,
        "mean_err_cm": mean_err,
        "p95_err_cm": p95_err,
        "worst_bone": worst_bone,
        "worst_bone_err_cm": float(per_bone_max[worst_i]),
        "num_frames": F,
        "num_sampled_frames": int(len(frame_indices)),
        "sample_stride": int(stride),
        "num_bones": len(bones),
    }

    if verbose:
        status = "PASS [OK]" if result["pass"] else "FAIL [X]"
        print(f"\n=== FBX Parity: {status} (bar < {PASS_BAR_CM} cm) ===")
        print(f"  frames     : {F} ({len(frame_indices)} sampled, stride {stride})")
        print(f"  bones      : {len(bones)}")
        print(f"  max error  : {max_err:.3f} cm")
        print(f"  mean error : {mean_err:.3f} cm")
        print(f"  p95 error  : {p95_err:.3f} cm")
        print(f"  worst bone : {worst_bone} ({per_bone_max[worst_i]:.3f} cm)")
        print(f"  per-bone max (cm):")
        for k, nm in enumerate(bones):
            mark = "  ***" if per_bone_max[k] > PASS_BAR_CM else ""
            print(f"    {nm:14s} {per_bone_max[k]:8.3f}{mark}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npy_path")
    ap.add_argument("fbx_path")
    ap.add_argument("source_rig_path", help="original rig the npy was built against (e.g. Manny.FBX)")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--bones", default=None, help="comma-separated bone names")
    ap.add_argument("--sample-stride", type=int, default=1,
                    help="Evaluate every Nth frame plus the last frame (default: 1).")
    ap.add_argument("--chunk-size", type=int, default=30,
                    help="Subprocess FBX evaluation chunk size (default: 30).")
    ap.add_argument("--unsafe-inprocess-fbx", action="store_true",
                    help="Use old in-process ufbx evaluation path.")
    args = ap.parse_args()
    bones = args.bones.split(",") if args.bones else None
    res = run_parity(args.npy_path, args.fbx_path, args.source_rig_path,
                     fps=args.fps, bone_names=bones,
                     sample_stride=args.sample_stride,
                     chunk_size=args.chunk_size,
                     safe_fbx_eval=not args.unsafe_inprocess_fbx)
    sys.exit(0 if res["pass"] else 1)


if __name__ == "__main__":
    main()
