"""World-orientation parity for solved NPYs versus evaluated animated FBXs.

Kept separate from the position-only parity gate; this reports measurements
without imposing a pass threshold.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from aimocap.retarget.fbx_rig import Skeleton


def _stats(values) -> dict:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if not len(x):
        return {"n": 0, "p50_deg": None, "p95_deg": None, "max_deg": None}
    return {"n": int(len(x)), "p50_deg": float(np.percentile(x, 50)),
            "p95_deg": float(np.percentile(x, 95)), "max_deg": float(np.max(x))}


def _npy_world_rotations(npy_path: str | Path, rig_path: str | Path) -> tuple[np.ndarray, list[str]]:
    animation = np.load(npy_path, allow_pickle=False)
    skel = Skeleton(str(rig_path))
    expected = 3 + 3 * skel.num_joints
    if animation.ndim != 2 or animation.shape[1] != expected:
        raise ValueError(f"expected solved NPY shape (F,{expected}), got {animation.shape}")
    f_count = len(animation)
    eulers = animation[:, 3:].reshape(f_count, skel.num_joints, 3)
    local = Rotation.from_euler("xyz", eulers.reshape(-1, 3)).as_quat().reshape(
        f_count, skel.num_joints, 4)
    world = np.empty_like(local)
    for f in range(f_count):
        _, world[f] = skel.get_forward_kinematics(local[f], root_translation=animation[f, :3])
    return world, list(skel.node_names)


def _ufbx_world_rotations(
    fbx_path: str | Path,
    sample_times: list[float],
    bone_names: list[str],
    *,
    chunk_size: int,
    work_dir: Path,
) -> dict[float, dict[str, np.ndarray]]:
    worker_module = "aimocap.retarget.orientation_parity_worker"
    out: dict[float, dict[str, np.ndarray]] = {}
    for chunk_idx, start in enumerate(range(0, len(sample_times), chunk_size)):
        times = sample_times[start:start + chunk_size]
        cmd = [sys.executable, "-m", worker_module, str(fbx_path),
               "--times", json.dumps(times), "--bones", json.dumps(bone_names)]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if proc.returncode:
            raise RuntimeError(
                f"ufbx orientation worker failed at sample chunk {chunk_idx} "
                f"({times[0]}..{times[-1]} sec, exit {proc.returncode}): "
                f"{proc.stderr.strip() or proc.stdout.strip()}"
            )
        rows = json.loads(proc.stdout)
        for row in rows:
            out[float(row["time_sec"])] = {
                name: np.asarray(q, dtype=np.float64)
                for name, q in row["world_quaternion_xyzw"].items()
            }
    return out


def run_orientation_parity(
    npy_path: str | Path,
    fbx_path: str | Path,
    rig_path: str | Path,
    *,
    fps: float = 30.0,
    subframe_factor: int = 4,
    chunk_size: int = 30,
    work_dir: str | Path | None = None,
) -> dict:
    """Compare every NPY key and uniform subframe sample on each key interval."""
    for path in (Path(npy_path), Path(fbx_path), Path(rig_path)):
        if not path.is_file():
            raise FileNotFoundError(path)
    npy_world, all_bones = _npy_world_rotations(npy_path, rig_path)
    frame_count = len(npy_world)
    bones = [name for name in all_bones if name in {
        "pelvis", "spine_01", "spine_02", "spine_03", "spine_04", "spine_05",
        "neck_01", "neck_02", "head", "clavicle_l", "upperarm_l", "lowerarm_l", "hand_l",
        "clavicle_r", "upperarm_r", "lowerarm_r", "hand_r", "thigh_l", "calf_l", "foot_l", "ball_l",
        "thigh_r", "calf_r", "foot_r", "ball_r",
    }]
    if not bones:
        raise ValueError("none of the expected body bones are present on the source rig")

    # Exporter convention: FBX frame 1 is rest; NPY frame f is keyed at FBX frame f+2.
    coords = [(f, 0.0) for f in range(frame_count)]
    for f in range(frame_count - 1):
        for sub in range(1, subframe_factor):
            coords.append((f, sub / subframe_factor))
    coords.sort(key=lambda pair: pair[0] + pair[1])
    sample_times = [(f + frac + 2.0) / fps for f, frac in coords]

    if work_dir is None:
        with tempfile.TemporaryDirectory(prefix="aimocap_orientation_parity_") as temp:
            fbx_world = _ufbx_world_rotations(fbx_path, sample_times, bones,
                                               chunk_size=max(1, chunk_size),
                                               work_dir=Path(temp))
    else:
        work = Path(work_dir)
        work.mkdir(parents=True, exist_ok=True)
        fbx_world = _ufbx_world_rotations(fbx_path, sample_times, bones,
                                           chunk_size=max(1, chunk_size), work_dir=work)

    key_errors = {bone: [] for bone in bones}
    sub_errors = {bone: [] for bone in bones}
    key_records = []
    sub_records = []
    for f, frac in coords:
        time = (f + frac + 2.0) / fps
        evaluated = fbx_world[time]
        row = {"npy_frame_coordinate": float(f + frac), "fbx_time_sec": float(time)}
        for bone in bones:
            idx = all_bones.index(bone)
            r_fbx = Rotation.from_quat(evaluated[bone])
            if frac == 0.0:
                r_ref = Rotation.from_quat(npy_world[f, idx])
                err = float(np.degrees((r_ref.inv() * r_fbx).magnitude()))
                key_errors[bone].append(err)
            else:
                slerp = Slerp([0.0, 1.0], Rotation.from_quat(npy_world[f:f + 2, idx]))
                r_ref = slerp([frac])[0]
                err = float(np.degrees((r_ref.inv() * r_fbx).magnitude()))
                sub_errors[bone].append(err)
            row[bone] = err
        (key_records if frac == 0.0 else sub_records).append(row)

    return {
        "method": "NPY source-rig FK world quaternions vs ufbx-evaluated FBX world rotations",
        "npy_frames": frame_count,
        "fbx_fps": float(fps),
        "subframe_factor": int(subframe_factor),
        "fbx_frame_mapping": "NPY frame f -> FBX time (f+2)/fps; frame 1 is rest",
        "between_key_reference": "shortest-arc SciPy Slerp between adjacent NPY world quaternions",
        "key_sample_count": len(key_records),
        "between_key_sample_count": len(sub_records),
        "per_bone": {
            bone: {"at_keys": _stats(key_errors[bone]),
                   "between_keys": _stats(sub_errors[bone])}
            for bone in bones
        },
        "overall": {"at_keys": _stats([v for values in key_errors.values() for v in values]),
                    "between_keys": _stats([v for values in sub_errors.values() for v in values])},
        "per_key_errors_deg": key_records,
        "per_subframe_errors_deg": sub_records,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("npy")
    parser.add_argument("fbx")
    parser.add_argument("rig")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--chunk-size", type=int, default=30)
    parser.add_argument("--output")
    args = parser.parse_args()
    result = run_orientation_parity(args.npy, args.fbx, args.rig,
                                    fps=args.fps, chunk_size=args.chunk_size)
    text = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    print(text)
