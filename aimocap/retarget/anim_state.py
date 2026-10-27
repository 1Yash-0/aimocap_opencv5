"""
Gimbal-safe animation-state conversion.

The retargeted .npy stores per-frame per-joint local rotations as **XYZ Euler
radians**: ``[root_t(3), e0x,e0y,e0z, e1x,e1y,e1z, ...]``.

Manny's rest pose is near a gimbal singularity (``pelvis`` rest Euler ≈
``[-90, -86, +90]`` deg). Re-decomposing a rotation through Euler near that
singularity yields *different but mathematically equivalent* Euler triples, so
the raw numbers flap frame-to-frame (looks like 80+ deg jitter, is actually the
same orientation). Any tool that consumes the *numbers* (Blender import, BVH
re-export) then sees garbage.

Fix: lift the animation to **quaternions** and sign-align them across frames so
the double-cover is continuous. Quaternions have no singularity. The BVH/gif
pipeline already does this implicitly; the FBX pipeline must do it explicitly.

This module is pure NumPy/SciPy. No Blender, no ufbx. Fully unit-testable.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def load_npy(npy_path: str) -> np.ndarray:
    """Load the solved animation array of shape ``(F, 3 + J*3)``."""
    arr = np.load(npy_path).astype(np.float64)
    if arr.ndim != 2 or arr.shape[1] < 6:
        raise ValueError(f"Expected (F, 3+J*3) array, got {arr.shape}")
    return arr


def split_root_and_eulers(npy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split into ``root_translation (F,3)`` and ``euler_rad (F,J,3)``.

    Root translation is in FBX/Z-up cm, relative to the FBX root rest
    translation (as written by ``engine.py``).
    """
    root_t = npy[:, 0:3].copy()
    eulers = npy[:, 3:].reshape(npy.shape[0], -1, 3).copy()
    return root_t, eulers


def eulers_to_quats(euler_rad: np.ndarray) -> np.ndarray:
    """XYZ Euler radians ``(F,J,3)`` -> quaternions xyzw ``(F,J,4)``.

    Uses scipy's intrinsic XYZ decomposition (matches Manny ``RotationOrder =
    XYZ`` = 0, verified against ufbx). No singularity; one canonical orientation
    per input triple.
    """
    F, J, _ = euler_rad.shape
    flat = euler_rad.reshape(F * J, 3)
    quats = Rotation.from_euler("xyz", flat).as_quat()  # (F*J, 4) xyzw
    return quats.reshape(F, J, 4)


def align_quat_continuity(quats: np.ndarray) -> np.ndarray:
    """Sign-align quaternions so consecutive frames take the short arc.

    ``q`` and ``-q`` represent the same rotation. Raw solver output / Euler
    conversion can flip signs arbitrarily. For a continuous animation we want
    each frame's quaternion to sit in the same hemisphere as the previous one
    (dot >= 0). Operates per joint across the time axis.

    This is the operation that removes the apparent "jitter": the orientation
    was never changing, only the representative sign was flapping.
    """
    out = quats.copy()
    # walk frames; flip to match previous frame's hemisphere
    for f in range(1, out.shape[0]):
        prev = out[f - 1]              # (J,4)
        cur = out[f]                   # (J,4)
        dots = np.sum(prev * cur, axis=-1)        # (J,)
        flip = dots < 0.0                           # (J,)
        cur[flip] = -cur[flip]
        out[f] = cur
    return out


def load_anim_state(npy_path: str) -> dict:
    """Load + lift a solved .npy to a gimbal-safe animation state.

    Returns
    -------
    dict with:
        root_t   : (F,3)  root translation, FBX Z-up cm (relative to FBX root rest)
        quats    : (F,J,4) continuous local quaternions (xyzw)
        num_frames, num_joints
    """
    npy = load_npy(npy_path)
    root_t, eulers = split_root_and_eulers(npy)
    quats = eulers_to_quats(eulers)
    quats = align_quat_continuity(quats)
    return {
        "root_t": root_t,
        "quats": quats,
        "num_frames": npy.shape[0],
        "num_joints": eulers.shape[1],
    }
