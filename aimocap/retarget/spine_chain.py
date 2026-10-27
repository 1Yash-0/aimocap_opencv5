"""Spine target distribution for the proxy skeleton.

The 3D input gives only 2 spine points (pelvis, neck) but a real rig has N
spine joints in between. To let the upper body bend, we distribute the
pelvis->neck measurement across N-1 intermediate joints.  The rest spine
is nearly straight in Manny, so a rigid rotation+scale produces a straight
line of targets — no curvature information.  During a forward bend the
real spine curves, so we add a forward arc to the intermediates
proportional to the bend angle.

Pure numpy. No Blender, no ufbx.
"""
from __future__ import annotations

import numpy as np


def distribute_spine_targets(
    pelvis: np.ndarray, neck: np.ndarray, rest_positions: np.ndarray
) -> np.ndarray:
    """Place N-1 intermediate spine targets between pelvis and neck, with
    curvature proportional to the bend angle.

    Parameters
    ----------
    pelvis, neck : (3,) the two measured spine endpoints.
    rest_positions : (N, 3) rest global positions of all spine joints (pelvis to neck).

    Returns
    -------
    (N-1, 3) intermediate target positions (excludes pelvis and neck).
    """
    pelvis = np.asarray(pelvis, dtype=np.float64).flatten()
    neck = np.asarray(neck, dtype=np.float64).flatten()
    rest_positions = np.asarray(rest_positions, dtype=np.float64)

    n_inter = len(rest_positions) - 2
    if n_inter <= 0:
        return np.zeros((0, 3), dtype=np.float64)

    rest_pelvis = rest_positions[0]
    rest_neck = rest_positions[-1]

    rest_vec = rest_neck - rest_pelvis
    tgt_vec = neck - pelvis

    rest_len = np.linalg.norm(rest_vec)
    tgt_len = np.linalg.norm(tgt_vec)

    if rest_len < 1e-9 or tgt_len < 1e-9:
        return np.linspace(pelvis, neck, n_inter + 2)[1:-1]

    scale = tgt_len / rest_len

    rest_dir = rest_vec / rest_len
    tgt_dir = tgt_vec / tgt_len

    # shortest path rotation
    axis = np.cross(rest_dir, tgt_dir)
    sin_angle = np.linalg.norm(axis)
    cos_angle = np.dot(rest_dir, tgt_dir)

    if sin_angle > 1e-9:
        axis = axis / sin_angle
        angle = np.arctan2(sin_angle, cos_angle)
        from scipy.spatial.transform import Rotation
        R = Rotation.from_rotvec(axis * angle).as_matrix()
    elif cos_angle < 0:  # 180 flip
        R = -np.eye(3)
    else:
        R = np.eye(3)

    # Apply rigid rotation + scale to intermediate joints
    inter_rest = rest_positions[1:-1] - rest_pelvis
    inter_tgt = (R @ inter_rest.T).T * scale + pelvis

    # Add curvature: arc the intermediates forward proportional to the bend
    # angle.  The "forward" direction is perpendicular to the chord, in the
    # plane defined by the rest-spine "forward" direction and the chord.
    bend_angle = abs(np.arctan2(sin_angle, cos_angle))
    if bend_angle > 0.35:  # > 20 deg bend: add curvature
        # Forward direction in rest frame: perpendicular to rest spine,
        # in the plane of rotation.
        if sin_angle > 1e-9:
            fwd_rest = np.cross(rest_dir, axis)
        else:
            fwd_rest = np.zeros(3)
        fwd_tgt = R @ fwd_rest if sin_angle > 1e-9 else np.zeros(3)
        # Project fwd_tgt perpendicular to the chord direction
        chord_dir = tgt_dir
        fwd_perp = fwd_tgt - np.dot(fwd_tgt, chord_dir) * chord_dir
        fwd_norm = np.linalg.norm(fwd_perp)
        if fwd_norm > 1e-6:
            fwd_perp = fwd_perp / fwd_norm
            arc_mag = 0.15 * bend_angle * tgt_len  # 15% of bend x length
            for k in range(n_inter):
                t = (k + 1) / (n_inter + 1)  # 0 < t < 1
                bulge = arc_mag * np.sin(np.pi * t)
                inter_tgt[k] += fwd_perp * bulge

    return inter_tgt
