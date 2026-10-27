"""
Root-frame orientation via orthonormal basis alignment.

Replaces the buggy 2-vector ``Rotation.align_vectors`` in ``analytic_init``.
That call misused align_vectors on non-orthogonal input (hip-line and spine
are heavily correlated when the actor leans, dot ~ -300), so Procrustes
returned a compromise rotation fitting neither vector — 13cm neck error.

Fix: build an orthonormal frame from each side (Gram-Schmidt), then solve the
rotation between two orthonormal frames. Two orthonormal bases always have the
same inner-product structure (identity), so the Procrustes fit is exact for
the frame's defining vectors and well-conditioned.

A pure rotation can't fit data whose proportions differ from the rest skeleton
(the actor's hip-width / spine-length differ from Manny's). So we also solve a
per-frame SCALE: the desired spine length vs the rest spine length. The root
is applied as R @ (s * v), letting the spine chain stretch/compress to the
actual measurement while limbs (whose lengths are matched individually)
remain rigid.

Frame convention (right-handed, matches Manny rest):
    z-axis = spine direction  (pelvis -> neck, "up" the body)
    x-axis = hip line         (right hip - left hip, "across" the body)
    y-axis = z cross x        ("forward", "out the chest")

This module is pure numpy/scipy. Unit-testable, no Blender, no ufbx.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def _orthonormal_frame(spine_dir: np.ndarray, hip_line: np.ndarray) -> np.ndarray:
    """Build a right-handed orthonormal basis (3x3, columns = axes) from two
    directions. Gram-Schmidt: z = spine, x = hip made perpendicular to z.
    """
    if not (np.all(np.isfinite(spine_dir)) and np.all(np.isfinite(hip_line))):
        return np.eye(3)
    nz = np.linalg.norm(spine_dir)
    if nz < 1e-9:
        return np.eye(3)
    z = spine_dir / nz
    # hip-line made orthogonal to z
    x = hip_line - np.dot(hip_line, z) * z
    nx = np.linalg.norm(x)
    lateral_mag = np.linalg.norm(hip_line)
    # Relative threshold: if the perpendicular component is < 10% of the
    # original lateral, the lateral is nearly parallel to z and the residual
    # is noise-dominated. Fall back to the world axis least aligned with z.
    if nx < 1e-9 or (lateral_mag > 1e-9 and nx < 0.1 * lateral_mag):
        cand = np.eye(3)
        dots = np.abs(cand @ z)
        x = cand[int(np.argmin(dots))]
        x = x - np.dot(x, z) * z
        nx = np.linalg.norm(x)
    if nx < 1e-9:
        return np.eye(3)
    x = x / nx
    y = np.cross(z, x)
    ny = np.linalg.norm(y)
    if ny < 1e-9:
        return np.eye(3)
    y = y / ny
    return np.column_stack([x, y, z])


def root_rotation(spine_dir_desired: np.ndarray, hip_line_desired: np.ndarray,
                  spine_dir_rest: np.ndarray, hip_line_rest: np.ndarray) -> Rotation:
    """Rotation mapping the rest root frame onto the desired root frame.

    All inputs are 3-vectors (directions, magnitude ignored).
    """
    if not (np.all(np.isfinite(spine_dir_desired)) and np.all(np.isfinite(hip_line_desired))
            and np.all(np.isfinite(spine_dir_rest)) and np.all(np.isfinite(hip_line_rest))):
        return Rotation.identity()
    try:
        F_desired = _orthonormal_frame(spine_dir_desired, hip_line_desired)
        F_rest = _orthonormal_frame(spine_dir_rest, hip_line_rest)
        R = F_desired @ F_rest.T
        if not np.all(np.isfinite(R)):
            return Rotation.identity()
        return Rotation.from_matrix(R)
    except Exception:
        return Rotation.identity()


def spine_scale(spine_len_desired: float, spine_len_rest: float) -> float:
    """Per-frame spine-length scale, clamped to a sane range to avoid blowups
    on degenerate frames. The spine compresses/stretches; limbs don't, so this
    scale is applied only to the spine chain, not the limbs."""
    if spine_len_rest < 1e-6:
        return 1.0
    s = spine_len_desired / spine_len_rest
    # clamp: a real spine doesn't change length by more than ~30% frame to frame
    return float(np.clip(s, 0.7, 1.4))

