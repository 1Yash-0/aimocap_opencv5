"""Spatiotemporal triangulation refinement (Anipose-style).

After per-frame triangulation, refine 3D trajectories over sliding windows
with three cost terms:

    L = Σ_{c,j,t} ρ(||Π_c(X_{j,t}) - x_{c,j,t}||)      # reprojection
      + α_time * Σ_{j,t} ||X_{j,t} - X_{j,t-1}||²        # temporal smoothness
      + α_bone * Σ_{b,t} (||X_{child} - X_{parent}|| - L_b)²  # bone length

This is directly modeled on Anipose's ``CameraGroup.optim_points()``.

The per-frame triangulation is independent per joint and per frame — no
temporal coupling, no skeletal coupling.  This module adds both.

All optimization is done in **OpenCV space** (Y-down, Z-forward) so the
projection matrices P = K[R|t] apply directly.  Results are converted to
the internal Y-up space before returning.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import lil_matrix





# ── COCO-17 bone pairs (parent, child) ───────────────────────────────────
# These are the skeleton bones used for the bone-length constraint.
# Indices follow the standard COCO-17 ordering:
#   0=nose, 1=L_eye, 2=R_eye, 3=L_ear, 4=R_ear,
#   5=L_shoulder, 6=R_shoulder, 7=L_elbow, 8=R_elbow,
#   9=L_wrist, 10=R_wrist, 11=L_hip, 12=R_hip,
#   13=L_knee, 14=R_knee, 15=L_ankle, 16=R_ankle
COCO_BONES = [
    (5, 7),   # L_shoulder → L_elbow
    (7, 9),   # L_elbow → L_wrist
    (6, 8),   # R_shoulder → R_elbow
    (8, 10),  # R_elbow → R_wrist
    (11, 13), # L_hip → L_knee
    (13, 15), # L_knee → L_ankle
    (12, 14), # R_hip → R_knee
    (14, 16), # R_knee → R_ankle
    (5, 6),   # L_shoulder → R_shoulder (clavicle span)
    (11, 12), # L_hip → R_hip (pelvis span)
    (5, 11),  # L_shoulder → L_hip (torso L)
    (6, 12),  # R_shoulder → R_hip (torso R)
    (0, 5),   # nose → L_shoulder (head/neck)
    (0, 6),   # nose → R_shoulder (head/neck)
]

# ── Extended bone pairs including feet (for 23-joint COCO+foot) ─────────
# Indices 17-22: 17=L_big_toe, 18=L_small_toe, 19=L_heel,
#                 20=R_big_toe, 21=R_small_toe, 22=R_heel
COCO_FOOT_BONES = [
    (15, 19),  # L_ankle → L_heel
    (19, 17),  # L_heel → L_big_toe
    (16, 22),  # R_ankle → R_heel
    (22, 20),  # R_heel → R_big_toe
]

# Combined bones for the full 23-joint skeleton
COCO_FULL_BONES = COCO_BONES + COCO_FOOT_BONES


def compute_bone_lengths(
    points3d: np.ndarray, bones: list[tuple[int, int]] | None = None,
) -> np.ndarray:
    """Compute per-bone median lengths from a 3D trajectory.

    Args:
        points3d: ``(F, K, 3)`` array of 3D points (NaN = missing).
        bones:    list of ``(parent, child)`` index pairs.

    Returns:
        ``(len(bones),)`` array of median bone lengths (cm or m, same
        as ``points3d``).  Zero for bones with no valid measurements.
    """
    if bones is None:
        bones = COCO_BONES
    F, K, _ = points3d.shape
    lengths = np.zeros(len(bones))
    for i, (p, c) in enumerate(bones):
        if p >= K or c >= K:
            continue
        diff = points3d[:, p, :] - points3d[:, c, :]
        norms = np.linalg.norm(diff, axis=-1)
        valid = np.isfinite(norms) & (norms > 1e-6)
        if valid.any():
            lengths[i] = np.median(norms[valid])
    return lengths


def _build_residual_fn(
    pts2d: np.ndarray,           # (T, C, K, 2)
    scores: np.ndarray,          # (T, C, K)
    proj_matrices: np.ndarray,   # (C, 3, 4)
    bone_lengths: np.ndarray,    # (B,)
    bones: list[tuple[int, int]],
    alpha_time: float,
    alpha_bone: float,
) -> callable:
    """Build the residual function for ``scipy.optimize.least_squares``.

    The state vector is flattened ``(T, K, 3)`` → ``T*K*3`` elements.

    Residual blocks:
        1. Reprojection: for each (t, c, j) with valid score,
           ``sqrt(conf) * (Π_c(X_{j,t}) - x_{c,j,t})``  → 2 values
        2. Temporal: ``sqrt(alpha_time) * (X_{j,t} - X_{j,t-1})``  → 3 values
        3. Bone length: ``sqrt(alpha_b) * (||X_child - X_parent|| - L_b)``  → 1 value

    Returns a callable ``fn(x_flat) -> residuals``.
    """
    T, C, K, _ = pts2d.shape
    B = len(bones)

    # Precompute valid mask for reprojection (score > threshold)
    valid_mask = scores > 0.3  # (T, C, K)
    n_reproj = int(valid_mask.sum()) * 2
    n_temporal = (T - 1) * K * 3
    n_bone = T * B
    n_total = n_reproj + n_temporal + n_bone

    def residual_fn(x_flat: np.ndarray) -> np.ndarray:
        X = x_flat.reshape(T, K, 3)
        res = np.empty(n_total)

        # ── 1. Reprojection residuals ───────────────────────────────
        idx = 0
        for t in range(T):
            for c in range(C):
                for j in range(K):
                    if not valid_mask[t, c, j]:
                        continue
                    P = proj_matrices[c]  # (3, 4)
                    Xh = np.array([X[t, j, 0], X[t, j, 1], X[t, j, 2], 1.0])
                    proj = P @ Xh
                    if abs(proj[2]) < 1e-10:
                        continue
                    proj2d = proj[:2] / proj[2]
                    w = np.sqrt(max(scores[t, c, j], 0.01))
                    res[idx:idx+2] = w * (proj2d - pts2d[t, c, j])
                    idx += 2

        # ── 2. Temporal smoothness residuals ────────────────────────
        t_start = idx
        for t in range(1, T):
            diff = X[t] - X[t-1]  # (K, 3)
            start = t_start + (t - 1) * K * 3
            res[start:start + K*3] = np.sqrt(alpha_time) * diff.ravel()

        # ── 3. Bone-length residuals ───────────────────────────────
        b_start = t_start + (T - 1) * K * 3
        for t in range(T):
            for i, (p, c) in enumerate(bones):
                diff = X[t, c] - X[t, p]
                length = np.linalg.norm(diff)
                res[b_start + t * B + i] = np.sqrt(alpha_bone) * (length - bone_lengths[i])

        return res

    # Build sparse Jacobian sparsity pattern for efficiency
    def build_jac_sparsity():
        jac = lil_matrix((n_total, T * K * 3), dtype=np.int8)
        idx = 0
        # Reprojection: each (t,c,j) depends on X[t,j]
        for t in range(T):
            for c in range(C):
                for j in range(K):
                    if not valid_mask[t, c, j]:
                        continue
                    var_base = (t * K + j) * 3
                    jac[idx, var_base:var_base+3] = 1
                    jac[idx+1, var_base:var_base+3] = 1
                    idx += 2
        # Temporal: diff depends on X[t,j] and X[t-1,j]
        for t in range(1, T):
            for j in range(K):
                var_t = (t * K + j) * 3
                var_t1 = ((t - 1) * K + j) * 3
                for d in range(3):
                    jac[idx + (t-1)*K*3 + j*3 + d, var_t + d] = 1
                    jac[idx + (t-1)*K*3 + j*3 + d, var_t1 + d] = 1
        idx += (T - 1) * K * 3
        # Bone length: depends on X[t, parent] and X[t, child]
        for t in range(T):
            for i, (p, c) in enumerate(bones):
                var_p = (t * K + p) * 3
                var_c = (t * K + c) * 3
                jac[idx + t * B + i, var_p:var_p+3] = 1
                jac[idx + t * B + i, var_c:var_c+3] = 1
        return jac

    return residual_fn, build_jac_sparsity, n_total


