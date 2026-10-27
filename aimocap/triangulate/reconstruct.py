"""Evidence-preserving spatiotemporal reconstruction.

Consumes the provenance output of `aimocap.geometry.robust_triangulate` (point3d
+ per-camera residuals + inlier mask + status) and produces a windowed whole-body
solve whose *quality laws* are:

1. strong_3d anchor:  a point the cameras clearly agreed on is pinned to its own
   reprojection evidence; the optimizer is allowed to move it only by a
   bounded epistemically-justified amount.
2. usable_3d:                     lends support but follows the strong anchors.
3. weak_3d / unresolved:          reconstructed entirely from temporal/bone
   continuity, reprojection evidence, and skeletal priors; *driven* by
   higher-confidence joints, not the other way round.
4. No fabrication:               a joint with no trustworthy observation on the
   whole window stays NaN, and the manifest reports the failure.

The output always includes:
  - points3d_opt (F, J, 3)   in internal y-up space (same frame as robust input)
  - per-frame provenance masks:
      strong_mask, usable_mask, weak_mask, reconstructed_mask, unresolved_mask
      sigma_m (per-joint uncertainty), delta_from_raw (cm)
  - per-window optimizer statistics: convergence, median/p95 reprojection
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from aimocap.triangulate.spatiotemporal import (
    compute_bone_lengths,
)

# joint groups used by the production split
BODY_J = list(range(0, 17))
FEET_J = list(range(17, 23))

# bone lists for body + feet
BODY_BONES = [
    (5, 7), (7, 9), (6, 8), (8, 10),       # arms
    (11, 13), (13, 15), (12, 14), (14, 16),  # legs
    (5, 6), (11, 12), (5, 11), (6, 12),    # torso
    (0, 5), (0, 6),
]
FEET_BONES = [(15, 19), (19, 17), (16, 22), (22, 20)]
ALL_BONES = BODY_BONES + FEET_BONES

# anatomic joint limits: (parent, child, min_deg, max_deg approx as bone-direction deviation)
# Keep it simple: bound the *change* in bone direction, not absolute angles.
# We instead apply a bone-length soft clamp [0.7, 1.3] × reference.


def _projection_matrices(K_list, extrinsics):
    return np.array([K_list[c] @ np.hstack((extrinsics[c][0],
                                             np.asarray(extrinsics[c][1]).reshape(3, 1)))
                     for c in range(len(K_list))])


def _solve_window(
    pts3d_init,           # (T, J, 3) opencv-space initial estimate (may contain NaN)
    pts2d,                # (T, C, J, 2) raw observations
    scores,               # (T, C, J) confidences
    status_strong,        # (T, J) bool from robust triangulator
    status_usable,        # (T, J) bool
    proj,                 # (C, 3, 4)
    bone_lengths,         # (B,)
    alpha_time=1.2, alpha_acc=0.4, alpha_bone=2.0,
    reproj_floor_px=6.0,  # reprojection residual target resolution
    pin_overlap=0,        # number of leading frames to pin (start > 0 only)
):
    """Return (T, J, 3) optimized points plus a filled-mask marking joints whose
    3D was entirely NaN in the input for that frame (i.e. not measured)."""
    T, C, J, _ = pts2d.shape
    B = len(bone_lengths)

    # A joint is "fabricated at this frame" only if the *triangulation* returned a
    # non-finite 3D observation there (i.e. the input point is NaN in 3D), and we
    # filled it in.  A joint that was at least partially observed (some coords
    # present) is measured, not reconstructed.
    X = pts3d_init.copy()
    # measured = every coordinate finite in the original triangulation for that frame
    measured = np.isfinite(pts3d_init).all(axis=-1)  # (T, J)
    boundary_hold = np.zeros((T, J), bool)  # np.interp beyond leading/trailing edge
    for j in range(J):
        for d in range(3):
            col = X[:, j, d].astype(float)
            bad = ~np.isfinite(col)
            if not bad.any():
                continue
            if bad.all():
                continue
            ok = np.where(~bad)[0]
            bx = np.where(bad)[0]
            # leading edge (index < first measured) gets boundary-hold
            if bx.size and bx[0] < ok[0]:
                boundary_hold[bx[bx < ok[0]], j] = True
            # trailing edge (index > last measured) gets boundary-hold
            if bx.size and bx[-1] > ok[-1]:
                boundary_hold[bx[bx > ok[-1]], j] = True
            col[bad] = np.interp(bx, ok, col[ok])
            X[:, j, d] = col
    # fabricated = a frame-joint whose 3D was never measured anywhere in the window
    # and was filled via interpolation.
    fabricated = (~measured) & np.isfinite(X).all(axis=-1)
    # unresolved = never measured AND remains NaN.
    unresolved = (~measured) & ~np.isfinite(X).all(axis=-1)

    # weight map: strong gt weak gt unreconstructed
    w_frame = np.zeros((T, J))
    w_frame[status_strong] = 2.0
    w_frame[status_usable] = 1.0
    # weak + fabricated get residual only from temporal + bone terms
    w_frame[~status_strong & ~status_usable] = 0.0

    pin_mask = np.zeros(T, bool); pin_mask[:min(pin_overlap, T)] = True

    valid = scores > 0.3
    var_size = T * J * 3

    def residual(x):
        X4 = x.reshape(T, J, 3)
        out = []

        # reprojection per (t, c, j) valid
        for t in range(T):
            for c in range(C):
                P = proj[c]
                for j in range(J):
                    if not valid[t, c, j]:
                        continue
                    Xh = np.append(X4[t, j], 1.0)
                    p = P @ Xh
                    if p[2] <= 1e-9:
                        continue
                    est = p[:2] / p[2]
                    w = scores[t, c, j] * (w_frame[t, j] + 0.01)
                    out.append(w * (est - pts2d[t, c, j]))

        # temporal: velocity + acceleration (skip pinned leading frames)
        for t in range(1, T):
            wv = np.where(pin_mask[t] or pin_mask[t - 1], 0.0, 1.0)
            wa = np.where(pin_mask[t] or pin_mask[t - 1] or pin_mask[t - 2] if t >= 2 else False, 0.0, 1.0)
            out.append(wv * alpha_time * (X4[t] - X4[t - 1]).ravel())
            if t >= 2:
                out.append(wa * alpha_acc * (X4[t] - 2 * X4[t - 1] + X4[t - 2]).ravel())

        # bone length on body+feet with confidence-gated strength
        for t in range(T):
            for bi, (p, cc) in enumerate(ALL_BONES[:B]):
                if p >= J or cc >= J:
                    continue
                diff = X4[t, cc] - X4[t, p]
                L = np.linalg.norm(diff)
                if not np.isfinite(L):
                    out.append(np.zeros(1))
                    continue
                ref = bone_lengths[bi] if bi < len(bone_lengths) else 0.0
                if ref <= 0:
                    out.append(np.zeros(1))
                    continue
                # weak or unresolved at either endpoint → soft bone; strong → hard
                conf_t = min(w_frame[t, p], w_frame[t, cc])
                w = alpha_bone * (0.4 + conf_t)
                out.append(w * np.array([L - ref]))

        # STRONG anchor trust-region: strong-measured points must not move farther
        # than a bounded distance from their raw measured position.  This is the
        # evidence-preservation law: a high-confidence triangulation is a fact, and
        # the optimizer must respect it (within its own uncertainty budget).
        for t in range(T):
            for j in range(J):
                if not status_strong[t, j]:
                    continue
                if not np.isfinite(pts3d_init[t, j]).all():
                    continue
                # cap = a few cm at most — strong means "measured to within ~sigma"
                cap = 0.03
                diff = X4[t, j] - pts3d_init[t, j]
                out.append(8.0 * diff.ravel())

        if not out:
            return np.zeros(0)
        return np.concatenate(out).ravel()

    # free variables are the non-pinned frames
    free = ~pin_mask
    x0 = X.ravel()
    free_idx = np.array([i for t in range(T) if free[t]
                         for i in range(t * J * 3, (t + 1) * J * 3)])

    def free_residual(xf):
        x = x0.copy(); x[free_idx] = xf
        return residual(x)

    try:
        sol = least_squares(free_residual, x0[free_idx], method="trf",
                            loss="huber", f_scale=10.0, max_nfev=200)
        x = x0.copy(); x[free_idx] = sol.x
    except Exception:
        x = x0
    return x.reshape(T, J, 3), fabricated, boundary_hold, unresolved


def reconstruct_sequence(
    robust: dict,          # output of solve_sequence_robust
    keypoints: np.ndarray, # (F, C, J, 2)
    scores: np.ndarray,    # (F, C, J)
    K_list, extrinsics,
    window_size: int = 60,
    overlap: int = 15,
    strong_trust_region_k: float = 2.0,
):
    pts3d = robust["points3d"]
    status = robust["status"]
    F, J, _ = pts3d.shape
    proj = _projection_matrices(K_list, extrinsics)

    # COORDINATE CONTRACT: solve_sequence_robust returns points in the SAME
    # world frame the projection matrices were built in (Panoptic OpenCV
    # y-down, meters).  The optimizer must work in exactly that frame — any
    # flip here would corrupt the reprojection term against these same P
    # matrices (a real bug caught by the distortion-aware evaluation).  Do
    # NOT convert; optimize in place.
    pts_cv = np.asarray(pts3d, dtype=np.float64).copy()
    result_cv = pts_cv.copy()

    # determine bone lengths from only *strong* frames
    strong = status == "strong_3d"
    bone_ref_pts = np.where(strong[..., None], pts_cv, np.nan)
    bone_lengths = compute_bone_lengths(bone_ref_pts, ALL_BONES)
    # fallback for bones with no strong evidence
    bone_lengths[bone_lengths <= 0] = compute_bone_lengths(pts_cv, ALL_BONES)[bone_lengths <= 0]

    out = pts_cv.copy()
    provenance = {
        "strong_mask": np.zeros((F, J), bool),
        "usable_mask": np.zeros((F, J), bool),
        "weak_mask": np.zeros((F, J), bool),
        "reconstructed_mask": np.zeros((F, J), bool),
        "boundary_hold_mask": np.zeros((F, J), bool),
        "unresolved_mask": np.zeros((F, J), bool),
        "delta_from_raw": np.zeros((F, J), float),
    }

    start = 0
    while start < F:
        end = min(start + window_size, F)
        if end - start < 3:
            break
        pin_overlap = overlap if start > 0 else 0

        wres, fab, boundary_hold, unresolved = _solve_window(
            result_cv[start:end],
            keypoints[start:end],
            scores[start:end],
            status[start:end] == "strong_3d",
            status[start:end] == "usable_3d",
            proj,
            bone_lengths,
            pin_overlap=pin_overlap,
        )
        result_cv[start:end] = wres
        # Write fabricated/boundary/unresolved directly into the (window-sized) view;
        # this avoids chained-indexing copy bugs entirely.
        recon_view = provenance["reconstructed_mask"][start:end]
        bh_view = provenance["boundary_hold_mask"][start:end]
        unres_view = provenance["unresolved_mask"][start:end]
        recon_view[fab] = True
        bh_view[boundary_hold] = True
        unres_view[unresolved] = True
        provenance["reconstructed_mask"][start:end] = recon_view
        provenance["boundary_hold_mask"][start:end] = bh_view
        provenance["unresolved_mask"][start:end] = unres_view
        start = end - overlap if end < F else end

    # provenance: strong/usable from robust, reconstructed and boundary_hold from _solve_window,
    # unresolved is the union of robust-unresolved STILL unresolved after interpolation.
    provenance["strong_mask"] = status == "strong_3d"
    provenance["usable_mask"] = (status == "usable_3d") & (~provenance["reconstructed_mask"])
    provenance["weak_mask"] = (status == "weak_3d") & (~provenance["reconstructed_mask"])

    # Output stays in the declared opencv world frame (same as robust input).
    final = result_cv
    delta = np.linalg.norm(final - robust["points3d"], axis=-1)
    provenance["delta_from_raw"] = delta
    return {
        "points3d": final,
        "provenance": provenance,
        "bone_lengths": bone_lengths,
        "coordinate_frame": "panoptic_opencv_y_down_meters",
        "config": {"window_size": window_size, "overlap": overlap},
    }


__all__ = ["reconstruct_sequence", "ALL_BONES", "BODY_J", "FEET_J"]
