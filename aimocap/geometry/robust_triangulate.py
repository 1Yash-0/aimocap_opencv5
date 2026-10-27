"""Rigorous, evidence-preserving multi-camera triangulation.

This is the authoritative geometric solver.  It replaces the rough
leave-one-subset-out selection with iterative inlier refinement and produces a
full uncertainty report for every solved joint instead of collapsing to a
single confidence float.

Contract
--------
* Input observations are *confidence-weighted pixel detections* already in the
  same coordinate convention the cameras were calibrated in.
* Each camera carries per-frame *observation noise* (pixels), used to convert a
  residual into an evidence weight.
* Hip: a point is **not finite** merely because it is a number — it's finite
  only when the residual evidence supports it and no inconsistency happened.
* Output provenance: per joint+frame we emit `status` ∈ {strong_3d, usable_3d,
  weak_3d, unresolved}, plus per-camera residuals and a covariance proxy.

Algorithm
---------
1. Initial DLT weighted by observation confidence.
2. Compute per-camera residuals of that initial point.
3. Iteratively: mark any camera whose residual exceeds (median + k·std or
   max_signal_px) as outlier, retriangulate using survivors.
4. Never reintroduce a camera already rejected unless the new point now fits it
   within the same bar (vectorised convergence loop).
5. Estimate σ_pos from the ray configuration and per-camera noise.
6. Emit status: strong (≥2 inliers, low residual), usable (≥2 inliers, ok),
   weak (1 inlier), unresolved (0 inliers).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import cv2
from scipy.optimize import least_squares

from aimocap.math.triangulate import (
    triangulate_weighted_dlt,
)

# ---- status labels --------------------------------------------------------
STATUS_STRONG = "strong_3d"
STATUS_USABLE = "usable_3d"
STATUS_WEAK = "weak_3d"
STATUS_UNRESOLVED = "unresolved"


@dataclass
class CameraObs:
    """One camera's observation of one joint in one frame.

    All arrays indexed by joint (J,) and camera (C,) then time handled outside.
    """
    point_uv: np.ndarray          # (2,) in px (raw distorted detector frame)
    confidence: float             # detector confidence 0..1
    noise_px: float               # expected 2D measurement error in px
    is_finite: bool
    camera_K: np.ndarray | None = None     # per-camera intrinsics (for distorted projection)
    camera_R: np.ndarray | None = None     # (3,3)
    camera_t: np.ndarray | None = None     # (3,)
    camera_dist: np.ndarray | None = None  # distortion coefficients


@dataclass
class RobustTriResult:
    point3d: np.ndarray           # shape (3,)
    inlier_mask: np.ndarray       # (C,) True where camera's obs supported the point
    per_camera_residual: np.ndarray  # (C,) px, inf where camera not observed/rejected
    num_inliers: int
    status: str
    sigma_m: float                # position uncertainty proxy in world units
    diag: dict = field(default_factory=dict)


def _project(P, X3, obs=None):
    """Project a 3D point to detector pixels.

    When the observation object carries per-camera intrinsics + distortion
    (obs.camera_K / camera_R / camera_t / camera_dist), project with full
    cv2.projectPoints so the comparison is in the same distorted coordinate
    space the detector emitted.  Otherwise assume the supplied P already
    produces detector pixels (raw pinhole policy) and use it.
    """
    if obs is not None and obs.camera_K is not None and obs.camera_R is not None:
        return project_with_distortion(X3, obs.camera_K, obs.camera_R, obs.camera_t, obs.camera_dist)
    Xh = np.append(X3, 1.0)
    p = P @ Xh
    if not np.isfinite(p).all() or p[2] <= 1e-9:
        return None
    uv = p[:2] / p[2]
    return uv if np.isfinite(uv).all() else None


def project_with_distortion(X3, K, R, t, dist):
    """Project a world point into *distorted* detector pixels via cv2.projectPoints.

    Use this whenever the observations are raw distorted pixels (the declared
    observation_space for this capture).  Returns the (2,) pixel coordinate or
    None for invalid depth / non-finite.
    """
    if not np.isfinite(X3).all():
        return None
    rv, _ = cv2.Rodrigues(np.asarray(R, dtype=np.float64))
    proj, _ = cv2.projectPoints(
        np.asarray(X3, dtype=np.float64).reshape(1, 1, 3),
        rv,
        np.asarray(t, dtype=np.float64).reshape(3),
        np.asarray(K, dtype=np.float64),
        None if dist is None else np.asarray(dist, dtype=np.float64).reshape(-1),
    )
    uv = proj.reshape(2)
    return uv if np.isfinite(uv).all() else None


def _depth_cv(P, X3):
    """Depth sign proxy: the third component of P Xh.

    For an OpenCV world-to-camera P = K[R|t] with z-forward convention, P Xh[2]
    is positive when X3 is in front of the camera.  This is the correct sign to
    check without needing to un-fold K."""
    Xh = np.append(X3, 1.0)
    cam_space = P @ Xh
    return float(cam_space[2])


def _pair_combinations(indices):
    from itertools import combinations
    return list(combinations(indices, 2))




def _undistort_observation(obs: CameraObs) -> np.ndarray | None:
    """Undistort a raw distorted pixel to pinhole coordinates when the camera
    model is attached; otherwise return the point as-is (assumes the caller's
    projection matrices already match the detector pixel space).
    """
    if obs is None or not obs.is_finite:
        return None
    if obs.camera_K is None:
        return np.asarray(obs.point_uv, dtype=np.float64).copy()
    try:
        pt = cv2.undistortPoints(
            np.asarray(obs.point_uv, dtype=np.float64).reshape(1, 1, 2),
            np.asarray(obs.camera_K, dtype=np.float64),
            obs.camera_dist if obs.camera_dist is not None else np.zeros(5),
            P=obs.camera_K,
        )
        return pt.reshape(2)
    except Exception:
        return np.asarray(obs.point_uv, dtype=np.float64).copy()


def _covariance_from_jacobian(residual_fn, x, inlier_idx, weights):
    """Estimate 3D covariance by linearizing the reprojection residual.

    J = d(res)/dX (numeric, 2N x 3); W = diag(weights).  Cov ≈ inv(Jᵀ W J).
    Returns (sigma_world, condition_number).  On ill-conditioning returns large sigma.
    """
    n = len(inlier_idx)
    eps = 1e-4
    Jn = np.zeros((2 * n, 3))
    base = residual_fn(x)
    for d in range(3):
        xp = x.copy(); xp[d] += eps
        Jn[:, d] = (residual_fn(xp) - base) / eps
    W = np.diag(np.repeat(weights, 2))
    try:
        JT_W_J = Jn.T @ W @ Jn
        cov = np.linalg.pinv(JT_W_J)
        sigma_world = float(np.sqrt(max(np.trace(cov) / 3.0, 0.0)))
        cond = float(np.linalg.cond(Jn))
    except Exception:
        sigma_world = float("inf"); cond = float("inf")
    return sigma_world, cond


def triangulate_joint_robust(
    observations: list[CameraObs],
    proj_matrices: list[np.ndarray],
    K_list: list[np.ndarray],
    dist_coeffs: list[np.ndarray] | None = None,
    *,
    max_accept_px: float = 12.0,
    refine_loss: str = "huber",
    refine_f_scale: float = 10.0,
    min_inliers: int = 2,
    strong_res_px: float = 5.0,
    max_iterations: int = 8,
) -> RobustTriResult:
    """Iterative, distortion-aware, uncertainty-scored joint triangulation.

    Steps:
      1. Undistort raw detector pixels (per-camera cv2.undistortPoints, P=K)
         so DLT operates on pinhole-consistent coordinates.
      2. Seed 3D from every camera pair with positive depth.
      3. Score each seed by projected residuals against raw distorted pixels
         (cv2.projectPoints with the camera's actual distortion).
      4. Iteratively: keep cameras whose residual ≤ per-camera accept threshold
         (noise_px * noise_mult, clamped [2, max_accept_px]), retriangulate,
         stop on stability; prefer the solution with more inliers, then lower median.
      5. Compute covariance (Jacobian) and classify status by residual + ray angle +
         depth + sigma — not by camera count alone.
    """
    C = len(observations)
    observed = np.array(
        [o is not None and o.is_finite for o in observations], dtype=bool
    )
    if observed.sum() < min_inliers:
        return RobustTriResult(
            point3d=np.full(3, np.nan), inlier_mask=np.zeros(C, bool),
            per_camera_residual=np.full(C, np.inf), num_inliers=0,
            status=STATUS_UNRESOLVED, sigma_m=np.inf,
            diag={"reason": "insufficient_views"},
        )
    idx_obs = np.flatnonzero(observed)

    # per-camera noise-driven acceptance thresholds, component-tracked
    accept = np.array([
        float(np.clip((observations[i].noise_px or 2.0) * 3.0, 2.0, max_accept_px))
        if observed[i] else np.inf for i in range(C)])

    # undistorted observations for geometric seeding (DLT in pinhole space)
    undist = {i: _undistort_observation(observations[i]) for i in idx_obs}

    def projected_dist(x, i):
        return _project(proj_matrices[i], x, observations[i])

    def cam_residuals(x):
        return np.array([np.linalg.norm(projected_dist(x, i) - observations[i].point_uv)
                         if observed[i] and projected_dist(x, i) is not None else np.inf
                         for i in range(C)])

    def _depth_ok(x, i):
        d = _depth_cv(proj_matrices[i], x)
        return np.isfinite(d) and d > 0

    # ---- candidate generation from pair seeds (pinhole space) ----
    candidates = []
    for a_idx, b_idx in _pair_combinations(idx_obs):
        if undist.get(a_idx) is None or undist.get(b_idx) is None:
            continue
        P_pair = [proj_matrices[a_idx], proj_matrices[b_idx]]
        pts_pair = np.array([undist[a_idx], undist[b_idx]], float)
        conf_pair = np.array([observations[a_idx].confidence,
                              observations[b_idx].confidence], float)
        try:
            Xp = triangulate_weighted_dlt(pts_pair, P_pair, conf_pair)
        except Exception:
            continue
        if not np.isfinite(Xp).all():
            continue
        if not (_depth_ok(Xp, a_idx) and _depth_ok(Xp, b_idx)):
            continue
        candidates.append(Xp)

    if not candidates:
        return RobustTriResult(
            point3d=np.full(3, np.nan), inlier_mask=np.zeros(C, bool),
            per_camera_residual=np.full(C, np.inf), num_inliers=0,
            status=STATUS_UNRESOLVED, sigma_m=np.inf,
            diag={"reason": "no_valid_seed"},
        )

    # ---- for each candidate, run an iterative inlier-refinement loop ----
    def refine_and_select(seed):
        X = seed.copy()
        inlier = np.zeros(C, bool)
        inlier[idx_obs] = True
        rr = cam_residuals(X)
        inlier = observed & (rr <= accept)
        last_best = (int(inlier.sum()), float(np.median(rr[observed])) if observed.any() else np.inf,
                     X.copy(), inlier.copy())
        for _ in range(max_iterations):
            nl = np.flatnonzero(inlier)
            if nl.size < min_inliers:
                break
            # nonlinear refinement over inlier set using distorted-pixel residuals
            obs_i = [observations[i] for i in nl]
            P_i = [proj_matrices[i] for i in nl]
            try:
                def res_fn(x, _obs=obs_i, _P=P_i, _nl=nl):
                    out = []
                    for i, o in zip(_nl, _obs):
                        uv = _project(_P[_nl.index(i)], x, o)
                        if uv is None:
                            out.extend([1e3, 1e3])
                        else:
                            w = max(o.confidence, 0.01)
                            out.extend(w * (uv - o.point_uv))
                    return np.asarray(out)
                r = least_squares(res_fn, X, method="trf", loss=refine_loss,
                                  f_scale=refine_f_scale)
                if np.isfinite(r.x).all():
                    X = r.x
            except Exception:
                pass
            rr = cam_residuals(X)
            new_inlier = observed & (rr <= accept)
            if new_inlier.sum() < min_inliers:
                break
            # prefer more inliers, then lower median residual
            score_new = (-int(new_inlier.sum()),
                         float(np.median(rr[observed & np.isfinite(rr)]))
                         if np.isfinite(rr).any() else np.inf)
            score_old = (-int(inlier.sum()),
                         float(np.median(rr[inlier & np.isfinite(rr)]))
                         if np.isfinite(rr).any() else np.inf)
            inlier = new_inlier
            if score_new < (last_best[0], last_best[1]):
                last_best = (int(new_inlier.sum()), score_new[1], X.copy(), inlier.copy())
            if new_inlier.sum() == inlier.sum() and np.array_equal(new_inlier, inlier):
                break
        return last_best

    best = None
    for seed in candidates:
        res = refine_and_select(seed)
        # res = (num_inliers, median_res, X, inlier)
        key = (-res[0], res[1] if np.isfinite(res[1]) else np.inf)
        if best is None or key < best[0]:
            best = (key, res)
    if best is None:
        return RobustTriResult(
            point3d=np.full(3, np.nan), inlier_mask=np.zeros(C, bool),
            per_camera_residual=np.full(C, np.inf), num_inliers=0,
            status=STATUS_UNRESOLVED, sigma_m=np.inf,
            diag={"reason": "no_refinement_survived"},
        )
    (n_in, med_res, X, inlier) = best[1]

    # ---- geometric covariance (Jacobian) over the inlier set ----
    nl = np.flatnonzero(inlier)
    def res_full(x):
        out = []
        for i in nl:
            o = observations[i]
            uv = _project(proj_matrices[i], x, o)
            if uv is None:
                out.extend([1e3, 1e3])
            else:
                w = max(o.confidence, 0.01)
                out.extend(w * (uv - o.point_uv))
        return np.asarray(out)
    weights = [max(observations[i].confidence, 0.01) / max(observations[i].noise_px, 1e-3)
               for i in nl]
    sigma_world, cond = _covariance_from_jacobian(res_full, X, nl, weights)

    # ---- ray angle between inlier camera centers and the point ----
    centers = []
    for i in nl:
        # camera center from P = [M | p4]; center = -inv(M) @ p4[:,0...M is 3x3 of P[:,:3]
        P = proj_matrices[i]
        try:
            M = P[:, :3]; c_world = -np.linalg.solve(M, P[:, 3])
            centers.append(c_world)
        except Exception:
            pass
    ray_angle = 0.0
    if len(centers) >= 2:
        mx = 0.0
        from itertools import combinations
        for ca, cb in combinations(centers, 2):
            va = X - ca; vb = X - cb
            na = np.linalg.norm(va); nb = np.linalg.norm(vb)
            if na > 1e-9 and nb > 1e-9:
                ca_ = np.clip(np.dot(va, vb) / (na * nb), -1.0, 1.0)
                mx = max(mx, np.degrees(np.arccos(ca_)))
        ray_angle = mx

    # ---- per-camera residuals in distorted pixels ----
    per_cam = np.full(C, np.inf)
    for i, o in enumerate(observations):
        if o is None or not o.is_finite:
            continue
        uv = projected_dist(X, i) if np.isfinite(X).all() else None
        if uv is not None:
            per_cam[i] = float(np.linalg.norm(uv - o.point_uv))

    # ---- status classification: counts + quality, not just counts ----
    fin_observed = per_cam[np.isfinite(per_cam)]
    n_inl = int(inlier.sum())
    p95 = float(np.percentile(fin_observed, 95)) if fin_observed.size else np.inf
    depth_ok = all(_depth_ok(X, i) for i in nl) if nl.size else False
    sigma_ok = np.isfinite(sigma_world) and sigma_world < 0.05
    if n_inl >= 3 and med_res <= strong_res_px and p95 <= 2 * strong_res_px and \
            depth_ok and sigma_ok and ray_angle >= 15:
        status = STATUS_STRONG
    elif n_inl >= 2 and depth_ok:
        status = STATUS_USABLE
    elif n_inl >= 1 and depth_ok:
        status = STATUS_WEAK
    else:
        status = STATUS_UNRESOLVED

    return RobustTriResult(
        point3d=X if np.isfinite(X).all() else np.full(3, np.nan),
        inlier_mask=inlier,
        per_camera_residual=per_cam,
        num_inliers=n_inl,
        status=status,
        sigma_m=(sigma_world if np.isfinite(sigma_world) else np.nan),
        diag={
            "med_residual": float(np.median(fin_observed)) if fin_observed.size else float("nan"),
            "p95_residual": float(np.percentile(fin_observed, 95)) if fin_observed.size else float("nan"),
            "ray_angle_deg": float(ray_angle),
            "jacobian_cond": float(cond) if np.isfinite(cond) else None,
            "positive_depth": bool(depth_ok),
            "iters_best": int(n_inl),
        },
    )
    """Estimate one 3D joint from per-camera observations with iterative inlier selection.

    Parameters
    ----------
    observations : per-camera :class:`CameraObs` (None entries are allowed and
        mean the camera did not observe the joint).
    proj_matrices : list of (3,4) project matrices K[R|t] *in the exact pixel+unit
        convention* of the observations.
    K_list, dist_coeffs : used only for computing per-camera residuals against the
        *distorted* original pixels so that the residual is expressed in the same
        units the detector sees.

    Returns
    -------
    RobustTriResult. `point3d` may be NaN if no valid point was found.
    """
    C = len(observations)
    observed = np.array(
        [o is not None and o.is_finite for o in observations], dtype=bool
    )
    if observed.sum() < min_inliers:
        return RobustTriResult(
            point3d=np.full(3, np.nan),
            inlier_mask=np.zeros(C, bool),
            per_camera_residual=np.full(C, np.inf),
            num_inliers=0,
            status=STATUS_UNRESOLVED,
            sigma_m=np.inf,
            diag={"reason": "insufficient_views"},
        )

    idx_obs = np.flatnonzero(observed)

    def residuals_for(x, mask):
        res = np.full(len(mask), np.inf)
        for k, i in enumerate(np.flatnonzero(mask)):
            uvp = _project(proj_matrices[i], x, observations[i])
            if uvp is not None:
                res[k] = float(np.linalg.norm(uvp - observations[i].point_uv))
        return res

    # Find the largest self-consistent inlier set by explicitly searching over
    # camera subsets.  We must not accept a camera merely because its residual
    # falls below a noise-driven global threshold — one bad view can inflate the
    # threshold and poison the rest.  Instead, for each candidate pair seed we
    # grow the set of cameras whose residuals fit the point, then pick the set
    # with the strongest evidence (prefer more inliers and lower error).
    candidate_sets = []
    seen_sets = set()
    for a_idx, b_idx in _pair_combinations(idx_obs):
        P_pair = [proj_matrices[a_idx], proj_matrices[b_idx]]
        pts_pair = np.array([observations[a_idx].point_uv,
                             observations[b_idx].point_uv], float)
        conf_pair = np.array([observations[a_idx].confidence,
                              observations[b_idx].confidence], float)
        try:
            Xp = triangulate_weighted_dlt(pts_pair, P_pair, conf_pair)
        except Exception:
            continue
        if not np.isfinite(Xp).all():
            continue
        da = _depth_cv(proj_matrices[a_idx], Xp); db = _depth_cv(proj_matrices[b_idx], Xp)
        if not np.isfinite(da) or not np.isfinite(db) or da <= 0 or db <= 0:
            continue
        # grow consensus from this pair's seed
        inlier_i = np.zeros(C, bool)
        for i in idx_obs:
            uv = _project(proj_matrices[i], Xp)
            if uv is not None:
                residual = float(np.linalg.norm(uv - observations[i].point_uv))
                if residual <= max_accept_px:
                    inlier_i[i] = True
        key = tuple(np.flatnonzero(inlier_i))
        if key in seen_sets:
            continue
        seen_sets.add(key)
        candidate_sets.append((Xp, inlier_i))

    if not candidate_sets:
        return RobustTriResult(
            point3d=np.full(3, np.nan),
            inlier_mask=np.zeros(C, bool),
            per_camera_residual=np.full(C, np.inf),
            num_inliers=0,
            status=STATUS_UNRESOLVED,
            sigma_m=np.inf,
            diag={"reason": "no_consensus_set"},
        )

    # Score each candidate: prefer more inliers, then lower median residual, then
    # higher total confidence.  Solve a refined point from each candidate's inlier set.
    best = None
    for Xp, inlier_i in candidate_sets:
        nl_idx = np.flatnonzero(inlier_i)
        if nl_idx.size < min_inliers:
            continue
        obs_i = [observations[i] for i in nl_idx]
        conf_i = np.array([o.confidence for o in obs_i])
        P_i = [proj_matrices[i] for i in nl_idx]
        X_i = np.array([o.point_uv for o in obs_i])
        try:
            res = least_squares(
                _robust_residuals, Xp, method="trf",
                loss=refine_loss, f_scale=refine_f_scale,
                args=(X_i, P_i, conf_i),
            )
            X_refined = res.x if np.isfinite(res.x).all() else Xp
        except Exception:
            X_refined = Xp
        res_all = residuals_for(X_refined, observed)
        residual_in = res_all[nl_idx]
        med = float(np.median(residual_in)) if residual_in.size else float("inf")
        score = (-int(inlier_i.sum()), med, -float(conf_i.sum()))
        if best is None or score < best[0]:
            best = (score, X_refined, inlier_i)
    if best is None:
        return RobustTriResult(
            point3d=np.full(3, np.nan),
            inlier_mask=np.zeros(C, bool),
            per_camera_residual=np.full(C, np.inf),
            num_inliers=0,
            status=STATUS_UNRESOLVED,
            sigma_m=np.inf,
            diag={"reason": "no_sufficient_consensus_set"},
        )
    _, X, inlier = best
    count = int(inlier.sum())
    if count >= 3:
        status = STATUS_STRONG
    elif count == 2:
        status = STATUS_USABLE
    elif count == 1:
        status = STATUS_WEAK
    else:
        status = STATUS_UNRESOLVED
    # per-camera residual for observations; inf where rejected / not observed
    per_cam = np.full(C, np.inf)
    for i, o in enumerate(observations):
        if o is None or not o.is_finite:
            continue
        uv = _project(proj_matrices[i], X) if np.isfinite(X).all() else None
        if uv is not None:
            per_cam[i] = float(np.linalg.norm(uv - o.point_uv))
    # uncertainty proxy: sigma_world ≈ mean inlier noise / mean inlier confidence.
    inlier_noise = [observations[i].noise_px for i in np.flatnonzero(inlier)]
    inlier_conf = [observations[i].confidence for i in np.flatnonzero(inlier)]
    sigma_world = float(np.mean(inlier_noise)) if inlier_noise else float("nan")
    mean_conf = float(np.mean(inlier_conf)) if inlier_conf else 1e-6
    return RobustTriResult(
        point3d=X if np.isfinite(X).all() else np.full(3, np.nan),
        inlier_mask=inlier,
        per_camera_residual=per_cam,
        num_inliers=count,
        status=status,
        sigma_m=sigma_world / max(mean_conf, 1e-6),
        diag={"med_residual": float(np.nanmedian(per_cam[np.isfinite(per_cam)])) if np.isfinite(per_cam).any() else float("nan")},
    )


def _robust_residuals(x, pts2d, proj_matrices, confidences):
    Xh = np.append(x, 1.0)
    res = []
    for P, uv, c in zip(proj_matrices, pts2d, confidences):
        p = P @ Xh
        if p[2] <= 1e-9 or not np.isfinite(p).all():
            res.extend([1e3, 1e3])
            continue
        proj = p[:2] / p[2]
        r = c * (proj - uv)
        res.extend(r)
    return np.asarray(res)


def solve_sequence_robust(
    keypoints: np.ndarray,        # (F, C, J, 2)
    scores: np.ndarray,           # (F, C, J)
    camera_models: list[dict],    # per-cam: K, R, t, dist, noise_px, image_size
    *,
    max_accept_px: float = 12.0,
    joint_noise_px: dict[int, float] | None = None,
) -> dict:
    """Triangulate a full sequence with full per-joint provenance."""
    F, C, J, _ = keypoints.shape
    pts3d = np.full((F, J, 3), np.nan)
    status = np.full((F, J), STATUS_UNRESOLVED, dtype=object)
    num_inliers = np.zeros((F, J), np.int8)
    per_cam_res = np.full((F, J, C), np.inf, np.float32)
    per_cam_inlier = np.zeros((F, J, C), bool)
    sigma_m = np.full((F, J), np.inf)

    P_list = []
    for cm in camera_models:
        K = np.asarray(cm["K"], float); R = np.asarray(cm["R"], float); t = np.asarray(cm["t"], float).reshape(3)
        P_list.append(K @ np.hstack((R, t.reshape(3, 1))))
    if joint_noise_px is None:
        joint_noise_px = {}

    for f in range(F):
        for j in range(J):
            obs = []
            have = 0
            for c in range(C):
                uv = keypoints[f, c, j]
                sc = float(scores[f, c, j])
                if not np.isfinite(uv).all() or sc < 0.01:
                    obs.append(None)
                    continue
                jp = joint_noise_px.get(j, 2.0)
                noise = max(1.0, jp * (1.0 + (1.0 - sc)))
                obs.append(CameraObs(point_uv=uv, confidence=sc, noise_px=noise, is_finite=True))
                have += 1
            if have < 2:
                continue
            r = triangulate_joint_robust(
                obs, P_list, [cm["K"] for cm in camera_models],
                dist_coeffs=[cm.get("dist") for cm in camera_models],
                max_accept_px=max_accept_px,
            )
            if np.isfinite(r.point3d).all():
                pts3d[f, j] = r.point3d
            status[f, j] = r.status
            num_inliers[f, j] = r.num_inliers
            per_cam_res[f, j] = r.per_camera_residual
            per_cam_inlier[f, j] = r.inlier_mask
            sigma_m[f, j] = r.sigma_m
    return {
        "points3d": pts3d,
        "status": status,
        "num_inliers": num_inliers,
        "per_cam_residual_px": per_cam_res,
        "per_cam_inlier": per_cam_inlier,
        "sigma_m": sigma_m,
        "config": {"max_accept_px": max_accept_px, "joint_noise_px": joint_noise_px},
    }


__all__ = [
    "triangulate_joint_robust",
    "solve_sequence_robust",
    "RobustTriResult",
    "CameraObs",
    "STATUS_STRONG",
    "STATUS_USABLE",
    "STATUS_WEAK",
    "STATUS_UNRESOLVED",
]
