"""Triangulation orchestration engine."""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import combinations
from typing import NamedTuple

import cv2
import numpy as np

from aimocap.math.coords import opencv_to_internal
from aimocap.math.triangulate import triangulate_weighted_dlt
from aimocap.triangulate import confidence3d
from aimocap.triangulate.config import GeometryConfig
from aimocap.triangulate.temporal_filter import filter_skeleton3d_rts


def _max_pairwise_angle(cams, target_col: np.ndarray, C_list: list[np.ndarray]) -> float:
    """Maximum angle (degrees) subtended at ``target_col`` by any pair of camera centers."""
    max_ang = 0.0
    for c1, c2 in combinations(cams, 2):
        v1 = target_col - C_list[c1]
        v2 = target_col - C_list[c2]
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 > 0 and n2 > 0:
            cos_t = np.clip(np.dot(v1.flatten(), v2.flatten()) / (n1 * n2), -1.0, 1.0)
            ang = np.arccos(cos_t) * 180 / np.pi
            if ang > max_ang:
                max_ang = ang
    return max_ang


@dataclass(frozen=True)
class TriangulationDiagnostics:
    """Per-frame/per-joint observability emitted by triangulation.

    All arrays are in the original OpenCV camera projection space, except
    ``points3d`` which is returned in the project's Y-up internal space to
    preserve the historical downstream contract.
    """

    points3d: np.ndarray
    candidate_points3d: np.ndarray
    confidence: np.ndarray
    candidate_inlier_mask: np.ndarray
    inlier_mask: np.ndarray
    # Cameras that corroborate the point for diagnostics/gating.  Legacy trust
    # is binary, so this equals ``inlier_mask``.  Soft trust is continuous and
    # this mask is ``camera_trust >= GeometryConfig.soft_support_min_trust``;
    # unlike ``inlier_mask`` (trust > 0.5) it stays independent of the stricter
    # 0.5 mask that the soft solver uses for its reported switch state.
    support_mask: np.ndarray
    camera_trust: np.ndarray
    reprojection_error_px: np.ndarray
    candidate_num_inliers: np.ndarray
    num_inliers: np.ndarray
    low_quality_mask: np.ndarray
    ray_angle_deg: np.ndarray
    observed_camera_count: np.ndarray
    positive_depth_mask: np.ndarray
    observation_provenance: np.ndarray
    camera_rejection_reason: np.ndarray
    # Accepted left/right swap flags, indexed (frame, camera, group) with the
    # group order of ``LR_SWAP_GROUPS`` (arm, leg, hands).  All False when
    # ``GeometryConfig.lr_swap_fix`` is off.
    lr_swap_mask: np.ndarray
    # P3.6 per-point 3D confidence and its raw features.  ``None`` unless
    # ``GeometryConfig.confidence3d`` is enabled on the soft policy.
    confidence3d: np.ndarray | None = None
    confidence3d_features: np.ndarray | None = None
    # Magnitude of the P3.5 smoother correction in metres, ``(frames, joints)``.
    # Populated only when confidence3d needs the smoother innovation feature.
    smoother_innovation_m: np.ndarray | None = None


# Left/right label groups tested by ``lr_swap_fix``: (group name, joint pairs).
LR_SWAP_GROUPS: tuple[tuple[str, tuple[tuple[int, int], ...]], ...] = (
    ("arm", ((5, 6), (7, 8), (9, 10))),
    ("leg", ((11, 12), (13, 14), (15, 16), (17, 20), (18, 21), (19, 22))),
    ("hands", tuple((91 + offset, 112 + offset) for offset in range(21))),
)
LR_SWAP_GROUP_NAMES: tuple[str, ...] = tuple(name for name, _ in LR_SWAP_GROUPS)


def _project_point(
    point3d_cv: np.ndarray,
    P: np.ndarray,
    *,
    camera_model: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None,
    min_depth: float = 0.0,
) -> np.ndarray | None:
    """Project one OpenCV-space 3D point. Return None for invalid depth."""
    if camera_model is not None:
        K, R, t, dist = camera_model
        camera_point = np.asarray(R, dtype=np.float64) @ np.asarray(point3d_cv, dtype=np.float64).reshape(3)
        camera_point += np.asarray(t, dtype=np.float64).reshape(3)
        invalid_depth = (
            (min_depth > 0.0 and camera_point[2] <= min_depth)
            or (min_depth <= 0.0 and abs(camera_point[2]) < 1e-12)
        )
        if not np.isfinite(camera_point).all() or invalid_depth:
            return None
        rvec, _ = cv2.Rodrigues(np.asarray(R, dtype=np.float64))
        projected, _ = cv2.projectPoints(
            np.asarray(point3d_cv, dtype=np.float64).reshape(1, 3),
            rvec,
            np.asarray(t, dtype=np.float64).reshape(3, 1),
            np.asarray(K, dtype=np.float64),
            np.asarray(dist, dtype=np.float64),
        )
        result = projected.reshape(2)
        return result if np.isfinite(result).all() else None
    X = np.append(point3d_cv, 1.0)
    p = P @ X
    if not np.isfinite(p).all():
        return None
    # The historical projection-matrix-only API allowed either OpenCV depth
    # sign, because some callers supplied a world frame with the opposite
    # viewing convention.  Enforce positive depth only when the new geometry
    # contract asks for a physical minimum depth.
    if min_depth > 0.0 and float(p[2]) <= min_depth:
        return None
    if min_depth <= 0.0 and abs(float(p[2])) < 1e-12:
        return None
    return p[:2] / p[2]


def _reprojection_errors(
    point3d_cv: np.ndarray,
    pts2d: np.ndarray,
    P_list: list[np.ndarray],
    *,
    camera_models: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] | None = None,
    min_depth: float = 0.0,
) -> np.ndarray:
    err = np.full((len(P_list),), np.inf, dtype=np.float64)
    for i, P in enumerate(P_list):
        model = None if camera_models is None else camera_models[i]
        proj = _project_point(point3d_cv, P, camera_model=model, min_depth=min_depth)
        if proj is not None:
            err[i] = float(np.linalg.norm(proj - pts2d[i]))
    return err


def _robust_influence(err_px: np.ndarray, scale_px: float, loss: str) -> np.ndarray:
    """IRLS influence ``rho'(r)/r`` for one robust kernel (unit-free, in [0, 1])."""
    scale = max(float(scale_px), 1e-6)
    finite = np.isfinite(err_px)
    safe = np.where(finite, np.maximum(np.asarray(err_px, dtype=np.float64), 0.0), np.inf)
    if loss == "cauchy":
        return np.where(finite, 1.0 / (1.0 + (safe / scale) ** 2), 0.0)
    return np.where(finite, np.where(safe <= scale, 1.0, scale / np.maximum(safe, 1e-9)), 0.0)


class _LRObservation(NamedTuple):
    """One joint/camera observation used by the left/right swap test."""

    point: np.ndarray  # undistorted 2D point used by the solve
    observed: np.ndarray  # original (distorted) pixel residual target
    weight: float
    model: tuple | None


def _huber_cost(residual_px: np.ndarray, scale_px: float) -> np.ndarray:
    """Huber loss value (not the IRLS influence) in weighted-pixel units."""
    scale = max(float(scale_px), 1e-6)
    residual = np.abs(np.asarray(residual_px, dtype=np.float64))
    return np.where(
        residual <= scale,
        0.5 * residual * residual,
        scale * (residual - 0.5 * scale),
    )


def _gather_lr_observations(
    keypoints: np.ndarray,
    scores: np.ndarray,
    cfg: GeometryConfig,
    frame: int,
    camera_models: list | None,
    image_sizes,
) -> dict[int, dict[int, _LRObservation]]:
    """Per joint/camera observations for the swap test, mirroring engine gating."""
    num_cameras = keypoints.shape[1]
    observations: dict[int, dict[int, _LRObservation]] = {}
    for _, pairs in LR_SWAP_GROUPS:
        for left, right in pairs:
            for joint in (left, right):
                if joint in observations or joint >= keypoints.shape[2]:
                    continue
                part = cfg.confidence_for_joint(joint)
                records: dict[int, _LRObservation] = {}
                for camera in range(num_cameras):
                    score = float(scores[frame, camera, joint])
                    if not np.isfinite(score) or score < part.min_conf:
                        continue
                    xy = keypoints[frame, camera, joint]
                    if not np.isfinite(xy).all():
                        continue
                    if image_sizes is not None:
                        width, height = image_sizes[camera]
                        if not (0.0 <= xy[0] < float(width) and 0.0 <= xy[1] < float(height)):
                            continue
                    observed = np.asarray(xy, dtype=np.float64)
                    point = observed
                    model = None if camera_models is None else camera_models[camera]
                    if model is not None:
                        K, _, _, dist = model
                        point = cv2.undistortPoints(
                            observed.reshape(1, 1, 2), K, dist, P=K
                        ).reshape(2)
                    records[camera] = _LRObservation(
                        point=point,
                        observed=observed,
                        weight=part.weight(score),
                        model=model,
                    )
                observations[joint] = records
    return observations


def _lr_group_cost(
    observations: dict[int, dict[int, _LRObservation]],
    pairs: tuple[tuple[int, int], ...],
    swap_flags: np.ndarray,
    P_list: list[np.ndarray],
    cfg: GeometryConfig,
) -> float:
    """Robust reprojection cost of one left/right group under ``swap_flags``.

    For every left/right joint pair both sides are re-triangulated from the
    labelled observations of all cameras (a camera with a flip uses the
    opposite joint's point) and the weighted Huber residual of every used
    observation is summed.  An infeasible labelling (fewer than two cameras on
    either side, invalid depth or a non-finite solve) returns ``inf``.
    """
    total = 0.0
    for left, right in pairs:
        for joint, other in ((left, right), (right, left)):
            cams: list[int] = []
            points: list[np.ndarray] = []
            observed: list[np.ndarray] = []
            weights: list[float] = []
            models: list = []
            projections: list[np.ndarray] = []
            for camera in sorted(set(observations[joint]) | set(observations[other])):
                source = other if bool(swap_flags[camera]) else joint
                item = observations[source].get(camera)
                if item is None:
                    continue
                cams.append(camera)
                points.append(item.point)
                observed.append(item.observed)
                weights.append(item.weight)
                models.append(item.model)
                projections.append(P_list[camera])
            if len(cams) < 2:
                return float("inf")
            try:
                point3d = triangulate_weighted_dlt(
                    np.asarray(points, dtype=np.float64),
                    projections,
                    np.clip(np.asarray(weights, dtype=np.float64), 1e-6, 1.0),
                )
            except Exception:
                return float("inf")
            if not np.isfinite(point3d).all():
                return float("inf")
            errors = _reprojection_errors(
                point3d,
                np.asarray(observed, dtype=np.float64),
                projections,
                camera_models=models,
                min_depth=cfg.min_depth_m,
            )
            if not np.isfinite(errors).all():
                return float("inf")
            total += float(
                np.sum(
                    np.asarray(weights, dtype=np.float64)
                    * _huber_cost(errors, cfg.soft_scale_px)
                )
            )
    return total


def _detect_lr_swap_flags(
    observations: dict[int, dict[int, _LRObservation]],
    P_list: list[np.ndarray],
    cfg: GeometryConfig,
    num_cameras: int,
) -> np.ndarray:
    """Per-camera swap test for every left/right group of one frame.

    Flips are accepted strongest-improvement first: with three cameras a
    *majority* relabelling can also explain the observations (the mirrored
    skeleton), so the camera whose own flip leaves the lowest group cost is
    chosen before a smaller improvement from a camera that merely joins the
    wrong majority.
    """
    raw = np.zeros((num_cameras, len(LR_SWAP_GROUPS)), dtype=bool)
    for group_index, (_, pairs) in enumerate(LR_SWAP_GROUPS):
        if any(
            left not in observations or right not in observations
            for left, right in pairs
        ):
            continue
        flags = np.zeros(num_cameras, dtype=bool)
        base_cost = _lr_group_cost(observations, pairs, flags, P_list, cfg)
        if not np.isfinite(base_cost):
            continue
        remaining = [
            camera
            for camera in range(num_cameras)
            if any(
                camera in observations[left] or camera in observations[right]
                for left, right in pairs
            )
        ]
        while remaining:
            best_camera: int | None = None
            best_cost = base_cost
            for camera in remaining:
                trial = flags.copy()
                trial[camera] = not trial[camera]
                cost = _lr_group_cost(observations, pairs, trial, P_list, cfg)
                if np.isfinite(cost) and cost < best_cost:
                    best_camera = camera
                    best_cost = cost
            if (
                best_camera is None
                or best_cost * cfg.lr_swap_ratio > base_cost
            ):
                break
            flags[best_camera] = True
            base_cost = best_cost
            remaining.remove(best_camera)
        raw[:, group_index] = flags
    return raw


def repair_lr_swaps(
    keypoints: np.ndarray,
    scores: np.ndarray,
    K_list: list[np.ndarray],
    extrinsics: list[tuple[np.ndarray, np.ndarray]],
    cfg: GeometryConfig,
    dist_coeffs: list[np.ndarray] | None = None,
    image_sizes=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Relabel left/right groups that a camera consistently swaps.

    Returns copies of ``keypoints`` and ``scores`` with the accepted swaps
    applied per frame/camera, plus the accepted ``(frame, camera, group)``
    boolean mask.  A raw detection is only applied when it has been present for
    two consecutive frames or continues the previous frame's accepted decision.
    """
    num_frames, num_cameras, num_kpts, _ = keypoints.shape
    P_list = [K_list[c] @ np.hstack(extrinsics[c]) for c in range(num_cameras)]
    camera_models = None
    if dist_coeffs is not None:
        camera_models = [
            (
                np.asarray(K_list[c], dtype=np.float64),
                np.asarray(extrinsics[c][0], dtype=np.float64),
                np.asarray(extrinsics[c][1], dtype=np.float64),
                np.asarray(dist_coeffs[c], dtype=np.float64),
            )
            for c in range(num_cameras)
        ]
    repaired_keypoints = np.array(keypoints, dtype=np.float64, copy=True)
    repaired_scores = np.array(scores, dtype=np.float64, copy=True)
    swap_mask = np.zeros(
        (num_frames, num_cameras, len(LR_SWAP_GROUPS)), dtype=bool
    )
    prev_raw = np.zeros((num_cameras, len(LR_SWAP_GROUPS)), dtype=bool)
    prev_accepted = np.zeros_like(prev_raw)
    for frame in range(num_frames):
        observations = _gather_lr_observations(
            repaired_keypoints, repaired_scores, cfg, frame, camera_models, image_sizes
        )
        raw = _detect_lr_swap_flags(observations, P_list, cfg, num_cameras)
        accepted = raw & (prev_raw | prev_accepted)
        for group_index, (_, pairs) in enumerate(LR_SWAP_GROUPS):
            for camera in np.flatnonzero(accepted[:, group_index]):
                for left, right in pairs:
                    if left >= num_kpts or right >= num_kpts:
                        continue
                    repaired_keypoints[frame, camera, left], repaired_keypoints[
                        frame, camera, right
                    ] = (
                        repaired_keypoints[frame, camera, right].copy(),
                        repaired_keypoints[frame, camera, left].copy(),
                    )
                    repaired_scores[frame, camera, left], repaired_scores[
                        frame, camera, right
                    ] = (
                        repaired_scores[frame, camera, right],
                        repaired_scores[frame, camera, left],
                    )
        swap_mask[frame] = accepted
        prev_raw = raw
        prev_accepted = accepted
    return repaired_keypoints, repaired_scores, swap_mask


class _SoftObservation:
    """One joint's valid observations for the soft camera-trust solve."""

    __slots__ = ("cams", "pts2d", "residual_pts2d", "confidences", "weights", "models")

    def __init__(self, cams, pts2d, residual_pts2d, confidences, weights, models):
        self.cams = np.asarray(cams, dtype=np.int32)
        self.pts2d = np.asarray(pts2d, dtype=np.float64)
        self.residual_pts2d = np.asarray(residual_pts2d, dtype=np.float64)
        self.confidences = np.asarray(confidences, dtype=np.float64)
        self.weights = np.asarray(weights, dtype=np.float64)
        self.models = None if models is None else list(models)


def _soft_irls_refine(x0, observation, P_list, min_depth, cfg, base_weights=None):
    """Refine ``x0`` with ``cfg.soft_iterations`` IRLS steps (base weights + kernel)."""
    sub_P = [P_list[int(camera)] for camera in observation.cams]
    models = (
        None
        if observation.models is None
        else list(observation.models)
    )
    weights = (
        observation.weights
        if base_weights is None
        else np.asarray(base_weights, dtype=np.float64)
    )
    x = np.asarray(x0, dtype=np.float64)
    err = _reprojection_errors(
        x,
        observation.residual_pts2d,
        sub_P,
        camera_models=models,
        min_depth=min_depth,
    )
    for _ in range(int(cfg.soft_iterations)):
        influence = _robust_influence(err, cfg.soft_scale_px, cfg.robust_loss)
        new_weights = np.clip(weights * influence, 1e-6, 1.0)
        try:
            candidate = triangulate_weighted_dlt(observation.pts2d, sub_P, new_weights)
        except Exception:
            break
        if not np.isfinite(candidate).all():
            break
        candidate_err = _reprojection_errors(
            candidate,
            observation.residual_pts2d,
            sub_P,
            camera_models=models,
            min_depth=min_depth,
        )
        x = candidate
        err = candidate_err
    return x, err


def _soft_initial_point(observation, P_list):
    """All-camera weighted-DLT initialisation for the IRLS refinement."""
    sub_P = [P_list[int(camera)] for camera in observation.cams]
    weights = np.clip(observation.weights, 1e-6, 1.0)
    try:
        x = triangulate_weighted_dlt(observation.pts2d, sub_P, weights)
    except Exception:
        return None
    return x if np.isfinite(x).all() else None


def _soft_pair_initial_point(observation, P_list, min_depth, cfg):
    """Best minimal-pair solution scored by cross-camera support.

    A three-camera frame where every camera disagrees has no consensus to start
    the IRLS from; a minimal-pair initialisation (cheap RANSAC, as in Panoptic's
    per-finger solve) finds the dominant pair and lets the robust weights decide
    the final influence of the remaining cameras.
    """
    n = len(observation.cams)
    if n < 3:
        return None
    models = None if observation.models is None else list(observation.models)
    threshold = 2.0 * cfg.soft_scale_px
    best_x = None
    best_key: tuple[float, float] | None = None
    pairs = list(combinations(range(n), 2))
    if n > 6:
        # Production pipelines use 3 cameras; for larger rigs only the most
        # confident pairs are worth the extra solves.
        ranked = sorted(
            pairs,
            key=lambda pair: -(observation.confidences[pair[0]] + observation.confidences[pair[1]]),
        )
        pairs = ranked[:10]
    for left, right in pairs:
        index = [left, right]
        try:
            x = triangulate_weighted_dlt(
                observation.pts2d[index],
                [P_list[int(observation.cams[i])] for i in index],
                np.clip(observation.weights[index], 1e-6, 1.0),
            )
        except Exception:
            continue
        if not np.isfinite(x).all():
            continue
        err = _reprojection_errors(
            x,
            observation.residual_pts2d,
            [P_list[int(camera)] for camera in observation.cams],
            camera_models=models,
            min_depth=min_depth,
        )
        support = int(np.sum(np.isfinite(err) & (err <= threshold)))
        median = float(np.nanmedian(err)) if np.isfinite(err).any() else float("inf")
        key = (-float(support), median)
        if best_key is None or key < best_key:
            best_key = key
            best_x = x
    return best_x


def _anatomy_parent(joint: int) -> int | None:
    """Proximal joint of the bone used by the F2 anatomical tie-break.

    Only limb bones with a stable measured length are listed (COCO-WholeBody
    body arms/legs and the five finger chains of each hand).  Face joints are
    intentionally skipped.
    """
    if 91 <= joint <= 132:
        base = 91 if joint < 112 else 112
        offset = int(joint) - base
        if offset == 0:
            return 9 if base == 91 else 10
        for chain in (
            (0, 1, 2, 3, 4),
            (0, 5, 6, 7, 8),
            (0, 9, 10, 11, 12),
            (0, 13, 14, 15, 16),
            (0, 17, 18, 19, 20),
        ):
            if offset in chain[1:]:
                index = chain.index(offset)
                return base + chain[index - 1]
        return None
    return {
        7: 5, 9: 7, 8: 6, 10: 8,
        13: 11, 15: 13, 14: 12, 16: 14,
    }.get(int(joint))


def _robust_reprojection_cost(err_px, scale_px: float, loss: str) -> float:
    """Summed robust reprojection cost in pixel units; non-finite is penalised."""
    err = np.asarray(err_px, dtype=np.float64)
    finite = np.isfinite(err)
    if not finite.any():
        return float("inf")
    values = err[finite]
    scale = float(scale_px)
    if loss == "cauchy":
        rho = 0.5 * scale * scale * np.log1p(np.square(values / scale))
    else:
        rho = np.where(
            values <= scale,
            0.5 * np.square(values),
            scale * (values - 0.5 * scale),
        )
    return float(np.sum(rho)) + float(np.sum(~finite)) * 1.0e6


def _prior_prediction(
    pts3d: np.ndarray, frame: int, joint: int, cfg: GeometryConfig
) -> np.ndarray | None:
    """Previous-frame position or constant-velocity prediction for the F2 prior."""
    if frame < 1:
        return None
    previous = pts3d[frame - 1, joint]
    if not np.all(np.isfinite(previous)):
        return None
    if cfg.soft_prior_velocity and frame >= 2:
        before = pts3d[frame - 2, joint]
        if np.all(np.isfinite(before)):
            return 2.0 * previous - before
    return np.asarray(previous, dtype=np.float64)


def _anatomy_reference(
    pts3d: np.ndarray, frame: int, joint: int, window: int = 30
) -> float | None:
    """Running median measured bone length to the anatomy parent, in metres."""
    parent = _anatomy_parent(joint)
    if parent is None or frame < 1:
        return None
    start = max(0, int(frame) - int(window))
    child = pts3d[start:frame, joint]
    proximal = pts3d[start:frame, parent]
    usable = np.isfinite(child).all(axis=-1) & np.isfinite(proximal).all(axis=-1)
    if not usable.any():
        return None
    lengths = np.linalg.norm(child[usable] - proximal[usable], axis=-1)
    median = float(np.median(lengths))
    return median if np.isfinite(median) and median > 1.0e-6 else None


def _anatomy_penalty(
    point: np.ndarray, parent_point: np.ndarray | None, reference_m: float | None
) -> float:
    """Relative bone-length deviation (unit-free; both lengths in metres)."""
    if parent_point is None or reference_m is None or reference_m <= 0.0:
        return 0.0
    if not np.all(np.isfinite(parent_point)):
        return 0.0
    measured = float(np.linalg.norm(np.asarray(point, dtype=np.float64) - parent_point))
    return abs(measured - float(reference_m)) / float(reference_m)


def _soft_prior_solve(
    observation,
    P_list,
    min_depth,
    cfg,
    predicted: np.ndarray,
    anatomy_reference_m: float | None,
    anatomy_parent_point: np.ndarray | None,
):
    """Choose a camera set with the temporal prior and initialise the IRLS.

    Candidates are every minimal pair plus the full observed set.  Each is
    scored by the summed robust reprojection cost over *all* observed cameras,
    the weighted distance to ``predicted`` (millimetres) and a small anatomical
    tie-break.  The winning set keeps its confidence weights; cameras outside
    it start at ``cfg.soft_prior_downweight`` so the IRLS cannot immediately
    drag the solution back to a self-consistent wrong consensus.
    """
    n = len(observation.cams)
    if n < 3:
        return None
    pairs = list(combinations(range(n), 2))
    if n > 6:
        ranked = sorted(
            pairs,
            key=lambda pair: -(
                observation.confidences[pair[0]] + observation.confidences[pair[1]]
            ),
        )
        pairs = ranked[:10]
    candidates = pairs + [tuple(range(n))]

    models = None if observation.models is None else list(observation.models)
    all_P = [P_list[int(camera)] for camera in observation.cams]
    best = None
    best_cost = float("inf")
    reprojection_best = None
    reprojection_cost = float("inf")
    for candidate in candidates:
        index = list(candidate)
        try:
            point = triangulate_weighted_dlt(
                observation.pts2d[index],
                [all_P[i] for i in index],
                np.clip(observation.weights[index], 1e-6, 1.0),
            )
        except Exception:
            continue
        if not np.isfinite(point).all():
            continue
        err = _reprojection_errors(
            point,
            observation.residual_pts2d,
            all_P,
            camera_models=models,
            min_depth=min_depth,
        )
        reprojection = _robust_reprojection_cost(
            err, cfg.soft_scale_px, cfg.robust_loss
        )
        if reprojection < reprojection_cost:
            reprojection_cost = reprojection
            reprojection_best = candidate
        cost = reprojection
        cost += cfg.soft_prior_w_temporal * float(
            np.linalg.norm(point - predicted)
        ) * 1000.0
        cost += cfg.soft_prior_w_anat * _anatomy_penalty(
            point, anatomy_parent_point, anatomy_reference_m
        )
        if cost < best_cost:
            best_cost = cost
            best = (point, candidate, err)
    if best is None:
        return None
    point, chosen, chosen_err = best
    finite = np.asarray(chosen_err)[np.isfinite(chosen_err)]
    residual_max = float(np.max(finite)) if finite.size else float("inf")
    base_weights = np.clip(observation.weights, 1e-6, 1.0).copy()
    for local in range(n):
        if local not in chosen:
            base_weights[local] = max(
                1e-6, base_weights[local] * float(cfg.soft_prior_downweight)
            )
    return point, base_weights, chosen, residual_max, reprojection_best


def _soft_observation_solve(
    observation,
    P_list,
    min_depth,
    cfg,
    prior: np.ndarray | None = None,
    anatomy_reference_m: float | None = None,
    anatomy_parent_point: np.ndarray | None = None,
):
    """Robust solve: initialise, IRLS-refine, retry from a pair if unsupported.

    The solve uses undistorted points; the robust kernel measures the residual at
    the *observed* pixels, so ``soft_scale_px`` is a true image-space scale.
    With the F2 temporal prior the winning camera set provides the initial
    point and weights; without a prior this is the P3.7 behaviour.
    """
    chosen = None
    prior_residual_max = None
    reprojection_best = None
    if cfg.soft_temporal_prior and prior is not None and len(observation.cams) >= 3:
        prior_result = _soft_prior_solve(
            observation,
            P_list,
            min_depth,
            cfg,
            prior,
            anatomy_reference_m,
            anatomy_parent_point,
        )
        if prior_result is not None:
            initial, base_weights, chosen, prior_residual_max, reprojection_best = (
                prior_result
            )
            x, err = _soft_irls_refine(
                initial, observation, P_list, min_depth, cfg, base_weights=base_weights
            )
            return x, err, chosen, prior_residual_max, reprojection_best
    initial = _soft_initial_point(observation, P_list)
    if initial is None:
        return None, None, None, None, None
    x, err = _soft_irls_refine(initial, observation, P_list, min_depth, cfg)
    support_threshold = 2.0 * cfg.soft_scale_px
    support = int(np.sum(np.isfinite(err) & (err <= support_threshold)))
    if support < min(2, len(observation.cams)) and len(observation.cams) >= 3:
        pair_initial = _soft_pair_initial_point(observation, P_list, min_depth, cfg)
        if pair_initial is not None:
            pair_x, pair_err = _soft_irls_refine(
                pair_initial, observation, P_list, min_depth, cfg
            )
            pair_support = int(
                np.sum(np.isfinite(pair_err) & (pair_err <= support_threshold))
            )
            if pair_support > support:
                x, err = pair_x, pair_err
    return x, err, chosen, prior_residual_max, reprojection_best



def _soft_support_decision(local_trust, err_local, cfg):
    """F1 decision for one joint-frame: emit plus support/fit camera masks."""
    trust = np.asarray(local_trust, dtype=np.float64)
    err = np.asarray(err_local, dtype=np.float64)
    support = trust >= cfg.soft_support_min_trust
    fit = support & np.isfinite(err) & (err <= 2.0 * cfg.soft_scale_px)
    if cfg.soft_support_gate and (
        int(np.count_nonzero(support)) < 2 or int(np.count_nonzero(fit)) < 2
    ):
        return False, support, fit
    return True, support, fit


def _soft_group_first_pass(
    items, P_list, min_depth, cfg, pts3d=None, frame=None, lockin_state=None
):
    """Solve every joint of one trust group with all of its cameras."""
    solved: dict[int, tuple[np.ndarray, dict[int, float]]] = {}
    for joint, observation in items:
        prior = None
        anatomy_reference_m = None
        anatomy_parent_point = None
        use_prior = False
        if cfg.soft_temporal_prior and pts3d is not None and frame is not None:
            prior = _prior_prediction(pts3d, int(frame), int(joint), cfg)
            escaped = (
                lockin_state is not None
                and int(lockin_state[joint]) >= int(cfg.soft_prior_lockin_frames)
            )
            use_prior = prior is not None and not escaped
            if use_prior:
                parent = _anatomy_parent(int(joint))
                if parent is not None:
                    anatomy_reference_m = _anatomy_reference(
                        pts3d, int(frame), int(joint)
                    )
                    if np.all(np.isfinite(pts3d[int(frame), parent])):
                        anatomy_parent_point = np.asarray(
                            pts3d[int(frame), parent], dtype=np.float64
                        )
                    elif int(frame) >= 1 and np.all(
                        np.isfinite(pts3d[int(frame) - 1, parent])
                    ):
                        anatomy_parent_point = np.asarray(
                            pts3d[int(frame) - 1, parent], dtype=np.float64
                        )
        x, err, chosen, prior_residual_max, reprojection_best = _soft_observation_solve(
            observation,
            P_list,
            min_depth,
            cfg,
            prior=prior if use_prior else None,
            anatomy_reference_m=anatomy_reference_m,
            anatomy_parent_point=anatomy_parent_point,
        )
        if lockin_state is not None and cfg.soft_temporal_prior:
            locked_set = (
                use_prior
                and chosen is not None
                and len(chosen) < len(observation.cams)
                and chosen != reprojection_best
            )
            if (
                locked_set
                and prior_residual_max is not None
                and prior_residual_max > cfg.soft_prior_lockin_scale * cfg.soft_scale_px
            ):
                lockin_state[joint] += 1
            else:
                lockin_state[joint] = 0
        if x is None:
            continue
        per_camera = {
            int(camera): float(value)
            for camera, value in zip(observation.cams, err)
            if np.isfinite(value)
        }
        solved[int(joint)] = (x, per_camera)
    return solved


def _aggregate_camera_residuals(solved) -> dict[int, float]:
    """Median px residual per camera over the group's joints (Panoptic-style)."""
    values: dict[int, list[float]] = {}
    for _, per_camera in solved.values():
        for camera, value in per_camera.items():
            if np.isfinite(value):
                values.setdefault(int(camera), []).append(float(value))
    return {
        camera: float(np.median(samples))
        for camera, samples in values.items()
        if samples
    }


def _group_ray_angle_deg(solved, cameras: list[int], C_list: list[np.ndarray]) -> float:
    if not solved or len(cameras) < 2:
        return 0.0
    points = np.stack([point for point, _ in solved.values()], axis=0)
    target = np.median(points, axis=0).reshape((3, 1))
    return _max_pairwise_angle(cameras, target, C_list)


def _soft_group_trust(
    items,
    *,
    P_list,
    C_list,
    min_depth,
    cfg,
    prev_trust: np.ndarray,
    num_cameras: int,
    pts3d=None,
    frame=None,
    lockin_state=None,
) -> tuple[dict[int, tuple[np.ndarray, dict[int, float]]], np.ndarray]:
    """Return the group's first-pass solves and its final per-camera trust vector."""
    first = _soft_group_first_pass(
        items,
        P_list,
        min_depth,
        cfg,
        pts3d=pts3d,
        frame=frame,
        lockin_state=lockin_state,
    )
    residuals = _aggregate_camera_residuals(first)
    target = np.ones(num_cameras, dtype=np.float64)
    if residuals:
        cameras = sorted(residuals)
        target[cameras] = _robust_influence(
            np.array([residuals[camera] for camera in cameras]),
            cfg.soft_scale_px,
            cfg.robust_loss,
        )

    # Leave-one-out: a camera that is the odd one out has a much larger
    # residual than the agreeing majority at the robust consensus.  With three
    # cameras a re-solved pair always fits itself, so the honest LOO statistic
    # is the dropped camera's consensus residual relative to the remaining
    # median; the remaining cameras must agree (all within the mask threshold)
    # and keep a usable ray angle.  The down-weight is continuous.
    if len(residuals) >= 3:
        cameras = sorted(residuals)
        worst = max(cameras, key=lambda camera: residuals[camera])
        remaining = [camera for camera in cameras if camera != worst]
        remaining_values = [residuals[camera] for camera in remaining]
        remaining_median = float(np.median(remaining_values))
        if (
            residuals[worst] >= cfg.soft_scale_px
            and max(remaining_values) <= 2.0 * cfg.soft_scale_px
            and residuals[worst] >= cfg.soft_loo_ratio * max(remaining_median, 1e-6)
        ):
            angle = _group_ray_angle_deg(first, remaining, C_list)
            if angle >= cfg.min_ray_angle_deg:
                reduction = residuals[worst] / max(remaining_median, 1e-6)
                target[worst] *= float(
                    np.clip(cfg.soft_loo_ratio / reduction, cfg.soft_loo_floor, 1.0)
                )

    trust = np.ones(num_cameras, dtype=np.float64)
    fast_px = cfg.soft_fast_change_scale * cfg.soft_scale_px
    for camera, residual in residuals.items():
        new_trust = float(np.clip(target[camera], cfg.soft_loo_floor, 1.0))
        if cfg.soft_ema_alpha <= 0.0 or residual > fast_px:
            blended = new_trust
        else:
            blended = (
                cfg.soft_ema_alpha * float(prev_trust[camera])
                + (1.0 - cfg.soft_ema_alpha) * new_trust
            )
        trust[camera] = float(np.clip(blended, 0.0, 1.0))
    return first, trust


def soft_trust_groups(num_joints: int) -> tuple[list[tuple[int, ...]], np.ndarray]:
    """Return trust groups and the joint -> group assignment.

    Body, feet and face joints are independent groups (one group per joint), so
    the face policy is unchanged.  Hand joints follow the Panoptic per-finger
    rule: the wrist/palm is its own group and each finger is a chain
    ``(wrist, four finger joints)``; finger joints share the chain's trust so a
    finger can never be split across camera sets.
    """
    groups: list[tuple[int, ...]] = [(joint,) for joint in range(min(int(num_joints), 91))]
    assignment = np.arange(int(num_joints), dtype=np.int64)
    for base in (91, 112):
        if base + 21 > num_joints:
            continue
        groups.append((base,))
        assignment[base] = len(groups) - 1
        for chain in ((0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (0, 9, 10, 11, 12),
                      (0, 13, 14, 15, 16), (0, 17, 18, 19, 20)):
            groups.append(tuple(base + offset for offset in chain))
            for offset in chain[1:]:
                assignment[base + offset] = len(groups) - 1
    return groups, assignment


def _temporal_anatomy_penalties(
    candidate: np.ndarray,
    *,
    prev_point3d: np.ndarray | None,
    prev_velocity3d: np.ndarray | None,
    connected_point3d: np.ndarray | None,
    ref_bone_length_cm: float | None,
    legacy_units: bool,
) -> tuple[float, float]:
    """Return the (temporal, anatomical) scoring penalties for one candidate.

    The engine works in metres.  ``w_temporal`` was tuned on centimetre
    displacements, so the temporal term converts the displacement to
    centimetres (``disp_cm``).  The anatomical table is authored in centimetres,
    so ``ref_bone_length_cm`` is converted to metres before the relative
    deviation is computed; the penalty itself is a percentage and stays
    unit-free.

    ``legacy_units=True`` reproduces the historical numbers exactly: metres
    were fed to the cm-named temporal term, the centimetre table was compared
    against metres (making the bone penalty effectively constant per joint),
    and the 0.5-unit speed gate was applied to metre steps instead of
    centimetre steps.  It exists only for bit-exact reproduction of pre-fix
    output.
    """
    scale = 1.0 if legacy_units else 100.0
    temp_penalty = 0.0
    if prev_point3d is not None and np.all(np.isfinite(prev_point3d)):
        delta = candidate - prev_point3d
        displacement = float(np.linalg.norm(delta)) * scale
        cos_theta_v = 1.0
        if prev_velocity3d is not None and np.all(np.isfinite(prev_velocity3d)):
            prev_step = float(np.linalg.norm(prev_velocity3d)) * scale
            if displacement > 0.5 and prev_step > 0.5:
                cos_theta_v = float(
                    np.dot(delta, prev_velocity3d)
                    / (float(np.linalg.norm(delta)) * float(np.linalg.norm(prev_velocity3d)))
                )
        temp_penalty = displacement * (1.0 + max(0.0, -cos_theta_v) * 2.0)

    anat_penalty = 0.0
    if (
        connected_point3d is not None
        and ref_bone_length_cm is not None
        and ref_bone_length_cm > 0.0
        and np.all(np.isfinite(connected_point3d))
    ):
        meas_len_m = float(np.linalg.norm(candidate - connected_point3d))
        ref_len_m = ref_bone_length_cm if legacy_units else ref_bone_length_cm / 100.0
        dev_pct = abs(meas_len_m - ref_len_m) / ref_len_m * 100.0
        if dev_pct > 10.0:
            anat_penalty = (dev_pct - 10.0) / 10.0
    return temp_penalty, anat_penalty


def _triangulate_with_inlier_selection(
    pts2d: np.ndarray,
    P_list: list[np.ndarray],
    confidences: np.ndarray,
    camera_indices: np.ndarray,
    reproj_threshold_px: float,
    camera_models: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] | None = None,
    min_depth: float = 0.0,
    residual_pts2d: np.ndarray | None = None,
    hard_reproj_threshold_px: float | None = None,
    prev_inliers: set[int] | None = None,
    hysteresis_margin_px: float = 0.0,
    camera_centers: list[np.ndarray] | None = None,
    prev_point3d: np.ndarray | None = None,
    prev_velocity3d: np.ndarray | None = None,
    connected_point3d: np.ndarray | None = None,
    ref_bone_length_cm: float | None = None,
    w_temporal: float = 0.0,
    w_geom: float = 0.0,
    w_anat: float = 0.0,
    legacy_units: bool = False,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Triangulate one joint with leave-one-out style inlier selection.

    Evaluates camera subsets of size >= 2 using consensus dominance, reprojection residual,
    continuous geometry reliability, directional temporal continuity, and anatomical sanity.

    ``confidences`` stay the calibrated probabilities used for scoring and the
    emitted 3D confidence; ``weights`` (optional) are the per-observation
    weighted-DLT weights ``p ** gamma``.  When omitted the two are identical.
    """
    n = len(P_list)
    if n < 2:
        raise ValueError("At least 2 views are required.")
    tri_weights = confidences if weights is None else np.asarray(weights, dtype=np.float64)
    if tri_weights.shape != np.asarray(confidences).shape:
        raise ValueError("weights must have the same shape as confidences")
    residual_points = pts2d if residual_pts2d is None else np.asarray(residual_pts2d, dtype=np.float64)
    if residual_points.shape != pts2d.shape:
        raise ValueError("residual_pts2d must have the same shape as pts2d")

    best_key: tuple[float, ...] | None = None
    best_score: float | None = None
    best_point: np.ndarray | None = None
    best_err_local: np.ndarray | None = None
    best_subset: tuple[int, ...] | None = None

    use_multiobj = (w_temporal > 0.0 or w_geom > 0.0 or w_anat > 0.0)

    subsets: list[tuple[int, ...]] = []
    if n <= 5:
        for size in range(2, n + 1):
            subsets.extend(tuple(x) for x in combinations(range(n), size))
    else:
        subsets.extend(tuple(x) for x in combinations(range(n), 2))
        subsets.append(tuple(range(n)))

    for subset in subsets:
        sub_idx = np.array(subset, dtype=np.int32)
        try:
            x = triangulate_weighted_dlt(
                pts2d[sub_idx], [P_list[i] for i in sub_idx], tri_weights[sub_idx],
            )
        except Exception:
            continue
        if not np.isfinite(x).all():
            continue
        err_all = _reprojection_errors(
            x, residual_points, P_list, camera_models=camera_models, min_depth=min_depth
        )
        finite = np.isfinite(err_all)
        if not finite.any():
            continue
        hard = reproj_threshold_px if hard_reproj_threshold_px is None else hard_reproj_threshold_px
        global_inliers = np.isfinite(err_all) & (err_all <= hard)
        support = int(global_inliers.sum())
        if support < 2:
            continue
        global_err = err_all[global_inliers]
        robust_err = float(np.median(global_err) + 0.25 * np.max(global_err))
        held_out = err_all[~global_inliers]
        held_out_penalty = float(np.median(held_out)) if held_out.size else 0.0
        confidence_sum = float(np.sum(confidences[global_inliers]))

        # Temporal hysteresis: check if candidate matches previous winning inliers
        scored_robust_err = robust_err
        if prev_inliers is not None and len(prev_inliers) > 0 and hysteresis_margin_px > 0.0:
            candidate_cams = set(camera_indices[global_inliers])
            if candidate_cams == prev_inliers:
                scored_robust_err = max(0.0, robust_err - hysteresis_margin_px)

        if use_multiobj:
            # 1. Geometry penalty: ray intersection angle
            geom_penalty = 0.0
            if camera_centers is not None and len(subset) >= 2:
                c1_idx = camera_indices[subset[0]]
                c2_idx = camera_indices[subset[1]]
                if c1_idx < len(camera_centers) and c2_idx < len(camera_centers):
                    v1 = x.reshape((3, 1)) - camera_centers[c1_idx]
                    v2 = x.reshape((3, 1)) - camera_centers[c2_idx]
                    n1 = float(np.linalg.norm(v1))
                    n2 = float(np.linalg.norm(v2))
                    if n1 > 1e-6 and n2 > 1e-6:
                        cos_t = np.clip(np.dot(v1.flatten(), v2.flatten()) / (n1 * n2), -1.0, 1.0)
                        ray_angle_deg = float(np.arccos(cos_t) * 180.0 / np.pi)
                        sin_theta = float(max(0.01, np.sin(ray_angle_deg * np.pi / 180.0)))
                        geom_penalty = float(1.0 - sin_theta)

            # 2. Directional temporal displacement penalty + 3. Weak
            # anatomical bone-length penalty.  Both are centimetre-aware; see
            # _temporal_anatomy_penalties for the unit contract.
            temp_penalty, anat_penalty = _temporal_anatomy_penalties(
                x,
                prev_point3d=prev_point3d,
                prev_velocity3d=prev_velocity3d,
                connected_point3d=connected_point3d,
                ref_bone_length_cm=ref_bone_length_cm,
                legacy_units=legacy_units,
            )

            combined_score = (
                scored_robust_err
                + w_temporal * temp_penalty
                + w_geom * (geom_penalty * 5.0)
                + w_anat * (anat_penalty * 5.0)
                - support * 100.0
            )

            if best_score is None or combined_score < best_score:
                best_score = combined_score
                best_point = x
                best_err_local = err_all
                best_subset = subset
        else:
            # Lexicographic ordering: maximize consensus, minimize global error,
            # minimize held-out disagreement, then maximize confidence.
            key = (-float(support), scored_robust_err, held_out_penalty, -confidence_sum)
            if best_key is None or key < best_key:
                best_key = key
                best_point = x
                best_err_local = err_all
                best_subset = subset

    if best_point is None or best_err_local is None or best_subset is None:
        x = triangulate_weighted_dlt(pts2d, P_list, tri_weights)
        err_all = _reprojection_errors(
            x, residual_points, P_list, camera_models=camera_models, min_depth=min_depth
        )
    else:
        x = best_point
        err_all = best_err_local

    hard_threshold = (
        reproj_threshold_px if hard_reproj_threshold_px is None else hard_reproj_threshold_px
    )
    inliers_local = (err_all <= hard_threshold) & np.isfinite(err_all)
    if int(inliers_local.sum()) >= 2 and int(inliers_local.sum()) < n:
        try:
            x = triangulate_weighted_dlt(
                pts2d[inliers_local],
                [P_list[i] for i in np.flatnonzero(inliers_local)],
                tri_weights[inliers_local],
            )
            err_all = _reprojection_errors(
                x, residual_points, P_list, camera_models=camera_models, min_depth=min_depth
            )
            inliers_local = (err_all <= hard_threshold) & np.isfinite(err_all)
        except Exception:
            pass

    med_err = float(np.nanmedian(np.where(np.isfinite(err_all), err_all, np.nan)))
    conf = float(np.mean(confidences[inliers_local])) if int(inliers_local.sum()) else 0.0
    conf *= float(1.0 / (1.0 + max(0.0, med_err) / max(1.0, hard_threshold)))
    return x, inliers_local, err_all, conf


def _triangulate_sequence_soft(
    keypoints: np.ndarray,
    scores: np.ndarray,
    K_list: list[np.ndarray],
    extrinsics: list[tuple[np.ndarray, np.ndarray]],
    cfg: GeometryConfig,
    dist_coeffs: list[np.ndarray] | None,
    image_sizes: list[tuple[int, int]] | np.ndarray | None,
    lr_swap_mask: np.ndarray | None = None,
) -> TriangulationDiagnostics:
    """Soft camera-trust triangulation (``GeometryConfig.camera_policy='soft'``).

    Per joint/frame: IRLS robust solve, continuous leave-one-out trust, temporal
    EMA of that trust, then a final confidence-and-trust weighted DLT.  Hand
    joints share one trust vector per finger chain.  The hard inlier mask is
    kept for consumers as ``trust > 0.5``; the continuous trust is reported in
    ``camera_trust``.
    """
    num_frames, num_cameras, num_kpts, _ = keypoints.shape
    if len(K_list) != num_cameras or len(extrinsics) != num_cameras:
        raise ValueError("camera calibration count must match keypoints camera dimension")
    if dist_coeffs is not None and len(dist_coeffs) != num_cameras:
        raise ValueError("dist_coeffs count must match keypoints camera dimension")

    part_confidence = [cfg.confidence_for_joint(k) for k in range(num_kpts)]
    reproj_threshold_px = cfg.max_reprojection_error_px
    P_list = [K_list[c] @ np.hstack(extrinsics[c]) for c in range(num_cameras)]
    camera_models = None
    if dist_coeffs is not None:
        camera_models = [
            (np.asarray(K_list[c], dtype=np.float64),
             np.asarray(extrinsics[c][0], dtype=np.float64),
             np.asarray(extrinsics[c][1], dtype=np.float64),
             np.asarray(dist_coeffs[c], dtype=np.float64))
            for c in range(num_cameras)
        ]

    # The subset-search accumulators are intentionally identical to the legacy
    # path so existing consumers and gates keep working.
    pts3d = np.full((num_frames, num_kpts, 3), np.nan, dtype=np.float64)
    candidate_pts3d = np.full((num_frames, num_kpts, 3), np.nan, dtype=np.float64)
    confidence = np.zeros((num_frames, num_kpts), dtype=np.float32)
    candidate_inlier_mask = np.zeros((num_frames, num_kpts, num_cameras), dtype=bool)
    inlier_mask = np.zeros((num_frames, num_kpts, num_cameras), dtype=bool)
    support_mask = np.zeros((num_frames, num_kpts, num_cameras), dtype=bool)
    camera_trust = np.zeros((num_frames, num_kpts, num_cameras), dtype=np.float32)
    reprojection_error_px = np.full((num_frames, num_kpts, num_cameras), np.nan, dtype=np.float32)
    candidate_num_inliers = np.zeros((num_frames, num_kpts), dtype=np.uint8)
    num_inliers = np.zeros((num_frames, num_kpts), dtype=np.uint8)
    ray_angle_deg = np.zeros((num_frames, num_kpts), dtype=np.float32)
    observed_camera_count = np.zeros((num_frames, num_kpts), dtype=np.uint8)
    positive_depth_mask = np.zeros((num_frames, num_kpts, num_cameras), dtype=bool)
    observation_provenance = np.full((num_frames, num_kpts), "unobserved", dtype="<U32")
    camera_rejection_reason = np.full(
        (num_frames, num_kpts, num_cameras), "not_observed", dtype="<U32"
    )

    C_list = [-extrinsics[c][0].T @ extrinsics[c][1] for c in range(num_cameras)]
    low_quality_mask = np.zeros(num_frames, dtype=bool)
    groups, group_of_joint = soft_trust_groups(num_kpts)
    prev_trust = np.ones((len(groups), num_cameras), dtype=np.float64)
    lockin_state = np.zeros(num_kpts, dtype=np.int32)

    for f in range(num_frames):
        valid_cam_for_frame = list(range(num_cameras))
        max_angle = _max_pairwise_angle(valid_cam_for_frame, np.zeros((3, 1)), C_list)
        if max_angle < 35.0:
            low_quality_mask[f] = True

        observations: list[_SoftObservation | None] = [None] * num_kpts
        for k in range(num_kpts):
            part = part_confidence[k]
            valid_pts2d = []
            valid_observed_pts2d = []
            valid_conf = []
            valid_weight = []
            valid_cam = []
            valid_models = []

            for c in valid_cam_for_frame:
                score = float(scores[f, c, k])
                if not np.isfinite(score) or score < part.min_conf:
                    camera_rejection_reason[f, k, c] = "low_confidence"
                    continue
                if not np.isfinite(keypoints[f, c, k]).all():
                    camera_rejection_reason[f, k, c] = "nonfinite_2d"
                    continue
                if image_sizes is not None:
                    width, height = image_sizes[c]
                    x, y = keypoints[f, c, k]
                    if not (0.0 <= x < float(width) and 0.0 <= y < float(height)):
                        camera_rejection_reason[f, k, c] = "outside_image"
                        continue
                observed_point = np.asarray(keypoints[f, c, k], dtype=np.float64)
                point = observed_point.copy()
                valid_observed_pts2d.append(observed_point)
                if camera_models is not None:
                    K, _, _, dist = camera_models[c]
                    point = cv2.undistortPoints(
                        point.reshape(1, 1, 2), K, dist, P=K
                    ).reshape(2)
                    valid_models.append(camera_models[c])
                valid_pts2d.append(point)
                valid_conf.append(score)
                valid_weight.append(part.weight(score))
                valid_cam.append(c)

            observed_camera_count[f, k] = np.uint8(len(valid_pts2d))
            if len(valid_pts2d) == 0:
                continue
            if len(valid_pts2d) < 2:
                observation_provenance[f, k] = "one_camera_only"
                continue
            observations[k] = _SoftObservation(
                valid_cam,
                valid_pts2d,
                valid_observed_pts2d,
                valid_conf,
                valid_weight,
                valid_models if camera_models is not None else None,
            )

        for group_index, members in enumerate(groups):
            items = [
                (k, observations[k]) for k in members if observations[k] is not None
            ]
            if not items:
                continue
            first, trust = _soft_group_trust(
                items,
                P_list=P_list,
                C_list=C_list,
                min_depth=cfg.min_depth_m,
                cfg=cfg,
                prev_trust=prev_trust[group_index],
                num_cameras=num_cameras,
                pts3d=pts3d,
                frame=f,
                lockin_state=lockin_state,
            )
            prev_trust[group_index] = trust

            for k, observation in items:
                group_final = first.get(k)
                if group_final is None:
                    observation_provenance[f, k] = "geometry_rejected"
                    continue
                candidate_point = group_final[0]

                local_trust = trust[observation.cams]
                final_weights = np.clip(observation.weights * local_trust, 1e-6, 1.0)
                sub_P = [P_list[int(camera)] for camera in observation.cams]
                models = (
                    None
                    if observation.models is None
                    else list(observation.models)
                )
                try:
                    point = triangulate_weighted_dlt(
                        observation.pts2d, sub_P, final_weights
                    )
                except Exception:
                    observation_provenance[f, k] = "geometry_rejected"
                    continue
                if not np.isfinite(point).all():
                    observation_provenance[f, k] = "geometry_rejected"
                    continue
                err_local = _reprojection_errors(
                    point,
                    observation.residual_pts2d,
                    sub_P,
                    camera_models=models,
                    min_depth=cfg.min_depth_m,
                )
                local_mask = local_trust > 0.5
                emit, local_support, fit_support = _soft_support_decision(
                    local_trust, err_local, cfg
                )
                med_err = float(
                    np.nanmedian(np.where(np.isfinite(err_local), err_local, np.nan))
                )
                conf3d = (
                    float(np.mean(observation.confidences[local_mask]))
                    if bool(local_mask.any())
                    else 0.0
                )
                conf3d *= 1.0 / (1.0 + max(0.0, med_err) / max(1.0, reproj_threshold_px))

                camera_trust[f, k, observation.cams] = local_trust.astype(np.float32)
                candidate_pts3d[f, k] = candidate_point
                candidate_inlier_mask[f, k, observation.cams] = local_mask
                inlier_mask[f, k, observation.cams] = local_mask
                support_mask[f, k, observation.cams] = local_support
                reprojection_error_px[f, k, observation.cams] = err_local.astype(np.float32)
                confidence[f, k] = np.float32(conf3d if emit else 0.0)
                num_inliers[f, k] = np.uint8(local_mask.sum())
                candidate_num_inliers[f, k] = np.uint8(local_mask.sum())

                if not emit:
                    # F1: a point whose depth is set by low-trust cameras (or
                    # that no two supporting cameras actually fit) is not
                    # emitted; the smoother may only bridge its existing
                    # short-gap limit.
                    pts3d[f, k] = np.nan
                    observation_provenance[f, k] = "insufficient_support"
                    for local, c in enumerate(observation.cams):
                        if not local_support[local]:
                            camera_rejection_reason[f, k, c] = "low_support"
                        elif not fit_support[local]:
                            camera_rejection_reason[f, k, c] = "high_reprojection"
                        else:
                            camera_rejection_reason[f, k, c] = "insufficient_support"
                    continue

                # Soft mode reports the trust mask but does not let it erase the
                # point: the robust solve already decided the influence of every
                # observed camera, and a 3-camera frame with one trusted camera
                # still carries two corroborating (down-weighted) observations.
                # The legacy ``min_inliers`` gate therefore applies to the legacy
                # policy only; with F1 a point needs >= 2 supporting cameras.
                pts3d[f, k] = point
                observation_provenance[f, k] = (
                    "observed_2plus"
                    if num_inliers[f, k] == observed_camera_count[f, k]
                    else "observed_with_rejected_views"
                )

                for c in observation.cams:
                    R_c, t_c = extrinsics[int(c)]
                    depth = float(
                        (np.asarray(R_c) @ point.reshape(3) + np.asarray(t_c).reshape(3))[2]
                    )
                    positive_depth_mask[f, k, c] = depth > cfg.min_depth_m
                for local, c in enumerate(observation.cams):
                    if not positive_depth_mask[f, k, c]:
                        camera_rejection_reason[f, k, c] = "nonpositive_depth"
                    elif not np.isfinite(err_local[local]):
                        camera_rejection_reason[f, k, c] = "invalid_projection"
                    elif not local_mask[local]:
                        camera_rejection_reason[f, k, c] = (
                            "high_reprojection"
                            if err_local[local] > reproj_threshold_px
                            else "low_trust"
                        )
                    else:
                        camera_rejection_reason[f, k, c] = "inlier"

                # Ray angle is a property of the cameras that corroborate the
                # point.  In soft mode the continuous $\geq$ support threshold
                # defines that set; ``inlier_mask`` keeps its historical
                # trust > 0.5 meaning for switch/confidence consumers.
                support_cams = np.flatnonzero(support_mask[f, k])
                if len(support_cams) >= 2:
                    max_ang = _max_pairwise_angle(
                        support_cams, point.reshape((3, 1)), C_list
                    )
                    ray_angle_deg[f, k] = np.float32(max_ang)
                inlier_cams = np.flatnonzero(inlier_mask[f, k])
                if len(inlier_cams) >= 2:
                    max_ang = float(ray_angle_deg[f, k])
                    if max_ang < cfg.min_ray_angle_deg:
                        pts3d[f, k] = np.nan
                        confidence[f, k] = 0.0
                        inlier_mask[f, k] = False
                        support_mask[f, k] = False
                        num_inliers[f, k] = 0
                        observation_provenance[f, k] = "low_ray_angle"
                        for c in inlier_cams:
                            camera_rejection_reason[f, k, c] = "low_ray_angle"

    pts3d_internal = opencv_to_internal(pts3d)
    candidate_pts3d_internal = opencv_to_internal(candidate_pts3d)
    return TriangulationDiagnostics(
        points3d=pts3d_internal,
        candidate_points3d=candidate_pts3d_internal,
        confidence=confidence,
        candidate_inlier_mask=candidate_inlier_mask,
        inlier_mask=inlier_mask,
        support_mask=support_mask,
        camera_trust=camera_trust,
        reprojection_error_px=reprojection_error_px,
        candidate_num_inliers=candidate_num_inliers,
        num_inliers=num_inliers,
        low_quality_mask=low_quality_mask,
        ray_angle_deg=ray_angle_deg,
        observed_camera_count=observed_camera_count,
        positive_depth_mask=positive_depth_mask,
        observation_provenance=observation_provenance,
        camera_rejection_reason=camera_rejection_reason,
        lr_swap_mask=(
            np.zeros((num_frames, num_cameras, len(LR_SWAP_GROUPS)), dtype=bool)
            if lr_swap_mask is None
            else lr_swap_mask
        ),
    )


def _apply_temporal_filter(
    diagnostics: TriangulationDiagnostics,
    cfg: GeometryConfig,
    fps: float | None,
) -> TriangulationDiagnostics:
    """Apply the opt-in P3.5 smoother to the camera-world point sequence."""
    if cfg.temporal_filter == "none":
        return diagnostics
    if fps is None or not np.isfinite(fps) or fps <= 0.0:
        raise ValueError(
            "fps must be provided and positive when temporal_filter is enabled"
        )
    points = filter_skeleton3d_rts(
        diagnostics.points3d,
        float(fps),
        sigma_m=cfg.temporal_filter_sigma_m,
        q_m2_s5=cfg.temporal_filter_q,
        maneuver_boost=cfg.temporal_filter_maneuver_boost,
        trust=diagnostics.camera_trust,
        reprojection_error_px=diagnostics.reprojection_error_px,
        max_gap_s=cfg.temporal_filter_max_gap_s,
        fast_q_boost=cfg.temporal_filter_fast_boost,
        spike_gate=cfg.temporal_spike_gate,
        mad_window=cfg.temporal_mad_window,
        mad_scale=cfg.temporal_mad_scale,
        mad_min_samples=cfg.temporal_mad_min_samples,
        maneuver_cap=cfg.temporal_maneuver_cap,
    )
    result = replace(diagnostics, points3d=points)
    if cfg.confidence3d:
        # Sizes of the smoother correction at each point; the P3.6 confidence
        # model uses it as the "smoother innovation" runtime signal.
        raw = diagnostics.points3d
        finite = np.isfinite(raw).all(axis=-1) & np.isfinite(points).all(axis=-1)
        innovation = np.full(raw.shape[:2], np.nan, dtype=np.float64)
        innovation[finite] = np.linalg.norm(points[finite] - raw[finite], axis=-1)
        result = replace(result, smoother_innovation_m=innovation)
    return result


def _attach_confidence3d(
    diagnostics: TriangulationDiagnostics,
    keypoints: np.ndarray,
    scores: np.ndarray,
) -> TriangulationDiagnostics:
    """Add the opt-in P3.6 per-point 3D confidence to soft diagnostics."""
    finite = np.isfinite(diagnostics.points3d).all(axis=-1)
    features = confidence3d.build_features(
        diagnostics.camera_trust,
        diagnostics.reprojection_error_px,
        diagnostics.ray_angle_deg,
        keypoints=keypoints,
        scores=scores,
        innovation_m=diagnostics.smoother_innovation_m,
        points_finite=finite,
    )
    model = confidence3d.default_model()
    values = confidence3d.predict_confidence3d(
        features, model, valid=finite, num_joints=features.shape[1]
    )
    return replace(
        diagnostics,
        confidence3d=values,
        confidence3d_features=features.astype(np.float32),
    )


def triangulate_sequence(
    keypoints: np.ndarray,
    scores: np.ndarray,
    K_list: list[np.ndarray],
    extrinsics: list[tuple[np.ndarray, np.ndarray]],
    min_conf: float = 0.5,
    min_aspect_ratio: float = 0.0,
    dist_coeffs: list[np.ndarray] | None = None,
    geometry_config: GeometryConfig | None = None,
    image_sizes: list[tuple[int, int]] | np.ndarray | None = None,
    fps: float | None = None,
) -> np.ndarray:
    """Back-compatible wrapper returning only Y-up 3D points."""
    return triangulate_sequence_with_diagnostics(
        keypoints,
        scores,
        K_list,
        extrinsics,
        min_conf=min_conf,
        min_aspect_ratio=min_aspect_ratio,
        dist_coeffs=dist_coeffs,
        geometry_config=geometry_config,
        image_sizes=image_sizes,
        fps=fps,
    ).points3d


def triangulate_sequence_with_diagnostics(
    keypoints: np.ndarray,
    scores: np.ndarray,
    K_list: list[np.ndarray],
    extrinsics: list[tuple[np.ndarray, np.ndarray]],
    min_conf: float = 0.5,
    reproj_threshold_px: float = 5.0,
    min_aspect_ratio: float = 0.0,
    f_scale: float = 10.0,
    dist_coeffs: list[np.ndarray] | None = None,
    geometry_config: GeometryConfig | None = None,
    image_sizes: list[tuple[int, int]] | np.ndarray | None = None,
    min_inliers: int = 2,
    hysteresis_margin_px: float = 0.0,
    fps: float | None = None,
) -> TriangulationDiagnostics:
    """
    Triangulate a 3D skeleton sequence and retain geometric diagnostics.

    Returns points in the Y-up internal coordinate space. Diagnostic reprojection
    errors and inlier masks are indexed as (frame, joint, camera).
    """
    num_frames, num_cameras, num_kpts, _ = keypoints.shape
    if len(K_list) != num_cameras or len(extrinsics) != num_cameras:
        raise ValueError("camera calibration count must match keypoints camera dimension")
    if dist_coeffs is not None and len(dist_coeffs) != num_cameras:
        raise ValueError("dist_coeffs count must match keypoints camera dimension")
    cfg = geometry_config or GeometryConfig(
        min_detection_conf=min_conf,
        max_reprojection_error_px=reproj_threshold_px,
        hard_reprojection_error_px=max(12.0, reproj_threshold_px),
        min_ray_angle_deg=0.0,
        min_depth_m=0.0,
        f_scale=f_scale,
        min_inliers=min_inliers,
        hysteresis_margin_px=hysteresis_margin_px,
    )
    lr_swap_mask = np.zeros(
        (num_frames, num_cameras, len(LR_SWAP_GROUPS)), dtype=bool
    )
    if cfg.lr_swap_fix:
        keypoints, scores, lr_swap_mask = repair_lr_swaps(
            keypoints,
            scores,
            K_list,
            extrinsics,
            cfg,
            dist_coeffs=dist_coeffs,
            image_sizes=image_sizes,
        )
    if cfg.camera_policy == "soft":
        diagnostics = _triangulate_sequence_soft(
            keypoints,
            scores,
            K_list,
            extrinsics,
            cfg=cfg,
            dist_coeffs=dist_coeffs,
            image_sizes=image_sizes,
            lr_swap_mask=lr_swap_mask,
        )
        diagnostics = _apply_temporal_filter(diagnostics, cfg, fps)
        if cfg.confidence3d:
            diagnostics = _attach_confidence3d(diagnostics, keypoints, scores)
        return diagnostics
    part_confidence = [cfg.confidence_for_joint(k) for k in range(num_kpts)]
    reproj_threshold_px = cfg.max_reprojection_error_px
    f_scale = cfg.f_scale
    # Compute projection matrices P = K[R|t] for all cameras
    P_list = [K_list[c] @ np.hstack(extrinsics[c]) for c in range(num_cameras)]
    camera_models = None
    if dist_coeffs is not None:
        camera_models = [
            (np.asarray(K_list[c], dtype=np.float64),
             np.asarray(extrinsics[c][0], dtype=np.float64),
             np.asarray(extrinsics[c][1], dtype=np.float64),
             np.asarray(dist_coeffs[c], dtype=np.float64))
            for c in range(num_cameras)
        ]
    
    pts3d = np.full((num_frames, num_kpts, 3), np.nan, dtype=np.float64)
    candidate_pts3d = np.full((num_frames, num_kpts, 3), np.nan, dtype=np.float64)
    confidence = np.zeros((num_frames, num_kpts), dtype=np.float32)
    candidate_inlier_mask = np.zeros((num_frames, num_kpts, num_cameras), dtype=bool)
    inlier_mask = np.zeros((num_frames, num_kpts, num_cameras), dtype=bool)
    # Legacy trust is the hard mask itself: consumers that read `camera_trust`
    # get the same 0/1 decision the legacy solver made.
    camera_trust = np.zeros((num_frames, num_kpts, num_cameras), dtype=np.float32)
    support_mask = inlier_mask
    reprojection_error_px = np.full((num_frames, num_kpts, num_cameras), np.nan, dtype=np.float32)
    candidate_num_inliers = np.zeros((num_frames, num_kpts), dtype=np.uint8)
    num_inliers = np.zeros((num_frames, num_kpts), dtype=np.uint8)
    ray_angle_deg = np.zeros((num_frames, num_kpts), dtype=np.float32)
    observed_camera_count = np.zeros((num_frames, num_kpts), dtype=np.uint8)
    positive_depth_mask = np.zeros((num_frames, num_kpts, num_cameras), dtype=bool)
    observation_provenance = np.full(
        (num_frames, num_kpts), "unobserved", dtype="<U32"
    )
    camera_rejection_reason = np.full(
        (num_frames, num_kpts, num_cameras), "not_observed", dtype="<U32"
    )
    
    C_list = [-extrinsics[c][0].T @ extrinsics[c][1] for c in range(num_cameras)]
    low_quality_mask = np.zeros(num_frames, dtype=bool)
    prev_inlier_sets: list[set[int]] = [set() for _ in range(num_kpts)]
    
    for f in range(num_frames):
        # Camera-wide aspect-ratio rejection is intentionally retired.  The
        # argument remains for source compatibility, but observation quality
        # is decided per joint below so a squat/T-pose cannot erase a camera.
        valid_cam_for_frame = list(range(num_cameras))
                
        # Compute baseline angle of surviving cameras to set diagnostic low_quality_mask
        # Subject sits near the origin, so a fixed target column approximates the shared view center.
        max_angle = _max_pairwise_angle(valid_cam_for_frame, np.zeros((3, 1)), C_list)
        if max_angle < 35.0:
            low_quality_mask[f] = True

        for k in range(num_kpts):
            part = part_confidence[k]
            valid_pts2d = []
            valid_observed_pts2d = []
            valid_P = []
            valid_conf = []
            valid_weight = []
            valid_cam = []
            valid_models = []
            
            for c in valid_cam_for_frame:
                score = float(scores[f, c, k])
                if not np.isfinite(score) or score < part.min_conf:
                    camera_rejection_reason[f, k, c] = "low_confidence"
                    continue
                if not np.isfinite(keypoints[f, c, k]).all():
                    camera_rejection_reason[f, k, c] = "nonfinite_2d"
                    continue
                if image_sizes is not None:
                    width, height = image_sizes[c]
                    x, y = keypoints[f, c, k]
                    if not (0.0 <= x < float(width) and 0.0 <= y < float(height)):
                        camera_rejection_reason[f, k, c] = "outside_image"
                        continue
                observed_point = np.asarray(keypoints[f, c, k], dtype=np.float64)
                point = observed_point.copy()
                valid_observed_pts2d.append(observed_point)
                if camera_models is not None:
                    K, _, _, dist = camera_models[c]
                    point = cv2.undistortPoints(
                        point.reshape(1, 1, 2), K, dist, P=K
                    ).reshape(2)
                    valid_models.append(camera_models[c])
                valid_pts2d.append(point)
                valid_P.append(P_list[c])
                valid_conf.append(score)
                valid_weight.append(part.weight(score))
                valid_cam.append(c)

            observed_camera_count[f, k] = np.uint8(len(valid_pts2d))
            if len(valid_pts2d) == 0:
                continue
            if len(valid_pts2d) < 2:
                observation_provenance[f, k] = "one_camera_only"
                    
                    
            if len(valid_pts2d) >= 2:
                valid_pts2d_arr = np.array(valid_pts2d, dtype=np.float64)
                valid_observed_pts2d_arr = np.array(valid_observed_pts2d, dtype=np.float64)
                valid_conf_arr = np.array(valid_conf, dtype=np.float64)
                valid_weight_arr = np.array(valid_weight, dtype=np.float64)
                valid_cam_arr = np.array(valid_cam, dtype=np.int32)

                prev_p = pts3d[f - 1, k] if f >= 1 and np.all(np.isfinite(pts3d[f - 1, k])) else None
                prev_v = (
                    (pts3d[f - 1, k] - pts3d[f - 2, k])
                    if f >= 2 and np.all(np.isfinite(pts3d[f - 1, k])) and np.all(np.isfinite(pts3d[f - 2, k]))
                    else None
                )

                # Connected joint for anatomical bone sanity.  The nominal
                # lengths are centimetres (adult anthropometry); the engine
                # converts them to metres in _temporal_anatomy_penalties.
                conn_p = None
                ref_len_cm = None
                LIMB_CONNECTIONS = {
                    12: (14, 39.5),  # hip_r -> knee_r
                    14: (16, 37.5),  # knee_r -> ankle_r
                    16: (14, 37.5),  # ankle_r -> knee_r
                    11: (13, 40.0),  # hip_l -> knee_l
                    13: (15, 41.6),  # knee_l -> ankle_l
                    15: (13, 41.6),  # ankle_l -> knee_l
                    8: (6, 30.0),    # elbow_r -> shoulder_r
                    10: (8, 26.4),   # wrist_r -> elbow_r
                    7: (5, 29.4),    # elbow_l -> shoulder_l
                    9: (7, 25.9),    # wrist_l -> elbow_l
                }
                if k in LIMB_CONNECTIONS:
                    parent_k, nominal_len = LIMB_CONNECTIONS[k]
                    if np.all(np.isfinite(pts3d[f, parent_k])):
                        conn_p = pts3d[f, parent_k]
                        ref_len_cm = nominal_len
                    elif f >= 1 and np.all(np.isfinite(pts3d[f - 1, parent_k])):
                        conn_p = pts3d[f - 1, parent_k]
                        ref_len_cm = nominal_len

                if len(valid_pts2d_arr) >= 3 or (cfg.w_temporal > 0.0 or cfg.w_geom > 0.0 or cfg.w_anat > 0.0):
                    pt, inliers_local, err_local, conf3d = _triangulate_with_inlier_selection(
                        valid_pts2d_arr,
                        valid_P,
                        valid_conf_arr,
                        valid_cam_arr,
                        reproj_threshold_px,
                        camera_models=valid_models if camera_models is not None else None,
                        min_depth=cfg.min_depth_m,
                        residual_pts2d=valid_observed_pts2d_arr,
                        hard_reproj_threshold_px=cfg.hard_reprojection_error_px,
                        prev_inliers=prev_inlier_sets[k],
                        hysteresis_margin_px=cfg.hysteresis_margin_px,
                        camera_centers=C_list,
                        prev_point3d=prev_p,
                        prev_velocity3d=prev_v,
                        connected_point3d=conn_p,
                        ref_bone_length_cm=ref_len_cm,
                        w_temporal=cfg.w_temporal,
                        w_geom=cfg.w_geom,
                        w_anat=cfg.w_anat,
                        legacy_units=cfg.legacy_unit_behaviour,
                        weights=valid_weight_arr if part.gamma != 1.0 else None,
                    )
                    candidate_inlier_mask[f, k, valid_cam_arr] = inliers_local
                    inlier_mask[f, k, valid_cam_arr] = inliers_local
                    camera_trust[f, k, valid_cam_arr] = inliers_local.astype(np.float32)
                    reprojection_error_px[f, k, valid_cam_arr] = err_local.astype(np.float32)
                    confidence[f, k] = np.float32(conf3d)
                else:
                    pt = triangulate_weighted_dlt(
                        valid_pts2d_arr,
                        valid_P,
                        valid_weight_arr if part.gamma != 1.0 else valid_conf_arr,
                    )
                    err_local = _reprojection_errors(
                        pt, valid_observed_pts2d_arr, valid_P,
                        camera_models=valid_models if camera_models is not None else None,
                        min_depth=cfg.min_depth_m,
                    )
                    reprojection_error_px[f, k, valid_cam_arr] = err_local.astype(np.float32)
                    candidate_inlier_mask[f, k, valid_cam_arr] = (
                        np.isfinite(err_local) & (err_local <= cfg.hard_reprojection_error_px)
                    )
                    inlier_mask[f, k, valid_cam_arr] = candidate_inlier_mask[f, k, valid_cam_arr]
                    camera_trust[f, k, valid_cam_arr] = inlier_mask[f, k, valid_cam_arr].astype(np.float32)
                    med_err = float(np.nanmedian(np.where(np.isfinite(err_local), err_local, np.nan)))
                    confidence[f, k] = np.float32(
                        np.mean(valid_conf_arr) / (1.0 + max(0.0, med_err) / max(1.0, reproj_threshold_px))
                    )
                num_inliers[f, k] = np.uint8(inlier_mask[f, k].sum())
                prev_inlier_sets[k] = set(np.flatnonzero(inlier_mask[f, k]))
                if np.isfinite(pt).all():
                    candidate_pts3d[f, k] = pt
                    candidate_num_inliers[f, k] = np.uint8(candidate_inlier_mask[f, k].sum())

                min_inliers_req = cfg.min_inliers
                if num_inliers[f, k] < min_inliers_req:
                    pts3d[f, k] = np.nan
                    confidence[f, k] = 0.0
                    observation_provenance[f, k] = (
                        "geometry_rejected" if num_inliers[f, k] < 2 else "below_min_inliers"
                    )
                else:
                    pts3d[f, k] = pt
                    observation_provenance[f, k] = (
                        "observed_2plus" if num_inliers[f, k] == observed_camera_count[f, k]
                        else "observed_with_rejected_views"
                    )

                if np.isfinite(pt).all():
                    for c in valid_cam_arr:
                        R_c, t_c = extrinsics[c]
                        depth = float((np.asarray(R_c) @ pt.reshape(3) + np.asarray(t_c).reshape(3))[2])
                        positive_depth_mask[f, k, c] = depth > cfg.min_depth_m
                    for local, c in enumerate(valid_cam_arr):
                        if not positive_depth_mask[f, k, c]:
                            camera_rejection_reason[f, k, c] = "nonpositive_depth"
                        elif not np.isfinite(err_local[local]):
                            camera_rejection_reason[f, k, c] = "invalid_projection"
                        elif err_local[local] > reproj_threshold_px:
                            camera_rejection_reason[f, k, c] = "high_reprojection"
                        elif inlier_mask[f, k, c]:
                            camera_rejection_reason[f, k, c] = "inlier"

                # Compute ray angle
                inlier_cams = np.flatnonzero(inlier_mask[f, k])
                if len(inlier_cams) >= 2:
                    max_ang = _max_pairwise_angle(inlier_cams, pt.reshape((3, 1)), C_list)
                    ray_angle_deg[f, k] = np.float32(max_ang)
                    if max_ang < cfg.min_ray_angle_deg:
                        pts3d[f, k] = np.nan
                        confidence[f, k] = 0.0
                        inlier_mask[f, k] = False
                        num_inliers[f, k] = 0
                        observation_provenance[f, k] = "low_ray_angle"
                        for c in inlier_cams:
                            camera_rejection_reason[f, k, c] = "low_ray_angle"
                
    # Convert from OpenCV (Y-down) to internal (Y-up) space
    pts3d_internal = opencv_to_internal(pts3d)
    candidate_pts3d_internal = opencv_to_internal(candidate_pts3d)
    
    diagnostics = TriangulationDiagnostics(
        points3d=pts3d_internal,
        candidate_points3d=candidate_pts3d_internal,
        confidence=confidence,
        candidate_inlier_mask=candidate_inlier_mask,
        inlier_mask=inlier_mask,
        support_mask=support_mask,
        camera_trust=camera_trust,
        reprojection_error_px=reprojection_error_px,
        candidate_num_inliers=candidate_num_inliers,
        num_inliers=num_inliers,
        low_quality_mask=low_quality_mask,
        ray_angle_deg=ray_angle_deg,
        observed_camera_count=observed_camera_count,
        positive_depth_mask=positive_depth_mask,
        observation_provenance=observation_provenance,
        camera_rejection_reason=camera_rejection_reason,
        lr_swap_mask=lr_swap_mask,
    )
    return _apply_temporal_filter(diagnostics, cfg, fps)
