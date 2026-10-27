"""Deterministic rigid-segment conditioning for solver target tracks.

This module contains the promoted C3h target operation.  It is intentionally
NumPy-only and uses centimetre-space float64 arithmetic throughout:

* pairwise median distances and classical MDS estimate a segment template;
* the MDS template and its single mirror are compared using proper Kabsch;
* two proper Procrustes-mean iterations refine the selected template;
* a robust exhaustive three-point Kabsch fit, followed by inlier refit,
  accepts a frame only with at least three inliers and RMS <= ``TAU_FILL_CM``;
* accepted frames replace every segment point with the rigid prediction;
  rejected frames are left unchanged (fail closed).

Segment membership is declarative.  The default table is the P1 face and hand
set, but callers may pass another mapping of segment names to track names.
"""
from __future__ import annotations

import itertools
from typing import Any, Mapping, Sequence

import numpy as np

TAU_FILL_CM = 1.0
RIGIDITY_SIGMA_LIMIT_CM = 0.5 * TAU_FILL_CM
MIN_INLIERS = 3
PROCRUSTES_ITERATIONS = 2

# Names, rather than indices, are the production contract.  Keep insertion
# order stable because diagnostics and tie-breaking are part of reproducibility.
RIGID_SEGMENTS: dict[str, tuple[str, ...]] = {
    "head_face": ("nose", "left_eye", "right_eye", "left_ear", "right_ear"),
    "hand_l": (
        "hand_wrist_l", "hand_index_mcp_l", "hand_middle_mcp_l",
        "hand_ring_mcp_l", "hand_pinky_mcp_l",
    ),
    "hand_r": (
        "hand_wrist_r", "hand_index_mcp_r", "hand_middle_mcp_r",
        "hand_ring_mcp_r", "hand_pinky_mcp_r",
    ),
}


def _copy_tracks(tracks: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {str(name): np.asarray(value, dtype=np.float64).copy() for name, value in tracks.items()}


def _valid_track_map(tracks: Mapping[str, np.ndarray]) -> tuple[int | None, str | None]:
    lengths: set[int] = set()
    for name, value in tracks.items():
        arr = np.asarray(value)
        if arr.ndim != 2 or arr.shape[1] != 3:
            return None, f"track {name!r} must have shape (F, 3), got {arr.shape}"
        lengths.add(int(arr.shape[0]))
    if len(lengths) > 1:
        return None, "all tracks must have the same frame count"
    return (next(iter(lengths)) if lengths else 0), None


def _kabsch(template: np.ndarray, observed: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Return row-vector proper rotation and translation, template -> observed."""
    if template.ndim != 2 or observed.shape != template.shape or len(template) < MIN_INLIERS:
        return None
    if not np.isfinite(template).all() or not np.isfinite(observed).all():
        return None
    x_mean = np.mean(template, axis=0)
    y_mean = np.mean(observed, axis=0)
    x = template - x_mean
    y = observed - y_mean
    try:
        u, _s, vt = np.linalg.svd(x.T @ y)
    except np.linalg.LinAlgError:
        return None
    rotation = u @ vt
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    translation = y_mean - x_mean @ rotation
    return rotation, translation


def _apply_fit(template: np.ndarray, fit: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    rotation, translation = fit
    return template @ rotation + translation


def _classical_mds(distances: np.ndarray) -> np.ndarray:
    count = int(distances.shape[0])
    centering = np.eye(count, dtype=np.float64) - np.ones((count, count), dtype=np.float64) / count
    gram = -0.5 * centering @ (distances * distances) @ centering
    gram = 0.5 * (gram + gram.T)
    values, vectors = np.linalg.eigh(gram)
    order = np.argsort(values)[::-1]
    values = values[order]
    vectors = vectors[:, order]
    positive = np.maximum(values[:3], 0.0)
    coords = vectors[:, :3] * np.sqrt(positive)[None, :]
    coords -= np.mean(coords, axis=0, keepdims=True)
    return coords


def _estimate_template(tracks: Mapping[str, np.ndarray], names: Sequence[str]) -> tuple[np.ndarray, dict[str, Any]] | None:
    points = np.stack([np.asarray(tracks[name], dtype=np.float64) for name in names], axis=1)
    all_finite = np.isfinite(points).all(axis=(1, 2))
    if not np.any(all_finite):
        return None
    full = points[all_finite]
    count = len(names)
    distances = np.zeros((count, count), dtype=np.float64)
    pair_counts: dict[str, int] = {}
    for i in range(count):
        for j in range(i + 1, count):
            values = np.linalg.norm(full[:, i] - full[:, j], axis=1)
            distances[i, j] = distances[j, i] = float(np.median(values))
            pair_counts[f"{names[i]}|{names[j]}"] = int(len(values))
    template = _classical_mds(distances)
    return template, {
        "method": "pairwise_median_distances_classical_mds",
        "all_finite_frames": int(len(full)),
        "pairwise_median_distances_cm": distances.tolist(),
        "pair_counts": pair_counts,
    }


def _median_residual(template: np.ndarray, points: np.ndarray) -> float:
    residuals: list[float] = []
    for frame in points:
        fit = _kabsch(template, frame)
        if fit is None:
            continue
        prediction = _apply_fit(template, fit)
        residuals.extend(np.linalg.norm(frame - prediction, axis=1).tolist())
    values = np.asarray(residuals, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if len(values) else float("inf")


def _refine_template(template: np.ndarray, points: np.ndarray) -> np.ndarray:
    current = np.asarray(template, dtype=np.float64).copy()
    for _ in range(PROCRUSTES_ITERATIONS):
        aligned: list[np.ndarray] = []
        for frame in points:
            fit = _kabsch(current, frame)
            if fit is None:
                continue
            rotation, translation = fit
            aligned.append((frame - translation) @ rotation.T)
        if not aligned:
            break
        current = np.mean(np.stack(aligned, axis=0), axis=0)
        current -= np.mean(current, axis=0, keepdims=True)
    return current


def build_template(tracks: Mapping[str, np.ndarray], names: Sequence[str]) -> tuple[np.ndarray, dict[str, Any]] | None:
    """Build a chirality-selected, two-pass Procrustes-refined template."""
    estimated = _estimate_template(tracks, names)
    if estimated is None:
        return None
    mds, mds_diag = estimated
    points = np.stack([np.asarray(tracks[name], dtype=np.float64) for name in names], axis=1)
    full = points[np.isfinite(points).all(axis=(1, 2))]
    mirror = mds * np.asarray([1.0, 1.0, -1.0], dtype=np.float64)[None, :]
    medians = (_median_residual(mds, full), _median_residual(mirror, full))
    selected_index = int(np.argmin(np.asarray(medians, dtype=np.float64)))
    selected = (mds if selected_index == 0 else mirror).copy()
    before_refine = selected.copy()
    selected = _refine_template(selected, full)
    return selected, {
        "method": "classical_mds_plus_chirality_selection_plus_proper_procrustes_mean",
        "mirror_transform": "T @ diag(1,1,-1)",
        "all_finite_frames": int(len(full)),
        "candidate_median_residual_cm": {"mds": float(medians[0]), "mirror": float(medians[1])},
        "selected_chirality": "mds" if selected_index == 0 else "mirror",
        "procrustes_mean_iterations": PROCRUSTES_ITERATIONS,
        "selected_template_before_refine": before_refine.tolist(),
        "mds_diagnostics": mds_diag,
    }


def _best_hypothesis(template: np.ndarray, observed: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray], np.ndarray] | None:
    indices = np.flatnonzero(valid)
    if len(indices) < MIN_INLIERS:
        return None
    best: tuple[int, float, tuple[int, ...], tuple[np.ndarray, np.ndarray], np.ndarray] | None = None
    for combo in itertools.combinations(indices.tolist(), MIN_INLIERS):
        combo_array = np.asarray(combo, dtype=int)
        fit = _kabsch(template[combo_array], observed[combo_array])
        if fit is None:
            continue
        prediction = _apply_fit(template, fit)
        residual = np.linalg.norm(observed - prediction, axis=1)
        inliers = valid & np.isfinite(residual) & (residual < TAU_FILL_CM)
        count = int(np.sum(inliers))
        score_sum = float(np.sum(residual[inliers])) if count else float("inf")
        candidate = (count, -score_sum, tuple(combo), fit, inliers)
        if best is None or candidate[:3] > best[:3]:
            best = candidate
    if best is None:
        return None
    _count, _negative_sum, _combo, _initial_fit, initial_inliers = best
    if int(np.sum(initial_inliers)) < MIN_INLIERS:
        return None
    refit = _kabsch(template[initial_inliers], observed[initial_inliers])
    if refit is None:
        return None
    prediction = _apply_fit(template, refit)
    residual = np.linalg.norm(observed - prediction, axis=1)
    final_inliers = valid & np.isfinite(residual) & (residual < TAU_FILL_CM)
    if int(np.sum(final_inliers)) < MIN_INLIERS:
        final_inliers = initial_inliers
    return prediction, refit, final_inliers


def _sigma(tracks: Mapping[str, np.ndarray], names: Sequence[str], template: np.ndarray) -> tuple[float, int, int]:
    points = np.stack([np.asarray(tracks[name], dtype=np.float64) for name in names], axis=1)
    all_finite = np.isfinite(points).all(axis=(1, 2))
    residuals: list[float] = []
    fitted = 0
    for frame in points[all_finite]:
        fit = _kabsch(template, frame)
        if fit is None:
            continue
        fitted += 1
        residuals.extend(np.linalg.norm(frame - _apply_fit(template, fit), axis=1).tolist())
    values = np.asarray(residuals, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return 0.0, fitted, 0
    centre = float(np.median(values))
    return 1.4826 * float(np.median(np.abs(values - centre))), fitted, int(len(values))


def rigid_denoise_targets(
    tracks: Mapping[str, np.ndarray],
    segments: Mapping[str, Sequence[str]] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Apply C3h to named segments and return ``(tracks, diagnostics)``.

    Invalid input shapes, inconsistent frame counts, missing segment names, or
    segments with no all-finite template frame are handled fail-closed: the
    returned tracks are an unchanged float64 copy and diagnostics identify the
    reason.  A valid segment whose rigidity gate fails is likewise unchanged.
    """
    out = _copy_tracks(tracks)
    frame_count, error = _valid_track_map(out)
    selected = RIGID_SEGMENTS if segments is None else segments
    diagnostics: dict[str, Any] = {
        "enabled": error is None,
        "method": "C3h",
        "tau_fill_cm": TAU_FILL_CM,
        "rigidity_sigma_limit_cm": RIGIDITY_SIGMA_LIMIT_CM,
        "segments": {},
    }
    if error is not None:
        diagnostics.update({"enabled": False, "reason": error})
        return out, diagnostics
    for segment, declared_names in selected.items():
        names = tuple(str(name) for name in declared_names)
        missing = [name for name in names if name not in out]
        row: dict[str, Any] = {"track_names": list(names), "frames_total": int(frame_count or 0)}
        if len(names) < MIN_INLIERS or missing:
            row.update({"enabled": False, "reason": "missing_or_too_short_segment", "missing_tracks": missing})
            diagnostics["segments"][str(segment)] = row
            continue
        built = build_template(out, names)
        if built is None:
            row.update({"enabled": False, "reason": "no_all_finite_template_frame"})
            diagnostics["segments"][str(segment)] = row
            continue
        template, template_diag = built
        sigma, fitted_frames, residual_count = _sigma(out, names, template)
        gate_pass = bool(sigma < RIGIDITY_SIGMA_LIMIT_CM)
        accepted = rejected = replacements = 0
        rms_values: list[float] = []
        points = np.stack([out[name] for name in names], axis=1)
        if gate_pass:
            for frame_index, frame in enumerate(points):
                valid = np.isfinite(frame).all(axis=1)
                result = _best_hypothesis(template, frame, valid)
                if result is None:
                    rejected += 1
                    continue
                prediction, _fit, _inliers = result
                residual = np.linalg.norm(frame - prediction, axis=1)
                inliers = valid & np.isfinite(residual) & (residual < TAU_FILL_CM)
                if int(np.sum(inliers)) < MIN_INLIERS:
                    rejected += 1
                    continue
                fit_rms = float(np.sqrt(np.mean(residual[inliers] ** 2)))
                if not np.isfinite(fit_rms) or fit_rms > TAU_FILL_CM:
                    rejected += 1
                    continue
                for point_index, name in enumerate(names):
                    if np.isfinite(prediction[point_index]).all():
                        out[name][frame_index] = prediction[point_index]
                        replacements += 1
                rms_values.append(fit_rms)
                accepted += 1
        else:
            rejected = int(frame_count or 0)
        row.update({
            "enabled": True,
            "template": template.tolist(),
            "template_diagnostics": template_diag,
            "rigidity_sigma_cm": float(sigma),
            "rigidity_gate_pass": gate_pass,
            "fitted_frames": int(fitted_frames),
            "residual_count": int(residual_count),
            "frames_rigid_consistent": int(accepted),
            "frames_consistency_rejected": int(rejected),
            "all_point_replacements": int(replacements),
            "fit_rms_p95_cm": float(np.percentile(rms_values, 95)) if rms_values else None,
        })
        diagnostics["segments"][str(segment)] = row
    return out, diagnostics


# Descriptive aliases for callers and tests.
process_rigid_segments = rigid_denoise_targets
apply_rigid_segments = rigid_denoise_targets

__all__ = [
    "MIN_INLIERS", "PROCRUSTES_ITERATIONS", "RIGID_SEGMENTS",
    "RIGIDITY_SIGMA_LIMIT_CM", "TAU_FILL_CM", "build_template",
    "rigid_denoise_targets", "process_rigid_segments", "apply_rigid_segments",
]
