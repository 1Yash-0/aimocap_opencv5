"""Robust observation conditioning for solver-consumed 3-D target tracks.

The implementation is deliberately self-contained: it uses NumPy only, keeps all
arithmetic in float64, and has no dependency on evaluation or production solver
modules.  Tracks are smoothed with a constant-acceleration Kalman model driven
by continuous white-jerk noise.  Student-t IRLS protects the state estimate from
isolated triangulation failures; a windowed innovation test increases process
noise during genuine maneuvers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ConditioningConfig:
    """Configuration for :func:`condition_target_tracks`.

    Units are centimetres, seconds, and radians are not used by this module.
    ``q_grid`` is the white-jerk spectral-density grid in cm²/s⁵.
    """

    window: int = 9
    min_quasi_static_frames: int = 30
    quasi_static_speed_cm_s: float = 10.0
    sigma_floor_cm: float = 0.05
    irls_passes: int = 3
    student_t_nu: float = 4.0
    max_bridge_frames: int | None = None
    q_grid: tuple[float, ...] = field(
        default_factory=lambda: tuple(np.logspace(0.0, 9.0, 40).tolist())
    )
    maneuver_window: int = 5
    maneuver_quantile_z: float = 2.3263478740408408  # chi-square 99% normal quantile
    maneuver_max_factor: float = 100.0
    local_window_seconds: float = 1.0
    local_q_hop_frames: int = 3
    local_log_smooth_seconds: float = 0.3
    maneuver_boost_enabled: bool = True


_F = np.ndarray


def _as_config(config: ConditioningConfig | None) -> ConditioningConfig:
    return config if config is not None else ConditioningConfig()


def _validate_fps(fps: float) -> float:
    value = float(fps)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"fps must be finite and positive, got {fps!r}")
    return value


def _jerk_transition(dt: float) -> np.ndarray:
    return np.array(
        [[1.0, dt, 0.5 * dt * dt], [0.0, 1.0, dt], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _jerk_process(dt: float, q: float) -> np.ndarray:
    # Continuous white jerk, standard constant-acceleration discretization.
    return float(q) * np.array(
        [
            [dt**5 / 20.0, dt**4 / 8.0, dt**3 / 6.0],
            [dt**4 / 8.0, dt**3 / 3.0, dt**2 / 2.0],
            [dt**3 / 6.0, dt**2 / 2.0, dt],
        ],
        dtype=np.float64,
    )


def _sym_psd(matrix: np.ndarray, floor: float = 1e-12) -> np.ndarray:
    matrix = (matrix + matrix.T) * 0.5
    values, vectors = np.linalg.eigh(matrix)
    return (vectors * np.maximum(values, floor)) @ vectors.T


def _sym_covariance(matrix: np.ndarray, floor: float = 1e-12) -> np.ndarray:
    """Fast covariance symmetrization for the forward recursion.

    The Kalman update starts PSD and the Joseph-equivalent scalar-position
    update below preserves that property up to roundoff.  Full eigendecomposition
    is reserved for the much smaller RTS pass; the q-grid likelihood evaluates
    this path many times.
    """
    matrix = (matrix + matrix.T) * 0.5
    diagonal = np.maximum(np.diag(matrix), floor)
    return matrix + np.diag(diagonal - np.diag(matrix))


def _sym_psd_batch(matrix: np.ndarray, floor: float = 1e-12) -> np.ndarray:
    """Fast batched covariance symmetrization for the RTS recursion."""
    matrix = (matrix + np.swapaxes(matrix, -1, -2)) * 0.5
    diagonal = np.maximum(np.diagonal(matrix, axis1=-2, axis2=-1), floor)
    matrix[..., range(3), range(3)] = diagonal
    return matrix


def _local_quadratic_residuals(values: np.ndarray, fps: float, config: ConditioningConfig):
    """Return LOO residuals and local fitted speed for a track.

    The centre sample is omitted from every local fit.  The centre prediction
    variance factor ``1 + x0 (X'X)^-1 x0`` is the analytic LOO correction;
    dividing the raw prediction residual by its square root avoids interpreting
    extrapolation uncertainty as measurement noise.
    """
    x = np.asarray(values, dtype=np.float64)
    n = len(x)
    half = max(1, int(config.window) // 2)
    residual = np.full((n, 3), np.nan, dtype=np.float64)
    speed = np.full(n, np.nan, dtype=np.float64)
    offsets = np.arange(-half, half + 1, dtype=np.float64)
    offsets = offsets[offsets != 0.0]
    design_all = np.column_stack((np.ones(len(offsets)), offsets, offsets * offsets))

    # The frame loop is intentional: each fit has a different finite mask and
    # the model is tiny; all three spatial axes are solved in one least-squares
    # operation when their masks agree.
    for frame in range(n):
        left = max(0, frame - half)
        right = min(n, frame + half + 1)
        local_offsets = np.arange(left - frame, right - frame, dtype=np.float64)
        keep = local_offsets != 0.0
        local_offsets = local_offsets[keep]
        if len(local_offsets) < 3:
            continue
        y = x[left:right][keep]
        finite = np.isfinite(y).all(axis=1)
        if int(finite.sum()) < 3:
            continue
        d = local_offsets[finite]
        y = y[finite]
        design = np.column_stack((np.ones(len(d)), d, d * d))
        try:
            gram_inv = np.linalg.pinv(design.T @ design, rcond=1e-12)
            beta = gram_inv @ design.T @ y
        except np.linalg.LinAlgError:
            continue
        prediction = beta[0]
        leverage = float(np.array([1.0, 0.0, 0.0]) @ gram_inv @ np.array([1.0, 0.0, 0.0]))
        correction = np.sqrt(max(1.0 + leverage, 1.0))
        if np.isfinite(x[frame]).all():
            residual[frame] = (x[frame] - prediction) / correction
        speed[frame] = float(np.linalg.norm(beta[1]) * fps)
    return residual, speed


def _estimate_sigma(values: np.ndarray, fps: float, config: ConditioningConfig):
    residual, local_speed = _local_quadratic_residuals(values, fps, config)
    valid = np.isfinite(residual).all(axis=1)
    quasi = valid & np.isfinite(local_speed) & (local_speed < config.quasi_static_speed_cm_s)
    use = quasi if int(quasi.sum()) >= config.min_quasi_static_frames else valid
    samples = residual[use].reshape(-1)
    samples = samples[np.isfinite(samples)]
    if samples.size:
        centre = float(np.median(samples))
        sigma = 1.4826 * float(np.median(np.abs(samples - centre)))
        if sigma <= 1e-12:
            sigma = float(np.sqrt(np.mean((samples - centre) ** 2)))
    else:
        finite = values[np.isfinite(values).all(axis=1)]
        sigma = float(np.std(finite)) if finite.size else config.sigma_floor_cm
    sigma = max(float(config.sigma_floor_cm), sigma)
    return sigma, residual, local_speed, int(quasi.sum()), int(valid.sum())


def _initial_states(values: np.ndarray):
    n = len(values)
    first = np.zeros(3, dtype=np.int64)
    initial = np.zeros(3, dtype=np.float64)
    available = np.zeros(3, dtype=bool)
    for axis in range(3):
        where = np.flatnonzero(np.isfinite(values[:, axis]))
        if len(where):
            first[axis] = int(where[0])
            initial[axis] = float(values[where[0], axis])
            available[axis] = True
    return first, initial, available


def _filter(
    values: np.ndarray,
    fps: float,
    q: float,
    measurement_variance: np.ndarray,
    q_scale: np.ndarray | None = None,
):
    """Forward filter, vectorized over axes and retaining innovation traces."""
    y = np.asarray(values, dtype=np.float64)
    n = len(y)
    dt = 1.0 / fps
    transition = _jerk_transition(dt)
    q_scale = np.ones(n, dtype=np.float64) if q_scale is None else np.asarray(q_scale, dtype=np.float64)
    x_pred = np.zeros((n, 3, 3), dtype=np.float64)
    p_pred = np.zeros((n, 3, 3, 3), dtype=np.float64)
    x_filt = np.zeros_like(x_pred)
    p_filt = np.zeros_like(p_pred)
    innovations = np.full((n, 3), np.nan, dtype=np.float64)
    innovation_var = np.full((n, 3), np.nan, dtype=np.float64)
    _, initial, available = _initial_states(y)
    if not available.any():
        return x_pred, p_pred, x_filt, p_filt, innovations, innovation_var

    state = np.zeros((3, 3), dtype=np.float64)
    state[:, 0] = initial
    covariance = np.zeros((3, 3, 3), dtype=np.float64)
    for axis in range(3):
        covariance[axis] = np.diag([1e6, 1e6, 1e6])
    for frame in range(n):
        if frame:
            q_matrix = _jerk_process(dt, q * max(float(q_scale[frame]), 1e-12))
            state = state @ transition.T
            covariance = transition @ covariance @ transition.T + q_matrix
            covariance = (covariance + np.swapaxes(covariance, 1, 2)) * 0.5
            diagonal = np.maximum(np.diagonal(covariance, axis1=1, axis2=2), 1e-12)
            covariance[:, range(3), range(3)] = diagonal
        x_pred[frame] = state
        p_pred[frame] = covariance
        observed = np.isfinite(y[frame]) & available
        variance = max(float(measurement_variance[frame]), 1e-12)
        innovation = np.where(observed, y[frame] - state[:, 0], 0.0)
        innovation_variance = np.maximum(covariance[:, 0, 0] + variance, 1e-12)
        gain = covariance[:, :, 0] / innovation_variance[:, None]
        update = observed.astype(np.float64)
        state = state + gain * (innovation * update)[:, None]
        covariance = covariance - gain[:, :, None] * covariance[:, 0, :][:, None, :] * update[:, None, None]
        covariance = (covariance + np.swapaxes(covariance, 1, 2)) * 0.5
        diagonal = np.maximum(np.diagonal(covariance, axis1=1, axis2=2), 1e-12)
        covariance[:, range(3), range(3)] = diagonal
        innovations[frame, observed] = innovation[observed]
        innovation_var[frame, observed] = innovation_variance[observed]
        x_filt[frame] = state
        p_filt[frame] = covariance
    return x_pred, p_pred, x_filt, p_filt, innovations, innovation_var




def _filter_grid(
    values: np.ndarray,
    fps: float,
    q_grid: np.ndarray,
    measurement_variance: np.ndarray,
):
    """Forward-filter all q candidates together for likelihood selection.

    This is algebraically the same recursion as :func:`_filter`, with a
    leading candidate dimension.  Keeping the time recursion in one loop
    removes the Python loop over the q grid while retaining float64 state and
    the scalar-position observation update.
    """
    y = np.asarray(values, dtype=np.float64)
    candidates = np.asarray(q_grid, dtype=np.float64)
    n = len(y)
    count = len(candidates)
    dt = 1.0 / fps
    transition = _jerk_transition(dt)
    base_process = _jerk_process(dt, 1.0)
    innovations = np.full((count, n, 3), np.nan, dtype=np.float64)
    innovation_var = np.full((count, n, 3), np.nan, dtype=np.float64)
    _, initial, available = _initial_states(y)
    if not available.any() or count == 0:
        return innovations, innovation_var

    state = np.zeros((count, 3, 3), dtype=np.float64)
    state[:, :, 0] = initial
    covariance = np.zeros((count, 3, 3, 3), dtype=np.float64)
    covariance[:, :, range(3), range(3)] = 1e6
    q_matrix = base_process[None, :, :] * candidates[:, None, None]
    for frame in range(n):
        if frame:
            state = state @ transition.T
            covariance = transition @ covariance @ transition.T + q_matrix[:, None, :, :]
            covariance = (covariance + np.swapaxes(covariance, 2, 3)) * 0.5
            diagonal = np.maximum(np.diagonal(covariance, axis1=2, axis2=3), 1e-12)
            covariance[:, :, range(3), range(3)] = diagonal
        observed = np.isfinite(y[frame]) & available
        variance = max(float(measurement_variance[frame]), 1e-12)
        innovation = np.where(observed[None, :], y[frame][None, :] - state[:, :, 0], 0.0)
        innovation_variance = np.maximum(covariance[:, :, 0, 0] + variance, 1e-12)
        gain = covariance[:, :, :, 0] / innovation_variance[:, :, None]
        update = observed.astype(np.float64)[None, :]
        state = state + gain * (innovation * update)[:, :, None]
        covariance = covariance - gain[:, :, :, None] * covariance[:, :, 0, :][:, :, None, :] * update[:, :, None, None]
        covariance = (covariance + np.swapaxes(covariance, 2, 3)) * 0.5
        diagonal = np.maximum(np.diagonal(covariance, axis1=2, axis2=3), 1e-12)
        covariance[:, :, range(3), range(3)] = diagonal
        innovations[:, frame, observed] = innovation[:, observed]
        innovation_var[:, frame, observed] = innovation_variance[:, observed]
    return innovations, innovation_var

def _smooth(
    values: np.ndarray,
    fps: float,
    q: float,
    measurement_variance: np.ndarray,
    q_scale: np.ndarray | None = None,
):
    filtered = _filter(values, fps, q, measurement_variance, q_scale=q_scale)
    x_pred, p_pred, x_filt, p_filt, innovations, innovation_var = filtered
    n = len(values)
    x_smooth = x_filt.copy()
    p_smooth = p_filt.copy()
    transition = _jerk_transition(1.0 / fps)
    gains = None
    if n > 1:
        try:
            # np.linalg.solve broadcasts over time and spatial axes.  This
            # removes the Python solve call for every frame and axis.
            lhs = np.swapaxes(p_pred[1:], -1, -2)
            rhs = np.swapaxes(p_filt[:-1] @ transition.T, -1, -2)
            gains = np.swapaxes(np.linalg.solve(lhs, rhs), -1, -2)
        except np.linalg.LinAlgError:
            # Retain the legacy per-axis pseudo-inverse fallback if any
            # batched system is singular.
            gains = None
    if gains is None:
        for frame in range(n - 2, -1, -1):
            for axis in range(3):
                try:
                    gain = np.linalg.solve(p_pred[frame + 1, axis].T,
                                           (p_filt[frame, axis] @ transition.T).T).T
                except np.linalg.LinAlgError:
                    gain = p_filt[frame, axis] @ transition.T @ np.linalg.pinv(p_pred[frame + 1, axis])
                delta = x_smooth[frame + 1, axis] - x_pred[frame + 1, axis]
                x_smooth[frame, axis] = x_filt[frame, axis] + gain @ delta
                p_smooth[frame, axis] = _sym_psd(
                    p_filt[frame, axis] + gain @ (p_smooth[frame + 1, axis] - p_pred[frame + 1, axis]) @ gain.T
                )
    else:
        for frame in range(n - 2, -1, -1):
            gain = gains[frame]
            delta = x_smooth[frame + 1] - x_pred[frame + 1]
            x_smooth[frame] = x_filt[frame] + np.matmul(gain, delta[..., None])[..., 0]
            covariance = (
                p_filt[frame]
                + gain @ (p_smooth[frame + 1] - p_pred[frame + 1])
                @ np.swapaxes(gain, -1, -2)
            )
            p_smooth[frame] = _sym_psd_batch(covariance)
    positions = x_smooth[:, :, 0]
    variance = np.maximum(np.diagonal(p_smooth, axis1=2, axis2=3)[..., 0], 1e-12)
    return positions, variance, innovations, innovation_var


def _filter_grid_windows(
    values: np.ndarray,
    fps: float,
    q_grid: np.ndarray,
    measurement_variance: np.ndarray,
):
    """Forward-filter equal-length local windows and q candidates together."""
    y = np.asarray(values, dtype=np.float64)
    candidates = np.asarray(q_grid, dtype=np.float64)
    batch, n, _ = y.shape
    count = len(candidates)
    dt = 1.0 / fps
    transition = _jerk_transition(dt)
    base_process = _jerk_process(dt, 1.0)
    flat_count = batch * count
    axis_count = flat_count * 3
    innovations = np.full((flat_count, n, 3), np.nan, dtype=np.float64)
    innovation_var = np.full_like(innovations, np.nan)
    finite = np.isfinite(y)
    available = finite.any(axis=1)
    if count == 0 or not available.any():
        return innovations.reshape(batch, count, n, 3), innovation_var.reshape(batch, count, n, 3)
    initial = np.zeros((batch, 3), dtype=np.float64)
    for axis in range(3):
        present = available[:, axis]
        first = np.argmax(finite[:, :, axis], axis=1)
        initial[:, axis] = np.where(present, y[np.arange(batch), first, axis], 0.0)
    repeated_initial = np.repeat(initial, count, axis=0)
    state = np.zeros((axis_count, 3), dtype=np.float64)
    state[:, 0] = repeated_initial.reshape(-1)
    covariance = np.zeros((axis_count, 3, 3), dtype=np.float64)
    covariance[:, range(3), range(3)] = 1e6
    flat_q = np.tile(candidates, batch)
    flat_y = np.repeat(y, count, axis=0)
    flat_available = np.repeat(available, count, axis=0)
    flat_variance = np.repeat(np.asarray(measurement_variance, dtype=np.float64), count, axis=0)
    q_matrix = np.repeat(base_process[None, :, :] * flat_q[:, None, None], 3, axis=0)
    covariance_path = np.einsum_path(
        "ij,bjk,kl->bil", transition, covariance, transition.T, optimize="greedy"
    )[0]
    state_path = np.einsum_path(
        "bi,ij->bj", state, transition.T, optimize="greedy"
    )[0]
    for frame in range(n):
        if frame:
            state = np.einsum("bi,ij->bj", state, transition.T, optimize=state_path)
            covariance = np.einsum(
                "ij,bjk,kl->bil", transition, covariance, transition.T, optimize=covariance_path
            ) + q_matrix
            covariance = (covariance + np.swapaxes(covariance, 1, 2)) * 0.5
            diagonal = np.maximum(np.diagonal(covariance, axis1=1, axis2=2), 1e-12)
            covariance[:, range(3), range(3)] = diagonal
        observed = np.isfinite(flat_y[:, frame]) & flat_available
        variance = np.maximum(flat_variance[:, frame], 1e-12)
        state_axes = state.reshape(flat_count, 3, 3)
        innovation = np.where(observed, flat_y[:, frame] - state_axes[:, :, 0], 0.0)
        innovation_variance = np.maximum(
            covariance[:, 0, 0].reshape(flat_count, 3) + variance[:, None], 1e-12
        )
        gain = covariance[:, :, 0] / np.repeat(innovation_variance, 3, axis=0)
        update = observed.astype(np.float64)
        state = state + gain * (innovation * update).reshape(-1, 1)
        covariance = covariance - (
            gain[:, :, None]
            * covariance[:, 0, :][:, None, :]
            * update.reshape(-1, 1, 1)
        )
        covariance = (covariance + np.swapaxes(covariance, 1, 2)) * 0.5
        diagonal = np.maximum(np.diagonal(covariance, axis1=1, axis2=2), 1e-12)
        covariance[:, range(3), range(3)] = diagonal
        innovations[:, frame, :] = innovation
        innovation_var[:, frame, :] = innovation_variance
    return (
        innovations.reshape(batch, count, n, 3),
        innovation_var.reshape(batch, count, n, 3),
    )

def _select_q_windows(
    values: np.ndarray,
    fps: float,
    sigma: float,
    weights: np.ndarray,
    config: ConditioningConfig,
) -> np.ndarray:
    """Select q independently for a stack of equal-length local windows."""
    clipped = np.clip(np.asarray(weights, dtype=np.float64), 0.05, 1.0)
    measurement_variance = sigma * sigma / clipped
    candidates = np.asarray(config.q_grid, dtype=np.float64)
    innovations, innovation_var = _filter_grid_windows(
        values, fps, candidates, measurement_variance
    )
    best = np.full(len(values), float(candidates[0]), dtype=np.float64)
    best_scores = np.full(len(values), float("inf"), dtype=np.float64)
    valid_values = np.isfinite(values)[:, None, :, :]
    valid_weights = clipped[:, None, :, None]
    for index, candidate in enumerate(candidates):
        valid = valid_values & np.isfinite(innovations[:, index:index + 1]) & np.isfinite(innovation_var[:, index:index + 1])
        terms = np.where(
            valid,
            valid_weights * (
                np.log(innovation_var[:, index:index + 1])
                + innovations[:, index:index + 1] ** 2
                / innovation_var[:, index:index + 1]
            ),
            0.0,
        )
        scores = np.sum(terms, axis=(2, 3))[:, 0]
        better = scores < best_scores
        best_scores[better] = scores[better]
        best[better] = float(candidate)
    return best

def _weighted_log_likelihood(values: np.ndarray, innovations: np.ndarray,
                             innovation_var: np.ndarray, weights: np.ndarray) -> float:
    valid = np.isfinite(values) & np.isfinite(innovations) & np.isfinite(innovation_var)
    if not valid.any():
        return float("inf")
    w = np.broadcast_to(weights[:, None], values.shape)
    return float(np.sum(w[valid] * (np.log(innovation_var[valid]) + innovations[valid] ** 2 / innovation_var[valid])))


def _select_q(values: np.ndarray, fps: float, sigma: float, weights: np.ndarray,
              config: ConditioningConfig,
              measurement_variance: np.ndarray | None = None) -> float:
    weights = np.clip(np.asarray(weights, dtype=np.float64), 0.05, 1.0)
    if measurement_variance is None:
        measurement_variance = np.full(len(values), sigma * sigma, dtype=np.float64) / weights
        candidates = np.asarray(config.q_grid, dtype=np.float64)
        innovations, innovation_var = _filter_grid(values, fps, candidates, measurement_variance)
        best_q = float(candidates[0])
        best_score = float("inf")
        # Retain the original ordered comparison so ties select the first q_grid
        # candidate exactly as before.
        for index, candidate in enumerate(candidates):
            score = _weighted_log_likelihood(
                values, innovations[index], innovation_var[index], weights
            )
            if score < best_score:
                best_score = score
                best_q = float(candidate)
        return best_q

    # Optional heteroscedastic observations.  This branch is intentionally
    # unreachable from the legacy API path, preserving the exact default
    # arithmetic while allowing a per-frame variance for experiments.
    variance = np.asarray(measurement_variance, dtype=np.float64)
    if variance.shape != (len(values),) or not np.isfinite(variance).all() or np.any(variance <= 0.0):
        raise ValueError("measurement_variance must be finite and positive with shape (F,)")
    candidates = np.asarray(config.q_grid, dtype=np.float64)
    innovations, innovation_var = _filter_grid(values, fps, candidates, variance)
    reference = max(float(sigma * sigma), 1e-12)
    sample_weights = np.clip(reference / variance, 0.05, 1.0)
    best_q = float(candidates[0])
    best_score = float("inf")
    for index, candidate in enumerate(candidates):
        valid = np.isfinite(values) & np.isfinite(innovations[index]) & np.isfinite(innovation_var[index])
        terms = np.where(
            valid,
            sample_weights[:, None] * (
                np.log(innovation_var[index])
                + innovations[index] * innovations[index] / innovation_var[index]
            ),
            0.0,
        )
        score = float(np.sum(terms))
        if score < best_score:
            best_score = score
            best_q = float(candidate)
    return best_q


def _chi_square_99(dof: np.ndarray, z: float) -> np.ndarray:
    """Wilson-Hilferty chi-square quantile approximation, NumPy-only."""
    dof = np.maximum(np.asarray(dof, dtype=np.float64), 1.0)
    return dof * np.maximum(1.0 - 2.0 / (9.0 * dof) + z * np.sqrt(2.0 / (9.0 * dof)), 1e-12) ** 3


def _maneuver_scales(innovations: np.ndarray, innovation_var: np.ndarray,
                     weights: np.ndarray, config: ConditioningConfig,
                     observed_mask: np.ndarray | None = None):
    nis = np.sum(np.where(np.isfinite(innovations) & np.isfinite(innovation_var),
                          innovations * innovations / np.maximum(innovation_var, 1e-12), 0.0), axis=1)
    if observed_mask is None:
        observed_mask = np.ones(len(nis), dtype=bool)
    usable = (np.asarray(weights > 0.5, dtype=bool)
              & np.asarray(observed_mask, dtype=bool) & np.isfinite(nis))
    width = max(1, int(config.maneuver_window) // 2)
    padded_nis = np.pad(np.where(usable, nis, 0.0), (width, width), mode="edge")
    padded_count = np.pad(usable.astype(np.float64), (width, width), mode="edge")
    blocked = np.pad((~usable).astype(np.float64), (width, width), mode="edge")
    sums = np.convolve(padded_nis, np.ones(2 * width + 1), mode="valid")
    counts = np.convolve(padded_count, np.ones(2 * width + 1), mode="valid")
    blocked_count = np.convolve(blocked, np.ones(2 * width + 1), mode="valid")
    threshold = _chi_square_99(3.0 * np.maximum(counts, 1.0), config.maneuver_quantile_z)
    excess = np.maximum(sums / np.maximum(threshold, 1e-12) - 1.0, 0.0)
    # A window touching a rejected or missing sample is not evidence of a
    # maneuver.  In particular, never boost process noise inside a NaN gap.
    excess[blocked_count > 0.0] = 0.0
    log_factor = np.clip(excess * np.log10(config.maneuver_max_factor), 0.0,
                         np.log10(config.maneuver_max_factor))
    kernel = np.array([0.125, 0.25, 0.25, 0.25, 0.125], dtype=np.float64)
    log_factor = np.convolve(np.pad(log_factor, (2, 2), mode="edge"), kernel, mode="valid")
    scales = np.power(10.0, log_factor)
    scales[~np.asarray(observed_mask, dtype=bool)] = 1.0
    return scales, nis, threshold


def _local_q_profile(values: np.ndarray, fps: float, sigma: float,
                     weights: np.ndarray, global_q: float,
                     config: ConditioningConfig) -> np.ndarray:
    """Estimate a time-varying q from overlapping robust local windows.

    The profile is deliberately estimated from observations and IRLS weights,
    then smoothed in log space.  Missing samples are allowed to use the
    surrounding profile for the RTS bridge, but are never treated as evidence
    for a maneuver or given a special gap boost.
    """
    n = len(values)
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    width = max(3, int(round(float(config.local_window_seconds) * fps)))
    hop = max(1, int(config.local_q_hop_frames))
    centers = list(range(0, n, hop))
    if centers[-1] != n - 1:
        centers.append(n - 1)
    estimates = np.full(len(centers), float(global_q), dtype=np.float64)
    windows: list[np.ndarray] = []
    eligible: list[int] = []
    for k, center in enumerate(centers):
        lo = max(0, center - width // 2)
        hi = min(n, lo + width)
        lo = max(0, hi - width)
        window_values = values[lo:hi]
        if np.count_nonzero(np.isfinite(window_values).all(axis=1)) >= 4:
            # All windows have the same length for a fixed track, including
            # the clipped edge windows, so they can share one batched pass.
            windows.append(window_values)
            eligible.append(k)
    if windows:
        window_values = np.stack(windows, axis=0)
        # Recompute the adjusted starts exactly as the legacy loop does when
        # a right-edge slice is shorter than width.
        adjusted_weights = []
        for k in eligible:
            center = centers[k]
            lo = max(0, center - width // 2)
            hi = min(n, lo + width)
            lo = max(0, hi - width)
            adjusted_weights.append(weights[lo:hi])
        window_weights = np.stack(adjusted_weights, axis=0)
        selected = _select_q_windows(
            window_values, fps, sigma, window_weights, config
        )
        for k, estimate in zip(eligible, selected):
            estimates[k] = max(float(global_q), float(estimate))
    log_profile = np.interp(np.arange(n, dtype=np.float64), centers,
                            np.log(np.maximum(estimates, global_q)))
    smooth_frames = max(1, int(round(float(config.local_log_smooth_seconds) * fps)))
    kernel = np.ones(2 * smooth_frames + 1, dtype=np.float64)
    kernel /= np.sum(kernel)
    log_profile = np.convolve(np.pad(log_profile, (smooth_frames, smooth_frames), mode="edge"),
                              kernel, mode="valid")
    return np.maximum(np.exp(log_profile), float(global_q))


def _bandwidth_hz(q: float, sigma: float, fps: float) -> float:
    # The continuous measurement equivalent has spectral density sigma²*dt;
    # cubic-spline/jerk smoothing has its -3 dB scale at (q/S)^1/6.
    dt = 1.0 / fps
    omega = max(float(q) / max(sigma * sigma * dt, 1e-18), 0.0) ** (1.0 / 6.0)
    return float(np.clip(omega / (2.0 * np.pi), 0.0, 0.5 * fps))


def _condition_one(
    values: np.ndarray,
    fps: float,
    config: ConditioningConfig,
    measurement_variance: np.ndarray | None = None,
):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError(f"each target track must have shape (F, 3), got {values.shape}")
    n = len(values)
    output = np.full_like(values, np.nan, dtype=np.float64)
    finite_rows = np.isfinite(values).all(axis=1)
    if measurement_variance is not None:
        measurement_variance = np.asarray(measurement_variance, dtype=np.float64)
        if (measurement_variance.shape != (n,) or not np.isfinite(measurement_variance).all()
                or np.any(measurement_variance <= 0.0)):
            raise ValueError("measurement_variance must be finite and positive with shape (F,)")
    if n == 0 or not np.isfinite(values).any():
        return output, {
            "sigma_cm": float(config.sigma_floor_cm), "q_cm2_s5": float(config.q_grid[0]),
            "bandwidth_hz": 0.0, "outlier_count": 0, "boosted_fraction": 0.0,
            "bridged_frames": 0, "observed_frames": 0, "quasi_static_frames": 0,
        }
    sigma, _, _, quasi_count, _ = _estimate_sigma(values, fps, config)
    weights = np.ones(n, dtype=np.float64)
    q = _select_q(
        values, fps, sigma, weights, config,
        measurement_variance=measurement_variance,
    )
    final_variance = (
        np.full(n, sigma * sigma, dtype=np.float64)
        if measurement_variance is None else measurement_variance.copy()
    )
    final_weights = weights.copy()
    for _ in range(max(1, int(config.irls_passes))):
        final_weights = np.clip(weights, 0.05, 1.0)
        if measurement_variance is None:
            final_variance = sigma * sigma / final_weights
        else:
            final_variance = measurement_variance / final_weights
        positions, variance, _, _ = _smooth(values, fps, q, final_variance)
        residual = values - positions
        denom = np.maximum(variance + final_variance[:, None], 1e-12)
        z2 = np.sum(np.where(np.isfinite(residual), residual * residual / denom, 0.0), axis=1)
        z2[~finite_rows] = 0.0
        weights = np.ones(n, dtype=np.float64)
        weights[finite_rows] = (config.student_t_nu + 3.0) / (config.student_t_nu + z2[finite_rows])
        q = _select_q(
            values, fps, sigma, weights, config,
            measurement_variance=(
                None if measurement_variance is None
                else measurement_variance / np.maximum(weights, 0.05)
            ),
        )
    final_weights = np.clip(weights, 0.05, 1.0)
    if measurement_variance is None:
        final_variance = sigma * sigma / final_weights
    else:
        final_variance = measurement_variance / final_weights
    _, _, innovations, innovation_var = _smooth(values, fps, q, final_variance)
    local_q = _local_q_profile(values, fps, sigma, final_weights, q, config)
    scales, nis, threshold = _maneuver_scales(
        innovations, innovation_var, final_weights, config, observed_mask=finite_rows)
    if not config.maneuver_boost_enabled:
        scales = np.ones(n, dtype=np.float64)
    # q_scale is relative to the global ML q.  The final pass is a genuine
    # time-varying-Q RTS smoother; local adaptation is lower-bounded by q.
    q_scale = np.maximum(local_q / max(q, 1e-12), 1.0) * scales
    positions, variance, _, _ = _smooth(
        values, fps, q, final_variance, q_scale=q_scale
    )
    output[:, :] = positions
    for axis in range(3):
        if not np.isfinite(values[:, axis]).any():
            output[:, axis] = np.nan

    max_gap = int(config.max_bridge_frames if config.max_bridge_frames is not None
                  else round(0.5 * fps))
    gap_lengths: list[int] = []
    gap_classes: dict[str, int] = {"clip_edge": 0, "short_interior": 0, "long_interior": 0}
    start = 0
    while start < n:
        if finite_rows[start]:
            start += 1
            continue
        end = start
        while end < n and not finite_rows[end]:
            end += 1
        length = end - start
        gap_lengths.append(length)
        left = start > 0
        right = end < n
        if not left or not right:
            gap_classes["clip_edge"] += length
            # Never extrapolate a CA state beyond the clip. Hold the nearest
            # smoothed observation with zero velocity instead.
            nearest = output[end] if not left else output[start - 1]
            output[start:end] = nearest
        elif length > max_gap:
            gap_classes["long_interior"] += length
            # Long gaps use the legacy-equivalent straight bridge between the
            # two smoothed boundary positions, avoiding CA extrapolation.
            left_value, right_value = output[start - 1].copy(), output[end].copy()
            for frame in range(start, end):
                alpha = (frame - (start - 1)) / float(end - (start - 1))
                output[frame] = (1.0 - alpha) * left_value + alpha * right_value
        else:
            gap_classes["short_interior"] += length
            # Short gaps retain the RTS bridge, whose boundary velocities are
            # continuous. q boosts are disabled on all missing samples above.
        start = end
    finite_output = np.isfinite(output).all(axis=1)
    bridged = int(np.count_nonzero(~finite_rows & finite_output))
    boosted = scales > 1.0 + 1e-12
    local_q_p50 = float(np.percentile(local_q, 50)) if len(local_q) else float(q)
    local_q_p95 = float(np.percentile(local_q, 95)) if len(local_q) else float(q)
    return output, {
        "sigma_cm": float(sigma),
        "q_cm2_s5": float(q),
        "local_q_p50_cm2_s5": local_q_p50,
        "local_q_p95_cm2_s5": local_q_p95,
        "bandwidth_hz": _bandwidth_hz(q, sigma, fps),
        "outlier_count": int(np.count_nonzero(final_weights < 0.2)),
        "boosted_fraction": float(np.mean(boosted)) if n else 0.0,
        "boosted_frames": int(np.count_nonzero(boosted)),
        "bridged_frames": bridged,
        "gap_classes": gap_classes,
        "long_gap_frames": int(sum(g for g in gap_lengths if g > max_gap)),
        "long_gap_count": int(sum(g > max_gap for g in gap_lengths)),
        "observed_frames": int(np.count_nonzero(finite_rows)),
        "quasi_static_frames": int(quasi_count),
        "max_bridge_frames": max_gap,
        "maneuver_threshold_nis_p99_mean": float(np.nanmax(threshold)) if len(threshold) else 0.0,
        "maneuver_nis_max": float(np.nanmax(nis)) if np.isfinite(nis).any() else 0.0,
    }


def _midpoint(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.full_like(a, np.nan, dtype=np.float64)
    valid = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
    out[valid] = (a[valid] + b[valid]) * 0.5
    return out


def condition_target_tracks(
    tracks: dict[str, np.ndarray],
    fps: float,
    config: ConditioningConfig | None = None,
    measurement_variance: dict[str, np.ndarray] | np.ndarray | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Condition all solver-consumed target tracks deterministically.

    ``tracks`` uses NaN rows/components for unobserved values.  Every track
    with at least one finite observation becomes finite across its bridgeable
    timeline.  Pelvis and neck are derived *after* conditioning their source
    hip/shoulder tracks, matching ``extract_mocap_points`` exactly.

    ``measurement_variance`` is optional and experimental.  It is either a
    length-F array broadcast to every independent track or a mapping from
    track name to a length-F array in the track's squared position units.  A
    missing mapping entry uses the legacy homoscedastic estimate.  Leaving it
    as ``None`` takes the original code path and preserves legacy results.
    """
    cfg = _as_config(config)
    fps = _validate_fps(fps)
    if not isinstance(tracks, dict):
        raise TypeError("tracks must be a dict[str, (F, 3) array]")
    raw = {str(name): np.asarray(value, dtype=np.float64) for name, value in tracks.items()}
    lengths = {len(value) for value in raw.values() if value.ndim >= 1}
    if len(lengths) > 1:
        raise ValueError("all target tracks must have the same frame count")
    if measurement_variance is None:
        variance_map: dict[str, np.ndarray] = {}
    elif isinstance(measurement_variance, dict):
        variance_map = {str(name): np.asarray(value, dtype=np.float64) for name, value in measurement_variance.items()}
    else:
        shared = np.asarray(measurement_variance, dtype=np.float64)
        variance_map = {name: shared for name in raw}
    conditioned: dict[str, np.ndarray] = {}
    diagnostics: dict[str, Any] = {
        "enabled": True,
        "fps": float(fps),
        "model": "constant_acceleration_white_jerk",
        "irls_passes": int(cfg.irls_passes),
        "student_t_nu": float(cfg.student_t_nu),
        "tracks": {},
        "derived_tracks": {},
    }
    if measurement_variance is not None:
        diagnostics["measurement_variance"] = True
        diagnostics["measurement_variance_tracks"] = sorted(name for name in variance_map if name in raw)
    # Derived tracks are deliberately omitted from the independent filtering
    # set so the conditioned base observations define their exact midpoints.
    for name, value in raw.items():
        if name in {"pelvis", "neck"} and (
            (name == "pelvis" and "hip_l" in raw and "hip_r" in raw)
            or (name == "neck" and "shoulder_l" in raw and "shoulder_r" in raw)
        ):
            continue
        conditioned[name], diagnostics["tracks"][name] = _condition_one(
            value, fps, cfg, variance_map.get(name)
        )
    if "hip_l" in conditioned and "hip_r" in conditioned:
        conditioned["pelvis"] = _midpoint(conditioned["hip_l"], conditioned["hip_r"])
        diagnostics["derived_tracks"]["pelvis"] = "mean(hip_l, hip_r) after conditioning"
        diagnostics["tracks"]["pelvis"] = {"derived": True}
    elif "pelvis" in raw:
        conditioned["pelvis"], diagnostics["tracks"]["pelvis"] = _condition_one(
            raw["pelvis"], fps, cfg, variance_map.get("pelvis")
        )
    if "shoulder_l" in conditioned and "shoulder_r" in conditioned:
        conditioned["neck"] = _midpoint(conditioned["shoulder_l"], conditioned["shoulder_r"])
        diagnostics["derived_tracks"]["neck"] = "mean(shoulder_l, shoulder_r) after conditioning"
        diagnostics["tracks"]["neck"] = {"derived": True}
    elif "neck" in raw:
        conditioned["neck"], diagnostics["tracks"]["neck"] = _condition_one(
            raw["neck"], fps, cfg, variance_map.get("neck")
        )
    diagnostics["track_count"] = len(conditioned)
    diagnostics["finite_output_tracks"] = int(sum(np.isfinite(v).all(axis=1).any() for v in conditioned.values()))
    return conditioned, diagnostics


__all__ = ["ConditioningConfig", "condition_target_tracks"]
