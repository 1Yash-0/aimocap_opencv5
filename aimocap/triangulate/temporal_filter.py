"""Opt-in offline 3D temporal smoother for triangulated skeletons (P3.5).

A bidirectional constant-acceleration Kalman filter + Rauch-Tung-Striebel (RTS)
smoother is applied independently to every joint coordinate of a
``(frames, joints, 3)`` metric point sequence.

Design constraints
------------------
* **Offline / zero phase.**  The forward Kalman pass is followed by the RTS
  backward pass, so there is no causal lag on the output.  This module is only
  suitable for recorded clips, not live streaming.
* **Measurement noise from the triangulator.**  ``camera_trust`` and
  ``reprojection_error_px`` from P3.3 are converted into a per-point position
  standard deviation.  Fewer / less trusted cameras and larger residuals give
  larger noise.
* **Robust update.**  A Huber weight on the normalized innovation rejects
  single-frame spikes (left/right flips, bad triangulations) instead of letting
  them pull the trajectory.
* **Fast motion is preserved.**  Process noise is scaled per joint group,
  inflated when the innovation is large (a maneuver), and opened further where
  the joint's own recent speed is in its upper quantiles.  Real fast movement
  therefore follows the measurements instead of being damped like noise.
  Slow, quasi-static frames keep the low base process noise and are smoothed
  more strongly.
* **Gaps stay gaps.**  Missing points are prediction-only.  Short interior gaps
  are bridged; gaps longer than ``max_gap_frames`` are left as NaN, as are
  leading/trailing gaps.  Nothing is extrapolated beyond the clip.

The filter is opt-in via ``GeometryConfig.temporal_filter = "robust_rts"`` and
defaults to ``"none"`` (bit-identical old output).  It operates on the internal
camera-world frame (Y-up, metres), before any floor transform.

References: Kalman (1960), Rauch-Tung-Striebel (1965); the constant-acceleration
white-jerk model follows the same formulation used by the phase-4 conditioning
code (``aimocap/retarget/observation_conditioning.py``), but this module works
on raw 133-joint skeleton points in metres and consumes triangulation trust.
"""
from __future__ import annotations

import warnings

import numpy as np

# Process-noise multipliers by COCO-WholeBody group.  Hands and face are
# allowed to move faster than the torso/legs, feet slightly more than the body.
GROUP_Q_SCALE = np.ones(133, dtype=np.float64)
GROUP_Q_SCALE[17:23] = 1.5   # feet
GROUP_Q_SCALE[23:91] = 1.5   # face
GROUP_Q_SCALE[91:133] = 2.5  # hands

# Priors for a fresh track: 1 m/s velocity and 10 m/s^2 acceleration (1 sigma).
_INITIAL_VELOCITY_STD = 1.0
_INITIAL_ACCEL_STD = 10.0

__all__ = ["filter_skeleton3d_rts", "measurement_sigma_m"]


def measurement_sigma_m(
    points: np.ndarray,
    camera_trust: np.ndarray | None,
    reprojection_error_px: np.ndarray | None,
    base_sigma_m: float,
    *,
    reprojection_scale_px: float = 20.0,
    min_factor: float = 0.35,
    max_factor: float = 4.0,
) -> np.ndarray:
    """Per joint-frame position sigma from triangulation trust and residuals.

    ``points`` is ``(frames, joints, 3)``; the returned array is
    ``(frames, joints)`` in metres.  Non-finite points get ``inf`` (no update).
    """
    frames, joints, _ = points.shape
    finite = np.isfinite(points).all(axis=-1)
    if camera_trust is None:
        trust_sum = np.full((frames, joints), 2.0, dtype=np.float64)
    else:
        trust = np.asarray(camera_trust, dtype=np.float64)
        if trust.shape[:2] != (frames, joints):
            raise ValueError("camera_trust must have shape (frames, joints, cameras)")
        trust = np.where(np.isfinite(trust), np.clip(trust, 0.0, 1.0), 0.0)
        trust_sum = trust.sum(axis=-1)
    trust_sum = np.maximum(trust_sum, 0.5)

    if reprojection_error_px is None:
        residual_rms = np.full((frames, joints), 5.0, dtype=np.float64)
    else:
        residual = np.asarray(reprojection_error_px, dtype=np.float64)
        if residual.shape[:2] != (frames, joints):
            raise ValueError(
                "reprojection_error_px must have shape (frames, joints, cameras)"
            )
        usable = np.isfinite(residual) & (residual > 0.0)
        squared = np.where(usable, residual * residual, 0.0)
        count = usable.sum(axis=-1)
        residual_rms = np.sqrt(
            squared.sum(axis=-1) / np.maximum(count, 1)
        )
        residual_rms = np.where(count > 0, residual_rms, 5.0)

    # More trusted cameras reduce noise (roughly sqrt(N)); larger residuals
    # increase it.  Clipping keeps a single pathological camera from producing
    # an infinite sigma or a perfectly trusted point from producing zero noise.
    factor = (1.0 + residual_rms / float(reprojection_scale_px)) / np.sqrt(trust_sum)
    factor = np.clip(factor, min_factor, max_factor)
    sigma = float(base_sigma_m) * factor
    sigma = np.where(finite, sigma, np.inf)
    return sigma


def _batched_transition(dt: float) -> np.ndarray:
    return np.array(
        [
            [1.0, dt, 0.5 * dt * dt],
            [0.0, 1.0, dt],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _batched_process_noise(dt: float) -> np.ndarray:
    """Unit white-jerk covariance factor; multiply by q (m^2/s^5)."""
    dt2 = dt * dt
    dt3 = dt2 * dt
    dt4 = dt3 * dt
    dt5 = dt4 * dt
    return np.array(
        [
            [dt5 / 20.0, dt4 / 8.0, dt3 / 6.0],
            [dt4 / 8.0, dt3 / 3.0, dt2 / 2.0],
            [dt3 / 6.0, dt2 / 2.0, dt],
        ],
        dtype=np.float64,
    )


def _median_filter5(values: np.ndarray) -> np.ndarray:
    """NaN-aware 5-frame median along axis 0."""
    padded = np.pad(values, ((2, 2), (0, 0)), mode="constant", constant_values=np.nan)
    windows = np.lib.stride_tricks.sliding_window_view(padded, 5, axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(windows, axis=-1)


def _fast_motion_q_scale(
    points: np.ndarray,
    fps: float,
    fast_q_boost: float,
    low_quantile: float,
    high_quantile: float,
) -> np.ndarray:
    """Per-coordinate process-noise multiplier that opens up on fast motion.

    The raw (unsmoothed) 5-frame median speed of each joint is compared with
    that joint's own speed distribution over the clip.  Below ``low_quantile``
    the multiplier is 1 (base process noise); above ``high_quantile`` it is
    ``fast_q_boost``; in between it ramps linearly.  This keeps fast limbs from
    being damped while the base process noise is free to be low enough to
    smooth the slower majority of frames.
    """
    frames, joints, _ = points.shape
    velocity = np.full_like(points, np.nan)
    velocity[1:-1] = (points[2:] - points[:-2]) * (0.5 * fps)
    speed = np.linalg.norm(velocity, axis=-1)
    speed = _median_filter5(speed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        low = np.nanpercentile(speed, low_quantile, axis=0)
        high = np.nanpercentile(speed, high_quantile, axis=0)
    span = high - low
    usable = np.isfinite(span) & (span > 1e-6)
    ramp = np.zeros((frames, joints), dtype=np.float64)
    if usable.any():
        normalized = (speed[:, usable] - low[usable]) / span[usable]
        ramp[:, usable] = np.clip(normalized, 0.0, 1.0)
    ramp = np.where(np.isfinite(ramp), ramp, 0.0)
    scale = 1.0 + (float(fast_q_boost) - 1.0) * ramp
    return np.repeat(scale[:, :, None], 3, axis=2).reshape(frames, joints * 3).T


def _fill_short_gaps(
    smoothed: np.ndarray,
    observed: np.ndarray,
    max_gap_frames: int,
) -> np.ndarray:
    """Return a copy where interior observed-gap runs <= max_gap_frames are kept."""
    out = np.array(smoothed, copy=True)
    frames = observed.shape[0]
    for joint in range(observed.shape[1]):
        visible = observed[:, joint]
        if not visible.any():
            out[:, joint] = np.nan
            continue
        first = int(np.argmax(visible))
        last = frames - 1 - int(np.argmax(visible[::-1]))
        out[:first, joint] = np.nan
        out[last + 1 :, joint] = np.nan
        index = first
        while index <= last:
            if visible[index]:
                index += 1
                continue
            start = index
            while index <= last and not visible[index]:
                index += 1
            if index - start > max_gap_frames:
                out[start:index, joint] = np.nan
    return out


def filter_skeleton3d_rts(
    points: np.ndarray,
    fps: float,
    *,
    sigma_m: float = 0.012,
    q_m2_s5: float = 30000.0,
    maneuver_boost: float = 100.0,
    trust: np.ndarray | None = None,
    reprojection_error_px: np.ndarray | None = None,
    max_gap_s: float = 0.2,
    robust_c: float = 3.0,
    robust_min_weight: float = 0.05,
    maneuver_threshold: float = 6.25,
    fast_q_boost: float = 50.0,
    fast_speed_low_quantile: float = 50.0,
    fast_speed_high_quantile: float = 70.0,
    spike_gate: bool = False,
    mad_window: int = 15,
    mad_scale: float = 4.0,
    mad_min_samples: int = 8,
    maneuver_cap: float = 1.0,
) -> np.ndarray:
    """Zero-phase constant-acceleration RTS smoothing of a 3D point sequence.

    Parameters
    ----------
    points:
        ``(frames, joints, 3)`` float array in metres, internal camera-world
        frame.  NaN marks an unobserved joint-frame.
    fps:
        Sequence frame rate (Hz), used for the transition and process-noise
        matrices and for the gap limit.
    sigma_m:
        Base per-axis measurement standard deviation at full trust.
    q_m2_s5:
        White-jerk spectral density in m^2/s^5 (base process noise).
    maneuver_boost:
        Maximum multiplicative process-noise inflation when the normalized
        innovation exceeds ``maneuver_threshold``.
    fast_q_boost, fast_speed_low_quantile, fast_speed_high_quantile:
        Per-joint speed-adaptive process-noise boost.  Below the joint's
        ``fast_speed_low_quantile`` of raw 5-frame median speed the base ``q``
        applies; above ``fast_speed_high_quantile`` it is multiplied by
        ``fast_q_boost``; in between it ramps linearly.  This protects fast
        motion from over-smoothing without using any ground truth.
    trust, reprojection_error_px:
        Optional ``(frames, joints, cameras)`` diagnostics from triangulation.
    max_gap_s:
        Interior gaps up to ``round(fps * max_gap_s)`` frames are bridged by the
        smoother; longer gaps stay NaN.
    robust_c, robust_min_weight:
        Huber threshold on the normalized innovation and its weight floor.
    maneuver_threshold:
        Squared normalized innovation above which process noise is inflated.
    spike_gate, mad_window, mad_scale, mad_min_samples:
        P3.8 spike guard.  When ``spike_gate`` is on, a measured coordinate whose
        raw innovation exceeds ``mad_scale`` times the robust MAD of the last
        ``mad_window`` innovations (once ``mad_min_samples`` are available) and
        the measurement-noise floor ``robust_c * sigma`` is rejected (zero Kalman
        gain).  An innovation that was already large on the previous frame is
        accepted, so a real step (or fast oscillation) is followed on its second
        frame while an isolated one-frame spike is rejected.
    maneuver_cap:
        Fraction of the normalized-innovation excess allowed to inflate process
        noise.  ``1.0`` reproduces the P3.5 behaviour; lower values stop a
        single-frame spike from opening the filter.

    Returns
    -------
    ``(frames, joints, 3)`` float64 array.  Observed frames are smoothed;
    short interior gaps may be filled; everything else is NaN.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 3 or points.shape[2] != 3:
        raise ValueError("points must have shape (frames, joints, 3)")
    frames, joints, _ = points.shape
    if frames < 3 or joints == 0:
        return np.array(points, copy=True)
    fps = float(fps)
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError("fps must be finite and positive")
    if sigma_m <= 0.0:
        raise ValueError("sigma_m must be positive")
    if q_m2_s5 <= 0.0:
        raise ValueError("q_m2_s5 must be positive")
    if maneuver_boost < 0.0:
        raise ValueError("maneuver_boost must be non-negative")
    if max_gap_s < 0.0:
        raise ValueError("max_gap_s must be non-negative")
    if not 0.0 < robust_min_weight <= 1.0:
        raise ValueError("robust_min_weight must be in (0, 1]")
    if robust_c <= 0.0:
        raise ValueError("robust_c must be positive")
    if maneuver_threshold <= 0.0:
        raise ValueError("maneuver_threshold must be positive")
    if fast_q_boost < 1.0:
        raise ValueError("fast_q_boost must be at least 1")
    if not 0.0 <= fast_speed_low_quantile < fast_speed_high_quantile <= 100.0:
        raise ValueError("fast speed quantiles must satisfy 0 <= low < high <= 100")
    if int(mad_window) < 3:
        raise ValueError("mad_window must be >= 3")
    if mad_scale <= 0.0:
        raise ValueError("mad_scale must be positive")
    if int(mad_min_samples) < 1:
        raise ValueError("mad_min_samples must be >= 1")
    if not 0.0 <= maneuver_cap <= 1.0:
        raise ValueError("maneuver_cap must be in [0, 1]")

    dt = 1.0 / fps
    transition = _batched_transition(dt)
    process_unit = _batched_process_noise(dt)
    max_gap_frames = int(round(max_gap_s * fps))

    # Flatten joints x axes so the whole sequence can be filtered with batched
    # 3x3 algebra (no Python loop over joints).
    count = joints * 3
    z = points.reshape(frames, count).T  # (coordinates, frames)
    finite_coord = np.isfinite(z)
    sigma = measurement_sigma_m(
        points, trust, reprojection_error_px, sigma_m
    ).reshape(frames, joints, 1)  # broadcast, same sigma for x/y/z
    sigma = np.repeat(sigma, 3, axis=2).reshape(frames, count).T
    variance = np.square(sigma)
    if joints > GROUP_Q_SCALE.size:
        raise ValueError(
            f"points has {joints} joints but only {GROUP_Q_SCALE.size} group scales exist"
        )
    q = q_m2_s5 * np.repeat(GROUP_Q_SCALE[:joints], 3)
    fast_scale = _fast_motion_q_scale(
        points,
        fps,
        fast_q_boost,
        fast_speed_low_quantile,
        fast_speed_high_quantile,
    )

    x_filtered = np.zeros((frames, count, 3), dtype=np.float64)
    p_filtered = np.zeros((frames, count, 3, 3), dtype=np.float64)
    x_predicted = np.zeros((frames, count, 3), dtype=np.float64)
    p_predicted = np.zeros((frames, count, 3, 3), dtype=np.float64)

    x = np.zeros((count, 3), dtype=np.float64)
    p = np.zeros((count, 3, 3), dtype=np.float64)
    p[:, 0, 0] = 1.0e6
    p[:, 1, 1] = _INITIAL_VELOCITY_STD ** 2
    p[:, 2, 2] = _INITIAL_ACCEL_STD ** 2
    q_scale = np.ones(count, dtype=np.float64)
    innovation_history = np.full((int(mad_window), count), np.nan, dtype=np.float64)
    history_index = 0
    last_large = np.zeros(count, dtype=bool)
    rejected = np.zeros((frames, count), dtype=bool)

    for frame in range(frames):
        if frame == 0:
            x_predicted[0] = x
            p_predicted[0] = p
        else:
            q_frame = q * q_scale * fast_scale[:, frame]
            x = x @ transition.T
            p = (
                np.einsum("ij,cjk,lk->cil", transition, p, transition)
                + process_unit[None, :, :] * q_frame[:, None, None]
            )
            x_predicted[frame] = x
            p_predicted[frame] = p

        measured = finite_coord[:, frame]
        if measured.any():
            covariance = p_predicted[frame]
            position_var = covariance[:, 0, 0]
            raw_innovation = z[:, frame] - x_predicted[frame, :, 0]
            # Only measured coordinates update.  Dummy finite values keep the
            # unmeasured lanes NaN-free; the final np.where restores prediction.
            innovation = np.where(measured, raw_innovation, 0.0)
            frame_variance = np.where(measured, variance[:, frame], 1.0)
            innovation_var = position_var + frame_variance
            normalized_sq = np.where(
                measured, np.square(innovation) / innovation_var, 0.0
            )

            # Huber weight: spikes get down-weighted, normal frames unchanged.
            normalized = np.sqrt(np.maximum(normalized_sq, 0.0))
            weight = np.clip(robust_c / np.maximum(normalized, 1e-12), robust_min_weight, 1.0)
            weight = np.where(normalized <= robust_c, 1.0, weight)

            # P3.8 spike guard: reject an isolated innovation that is far from
            # the recent robust spread.  A real step (or a fast oscillation)
            # produces a large innovation on consecutive frames; the second
            # frame is accepted so the filter follows it instead of locking the
            # old position.  Only isolated one-frame excursions are gated.
            gated = np.zeros(count, dtype=bool)
            if spike_gate:
                samples = np.sum(np.isfinite(innovation_history), axis=0)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    center = np.nanmedian(innovation_history, axis=0)
                    spread = 1.4826 * np.nanmedian(
                        np.abs(innovation_history - center), axis=0
                    )
                threshold = np.maximum(
                    mad_scale * spread, robust_c * np.sqrt(frame_variance)
                )
                large = (
                    measured
                    & (samples >= mad_min_samples)
                    & np.isfinite(threshold)
                    & (np.abs(raw_innovation) > threshold)
                )
                sustained = large & last_large
                gated = large & ~sustained
                rejected[frame, gated] = True
                innovation_history[history_index, measured] = raw_innovation[measured]
                history_index = (history_index + 1) % innovation_history.shape[0]
                last_large = large

            effective_var = np.where(measured, frame_variance / weight, 1.0)
            # A gated measurement is rejected outright (infinite variance, zero
            # Kalman gain); a sustained step was already accepted above.
            effective_var = np.where(gated, np.inf, effective_var)

            gain = covariance[:, :, 0] / (position_var + effective_var)[:, None]
            update = gain * innovation[:, None]
            updated_x = x_predicted[frame] + update
            updated_p = covariance - gain[:, :, None] * covariance[:, 0, :][:, None, :]
            # Symmetrize to control round-off.
            updated_p = 0.5 * (updated_p + np.swapaxes(updated_p, 1, 2))

            excess = np.maximum(normalized_sq / maneuver_threshold - 1.0, 0.0)
            frame_q_scale = 1.0 + maneuver_boost * np.minimum(excess, maneuver_cap)
            # A rejected measurement must not open the filter for the next
            # prediction; only accepted measurements can signal a maneuver.
            frame_q_scale = np.where(gated, 1.0, frame_q_scale)

            x = np.where(measured[:, None], updated_x, x_predicted[frame])
            p = np.where(measured[:, None, None], updated_p, covariance)
            q_scale = np.where(measured, frame_q_scale, 1.0)
        else:
            x = x_predicted[frame]
            p = p_predicted[frame]
            q_scale = np.ones(count, dtype=np.float64)

        x_filtered[frame] = x
        p_filtered[frame] = p

    # RTS backward pass: x_s[t] = x_f[t] + C (x_s[t+1] - x_p[t+1]).
    x_smoothed = np.array(x_filtered, copy=True)
    for frame in range(frames - 2, -1, -1):
        gain_backward = np.linalg.solve(
            p_predicted[frame + 1],
            np.einsum("ij,cjk->cik", transition, p_filtered[frame]),
        )
        gain_backward = np.swapaxes(gain_backward, 1, 2)
        residual = x_smoothed[frame + 1] - x_predicted[frame + 1]
        x_smoothed[frame] = x_filtered[frame] + np.einsum(
            "cij,cj->ci", gain_backward, residual
        )

    positions = x_smoothed[:, :, 0].reshape(frames, joints, 3)
    # A gated measurement is not an observation: treat the whole joint-frame as
    # missing so an isolated rejected point cannot survive as a prediction-only
    # extrapolation.  Short rejected runs may still be bridged by the gap policy.
    observed = np.isfinite(points).all(axis=-1)
    if spike_gate:
        observed &= ~rejected.reshape(frames, joints, 3).any(axis=-1)
    positions = _fill_short_gaps(positions, observed, max_gap_frames)
    # Preserve the exact NaN pattern outside bridged gaps and force non-finite
    # points at the original unobserved frames beyond the gap policy.
    positions = np.where(np.isfinite(positions), positions, np.nan)
    return positions
