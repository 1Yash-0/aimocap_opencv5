"""Temporal filtering for 3D keypoints."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt
import math

def filter_skeleton3d(
    skeleton3d: np.ndarray,
    fps: float = 30.0,
    cutoff_freq: float = 3.0,
    order: int = 2
) -> np.ndarray:
    """
    Apply a zero-phase Butterworth lowpass filter to 3D skeleton keypoints.
    Also interpolates over missing (NaN) frames.
    
    Args:
        skeleton3d: (F, K, 3) float array of 3D keypoints.
        fps: The frame rate of the capture.
        cutoff_freq: Lowpass cutoff frequency in Hz.
        order: Filter order.
        
    Returns:
        Filtered (F, K, 3) array.
    """
    num_frames, num_kpts, _ = skeleton3d.shape
    filtered = np.copy(skeleton3d)
    
    # Need at least 9 frames to apply filtfilt safely
    if num_frames < 9:
        return filtered
        
    nyq = 0.5 * fps
    normal_cutoff = cutoff_freq / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    
    for k in range(num_kpts):
        for dim in range(3):
            seq = filtered[:, k, dim]
            
            # Interpolate NaNs
            valid_mask = ~np.isnan(seq)
            if not np.any(valid_mask):
                continue
                
            # If some are missing but not all, interpolate
            if not np.all(valid_mask):
                valid_idx = np.where(valid_mask)[0]
                all_idx = np.arange(num_frames)
                seq = np.interp(all_idx, valid_idx, seq[valid_idx])
                
            # Filter
            try:
                # padlen must be less than length of seq
                padlen = min(15, len(seq) - 1)
                smoothed = filtfilt(b, a, seq, padlen=padlen)
                filtered[:, k, dim] = smoothed
            except ValueError:
                # Fallback to unfiltered if filtfilt fails
                filtered[:, k, dim] = seq
                
    return filtered


# ── Stage 5: Gap filling ──────────────────────────────────────────────────────

# Per-joint gap-fill: interpolate NaNs using scipy cubic spline (or linear if
# <4 valid samples). Returns the filled array plus structured gap log.
LONG_GAP_THRESHOLD = 15  # frames


def fill_gaps_with_logging(
    skeleton3d: np.ndarray,
    joint_names: list,
    fps: float = 30.0,
) -> tuple:
    """
    Fill NaN gaps in a (F, K, 3) skeleton array.

    Gap-fill policy:
      - Gaps <= LONG_GAP_THRESHOLD frames: cubic spline (or linear if <4 valid
        samples). Measured data, reliable interpolation.
      - Gaps > LONG_GAP_THRESHOLD frames: linear fill, flagged as
        ``reconstructed=True`` in the log. These are fabricated motion;
        they will be replaced by Stage 6's kinematic solve.

    Returns:
        filled:          (F, K, 3) fully interpolated array (no NaN if any
                         valid data exists for the joint)
        gap_log:         list of dicts: {joint, joint_idx, start_frame,
                                         end_frame, gap_length, long_gap,
                                         reconstructed, fill_method}
        long_gap_counts: {joint_name: count_of_long_gaps}
    """
    from scipy.interpolate import interp1d

    F, K, _ = skeleton3d.shape
    filled = skeleton3d.copy()
    gap_log = []
    long_gap_counts = {n: 0 for n in joint_names}

    for k in range(K):
        jname = joint_names[k] if k < len(joint_names) else f"joint_{k}"
        for dim in range(3):
            seq = filled[:, k, dim].copy()
            valid_mask = ~np.isnan(seq)
            if valid_mask.all():
                continue  # no gaps
            if not valid_mask.any():
                continue  # all NaN -- nothing to fill

            # Find contiguous NaN runs
            nan_mask = np.isnan(seq)
            changes = np.diff(nan_mask.astype(int), prepend=0, append=0)
            starts = np.where(changes == 1)[0]
            ends   = np.where(changes == -1)[0]  # exclusive

            valid_idx  = np.where(valid_mask)[0]
            valid_vals = seq[valid_mask]

            for s, e in zip(starts, ends):
                gap_len = int(e - s)
                is_long = gap_len > LONG_GAP_THRESHOLD

                # Policy: cubic for short gaps, linear for long ones
                if is_long:
                    fill_method = "linear"
                else:
                    fill_method = "cubic" if len(valid_idx) >= 4 else "linear"

                # Log once per gap (dim==0 sentinel)
                if dim == 0:
                    rec = {
                        "joint":        jname,
                        "joint_idx":    k,
                        "start_frame":  int(s),
                        "end_frame":    int(e - 1),  # inclusive
                        "gap_length":   gap_len,
                        "long_gap":     is_long,
                        "reconstructed": is_long,    # long gaps = fabricated motion
                        "fill_method":  fill_method,
                    }
                    gap_log.append(rec)
                    if is_long:
                        long_gap_counts[jname] = long_gap_counts.get(jname, 0) + 1

                # Fill only this gap segment using the appropriate method
                # Build a local interpolator for this dimension
                gap_frames = np.arange(s, e)
                f_interp = interp1d(
                    valid_idx, valid_vals,
                    kind=fill_method,
                    bounds_error=False,
                    fill_value=(valid_vals[0], valid_vals[-1]),
                )
                seq[s:e] = f_interp(gap_frames)

            filled[:, k, dim] = seq

    return filled, gap_log, long_gap_counts



# ── Stage 5: One-Euro smoothing over 3D skeleton ─────────────────────────────

def filter_skeleton_one_euro(
    skeleton3d: np.ndarray,
    fps: float = 30.0,
    min_cutoff: float = 1.0,
    beta: float = 0.007,
) -> np.ndarray:
    """
    Apply One-Euro filter independently to each joint's (x, y, z) trajectory.
    Input skeleton3d must already be gap-filled (no NaN).

    Args:
        skeleton3d: (F, K, 3) float array.
        fps:        Capture frame rate.
        min_cutoff: One-Euro min_cutoff parameter (Hz). Lower = smoother but
                    more lag. Typical range 0.5–2.0 for mocap.
        beta:       One-Euro speed coefficient. Higher = less lag on fast motion.
                    Typical range 0.001–0.05.

    Returns:
        Smoothed (F, K, 3) array.
    """
    F, K, _ = skeleton3d.shape
    out = skeleton3d.copy()
    dt = 1.0 / fps

    for k in range(K):
        # Find first finite frame as x0 (handles all-NaN joints gracefully)
        x0 = None
        f0 = 0
        for f in range(F):
            if np.isfinite(skeleton3d[f, k]).all():
                x0 = skeleton3d[f, k]
                f0 = f
                break
        if x0 is None:
            continue  # all-NaN joint — leave as NaN

        filt = OneEuroFilter(0.0, x0, min_cutoff=min_cutoff, beta=beta)
        out[f0, k] = x0
        for f in range(f0 + 1, F):
            x = skeleton3d[f, k]
            if np.isfinite(x).all():
                out[f, k] = filt(f * dt, x)
            else:
                out[f, k] = filt(f * dt, filt.x_prev)  # hold last known

    return out


# ── Stage 5: Bone-length normalization (median, root-walk) ───────────────────

# COCO-17 skeleton tree: (child_idx, parent_idx, bone_key or None)
# Both hips (11, 12) AND shoulders (5, 6) are fixed anchors.
# This means: torso and shoulder_width are NOT normalized (they'd require
# moving an anchor). Only limbs are normalized: elbows/wrists from fixed
# shoulders, knees/ankles from fixed hips. This keeps MPJPE root (mid-hip)
# AND the global upper-body position stable.
SKELETON_TREE = [
    # Fixed anchors -- never moved
    (11, None, None),   # l_hip
    (12, None, None),   # r_hip
    (5,  None, None),   # l_shoulder
    (6,  None, None),   # r_shoulder
    # Legs (hip -> knee -> ankle)
    (13, 11,   "l_thigh"),
    (15, 13,   "l_shin"),
    (14, 12,   "r_thigh"),
    (16, 14,   "r_shin"),
    # Arms (shoulder -> elbow -> wrist)
    (7,  5,    "l_upper_arm"),
    (9,  7,    "l_forearm"),
    (8,  6,    "r_upper_arm"),
    (10, 8,    "r_forearm"),
    # Head chain (no bone-length rescaling -- face joints unreliable)
    (0,  5,    None),
    (1,  0,    None),
    (2,  0,    None),
    (3,  0,    None),
    (4,  0,    None),
]




class OneEuroFilter:
    def __init__(self, t0, x0, dx0=None, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = x0
        self.dx_prev = dx0 if dx0 is not None else np.zeros_like(x0)
        self.t_prev = t0

    def __call__(self, t, x):
        t_e = t - self.t_prev
        if t_e <= 0:
            return x

        # The filtered derivative of the signal.
        a_d = (2 * math.pi * self.d_cutoff * t_e) / (2 * math.pi * self.d_cutoff * t_e + 1)
        dx = (x - self.x_prev) / t_e
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev

        # The filtered signal.
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = (2 * math.pi * cutoff * t_e) / (2 * math.pi * cutoff * t_e + 1)
        x_hat = a * x + (1 - a) * self.x_prev

        # Memorize the previous values.
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t

        return x_hat

def filter_params_one_euro(params_seq: np.ndarray, fps: float = 30.0, min_cutoff=1.0, beta=0.007) -> np.ndarray:
    """
    Apply One-Euro filter to a sequence of kinematic parameters (root pos + joint rotations).
    
    Args:
        params_seq: (num_frames, 54) array
        fps: Frames per second
        
    Returns:
        Filtered (num_frames, 54) array
    """
    num_frames = params_seq.shape[0]
    if num_frames == 0:
        return params_seq
        
    filtered = np.zeros_like(params_seq)
    filtered[0] = params_seq[0]
    
    dt = 1.0 / fps
    f = OneEuroFilter(0.0, params_seq[0], min_cutoff=min_cutoff, beta=beta)
    
    for i in range(1, num_frames):
        filtered[i] = f(i * dt, params_seq[i])
        
    return filtered

from scipy.spatial.transform import Rotation

def filter_params_one_euro_quaternion(params_seq: np.ndarray, fps: float = 30.0, min_cutoff=1.0, beta=0.007) -> np.ndarray:
    """
    Apply One-Euro filter to kinematic parameters.
    Root position is filtered linearly.
    Joint rotations are filtered by filtering their angular velocity to avoid quaternion wraparound.

    Args:
        params_seq: (num_frames, 3 + num_joints*3) array -- root pos (3)
            followed by one 3D rotation vector per joint.  The joint count
            is inferred from the array width, so this works for any rig.
        fps: Frames per second

    Returns:
        Filtered (num_frames, 3 + num_joints*3) array
    """
    num_frames, n_params = params_seq.shape
    if num_frames == 0:
        return params_seq

    # Infer joint count from the array width (root pos + one rotvec/joint).
    n_joints = (n_params - 3) // 3

    filtered = np.zeros_like(params_seq)
    filtered[0] = params_seq[0]

    dt = 1.0 / fps

    # Root pos filter
    pos_filter = OneEuroFilter(0.0, params_seq[0, :3], min_cutoff=min_cutoff, beta=beta)

    # One One-Euro filter per joint's 3D angular velocity (initial = 0).
    omega_filters = [OneEuroFilter(0.0, np.zeros(3), min_cutoff=min_cutoff, beta=beta) for _ in range(n_joints)]

    R_filtered_prev = [Rotation.from_rotvec(params_seq[0, 3+i*3 : 3+i*3+3]) for i in range(n_joints)]

    for f in range(1, num_frames):
        # Filter position
        filtered[f, :3] = pos_filter(f * dt, params_seq[f, :3])

        for i in range(n_joints):
            R_raw_current = Rotation.from_rotvec(params_seq[f, 3+i*3 : 3+i*3+3])

            # Delta rotation from filtered prev to raw current
            delta_R = R_raw_current * R_filtered_prev[i].inv()
            omega_raw = delta_R.as_rotvec() / dt

            # Filter the angular velocity
            omega_filtered = omega_filters[i](f * dt, omega_raw)

            # Integrate
            R_filtered_current = Rotation.from_rotvec(omega_filtered * dt) * R_filtered_prev[i]

            filtered[f, 3+i*3 : 3+i*3+3] = R_filtered_current.as_rotvec()
            R_filtered_prev[i] = R_filtered_current

    return filtered

