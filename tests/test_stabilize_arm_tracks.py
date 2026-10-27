"""Regression for Stage-2 arm endpoint filtering and fast-motion adaptation."""
import numpy as np

from aimocap.retarget.temporal import stabilize_mocap_fit_targets

FPS = 29.97002997002997
LOWER = ("pelvis", "hip_l", "knee_l", "ankle_l", "hip_r", "knee_r", "ankle_r")
ARMS = ("shoulder_l", "shoulder_r", "elbow_l", "elbow_r", "wrist_l", "wrist_r")


def _track(base, frames=120):
    out = np.tile(np.asarray(base, dtype=float), (frames, 1))
    start = 40
    frequency = 0.5
    amplitude = 150.0 / (2.0 * np.pi * frequency)
    t = (np.arange(frames) - start) / FPS
    out[start:, 0] += amplitude * np.sin(2.0 * np.pi * frequency * t[start:])
    return out


def _tracks():
    tracks = {
        "pelvis": _track((0.0, 0.0, 90.0)),
        "hip_l": _track((-10.0, 0.0, 90.0)),
        "knee_l": _track((-12.0, 0.0, 48.0)),
        "ankle_l": _track((-12.0, 0.0, 5.0)),
        "hip_r": _track((10.0, 0.0, 90.0)),
        "knee_r": _track((12.0, 0.0, 48.0)),
        "ankle_r": _track((12.0, 0.0, 5.0)),
    }
    rng = np.random.default_rng(147)
    bases = {
        "shoulder_l": (-20.0, 0.0, 140.0), "elbow_l": (-42.0, 0.0, 100.0),
        "wrist_l": (-64.0, 0.0, 100.0), "shoulder_r": (20.0, 0.0, 140.0),
        "elbow_r": (42.0, 0.0, 100.0), "wrist_r": (64.0, 0.0, 100.0),
    }
    for name, base in bases.items():
        track = _track(base)
        signs = rng.choice(np.array([-1.0, 1.0]), size=30)
        track[10:40, 0] += 0.6 * signs  # 1.2 cm peak-to-peak held noise
        tracks[name] = track
    return tracks


def _median5_residual_p95(track, frame_slice):
    x = np.asarray(track, dtype=float)
    padded = np.pad(x, ((2, 2), (0, 0)), mode="edge")
    baseline = np.median(np.lib.stride_tricks.sliding_window_view(padded, 5, axis=0), axis=-1)
    magnitude = np.linalg.norm(x - baseline, axis=1)
    return float(np.percentile(magnitude[frame_slice], 95))


def _speed_p95(track, frame_slice):
    speed = np.linalg.norm(np.diff(track, axis=0), axis=1) * FPS
    return float(np.percentile(speed[frame_slice], 95))


def test_arm_hf_is_attenuated_fast_motion_preserved_and_legs_unchanged():
    tracks = _tracks()
    filtered, diag = stabilize_mocap_fit_targets(
        {k: v.copy() for k, v in tracks.items()}, fps=FPS, cutoff_hz=1.5, adaptive=True)
    repeated, _ = stabilize_mocap_fit_targets(
        {k: v.copy() for k, v in tracks.items()}, fps=FPS, cutoff_hz=1.5, adaptive=True)

    held = slice(12, 38)
    for name in ARMS:
        before = _median5_residual_p95(tracks[name], held)
        after = _median5_residual_p95(filtered[name], held)
        assert before > 0.0
        assert after < 0.5 * before, (name, before, after)
        assert np.array_equal(filtered[name], repeated[name])

    # Exclude the motion onset/endpoint; the fast sinusoid is ~150 cm/s.
    fast_pairs = slice(45, 114)
    before_speed = _speed_p95(tracks["wrist_l"], fast_pairs)
    after_speed = _speed_p95(filtered["wrist_l"], fast_pairs)
    assert abs(after_speed - before_speed) / before_speed < 0.15
    assert diag["arm_effective_cutoff_hz"] > diag["cutoff_hz"]

    # The adaptive cutoff that feeds the original leg filters remains based on
    # the original lower-body set; adding arm tracks is bit-identical for legs.
    leg_only, _ = stabilize_mocap_fit_targets(
        {name: tracks[name].copy() for name in LOWER},
        fps=FPS, cutoff_hz=1.5, adaptive=True)
    for name in LOWER:
        assert np.array_equal(filtered[name], leg_only[name]), name

    # Missing optional arm tracks are skipped; returned keys stay unchanged.
    partial = {name: tracks[name].copy() for name in LOWER}
    partial["shoulder_l"] = tracks["shoulder_l"].copy()
    partial_out, partial_diag = stabilize_mocap_fit_targets(partial, fps=FPS)
    assert partial_diag["enabled"] is True
    assert partial_out.keys() == partial.keys()
