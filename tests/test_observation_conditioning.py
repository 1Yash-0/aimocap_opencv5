import numpy as np

from aimocap.retarget.observation_conditioning import (
    ConditioningConfig,
    condition_target_tracks,
)


def test_static_white_noise_is_reduced_more_than_fivefold():
    rng = np.random.default_rng(1001)
    observed = rng.normal(0.0, 0.5, size=(300, 3))
    conditioned, _ = condition_target_tracks({"point": observed}, fps=60.0)
    assert np.std(observed) / np.std(conditioned["point"]) > 5.0


def test_fast_min_jerk_reach_preserves_peak_speed_and_beats_noise_rms():
    rng = np.random.default_rng(1002)
    fps = 60.0
    frames = 300
    t = np.arange(frames, dtype=np.float64) / fps
    duration = 0.8
    u = np.clip(t / duration, 0.0, 1.0)
    profile = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
    velocity = (30.0 * u**2 - 60.0 * u**3 + 30.0 * u**4) / duration
    profile *= 300.0 / np.max(velocity)
    truth = np.zeros((frames, 3), dtype=np.float64)
    truth[:, 0] = profile
    observed = truth + rng.normal(0.0, 0.5, size=truth.shape)
    conditioned, _ = condition_target_tracks({"point": observed}, fps=fps)
    peak_truth = np.max(np.linalg.norm(np.diff(truth, axis=0), axis=1) * fps)
    peak_output = np.max(np.linalg.norm(np.diff(conditioned["point"], axis=0), axis=1) * fps)
    rms = np.sqrt(np.mean((conditioned["point"] - truth) ** 2))
    assert abs(peak_output - peak_truth) / peak_truth < 0.10
    assert rms < 0.5


def test_single_frame_outliers_are_rejected():
    rng = np.random.default_rng(1003)
    observed = rng.normal(0.0, 0.5, size=(300, 3))
    outlier_frames = np.arange(0, len(observed), 50)
    observed[outlier_frames] += 20.0
    conditioned, diagnostics = condition_target_tracks({"point": observed}, fps=60.0)
    error = np.linalg.norm(conditioned["point"][outlier_frames], axis=1)
    assert np.all(error < 2.0 * 0.5)
    assert diagnostics["tracks"]["point"]["outlier_count"] == len(outlier_frames)


def test_nan_gaps_are_bridged_and_long_gap_is_flagged():
    observed = np.zeros((80, 3), dtype=np.float64)
    observed[:, 0] = np.arange(80, dtype=np.float64)
    observed[25:50] = np.nan
    conditioned, diagnostics = condition_target_tracks({"point": observed}, fps=30.0)
    assert np.isfinite(conditioned["point"]).all()
    track_diag = diagnostics["tracks"]["point"]
    assert track_diag["bridged_frames"] == 25
    assert track_diag["long_gap_frames"] == 25


def test_conditioning_is_bitwise_deterministic():
    rng = np.random.default_rng(1004)
    observed = rng.normal(size=(100, 3))
    observed[10:14] = np.nan
    first, first_diag = condition_target_tracks({"point": observed}, fps=29.97)
    second, second_diag = condition_target_tracks({"point": observed}, fps=29.97)
    assert np.array_equal(first["point"], second["point"])
    assert first_diag == second_diag


def test_pelvis_and_neck_are_recomputed_from_conditioned_base_tracks():
    frames = 120
    rng = np.random.default_rng(1005)
    hip_l = rng.normal(size=(frames, 3))
    hip_r = rng.normal(size=(frames, 3))
    shoulder_l = rng.normal(size=(frames, 3))
    shoulder_r = rng.normal(size=(frames, 3))
    conditioned, diagnostics = condition_target_tracks(
        {
            "hip_l": hip_l,
            "hip_r": hip_r,
            "shoulder_l": shoulder_l,
            "shoulder_r": shoulder_r,
            "pelvis": np.full((frames, 3), 999.0),
            "neck": np.full((frames, 3), 999.0),
        },
        fps=30.0,
        config=ConditioningConfig(),
    )
    assert np.array_equal(conditioned["pelvis"], (conditioned["hip_l"] + conditioned["hip_r"]) * 0.5)
    assert np.array_equal(conditioned["neck"], (conditioned["shoulder_l"] + conditioned["shoulder_r"]) * 0.5)
    assert diagnostics["derived_tracks"] == {
        "pelvis": "mean(hip_l, hip_r) after conditioning",
        "neck": "mean(shoulder_l, shoulder_r) after conditioning",
    }



def test_local_q_preserves_late_fast_reach_after_slow_section():
    rng = np.random.default_rng(9001)
    fps = 60.0
    frames = 1260  # 20 s slow section followed by a 1 s reach
    t = np.arange(frames, dtype=np.float64) / fps
    u = np.clip((t - 20.0) / 1.0, 0.0, 1.0)
    profile = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
    truth = np.zeros((frames, 3), dtype=np.float64)
    truth[:, 0] = 160.0 * profile  # peak speed is 300 cm/s
    observed = truth + rng.normal(0.0, 0.5, size=truth.shape)
    conditioned, _ = condition_target_tracks({"point": observed}, fps=fps)
    output = conditioned["point"]
    truth_speed = np.linalg.norm(np.diff(truth, axis=0), axis=1) * fps
    output_speed = np.linalg.norm(np.diff(output, axis=0), axis=1) * fps
    peak_truth = np.max(truth_speed[1200:])
    peak_output = np.max(output_speed[1200:])
    assert abs(peak_output - peak_truth) / peak_truth < 0.10
    raw_noise = np.std(observed[:1200, 0] - truth[:1200, 0])
    output_noise = np.std(output[:1200, 0] - truth[:1200, 0])
    assert raw_noise / output_noise > 4.0
