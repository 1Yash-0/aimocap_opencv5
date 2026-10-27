"""Tests for the knee-flexion movement analysis shown on result pages."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from service import insights
from service.insights import KneeAnalysis, analyse, chart_svg, knee_flexion, summary_rows, tracking_error_deg

IDX = insights.IDX


def _leg_points(frames: int, left_deg: float, right_deg: float) -> np.ndarray:
    """(frames, 12, 3) joints with each leg bent by the given flexion, in the 12-joint layout.

    The thigh points up from the knee; the shank rotates by the flexion about the x axis.
    Knee angles are invariant to the orientation of the frame, so the layout is exact.
    """
    points = np.zeros((frames, len(insights.JOINT_NAMES), 3))
    for side, x, deg in (("L", -0.1, left_deg), ("R", 0.1, right_deg)):
        phi = np.radians(deg)
        hip = np.array([x, 0.0, 0.0])
        knee = hip + np.array([0.0, -0.45, 0.0])
        ankle = knee + 0.45 * np.array([0.0, -np.cos(phi), np.sin(phi)])
        points[:, IDX[f"{side}_hip"]] = hip
        points[:, IDX[f"{side}_knee"]] = knee
        points[:, IDX[f"{side}_ankle"]] = ankle
    return points


def _analysis(frames: int = 120, mocap: bool = True) -> KneeAnalysis:
    seconds = np.arange(frames) / 60.0
    ours = {"L": np.full(frames, 60.0), "R": np.full(frames, 50.0)}
    truth = {"L": np.full(frames, 61.0), "R": np.full(frames, 49.0)} if mocap else None
    return KneeAnalysis(seconds=seconds, ours=ours, mocap=truth)


def test_knee_flexion_is_zero_for_a_straight_leg_and_ninety_for_a_right_angle():
    straight = knee_flexion(_leg_points(1, 0.0, 0.0), "L")
    bent = knee_flexion(_leg_points(1, 90.0, 90.0), "R")
    assert np.allclose(straight, 0.0, atol=1e-6)
    assert np.allclose(bent, 90.0, atol=1e-6)


def test_summary_rows_report_peaks_and_left_right_difference():
    rows = summary_rows(_analysis())
    assert [row["source"] for row in rows] == ["AIMoCap (phones)", "Marker motion capture (lab)"]
    assert rows[0]["left"] == "60" and rows[0]["right"] == "50"
    assert rows[0]["asymmetry"] == "10.0"
    assert rows[0]["peak_difference"] == "10.0"
    assert rows[1]["asymmetry"] == "12.0"


def test_peak_difference_does_not_cancel_alternating_walking_cycles_and_handles_nans():
    left = np.tile([0.0, 90.0], 60)
    right = np.tile([90.0, 0.0], 60)
    alternating = KneeAnalysis(seconds=np.arange(120) / 60.0, ours={"L": left, "R": right}, mocap=None)
    row = summary_rows(alternating)[0]
    assert row["asymmetry"] == "90.0"
    assert row["peak_difference"] == "0.0"

    missing = KneeAnalysis(
        seconds=np.arange(2) / 60.0,
        ours={"L": np.full(2, np.nan), "R": np.full(2, np.nan)},
        mocap=None,
    )
    assert summary_rows(missing)[0]["peak_difference"] == "n/a"


def test_tracking_error_is_the_median_absolute_difference():
    assert tracking_error_deg(_analysis()) == "1.0"
    assert tracking_error_deg(_analysis(mocap=False)) is None
    assert len(summary_rows(_analysis(mocap=False))) == 1


def test_chart_draws_every_series_with_non_empty_paths():
    svg = chart_svg(_analysis())
    assert svg.startswith("<svg") and 'role="img"' in svg
    for css in ("ours-l", "ours-r", "mocap-l", "mocap-r"):
        assert f'class="line {css}"' in svg
    paths = re.findall(r' d="([^"]*)"', svg)
    assert len(paths) == 4
    assert all(d.startswith("M") and "L" in d for d in paths)


def test_chart_starts_a_new_segment_after_a_gap_and_omits_mocap_when_absent():
    result = _analysis(frames=60, mocap=False)
    result.ours["L"][20:25] = np.nan
    svg = chart_svg(result)
    path_d = re.search(r'class="line ours-l" d="([^"]*)"', svg).group(1)
    assert path_d.count("M") == 2
    assert "mocap-" not in svg


def test_chart_handles_a_series_with_no_finite_values():
    result = _analysis(frames=30, mocap=False)
    result.ours["L"][:] = np.nan
    result.ours["R"][:] = np.nan
    svg = chart_svg(result)
    assert 'class="line ours-l" d=""' in svg


def test_analyse_returns_none_without_a_skeleton_file(tmp_path: Path):
    assert analyse(tmp_path / "missing.npz", None) is None
