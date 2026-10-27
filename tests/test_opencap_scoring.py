"""Scoring on synthetic tracks, TRC parsing, and the stage event log."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from evaluation.opencap.events import EventLog
from evaluation.opencap.mocap import JOINT_NAMES, load_trc, mocap_joints, resample_rate
from evaluation.opencap.scoring import global_rigid_error, lag0_accuracy, smoothness

RATE = 100.0
STEP = RATE / 60.0


def _rotation(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _mocap(rows: int = 1200) -> np.ndarray:
    """Smooth synthetic virtual-joint trajectories at 100 Hz, shape (rows, 12, 3), mm."""
    rng = np.random.default_rng(11)
    t = np.arange(rows) / RATE
    base = rng.uniform(-300.0, 300.0, size=(12, 3))
    phase = rng.uniform(0.0, 2 * np.pi, size=(12, 3))
    freq = rng.uniform(0.5, 1.5, size=(12, 3))
    return base + 80.0 * np.sin(2 * np.pi * freq * t[:, None, None] + phase)


def _window(mocap: np.ndarray, frames: int) -> np.ndarray:
    """The 60 Hz window that lag zero compares against the mocap."""
    rows = np.arange(frames) * STEP
    return np.stack([np.interp(rows, np.arange(len(mocap)), mocap[:, j, a])
                     for j in range(12) for a in range(3)], axis=-1).reshape(frames, 12, 3)


def test_exact_rigid_copy_scores_zero_error_and_unit_speed_ratio() -> None:
    mocap = _mocap()
    ours = _window(mocap, frames=600)
    R = _rotation(0.4)
    ours_moved = ours @ R.T + np.array([120.0, -40.0, 900.0])
    lag0 = lag0_accuracy(ours_moved, mocap, STEP)
    assert lag0 is not None and lag0["median_mm"] < 1e-6 and lag0["frames"] == 600
    rigid = global_rigid_error(ours_moved, mocap, STEP)
    assert rigid is not None and rigid["median_mm"] < 1e-6
    # Smoothness is evaluated in the frame shared with the mocap (no alignment), as in the reports.
    smooth = smoothness(ours, mocap, STEP)
    assert smooth["acceleration_error_median_mm_per_frame2"] < 1e-6
    assert smooth["peak_speed_ratio_p95"] == pytest.approx(1.0, rel=1e-9)
    rotated = smoothness(ours_moved, mocap, STEP)
    assert rotated["acceleration_error_median_mm_per_frame2"] > 0.0


def test_noise_increases_error_and_frames_outside_mocap_are_not_scored() -> None:
    mocap = _mocap(rows=1200)  # 12 s of mocap = 720 frames at 60 Hz
    frames = 760  # 40 frames extend past the recording
    clean = _window(mocap, frames=frames)
    rng = np.random.default_rng(3)
    noisy = clean + rng.normal(0.0, 8.0, size=clean.shape)
    lag0 = lag0_accuracy(noisy, mocap, STEP)
    assert lag0 is not None
    assert lag0["frames"] <= int(np.floor((len(mocap) - 1) / STEP)) + 1
    assert lag0["median_mm"] > 0.0
    smooth = smoothness(noisy, mocap, STEP)
    assert smooth["acceleration_error_median_mm_per_frame2"] > 0.0
    assert smooth["frames"] <= int(np.floor((len(mocap) - 1) / STEP)) + 1


def test_trc_round_trip_and_virtual_joints(tmp_path: Path) -> None:
    names = ["R_HJC", "L_HJC", "r_knee", "L_knee", "r_ankle", "L_ankle", "R_Shoulder", "L_Shoulder",
             "R_elbow_med", "R_elbow_lat", "L_elbow_med", "L_elbow_lat", "R_wrist_radius", "R_wrist_ulna",
             "L_wrist_radius", "L_wrist_ulna"]
    frames = 5
    rng = np.random.default_rng(5)
    data = rng.uniform(-500, 500, size=(frames, len(names), 3))
    header = [
        "PathFileType\t4\t(X/Y/Z)\tfixture.trc",
        "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames",
        f"{RATE:g}\t{RATE:g}\t{frames}\t{len(names)}\tmm\t{RATE:g}\t1\t{frames}",
        "Frame#\tTime\t" + "\t\t".join(names) + "\t",
        "\t\t" + "\t".join(f"X{i + 1}\tY{i + 1}\tZ{i + 1}" for i in range(len(names))),
    ]
    rows = []
    for f in range(frames):
        values = "\t".join(f"{v:.6f}" for v in data[f].reshape(-1))
        rows.append(f"{f + 1}\t{f / RATE:.6f}\t{values}")
    path = tmp_path / "fixture.trc"
    path.write_text("\n".join(header + rows) + "\n", encoding="utf-8")
    trc = load_trc(path)
    assert trc.rate == RATE
    assert trc.names == tuple(names)
    assert np.allclose(trc.data, data, atol=1e-5)
    joints = mocap_joints(trc)
    assert joints.shape == (frames, 12, 3)
    assert np.allclose(joints[:, JOINT_NAMES.index("R_hip")], data[:, 0])  # R_hip is R_HJC
    assert np.allclose(joints[:, JOINT_NAMES.index("R_wr")], data[:, 12:14].mean(axis=1))
    down = resample_rate(joints, RATE, 60.0)
    assert down.shape[0] == int(np.floor((frames - 1) * 60.0 / RATE)) + 1


def test_event_log_writes_started_and_completed(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    with EventLog(path) as log:
        with log.stage("alpha"):
            pass
        log.emit("beta", "started")
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [(r["stage"], r["status"]) for r in records] == [
        ("alpha", "started"), ("alpha", "completed"), ("beta", "started")]
    assert all(set(r) == {"ts", "stage", "status", "seconds", "message"} for r in records)
    assert records[1]["seconds"] >= 0.0
    assert records[0]["seconds"] is None


def test_event_log_records_failure_and_reraises(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    with pytest.raises(RuntimeError, match="boom"):
        with EventLog(path) as log:
            with log.stage("gamma"):
                raise RuntimeError("boom")
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert records[-1]["status"] == "failed"
    assert records[-1]["stage"] == "gamma"
    assert "RuntimeError: boom" in records[-1]["message"]


def test_event_log_rejects_unknown_status(tmp_path: Path) -> None:
    with EventLog(tmp_path / "events.jsonl") as log:
        with pytest.raises(ValueError):
            log.emit("delta", "running")
