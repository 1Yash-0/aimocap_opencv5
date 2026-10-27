"""Geometric synchronisation recovers known per-camera offsets from synthetic 2D tracks."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from aimocap.calib.io import save_calibration
from evaluation.opencap.sync import cut_window, estimate_sync, read_sync_file, write_sync_file

WIDTH, HEIGHT = 720, 1280
FOCAL = 900.0
TRUE_OFFSETS = {"Cam0": 0, "Cam1": 5, "Cam2": -7}
FRAMES = 260
JOINTS = 23


def _look_at(centre: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """World-to-camera rotation and translation for a camera at ``centre`` looking at ``target``.

    World y points down, matching the OpenCV camera frame.
    """
    z = target - centre
    z /= np.linalg.norm(z)
    x = np.cross(np.array([0.0, 1.0, 0.0]), z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    t = -R @ centre
    return R, t


def _body_trajectory(times: np.ndarray) -> np.ndarray:
    """Synthetic 23-joint body moving in 3D (millimetres); the motion gives depth cues."""
    rng = np.random.default_rng(7)
    base = rng.uniform(-250.0, 250.0, size=(JOINTS, 3))
    base[:, 1] = np.linspace(-900.0, 600.0, JOINTS)  # rough vertical spread, y down
    phase = rng.uniform(0.0, 2 * np.pi, size=JOINTS)
    amp = rng.uniform(40.0, 160.0, size=(JOINTS, 3))
    out = np.empty((len(times), JOINTS, 3))
    for i, tau in enumerate(times):
        out[i] = base + amp * np.sin(2 * np.pi * tau / 90.0 + phase)[:, None]
    return out


def _project(points: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    cam = points @ R.T + t.reshape(3)
    uv = (cam @ K.T)
    return uv[..., :2] / uv[..., 2:3]


def _write_fixture(tmp: Path) -> tuple[Path, Path]:
    K = np.array([[FOCAL, 0.0, WIDTH / 2], [0.0, FOCAL, HEIGHT / 2], [0.0, 0.0, 1.0]])
    names = list(TRUE_OFFSETS)
    centres = [np.array([3000.0, 0.0, 0.0]), np.array([0.0, 0.0, 3000.0]), np.array([-2400.0, 0.0, -2400.0])]
    rt = []
    for centre in centres:
        R, t = _look_at(centre, np.array([0.0, -200.0, 0.0]))
        rt.append((R, t.reshape(3, 1)))
    save_calibration(
        tmp / "calib.json",
        [K for _ in names],
        rt,
        image_sizes=[(WIDTH, HEIGHT)] * len(names),
        dist_coeffs=[np.zeros(5) for _ in names],
        camera_names=names,
        units="mm",
        calibration_source="checkerboard",
        accuracy_class="production",
        scale_source="checkerboard",
    )

    timeline = np.arange(FRAMES + 40, dtype=float) - 20.0
    world = _body_trajectory(timeline)
    keypoints = np.zeros((FRAMES, len(names), JOINTS, 2))
    for c, name in enumerate(names):
        R, t = rt[c]
        frame_times = np.arange(FRAMES) - TRUE_OFFSETS[name]  # camera frame f shows time f - offset
        idx = np.round(frame_times + 20.0).astype(int)
        keypoints[:, c] = _project(world[idx], K, R, t)
    scores = np.ones((FRAMES, len(names), JOINTS))
    npz = tmp / "multipose.npz"
    np.savez(
        npz,
        keypoints=keypoints,
        scores=scores,
        camera_names=np.asarray(names),
        image_sizes=np.asarray([[WIDTH, HEIGHT]] * len(names), dtype=np.int32),
        fps=np.asarray(60.0),
    )
    return npz, tmp / "calib.json"


def test_geometric_sync_recovers_known_offsets(tmp_path: Path) -> None:
    npz, calib = _write_fixture(tmp_path)
    sync = estimate_sync(
        npz,
        calib,
        epipolar_max_lag=30,
        triangulation_max_lag=30,
        rounds=2,
    )
    assert sync["uses_mocap"] is False and sync["uses_speed_signal"] is False
    assert sync["reference"] == "Cam0"
    for name, expected in TRUE_OFFSETS.items():
        estimated = sync["cameras"][name]["offset_vs_cam0"]
        assert abs(estimated - expected) <= 1, f"{name}: estimated {estimated}, expected {expected}"


def test_sync_file_round_trip_and_window_cut(tmp_path: Path) -> None:
    npz, calib = _write_fixture(tmp_path)
    sync = estimate_sync(npz, calib, epipolar_max_lag=30, triangulation_max_lag=30)
    sync["anchor"] = {"cam0_start_frame": 40, "source": "test", "resolved": True}
    path = write_sync_file(tmp_path / "sync.json", sync)
    loaded = read_sync_file(path)
    offsets = {name: int(v["offset_vs_cam0"]) for name, v in loaded["cameras"].items()}
    info = cut_window(npz, tmp_path / "window.npz", offsets, anchor_frame=40, n_frames=100,
                      sync_status="TEST")
    window = np.load(tmp_path / "window.npz", allow_pickle=True)
    assert window["keypoints"].shape[0] == info["frames"] == 100
    assert window["source_frame_indices"][0].tolist() == [40 + offsets[n] for n in window["camera_names"]]
    assert str(window["sync_status"]) == "TEST"
