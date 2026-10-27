"""Frame-rate resolution for the ``triangulate`` CLI.

Regression for the silent 30 fps default: the triangulation smoother and the
legacy kinematics filters must use the capture rate of the input, and the rate
(and where it came from) must be recorded in the provenance manifest.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import aimocap.math.clusters as clusters_module
import aimocap.math.filter as filter_module
import aimocap.triangulate.engine as engine_module
from aimocap import cli
from aimocap.calib.io import save_calibration

NUM_FRAMES = 12
NUM_JOINTS = 133


def _write_calibration(path: Path) -> Path:
    """Two cameras, identity rotation, 1 m baseline, production metric scale."""
    K = np.array([[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]])
    save_calibration(
        path,
        [K, K.copy()],
        [(np.eye(3), np.zeros((3, 1))), (np.eye(3), np.array([[-1.0], [0.0], [0.0]]))],
        image_sizes=[(1920, 1080), (1920, 1080)],
        camera_names=["cam0", "cam1"],
        units="meters",
        calibration_source="checkerboard",
        accuracy_class="production",
        scale_source="checkerboard",
    )
    return path


def _keypoints() -> np.ndarray:
    """Project a slowly drifting random 133-joint cloud into both cameras."""
    rng = np.random.default_rng(0)
    base = rng.uniform(-0.4, 0.4, size=(NUM_JOINTS, 3))
    base[:, 2] += 3.0  # 3 m in front of both cameras
    K = np.array([[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]])
    kpts = np.zeros((NUM_FRAMES, 2, NUM_JOINTS, 2), dtype=np.float32)
    for f in range(NUM_FRAMES):
        pts = base + 0.01 * rng.standard_normal((NUM_JOINTS, 3))
        for c, t in enumerate((np.zeros(3), np.array([-1.0, 0.0, 0.0]))):
            cam = pts + t
            uv = (K @ (cam / cam[:, 2:3]).T).T[:, :2]
            kpts[f, c] = uv.astype(np.float32)
    return kpts


def _write_pose_npz(path: Path, **extra) -> Path:
    """Write a pose NPZ; ``extra`` adds keys such as fps or timestamps."""
    np.savez(
        path,
        keypoints=_keypoints(),
        scores=np.ones((NUM_FRAMES, 2, NUM_JOINTS), dtype=np.float32),
        image_sizes=np.array([[1920, 1080], [1920, 1080]]),
        camera_names=np.array(["cam0", "cam1"]),
        **extra,
    )
    return path


def _triangulate_argv(npz: Path, calib: Path, out: Path, *extra: str) -> list[str]:
    return [
        "triangulate", str(npz), str(calib), "-o", str(out),
        "--allow-unverified-sync", "--allow-unsafe-geometry", *extra,
    ]


def _manifest_config(out: Path) -> dict:
    sidecar = out.with_suffix(out.suffix + ".manifest.json")
    return json.loads(sidecar.read_text(encoding="utf-8"))["all_config_values"]


@pytest.fixture
def smoother_fps(monkeypatch) -> list[float]:
    """Record the fps the RTS smoother is called with (the call still runs)."""
    seen: list[float] = []
    real = engine_module.filter_skeleton3d_rts

    def spy(points, fps, *args, **kwargs):
        seen.append(float(fps))
        return real(points, fps, *args, **kwargs)

    monkeypatch.setattr(engine_module, "filter_skeleton3d_rts", spy)
    return seen


def test_npz_fps_60_drives_smoother_and_manifest(tmp_path, smoother_fps):
    calib = _write_calibration(tmp_path / "calib.json")
    npz = _write_pose_npz(tmp_path / "pose.npz", fps=np.asarray(60.0))
    out = tmp_path / "skeleton3d.npz"

    assert cli.main(_triangulate_argv(npz, calib, out)) == 0

    assert smoother_fps, "RTS smoother was not called"
    assert all(f == 60.0 for f in smoother_fps)
    assert 1.0 / smoother_fps[0] == pytest.approx(1.0 / 60.0)
    cfg = _manifest_config(out)
    assert cfg["fps"] == 60.0
    assert cfg["fps_source"] == "npz:fps"


def test_fps_derived_from_timestamps_when_fps_key_missing(tmp_path, smoother_fps):
    calib = _write_calibration(tmp_path / "calib.json")
    npz = _write_pose_npz(tmp_path / "pose.npz", timestamps=np.arange(NUM_FRAMES) / 60.0)
    out = tmp_path / "skeleton3d.npz"

    assert cli.main(_triangulate_argv(npz, calib, out)) == 0

    assert smoother_fps and all(f == pytest.approx(60.0) for f in smoother_fps)
    cfg = _manifest_config(out)
    assert cfg["fps"] == pytest.approx(60.0)
    assert cfg["fps_source"] == "npz:timestamps"


def test_missing_fps_fails_clearly_without_flag(tmp_path, capsys):
    calib = _write_calibration(tmp_path / "calib.json")
    npz = _write_pose_npz(tmp_path / "pose.npz")
    out = tmp_path / "skeleton3d.npz"

    assert cli.main(_triangulate_argv(npz, calib, out)) == 2

    err = capsys.readouterr().err
    assert "--fps" in err
    assert "capture frame rate" in err
    assert not out.exists()


def test_cli_fps_overrides_npz_fps(tmp_path, smoother_fps):
    calib = _write_calibration(tmp_path / "calib.json")
    npz = _write_pose_npz(tmp_path / "pose.npz", fps=np.asarray(30.0))
    out = tmp_path / "skeleton3d.npz"

    assert cli.main(_triangulate_argv(npz, calib, out, "--fps", "60")) == 0

    assert smoother_fps and all(f == 60.0 for f in smoother_fps)
    cfg = _manifest_config(out)
    assert cfg["fps"] == 60.0
    assert cfg["fps_source"] == "--fps"


def test_legacy_kinematics_uses_resolved_fps(tmp_path, monkeypatch):
    seen: dict[str, list[float]] = {"one_euro": [], "cluster": []}
    real_one_euro = filter_module.filter_params_one_euro_quaternion
    real_cluster = clusters_module.filter_cluster_relative

    def one_euro_spy(params_seq, *args, **kwargs):
        seen["one_euro"].append(float(kwargs["fps"]))
        return real_one_euro(params_seq, *args, **kwargs)

    def cluster_spy(*args, **kwargs):
        seen["cluster"].append(float(kwargs["fps"]))
        return real_cluster(*args, **kwargs)

    monkeypatch.setattr(filter_module, "filter_params_one_euro_quaternion", one_euro_spy)
    monkeypatch.setattr(clusters_module, "filter_cluster_relative", cluster_spy)

    calib = _write_calibration(tmp_path / "calib.json")
    npz = _write_pose_npz(tmp_path / "pose.npz", fps=np.asarray(60.0))
    out = tmp_path / "skeleton3d.npz"

    assert cli.main(_triangulate_argv(npz, calib, out, "--legacy-kinematics")) == 0

    assert seen["one_euro"] and all(f == 60.0 for f in seen["one_euro"])
    assert seen["cluster"] and all(f == 60.0 for f in seen["cluster"])


def test_resolver_rejects_non_positive_cli_fps(tmp_path):
    npz = _write_pose_npz(tmp_path / "pose.npz", fps=np.asarray(60.0))
    with np.load(npz) as data:
        for bad in (0.0, -30.0, float("nan"), float("inf")):
            with pytest.raises(ValueError, match="--fps must be a positive"):
                cli._resolve_triangulate_fps(data, npz, bad)


def test_resolver_uses_sync_manifest_referenced_by_provenance(tmp_path):
    sync_path = tmp_path / "sync.json"
    sync_path.write_text(json.dumps({"fps": 50.0, "reference_id": "cam0"}), encoding="utf-8")
    provenance = {"all_config_values": {"sync_manifest": str(sync_path)}}
    npz = _write_pose_npz(
        tmp_path / "pose.npz",
        provenance_manifest=np.asarray(json.dumps(provenance)),
    )
    with np.load(npz) as data:
        fps, source = cli._resolve_triangulate_fps(data, npz, None)
    assert fps == 50.0
    assert source.startswith("sync_manifest:")


def test_resolver_prefers_npz_fps_over_timestamps_and_sync(tmp_path):
    npz = _write_pose_npz(
        tmp_path / "pose.npz",
        fps=np.asarray(25.0),
        timestamps=np.arange(NUM_FRAMES) / 60.0,
    )
    with np.load(npz) as data:
        assert cli._resolve_triangulate_fps(data, npz, None) == (25.0, "npz:fps")
