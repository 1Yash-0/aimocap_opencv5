"""Trial extraction from an archive, and the crop-to-source pixel mapping."""

from __future__ import annotations

import json
import pickle
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from evaluation.opencap.crops import ASPECT, largest_blob_box, map_keypoints
from evaluation.opencap.dataset import CAMERAS, prepare_trial, read_trial

ROOT = "LabValidation_withVideos"
SUBJECT = "subject9"
TRIAL = "trial1"


def _tiny_video(path: Path, frames: int = 12) -> bytes:
    """Write a small 60 fps MJPG video and return its bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 60.0, (64, 48))
    for i in range(frames):
        writer.write(np.full((48, 64, 3), i * 10 % 255, dtype=np.uint8))
    writer.release()
    return path.read_bytes()


def _build_archive(tmp: Path) -> Path:
    archive = tmp / "archive.zip"
    scratch = tmp / "scratch"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(f"{ROOT}/{SUBJECT}/sessionMetadata.yaml",
                    "checkerBoard:\n  squareSideLength_mm: 35.0\n")
        for cam in CAMERAS:
            base = f"{ROOT}/{SUBJECT}/VideoData/Session0/{cam}"
            zf.writestr(f"{base}/cameraIntrinsicsExtrinsics.pickle",
                        pickle.dumps({"intrinsicMat": np.eye(3), "distortion": np.zeros(5),
                                      "rotation": np.eye(3), "translation": np.zeros((3, 1))}))
            zf.writestr(f"{base}/extrinsics/extrinsics.avi",
                        _tiny_video(scratch / f"{cam}_extr.avi"))
            zf.writestr(f"{base}/{TRIAL}/{TRIAL}.avi", _tiny_video(scratch / f"{cam}_raw.avi"))
            zf.writestr(f"{base}/{TRIAL}/{TRIAL}_syncdWithMocap.avi",
                        _tiny_video(scratch / f"{cam}_syncd.avi", frames=6))
            zf.writestr(f"{base}/other/other.avi", b"not needed")
        zf.writestr(f"{ROOT}/{SUBJECT}/MarkerData/Mocap/{TRIAL}.trc", "fixture\n")
        zf.writestr(f"{ROOT}/{SUBJECT}/MarkerData/Video/HRNet/3-cameras/{TRIAL}_videoAndMocap.trc", "fixture\n")
    return archive


def test_prepare_trial_extracts_only_needed_members_and_fixes_board(tmp_path: Path) -> None:
    archive = _build_archive(tmp_path)
    out = tmp_path / "trial_out"
    info = prepare_trial(archive, SUBJECT, TRIAL, out)
    assert info["session"] == "Session0"
    assert info["fps"] == 60.0
    assert info["board"]["inner_corners"] == [11, 8]
    assert info["board"]["square_mm"] == 60.0
    assert info["board"]["metadata_square_mm"] == 35.0
    assert any("35" in note for note in info["notes"])
    assert info["has_mocap"] is True
    assert info["files"]["mocap"] == f"mocap/{TRIAL}.trc"
    assert set(info["files"]["videos"]) == set(CAMERAS)
    assert set(info["files"]["syncd"]) == set(CAMERAS)
    assert set(info["files"]["hrnet"]) == {"3"}
    assert not (out / "other").exists()
    for cam in CAMERAS:
        assert (out / "videos" / f"{cam}.avi").exists()
        assert (out / "intrinsics" / f"{cam}.pickle").exists()
    assert read_trial(out)["trial"] == TRIAL
    assert json.loads((out / "trial.json").read_text(encoding="utf-8"))["cams"] == list(CAMERAS)


def test_missing_trial_fails_closed(tmp_path: Path) -> None:
    archive = _build_archive(tmp_path)
    with pytest.raises(FileNotFoundError):
        prepare_trial(archive, SUBJECT, "absent_trial", tmp_path / "missing")


def test_crop_keypoints_map_back_to_source_pixels() -> None:
    params = {"step": 3, "start_frame": 10, "row0": 3, "out_w": 422, "out_h": 768}
    rows = np.arange(40, dtype=float)
    cam_map = np.stack([100.0 + rows, 50.0 + 2 * rows, np.full_like(rows, 0.5)], axis=1)
    keypoints = np.array([[[211.0, 384.0], [0.0, 768.0]]] * 4)  # four frames, two joints
    mapped = map_keypoints(keypoints, params, cam_map)
    sy_scale = 0.5 * 422 / (ASPECT * 768)
    for i in range(4):
        row = (10 + i) // 3 - 3
        expected_x = keypoints[i, :, 0] * 0.5 + cam_map[row, 0]
        expected_y = keypoints[i, :, 1] * sy_scale + cam_map[row, 1]
        assert np.allclose(mapped[i, :, 0], expected_x)
        assert np.allclose(mapped[i, :, 1], expected_y)


def test_blob_box_finds_a_moving_block() -> None:
    diff = np.zeros((90, 160), dtype=np.uint8)
    diff[30:60, 40:80] = 200
    box = largest_blob_box(diff, scale=1.0)
    assert box is not None
    x0, y0, x1, y1 = box
    assert 35 <= x0 <= 45 and 25 <= y0 <= 35
    assert 75 <= x1 <= 85 and 55 <= y1 <= 65
