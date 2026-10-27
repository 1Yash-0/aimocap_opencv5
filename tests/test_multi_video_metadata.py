"""Tests for 2D capture provenance emitted by the multi-video stage."""

from __future__ import annotations

import numpy as np
import pytest

from aimocap.pose import multi_video


class _FakeCapture:
    def __init__(self, _path):
        self.frames = [np.zeros((8, 12, 3), dtype=np.uint8) for _ in range(3)]
        self.index = 0

    def isOpened(self):
        return True

    def get(self, prop):
        if prop == multi_video.cv2.CAP_PROP_FPS:
            return 30.0
        if prop == multi_video.cv2.CAP_PROP_POS_FRAMES:
            return float(self.index)
        if prop == multi_video.cv2.CAP_PROP_POS_MSEC:
            return max(0, self.index - 1) * 1000.0 / 30.0
        return 0.0

    def read(self):
        if self.index >= len(self.frames):
            return False, None
        frame = self.frames[self.index]
        self.index += 1
        return True, frame

    def release(self):
        pass


class _SeekableCapture(_FakeCapture):
    def __init__(self, path):
        super().__init__(path)
        self.frames = [np.zeros((8, 12, 3), dtype=np.uint8) for _ in range(10)]

    def set(self, prop, value):
        if prop == multi_video.cv2.CAP_PROP_POS_MSEC:
            self.index = max(0, int(round(float(value) * 30.0 / 1000.0)))
            return True
        if prop == multi_video.cv2.CAP_PROP_POS_FRAMES:
            self.index = max(0, int(round(float(value))))
            return True
        return False


class _WrongTimestampCapture(_SeekableCapture):
    def get(self, prop):
        value = super().get(prop)
        if prop == multi_video.cv2.CAP_PROP_POS_MSEC:
            return value + 100.0
        return value


class _FakeEstimator:
    def estimate(self, _frame, pick="all"):
        return []


class _FakePose:
    def __init__(self, area, visible_body=17):
        self.area = area
        self.scores = np.ones(133, dtype=np.float32)


def test_single_subject_selection_rejects_ambiguous_visible_people():
    poses = [_FakePose(100.0), _FakePose(80.0)]

    subject, status = multi_video._select_single_subject(poses, threshold=0.3)

    assert subject is None
    assert status == "ambiguous"


def test_multi_video_writes_timing_and_sync_provenance(tmp_path, monkeypatch):
    source = tmp_path / "cam_a.mp4"
    source.write_bytes(b"placeholder")
    monkeypatch.setattr(multi_video.cv2, "VideoCapture", _FakeCapture)

    output = tmp_path / "poses.npz"
    multi_video.run_multi_video(
        [source],
        output,
        estimator=_FakeEstimator(),
        max_frames=2,
        progress_every=0,
    )

    with np.load(output) as data:
        assert data["keypoints"].shape == (2, 1, 133, 2)
        assert data["timestamps"].shape == (2,)
        assert data["source_frame_indices"].shape == (2, 1)
        assert data["source_timestamps"].shape == (2, 1)
        assert data["requested_timestamps"].shape == (2, 1)
        assert data["subject_status"].shape == (2, 1)
        assert data["fps"] == pytest.approx(30.0)
        assert str(data["sync_validation"]) == "{}"
        assert str(data["sync_status"]) == "UNVERIFIED_EQUAL_FRAME_INDEX"
        assert str(data["coordinate_space"]) == "image_pixels"


def test_multi_video_require_sync_rejects_unverified_capture(tmp_path, monkeypatch):
    source = tmp_path / "cam_a.mp4"
    source.write_bytes(b"placeholder")
    monkeypatch.setattr(multi_video.cv2, "VideoCapture", _FakeCapture)

    with pytest.raises(ValueError, match="synchronization manifest is required"):
        multi_video.run_multi_video(
            [source],
            tmp_path / "poses.npz",
            estimator=_FakeEstimator(),
            max_frames=1,
            require_sync=True,
        )


def test_multi_video_uses_manifest_reference_when_it_is_not_first(
    tmp_path, monkeypatch
):
    first = tmp_path / "cam_a.mp4"
    reference = tmp_path / "cam_b.mp4"
    first.write_bytes(b"placeholder")
    reference.write_bytes(b"placeholder")
    monkeypatch.setattr(multi_video.cv2, "VideoCapture", _SeekableCapture)

    manifest = {
        "status": "VERIFIED_AUDIO_CLAP",
        "reference_id": "cam_b",
        "source_paths": {"cam_a": str(first), "cam_b": str(reference)},
        "cameras": {
            "cam_a": {"offset_ms": -100.0},
            "cam_b": {"offset_ms": 0.0},
        },
    }
    output = tmp_path / "poses.npz"
    multi_video.run_multi_video(
        [first, reference],
        output,
        estimator=_FakeEstimator(),
        max_frames=1,
        progress_every=0,
        sync_manifest=manifest,
        require_sync=True,
    )

    with np.load(output) as data:
        assert str(data["sync_status"]) == "VERIFIED"
        assert str(data["sync_reference_camera"]) == "cam_b"
        assert data["sync_camera_ids"].tolist() == ["cam_a", "cam_b"]
        assert "max_abs_error_s" in str(data["sync_validation"])
        assert data["requested_timestamps"][0, 0] == pytest.approx(0.1)
        assert data["requested_timestamps"][0, 1] == pytest.approx(0.0)
        assert data["source_timestamps"][0, 0] == pytest.approx(0.1)
        assert data["source_timestamps"][0, 1] == pytest.approx(0.0)


def test_multi_video_rejects_manifest_without_every_input_camera(tmp_path, monkeypatch):
    first = tmp_path / "cam_a.mp4"
    second = tmp_path / "cam_b.mp4"
    first.write_bytes(b"placeholder")
    second.write_bytes(b"placeholder")
    monkeypatch.setattr(multi_video.cv2, "VideoCapture", _SeekableCapture)

    with pytest.raises(ValueError, match="does not map every input camera"):
        multi_video.run_multi_video(
            [first, second],
            tmp_path / "poses.npz",
            estimator=_FakeEstimator(),
            max_frames=1,
            progress_every=0,
            sync_manifest={
                "status": "VERIFIED_AUDIO_CLAP",
                "reference_id": "cam_a",
                "source_paths": {"cam_a": str(first)},
                "cameras": {"cam_a": {"offset_ms": 0.0}},
            },
            require_sync=True,
        )


def test_multi_video_skips_unavailable_head_for_positive_camera_offset(
    tmp_path, monkeypatch
):
    first = tmp_path / "cam_a.mp4"
    reference = tmp_path / "cam_b.mp4"
    first.write_bytes(b"placeholder")
    reference.write_bytes(b"placeholder")
    monkeypatch.setattr(multi_video.cv2, "VideoCapture", _SeekableCapture)

    output = tmp_path / "poses.npz"
    multi_video.run_multi_video(
        [first, reference],
        output,
        estimator=_FakeEstimator(),
        max_frames=1,
        progress_every=0,
        sync_manifest={
            "status": "VERIFIED_AUDIO_CLAP",
            "reference_id": "cam_b",
            "source_paths": {"cam_a": str(first), "cam_b": str(reference)},
            "cameras": {
                "cam_a": {"offset_ms": 100.0},
                "cam_b": {"offset_ms": 0.0},
            },
        },
        require_sync=True,
    )

    with np.load(output) as data:
        assert data["requested_timestamps"][0, 1] == pytest.approx(0.1)
        assert data["requested_timestamps"][0, 0] == pytest.approx(0.0)


def test_multi_video_rejects_verified_manifest_when_decode_timestamps_drift(
    tmp_path, monkeypatch
):
    source = tmp_path / "cam_a.mp4"
    source.write_bytes(b"placeholder")
    monkeypatch.setattr(multi_video.cv2, "VideoCapture", _WrongTimestampCapture)

    with pytest.raises(ValueError, match="timestamp error"):
        multi_video.run_multi_video(
            [source],
            tmp_path / "poses.npz",
            estimator=_FakeEstimator(),
            max_frames=1,
            progress_every=0,
            sync_manifest={
                "status": "VERIFIED_AUDIO_CLAP",
                "reference_id": "cam_a",
                "source_paths": {"cam_a": str(source)},
                "cameras": {"cam_a": {"offset_ms": 0.0}},
            },
            require_sync=True,
        )
