"""Integration tests: temporal single-subject tracking inside run_multi_video.

Regression for the frame-191 failure mode: a single-subject capture gate raised
"ambiguous performer selection" when a detector emitted two visible people on
one frame.  The temporal tracker must keep the tracked identity instead of
raising or switching to the larger newcomer.
"""

from __future__ import annotations

import numpy as np
import pytest

from aimocap.pose import multi_video


class _Cap:
    def __init__(self, _path, n=6):
        self.frames = [np.zeros((8, 12, 3), dtype=np.uint8) for _ in range(n)]
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


def _person(cx, cy, conf=0.9):
    kp = np.zeros((133, 2), dtype=np.float32)
    sc = np.zeros((133,), dtype=np.float32)
    base = np.array([
        [0, -90], [-15, -95], [15, -95], [-25, -95], [25, -95],
        [-40, -40], [40, -40], [-55, 30], [55, 30],
        [-60, 95], [60, 95], [-20, 90], [20, 90],
        [-25, 160], [25, 160], [-25, 230], [25, 230],
    ], dtype=np.float32)
    kp[:17] = base + np.array([cx, cy], dtype=np.float32)
    sc[:17] = conf
    pose = type("P", (), {})()
    pose.keypoints = kp
    pose.scores = sc
    valid = sc[:17] > 0
    pts = kp[:17][valid]
    pose.bbox = np.array([pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()])
    pose.area = float((pose.bbox[2] - pose.bbox[0]) * (pose.bbox[3] - pose.bbox[1]))
    return pose


class _AmbigEstimator:
    """Lock onto subject at x=200, but for one middle frame also report a bigger
    distractor at the far right.  Temporal tracker must keep x=200 identity."""

    def estimate(self, _frame, pick="all"):
        # call count encoded by reading a counter the test drives externally
        idx = _AmbigEstimator.calls
        _AmbigEstimator.calls += 1
        poses = [_person(200, 100)]
        if idx == 3:  # ambiguous frame
            poses.append(_person(560, 300))  # second visible, different location
        return poses


_AmbigEstimator.calls = 0


def test_temporal_tracker_handles_ambiguous_frame_without_raising(tmp_path, monkeypatch):
    monkeypatch.setattr(multi_video.cv2, "VideoCapture", _Cap)
    _AmbigEstimator.calls = 0
    src = tmp_path / "cam_a.mp4"
    src.write_bytes(b"x")
    out = tmp_path / "poses.npz"
    multi_video.run_multi_video(
        [src], out, estimator=_AmbigEstimator(), max_frames=6,
        progress_every=0, single_subject=True, tracker="temporal",
    )
    with np.load(out) as d:
        assert str(d["subject_selection"]) == "temporal"
        # Wrist (joint 9) x should stay near the tracked subject (~140-200), not jump to 560.
        wrist_x = d["keypoints"][:, 0, 9, 0]
        assert np.all(np.isfinite(wrist_x))
        assert wrist_x.max() - wrist_x.min() < 200  # no jump to the distractor


def test_strict_tracker_still_raises_on_ambiguity(tmp_path, monkeypatch):
    monkeypatch.setattr(multi_video.cv2, "VideoCapture", _Cap)
    _AmbigEstimator.calls = 0
    src = tmp_path / "cam_a.mp4"
    src.write_bytes(b"x")
    with pytest.raises(ValueError, match="ambiguous"):
        multi_video.run_multi_video(
            [src], tmp_path / "p.npz", estimator=_AmbigEstimator(),
            max_frames=6, progress_every=0, single_subject=True, tracker="strict",
        )


def test_temporal_is_default_tracker(tmp_path, monkeypatch):
    monkeypatch.setattr(multi_video.cv2, "VideoCapture", _Cap)
    _AmbigEstimator.calls = 0
    src = tmp_path / "cam_a.mp4"
    src.write_bytes(b"x")
    out = tmp_path / "poses.npz"
    multi_video.run_multi_video(
        [src], out, estimator=_AmbigEstimator(), max_frames=6,
        progress_every=0, single_subject=True,
    )
    with np.load(out) as d:
        assert str(d["subject_selection"]) == "temporal"
