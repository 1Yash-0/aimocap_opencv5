"""Tests for the temporally consistent single-subject tracker."""

from __future__ import annotations

import numpy as np

from aimocap.pose.tracker import SingleSubjectTracker


class _Pose:
    """Minimal pose stub matching the estimator-side duck type."""

    def __init__(self, keypoints, scores, bbox=None):
        self.keypoints = np.asarray(keypoints, dtype=np.float32)
        self.scores = np.asarray(scores, dtype=np.float32)
        if bbox is None:
            valid = self.scores[:17] > 0
            pts = self.keypoints[:17][valid]
            bbox = [pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()]
        self.bbox = np.asarray(bbox, dtype=np.float32)
        self.area = float((self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1]))


def _person(cx, cy, scale=1.0, conf=0.9):
    """Make a 17-joint stand-up person centered at (cx, cy)."""
    kp = np.zeros((133, 2), dtype=np.float32)
    sc = np.zeros((133,), dtype=np.float32)
    offsets = np.array([
        [0, -90], [-15, -95], [15, -95], [-25, -95], [25, -95],  # head
        [-40, -40], [40, -40],  # shoulders
        [-55, 30], [55, 30],    # elbows
        [-60, 95], [60, 95],    # wrists
        [-20, 90], [20, 90],    # hips
        [-25, 160], [25, 160],  # knees
        [-25, 230], [25, 230],  # ankles
    ], dtype=np.float32) * scale
    kp[:17, 0] = cx + offsets[:, 0]
    kp[:17, 1] = cy + offsets[:, 1]
    sc[:17] = conf
    return _Pose(kp, sc)


SHAPE = (1080, 1920, 3)


def test_tracker_locks_and_tracks_continuous_subject():
    tr = SingleSubjectTracker(threshold=0.3)
    p0, s0 = tr.select([_person(500, 400)], SHAPE)
    assert s0 == "selected"
    for f in range(1, 20):
        pose, status = tr.select([_person(500 + f * 3, 400)], SHAPE)
        assert status == "tracked"
        assert abs(pose.bbox[0] - (p0.bbox[0])) < 200  # followed the moving box


def test_tracker_does_not_steal_stronger_newcomer():
    tr = SingleSubjectTracker(threshold=0.3)
    # Lock on the left subject.
    tr.select([_person(400, 400)], SHAPE)
    for f in range(10):
        tr.select([_person(400 + f * 2, 400)], SHAPE)
    # A second, *larger* person appears on the far right.
    pose, status = tr.select(
        [_person(422, 400), _person(1500, 400, scale=2.0)], SHAPE
    )
    # Tracker must keep the continuos left subject, not the bigger newcomer.
    assert status == "tracked"
    assert pose.bbox[0] < 900  # still the left subject


def test_tracker_carries_through_short_gap_and_recovers():
    tr = SingleSubjectTracker(threshold=0.3, max_gap=5)
    tr.select([_person(600, 400)], SHAPE)
    for _ in range(5):
        tr.select([_person(600, 400)], SHAPE)
    # 3 frames with no detection -> carried
    for _ in range(3):
        pose, status = tr.select([], SHAPE)
        assert status == "carried"
        assert pose is None
    # Subject reappears nearby -> recovers
    pose, status = tr.select([_person(620, 400)], SHAPE)
    assert status == "tracked"


def test_tracker_releases_after_max_gap():
    tr = SingleSubjectTracker(threshold=0.3, max_gap=3)
    tr.select([_person(600, 400)], SHAPE)
    for _ in range(3):
        tr.select([_person(600, 400)], SHAPE)
    for _ in range(4):
        tr.select([], SHAPE)
    pose, status = tr.select([], SHAPE)
    assert status == "none"


def test_tracker_prefers_continuity_over_size_between_frames():
    tr = SingleSubjectTracker(threshold=0.3)
    tr.select([_person(700, 400)], SHAPE)
    tr.select([_person(700, 400)], SHAPE)
    # Next frame: a small near-continuation vs a huge far-away box.
    near = _person(705, 402, scale=0.95)
    far = _person(1600, 800, scale=3.0)
    pose, status = tr.select([far, near], SHAPE)
    assert status == "tracked"
    assert pose is near


def test_tracker_init_picks_strongest_visible_subject():
    tr = SingleSubjectTracker(threshold=0.3)
    weak = _person(1500, 800, conf=0.1)   # below visibility floor
    strong = _person(500, 400, conf=0.9)
    pose, status = tr.select([weak, strong], SHAPE)
    assert status == "selected"
    assert pose is strong
