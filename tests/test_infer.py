"""Tests for the 2D pose inference wrapper.

Uses the bundled CIGPose model + YOLOX detector on a real COCO image (cached
locally). These are integration tests — they need the models present under
models/ and the test image under data/test/. Skipped gracefully if absent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import cv2

from aimocap.pose.infer import Pose2D, PoseEstimator, _CIGPOSE_AVAILABLE
from aimocap.pose.keypoints import N_KEYPOINTS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
POSE_MODEL = PROJECT_ROOT / "models" / "cigpose-l_coco-wholebody_384x288.onnx"
DETECTOR = PROJECT_ROOT / "models" / "yolox_nano.onnx"
TEST_IMG = PROJECT_ROOT / "data" / "test" / "000000000785.jpg"

needs_models = pytest.mark.skipif(
    not _CIGPOSE_AVAILABLE or not POSE_MODEL.exists() or not DETECTOR.exists(),
    reason="CIGPose models or package not present — run scripts/fetch_models.py",
)
needs_test_img = pytest.mark.skipif(
    not TEST_IMG.exists(),
    reason="test image not present — run scripts/fetch_test_images.py",
)


# ---------------------------------------------------------------------------
# Unit: Pose2D dataclass enforces shapes
# ---------------------------------------------------------------------------

def test_pose2d_rejects_bad_shapes():
    with pytest.raises(ValueError):
        Pose2D(
            keypoints=np.zeros((17, 2), dtype=np.float32),
            scores=np.zeros(17, dtype=np.float32),
            bbox=np.zeros(4, dtype=np.float32),
        )


def test_pose2d_accepts_valid_input():
    p = Pose2D(
        keypoints=np.zeros((N_KEYPOINTS, 2), dtype=np.float32),
        scores=np.full(N_KEYPOINTS, 0.5, dtype=np.float32),
        bbox=np.array([0, 0, 10, 10], dtype=np.float32),
    )
    assert p.area == 100.0
    assert 0.0 <= p.mean_score() <= 1.0


# ---------------------------------------------------------------------------
# Integration: estimator on a real image
# ---------------------------------------------------------------------------

@needs_models
@needs_test_img
class TestEstimatorOnRealImage:
    """Run the full detect+pose pipeline on a known full-body COCO image."""

    @classmethod
    @pytest.fixture(scope="class")
    def estimator(cls):
        return PoseEstimator()

    @classmethod
    @pytest.fixture(scope="class")
    def frame(cls):
        f = cv2.imread(str(TEST_IMG))
        assert f is not None, f"could not read {TEST_IMG}"
        return f

    @classmethod
    @pytest.fixture(scope="class")
    def poses(cls, estimator, frame):
        return estimator.estimate(frame, pick="all")

    def test_detects_at_least_one_person(self, poses):
        assert len(poses) >= 1

    def test_keypoint_shapes(self, poses):
        p = poses[0]
        assert p.keypoints.shape == (N_KEYPOINTS, 2)
        assert p.scores.shape == (N_KEYPOINTS,)
        assert p.bbox.shape == (4,)

    def test_keypoints_finite_and_in_frame(self, poses, frame):
        p = poses[0]
        assert np.isfinite(p.keypoints).all()
        h, w = frame.shape[:2]
        # all keypoints within frame bounds (allow small slack for crop edge)
        x, y = p.keypoints[:, 0], p.keypoints[:, 1]
        assert (x >= -5).all() and (x <= w + 5).all()
        assert (y >= -5).all() and (y <= h + 5).all()

    def test_scores_nonnegative(self, poses):
        p = poses[0]
        # SimCC max-logit scores are not bounded to [0,1] but must be >= 0.
        assert (p.scores >= 0).all()

    def test_body_confidence_reasonable(self, poses):
        # On a clean full-body standing shot, body keypoints should be confident.
        from aimocap.pose.keypoints import BODY_17
        body = np.array(BODY_17)
        p = poses[0]
        assert p.scores[body].mean() > 0.3, (
            f"body mean confidence {p.scores[body].mean():.3f} unexpectedly low"
        )
