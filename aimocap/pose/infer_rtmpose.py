"""RTMPose 2D whole-body pose inference (rtmlib, ONNX Runtime, GPU).

A drop-in alternative to ``aimocap.pose.infer.PoseEstimator`` (CIGPose). Same
public interface: construct once, call ``.estimate(frame)`` per frame, get a
list of ``Pose2D`` (133 COCO-wholebody keypoints).

Why RTMPose over CIGPose: RTMPose (rtmlib's ``rtmw-dw-x-l`` wholebody model) is
trained on a broader multi-domain cocktail dataset and generalizes better to
extreme camera angles (e.g. side views) where CIGPose was observed to produce
jumpy/wrong leg keypoints. Both emit the identical 133-keypoint layout, so the
output is a true drop-in for the triangulation + retarget pipeline.

GPU note: importing ``aimocap`` first is REQUIRED — it loads ``aimocap._gpu``
which registers the cu12 DLLs with the Windows loader BEFORE onnxruntime is
imported, so the CUDA Execution Provider activates. Without it, ORT silently
falls back to CPU.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# MUST come before any rtmlib/onnxruntime import: registers cu12 DLLs for GPU.
import aimocap  # noqa: F401
from rtmlib import Wholebody

from aimocap.pose.keypoints import Pose2D  # canonical 133-keypoint dataclass


class RTMPoseEstimator:
    """Loads RTMPose wholebody (detection + pose) and runs per-frame inference.

    Same public surface as ``PoseEstimator``: ``estimate(frame, pick=...)`` and
    ``active_providers``. Returns ``Pose2D`` objects with the same keypoint
    convention, so downstream code is backend-agnostic.

    Parameters
    ----------
    device : 'cuda' (default, needs the cu12 DLLs via aimocap._gpu) or 'cpu'.
    det, pose : optional explicit ONNX paths. None = rtmlib's auto-downloaded
                defaults (yolox_m detector + rtmw-dw-x-l wholebody pose).
    det_input_size, pose_input_size : model input resolution.
    """

    def __init__(
        self,
        device: str = "cuda",
        det: str | Path | None = None,
        pose: str | Path | None = None,
        det_input_size: tuple[int, int] = (640, 640),
        pose_input_size: tuple[int, int] = (288, 384),
    ) -> None:
        self._wholebody = Wholebody(
            det=str(det) if det else None,
            pose=str(pose) if pose else None,
            det_input_size=det_input_size,
            pose_input_size=pose_input_size,
            to_openpose=False,        # keep COCO-wholebody index 0..132 (matches CIGPose)
            backend="onnxruntime",
            device=device,
        )
        # Surface the actual ORT provider in use (CUDA vs CPU) for logging.
        self.active_providers = self._probe_providers()

    def _probe_providers(self) -> list[str]:
        # rtmlib hides the session; reach in to the pose head's ORT session.
        # Newer rtmlib exposes det_model/pose_model with .session attributes.
        try:
            for holder in (getattr(self._wholebody, "pose_model", None),
                           getattr(self._wholebody, "det_model", None),
                           getattr(getattr(self._wholebody, "pose_estimator", None), "sess", None)):
                sess = getattr(holder, "session", holder)
                if sess is not None and hasattr(sess, "get_providers"):
                    return list(sess.get_providers())
        except Exception:
            pass
        try:
            sess = getattr(self._wholebody.pose_estimator, "sess", None)
            if sess is not None and hasattr(sess, "get_providers"):
                return list(sess.get_providers())
        except Exception:
            pass
        return ["unknown"]

    def estimate(self, frame: np.ndarray, pick: str = "largest") -> list[Pose2D]:
        """Estimate whole-body poses for all detected persons in ``frame``.

        ``pick`` mirrors PoseEstimator:
          - "all"     : return every detected person
          - "largest" : return only the largest-area person (list of len 1)
          - "best"    : return only the highest mean-confidence person
        """
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"frame must be HxWx3 BGR, got {frame.shape}")

        kpts, scores = self._wholebody(frame)   # (N, 133, 2), (N, 133)
        if kpts is None or len(kpts) == 0:
            return []

        poses: list[Pose2D] = []
        for i in range(len(kpts)):
            bbox = self._bbox_from_kpts(kpts[i])
            poses.append(Pose2D(
                keypoints=np.asarray(kpts[i], dtype=np.float32),
                scores=np.asarray(scores[i], dtype=np.float32),
                bbox=bbox,
            ))

        if pick == "all" or not poses:
            return poses
        if pick == "largest":
            return [max(poses, key=lambda p: p.area)]
        if pick == "best":
            return [max(poses, key=lambda p: p.mean_score())]
        raise ValueError(f"unknown pick={pick!r}; use 'all'|'largest'|'best'")

    @staticmethod
    def _bbox_from_kpts(kpts: np.ndarray) -> np.ndarray:
        """Derive a tight bbox [x1,y1,x2,y2] from keypoints (for Pose2D.area)."""
        valid = kpts[np.isfinite(kpts).all(axis=1)]
        if len(valid) == 0:
            return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        x1, y1 = valid.min(axis=0)
        x2, y2 = valid.max(axis=0)
        return np.array([x1, y1, x2, y2], dtype=np.float32)


__all__ = ["RTMPoseEstimator"]
