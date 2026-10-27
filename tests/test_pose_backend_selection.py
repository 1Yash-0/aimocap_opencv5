# tests/test_pose_backend_selection.py
"""WP1.A: pose-backend selection + provenance (CPU-only, no inference)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest

from aimocap import cli as cli_mod


def _parse(argv):
    return cli_mod.build_parser().parse_args(argv)


def test_multipose_backend_default_is_rtmw():
    # 2026-10-17: default switched to RTMW-x + smoothbox + subpixel (user approved;
    # evidence outputs/phase1_2d/FINDINGS_20261017.md). Legacy stays selectable.
    args = _parse(["multipose", "a.mp4", "-o", "b.npz"])
    assert args.pose_backend == "rtmw"
    assert args.crop_mode == "smoothbox" and args.simcc_decode == "subpixel"
    legacy = _parse(["multipose", "a.mp4", "-o", "b.npz", "--pose-backend", "cigpose",
                     "--crop-mode", "detector", "--simcc-decode", "argmax",
                     "--part-refine", "off"])
    assert (legacy.pose_backend, legacy.crop_mode, legacy.simcc_decode,
            legacy.part_refine) == ("cigpose", "detector", "argmax", "off")


def test_multipose_part_refine_default_is_face_with_legacy_off():
    # S3 clip2 unseen gate: face passed; hands did not, so only face is on by
    # default at the CLI. The library default remains opt-in (tested elsewhere).
    args = _parse(["multipose", "a.mp4", "-o", "b.npz"])
    assert args.part_refine == "hands_face"
    legacy = _parse(["multipose", "a.mp4", "-o", "b.npz", "--part-refine", "off"])
    assert legacy.part_refine == "off"


def test_multipose_backend_accepts_rtmpose():
    args = _parse(["multipose", "a.mp4", "-o", "b.npz", "--pose-backend", "rtmpose"])
    assert args.pose_backend == "rtmpose"


def test_multipose_backend_rejects_unknown():
    with pytest.raises(SystemExit):
        _parse(["multipose", "a.mp4", "-o", "b.npz", "--pose-backend", "openpose"])


def test_build_estimator_routes_cigpose(monkeypatch):
    seen = {}

    class FakeCig:
        def __init__(self, **kwargs):
            seen["kwargs"] = kwargs

    import aimocap.pose.infer as infer_mod
    monkeypatch.setattr(infer_mod, "PoseEstimator", FakeCig)
    cli_mod._build_pose_estimator("cigpose", "mymodel.onnx")
    assert seen["kwargs"] == {"pose_model": "mymodel.onnx"}


def test_build_estimator_routes_rtmpose(monkeypatch):
    import aimocap.pose.infer_rtmpose as rt_mod

    class FakeRT:
        def __init__(self, **kwargs):
            raise AssertionError("must not construct real estimator")

    created = {}
    monkeypatch.setattr(rt_mod, "RTMPoseEstimator", lambda **k: created.setdefault("made", True))
    assert cli_mod._build_pose_estimator("rtmpose", None) is True
    assert created == {"made": True}


def test_build_estimator_unknown_backend_falls_back_to_cigpose_default(monkeypatch):
    import aimocap.pose.infer as infer_mod

    made = {}
    monkeypatch.setattr(infer_mod, "PoseEstimator", lambda **k: made.setdefault("made", True))
    # Any non-rtmpose value keeps legacy CIGPose construction (default preserved).
    assert cli_mod._build_pose_estimator("cigpose", None) is True
