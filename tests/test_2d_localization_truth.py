# tests/test_2d_localization_truth.py
"""P1.A: harness trust tests (CPU-only, deterministic).

The localization verdicts rest on projecting GT through the production
bundle exactly as the engine does. These pin the projection path and the
GT convention; the live verdict additionally requires the script's own
self-check (bit-near reproduction of the artifact's inlier residuals).
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

if importlib.util.find_spec("measure_2d_localization_truth") is None:
    pytest.skip(
        "scripts/measure_2d_localization_truth.py is not part of this repository",
        allow_module_level=True,
    )

from measure_2d_localization_truth import COCO17_TO_PAN19, project_frame  # noqa: E402


def test_coco17_to_pan19_map_shape():
    assert len(COCO17_TO_PAN19) == 17
    assert len(set(COCO17_TO_PAN19)) == 17
    assert set(COCO17_TO_PAN19) <= set(range(19))


def test_project_frame_pinhole_known_value():
    import cv2
    K = np.array([[1500, 0, 960], [0, 1500, 540], [0, 0, 1]], float)
    R = np.eye(3)
    t = np.zeros(3)
    dist = np.zeros(5)
    proj = project_frame([(K, R, t, dist)], np.array([[0.0, 0.0, 10.0]]))
    assert proj.shape == (1, 1, 2)
    assert abs(proj[0, 0, 0] - 960.0) < 1e-6
    assert abs(proj[0, 0, 1] - 540.0) < 1e-6


def test_project_frame_nan_safe():
    K = np.eye(3)
    R = np.eye(3)
    t = np.zeros(3)
    dist = np.zeros(5)
    proj = project_frame([(K, R, t, dist)],
                         np.array([[np.nan, 0.0, 1.0], [0.0, 0.0, 1.0]]))
    assert np.isnan(proj[0, 0]).all()
    assert np.all(np.isfinite(proj[1, 0]))
