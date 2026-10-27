# tests/test_retarget_input_gate.py
"""WP2.B: fail-closed core-target gate (CPU-only, deterministic).

A structurally broken target track (e.g. an untriangulated hip poisoning
the pelvis midpoint) must refuse with a per-track report instead of solving
into NaN roots. Healthy inputs must pass the gate untouched.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aimocap.retarget.engine import retarget_to_fbx


def _npz(path: Path, hip_r_finite_frac: float) -> None:
    """(300, 133, 3) m-skeleton; hip_r finite on the leading fraction."""
    rng = np.random.default_rng(0)
    pts = rng.normal(0, 0.5, size=(300, 133, 3))
    pts[:, :, 1] += 1.0
    n_fin = int(300 * hip_r_finite_frac)
    pts[n_fin:, 12, :] = np.nan  # wholebody hip_r = 12
    np.savez(path, skeleton3d=pts, units=np.asarray("m"),
             confidence=np.ones((300, 133), dtype=np.float32))


def test_collapsed_track_refuses_with_report(tmp_path: Path):
    src = tmp_path / "in.npz"
    _npz(src, 0.27)  # RTMPose-like collapse
    with pytest.raises(ValueError, match="unreliable required target tracks") as exc:
        retarget_to_fbx(str(src), "rig.fbx", str(tmp_path / "o.bvh"))
    assert "hip_r" in str(exc.value) and "pelvis" in str(exc.value)


def test_healthy_tracks_pass_gate(tmp_path: Path):
    src = tmp_path / "in.npz"
    _npz(src, 1.0)
    # Passes the gate (proceeds to rig load, which fails on the fake path).
    with pytest.raises(Exception, match="(?i)(fbx|rig|FBX|Manny|exist|found|load)"):
        retarget_to_fbx(str(src), "rig.fbx", str(tmp_path / "o.bvh"))


def test_threshold_is_configurable(tmp_path: Path):
    src = tmp_path / "in.npz"
    _npz(src, 0.40)
    with pytest.raises(ValueError, match="unreliable"):
        retarget_to_fbx(str(src), "rig.fbx", str(tmp_path / "o.bvh"))
    with pytest.raises(Exception, match="(?i)(fbx|rig|FBX|Manny|exist|found|load)"):
        retarget_to_fbx(str(src), "rig.fbx", str(tmp_path / "o.bvh"),
                        min_core_finite_fraction=0.3)
