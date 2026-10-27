# tests/test_retarget_input_contract.py
"""WP0.4: pin the retarget input contract's fail-closed behavior (CPU-only).

Locks down already-intended/current validation in `retarget_to_fbx`
without running the solver: joint-count, units, finiteness, and grounding
gates. All cases fail before any rig is loaded, so no FBX asset is needed.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aimocap.calib.floor import RegistrationGateError
from aimocap.retarget.engine import retarget_to_fbx


def _npz(path: Path, n_joints: int, units: str, finite: bool = True) -> None:
    pts = np.zeros((2, n_joints, 3), dtype=np.float64)
    if finite:
        pts[..., 1] = 100.0
    else:
        pts[...] = np.nan
    np.savez(path, skeleton3d=pts, units=np.asarray(units))


def test_rejects_wrong_joint_count(tmp_path: Path):
    src = tmp_path / "in.npz"
    _npz(src, 20, "m")
    with pytest.raises(ValueError, match="133 joints"):
        retarget_to_fbx(str(src), "rig.fbx", str(tmp_path / "o.bvh"))


def test_rejects_unknown_units(tmp_path: Path):
    src = tmp_path / "in.npz"
    _npz(src, 133, "furlongs")
    with pytest.raises(ValueError, match="units"):
        retarget_to_fbx(str(src), "rig.fbx", str(tmp_path / "o.bvh"))


def test_rejects_all_nan_input(tmp_path: Path):
    src = tmp_path / "in.npz"
    _npz(src, 133, "m", finite=False)
    with pytest.raises(ValueError, match="no finite"):
        retarget_to_fbx(str(src), "rig.fbx", str(tmp_path / "o.bvh"))


def test_require_grounded_refuses_unregistered_transform(tmp_path: Path):
    src = tmp_path / "in.npz"
    _npz(src, 133, "m")
    with pytest.raises(RegistrationGateError):
        retarget_to_fbx(str(src), "rig.fbx", str(tmp_path / "o.bvh"),
                        require_grounded=True)
