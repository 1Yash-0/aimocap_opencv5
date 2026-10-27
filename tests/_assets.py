"""Skip markers for tests that need assets which are not distributed with this repository.

A test carrying one of these markers is skipped with a clear reason when the asset is
absent. Test bodies and assertions are unchanged.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

MANNY_FBX_PATH = ROOT / "Manny.FBX"
TRUE_SYNTHETIC_PATH = ROOT / "outputs" / "true_synthetic.npz"

requires_manny_fbx = pytest.mark.skipif(
    not MANNY_FBX_PATH.exists(),
    reason="Manny.FBX rig is not distributed with this repository; place an FBX rig at the repository root to run this test",
)

requires_true_synthetic = pytest.mark.skipif(
    not TRUE_SYNTHETIC_PATH.exists(),
    reason="outputs/true_synthetic.npz is generated from the FBX rig by tests/generate_true_synthetic.py and is not distributed",
)
