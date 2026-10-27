from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

from aimocap.retarget.engine import retarget_to_fbx
from aimocap.retarget.mocap_ik import MocapIKSolver

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
from test_foot_landmark_model import _base_points, _fixture


from tests._assets import requires_manny_fbx
@requires_manny_fbx
def test_default_options_equal_explicit_defaults():
    _, skel, targets, _ = _fixture(frames=1)
    measured = {key: value[0].copy() for key, value in targets.items()}
    implicit = MocapIKSolver(skel)
    explicit = MocapIKSolver(
        skel, head_target_mode="legacy", foot_landmark_weight=1.0
    )
    x = np.linspace(-0.1, 0.1, implicit.num_vars)
    a = implicit._residuals_with_ctx(x, implicit._precompute_frame(measured))
    b = explicit._residuals_with_ctx(x, explicit._precompute_frame(measured))
    assert np.array_equal(a, b)


@requires_manny_fbx
def test_foot_landmark_weight_scales_only_landmark_rows():
    _, skel, targets, _ = _fixture(frames=1)
    measured = {key: value[0].copy() for key, value in targets.items()}
    base = MocapIKSolver(skel, target_sequence=targets, foot_landmark_model="fk_offsets")
    scaled = MocapIKSolver(
        skel, target_sequence=targets, foot_landmark_model="fk_offsets",
        foot_landmark_weight=0.25,
    )
    x = np.linspace(-0.1, 0.1, base.num_vars)
    base_ctx = base._precompute_frame(measured)
    scaled_ctx = scaled._precompute_frame(measured)
    a = base._residuals_with_ctx(x, base_ctx)
    b = scaled._residuals_with_ctx(x, scaled_ctx)
    landmark_rows = base_ctx["foot_landmark_packet"].shape[0] * 3
    assert landmark_rows == 18
    assert np.array_equal(b[:-landmark_rows], a[:-landmark_rows])
    assert np.array_equal(b[-landmark_rows:], a[-landmark_rows:] * 0.25)


@requires_manny_fbx
def test_fixed_head_mode_reachable_through_engine(tmp_path):
    points, confidence = _base_points(frames=1)
    source = tmp_path / "input.npz"
    output = tmp_path / "fixed.bvh"
    np.savez(source, skeleton3d=points, units=np.asarray("cm"), confidence=confidence)
    retarget_to_fbx(
        str(source), str(ROOT / "Manny.FBX"), str(output), max_frames=1,
        mocap_target_stabilization=False, leg_stabilization=False,
        foot_lock=False, export_fbx=False, profile=True,
        head_target_mode="fixed", foot_landmark_weight=0.5,
    )
    profile = json.loads((tmp_path / "fixed.profile.json").read_text())
    options = profile["effective_options"]
    assert options["head_target_mode"] == "fixed"
    assert options["foot_landmark_weight"] == 0.5
