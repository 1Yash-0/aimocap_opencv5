from __future__ import annotations

import numpy as np

from aimocap.retarget.mocap_ik import MocapIKSolver
from tests.test_foot_landmark_model import _fixture


from tests._assets import requires_manny_fbx
@requires_manny_fbx
def test_face_position_default_is_bit_identical():
    _, skel, targets, _ = _fixture(frames=1)
    measured = {key: value[0].copy() for key, value in targets.items()}
    implicit = MocapIKSolver(skel)
    explicit = MocapIKSolver(skel, face_position_weight=1.0)
    x = np.linspace(-0.1, 0.1, implicit.num_vars)
    a = implicit._residuals_with_ctx(x, implicit._precompute_frame(measured))
    b = explicit._residuals_with_ctx(x, explicit._precompute_frame(measured))
    assert np.array_equal(a, b)


@requires_manny_fbx
def test_face_position_weight_scales_only_head_position_row():
    _, skel, targets, _ = _fixture(frames=1)
    measured = {key: value[0].copy() for key, value in targets.items()}
    base = MocapIKSolver(skel)
    zero = MocapIKSolver(skel, face_position_weight=0.0)
    scaled = MocapIKSolver(skel, face_position_weight=0.25)
    x = np.linspace(-0.1, 0.1, base.num_vars)
    a = base._residuals_with_ctx(x, base._precompute_frame(measured))
    b = zero._residuals_with_ctx(x, zero._precompute_frame(measured))
    c = scaled._residuals_with_ctx(x, scaled._precompute_frame(measured))
    head = skel.name_to_idx["head"]
    rows = slice(head * 3, head * 3 + 3)
    assert np.array_equal(b[:rows.start], a[:rows.start])
    assert np.array_equal(b[rows.stop:], a[rows.stop:])
    assert np.array_equal(b[rows], np.zeros(3))
    assert np.array_equal(c[rows], a[rows] * 0.25)
