"""Phase D: IK hardening — NaN-tolerant orientation terms + warm-start
safety net (CPU-only, deterministic).

D1: non-finite spine/hip/shoulder inputs must drop the orientation residual
terms (ori_weight=0) instead of NaN-poisoning the whole residual. Finite
inputs keep ori_weight=1.0 bit-exact.
D2: a non-finite solve result must never enter the temporal chain.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aimocap.retarget.mocap_ik import MocapIKSolver, MocapSkeleton


from tests._assets import requires_manny_fbx
def _solver(n=12, seed=5):
    rng = np.random.default_rng(seed)
    pts = rng.normal(0.0, 10.0, (n, 133, 3))
    pts[:, :, 1] += 100.0
    w = np.ones((n, 133), dtype=float)
    return MocapSkeleton(pts, w)


@requires_manny_fbx
def test_finite_inputs_keep_ori_weight_one():
    sk = _solver()
    solver = MocapIKSolver(sk)
    rng = np.random.default_rng(1)
    measured = {
        "pelvis": rng.normal(0, 1, 3), "neck": np.array([0.0, 5.0, 0.0]),
        "hip_l": np.array([-1.0, 0.0, 0.0]), "hip_r": np.array([1.0, 0.0, 0.0]),
        "shoulder_l": np.array([-1.2, 4.8, 0.0]), "shoulder_r": np.array([1.2, 4.8, 0.0]),
        "nose": np.array([0.0, 6.5, 0.5]),
    }
    ctx = solver._precompute_frame(measured, None, 0.03, None, None, frame_idx=0)
    from aimocap.retarget.mocap_ik import ORI_WEIGHT_SCALE
    assert ctx["ori_weight"] == ORI_WEIGHT_SCALE


@requires_manny_fbx
def test_nan_spine_inputs_drop_orientation_terms():
    sk = _solver()
    solver = MocapIKSolver(sk)
    measured = {
        "pelvis": np.full(3, np.nan), "neck": np.array([0.0, 5.0, 0.0]),
        "hip_l": np.array([-1.0, 0.0, 0.0]), "hip_r": np.array([1.0, 0.0, 0.0]),
        "shoulder_l": np.array([-1.2, 4.8, 0.0]), "shoulder_r": np.array([1.2, 4.8, 0.0]),
        "nose": np.array([0.0, 6.5, 0.5]),
    }
    ctx = solver._precompute_frame(measured, None, 0.03, None, None, frame_idx=0)
    assert ctx["ori_weight"] == 0.0, "non-finite spine inputs must drop orientation terms"
    # the position targets and weights stay finite/valid
    assert np.isfinite(ctx["w"]).all()


@requires_manny_fbx
def test_nan_hip_inputs_drop_orientation_terms():
    sk = _solver()
    solver = MocapIKSolver(sk)
    measured = {
        "pelvis": np.array([0.0, 0.0, 0.0]), "neck": np.array([0.0, 5.0, 0.0]),
        "hip_l": np.full(3, np.nan), "hip_r": np.array([1.0, 0.0, 0.0]),
        "shoulder_l": np.array([-1.2, 4.8, 0.0]), "shoulder_r": np.array([1.2, 4.8, 0.0]),
        "nose": np.array([0.0, 6.5, 0.5]),
    }
    ctx = solver._precompute_frame(measured, None, 0.03, None, None, frame_idx=0)
    assert ctx["ori_weight"] == 0.0


@requires_manny_fbx
def test_residual_finite_when_orientation_dropped():
    """With ori_weight=0 the residual vector must be finite despite NaN
    orientation targets (previously NaN poisoned the whole residual)."""
    sk = _solver()
    solver = MocapIKSolver(sk)
    measured = {
        "pelvis": np.full(3, np.nan), "neck": np.array([0.0, 5.0, 0.0]),
        "hip_l": np.array([-1.0, 0.0, 0.0]), "hip_r": np.array([1.0, 0.0, 0.0]),
        "shoulder_l": np.array([-1.2, 4.8, 0.0]), "shoulder_r": np.array([1.2, 4.8, 0.0]),
        "nose": np.array([0.0, 6.5, 0.5]),
    }
    ctx = solver._precompute_frame(measured, None, 0.03, None, None, frame_idx=0)
    x = np.zeros(solver.num_vars)
    x[0:3] = [0.0, 0.0, 0.0]
    r = solver._residuals_with_ctx(x, ctx)
    assert np.isfinite(r).all(), (
        "residual must stay finite when orientation terms are dropped; "
        f"got {int((~np.isfinite(r)).sum())} NaN entries")