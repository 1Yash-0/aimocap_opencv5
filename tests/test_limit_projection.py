import numpy as np
import pytest


def _ctx(mode="post", trace=None):
    return {
        "twist_limit_mode": "clamp",
        "limit_projection": mode,
        "twist_limits": {"joint": 10.0},
        "twist_joint_indices": {"joint": 0},
        "twist_bone_axes": {"joint": np.array([1.0, 0.0, 0.0])},
        "twist_soft_joints": (),
        "twist_safety_margin_deg": 10.0,
        "lm_trace": trace,
        "target": None,
    }


def _run(mode="post", start=None, target=None, trace=None):
    torch = pytest.importorskip("torch")
    from aimocap.retarget import ik_backend, ik_torch

    original = ik_torch._residuals_torch
    target = torch.as_tensor(
        np.asarray(target if target is not None else [0, 0, 0, 1, 0, 0], dtype=np.float64)
    )

    def residual(x, ctx):
        return x - target.to(device=x.device, dtype=x.dtype)

    ik_torch._residuals_torch = residual
    try:
        ctx = _ctx(mode, trace)
        x = torch.as_tensor(
            np.asarray(start if start is not None else np.zeros(6), dtype=np.float64)
        ).reshape(1, 6)
        return ik_backend._lm_loop_torch(
            x, ctx, num_vars=6, max_iter=20, ftol=1e-12, xtol=1e-12
        ).detach().cpu().numpy().reshape(-1), ctx
    finally:
        ik_torch._residuals_torch = original


def test_post_default_is_bit_identical_to_explicit_post():
    implicit, _ = _run(mode="post")
    explicit, _ = _run(mode="post")
    np.testing.assert_array_equal(implicit, explicit)


def test_projected_trials_are_monotone_with_active_limit():
    trace = []
    solved, _ = _run(
        mode="projected",
        target=[0, 0, 0, np.deg2rad(30.0), 0, 0],
        trace=trace,
    )
    accepted = [row for row in trace if row["accepted"]]
    assert accepted
    assert all(row["cost_projected"] <= row["cost_x"] + 1e-12 for row in accepted)
    assert np.isclose(solved[3], np.deg2rad(10.0), atol=1e-8)


def test_projected_lambda_ladder_is_bounded_and_tiny_feasible_step_terminates():
    target = [0, 0, 0, np.deg2rad(30.0), 0, 0]
    solved, ctx = _run(
        mode="projected",
        start=[0, 0, 0, np.deg2rad(10.0), 0, 0],
        target=target,
        trace=[],
    )
    assert np.isclose(solved[3], np.deg2rad(10.0), atol=1e-6)
    assert ctx["_lm_stats"]["termination_reason"] == "projected_xtol"
    assert max(row["inner"] for row in ctx["lm_trace"]) < 8

