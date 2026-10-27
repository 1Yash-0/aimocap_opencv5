"""IK backend dispatcher — selects the fastest available solver.

Priority:
  1. PyTorch CUDA (GPU, ~100x faster) — if torch.cuda.is_available()
  2. Scipy LM with precompute (CPU, ~2x faster than original) — always available
  3. (Future: Numba JIT CPU fallback)

The backend is selected automatically.  Override with env var:
  AIMOCAP_IK_BACKEND=torch|scipy|auto  (default: auto)
"""
from __future__ import annotations

import os
import numpy as np

# ── Backend selection ────────────────────────────────────────────────────

_BACKEND_OVERRIDE = os.environ.get("AIMOCAP_IK_BACKEND", "auto").lower()
_BACKEND_NAME = None  # set on first dispatch


def _try_torch_cuda():
    """Check if torch CUDA is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


_HAS_CUDA = _try_torch_cuda()


def dispatch_solve_frame(solver, ctx: dict, x0: np.ndarray) -> np.ndarray:
    """Select the best available backend and solve."""
    global _BACKEND_NAME

    backend = _BACKEND_OVERRIDE
    if backend == "auto":
        if _HAS_CUDA:
            backend = "torch"
        else:
            backend = "scipy"

    if backend == "torch" and _HAS_CUDA:
        _BACKEND_NAME = "torch-cuda"
        try:
            return _solve_torch(solver, ctx, x0)
        except Exception as e:
            import warnings
            warnings.warn(
                f"torch backend failed ({e}), falling back to scipy")
            _BACKEND_NAME = "scipy-fallback"
            return _solve_scipy_precomputed(solver, ctx, x0)
    else:
        _BACKEND_NAME = "scipy"
        return _solve_scipy_precomputed(solver, ctx, x0)




def _apply_twist_limits_numpy(x: np.ndarray, ctx: dict) -> np.ndarray:
    """Apply hard limits, using the hybrid safety bound for selected joints."""
    mode = ctx.get("twist_limit_mode", "clamp")
    if mode not in {"clamp", "hybrid"}:
        return x
    from scipy.spatial.transform import Rotation as _Rot
    from aimocap.retarget.swing_twist import clamp_twist
    soft_names = set(ctx.get("twist_soft_joints") or ())
    margin = float(ctx.get("twist_safety_margin_deg", 10.0))
    result = np.asarray(x, dtype=np.float64).copy()
    rv = result[3:].reshape(-1, 3)
    for jname, max_tw in (ctx.get("twist_limits") or {}).items():
        ji = ctx.get("twist_joint_indices", {}).get(jname)
        bone = ctx.get("twist_bone_axes", {}).get(jname)
        if ji is None or bone is None:
            continue
        current = rv[ji]
        if np.linalg.norm(current) < 1e-8:
            continue
        limit = float(max_tw) + margin if mode == "hybrid" and jname in soft_names else float(max_tw)
        q = _Rot.from_rotvec(current)
        q_clamped = clamp_twist(q, bone, limit)
        rv_new = q_clamped.as_rotvec()
        if np.linalg.norm(rv_new - current) > 1e-6:
            rv[ji] = rv_new
    return result


def _project_state_torch(x: "torch.Tensor", ctx: dict) -> "torch.Tensor":
    """Project a single batched torch state through the hard twist bounds.

    The anatomical projection is intentionally shared with the scipy path so
    both backends use the same swing-twist convention. This CPU round-trip is
    opt-in and occurs only for projected trials/accepted states.
    """
    import torch

    values = _apply_twist_limits_numpy(x.detach().cpu().numpy()[0], ctx)
    return torch.as_tensor(values, device=x.device, dtype=x.dtype).unsqueeze(0)


def _project_x0_numpy(x0: np.ndarray, ctx: dict) -> np.ndarray:
    """Project a warm start when the opt-in feasible-set mode is active."""
    if ctx.get("limit_projection", "post") != "projected":
        return np.asarray(x0, dtype=np.float64)
    return _apply_twist_limits_numpy(x0, ctx)


# ── Scipy path (with precompute — always available) ─────────────────────

def _solve_scipy_precomputed(solver, ctx: dict, x0: np.ndarray) -> np.ndarray:
    """Scipy LM using precomputed frame context (faster than original).

    The projected mode has no scipy least_squares step hook. It projects the
    warm start and result, then performs one final polish call from the
    projected state before projecting once more. Torch is the authoritative
    projected-trial implementation; this fallback preserves the same
    feasible-state contract without changing the default post mode.
    """
    from scipy.optimize import least_squares
    max_nfev = int(ctx.get("max_iter", os.environ.get("AIMOCAP_IK_MAX_NFEV", "500")))
    if max_nfev < 1:
        raise ValueError("AIMOCAP_IK_MAX_NFEV must be positive")
    projected = ctx.get("limit_projection", "post") == "projected"
    start = _project_x0_numpy(x0, ctx) if projected else x0
    res = least_squares(
        solver._residuals_with_ctx, start,
        args=(ctx,), method='lm', max_nfev=max_nfev,
        ftol=1e-6, xtol=1e-6, gtol=1e-6,
    )
    x = _project_x0_numpy(res.x, ctx) if projected else res.x
    if projected:
        polish = least_squares(
            solver._residuals_with_ctx, x,
            args=(ctx,), method='lm', max_nfev=1,
            ftol=1e-6, xtol=1e-6, gtol=1e-6,
        )
        x = _project_x0_numpy(polish.x, ctx)
        final_cost = float(np.sum(np.asarray(solver._residuals_with_ctx(x, ctx)) ** 2))
        nfev = int(getattr(res, "nfev", 0) or 0) + int(getattr(polish, "nfev", 0) or 0)
    else:
        # VQ-FIX2 (fallback parity): the torch path clamps twist per accepted
        # step; scipy has no step hook, so apply the same bounds once on the
        # final state. Soft mode puts the limit in the residual and off mode
        # deliberately leaves it unconstrained. Hybrid uses the nominal limit in
        # its residual and only the +10 degree safety limit here.
        x = _apply_twist_limits_numpy(res.x, ctx)
        final_cost = float(2.0 * getattr(res, "cost", 0.0))
        nfev = int(getattr(res, "nfev", 0) or 0)
    solver._last_backend_stats = {
        "backend": "scipy",
        "termination_reason": str(getattr(res, "message", "unknown")),
        "iterations": nfev,
        "status": int(getattr(res, "status", 0) or 0),
        "final_cost": final_cost,
    }
    return x


# ── Torch CUDA path ──────────────────────────────────────────────────────

def _solve_torch(solver, ctx: dict, x0: np.ndarray) -> np.ndarray:
    """GPU-accelerated solve using PyTorch CUDA."""
    import torch

    device = torch.device("cuda")
    # FP64 is used for production accuracy.  Benchmarking showed FP64 and
    # FP32 per-FK-call times are identical on the RTX 4060 for our batch
    # size (76×75) — the Ada Lovelace FP64 throttle doesn't apply to small
    # tensors.  Set AIMOCAP_IK_PRECISION=fp32 for rapid iteration (slightly
    # fewer iterations due to larger eps, but causes temporal divergence
    # over 300+ frames: 101mm MPJPE → 210mm).
    _precision = os.environ.get("AIMOCAP_IK_PRECISION", "fp64").lower()
    dtype = torch.float32 if _precision == "fp32" else torch.float64
    num_joints = ctx["num_joints"]
    num_vars = 3 + num_joints * 3

    # ── Build torch tensors from ctx (move to GPU) ───────────────────
    def to_t(arr):
        if arr is None:
            return None
        return torch.tensor(np.ascontiguousarray(arr), device=device, dtype=dtype)

    # C2.2: skeleton-static FK tensors + per-level index pairs are cached on
    # the solver (first call) instead of rebuilt per frame. Same values ->
    # bit-exact; saves ~5 GPU uploads per frame.
    cache = getattr(solver, "_fk_torch_cache", None)

    if cache is None or cache.get("dtype") != dtype:
        from aimocap.retarget.ik_torch import build_fk_levels
        levels_flat = []
        level_starts = [0]
        for lvl in solver.levels:
            levels_flat.extend(lvl.tolist())
            level_starts.append(len(levels_flat))
        cache = {
            "dtype": dtype,
            "device": device,
            "rest_offsets": torch.tensor(np.ascontiguousarray(solver.rest_offsets), device=device, dtype=dtype),
            "parents": torch.tensor(solver.parents, device=device, dtype=torch.long),
            "levels_flat": torch.tensor(levels_flat, device=device, dtype=torch.long),
            "level_starts": torch.tensor(level_starts, device=device, dtype=torch.long),
            "levels_pre": build_fk_levels(
                solver.parents, solver.levels, solver.rest_offsets,
                device, torch.long, dtype),
        }
        solver._fk_torch_cache = cache

    torch_ctx = {
        "num_joints": num_joints,
        "rest_t0": to_t(ctx["rest_t0"]),
        "rest_offsets": cache["rest_offsets"],
        "parents": cache["parents"],
        "levels_flat": cache["levels_flat"],
        "level_starts": cache["level_starts"],
        "levels_pre": cache["levels_pre"],
        "tgt_pos": to_t(ctx["tgt_pos"]),
        "w": to_t(ctx["w"]),
        "R_target_quats": to_t(ctx["R_target_quats"]),
        "spine_indices": torch.tensor(ctx["spine_indices"], device=device, dtype=torch.long),
        "n_spine": ctx["n_spine"],
        "ori_weight": ctx["ori_weight"],
        "head_orientation_weight": float(ctx.get("head_orientation_weight", 1.0)),
        "head_ori_active": ctx["head_ori_active"],
        "head_idx": ctx["head_idx"],
        "neck_idx": ctx["neck_idx"],
        "pelvis_idx": ctx["pelvis_idx"],
        "head_rest_dir": to_t(ctx["head_rest_dir"]) if ctx["head_rest_dir"] is not None else None,
        "nose": to_t(ctx["nose"]) if ctx["nose"] is not None else None,
        # Frame-based head orientation target (precomputed per frame)
        "head_target_quat": to_t(ctx["head_target_quat"]) if ctx.get("head_target_quat") is not None else None,
        # Kabsch head orientation data (disabled)
        "use_kabsch": ctx.get("use_kabsch", False),
        "canonical_head": to_t(ctx["canonical_head"]) if "canonical_head" in ctx else None,
        "face_kpts_obs": None,  # set below if Kabsch is active
        "face_kpts_template": None,  # set below
        "lat_axis": to_t(ctx["lat_axis"]),
        "prev_lat": ctx["prev_lat"],
        "prev_x": to_t(ctx["prev_x"]).unsqueeze(0) if ctx["prev_x"] is not None else None,
        "prev_prev_x": to_t(ctx["prev_prev_x"]).unsqueeze(0) if ctx.get("prev_prev_x") is not None else None,
        "accel_weight": ctx.get("accel_weight", 0.0),
        "init_x": to_t(ctx["init_x"]).unsqueeze(0) if ctx["init_x"] is not None else None,
        "temporal_weight": ctx["temporal_weight"],
        "init_weight": ctx["init_weight"],
        "limb_pinned_info": None,  # handled below
        "limb_twist_anchor": ctx.get("limb_twist_anchor", "init"),
        "limb_twist_pin": ctx.get("limb_twist_pin", "always"),
        "foot_transport_pins": [],
        "foot_landmark_packet": to_t(ctx.get("foot_landmark_packet")) if ctx.get("foot_landmark_packet") is not None else None,
        "foot_landmark_weight": float(ctx.get("foot_landmark_weight", 1.0)),
        "foot_transport_pin_scale": float(ctx.get("foot_transport_pin_scale", 30.0)),
        # VQ-FIX2: the per-accepted-step swing-twist clamp reads these from
        # ctx, but they were never copied into torch_ctx — the clamp (knees/
        # ankles/clavicles, and now the AXIAL_LIMITS forearm/wrist bounds)
        # was silently INERT on the torch path. They stay numpy; the clamp
        # block runs on the CPU-side state each accepted step.
        "twist_limits": ctx.get("twist_limits"),
        "twist_joint_indices": ctx.get("twist_joint_indices", {}),
        "twist_bone_axes": ctx.get("twist_bone_axes", {}),
        "twist_limit_mode": ctx.get("twist_limit_mode", "clamp"),
        "twist_soft_joints": ctx.get("twist_soft_joints", ()),
        "twist_safety_margin_deg": ctx.get("twist_safety_margin_deg", 10.0),
        "limit_projection": ctx.get("limit_projection", "post"),
        "lm_trace": ctx.get("lm_trace"),
        "twist_soft_band_rad": ctx.get("twist_soft_band_rad"),
        "twist_soft_k": ctx.get("twist_soft_k"),
        "redundant_dof_prior": ctx.get("redundant_dof_prior", "none"),
        "redundant_dof_joints": ctx.get("redundant_dof_joints", []),
        "redundant_dof_prev_global_quats": to_t(ctx.get("redundant_dof_prev_global_quats")) if ctx.get("redundant_dof_prev_global_quats") is not None else None,
        "redundant_dof_init_global_quats": to_t(ctx.get("redundant_dof_init_global_quats")) if ctx.get("redundant_dof_init_global_quats") is not None else None,
        "redundant_dof_kc": float(ctx.get("redundant_dof_kc", 1.0)),
        "redundant_dof_kr": float(ctx.get("redundant_dof_kr", 0.05)),
    }

    # Convert limb pinned info to torch tensors
    if ctx["limb_pinned_info"]:
        limb_infos = ctx["limb_pinned_info"]
        torch_ctx["limb_pinned_info"] = [{
            "p": info["p"],
            "c": info["c"],
            "bone": to_t(info["bone"]),
            "p_parent": info["p_parent"],
            "twist_target": info.get("twist_target", 0.0),
            "twist_weight": info.get("twist_weight", 1.0),
            "twist_pin_multiplier": info.get("twist_pin_multiplier", 1.0),
            "side": info["side"] if "side" in info else None,
        } for info in limb_infos]
    else:
        torch_ctx["limb_pinned_info"] = []

    torch_ctx["foot_transport_pins"] = [{
        "side": pin["side"],
        "joint_idx": pin["joint_idx"],
        "target_quaternion": to_t(pin["target_quaternion"]),
        "weight": float(pin.get("weight", 1.0)),
    } for pin in ctx.get("foot_transport_pins", ())]

    # Convert wrist targets to torch tensors (legacy)
    wrist_targets = ctx.get("wrist_targets", {})
    if wrist_targets:
        torch_ctx["wrist_targets"] = {wi: to_t(q) for wi, q in wrist_targets.items()}
    else:
        torch_ctx["wrist_targets"] = {}

    # ── Build Kabsch head rotation ─────────────────────────────────────
    # When use_kabsch is True, we have ≥3 face keypoints.  Compute the
    # Kabsch rotation ONCE here (it depends only on measured data, not on
    # the optimization state), convert to a quaternion, and pass that into
    # the residual.  The residual then only needs to compare the current
    # head rotation to this static target — no per-iteration SVD.
    torch_ctx["kabsch_quat"] = None
    if torch_ctx.get("use_kabsch") and torch_ctx.get("canonical_head") is not None:
        from aimocap.math.kabsch import compute_head_rotation_kabsch
        face_kpts = ctx.get("face_kpts", {})
        neck_pos = ctx["tgt_pos"][ctx["neck_idx"]]  # (3,) numpy
        R_kabsch = compute_head_rotation_kabsch(face_kpts, neck_pos)
        if R_kabsch is not None:
            # R_kabsch maps template → observed, i.e. it's the head's
            # global orientation relative to the neck frame.
            torch_ctx["kabsch_quat"] = to_t(
                R_kabsch.as_quat().astype(np.float32))  # (4,) xyzw
        else:
            torch_ctx["use_kabsch"] = False  # not enough non-collinear points

    # ── Run LM loop on GPU ───────────────────────────────────────────
    x = torch.tensor(x0, device=device, dtype=dtype).unsqueeze(0)  # (1, num_vars)
    x = _lm_loop_torch(x, torch_ctx, num_vars,
                        max_iter=int(ctx.get("max_iter", 200)))
    solver._last_backend_stats = dict(torch_ctx.get("_lm_stats", {
        "backend": "torch-cuda",
        "termination_reason": "unknown",
        "iterations": 0,
        "final_cost": float("nan"),
    }))
    return x.squeeze(0).cpu().numpy()


def _lm_loop_torch(
    x: "torch.Tensor",
    ctx: dict,
    num_vars: int,
    max_iter: int = 200,
    ftol: float = 1e-5,
    xtol: float = 1e-5,
    lambda_init: float = 1e-3,
) -> "torch.Tensor":
    """Levenberg-Marquardt loop on GPU with batched finite-diff Jacobian.

    The key optimization: instead of 75 sequential residual evals for the
    Jacobian (like scipy does), we batch all 75 perturbations into ONE GPU
    call — (76, 75) tensor, where row 0 is the unperturbed state and rows
    1..75 are x + eps*e_i.  This turns 75x GPU overhead into 1x.

    Benchmarking showed FP64 and FP32 have identical per-call speed on the
    RTX 4060 for our batch size (76×75) — the Ada Lovelace FP64 throttle
    doesn't apply to small tensors.  FP64 is used for temporal accuracy
    (prevents MPJPE divergence over 300+ frames).
    """
    import torch
    from aimocap.retarget.ik_torch import _residuals_torch

    device = x.device
    dtype = x.dtype
    lam = lambda_init
    eps = 1e-5 if dtype == torch.float32 else 1e-7

    # Pre-build the perturbation matrix: (num_vars+1, num_vars)
    I_aug = torch.eye(num_vars, device=device, dtype=dtype)
    I_aug = torch.cat([torch.zeros(1, num_vars, device=device, dtype=dtype), I_aug])

    projected = ctx.get("limit_projection", "post") == "projected"
    if projected:
        # The warm start is part of the feasible-state contract. This is
        # deliberately outside the default branch to keep post mode exact.
        x = _project_state_torch(x, ctx)

    # Initial cost
    with torch.no_grad():
        r0 = _residuals_torch(x, ctx)
        cost = (r0 ** 2).sum().item()
    trace = ctx.get("lm_trace")
    if trace is not None and not isinstance(trace, list):
        raise TypeError("ctx['lm_trace'] must be a list when provided")
    ctx["_lm_stats"] = {
        "backend": "torch-cuda",
        "termination_reason": "max_iterations",
        "iterations": 0,
        "final_cost": float(cost),
    }

    for iteration in range(max_iter):
        # ── Batched Jacobian: 1 GPU call for all 75 perturbations ──────
        with torch.no_grad():
            x_batch = x.expand(num_vars + 1, -1) + eps * I_aug  # (76, 75)
            r_batch = _residuals_torch(x_batch, ctx)             # (76, M)
            r0 = r_batch[0:1, :]  # (1, M)
            r_pert = r_batch[1:, :]  # (75, M)
            J = ((r_pert - r0) / eps).T  # (M, 75)

        # LM normal equations: (J^T J + lam * diag(JtJ)) dx = -J^T r
        JtJ = J.T @ J                         # (num_vars, num_vars)
        Jtr = J.T @ r0[0]                     # (num_vars,)
        diag_JtJ = torch.diag(JtJ)
        diag_JtJ = torch.where(diag_JtJ > 1e-10, diag_JtJ, torch.ones_like(diag_JtJ))

        # Try step with increasing lambda — C2.4 REJECTED and reverted: the
        # batched lambda ladder was measured NOT bit-exact end-to-end (the
        # LM trajectory diverged: max abs diff 6.265 at 74% of elements on
        # the 450-frame clip; iteration counts changed on warm-started
        # frames). The sequential loop is the bit-exact reference; retained
        # here verbatim.
        improved = False
        # Projected trials can repeatedly map back to the same feasible state
        # at an active hard limit. Keep the historical 20-trial ladder for the
        # default post mode, but bound the opt-in projected ladder so a stalled
        # constrained step cannot dominate frame runtime.
        max_inner = 8 if projected else 20
        for _inner in range(max_inner):
            A = JtJ + lam * torch.diag(diag_JtJ)
            try:
                dx = torch.linalg.solve(A, -Jtr)  # (num_vars,)
            except RuntimeError:
                lam *= 10.0
                continue

            x_new = x + dx.unsqueeze(0)
            with torch.no_grad():
                # In projected mode the trial itself is feasible. The cost
                # comparison and lambda update therefore use the projected
                # state, not the infeasible linearized trial.
                x_trial = _project_state_torch(x_new, ctx) if projected else x_new
                r_trial = _residuals_torch(x_trial, ctx)
                cost_trial = (r_trial ** 2).sum().item()
                cost_unprojected = cost_trial
                if projected and trace is not None:
                    # This value is diagnostic-only. Do not pay for a second
                    # residual evaluation in production projected solves.
                    r_unprojected = _residuals_torch(x_new, ctx)
                    cost_unprojected = (r_unprojected ** 2).sum().item()

            if trace is not None:
                if projected:
                    cost_projected = cost_trial
                else:
                    x_projected = _project_state_torch(x_new, ctx)
                    r_projected = _residuals_torch(x_projected, ctx)
                    cost_projected = (r_projected ** 2).sum().item()
                trace.append({
                    "iteration": int(iteration),
                    "inner": int(_inner),
                    "cost_x": float(cost),
                    "cost_x_plus_dx": float(cost_unprojected),
                    "cost_projected": float(cost_projected),
                    "accepted": bool(cost_trial < cost),
                })

            if projected:
                projected_step_norm = (x_trial - x).norm().item()
                if projected_step_norm <= xtol:
                    # A trial that projects back to the current feasible state
                    # cannot make meaningful constrained progress. Accept a
                    # strictly improving tiny step, otherwise retain x, and
                    # terminate before another lambda ladder/Jacobian pass.
                    if cost_trial < cost:
                        x = x_trial
                        cost = cost_trial
                    ctx["_lm_stats"].update({
                        "termination_reason": "projected_xtol",
                        "iterations": iteration + 1,
                        "final_cost": float(cost),
                    })
                    return x

            if cost_trial < cost:
                if projected:
                    step_norm = (x_trial - x).norm().item()
                    x = x_trial
                    cost_new = cost_trial
                else:
                    # Keep the historical post-step order and arithmetic
                    # untouched: acceptance is on cost(x+dx), then the
                    # accepted state is projected after that comparison.
                    step_norm = dx.norm().item()
                    x = x_new
                    cost_new = cost_trial
                    # ── Swing-twist twist clamping on hinge joints ────────
                    # After each accepted LM step, decompose each constrained
                    # bone's local rotation into swing + twist, clamp the twist
                    # to anatomical limits, and reconstruct.  This prevents
                    # caved-in knees, spinning hands, and twisted forearms.
                    if ctx.get("twist_limits") and ctx.get("twist_limit_mode", "clamp") in {"clamp", "hybrid"}:
                        x_np = _apply_twist_limits_numpy(x[0].cpu().numpy(), ctx)
                        x = torch.tensor(x_np, device=device, dtype=dtype).unsqueeze(0)
                rel_change = abs(cost - cost_new) / max(cost, 1e-10)
                if rel_change < ftol:
                    ctx["_lm_stats"].update({
                        "termination_reason": "ftol",
                        "iterations": iteration + 1,
                        "final_cost": float(cost_new),
                    })
                    return x
                cost = cost_new
                lam = max(lam * 0.3, 1e-12)
                improved = True
                break
            else:
                lam *= 10.0

        if not improved:
            ctx["_lm_stats"].update({
                "termination_reason": "stuck",
                "iterations": iteration + 1,
                "final_cost": float(cost),
            })
            return x  # converged or stuck
        if step_norm < xtol:
            ctx["_lm_stats"].update({
                "termination_reason": "xtol",
                "iterations": iteration + 1,
                "final_cost": float(cost),
            })
            return x

    ctx["_lm_stats"].update({
        "termination_reason": "max_iterations",
        "iterations": max_iter,
        "final_cost": float(cost),
    })
    return x
