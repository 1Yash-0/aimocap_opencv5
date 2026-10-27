"""GPU-accelerated IK solver using PyTorch CUDA.

Drop-in replacement for ``MocapIKSolver.solve_frame`` that produces identical
results but runs ~100x faster by:
  1. Precomputing all frame-constant quantities once per frame (not per eval)
  2. Computing the Jacobian analytically via torch.autograd (not finite-diff)
  3. Running the LM loop on GPU with batched tensor ops

Falls back to scipy if CUDA is unavailable or parity check fails.
"""
from __future__ import annotations


try:
    import torch
    _HAS_TORCH = True
    _HAS_CUDA = torch.cuda.is_available()
except ImportError:
    _HAS_TORCH = False
    _HAS_CUDA = False


# ── Quaternion helpers (xyzw convention, matching scipy) ─────────────────

def _rotvec_to_quat(rv: torch.Tensor) -> torch.Tensor:
    """Convert rotation vector(s) to quaternion(s).  rv: (..., 3) -> (..., 4) xyzw."""
    angle = torch.norm(rv, dim=-1, keepdim=True)  # (..., 1)
    # near-zero angle: identity quaternion (avoid div-by-zero)
    small = angle < 1e-8
    axis = torch.where(small, torch.zeros_like(rv), rv / (angle + 1e-12))
    s = torch.sin(angle / 2.0)
    c = torch.cos(angle / 2.0)
    return torch.cat([axis * s, c], dim=-1)  # xyzw


def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product.  q1, q2: (..., 4) xyzw -> (..., 4) xyzw."""
    x1, y1, z1, w1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    x2, y2, z2, w2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return torch.stack([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ], dim=-1)


def _quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vector v by quaternion q.  q: (..., 4) xyzw, v: (..., 3) -> (..., 3)."""
    xyz = q[..., :3]  # (..., 3)
    w = q[..., 3:4]   # (..., 1)
    # t = 2 * cross(q.xyz, v)
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    # v + w*t + cross(q.xyz, t)
    return v + w * t + torch.cross(xyz, t, dim=-1)


def _quat_inv(q: torch.Tensor) -> torch.Tensor:
    """Inverse of a unit quaternion = conjugate.  q: (..., 4) xyzw -> (..., 4)."""
    return torch.cat([-q[..., :3], q[..., 3:4]], dim=-1)


def _quat_to_rotvec(q: torch.Tensor) -> torch.Tensor:
    """Convert quaternion to rotation vector (SO(3) log map).
    q: (..., 4) xyzw -> (..., 3)."""
    # Ensure positive w (canonical form) for continuous angle
    w = q[..., 3:4]
    flip = w < 0
    q = torch.where(flip, -q, q)
    w = q[..., 3:4]
    xyz = q[..., :3]
    # angle = 2 * atan2(|xyz|, w)
    n_xyz = torch.norm(xyz, dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(n_xyz, w)
    # near-zero angle
    small = angle < 1e-8
    axis = torch.where(small, torch.zeros_like(xyz), xyz / (n_xyz + 1e-12))
    return axis * angle


def _extract_twist_angle_torch(q: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    """Extract signed swing-twist angle, matching ``clamp_twist``.

    ``q`` is (..., 4) xyzw and ``axis`` is (..., 3) unit length.  The
    projection is intentionally the same vector-part projection used by
    ``swing_twist_decompose`` in ``swing_twist.py``.  The canonical positive
    scalar part gives scipy's shortest signed rotvec branch.
    """
    axis = axis / (torch.linalg.vector_norm(axis, dim=-1, keepdim=True) + 1e-12)
    projected = (q[..., :3] * axis).sum(dim=-1, keepdim=True) * axis
    twist = torch.cat([projected, q[..., 3:4]], dim=-1)
    twist = twist / (torch.linalg.vector_norm(twist, dim=-1, keepdim=True) + 1e-12)
    flip = twist[..., 3:4] < 0
    twist = torch.where(flip, -twist, twist)
    return 2.0 * torch.atan2(
        (twist[..., :3] * axis).sum(dim=-1),
        twist[..., 3],
    )


def _soft_twist_hinge_torch(excess: torch.Tensor, delta: float) -> torch.Tensor:
    """C1 quadratic blend of a positive-part hinge.

    The input and ``delta`` are radians.  The branch is zero through
    ``-delta``; over the two-degree pre-limit band it is quadratic; above
    the limit it has unit slope.  The positive offset ``delta/2`` is the
    unique C1 continuation of that quadratic branch.
    """
    d = torch.as_tensor(delta, device=excess.device, dtype=excess.dtype)
    low = torch.zeros_like(excess)
    band = (excess + d) ** 2 / (2.0 * d)
    high = excess + d / 2.0
    return torch.where(excess <= -d, low, torch.where(excess < 0.0, band, high))



# ── Forward kinematics (batched, torch) ──────────────────────────────────

def _fk_torch(
    x: torch.Tensor,
    rest_t0: torch.Tensor,
    rest_offsets: torch.Tensor,
    parents: torch.Tensor,
    levels_flat: torch.Tensor,
    level_starts: torch.Tensor,
    num_joints: int,
    levels_pre: list | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward kinematics for a batch of states.

    x: (B, 3 + num_joints*3) — root translation + per-joint rotvecs
    levels_pre: optional precomputed per-level (children, par, off) tensors
      (C2.2: skeleton-static indexing hoisted out of the hot loop; same
      values, bit-exact). Built once per solver by ``build_fk_levels``.
    Returns: (global_pos (B, J, 3), global_quat (B, J, 4) xyzw)
    """
    B = x.shape[0]
    device = x.device
    dtype = x.dtype

    root_t = x[:, :3]                    # (B, 3)
    rotvecs = x[:, 3:].reshape(B, num_joints, 3)  # (B, J, 3)
    local_quat = _rotvec_to_quat(rotvecs)  # (B, J, 4)

    # C2.3: every entry (root + all levels) is written below, so empty is
    # bit-exact vs zeros and skips the memset kernels.
    gpos = torch.empty(B, num_joints, 3, device=device, dtype=dtype)
    grot = torch.empty(B, num_joints, 4, device=device, dtype=dtype)

    # Root
    gpos[:, 0, :] = rest_t0 + root_t     # (B, 3)
    grot[:, 0, :] = local_quat[:, 0, :]  # (B, 4)

    if levels_pre is not None:
        n_levels = len(levels_pre)
        level_pairs = levels_pre
    else:
        n_levels = len(level_starts) - 1
        level_pairs = []
        for li in range(n_levels):
            s = level_starts[li]
            e = level_starts[li + 1]
            children = levels_flat[s:e]      # (n_in_level,)
            par = parents[children]          # (n_in_level,)
            off = rest_offsets[children]     # (n_in_level, 3)
            level_pairs.append((children, par, off))

    for children, par, off in level_pairs:
        # Parent global rotation: (B, n_in_level, 4)
        qp = grot[:, par, :]
        # Parent applies rest offset of child: R_parent.apply(rest_offset[child])
        # Need (B, n_in_level, 3) = apply qp to off broadcast
        rotated_off = _quat_apply(qp, off.unsqueeze(0).expand(B, -1, -1))
        gpos[:, children, :] = gpos[:, par, :] + rotated_off
        # Global rot = parent * local
        ql = local_quat[:, children, :]
        grot[:, children, :] = _quat_mul(qp, ql)

    return gpos, grot


def build_fk_levels(parents, levels_list, rest_offsets, device, index_dtype, dtype) -> list:
    """C2.2: precompute per-level (children, par, off) index tensors once.

    Skeleton-static: called once per solver, reused for every frame and
    evaluation. parents/levels_list/rest_offsets are the NUMPY arrays
    (solver.parents, solver.levels, ctx["rest_offsets"]); the torch
    conversion matches ``to_t`` exactly so FK stays bit-exact.
    """
    import numpy as np
    parents_np = np.asarray(parents)
    rest_np = np.ascontiguousarray(rest_offsets)
    pairs = []
    for lvl in levels_list:
        children_np = np.asarray(lvl, dtype=np.int64)
        par_np = parents_np[children_np]
        off_np = rest_np[children_np]
        pairs.append((
            torch.tensor(children_np, device=device, dtype=index_dtype),
            torch.tensor(np.ascontiguousarray(par_np), device=device, dtype=index_dtype),
            torch.tensor(off_np, device=device, dtype=dtype),
        ))
    return pairs


# ── Residual computation (torch, differentiable) ─────────────────────────

def _residuals_torch(
    x: torch.Tensor,
    ctx: dict,
) -> torch.Tensor:
    """Compute the full residual vector.  x: (B, num_vars) -> (B, M).

    ctx contains all frame-constant quantities (precomputed once per frame).
    """
    B = x.shape[0]
    num_joints = ctx["num_joints"]
    device = x.device
    dtype = x.dtype

    # FK
    gpos, grot = _fk_torch(
        x, ctx["rest_t0"], ctx["rest_offsets"], ctx["parents"],
        ctx["levels_flat"], ctx["level_starts"], num_joints,
        levels_pre=ctx.get("levels_pre"))

    parts = []

    # ── Term 0: Position residual ────────────────────────────────────────
    # (global_pos - tgt_pos) * w[:, None]
    tgt_pos = ctx["tgt_pos"]             # (J, 3)
    w = ctx["w"]                         # (J,)
    # D1: NaN*0 = NaN in IEEE — replace NaN targets with 0 so w=0 joints
    # contribute exactly 0 (finite targets untouched, bit-exact).
    if not bool(torch.isfinite(tgt_pos).all()):
        tgt_pos = torch.where(torch.isfinite(tgt_pos), tgt_pos, torch.zeros_like(tgt_pos))
    res_pos = (gpos - tgt_pos.unsqueeze(0)) * w.unsqueeze(0).unsqueeze(-1)  # (B, J, 3)
    parts.append(res_pos.reshape(B, -1))

    # ── Terms 1-5 (inside ori_weight > 0) ────────────────────────────────
    if ctx["ori_weight"] > 0:
        ow = ctx["ori_weight"]

        # Term 1: Spine orientation
        spine_indices = ctx["spine_indices"]      # (n_spine,)
        R_targets = ctx["R_target_quats"]         # (n_spine, 4)
        n_spine = len(spine_indices)
        if n_spine > 0:
            R_current = grot[:, spine_indices, :]  # (B, n_spine, 4)
            # ori_res = (R_current * R_target.inv()).as_rotvec()
            R_t_inv = _quat_inv(R_targets.unsqueeze(0))  # (1, n_spine, 4)
            R_diff_spine = _quat_mul(R_current, R_t_inv)  # (B, n_spine, 4)
            ori_res = _quat_to_rotvec(R_diff_spine)       # (B, n_spine, 3)
            parts.append(ori_res.reshape(B, -1) * ow)

        # Term 2: Head orientation (constant length 3)
        # Frame-based head tracking: pin head global rotation to the
        # precomputed target from face keypoints. Falls back to neck
        # rotation (torso-follow) when target is unavailable.
        if ctx["head_ori_active"]:
            head_i = ctx["head_idx"]
            neck_i = ctx["neck_idx"]
            R_head_current = grot[:, head_i, :]    # (B, 4)

            htq = ctx.get("head_target_quat")
            if htq is not None:
                R_head_target = htq.unsqueeze(0).expand(B, -1)  # (B, 4)
            else:
                R_head_target = grot[:, neck_i, :]  # torso-follow fallback

            head_res = _quat_to_rotvec(_quat_mul(
                R_head_current, _quat_inv(R_head_target)))  # (B, 3)
            parts.append(head_res * ow * ctx.get("head_orientation_weight", 1.0))
        else:
            parts.append(torch.zeros(B, 3, device=device, dtype=dtype))

        # Term 3: Pelvis-neck lateral sway
        pelvis_i = ctx["pelvis_idx"]
        neck_i_s = ctx["neck_idx"]
        lat_axis = ctx["lat_axis"]               # (3,)
        neck_off = gpos[:, neck_i_s, :] - gpos[:, pelvis_i, :]  # (B, 3)
        lat = (neck_off * lat_axis).sum(dim=-1)  # (B,)
        SWAY_LIMIT = 8.0
        excess = torch.clamp(torch.abs(lat) - SWAY_LIMIT, min=0.0) * torch.sign(lat)
        parts.append((excess * 20.0).unsqueeze(-1))  # (B, 1)

        # Term 4: Sway velocity damping
        if ctx["prev_lat"] is not None:
            v = lat - ctx["prev_lat"]
            VEL_LIMIT = 0.5
            excess_v = torch.clamp(torch.abs(v) - VEL_LIMIT, min=0.0) * torch.sign(v)
            parts.append((excess_v * 10.0).unsqueeze(-1))  # (B, 1)
        else:
            parts.append(torch.zeros(B, 1, device=device, dtype=dtype))

        # Term 5: Limb twist
        if ctx["limb_pinned_info"] is not None and len(ctx["limb_pinned_info"]) > 0:
            limb_res_list = []
            for info in ctx["limb_pinned_info"]:
                p_idx = info["p"]           # parent proxy idx
                c_idx = info["c"]           # child proxy idx
                bone = info["bone"]         # (3,) normalized rest offset
                p_parent = info["p_parent"] # parent of parent
                twist_target = info.get("twist_target", 0.0)
                twist_weight = info.get("twist_weight", 1.0)
                pin_multiplier = info.get("twist_pin_multiplier", 1.0)
                if ctx.get("limb_twist_pin", "always") == "off":
                    continue
                
                # R_current = global_rots[p_parent].inv() * global_rots[p]
                R_pp = grot[:, p_parent, :]
                R_p = grot[:, p_idx, :]
                R_current = _quat_mul(_quat_inv(R_pp), R_p)
                rv = _quat_to_rotvec(R_current)    # (B, 3)
                if ctx.get("limb_twist_anchor") == "smoothed":
                    twist_rad = _extract_twist_angle_torch(
                        R_current, bone.unsqueeze(0).expand(B, -1))
                    twist_error = torch.atan2(
                        torch.sin(twist_rad - twist_target),
                        torch.cos(twist_rad - twist_target),
                    ) * twist_weight * pin_multiplier
                else:
                    twist_rad = (rv * bone).sum(dim=-1)  # (B,)
                    twist_error = (twist_rad - twist_target) * twist_weight * pin_multiplier
                limb_res_list.append(twist_error.unsqueeze(-1) * bone)  # (B, 3)
            if limb_res_list:
                limb_res = torch.stack(limb_res_list, dim=1)  # (B, n_pin, 3)
                parts.append(limb_res.reshape(B, -1) * ow)

    # Soft swing-twist limits are residuals, not projections.  The local
    # rotation and axis match the post-step clamp's inputs exactly.
    limit_mode = ctx.get("twist_limit_mode", "clamp")
    if limit_mode in {"soft", "hybrid"}:
        rv_local = x[:, 3:].reshape(B, num_joints, 3)
        q_local = _rotvec_to_quat(rv_local)
        soft_parts = []
        soft_names = (set(ctx.get("twist_limits") or {})
                      if limit_mode == "soft" else set(ctx.get("twist_soft_joints") or ()))
        delta = float(ctx.get("twist_soft_band_rad", 2.0 * 3.141592653589793 / 180.0))
        k = float(ctx.get("twist_soft_k", 180.0 / 3.141592653589793))
        for jname, max_tw_deg in (ctx.get("twist_limits") or {}).items():
            if jname not in soft_names:
                continue
            ji = ctx.get("twist_joint_indices", {}).get(jname)
            axis_np = ctx.get("twist_bone_axes", {}).get(jname)
            if ji is None or axis_np is None:
                continue
            axis = torch.as_tensor(axis_np, device=device, dtype=dtype)
            angle = _extract_twist_angle_torch(q_local[:, ji, :], axis.unsqueeze(0).expand(B, -1))
            excess = torch.abs(angle) - float(max_tw_deg) * 3.141592653589793 / 180.0
            soft_parts.append(_soft_twist_hinge_torch(excess, delta) * k)
        if soft_parts:
            parts.append(torch.stack(soft_parts, dim=-1))

    # Limb hinge constraints
    hinge_joints = ctx.get("hinge_joints", [])
    if hinge_joints:
        hinge_parts_swing = []
        hinge_parts_range = []
        rv_local = x[:, 3:].reshape(B, num_joints, 3)
        q_local = _rotvec_to_quat(rv_local)
        for hinge in hinge_joints:
            j = hinge["joint_idx"]
            h = torch.as_tensor(hinge["axis"], device=device, dtype=dtype)
            flex_min = torch.as_tensor(hinge["flex_min_rad"], device=device, dtype=dtype)
            flex_max = torch.as_tensor(hinge["flex_max_rad"], device=device, dtype=dtype)
            
            # Get rotation vector for this joint
            rv = x[:, 3+3*j:6+3*j]
            q = _rotvec_to_quat(rv)
            
            # Ensure positive scalar part (canonical form)
            w = q[:, 3:4]
            flip = w < 0
            q = torch.where(flip, -q, q)
            
            # Extract twist component
            v = q[:, :3]
            w = q[:, 3:4]
            proj = (v * h).sum(dim=-1, keepdim=True)
            theta = 2.0 * torch.atan2(proj, w)
            
            # Twist quaternion
            t = torch.cat([proj * h, w], dim=-1)
            t_norm = torch.linalg.vector_norm(t, dim=-1, keepdim=True)
            t = t / (t_norm + 1e-12)
            
            # Swing = q * conj(t)
            swing = _quat_mul(q, _quat_inv(t))
            
            # Swing residual
            res_swing = 30.0 * _quat_to_rotvec(swing)
            hinge_parts_swing.append(res_swing)
            
            # Range residual
            res_range_min = torch.relu(flex_min - theta)
            res_range_max = torch.relu(theta - flex_max)
            res_range = 30.0 * torch.cat([res_range_min, res_range_max], dim=-1)
            hinge_parts_range.append(res_range)
        
        if hinge_parts_swing:
            parts.append(torch.cat(hinge_parts_swing, dim=-1))
            parts.append(torch.cat(hinge_parts_range, dim=-1))

    packet = ctx.get("foot_landmark_packet")
    if packet is not None and packet.numel():
        bone = torch.round(packet[:, 0]).to(dtype=torch.long)
        pred = gpos[:, bone, :] + _quat_apply(
            grot[:, bone, :], packet[:, 4:7].unsqueeze(0).expand(B, -1, -1))
        residual = (pred - packet[:, 1:4].unsqueeze(0)) * packet[:, 7].unsqueeze(0).unsqueeze(-1)
        norm = torch.norm(residual, dim=-1, keepdim=True)
        factor = torch.sqrt(torch.clamp(2.0 / (norm + 1e-9), max=1.0))
        shaped = (residual * factor).reshape(B, -1)
        if ctx.get("foot_landmark_weight", 1.0) != 1.0:
            shaped = shaped * ctx["foot_landmark_weight"]
        parts.append(shaped)

    # Opt-in redundant-DOF Tikhonov prior.  Keep the default path exactly
    # unchanged by appending no rows unless explicitly enabled.
    if ctx.get("redundant_dof_prior", "none") == "minimal":
        prev_global = ctx.get("redundant_dof_prev_global_quats")
        init_global = ctx.get("redundant_dof_init_global_quats")
        for ji in ctx.get("redundant_dof_joints", ()):
            cur = grot[:, ji, :]
            if prev_global is not None:
                q_prev = prev_global.unsqueeze(0).expand(B, -1, -1)[:, ji, :]
                parts.append(_quat_to_rotvec(_quat_mul(
                    _quat_inv(q_prev), cur
                )) * ctx["redundant_dof_kc"])
            if init_global is not None:
                q_init = init_global.unsqueeze(0).expand(B, -1, -1)[:, ji, :]
                parts.append(_quat_to_rotvec(_quat_mul(
                    _quat_inv(q_init), cur
                )) * ctx["redundant_dof_kr"])

    # Term 6: Temporal (velocity - first-order: x - prev_x)
    if ctx["prev_x"] is not None and ctx["temporal_weight"] > 0:
        # Broadcast prev_x/init_x to batch size (they are (1, num_vars))
        prev_x_b = ctx["prev_x"].expand(B, -1)
        d_root = x[:, :3] - prev_x_b[:, :3]  # (B, 3)
        rv_cur = x[:, 3:].reshape(B, num_joints, 3)
        rv_prev = prev_x_b[:, 3:].reshape(B, num_joints, 3)
        q_cur = _rotvec_to_quat(rv_cur)
        q_prev = _rotvec_to_quat(rv_prev)
        d_rot = _quat_to_rotvec(_quat_mul(_quat_inv(q_prev), q_cur))  # (B, J, 3)
        state_delta = torch.cat([d_root, d_rot.reshape(B, -1)], dim=-1)
        parts.append(state_delta * ctx["temporal_weight"])

    # Term 6b: Acceleration penalty (second-order: x - 2*prev_x + prev_prev_x)
    # This is the KinePose-style acceleration penalty that prevents sudden
    # spikes and produces smoother motion IN the cost function, reducing
    # reliance on post-solve One-Euro filtering.
    if ctx.get("prev_prev_x") is not None and ctx.get("accel_weight", 0.0) > 0:
        prev_x_b = ctx["prev_x"].expand(B, -1)
        prev_prev_x_b = ctx["prev_prev_x"].expand(B, -1)
        # Acceleration = x - 2*prev_x + prev_prev_x
        # For root translation: simple second difference
        a_root = x[:, :3] - 2.0 * prev_x_b[:, :3] + prev_prev_x_b[:, :3]  # (B, 3)
        # For rotations: compute as (x - prev_x) - (prev_x - prev_prev_x)
        # in SO(3) log space, i.e. the difference of consecutive velocity deltas.
        rv_cur = x[:, 3:].reshape(B, num_joints, 3)
        rv_prev = prev_x_b[:, 3:].reshape(B, num_joints, 3)
        rv_prev2 = prev_prev_x_b[:, 3:].reshape(B, num_joints, 3)
        q_cur = _rotvec_to_quat(rv_cur)
        q_prev = _rotvec_to_quat(rv_prev)
        q_prev2 = _rotvec_to_quat(rv_prev2)
        # velocity delta: q_prev.inv * q_cur
        d_vel = _quat_to_rotvec(_quat_mul(_quat_inv(q_prev), q_cur))       # (B, J, 3)
        d_vel_prev = _quat_to_rotvec(_quat_mul(_quat_inv(q_prev2), q_prev))  # (B, J, 3)
        # acceleration = velocity - prev_velocity
        a_rot = d_vel - d_vel_prev  # (B, J, 3)
        state_accel = torch.cat([a_root, a_rot.reshape(B, -1)], dim=-1)
        parts.append(state_accel * ctx["accel_weight"])

    # Term 7: Init
    if ctx["init_x"] is not None:
        init_x_b = ctx["init_x"].expand(B, -1)
        d_root = x[:, :3] - init_x_b[:, :3]
        rv_cur = x[:, 3:].reshape(B, num_joints, 3)
        rv_init = init_x_b[:, 3:].reshape(B, num_joints, 3)
        q_cur = _rotvec_to_quat(rv_cur)
        q_init = _rotvec_to_quat(rv_init)
        d_rot = _quat_to_rotvec(_quat_mul(_quat_inv(q_init), q_cur))
        state_delta = torch.cat([d_root, d_rot.reshape(B, -1)], dim=-1)
        parts.append(state_delta * ctx["init_weight"])

    # VQ-FIX6: for an unhealthy foot observation, transport the foot's local
    # SO(3) orientation from the previous solved state (analytic init at f=0).
    # The scale is the same unit-equivalent orientation scale as limb twist pins.
    for pin in ctx.get("foot_transport_pins", ()):
        joint = pin["joint_idx"]
        rv_cur = x[:, 3:].reshape(B, num_joints, 3)[:, joint, :]
        q_cur = _rotvec_to_quat(rv_cur)
        q_target = pin["target_quaternion"].expand(B, -1)
        r_pin = _quat_to_rotvec(_quat_mul(_quat_inv(q_target), q_cur))
        parts.append(r_pin * (ctx.get("foot_transport_pin_scale", 30.0)
                              * pin.get("weight", 1.0)))

    # Residual length assertion — catches silent changes when residuals
    # are added/removed in only one code path (scipy vs torch).
    r = torch.cat(parts, dim=-1)  # (B, M)
    M = r.shape[-1]
    if "_resid_len" not in ctx:
        ctx["_resid_len"] = M
    elif M != ctx["_resid_len"]:
        raise RuntimeError(f"residual length changed: {M} != {ctx['_resid_len']}")
    return r


# ── LM solver (torch, GPU) ───────────────────────────────────────────────
