import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from aimocap.retarget.bvh import write_bvh
from aimocap.retarget.fbx_rig import Skeleton
from aimocap.retarget.foot_lock import apply_foot_lock
from aimocap.retarget.mocap_ik import MocapIKSolver
from aimocap.retarget.mocap_skeleton import (
    MocapSkeleton,
    extract_mocap_points,
    COCO,
)
from aimocap.retarget.temporal import stabilize_lower_body_animation
from aimocap.retarget.temporal import (
    fill_missing_target_tracks,
    stabilize_mocap_fit_targets,
)
from aimocap.retarget.observation_conditioning import condition_target_tracks
from aimocap.retarget.rigid_segments import RIGID_SEGMENTS, rigid_denoise_targets



def _module_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _effective_pipeline_options(
    pipeline_profile: str,
    target_conditioning: str | None,
    leg_stabilization: bool | None,
    foot_lock: bool | None,
) -> tuple[str, bool, bool]:
    if pipeline_profile not in {"v7", "v8"}:
        raise ValueError(f"pipeline_profile must be 'v7' or 'v8', got {pipeline_profile!r}")
    if pipeline_profile == "v8":
        return (
            "p1" if target_conditioning is None else target_conditioning,
            False if leg_stabilization is None else bool(leg_stabilization),
            False if foot_lock is None else bool(foot_lock),
        )
    return (
        "legacy" if target_conditioning is None else target_conditioning,
        True if leg_stabilization is None else bool(leg_stabilization),
        True if foot_lock is None else bool(foot_lock),
    )



def _validate_fbx_hierarchy(fbx_skel: Skeleton) -> tuple[int, int]:
    root_indices = [i for i, p in enumerate(fbx_skel.parents) if p == -1]
    if len(root_indices) != 1:
        raise ValueError(f"Expected exactly one FBX root, found {len(root_indices)}")

    for i, p in enumerate(fbx_skel.parents):
        if p >= i:
            raise ValueError(
                f"FBX hierarchy is not topologically sorted at {fbx_skel.node_names[i]}"
            )

    return root_indices[0], fbx_skel.name_to_idx["pelvis"]




def _compose_p1_targets(
    raw_targets: dict[str, np.ndarray], fps: float
) -> tuple[dict[str, np.ndarray], dict]:
    """Compose exact P1 solver targets from raw centimetre-space tracks."""
    c1b_targets, c1b_diag = condition_target_tracks(raw_targets, fps=fps)
    rigid_targets, rigid_diag = rigid_denoise_targets(
        raw_targets, segments=RIGID_SEGMENTS
    )
    rigid_targets = fill_missing_target_tracks(rigid_targets)
    merged = {
        name: np.asarray(value, dtype=np.float64).copy()
        for name, value in c1b_targets.items()
    }
    rigid_names = {name for names in RIGID_SEGMENTS.values() for name in names}
    for name in rigid_names:
        if name in rigid_targets:
            merged[name] = rigid_targets[name].copy()
    if "hip_l" in merged and "hip_r" in merged:
        merged["pelvis"] = (merged["hip_l"] + merged["hip_r"]) * 0.5
    if "shoulder_l" in merged and "shoulder_r" in merged:
        merged["neck"] = (merged["shoulder_l"] + merged["shoulder_r"]) * 0.5
    return merged, {
        "enabled": True,
        "mode": "p1",
        "composition": "C1b all tracks; C3h face and hands; legacy fill on C3h tracks; pelvis/neck midpoint recompute",
        "c1b": c1b_diag,
        "c3h": rigid_diag,
        "rigid_segments": {name: list(names) for name, names in RIGID_SEGMENTS.items()},
    }

def _compose_p2_targets(
    raw_targets: dict[str, np.ndarray], fps: float
) -> tuple[dict[str, np.ndarray], dict]:
    """Compose P2/FB targets: C3h observations followed by C1b."""
    c1b_targets, c1b_diag = condition_target_tracks(raw_targets, fps=fps)
    rigid_targets, rigid_diag = rigid_denoise_targets(
        raw_targets, segments=RIGID_SEGMENTS
    )
    merged = {
        name: np.asarray(value, dtype=np.float64).copy()
        for name, value in c1b_targets.items()
    }
    segment_c1b_diag = {}
    for segment, names in RIGID_SEGMENTS.items():
        conditioned, diag = condition_target_tracks(
            {name: np.asarray(rigid_targets[name], dtype=np.float64) for name in names},
            fps=fps,
        )
        segment_c1b_diag[segment] = diag
        for name in names:
            if name in conditioned:
                merged[name] = np.asarray(conditioned[name], dtype=np.float64).copy()
    if "hip_l" in merged and "hip_r" in merged:
        merged["pelvis"] = (merged["hip_l"] + merged["hip_r"]) * 0.5
    if "shoulder_l" in merged and "shoulder_r" in merged:
        merged["neck"] = (merged["shoulder_l"] + merged["shoulder_r"]) * 0.5
    return merged, {
        "enabled": True,
        "mode": "p2",
        "composition": "C1b body/all non-rigid tracks; C3h observations then C1b on face/hand segments; rejected C3h segments fall back to C1b(raw); pelvis/neck midpoint recompute",
        "c1b": c1b_diag,
        "c3h": rigid_diag,
        "segment_c1b": segment_c1b_diag,
        "rigid_segments": {name: list(names) for name, names in RIGID_SEGMENTS.items()},
    }


def retarget_to_fbx(
    triangulated_npz: str,
    fbx_rig_path: str,
    output_bvh: str,
    max_frames: int = None,
    fps: float = 30.0,
    mocap_target_stabilization: bool = True,
    mocap_target_cutoff_hz: float = 1.5,
    target_conditioning: str | None = None,
    leg_stabilization: bool | None = None,
    leg_stabilize_cutoff_hz: float = 1.5,
    leg_max_deg_per_frame: float = 30.0,
    foot_lock: bool | None = None,
    foot_lock_anchor_policy: str = "fixed",
    foot_lock_debug_dir: str | None = None,
    pose2d_npz: str | None = None,
    arm_orientation_rewrite: bool = False,
    export_fbx: bool = True,
    require_grounded: bool = False,
    min_core_finite_fraction: float = 0.5,
    profile: bool = False,
    pipeline_profile: str = "v7",
    foot_landmark_model: str = "proxy",
    limb_twist_anchor: str | None = None,
    two_pass_limbs: str = "all",
    two_pass_pass1_stride: int = 1,
    two_pass_pass1_max_iter: int | None = None,
    smoothed_limbs: str = "all",
    limb_twist_pin: str = "always",
    twist_limit_mode: str = "clamp",
    limit_projection: str = "post",
    soft_limit_joints=None,
    ankle_twist_limit_deg: float | None = None,
    redundant_dof_prior: str = "none",
    head_target_mode: str = "legacy",
    foot_landmark_weight: float = 1.0,
    face_position_weight: float = 1.0,
    foot_gate_reference: str = "solved",
    limb_hinge: str = "off",
):
    if limb_twist_anchor is None:
        limb_twist_anchor = "init"
    if limb_twist_anchor not in {"init", "continuous", "continuous_resid", "smoothed", "two_pass"}:
        raise ValueError(
            "limb_twist_anchor must be 'init', 'continuous', 'continuous_resid', "
            "'smoothed', or 'two_pass', got "
            f"{limb_twist_anchor!r}"
        )
    if two_pass_limbs not in {"all", "arms"}:
        raise ValueError(
            "two_pass_limbs must be 'all' or 'arms', got " f"{two_pass_limbs!r}"
        )
    if (not isinstance(two_pass_pass1_stride, (int, np.integer))
            or isinstance(two_pass_pass1_stride, bool)
            or two_pass_pass1_stride < 1):
        raise ValueError(
            "two_pass_pass1_stride must be a positive integer, "
            f"got {two_pass_pass1_stride!r}"
        )
    if two_pass_pass1_max_iter is not None:
        if (not isinstance(two_pass_pass1_max_iter, (int, np.integer))
                or isinstance(two_pass_pass1_max_iter, bool)
                or two_pass_pass1_max_iter < 1):
            raise ValueError(
                "two_pass_pass1_max_iter must be None or a positive integer, "
                f"got {two_pass_pass1_max_iter!r}"
            )
    if smoothed_limbs not in {"all", "arms"}:
        raise ValueError(
            "smoothed_limbs must be 'all' or 'arms', got " f"{smoothed_limbs!r}"
        )
    if limb_twist_pin not in {"always", "fade", "off"}:
        raise ValueError(
            "limb_twist_pin must be 'always', 'fade', or 'off', "
            f"got {limb_twist_pin!r}"
        )
    if twist_limit_mode not in {"clamp", "soft", "hybrid", "off"}:
        raise ValueError(
            "twist_limit_mode must be 'clamp', 'soft', 'hybrid', or 'off', "
            f"got {twist_limit_mode!r}"
        )
    if limit_projection not in {"post", "projected"}:
        raise ValueError(
            "limit_projection must be 'post' or 'projected', "
            f"got {limit_projection!r}"
        )
    if redundant_dof_prior not in {"none", "minimal"}:
        raise ValueError(
            "redundant_dof_prior must be 'none' or 'minimal', "
            f"got {redundant_dof_prior!r}"
        )
    if foot_landmark_model not in {"proxy", "fk_offsets"}:
        raise ValueError(
            f"foot_landmark_model must be 'proxy' or 'fk_offsets', got {foot_landmark_model!r}"
        )
    if foot_gate_reference not in {"solved", "observed"}:
        raise ValueError(
            f"foot_gate_reference must be 'solved' or 'observed', got {foot_gate_reference!r}"
        )
    if ankle_twist_limit_deg is not None:
        if (isinstance(ankle_twist_limit_deg, bool)
                or not np.isfinite(ankle_twist_limit_deg)
                or ankle_twist_limit_deg < 0.0):
            raise ValueError(
                "ankle_twist_limit_deg must be None or a finite non-negative float, "
                f"got {ankle_twist_limit_deg!r}"
            )
    effective_target_conditioning, effective_leg_stabilization, effective_foot_lock = (
        _effective_pipeline_options(
            pipeline_profile, target_conditioning, leg_stabilization, foot_lock
        )
    )
    target_conditioning = effective_target_conditioning
    leg_stabilization = effective_leg_stabilization
    foot_lock = effective_foot_lock
    # v8 provenance is mandatory: its opt-in preset always emits the profile
    # document with effective options and implementation hashes.
    profile = bool(profile or pipeline_profile == "v8")
    """WP-A: when ``profile`` is True, collect per-stage wall times and write
    ``<output_bvh stem>.profile.json``. When False, behavior is byte-identical
    to the un-instrumented path (no timing overhead in the hot loop)."""
    timings: dict = {}          # name -> cumulative seconds since start
    _t0 = time.perf_counter() if profile else None

    def _mark(name: str) -> None:
        if profile and _t0 is not None:
            timings[name] = round(time.perf_counter() - _t0, 4)

    print(f"Loading triangulated data from {triangulated_npz}")
    triangulated_data = np.load(triangulated_npz)

    if require_grounded:
        from aimocap.calib.floor import require_grounded_floor
        from aimocap.math.coords import FloorTransform, CAMERA_WORLD_FRAME

        if "floor_transform" in triangulated_data:
            raw_ft = triangulated_data["floor_transform"]
            if isinstance(raw_ft, np.ndarray) and raw_ft.ndim == 0:
                raw_ft = str(raw_ft)
            if isinstance(raw_ft, str):
                ft_dict = json.loads(raw_ft)
            elif isinstance(raw_ft, dict):
                ft_dict = raw_ft
            else:
                ft_dict = {}
            floor_transform = FloorTransform.from_dict(ft_dict)
        else:
            floor_transform = FloorTransform.identity()

        # Gate enforcement: fail-closed if unregistered or unverified
        require_grounded_floor(floor_transform)

        # Select floor-world skeleton
        if "skeleton3d_refined_floor" in triangulated_data:
            print("Using authoritative refined floor-world skeleton (skeleton3d_refined_floor)...")
            pts3d_raw = np.asarray(triangulated_data["skeleton3d_refined_floor"], dtype=np.float64)
        elif "skeleton3d_floor" in triangulated_data:
            print("Using raw floor-world skeleton (skeleton3d_floor)...")
            pts3d_raw = np.asarray(triangulated_data["skeleton3d_floor"], dtype=np.float64)
        else:
            print("Transforming camera-world skeleton to floor-world via validated floor transform...")
            raw_cam = np.asarray(triangulated_data["skeleton3d"], dtype=np.float64)
            pts3d_raw = floor_transform.transform_points(raw_cam, current_frame=CAMERA_WORLD_FRAME)
    else:
        pts3d_raw = np.asarray(triangulated_data["skeleton3d"], dtype=np.float64)
    if pts3d_raw.ndim != 3 or pts3d_raw.shape[-1] != 3:
        raise ValueError(f"retarget input skeleton3d must have shape (F,J,3), got {pts3d_raw.shape}")
    if pts3d_raw.shape[1] not in (133, 17):
        raise ValueError(
            f"retarget input skeleton3d must have 133 joints (COCO-WholeBody) or "
            f"17 joints (legacy body proxy), got shape {pts3d_raw.shape}"
        )
    if not np.isfinite(pts3d_raw).any():
        raise ValueError("retarget input contains no finite 3D points")
    units = str(triangulated_data["units"]) if "units" in triangulated_data else "unknown"
    if units in {"m", "meter", "meters"}:
        pts3d_raw *= 100.0
    elif units in {"cm", "centimeter", "centimeters"}:
        pass
    else:
        raise ValueError(
            f"retarget requires explicit skeleton units ('m' or 'cm'), got {units!r}"
        )

    # Y-up (triangulation) -> Z-up (Unreal/Manny): [x, y, z] -> [x, -z, y]
    pts3d = np.zeros_like(pts3d_raw)
    pts3d[..., 0] = pts3d_raw[..., 0]
    pts3d[..., 1] = -pts3d_raw[..., 2]
    pts3d[..., 2] = pts3d_raw[..., 1]

    if "confidence" in triangulated_data:
        weights = triangulated_data["confidence"]
    else:
        weights = np.any(pts3d != 0, axis=-1).astype(np.float32)

    num_frames = len(pts3d)
    if max_frames is not None and max_frames > 0:
        num_frames = min(num_frames, max_frames)

    mocap_target_pts = extract_mocap_points(pts3d)
    # Small-toe observations are needed only by the opt-in FK landmark model;
    # extraction is harmless for proxy mode and does not enter its residual.
    if pts3d.shape[1] > COCO["small_toe_r"]:
        for side in ("l", "r"):
            mocap_target_pts[f"small_toe_{side}"] = pts3d[:, COCO[f"small_toe_{side}"], :]
    _mark("extract_mocap_points")
    core_targets = (
        "pelvis", "hip_l", "knee_l", "ankle_l",
        "hip_r", "knee_r", "ankle_r", "shoulder_l", "shoulder_r",
    )
    finite_counts = {
        name: int(np.sum(np.isfinite(mocap_target_pts.get(name, np.empty((0, 3))).astype(float)).all(axis=1)))
        for name in core_targets
    }
    # WP2.B fail-closed input gate: refuse structurally broken target tracks
    # instead of solving into NaN roots / crashing downstream stages. A track
    # below min_core_finite_fraction is not a sparse-but-usable observation;
    # it is a collapsed chain (e.g. an untriangulated hip poisoning pelvis).
    invalid_core = {
        name: count / max(num_frames, 1)
        for name, count in finite_counts.items()
        if count / max(num_frames, 1) < min_core_finite_fraction
    }
    if invalid_core:
        detail = ", ".join(
            f"{name}={finite_counts[name]}/{num_frames} finite"
            for name in invalid_core
        )
        raise ValueError(
            "retarget input has unreliable required target tracks "
            f"(finite fraction < {min_core_finite_fraction}): " + detail
        )

    fbx_skel = Skeleton(fbx_rig_path)
    fbx_root_idx, pelvis_idx = _validate_fbx_hierarchy(fbx_skel)

    print("Stage 1: Fitting MocapSkeleton...")
    mocap_skel = MocapSkeleton(pts3d, weights, fbx_skel=fbx_skel)
    mocap_ik = MocapIKSolver(
        mocap_skel, target_sequence=mocap_target_pts,
        foot_landmark_model=foot_landmark_model,
        limb_twist_anchor=limb_twist_anchor,
        two_pass_limbs=two_pass_limbs,
        two_pass_pass1_stride=two_pass_pass1_stride,
        two_pass_pass1_max_iter=two_pass_pass1_max_iter,
        smoothed_limbs=smoothed_limbs,
        limb_twist_pin=limb_twist_pin,
        head_target_mode=head_target_mode,
        foot_landmark_weight=foot_landmark_weight,
        face_position_weight=face_position_weight,
        ankle_twist_limit_deg=ankle_twist_limit_deg,
        twist_limit_mode=twist_limit_mode,
        limit_projection=limit_projection,
        soft_limit_joints=soft_limit_joints,
        redundant_dof_prior=redundant_dof_prior,
        foot_gate_reference=foot_gate_reference,
        limb_hinge=limb_hinge,
        fps=fps,
    )
    gate_refs = mocap_ik.foot_gate_diagnostics()
    print("Foot target gate reference modes:")
    for side in ("l", "r"):
        ref = gate_refs["reference_length_cm"][side]
        ref_text = "direction-only" if ref is None else f"{ref:.2f} cm"
        print(f"  {side}: {gate_refs['reference_mode'][side]} ({ref_text})")

    if target_conditioning not in {"legacy", "robust", "p1", "p2"}:
        raise ValueError(
            f"target_conditioning must be 'legacy', 'robust', 'p1', or 'p2', got {target_conditioning!r}"
        )

    target_diag = {"enabled": False, "mode": target_conditioning}
    if target_conditioning == "legacy":
        # WP2AC conditioning pass (after the fail-closed gate so the gate still
        # sees raw coverage): fill NaN gaps in every measured target track so
        # analytic root init and the temporal warm-start chain cannot be
        # poisoned by a single unobserved frame. Runs before the lower-body
        # stabilizer so its per-segment finite-chunk filter no longer restarts
        # at every gap boundary.
        mocap_target_pts = fill_missing_target_tracks(mocap_target_pts)
        _mark("fill_missing_target_tracks")

        if mocap_target_stabilization:
            print("Stage 1a: Stabilizing lower-body fit targets...")
            mocap_target_pts, target_diag = stabilize_mocap_fit_targets(
                mocap_target_pts,
                fps=fps,
                cutoff_hz=mocap_target_cutoff_hz,
                debug_dir=foot_lock_debug_dir,
            )
            _mark("stabilize_mocap_fit_targets")
            if target_diag.get("enabled"):
                for name in ("ankle_l", "ankle_r"):
                    before = target_diag["before"][name]
                    after = target_diag["after"][name]
                    print(
                        f"  {name}: p95 speed "
                        f"{before['speed_p95_cm_per_sec']:.1f} -> "
                        f"{after['speed_p95_cm_per_sec']:.1f} cm/s"
                    )
            else:
                print(f"  skipped: {target_diag.get('reason', 'unknown reason')}")
    elif target_conditioning == "robust":
        if mocap_target_stabilization:
            print("Stage 1a: Robust observation conditioning...")
            mocap_target_pts, target_diag = condition_target_tracks(
                mocap_target_pts,
                fps=fps,
            )
            target_diag["mode"] = "robust"
            _mark("condition_target_tracks")
            print(
                f"  conditioned {target_diag.get('track_count', 0)} tracks; "
                f"bridged {sum(item.get('bridged_frames', 0) for item in target_diag.get('tracks', {}).values() if isinstance(item, dict))} frames"
            )
        else:
            target_diag = {"enabled": False, "mode": "robust", "reason": "disabled"}
            _mark("fill_missing_target_tracks")
    elif target_conditioning == "p1":
        # P1 is an explicit composition: C1b conditions all solver tracks,
        # while C3h rigid denoising owns only face and hand segments.  The C3h
        # tracks receive the legacy gap fill after rigid replacement; pelvis and
        # neck are derived once, after the two paths are merged.
        print("Stage 1a: P1 composite target conditioning (C1b + C3h)...")
        mocap_target_pts, target_diag = _compose_p1_targets(mocap_target_pts, fps)
        _mark("condition_target_tracks")
        _mark("rigid_segments")
        _mark("fill_missing_target_tracks")
    else:
        print("Stage 1a: P2 composite target conditioning (C3h observations + C1b)...")
        mocap_target_pts, target_diag = _compose_p2_targets(mocap_target_pts, fps)
        _mark("condition_target_tracks")
        _mark("rigid_segments")

    limb_twist_diag = mocap_ik.prepare_limb_twist_anchor(
        mocap_target_pts, fps=fps, frame_count=num_frames
    )
    if limb_twist_anchor == "smoothed":
        print(
            "Limb twist anchor: smoothed offline target "
            f"(lambda={limb_twist_diag['lambda']:.6g}, cutoff={limb_twist_diag['cutoff_hz']:.1f} Hz)"
        )
    elif limb_twist_anchor == "two_pass":
        print(
            "Limb twist anchor: two-pass solved-state target "
            "(pass 1 init pins; pass 2 Butterworth order 2, zero-phase, cutoff=3.0 Hz)"
        )


    if foot_landmark_model == "fk_offsets":
        foot_landmark_diag = mocap_ik.calibrate_foot_landmark_offsets(
            mocap_target_pts,
            confidence=weights,
            frame_count=num_frames,
            stride=5,
            iterations=2,
        )
        _mark("foot_landmark_calibration")
        print(
            "Foot landmark model: "
            f"requested=fk_offsets effective={foot_landmark_diag.get('effective_model')} "
            f"status={foot_landmark_diag.get('status')}"
        )
    else:
        foot_landmark_diag = mocap_ik.foot_landmark_diagnostics()

    solved_root = []
    solved_local_quats = []
    frame_ms: list[float] = []
    solve_stats: list[dict] = []

    def _solve_pass(pass_number: int, seed_states: list[np.ndarray] | None = None,
                    frame_indices: np.ndarray | None = None):
        if limb_twist_anchor == "two_pass":
            mocap_ik.begin_two_pass(pass_number)
            if pass_number == 2:
                # Report limit occupancy on the final animation frames, not
                # twice-counted pass-1 and pass-2 solver calls.
                mocap_ik.reset_twist_limit_diagnostics()
        pass_root = []
        pass_quats = []
        pass_states = []
        prev_x = None
        prev_prev_x = None
        pass_frame_ms: list[float] = []
        frames = (np.arange(num_frames, dtype=np.int64)
                  if frame_indices is None else np.asarray(frame_indices, dtype=np.int64))
        # A strided sample compares each solve to the previous sampled frame.
        # Reduce temporal/acceleration penalties per sampled interval so the
        # longer jump is not over-constrained; stride=1 uses the old literals.
        stride_scale = (1.0 if pass_number != 1 or len(frames) < 2
                        else float(max(1, int(frames[1] - frames[0]))))
        temporal_weight = 0.03 if stride_scale == 1.0 else 0.03 / stride_scale
        accel_weight = 0.3 if stride_scale == 1.0 else 0.3 / (stride_scale * stride_scale)
        for f in frames:
            f = int(f)
            if f % 10 == 0:
                suffix = f" (pass {pass_number})" if limb_twist_anchor == "two_pass" else ""
                print(f"Solving Mocap Frame {f}/{num_frames}{suffix}")
            _fs = time.perf_counter() if profile else None
            measured = {k: v[f] for k, v in mocap_target_pts.items()}
            x_seed = seed_states[f] if seed_states is not None else None
            x_opt = mocap_ik.solve_frame(
                measured,
                prev_x=prev_x,
                temporal_weight=temporal_weight,
                frame_idx=f,
                prev_prev_x=prev_prev_x,
                accel_weight=accel_weight,
                x0=x_seed,
                max_iter=(two_pass_pass1_max_iter
                          if pass_number == 1 else None),
            )
            prev_prev_x = prev_x.copy() if prev_x is not None else None
            # D2 warm-start safety net: a non-finite solve result must never
            # enter the temporal chain. This is zero-effect on healthy frames.
            if not np.isfinite(x_opt).all():
                fallback = prev_x if (prev_x is not None and np.isfinite(prev_x).all()) else None
                if fallback is None and x_seed is not None and np.isfinite(x_seed).all():
                    fallback = x_seed
                if fallback is None:
                    try:
                        fallback = mocap_ik.analytic_init(
                            {k: v[f] for k, v in mocap_target_pts.items()})
                    except Exception:
                        fallback = None
                if fallback is not None and np.isfinite(fallback).all():
                    print(f"WARNING: frame {f} pass {pass_number} solve returned non-finite state; "
                          "using fallback (temporal/seed/analytic)", flush=True)
                    x_opt = fallback
                else:
                    raise ValueError(
                        f"frame {f}: solve returned non-finite state and no "
                        "finite fallback is available (input tracks are structurally broken)"
                    )
            solve_stat = dict(getattr(mocap_ik, "_last_solve_stats", {}))
            if limb_twist_anchor == "two_pass":
                solve_stat["pass"] = int(pass_number)
            solve_stats.append(solve_stat)
            pass_states.append(np.asarray(x_opt, dtype=np.float64).copy())
            prev_x = x_opt
            root_t, local_quats = mocap_ik._state_to_local_rotations(x_opt)
            pass_root.append(root_t)
            pass_quats.append(local_quats)
            if profile and _fs is not None:
                elapsed_ms = (time.perf_counter() - _fs) * 1000.0
                frame_ms.append(elapsed_ms)
                pass_frame_ms.append(elapsed_ms)
        return pass_states, pass_root, pass_quats, pass_frame_ms, frames

    pass1_started = time.perf_counter()
    pass1_stride = (int(two_pass_pass1_stride)
                    if limb_twist_anchor == "two_pass" else 1)
    pass1_indices = np.arange(0, num_frames, pass1_stride, dtype=np.int64)
    if pass1_indices[-1] != num_frames - 1:
        pass1_indices = np.r_[pass1_indices, num_frames - 1]
    pass1_states, pass1_root, pass1_quats, pass1_frame_ms, pass1_solved_indices = _solve_pass(
        1, frame_indices=pass1_indices
    )
    pass1_s = time.perf_counter() - pass1_started
    if limb_twist_anchor == "two_pass":
        limb_twist_diag = mocap_ik.prepare_two_pass_anchor(
            pass1_states, fps=fps, frame_count=num_frames,
            pass1_frame_indices=pass1_solved_indices,
        )
        # Pass 2 still visits every frame.  Use a deterministic warm-start
        # interpolation between sampled local rotations rather than solving
        # skipped pass-1 frames just to manufacture seeds.
        if len(pass1_solved_indices) == num_frames:
            pass1_seed_states = [np.asarray(state, dtype=np.float64).copy()
                                 for state in pass1_states]
        else:
            from scipy.spatial.transform import Slerp
            sampled_states = np.asarray(pass1_states, dtype=np.float64)
            pass1_seed_array = np.empty((num_frames, mocap_ik.num_vars), dtype=np.float64)
            pass1_seed_array[:, :3] = np.column_stack([
                np.interp(np.arange(num_frames), pass1_solved_indices, sampled_states[:, axis])
                for axis in range(3)
            ])
            sampled_rv = sampled_states[:, 3:].reshape(len(sampled_states), mocap_ik.num_joints, 3)
            seed_rv = np.empty((num_frames, mocap_ik.num_joints, 3), dtype=np.float64)
            for joint in range(mocap_ik.num_joints):
                if len(sampled_states) == 1:
                    seed_rv[:, joint] = sampled_rv[0, joint]
                else:
                    seed_rv[:, joint] = Slerp(
                        pass1_solved_indices.astype(np.float64),
                        Rotation.from_rotvec(sampled_rv[:, joint]),
                    )(np.arange(num_frames, dtype=np.float64)).as_rotvec()
            pass1_seed_array[:, 3:] = seed_rv.reshape(num_frames, -1)
            pass1_seed_states = list(pass1_seed_array)
        pass2_started = time.perf_counter()
        pass2_states, solved_root, solved_local_quats, pass2_frame_ms, _ = _solve_pass(
            2, seed_states=pass1_seed_states
        )
        pass2_s = time.perf_counter() - pass2_started
        mocap_ik.set_two_pass_timings(pass1_s, pass2_s)
        limb_twist_diag = mocap_ik.two_pass_anchor_diagnostics()
    else:
        solved_root = pass1_root
        solved_local_quats = pass1_quats
    _mark("solve_loop")

    ankle_twist_diag = {}
    if profile:
        from aimocap.retarget.swing_twist import signed_twist_angle
        for side in ("l", "r"):
            name = f"ankle_{side}"
            ji = mocap_ik._twist_joint_indices.get(name)
            axis = mocap_ik._twist_bone_axes.get(name)
            limit = mocap_ik._twist_limits.get(name)
            if ji is None or axis is None or limit is None or not solved_local_quats:
                continue
            values = np.asarray([
                np.degrees(abs(signed_twist_angle(Rotation.from_quat(local_q[ji]), axis)))
                for local_q in solved_local_quats
            ], dtype=np.float64)
            bound = float(limit)
            ankle_twist_diag[side] = {
                "limit_deg": bound,
                "p99_abs_deg": float(np.percentile(values, 99)),
                "max_abs_deg": float(np.max(values)),
                "bound_tolerance_deg": 1e-6,
                "bound_fraction": float(np.mean(values >= bound - 1e-6)),
            }

    if arm_orientation_rewrite:
        print("Stage 1b: Arm orientation rewrite...")
        from aimocap.retarget.arm_orient import (
            solve_upperarm_global,
            solve_hand_global_v2,
            resolve_bone_axis_parent_local,
            validate_arm_observation,
            rotation_angle_deg,
            Observation,
        )

        proxy_rest_global = [Rotation.identity()] * mocap_ik.num_joints

        arm_meta = {}
        for side in ("l", "r"):
            clav_name  = f"clavicle_{side}"
            upper_name = f"shoulder_{side}"    # proxy name for upper arm
            fore_name  = f"elbow_{side}"       # proxy name for forearm pivot
            hand_name  = f"wrist_{side}"       # proxy name for hand pivot
            
            names = [clav_name, upper_name, fore_name, hand_name]
            if not all(n in mocap_skel.name_to_idx for n in names):
                arm_meta[side] = None
                continue
            
            clav_i  = mocap_skel.name_to_idx[clav_name]
            upper_i = mocap_skel.name_to_idx[upper_name]
            fore_i  = mocap_skel.name_to_idx[fore_name]
            hand_i  = mocap_skel.name_to_idx[hand_name]
            
            rest_t = mocap_skel.rest_t  # shape (J, 3)
            
            e_upper_parent = resolve_bone_axis_parent_local(
                rest_t, proxy_rest_global[clav_i], upper_i, fore_i
            )
            e_fore_upper_local = resolve_bone_axis_parent_local(
                rest_t, proxy_rest_global[upper_i], fore_i, hand_i
            )
            
            arm_meta[side] = {
                "clav_i":  clav_i,
                "upper_i": upper_i,
                "fore_i":  fore_i,
                "hand_i":  hand_i,
                "e_upper_parent":     e_upper_parent,
                "e_fore_upper_local": e_fore_upper_local,
                "G_parent_rest": proxy_rest_global[clav_i],
                "G_rest_upper":  proxy_rest_global[upper_i],
                "G_rest_fore":   proxy_rest_global[fore_i],
                "G_rest_hand":   proxy_rest_global[hand_i],
            }
    else:
        print("Stage 1b: Arm orientation rewrite disabled (IK-only production mode).")
        arm_meta = {}

    def _globals_to_locals(mocap_ik, old_local_quats, replacements, solved_root):
        before_pos, before_global_q = mocap_ik.forward_kinematics(
            solved_root, old_local_quats
        )
        global_rots = list(Rotation.from_quat(before_global_q))
        
        for idx, G_new in replacements.items():
            global_rots[idx] = G_new
        
        new_local_quats = np.empty_like(old_local_quats)
        for idx in range(mocap_ik.num_joints):
            p = int(mocap_ik.parents[idx])
            if p < 0:
                L = global_rots[idx]
            else:
                L = global_rots[p].inv() * global_rots[idx]
            new_local_quats[idx] = L.as_quat()
        
        after_pos, after_global_q = mocap_ik.forward_kinematics(
            solved_root, new_local_quats
        )
        position_delta = np.linalg.norm(after_pos - before_pos, axis=1)
        assert np.max(position_delta) < 1e-9, (
            f"Arm rewrite moved proxy joint by {np.max(position_delta):.3e} cm"
        )
        replaced = set(replacements)
        for idx in range(mocap_ik.num_joints):
            if idx in replaced:
                continue
            assert rotation_angle_deg(
                Rotation.from_quat(before_global_q[idx]),
                Rotation.from_quat(after_global_q[idx]),
            ) < 1e-8, f"Non-arm global rotation changed at joint {idx}"
        return new_local_quats

    arm_state = {
        side: {"prev_upper_dir": None, "prev_fore_dir": None,
               "prev_phi": 0.0, "invalid_run": 0}
        for side in ("l", "r")
    }
    arm_source_histogram = {}
    arm_phi_upper = {"l": [], "r": []}
    arm_phi_fore  = {"l": [], "r": []}

    for f in range(num_frames):
        old_local_quats = solved_local_quats[f]
        
        _, global_q = mocap_ik.forward_kinematics(solved_root[f], old_local_quats)
        global_rots = list(Rotation.from_quat(global_q))
        
        replacements = {}
        
        for side in ("l", "r"):
            meta = arm_meta.get(side)
            if meta is None:
                continue
            
            clav_i  = meta["clav_i"]
            upper_i = meta["upper_i"]
            fore_i  = meta["fore_i"]
            hand_i  = meta["hand_i"]
            
            raw_sh = mocap_target_pts[f"shoulder_{side}"][f]
            raw_el = mocap_target_pts[f"elbow_{side}"][f]
            raw_wr = mocap_target_pts[f"wrist_{side}"][f]
            arm_obs, obs_gate = validate_arm_observation(
                raw_sh, raw_el, raw_wr, arm_state[side]
            )
            if arm_obs is None:
                arm_source_histogram[obs_gate] = arm_source_histogram.get(obs_gate, 0) + 1
                continue
            sh_w, el_w, wr_w = arm_obs
            
            D_parent = global_rots[clav_i] * meta["G_parent_rest"].inv()
            
            G_upper, obs_upper = solve_upperarm_global(
                D_parent=D_parent,
                G_parent_rest=meta["G_parent_rest"],
                e_upper_parent=meta["e_upper_parent"],
                e_fore_upper_local=meta["e_fore_upper_local"],
                G_rest_upper=meta["G_rest_upper"],
                sh_w=sh_w, el_w=el_w, wr_w=wr_w,
                phi_prev=arm_state[side]["prev_phi"],
                G_upper_ik=global_rots[upper_i],
            )
            
            # Preserve the solved forearm global rotation.  Recomputing it
            # from every raw wrist direction moves the proxy wrist whenever
            # triangulation noise differs from the IK pose.  The hand still
            # inherits the forearm delta below, but receives no second
            # independent orientation correction.
            G_fore = global_rots[fore_i]
            obs_fore = Observation(
                0.0, True, 1.0, "ik_only", "preserve_ik_forearm"
            )
            
            G_hand = solve_hand_global_v2(
                G_fore_w=G_fore,
                G_rest_fore=meta["G_rest_fore"],
                G_rest_hand=meta["G_rest_hand"],
            )
            
            replacements[upper_i] = G_upper
            replacements[fore_i]  = G_fore
            replacements[hand_i]  = G_hand
            
            arm_state[side]["prev_phi"] = obs_upper.value
            src = obs_gate if obs_gate == "held" else obs_upper.source
            arm_source_histogram[src] = arm_source_histogram.get(src, 0) + 1
            arm_phi_upper[side].append(obs_upper.value)
            arm_phi_fore[side].append(obs_fore.value)
        
        if not replacements:
            continue
        
        new_local_quats = _globals_to_locals(
            mocap_ik, old_local_quats, replacements, solved_root[f]
        )
        
        solved_local_quats[f] = new_local_quats

    print(f"  Arm source histogram: {arm_source_histogram}")
    for side in ("l", "r"):
        phis = np.array(arm_phi_upper[side])
        if len(phis) > 0:
            print(
                f"  Upper {side}: p05={np.percentile(np.degrees(phis), 5):.1f}° "
                f"p50={np.percentile(np.degrees(phis), 50):.1f}° "
                f"p95={np.percentile(np.degrees(phis), 95):.1f}°"
            )

    print("Stage 2: Rotation Transfer to FBX...")
    _mark("pre_transfer")
    fbx_rest_global_pos, fbx_rest_global_rot = fbx_skel.get_forward_kinematics()
    fbx_rest_global = Rotation.from_quat(fbx_rest_global_rot)
    fbx_rest_local = Rotation.from_quat(np.array(fbx_skel.rest_rotations))

    mocap_to_fbx_idx = {
        mocap_i: fbx_skel.name_to_idx[name]
        for mocap_i, name in mocap_skel.fbx_mapping.items()
    }
    fbx_to_mocap_idx = {
        fbx_i: mocap_i for mocap_i, fbx_i in mocap_to_fbx_idx.items()
    }
    # Hip/knee roll is underdetermined from sparse 3D joints, but handled in IK now.
    
    fbx_animation = np.zeros((num_frames, 3 + fbx_skel.num_joints * 3))
    previous_frame_local = None
    rotation_diag_indices = {
        name: fbx_skel.name_to_idx.get(name)
        for name in ("head", "neck_01", "pelvis")
    }

    for f in range(num_frames):
        mocap_pos, mocap_global_quats = mocap_ik.forward_kinematics(
            solved_root[f],
            solved_local_quats[f],
        )
        mocap_global = Rotation.from_quat(mocap_global_quats)

        frame_global = [None] * fbx_skel.num_joints
        frame_local = [None] * fbx_skel.num_joints

        for fbx_i in range(fbx_skel.num_joints):
            p = fbx_skel.parents[fbx_i]

            if fbx_i in fbx_to_mocap_idx:
                mocap_i = fbx_to_mocap_idx[fbx_i]
                # Proxy global rotation is a world-space delta from rest.
                R_global_target = mocap_global[mocap_i] * fbx_rest_global[fbx_i]
            elif p == -1:
                R_global_target = fbx_rest_global[fbx_i]
            else:
                # Preserve unmapped bones in rest local pose under the animated
                # parent, so twist/finger/intermediate bones inherit correctly.
                R_global_target = frame_global[p] * fbx_rest_local[fbx_i]

            if p == -1:
                R_local = R_global_target
            else:
                R_local = frame_global[p].inv() * R_global_target

            frame_global[fbx_i] = R_global_target
            frame_local[fbx_i] = R_local

        # Align Manny's pelvis position to the solved proxy pelvis. BVH writes
        # root translation relative to the FBX root rest translation.
        root_world = (
            solved_root[f]
            - frame_global[fbx_root_idx].apply(fbx_skel.rest_translations[pelvis_idx])
        )
        fbx_animation[f, 0:3] = root_world - fbx_skel.rest_translations[fbx_root_idx]

        frame_local_quats = np.array([r.as_quat() for r in frame_local])
        if previous_frame_local is not None and f % 30 == 0:
            dots = np.sum(previous_frame_local * frame_local_quats, axis=1)
            delta_deg = np.degrees(2.0 * np.arccos(np.clip(np.abs(dots), -1.0, 1.0)))
            values = {
                name: (delta_deg[idx] if idx is not None else float("nan"))
                for name, idx in rotation_diag_indices.items()
            }
            print(
                "[rotation-diagnostic] "
                f"frame={f} head_delta={values['head']:.2f}deg "
                f"neck_delta={values['neck_01']:.2f}deg "
                f"pelvis_delta={values['pelvis']:.2f}deg "
                f"max_delta={delta_deg.max():.2f}deg"
            )
        previous_frame_local = frame_local_quats
        eulers = Rotation.from_quat(frame_local_quats).as_euler("xyz", degrees=False)
        fbx_animation[f, 3:] = eulers.flatten()

    leg_diag = {}
    if leg_stabilization:
        print("Stage 3: Target-space lower-body stabilization...")
        _mark("transfer_loop")
        fbx_animation, leg_diag = stabilize_lower_body_animation(
            fbx_skel,
            fbx_animation,
            fps=fps,
            cutoff_hz=leg_stabilize_cutoff_hz,
            max_deg_per_frame=leg_max_deg_per_frame,
            debug_dir=foot_lock_debug_dir,
        )
        if leg_diag.get("enabled"):
            cap_report = leg_diag.get("velocity_cap", {})
            print(f"  velocity cap: {cap_report.get('total_exceedances', 0)} input deltas > "
                  f"{leg_max_deg_per_frame:.1f} deg/frame")
            for name in ("calf_l", "foot_l", "calf_r", "foot_r"):
                if name not in leg_diag["before"]:
                    continue
                before = leg_diag["before"][name]
                after = leg_diag["after"][name]
                print(
                    f"  {name}: p95 rot delta "
                    f"{before['p95_deg_per_frame']:.2f} -> "
                    f"{after['p95_deg_per_frame']:.2f} deg/frame"
                )
        else:
            print(f"  skipped: {leg_diag.get('reason', 'unknown reason')}")

    foot_diag = {}
    if foot_lock:
        print("Stage 4: Target-space foot locking...")
        _mark("leg_stabilization")
        fbx_animation, foot_diag = apply_foot_lock(
            fbx_skel,
            fbx_animation,
            pts3d[:num_frames],
            fps=fps,
            debug_dir=foot_lock_debug_dir,
            pose2d_npz=pose2d_npz,
            anchor_policy=foot_lock_anchor_policy,
        )
        if foot_diag.get("enabled"):
            for side in ("left", "right"):
                before = foot_diag["before"][side]
                after = foot_diag["after"][side]
                print(
                    f"  {side}: foot drift {before['foot_drift_cm']:.2f} -> "
                    f"{after['foot_drift_cm']:.2f} cm, ball drift "
                    f"{before['ball_drift_cm']:.2f} -> {after['ball_drift_cm']:.2f} cm"
                )
        else:
            print(f"  skipped: {foot_diag.get('reason', 'unknown reason')}")

    npy_out = str(output_bvh).replace(".bvh", "_solved.npy")
    print(f"Saving backup to {npy_out}")
    _mark("foot_lock")
    np.save(npy_out, fbx_animation)

    print(f"Saving BVH to {output_bvh}")
    write_bvh(output_bvh, fbx_skel, fbx_animation, fps=fps)
    _mark("bvh_save")
    
    # FBX export: direct fcurve injection (no COPY_TRANSFORMS, no rest-folding)
    fbx_out = str(output_bvh).replace(".bvh", ".fbx")
    if export_fbx:
        try:
            from aimocap.retarget.fbx_export import write_fbx
            print(f"Exporting FBX to {fbx_out}...")
            write_fbx(
                npy_path=npy_out,
                fbx_in_path=str(fbx_rig_path),
                fbx_out_path=fbx_out,
                fps=fps,
                bone_names=fbx_skel.node_names,
            )
        except Exception as exc:
            print(f"WARNING: FBX export skipped ({exc})")


    print("Done.")

    if profile:
        # Derive per-block durations from the cumulative offsets (dict order
        # is insertion order, which is execution order here).
        names = list(timings)
        durations = {}
        prev = 0.0
        for name in names:
            durations[name] = round(timings[name] - prev, 4)
            prev = timings[name]
        frame_stats = {}
        if frame_ms:
            fm = np.asarray(frame_ms, dtype=np.float64)
            frame_stats = {
                "n_frames_timed": int(fm.size),
                "p50_ms": float(np.percentile(fm, 50)),
                "p95_ms": float(np.percentile(fm, 95)),
                "max_ms": float(fm.max()),
                "total_s": float(fm.sum() / 1000.0),
            }
        profile_doc = {
            "pipeline_profile": pipeline_profile,
            "effective_options": {
                "pipeline_profile": pipeline_profile,
                "foot_landmark_model": foot_landmark_model,
                "foot_landmark_weight": float(foot_landmark_weight),
                "head_target_mode": head_target_mode,
                "face_position_weight": float(face_position_weight),
                "limb_twist_anchor": limb_twist_anchor,
                "two_pass_limbs": two_pass_limbs,
                "two_pass_pass1_stride": int(two_pass_pass1_stride),
                "two_pass_pass1_max_iter": (int(two_pass_pass1_max_iter)
                                              if two_pass_pass1_max_iter is not None else None),
                "smoothed_limbs": smoothed_limbs,
                "limb_twist_pin": limb_twist_pin,
                "twist_limit_mode": twist_limit_mode,
                "soft_limit_joints": sorted(mocap_ik.soft_limit_joints),
                "twist_safety_margin_deg": 10.0,
                "ankle_twist_limit_deg": ankle_twist_limit_deg,
                "redundant_dof_prior": redundant_dof_prior,
                "limb_hinge": limb_hinge,
                "foot_landmark_effective_model": mocap_ik.foot_landmark_model,
                "target_conditioning": target_conditioning,
                "mocap_target_stabilization": bool(mocap_target_stabilization),
                "mocap_target_cutoff_hz": float(mocap_target_cutoff_hz),
                "leg_stabilization": bool(leg_stabilization),
                "leg_stabilize_cutoff_hz": float(leg_stabilize_cutoff_hz),
                "leg_max_deg_per_frame": float(leg_max_deg_per_frame),
                "foot_lock": bool(foot_lock),
                "foot_lock_anchor_policy": foot_lock_anchor_policy,
                "arm_orientation_rewrite": bool(arm_orientation_rewrite),
                "export_fbx": bool(export_fbx),
                "require_grounded": bool(require_grounded),
                "fps": float(fps),
                "max_frames": max_frames,
            },
            "module_hashes": {
                "engine": _module_sha256(__file__),
                "rigid_segments": _module_sha256(str(Path(__file__).with_name("rigid_segments.py"))),
                "observation_conditioning": _module_sha256(str(Path(__file__).with_name("observation_conditioning.py"))),
                "temporal": _module_sha256(str(Path(__file__).with_name("temporal.py"))),
                "foot_lock": _module_sha256(str(Path(__file__).with_name("foot_lock.py"))),
            },
            "stage_wall_s_cumulative": timings,
            "stage_wall_s_per_block": durations,
            "solve_frame_ms": frame_stats,
            "solve_stats": solve_stats,
            "twist_limit_diagnostics": mocap_ik.twist_limit_diagnostics(),
            "ankle_twist_diagnostics": ankle_twist_diag,
            "foot_target_gate": mocap_ik.foot_gate_diagnostics(),
            "limb_twist_anchor": (
                limb_twist_diag if limb_twist_anchor == "two_pass"
                else mocap_ik.limb_twist_anchor_diagnostics()
            ),
            "limb_twist_pin": mocap_ik.limb_twist_pin_diagnostics(),
            "foot_landmark_model": mocap_ik.foot_landmark_diagnostics(),
            "target_conditioning": target_diag,
            "stage3_leg_stabilization": leg_diag,
            "stage4_foot_lock": foot_diag,
            "total_s": round(time.perf_counter() - _t0, 4),
        }
        profile_path = str(output_bvh).rsplit(".bvh", 1)[0] + ".profile.json"
        with open(profile_path, "w", encoding="utf-8") as fh:
            json.dump(profile_doc, fh, indent=1)
        print(f"Saved timing profile to {profile_path}")


retarget_sequence = retarget_to_fbx
