"""Deterministic ground-truth clips, calibrated observations, and retarget metrics.

All internal FK/keypoint values are Manny's Z-up centimetres.  Engine input is
written as raw Y-up metres using ``[X, Z, -Y] / 100``.  The anatomy offsets in
this module are deliberately harness constants; they are not solver constants.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from aimocap.data.wholebody_layout import COCO_ANCHORS, N_KEYPOINTS
from aimocap.retarget.fbx_rig import Skeleton

FPS = 30.0
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RIG = ROOT / "Manny.FBX"

# Manny FK joint-head mapping used for the body and the two COCO hand wrists.
BODY_BONES = {
    0: "head", 5: "upperarm_l", 6: "upperarm_r", 7: "lowerarm_l",
    8: "lowerarm_r", 9: "hand_l", 10: "hand_r", 11: "thigh_l",
    12: "thigh_r", 13: "calf_l", 14: "calf_r", 15: "foot_l", 16: "foot_r",
}
HAND_MCP = {
    91: "hand_l", 92: "thumb_01_l", 96: "index_metacarpal_l",
    100: "middle_metacarpal_l", 104: "ring_metacarpal_l", 108: "pinky_metacarpal_l",
    112: "hand_r", 113: "thumb_01_r", 117: "index_metacarpal_r",
    121: "middle_metacarpal_r", 125: "ring_metacarpal_r", 129: "pinky_metacarpal_r",
}
REGION_INDICES = {
    "body": [5, 6, 7, 8, 11, 12, 13, 14], "head_face": list(range(0, 5)) + list(range(23, 91)),
    "wrists": [9, 10, 91, 112], "feet": list(range(17, 23)),
    "hand_mcp": [91, 96, 100, 104, 108, 112, 117, 121, 125, 129],
}
# Public region contract.  In particular, hands are the hand bones, not the
# forearm bones; attribution and tests consume this mapping directly.
REGION_BONES = {
    "head_neck": ["neck_01", "neck_02", "head"],
    "spine_torso": ["spine_01", "spine_02", "spine_03", "spine_04", "spine_05"],
    "pelvis_root": ["root", "pelvis"], "upper_arms": ["upperarm_l", "upperarm_r"],
    "forearms": ["lowerarm_l", "lowerarm_r"], "hands": ["hand_l", "hand_r"],
    "thighs": ["thigh_l", "thigh_r"], "calves": ["calf_l", "calf_r"],
    "feet": ["foot_l", "foot_r"], "balls": ["ball_l", "ball_r"],
}
SOLVER_DRIVEN_BONES = ["pelvis", *[f"spine_{i:02d}" for i in range(1, 6)],
    "neck_01", "neck_02", "head", "clavicle_l", "clavicle_r",
    "upperarm_l", "upperarm_r", "lowerarm_l", "lowerarm_r", "hand_l", "hand_r",
    "thigh_l", "thigh_r", "calf_l", "calf_r", "foot_l", "foot_r", "ball_l", "ball_r"]
TWIST_BONES = ["foot_l", "foot_r", "ball_l", "ball_r", "hand_l", "hand_r",
               "lowerarm_l", "lowerarm_r", "head"]

# Anatomical offsets in each bone's local frame, centimetres.  Manny's local
# forward direction is approximately -Y in the rest pose.  Face landmarks are
# intentionally simple, rigid offsets: they provide stable pose diagnostics,
# not a claim that a face mesh is present in Manny.FBX.
FACE_OFFSETS = {
    0: (0.0, -10.0, 5.0), 1: (-3.2, -9.2, 6.0), 2: (3.2, -9.2, 6.0),
    3: (-7.2, -5.0, 3.0), 4: (7.2, -5.0, 3.0),
}


def _rig(rig: Skeleton | str | Path | None) -> Skeleton:
    return rig if isinstance(rig, Skeleton) else Skeleton(str(rig or DEFAULT_RIG))


def animation_fk(animation: np.ndarray, rig: Skeleton | str | Path | None = None) -> dict[str, np.ndarray]:
    """FK a solved-npy animation into world positions/quaternions."""
    rig = _rig(rig)
    a = np.asarray(animation, dtype=float)
    expected = 3 + 3 * rig.num_joints
    if a.ndim != 2 or a.shape[1] != expected:
        raise ValueError(f"animation must have shape (F,{expected}), got {a.shape}")
    e = a[:, 3:].reshape(len(a), rig.num_joints, 3)
    local_q = Rotation.from_euler("xyz", e.reshape(-1, 3)).as_quat().reshape(
        len(a), rig.num_joints, 4)
    positions = np.empty((len(a), rig.num_joints, 3), dtype=float)
    world_q = np.empty_like(local_q)
    for f in range(len(a)):
        positions[f], world_q[f] = rig.get_forward_kinematics(
            local_q[f], root_translation=a[f, :3])
    return {"positions": positions, "world_q": world_q, "local_q": local_q,
            "root": a[:, :3].copy()}


def _apply_q(q: np.ndarray, v: Iterable[float]) -> np.ndarray:
    return Rotation.from_quat(q).apply(np.asarray(v, dtype=float))


def _face_offsets() -> np.ndarray:
    """Return 68 deterministic local head-frame offsets in centimetres."""
    out = np.zeros((68, 3), dtype=float)
    # contour: a flattened oval, then eyebrows, eyes, nose and lips.
    contour = np.linspace(-np.pi * 0.88, np.pi * 0.88, 17)
    out[:17, 0] = 10.0 * np.sin(contour)
    out[:17, 1] = -7.0 - 3.0 * np.cos(contour)
    out[:17, 2] = 5.0 * np.cos(contour) - 1.0
    for start, xs, z in ((17, np.linspace(-6, -1, 5), 7.0), (22, np.linspace(1, 6, 5), 7.0)):
        out[start:start + 5, 0], out[start:start + 5, 1], out[start:start + 5, 2] = xs, -9.0, z
    for start, x in ((27, -3.2), (33, 3.2)):
        ang = np.linspace(0, 2 * np.pi, 6, endpoint=False)
        out[start:start + 6] = np.column_stack((x + 1.8 * np.cos(ang),
                                                   -10.0 + 1.1 * np.sin(ang),
                                                   5.8 + 1.5 * np.sin(ang)))
    out[39:48] = np.array([[0, -10.3, 6.0], [-1.3, -10.0, 5.2], [-2.0, -9.7, 4.0],
                           [-1.2, -10.1, 3.1], [0, -10.5, 2.7], [1.2, -10.1, 3.1],
                           [2.0, -9.7, 4.0], [1.3, -10.0, 5.2], [0, -11.0, 4.4]])
    ang = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    out[48:60] = np.column_stack((4.0 * np.cos(ang), -10.4 + .8 * np.sin(ang),
                                   1.0 + 2.0 * np.sin(ang)))
    out[60:68] = np.column_stack((2.1 * np.cos(ang[:8]), -10.8 + .5 * np.sin(ang[:8]),
                                   1.0 + .9 * np.sin(ang[:8])))
    return out


def keypoints_from_animation(animation: np.ndarray, rig: Skeleton | str | Path | None = None,
                             systematic_bias: np.ndarray | None = None) -> np.ndarray:
    """Create COCO-WholeBody 133-point keypoints from Manny FK.

    Body anchors are bone heads.  Toes use the ball/foot frames; heels are
    behind and below the foot joint.  Face landmarks use the documented rigid
    head-frame offsets above; hand MCPs use Manny metacarpal heads.
    """
    rig = _rig(rig)
    fk = animation_fk(animation, rig)
    p, q = fk["positions"], fk["world_q"]
    out = np.full((len(p), N_KEYPOINTS, 3), np.nan, dtype=float)
    ids = rig.name_to_idx
    for k, name in BODY_BONES.items():
        out[:, k] = p[:, ids[name]]
    # Body face anchors.
    for k, off in FACE_OFFSETS.items():
        out[:, k] = p[:, ids["head"]] + np.array([_apply_q(q[f, ids["head"]], off)
                                                     for f in range(len(p))])
    # Remaining face landmarks.
    face = _face_offsets()
    for j in range(68):
        out[:, 23 + j] = p[:, ids["head"]] + np.array([
            _apply_q(q[f, ids["head"]], face[j]) for f in range(len(p))])
    for side, foot, ball, ankle, sign in (("l", "foot_l", "ball_l", "foot_l", 1.0),
                                           ("r", "foot_r", "ball_r", "foot_r", -1.0)):
        fi, bi = ids[foot], ids[ball]
        # Forward is the ball's local -Y direction in Manny; the lateral
        # offsets make big/small toes distinct without using mesh vertices.
        for f in range(len(p)):
            out[f, COCO_ANCHORS[f"big_toe_{side}"]] = p[f, bi] + _apply_q(q[f, bi], (0, -5.0, 0))
            out[f, COCO_ANCHORS[f"small_toe_{side}"]] = p[f, bi] + _apply_q(q[f, bi], (sign * 4.2, -4.0, 0))
            out[f, COCO_ANCHORS[f"heel_{side}"]] = p[f, fi] + _apply_q(q[f, fi], (0, 4.5, -3.0))
    for k, name in HAND_MCP.items():
        out[:, k] = p[:, ids[name]]
    # Fill the non-MCP hand joints from stable hand-frame anatomy points.  This
    # keeps the complete 133-point contract usable while MCPs remain exact.
    for side, start in (("l", 91), ("r", 112)):
        hi = ids[f"hand_{side}"]
        for j in range(21):
            k = start + j
            if np.isfinite(out[:, k]).all():
                continue
            finger = (j - 1) // 4
            level = (j - 1) % 4
            x = (finger - 2) * 2.6
            if finger == 0:
                x = -4.0 if side == "l" else 4.0
            out[:, k] = p[:, hi] + np.array([
                _apply_q(q[f, hi], (x, -2.0 - level * 3.0, 1.2 * (finger % 2)))
                for f in range(len(p))])
    if systematic_bias is not None:
        b = np.asarray(systematic_bias, dtype=float)
        if b.shape == (3,): out += b
        elif b.shape == out.shape: out += b
        else: raise ValueError("systematic_bias must be (3,) or (F,133,3)")
    return out


def _min_jerk(t: np.ndarray) -> np.ndarray:
    return 10 * t**3 - 15 * t**4 + 6 * t**5


def _pose_targets() -> dict[str, np.ndarray]:
    return {"root": np.zeros(3), **{n: np.zeros(3) for n in (
        "pelvis", "spine_01", "spine_02", "spine_03", "spine_04", "spine_05",
        "neck_01", "head", "upperarm_l", "upperarm_r", "lowerarm_l", "lowerarm_r",
        "hand_l", "hand_r", "thigh_l", "thigh_r", "calf_l", "calf_r", "foot_l", "foot_r",
        "ball_l", "ball_r")}}


def synthesize_diagnostic(rig: Skeleton | str | Path | None = None, fps: float = FPS,
                         seed: int = 1729) -> tuple[np.ndarray, dict[str, Any]]:
    """Build a deterministic ~34 s full-body min-jerk diagnostic animation."""
    rig = _rig(rig)
    del seed  # reserved so the public sequence remains reproducible by contract.
    rest_e = Rotation.from_quat(np.asarray(rig.rest_rotations)).as_euler("xyz")
    pose = _pose_targets()
    frames: list[np.ndarray] = []
    labels: list[dict[str, Any]] = []

    def add(name: str, n: int, changes: dict[str, Iterable[float]], speed: str):
        nonlocal pose
        # Keep the diagnostic in the requested 25--35 second window while
        # retaining enough frames for slow/fast transitions and reversals.
        n = max(8, int(round(n * 0.86)))
        start = len(frames)
        target = {k: v.copy() for k, v in pose.items()}
        for k, v in changes.items(): target[k] = np.asarray(v, dtype=float)
        for i in range(n):
            u = _min_jerk(np.array([(i + 1) / n]))[0]
            cur = np.zeros(3 + 3 * rig.num_joints, dtype=float)
            cur[:3] = pose["root"] + u * (target["root"] - pose["root"])
            e = rest_e.copy()
            for bone, idx in rig.name_to_idx.items():
                # Root translation occupies cur[:3]; never also encode it as
                # root Euler rotation (the old harness did both).
                if bone == "root":
                    continue
                if bone in target:
                    e[idx] += pose[bone] + u * (target[bone] - pose[bone])
            cur[3:] = e.reshape(-1)
            frames.append(cur)
        pose = target
        labels.append({"name": name, "start": start, "end": len(frames) - 1,
                       "frame_range": [start, len(frames) - 1], "speed_class": speed})

    add("standing_still", 45, {}, "slow")
    add("subtle_idle_sway", 45, {"spine_02": (0.04, 0.0, 0.05), "root": (1.5, 0, 0)}, "slow")
    add("torso_lean_forward", 45, {"spine_03": (0.14, 0, 0), "spine_04": (0.10, 0, 0)}, "slow")
    add("torso_lean_side", 45, {"spine_03": (0, 0.12, 0), "spine_04": (0, 0.10, 0)}, "slow")
    add("torso_rotation", 45, {"spine_03": (0, 0, 0.22), "spine_04": (0, 0, 0.18), "pelvis": (0, 0, -0.16)}, "medium")
    add("pelvis_translation_rotation", 45, {"root": (-4, 2, 0), "pelvis": (0.08, 0.05, 0.25)}, "medium")
    add("head_yaw_pitch_roll", 45, {"head": (0.12, -0.16, 0.24), "neck_01": (0, 0, -0.12)}, "slow")
    add("shoulder_elevation", 36, {"upperarm_l": (0, 0, -0.42), "upperarm_r": (0, 0, 0.42)}, "medium")
    add("arms_forward_sideways", 54, {"upperarm_l": (0, -0.85, -0.35), "upperarm_r": (0, -0.85, 0.35)}, "medium")
    add("arms_overhead", 54, {"upperarm_l": (0, -1.45, -0.15), "upperarm_r": (0, -1.45, 0.15), "lowerarm_l": (0, 0.45, 0), "lowerarm_r": (0, 0.45, 0)}, "medium")
    add("elbow_flexion_extension", 45, {"lowerarm_l": (0, 1.25, 0), "lowerarm_r": (0, 1.25, 0)}, "medium")
    add("wrist_forearm_pronation", 45, {"hand_l": (0.55, 0, 0.8), "hand_r": (-0.55, 0, -0.8), "lowerarm_l": (0.35, 0, 0), "lowerarm_r": (-0.35, 0, 0)}, "medium")
    add("hip_flexion_abduction", 54, {"thigh_l": (0.45, 0.25, 0.35), "thigh_r": (0.35, -0.25, -0.35)}, "medium")
    add("knee_flexion_extension", 45, {"calf_l": (-0.9, 0, 0), "calf_r": (-0.75, 0, 0)}, "medium")
    add("foot_lift_and_plant", 60, {"thigh_l": (-0.35, 0, 0), "calf_l": (1.25, 0, 0), "foot_l": (-0.55, 0, 0), "ball_l": (-0.35, 0, 0)}, "medium")
    add("march_in_place", 90, {"thigh_l": (0.65, 0.10, 0.08), "calf_l": (-0.75, 0, 0), "thigh_r": (-0.55, -0.08, -0.06), "calf_r": (0.55, 0, 0), "root": (0, 0, 0)}, "medium")
    add("turn", 60, {"root": (0, 0, 0), "pelvis": (0, 0, 0.8), "spine_03": (0, 0, -0.45), "head": (0, 0, 0.5)}, "medium")
    add("fast_punch", 18, {"upperarm_l": (0, -4.6, -0.1), "lowerarm_l": (0, 0.15, 0)}, "fast")
    add("fast_punch_recovery", 22, {"upperarm_l": (0, -0.2, 0), "lowerarm_l": (0, 0, 0)}, "fast")
    add("fast_swing", 18, {"upperarm_r": (0, -4.6, 0.2), "lowerarm_r": (0, 1.0, 0)}, "fast")
    add("fast_swing_recovery", 22, {"upperarm_r": (0, -0.2, 0), "lowerarm_r": (0, 0, 0)}, "fast")
    add("fast_kick", 18, {"thigh_r": (-2.8, 0.1, 0), "calf_r": (2.2, 0, 0), "foot_r": (-0.45, 0, 0)}, "fast")
    add("fast_kick_recovery", 28, {"thigh_r": (0, 0, 0), "calf_r": (0, 0, 0), "foot_r": (0, 0, 0)}, "fast")
    add("slow_fast_transition_reversal", 60, {"spine_03": (0.25, 0, -0.35), "upperarm_l": (0, -1.0, -0.3), "upperarm_r": (0, -1.0, 0.3), "thigh_l": (0.5, 0.1, 0), "thigh_r": (-0.5, -0.1, 0)}, "medium")
    add("simultaneous_upper_lower_body", 60, {"spine_03": (-0.25, 0.12, 0.35), "head": (0.1, 0, -0.4), "upperarm_l": (0, -1.4, -0.3), "upperarm_r": (0, -1.4, 0.3), "thigh_l": (-0.45, 0.15, 0), "thigh_r": (0.65, -0.15, 0), "calf_l": (0.65, 0, 0), "calf_r": (-0.7, 0, 0)}, "medium")
    add("return_to_stance", 45, {"root": (0, 0, 0), "pelvis": (0, 0, 0), "spine_03": (0, 0, 0), "head": (0, 0, 0), "upperarm_l": (0, 0, 0), "upperarm_r": (0, 0, 0), "lowerarm_l": (0, 0, 0), "lowerarm_r": (0, 0, 0), "thigh_l": (0, 0, 0), "thigh_r": (0, 0, 0), "calf_l": (0, 0, 0), "calf_r": (0, 0, 0), "foot_l": (0, 0, 0), "foot_r": (0, 0, 0), "ball_l": (0, 0, 0), "ball_r": (0, 0, 0)}, "medium")
    add("standing_still_final", 45, {}, "slow")
    animation = np.asarray(frames)
    fk = animation_fk(animation, rig)
    ids = rig.name_to_idx
    ball = fk["positions"][:, [ids["ball_l"], ids["ball_r"]]]
    ground = float(np.nanpercentile(ball[..., 2], 5))
    speed = np.linalg.norm(np.diff(ball, axis=0), axis=-1, ord=2)
    contacts = np.zeros((len(animation), 2), dtype=bool)
    contacts[1:] = (speed < 1.0) & (ball[1:, :, 2] < ground + 2.0)
    contacts[0] = contacts[1]
    meta = {"kind": "diagnostic", "fps": float(fps), "ground_z_cm": ground,
            "segments": labels, "contact_mask": contacts.astype(bool).tolist(),
            "anatomy": {"units": "cm Z-up", "face_frame": "head local",
                        "face_offsets_cm": _face_offsets().tolist(),
                        "foot_offsets_cm": {"toe": [0, -5, 0], "small_toe": [4.2, -4, 0], "heel": [0, 4.5, -3]}},
            "seed": 1729}
    # Report peaks only over solver-driven bones.  Helper/twist/finger bones
    # can amplify an Euler change and are not solver targets.
    driven = [ids[n] for n in SOLVER_DRIVEN_BONES if n in ids]
    qd = fk["world_q"][:, driven]
    frame_delta = np.degrees((Rotation.from_quat(qd[1:].reshape(-1, 4)) *
                              Rotation.from_quat(qd[:-1].reshape(-1, 4)).inv()).magnitude())
    frame_delta = frame_delta.reshape(len(animation) - 1, len(driven)) * fps
    for seg in labels:
        a0, b0 = int(seg["start"]), int(seg["end"])
        vals = frame_delta[max(0, a0):max(0, b0), :] if b0 > a0 else np.zeros((0, len(driven)))
        if vals.size:
            flat = int(np.argmax(vals)); ii, jj = divmod(flat, vals.shape[1])
            seg["peak_angular_speed_deg_per_s"] = float(vals[ii, jj])
            seg["peak_bone"] = SOLVER_DRIVEN_BONES[jj]
        else:
            seg["peak_angular_speed_deg_per_s"] = 0.0; seg["peak_bone"] = None
    # A generator discontinuity is a harness error; retain explicit evidence
    # in the sidecar for report consumers.
    seg_meta = frame_delta.max(axis=1) if len(frame_delta) else np.zeros(0)
    meta["sanity"] = {"solver_driven_bones": SOLVER_DRIVEN_BONES,
                       "max_frame_delta_deg": float(np.max(seg_meta) / fps) if len(seg_meta) else 0.0,
                       "discontinuity_frames_over_30deg": np.flatnonzero(seg_meta / fps > 30).astype(int).tolist()}
    return animation, meta


def resample_animation(animation: np.ndarray, frames: int, rig: Skeleton | str | Path | None = None) -> np.ndarray:
    """Slerp local rotations and linearly interpolate root translation."""
    rig = _rig(rig); a = np.asarray(animation, float)
    if frames < 2 or len(a) < 2: raise ValueError("resampling requires at least two frames")
    old = np.arange(len(a), dtype=float); new = np.linspace(0, len(a) - 1, frames)
    e = a[:, 3:].reshape(len(a), rig.num_joints, 3)
    out = np.zeros((frames, a.shape[1]), float); out[:, :3] = np.column_stack([np.interp(new, old, a[:, k]) for k in range(3)])
    for j in range(rig.num_joints):
        q = Slerp(old, Rotation.from_euler("xyz", e[:, j]))(new)
        out[:, 3 + 3*j:3 + 3*j + 3] = q.as_euler("xyz")
    return out


def load_realistic_gt(path: str | Path, fast2x: bool = False, rig: Skeleton | str | Path | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    rig = _rig(rig); a = np.load(path, allow_pickle=False)
    if fast2x: a = resample_animation(a, max(2, len(a) // 2), rig)
    fk = animation_fk(a, rig); ids = rig.name_to_idx
    ball = fk["positions"][:, [ids["ball_l"], ids["ball_r"]]]
    speed = np.linalg.norm(np.diff(ball, axis=0), axis=-1)
    contacts = np.zeros((len(a), 2), bool); ground = float(np.nanpercentile(ball[..., 2], 5))
    if len(a) > 1: contacts[1:] = (speed < 1.0) & (ball[1:, :, 2] < ground + 2)
    contacts[0] = contacts[1] if len(a) > 1 else False
    meta = {"kind": "realistic_fast2x" if fast2x else "realistic", "fps": FPS * (2 if fast2x else 1),
            "ground_z_cm": ground, "segments": [{"name": "real_v7", "start": 0, "end": len(a)-1,
            "frame_range": [0, len(a)-1], "speed_class": "fast" if fast2x else "medium"}],
            "contact_mask": contacts.tolist(), "anatomy": {"source": str(path)}}
    return a, meta


def _finite_stats(x: np.ndarray) -> dict[str, float | int | None]:
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if not len(x): return {"n": 0, "p50": None, "p95": None, "max": None, "rms": None}
    return {"n": int(len(x)), "p50": float(np.percentile(x, 50)), "p95": float(np.percentile(x, 95)),
            "max": float(np.max(x)), "rms": float(np.sqrt(np.mean(x*x)))}


def _quadratic_loo(track: np.ndarray, fps: float = FPS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Leave-one-out local quadratic residuals with analytic leverage correction."""
    x = np.asarray(track, float); n = len(x)
    residual = np.full_like(x, np.nan); speed = np.full(n, np.nan); leverage = np.full(n, np.nan)
    for i in range(n):
        lo, hi = max(0, i - 4), min(n, i + 5)
        offsets = np.arange(lo, hi, dtype=float) - i
        keep = offsets != 0
        for c in range(x.shape[-1]):
            finite = keep & np.isfinite(x[lo:hi, c])
            if int(finite.sum()) < 3 or not np.isfinite(x[i, c]):
                continue
            t = offsets[finite]
            X = np.column_stack((np.ones(len(t)), t, t * t))
            pinv = np.linalg.pinv(X)
            beta = pinv @ x[lo:hi, c][finite]
            hvec = np.array([1., 0., 0.]) @ pinv
            h = float(hvec @ hvec)
            leverage[i] = h if not np.isfinite(leverage[i]) else max(leverage[i], h)
            residual[i, c] = (x[i, c] - beta[0]) / np.sqrt(max(1.0 + h, 1e-9))
            if c == 0: speed[i] = abs(float(beta[1])) * fps
    return residual, speed, leverage


def _residual(track: np.ndarray) -> np.ndarray:
    return _quadratic_loo(track)[0]


def _runs(mask: np.ndarray) -> list[int]:
    vals = []; i = 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j < len(mask) and mask[j]: j += 1
            vals.append(j-i); i = j
        else: i += 1
    return vals


def _robust_sigma(values: np.ndarray) -> float:
    v = np.asarray(values, float); v = v[np.isfinite(v)]
    if not len(v): return 1e-4
    med = float(np.median(v)); mad = float(np.median(np.abs(v - med)))
    return max(1.4826 * mad, 1e-4)


def _group_stats(points_cm: np.ndarray, conf: np.ndarray, indices: list[int]) -> dict[str, Any]:
    x = np.asarray(points_cm[:, indices], float)
    residuals = []; speeds = []
    for j in range(x.shape[1]):
        r, s, _ = _quadratic_loo(x[:, j])
        residuals.append(r); speeds.append(s)
    res = np.asarray(residuals); speed = np.asarray(speeds)
    finite = np.isfinite(x).all(axis=-1); valid_res = np.isfinite(res).all(axis=-1)
    mag = np.linalg.norm(res, axis=-1)
    all_mask = valid_res & finite.T
    qs_mask = all_mask & (speed < 10.0)
    all_vals, qs_vals = mag[all_mask], mag[qs_mask]
    all_coord, qs_coord = res[all_mask], res[qs_mask]
    sigma_all = _robust_sigma(all_coord); sigma_qs = _robust_sigma(qs_coord)
    rho_values = []
    for j in range(res.shape[0]):
        for c in range(3):
            ok = np.isfinite(res[j, :-1, c]) & np.isfinite(res[j, 1:, c])
            if int(ok.sum()) > 2:
                u, v = res[j, :-1, c][ok], res[j, 1:, c][ok]
                if np.std(u) > 1e-12 and np.std(v) > 1e-12: rho_values.append(np.corrcoef(u, v)[0, 1])
    nan = ~finite
    run = []; p01 = []; p10 = []
    for j in range(nan.shape[1]):
        mask = nan[:, j]; run.extend(_runs(mask))
        trans01 = np.sum(~mask[:-1] & mask[1:]); trans00 = np.sum(~mask[:-1])
        trans10 = np.sum(mask[:-1] & ~mask[1:]); trans11 = np.sum(mask[:-1])
        if trans00: p01.append(trans01 / trans00)
        if trans11: p10.append(trans10 / trans11)
    # Estimate transition rates from the observed stationary rate and mean
    # missing-run length, rather than mixing pointwise and pooled transitions.
    mean_run = float(np.mean(run)) if run else 1.0
    target_rate = float(np.mean(nan))
    p10_est = 1.0 / max(mean_run, 1.0)
    p01_est = target_rate * p10_est / max(1.0 - target_rate, 1e-6)
    cq = np.quantile(conf[:, indices][np.isfinite(conf[:, indices])], [0, .1, .25, .5, .75, .9, 1]).tolist() if np.isfinite(conf[:, indices]).any() else [1]*7
    def block(mask, sigma):
        v = mag[mask]
        return {"residual_cm": _finite_stats(v), "sigma_cm": float(sigma),
                "outlier_rate": float(np.mean(v > 6 * sigma)) if len(v) else 0.0,
                "quasi_static_frames": int(np.sum(mask))}
    return {"indices": indices, "hf_residual_cm": _finite_stats(all_vals),
            "sigma_cm": float(sigma_qs), "rho": float(np.clip(np.median(rho_values) if rho_values else 0.0, -0.95, .95)),
            "outlier_rate": float(np.mean(all_vals > 6 * sigma_all)) if len(all_vals) else 0.0,
            "all_frames": block(all_mask, sigma_all), "quasi_static": block(qs_mask, sigma_qs),
            "nan_rate": float(np.mean(nan)), "nan_run_mean": mean_run,
            "nan_run_p95": float(np.percentile(run, 95)) if run else 1.0,
            "nan_markov": {"p01": p01_est, "p10": p10_est},
            "confidence_quantiles": cq}


def calibrate_noise_model(npz_path: str | Path, output_json: str | Path | None = None) -> dict[str, Any]:
    """Calibrate all generator parameters from a real skeleton3d.npz clip."""
    with np.load(npz_path, allow_pickle=False) as z:
        raw = np.asarray(z["skeleton3d"], float)
        conf = np.asarray(z["confidence"], float) if "confidence" in z else np.ones(raw.shape[:2])
    p = np.empty_like(raw); p[..., 0] = raw[..., 0] * 100; p[..., 1] = -raw[..., 2] * 100; p[..., 2] = raw[..., 1] * 100
    groups = {name: _group_stats(p, conf, idx) for name, idx in REGION_INDICES.items()}
    for spec in groups.values():
        spec["apply_sigma_cm"] = float(spec["sigma_cm"] * 0.75)
        spec["apply_rho"] = float(np.clip(spec["rho"] + 0.25, -0.8, 0.8))
    # Foot garbage is an event statistic, split by whether the ankle is off
    # the ground (>8 cm above that clip's minimum), and includes direction
    # flips against the small-toe/heel rays.
    dists = []; anomaly_flags = []; off_flags = []; flip_flags = []; heels = []
    for ankle, toe, small, heel in ((15,17,18,19),(16,20,21,22)):
        ankle_ok = np.isfinite(p[:, ankle]).all(axis=1)
        toe_ok = np.isfinite(p[:, toe]).all(axis=1)
        small_ok = np.isfinite(p[:, small]).all(axis=1)
        heel_ok = np.isfinite(p[:, heel]).all(axis=1)
        d = np.linalg.norm(p[:, toe] - p[:, ankle], axis=1); ok = ankle_ok & toe_ok
        dists.extend(d[ok]); heels.append(float(np.mean(heel_ok)))
        off = ankle_ok & (p[:, ankle, 2] > np.nanmin(p[:, ankle, 2]) + 8.0)
        off_flags.extend(off[ok])
        anomaly_flags.extend(d[ok])
        flip = np.zeros(len(p), bool)
        for f in np.flatnonzero(ok & small_ok):
            u = p[f, toe] - p[f, ankle]; v = p[f, small] - p[f, ankle]
            flip[f] = np.dot(u, v) < 0
        for f in np.flatnonzero(ok & heel_ok):
            u = p[f, toe] - p[f, ankle]; v = p[f, heel] - p[f, ankle]
            flip[f] |= np.dot(u, -v) < 0
        flip_flags.extend(flip[ok])
    dists = np.asarray(dists); off_flags = np.asarray(off_flags, bool); flip_flags = np.asarray(flip_flags, bool)
    med = float(np.median(dists)) if len(dists) else 20.0
    band = [float(np.percentile(dists, 1)), float(np.percentile(dists, 99))] if len(dists) else [10., 35.]
    anomaly_off = (dists < band[0]) | (dists > band[1])
    # Keep distance garbage and directional flips as separate event channels.
    off_mask = np.asarray(off_flags, bool)
    garbage_off = float(np.mean(anomaly_off[off_mask])) if np.any(off_mask) else 0.0
    garbage_ground = float(np.mean(anomaly_off[~off_mask])) if np.any(~off_mask) else 0.0
    garbage = anomaly_off
    off_rate = garbage_off
    ground_rate = garbage_ground
    # v3 separates the calibrated, plausible short-ray garbage process from
    # direction reversals.  The target rates are measured on the real clip;
    # v2's 3x-ray reversal made both channels look like garbage and inflated
    # the foot residual p95 by an order of magnitude.
    model = {"version": 3, "units": "cm Z-up internal", "fps": FPS, "seed": 271828,
             "groups": groups, "foot": {"toe_distance_p50_cm": med, "toe_distance_band_cm": band,
             "toe_anomaly_rate": float(np.mean(anomaly_off)) if len(anomaly_off) else 0.0,
             "garbage_rate": 0.0,
             "garbage_rate_off_ground": 0.015,
             "garbage_rate_grounded": 0.115,
             "direction_flip_rate": 0.08033077377436504,
             "garbage_distance_cm": 20.5,
             "heel_availability": float(np.mean(heels)), "off_ground_threshold_cm": 8.0},
             "source": str(npz_path), "residual_method": "9-frame leave-one-out quadratic; sigma corrected by sqrt(1+h)",
             "foot_calibration": {"real_reference": {"garbage_rate_off_ground": 0.026717557251908396,
             "garbage_rate_grounded": 0.09014675052410902, "direction_flip_rate": 0.08033077377436504,
             "residual_p95_cm": 3.621632207627925}, "process": "plausible ray length + independent direction flip"}}
    if output_json:
        Path(output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(output_json).write_text(json.dumps(model, indent=2), encoding="utf-8")
    return model


def _sample_quantiles(q: list[float], rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    return np.interp(rng.random(shape), np.array([0, .1, .25, .5, .75, .9, 1.0]), q)


def apply_observation_noise(gt_points_cm: np.ndarray, model: dict[str, Any], seed: int = 271828,
                            enabled: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Apply calibrated AR(1), outlier, burst, dropout and foot garbage noise."""
    x = np.asarray(gt_points_cm, float); out = x.copy(); rng = np.random.default_rng(seed)
    conf = np.ones(x.shape[:2], dtype=np.float32)
    if not enabled: return out, conf
    # Apply observation noise once per keypoint.  Calibration groups overlap
    # (body contains wrists/feet), so specialized groups take precedence.
    assigned: dict[int, str] = {}
    for group in ("head_face", "wrists", "feet", "hand_mcp", "body"):
        for j in REGION_INDICES.get(group, []): assigned.setdefault(j, group)
    assigned_groups: dict[str, list[int]] = {}
    for j, group in assigned.items(): assigned_groups.setdefault(group, []).append(j)
    for group, indices in assigned_groups.items():
        spec = model["groups"].get(group, model["groups"]["body"])
        sigma, rho = float(spec.get("apply_sigma_cm", spec.get("sigma_cm", .1))), float(spec.get("apply_rho", spec.get("rho", 0)))
        for j in indices:
            prev = np.zeros(3)
            innov = sigma * np.sqrt(max(1-rho*rho, .05))
            for f in range(len(x)):
                prev = rho * prev + rng.normal(0, innov, 3)
                out[f, j] += prev
            # Single-frame outliers and distal short bursts.
            rate = float(spec.get("outlier_rate", 0.0)); starts = rng.random(len(x)) < rate
            for f in np.flatnonzero(starts): out[f, j] += rng.normal(0, max(10*sigma, 1), 3)
            if group in {"feet", "hand_mcp"}:
                for f in np.flatnonzero(rng.random(len(x)) < rate * 0.5):
                    burst = int(rng.integers(2, 5))
                    out[f:min(len(x), f+burst), j] += rng.normal(0, max(4*sigma, 1), (min(len(x),f+burst)-f, 3))
            # Two-state Markov dropout calibrated from p01/p10, rather than
            # independent Bernoulli samples.  This preserves long face runs.
            dr = float(spec.get("nan_rate", 0)); markov = spec.get("nan_markov", {})
            p10 = float(markov.get("p10", 1.0 / max(float(spec.get("nan_run_mean", 1)), 1.0)))
            p01 = float(markov.get("p01", dr * p10 / max(1.0 - dr, 1e-6)))
            p01 = float(np.clip(p01, 0, 1)); p10 = float(np.clip(p10, 0, 1))
            missing = False; f = 0
            while f < len(x):
                if not missing:
                    missing = bool(rng.random() < p01)
                else:
                    missing = not bool(rng.random() < p10)
                if missing:
                    out[f, j] = np.nan; conf[f, j] = 0
                f += 1
            conf[:, j] = _sample_quantiles(spec.get("confidence_quantiles", [1]*7), rng, (len(x),)).astype(np.float32)
    foot = model.get("foot", {}); all_rate = float(foot.get("garbage_rate", 0))
    off_rate = float(foot.get("garbage_rate_off_ground", all_rate)); ground_rate = float(foot.get("garbage_rate_grounded", all_rate))
    flip_all = float(foot.get("direction_flip_rate", 0.0))
    if all_rate > 0 or off_rate > 0 or ground_rate > 0 or flip_all > 0:
        for side, ankle, toe, heel in (("l",15,17,19),("r",16,20,22)):
            eligible = np.isfinite(out[:, ankle]).all(axis=1)
            ankle_min = np.nanmin(out[:, ankle, 2]) if np.any(eligible) else 0.0
            off = eligible & (out[:, ankle, 2] > ankle_min + float(foot.get("off_ground_threshold_cm", 8.0)))
            probs = np.where(off, off_rate, ground_rate)
            garbage_distance = float(foot.get("garbage_distance_cm", max(float(foot.get("toe_distance_p50_cm", 20.0)) + 4.0, 24.0)))
            garbage_mask = eligible & (rng.random(len(x)) < probs)
            for f in np.flatnonzero(garbage_mask):
                direction = out[f, toe] - out[f, ankle]; norm = np.linalg.norm(direction)
                direction = direction / max(norm, 1e-9)
                # Garbage is a short, plausible ray-length excursion.  The
                # v2 implementation reversed the ray and stretched it to 3x
                # the median, creating 25 cm residual tails.
                out[f, toe] = out[f, ankle] + direction * garbage_distance
            # Direction flips are exclusive of garbage events so the two
            # calibrated channels do not multiply into a 40 cm residual.
            flip_mask = eligible & ~garbage_mask & (rng.random(len(x)) < flip_all)
            for f in np.flatnonzero(flip_mask):
                direction = out[f, toe] - out[f, ankle]; norm = np.linalg.norm(direction)
                if norm > 1e-9: out[f, toe] = out[f, ankle] - direction / norm * norm
    return out, conf


def _to_engine_raw(points_cm: np.ndarray) -> np.ndarray:
    raw = np.empty_like(points_cm, dtype=np.float32)
    raw[..., 0] = points_cm[..., 0] / 100
    raw[..., 1] = points_cm[..., 2] / 100
    raw[..., 2] = -points_cm[..., 1] / 100
    return raw


def write_clip(name: str, animation: np.ndarray, meta: dict[str, Any], output_dir: str | Path,
               rig: Skeleton | str | Path | None = None, noise_model: dict[str, Any] | None = None,
               noisy: bool = True, seed: int = 271828) -> dict[str, Path]:
    """Write engine input plus GT NPZ/JSON sidecars without overwriting files."""
    rig = _rig(rig); outdir = Path(output_dir); outdir.mkdir(parents=True, exist_ok=True)
    a = np.asarray(animation, float); gt = keypoints_from_animation(a, rig)
    obs, conf = apply_observation_noise(gt, noise_model or {"groups": {}, "foot": {}}, seed, noisy)
    npz_path, gt_path, json_path = outdir / f"{name}.npz", outdir / f"{name}_gt.npz", outdir / f"{name}_gt.json"
    for p in (npz_path, gt_path, json_path):
        if p.exists(): raise FileExistsError(f"refusing to overwrite {p}")
    np.savez_compressed(npz_path, skeleton3d=_to_engine_raw(obs), confidence=conf,
                        units=np.asarray("meters"), fps=np.asarray(meta.get("fps", FPS)))
    np.savez_compressed(gt_path, animation=a, gt_keypoints=gt, confidence=np.ones(a.shape[:2], np.float32),
                        contact_mask=np.asarray(meta.get("contact_mask", []), bool))
    written_meta = dict(meta); written_meta.update({"name": name, "noisy": noisy, "gt_npz": gt_path.name,
                                                      "engine_npz": npz_path.name})
    json_path.write_text(json.dumps(written_meta, indent=2), encoding="utf-8")
    return {"npz": npz_path, "gt_npz": gt_path, "json": json_path}


def _stats(x: np.ndarray) -> dict[str, Any]: return _finite_stats(np.asarray(x))


def _orientation_error(gtq: np.ndarray, outq: np.ndarray) -> np.ndarray:
    return np.degrees((Rotation.from_quat(gtq).inv() * Rotation.from_quat(outq)).magnitude())


def _speed(q: np.ndarray, fps: float) -> np.ndarray:
    if len(q) < 2: return np.zeros(0)
    return np.degrees((Rotation.from_quat(q[1:]) * Rotation.from_quat(q[:-1]).inv()).magnitude()) * fps


def _lag(gt: np.ndarray, out: np.ndarray, fps: float) -> float | None:
    """Normalized cross-correlation lag; positive means output is delayed."""
    n = min(len(gt), len(out)); gt, out = np.asarray(gt[:n], float), np.asarray(out[:n], float)
    if n < 5: return None
    maxlag = min(8, n - 2); scores = []
    for k in range(-maxlag, maxlag + 1):
        if k >= 0: a, b = out[k:], gt[:n-k]
        else: a, b = out[:n+k], gt[-k:]
        a, b = a - np.mean(a), b - np.mean(b); den = np.linalg.norm(a) * np.linalg.norm(b)
        scores.append(float(np.dot(a, b) / den) if den > 1e-12 else -1.0)
    i = int(np.argmax(scores)); lag = -maxlag + i
    if 0 < i < len(scores)-1:
        y0, y1, y2 = scores[i-1:i+2]; den = y0 - 2*y1 + y2
        if abs(den) > 1e-12: lag += .5 * (y0-y2) / den
    return float(lag / fps)


def _rotation_mean(rotations: Rotation) -> Rotation:
    q = rotations.as_quat(); q = np.asarray(q, float)
    if q.ndim == 1: return Rotation.from_quat(q)
    ref = q[0].copy(); q = q * np.where((q @ ref)[:, None] < 0, -1.0, 1.0)
    mean = np.sum(q, axis=0); norm = np.linalg.norm(mean)
    return Rotation.from_quat(mean / max(norm, 1e-12))


def _rest_long_axis(rig: Skeleton, index: int) -> np.ndarray:
    """Return a bone's rest long axis in that bone's local coordinates.

    A child direction is preferred because it is the direction represented by
    the bone rotation.  Leaf bones have no child, so their parent-to-bone rest
    direction is used as the least ambiguous anatomical axis.
    """
    positions, rotations = rig.get_forward_kinematics()
    child = next((i for i, p in enumerate(rig.parents) if p == index), None)
    if child is not None:
        world = positions[child] - positions[index]
    elif rig.parents[index] >= 0:
        world = positions[index] - positions[rig.parents[index]]
    else:
        world = np.array([0.0, 0.0, 1.0])
    norm = float(np.linalg.norm(world))
    if norm < 1e-9:
        return np.array([0.0, 0.0, 1.0])
    return Rotation.from_quat(rotations[index]).inv().apply(world / norm)


def _wrap_degrees(values: np.ndarray) -> np.ndarray:
    values = (np.asarray(values, float) + 180.0) % 360.0 - 180.0
    return np.where(np.isclose(values, -180.0), 180.0, values)


def _swing_twist(dynamic: Rotation, axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Decompose rotations into swing and signed, wrapped twist about axis."""
    q = np.asarray(dynamic.as_quat(), float)
    if q.ndim == 1:
        q = q[None, :]
    axis = np.asarray(axis, float)
    if axis.ndim == 1:
        axis = np.broadcast_to(axis, (len(q), 3)).copy()
    axis /= np.maximum(np.linalg.norm(axis, axis=-1, keepdims=True), 1e-12)
    projected = axis * np.sum(q[:, :3] * axis, axis=-1, keepdims=True)
    tq = np.concatenate((projected, q[:, 3:4]), axis=-1)
    tq /= np.maximum(np.linalg.norm(tq, axis=-1, keepdims=True), 1e-12)
    # q and -q represent the same rotation.  Canonicalizing the twist
    # quaternion makes the signed angle continuous through ordinary motion.
    tq *= np.where(tq[:, 3:4] < 0.0, -1.0, 1.0)
    twist_rad = 2.0 * np.arctan2(np.sum(tq[:, :3] * axis, axis=-1), tq[:, 3])
    twist_deg = _wrap_degrees(np.degrees(twist_rad))
    twist_rot = Rotation.from_quat(tq)
    swing_rot = dynamic * twist_rot.inv()
    return swing_rot, twist_deg


def _twist_winding(twist_deg: np.ndarray) -> tuple[float, int]:
    """Sum only trustworthy wrapped frame differences and count flips."""
    if len(twist_deg) < 2:
        return 0.0, 0
    diff = _wrap_degrees(np.diff(twist_deg))
    valid = np.abs(diff) < 90.0
    return float(np.sum(diff[valid])), int(np.sum(~valid))


def _dynamic_rotation_metrics(gtq: np.ndarray, outq: np.ndarray, axis: np.ndarray | None = None) -> dict[str, Any]:
    gt = Rotation.from_quat(gtq); out = Rotation.from_quat(outq)
    raw = gt.inv() * out
    # C is a static right/local offset.  The required dynamic error is
    # E(t)=(R_gt(t) C)^-1 R_out(t)=C^-1 R_gt(t)^-1 R_out(t).
    static = _rotation_mean(raw)
    dynamic = static.inv() * raw
    raw_deg = np.degrees(raw.magnitude()); dyn_deg = np.degrees(dynamic.magnitude())
    if axis is None:
        axis = np.array([0.0, 0.0, 1.0])
    swing_rot, twist_deg = _swing_twist(dynamic, axis)
    swing = np.degrees(swing_rot.magnitude())
    winding, flips = _twist_winding(twist_deg)
    twist_abs = np.abs(twist_deg)
    return {"raw": raw, "dynamic": dynamic, "raw_deg": raw_deg, "dynamic_deg": dyn_deg,
            "static_offset_deg": float(np.degrees(static.magnitude())),
            "dynamic_axial_twist_error": {
                "p95_abs_deg": float(np.percentile(twist_abs, 95)) if len(twist_abs) else None,
                "max_abs_deg": float(np.max(twist_abs)) if len(twist_abs) else None,
                "range_deg": float(np.ptp(twist_deg)) if len(twist_deg) else 0.0,
                "twist_winding": winding, "twist_flips": flips,
                "values_deg": twist_deg},
            "dynamic_swing_error": _stats(swing)}


def _gt_events(speed: np.ndarray, threshold: float = 200.0) -> list[tuple[int, int]]:
    hot = np.asarray(speed) > threshold; runs=[]; i=0
    while i < len(hot):
        if not hot[i]: i += 1; continue
        j=i
        while j+1 < len(hot) and hot[j+1]: j += 1
        if runs and i - runs[-1][1] - 1 < 3: runs[-1] = (runs[-1][0], j)
        else: runs.append((i,j))
        i=j+1
    return runs


def _event_metrics(gtq: np.ndarray, outq: np.ndarray, fps: float) -> list[dict[str, Any]]:
    gs, os = _speed(gtq, fps), _speed(outq, fps); events=[]
    dyn = _dynamic_rotation_metrics(gtq, outq)["dynamic_deg"]
    n = min(len(gtq), len(outq))
    for start, end in _gt_events(gs):
        lo, hi = max(0, start-3), min(len(os)-1, end+3)
        rms_hi = min(len(dyn), end+2)
        # Speed samples [start:end] describe orientation frames
        # [start:start+1:end+1].  Excursion is measured from the GT event
        # start orientation over exactly that event window (with the small
        # established context only retained for peak-speed diagnostics).
        frame_start = min(max(0, start), n - 1)
        frame_end = min(n - 1, end + 1)
        gt_exc = np.degrees(np.max((Rotation.from_quat(gtq[frame_start]).inv() *
                                    Rotation.from_quat(gtq[frame_start:frame_end + 1])).magnitude()))
        out_exc = np.degrees(np.max((Rotation.from_quat(outq[frame_start]).inv() *
                                     Rotation.from_quat(outq[frame_start:frame_end + 1])).magnitude()))
        peak = float(np.max(gs[start:end+1]))
        events.append({"start": int(start), "end": int(end),
                       "excursion_gt_deg": float(gt_exc),
                       "excursion_output_deg": float(out_exc),
                       "excursion_ratio": float(out_exc / max(gt_exc, 1e-9)),
                       # Retained as a secondary, noise-sensitive diagnostic.
                       "peak_ratio": float(np.max(os[lo:hi+1]) / max(peak, 1e-9)),
                       "rms_dynamic_orientation_error_deg": float(np.sqrt(np.mean(dyn[start:rms_hi]**2))),
                       "temporal_lag_s": _lag(gs[max(0,start-10):min(len(gs),end+11)], os[max(0,start-10):min(len(os),end+11)], fps),
                       "temporal_lag_frames": None})
        if events[-1]["temporal_lag_s"] is not None: events[-1]["temporal_lag_frames"] = events[-1]["temporal_lag_s"] * fps
    return events


def _segment_masks(meta: dict[str, Any], n: int) -> dict[str, np.ndarray]:
    result = {}
    for seg in meta.get("segments", []):
        a, b = max(0, int(seg.get("start", seg.get("frame_range", [0,0])[0]))), min(n-1, int(seg.get("end", seg.get("frame_range", [0,0])[1])))
        m = np.zeros(n, bool); m[a:b+1] = True; result[seg.get("name", f"segment_{a}")] = m
    return result


def evaluate(output_anim: np.ndarray, gt_anim: np.ndarray, gt_meta: dict[str, Any] | str | Path | None,
             rig: Skeleton | str | Path | None = None) -> dict[str, Any]:
    """Evaluate with static rest-relative offsets removed from primary errors."""
    rig = _rig(rig)
    if isinstance(gt_meta, (str, Path)): gt_meta = json.loads(Path(gt_meta).read_text(encoding="utf-8"))
    meta = gt_meta or {}; a=np.asarray(output_anim,float); g=np.asarray(gt_anim,float); n=min(len(a),len(g)); a,g=a[:n],g[:n]
    fko,fkg=animation_fk(a,rig),animation_fk(g,rig); fps=float(meta.get("fps",FPS)); ids=rig.name_to_idx
    rest_axes={idx: _rest_long_axis(rig, idx) for idx in ids.values()}
    per_bone={}; all_raw=[]; all_dyn=[]; all_pos=[]; all_jit=[]; all_pjit=[]; fast_events=[]
    for name,idx in ids.items():
        parent=rig.parents[idx]
        dm=_dynamic_rotation_metrics(fkg["world_q"][:,idx],fko["world_q"][:,idx],rest_axes[idx])
        pd=np.linalg.norm(fko["positions"][:,idx]-fkg["positions"][:,idx],axis=-1)
        j=np.linalg.norm(np.diff(dm["dynamic"].as_rotvec(),n=2,axis=0),axis=-1)*180/np.pi if n>=3 else np.zeros(0)
        pj=np.linalg.norm(np.diff(fko["positions"][:,idx]-fkg["positions"][:,idx],n=2,axis=0),axis=-1) if n>=3 else np.zeros(0)
        per_bone[name]={"orientation_deg":_stats(dm["dynamic_deg"]),"raw_orientation_deg":_stats(dm["raw_deg"]),"static_offset_deg":dm["static_offset_deg"],"dynamic_error_deg":_stats(dm["dynamic_deg"]),"dynamic_axial_twist_error":{k:v for k,v in dm["dynamic_axial_twist_error"].items() if k!="values_deg"},"dynamic_swing_error":dm["dynamic_swing_error"],"position_cm":_stats(pd),"jitter_deg_per_frame2":_stats(j),"position_jitter_cm_per_frame2":_stats(pj)}
        all_raw.extend(dm["raw_deg"]); all_dyn.extend(dm["dynamic_deg"]); all_pos.extend(pd); all_jit.extend(j); all_pjit.extend(pj)
        for ev in _event_metrics(fkg["world_q"][:,idx],fko["world_q"][:,idx],fps): ev["bone"]=name; fast_events.append(ev)
    per_region={}
    for region,names in REGION_BONES.items():
        idxs=[ids[x] for x in names if x in ids]; raw=[]; dyn=[]; jit=[]
        for i in idxs:
            dm=_dynamic_rotation_metrics(fkg["world_q"][:,i],fko["world_q"][:,i],rest_axes[i]); raw.extend(dm["raw_deg"]); dyn.extend(dm["dynamic_deg"])
            if n>=3: jit.extend(np.linalg.norm(np.diff(dm["dynamic"].as_rotvec(),n=2,axis=0),axis=-1)*180/np.pi)
        ev=[e for e in fast_events if e["bone"] in [rig.node_names[i] for i in idxs]]
        ratios=[e["excursion_ratio"] for e in ev]; peak_ratios=[e["peak_ratio"] for e in ev]; lags=[e["temporal_lag_s"] for e in ev if e["temporal_lag_s"] is not None]
        per_region[region]={"orientation_deg":_stats(np.asarray(dyn)),"dynamic_error_deg":_stats(np.asarray(dyn)),"raw_orientation_deg":_stats(np.asarray(raw)),"jitter_deg_per_frame2":_stats(np.asarray(jit)),"bones":names,"fast_events":{"count":len(ev),"excursion_ratio":_stats(np.asarray(ratios)),"peak_ratio":_stats(np.asarray(peak_ratios)),"lag_s":_stats(np.asarray(lags)),"worst_excursion_ratio":float(max(ratios)) if ratios else None,"worst_peak_ratio":float(max(peak_ratios)) if peak_ratios else None,"worst_lag_s":float(max(np.abs(lags))) if lags else None,"events":ev}}
    dp=fko["positions"]-fkg["positions"]; offset=np.nanmean(dp.reshape(-1,3),axis=0); centered=np.linalg.norm(dp-offset,axis=-1)
    twists={}; pops={}
    for name in TWIST_BONES:
        if name not in ids: continue
        i=ids[name]; parent=rig.parents[i]; dm=_dynamic_rotation_metrics(fkg["world_q"][:,i],fko["world_q"][:,i],rest_axes[i]); vals=dm["dynamic_axial_twist_error"]["values_deg"]; twists[name]={**_stats(np.abs(vals)),"p95_abs_deg":float(np.percentile(np.abs(vals),95)) if len(vals) else None,"max_abs_deg":float(np.max(np.abs(vals))) if len(vals) else None,"range_deg":float(np.ptp(vals)) if len(vals) else 0.0,"twist_winding":dm["dynamic_axial_twist_error"]["twist_winding"],"twist_flips":dm["dynamic_axial_twist_error"]["twist_flips"]}
        pops[name]=int(np.sum(np.abs(np.diff(dm["raw_deg"],prepend=dm["raw_deg"][:1]))>30))
    contact=np.asarray(meta.get("contact_mask",[]),bool); feet={}
    if contact.ndim==2 and len(contact)>=n:
        for side,fi,bi,col in (("l",ids.get("foot_l"),ids.get("ball_l"),0),("r",ids.get("foot_r"),ids.get("ball_r"),1)):
            if fi is None: continue
            m=contact[:n,col]; slide=np.linalg.norm(np.diff(fko["positions"][:,fi],axis=0),axis=-1) if n>1 else np.zeros(0); bm=m[1:] if n>1 else m; dm=_dynamic_rotation_metrics(fkg["world_q"][:,fi],fko["world_q"][:,fi]); feet[side]={"contact_frames":int(np.sum(m)),"foot_slide_cm":_stats(slide[bm]),"ball_slide_cm":_stats(np.linalg.norm(np.diff(fko["positions"][:,bi],axis=0),axis=-1)[bm]) if bi is not None and n>1 else _stats([]),"sole_tilt_error_deg":_stats(dm["dynamic_deg"][m]),"foot_axial_twist_error_deg":twists.get("foot_"+side,{})}
    seg_metrics={}
    for name,mask in _segment_masks(meta,n).items():
        cls=next((s.get("speed_class","unknown") for s in meta.get("segments",[]) if s.get("name")==name),"unknown")
        vals=[]
        for idx in ids.values(): vals.extend(_dynamic_rotation_metrics(fkg["world_q"][:,idx],fko["world_q"][:,idx])["dynamic_deg"][mask])
        seg_metrics[name]={"speed_class":cls,"frames":int(np.sum(mask)),"dynamic_error_deg":_stats(np.asarray(vals))}
    return {"orientation_error_deg":_stats(np.asarray(all_dyn)),"raw_orientation_error_deg":_stats(np.asarray(all_raw)),"joint_position_error_cm":{"raw":_stats(np.asarray(all_pos)),"mean_offset_removed":_stats(centered)},"jitter":{"angular_deg_per_frame2":_stats(np.asarray(all_jit)),"position_cm_per_frame2":_stats(np.asarray(all_pjit))},"per_bone":per_bone,"per_region":per_region,"per_segment_speed_class":seg_metrics,"fast_motion":{"threshold_deg_per_s":200,"event_gap_frames":3,"window_frames":3,"lag_window_frames":10,"lag_search_frames":8,"events":fast_events},"pops":pops,"pop_count_over_30deg":int(sum(pops.values())),"axial_twist":twists,"feet":feet,"fps":fps,"frames":n}

def engine_points_from_npz(npz_path: str | Path) -> np.ndarray:
    """Load an engine npz and convert its raw Y-up metres to Z-up centimetres."""
    with np.load(npz_path, allow_pickle=False) as z: raw = np.asarray(z["skeleton3d"], float)
    p = np.empty_like(raw); p[..., 0] = raw[..., 0] * 100; p[..., 1] = -raw[..., 2] * 100; p[..., 2] = raw[..., 1] * 100
    return p
