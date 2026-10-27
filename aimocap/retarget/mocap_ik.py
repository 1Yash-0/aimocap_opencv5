import numpy as np
from scipy.spatial.transform import Rotation
from scipy.signal import butter, sosfiltfilt
from aimocap.retarget.swing_twist import STRAIGHT_FALLBACK_THRESHOLD_DEG


# Head orientation mode. "face_track" pins the head global rotation to the
# frame-based face target built from nose/eyes/ears (F_obs @ F_rest.T).
# Audit evidence (docs/phase_reports/VQ_FIXES.md, audit_head_flip): the face
# target is constructible on 450/450 frames of the 15 s user clip with
# healthy geometry (ear distance p50 18.2 cm vs the 5 cm gate), the head
# tracked the target to 0.008 deg mean (0.002 deg p95), zero orientation
# pops, unchanged neck jitter, and bit-identical determinism. The older
# raw-vector facing metric is structurally biased (~13 deg downward)
# and was superseded by the corrected yaw/pitch decomposition.
# "neck_follow" remains available to make the head a rigid neck follower.
HEAD_ORIENTATION_MODE = "face_track"

# Head target rest-frame construction. "legacy" preserves the historical
# clavicle-derived frame bit-for-bit. "fixed" uses the rig head rest rotation
# and the same canonical face landmark convention as F_obs. "calibrated"
# estimates the clip's face-to-torso offset from neutral-looking frames.
HEAD_TARGET_MODES = {"legacy", "fixed", "calibrated"}

# Orientation-residual unit scale (VQ-FIX1).
# The IK residual vector mixes position terms (cm) with orientation terms
# (radians). At scale 1.0 the LM can trade ~50 deg of neck/spine orientation
# for under 2 cm of position error: measured on the 450-frame user clip,
# ori=1 left the neck canted 53.6 deg from vertical (head facing ~85 deg
# off the face direction) while arm targets were already within ~1-2 cm;
# at ori=30 the neck tracks the data-derived chest frame (10.5 deg tilt,
# ~12 deg facing error) and the full-clip FK-vs-target position error
# DROPPED ~30% (7.71 -> 5.37 cm mean). A radian of orientation error
# displaces landmarks by a lever of roughly 30 cm, so 30 is the
# unit-equivalent weight, not a tuned knob. NaN-gated (D1) frames keep 0.
ORI_WEIGHT_SCALE = 30.0

# Foot gate reference modes
FOOT_GATE_REFERENCE_MODES = {"solved", "observed"}

# Continuation-angle observability thresholds bracket the existing 35-degree
# constrained_rotation guard symmetrically: below 20 degrees the bend plane is
# unobservable, above 50 degrees it is fully trusted.
LIMB_TWIST_OBSERVABILITY_LOW_DEG = 20.0
LIMB_TWIST_OBSERVABILITY_HIGH_DEG = 50.0
LIMB_TWIST_ANCHOR_MODES = {"init", "continuous", "continuous_resid", "smoothed", "two_pass"}
TWO_PASS_LIMB_MODES = {"all", "arms"}
SMOOTHED_LIMB_MODES = {"all", "arms"}
LIMB_TWIST_PIN_MODES = {"always", "fade", "off"}
# Fade spans the measured direction-noise band immediately below the
# straight-fallback guard: a 20-degree interval avoids a hard residual switch
# while retaining full pin authority where twist is unobservable. On the
# conditioned real_v7 clip, p95 absolute frame-to-frame theta steps were
# 6.72/7.89 deg for hips and 10.63/9.95 deg for shoulders (left/right);
# wrist-child values are 57-64 deg because they contain real hand motion,
# not direction noise. Thus theta_lo=15 deg covers the body-limb noise band
# while the high-motion wrist pins fade as the bend becomes observable.
LIMB_TWIST_PIN_THETA_HI_DEG = float(STRAIGHT_FALLBACK_THRESHOLD_DEG)
LIMB_TWIST_PIN_THETA_LO_DEG = LIMB_TWIST_PIN_THETA_HI_DEG - 20.0
TWIST_LIMIT_MODES = {"clamp", "soft", "hybrid", "off"}
LIMIT_PROJECTION_MODES = {"post", "projected"}
# Hybrid keeps the nominal limit in the differentiable cost for selected,
# numerically troublesome joints, but retains a hard post-step safety bound.
# The 10-degree margin is deliberate: it prevents the nominal barrier from
# reintroducing the +/-15-degree clavicle/upper-arm drift while still bounding
# pathological solver excursions before they can propagate to later frames.
TWIST_HYBRID_SAFETY_MARGIN_DEG = 10.0
DEFAULT_SOFT_LIMIT_JOINTS = frozenset({
    "clavicle_l", "clavicle_r", "ankle_l", "ankle_r",
})
ALL_LIMIT_JOINTS = frozenset({
    "knee_l", "knee_r", "ankle_l", "ankle_r",
    "clavicle_l", "clavicle_r", "elbow_l", "elbow_r", "wrist_l", "wrist_r",
})


def parse_soft_limit_joints(value) -> frozenset[str]:
    """Parse the configurable hybrid soft-limit joint set."""
    if value is None:
        return DEFAULT_SOFT_LIMIT_JOINTS
    if isinstance(value, str):
        names = [part.strip() for part in value.split(",") if part.strip()]
    else:
        names = [str(part).strip() for part in value if str(part).strip()]
    unknown = sorted(set(names) - ALL_LIMIT_JOINTS)
    if unknown:
        raise ValueError(
            "soft_limit_joints contains unknown joint(s): " + ", ".join(unknown)
        )
    return frozenset(names)

# Opt-in null-space regularization for redundant clavicle rotations.  The
# residuals are in radians while position residuals are centimetres: kc=1.0
# therefore makes one radian of unobservable rotation cost one centimetre-
# equivalent, while kr=0.05 is only a weak analytic-init/rest anchor.
REDUNDANT_DOF_PRIOR_MODES = {"none", "minimal"}
REDUNDANT_DOF_PRIOR_KC = 1.0
REDUNDANT_DOF_PRIOR_KR = 0.05
TWIST_SOFT_BAND_DEG = 2.0
TWIST_SOFT_BAND_RAD = float(np.deg2rad(TWIST_SOFT_BAND_DEG))
# Position residuals are centimetres.  The soft twist residual is radians,
# so 1 radian must be scaled by 1/(1 degree in radians) to make one degree
# of excess equal one centimetre-equivalent residual.
TWIST_SOFT_K = float(1.0 / np.deg2rad(1.0))


def soft_twist_hinge(excess_rad):
    """C1 quadratic blend of ReLU over the two-degree pre-limit band."""
    e = np.asarray(excess_rad, dtype=np.float64)
    d = TWIST_SOFT_BAND_RAD
    return np.where(
        e <= -d,
        0.0,
        np.where(e < 0.0, (e + d) ** 2 / (2.0 * d), e + d / 2.0),
    )


def limb_twist_pin_multiplier(target_positions: np.ndarray, p: int, c: int,
                               g: int | None) -> tuple[float, float | None]:
    """Return fade multiplier and bend angle from conditioned target points."""
    if g is None:
        return 1.0, None
    bone = np.asarray(target_positions[c], dtype=np.float64) - np.asarray(target_positions[p], dtype=np.float64)
    child = np.asarray(target_positions[g], dtype=np.float64) - np.asarray(target_positions[c], dtype=np.float64)
    if (not np.isfinite(bone).all() or not np.isfinite(child).all() or
            np.linalg.norm(bone) <= 1e-5 or np.linalg.norm(child) <= 1e-5):
        return 1.0, None
    theta = float(np.degrees(np.arccos(np.clip(
        np.dot(bone, child) / (np.linalg.norm(bone) * np.linalg.norm(child)),
        -1.0, 1.0))))
    if theta <= LIMB_TWIST_PIN_THETA_LO_DEG + 1e-12:
        return 1.0, theta
    if theta >= LIMB_TWIST_PIN_THETA_HI_DEG - 1e-12:
        return 0.0, theta
    u = (theta - LIMB_TWIST_PIN_THETA_LO_DEG) / (
        LIMB_TWIST_PIN_THETA_HI_DEG - LIMB_TWIST_PIN_THETA_LO_DEG
    )
    smooth = 3.0 * u * u - 2.0 * u * u * u
    return float(1.0 - smooth), theta


def _smooth_twist_sequence(values: np.ndarray, weights: np.ndarray,
                           fps: float, cutoff_hz: float = 3.0) -> np.ndarray:
    """Fit a wrapped/unwrapped twist sequence with a second-difference penalty.

    The objective is ``sum(w*(x-y)^2) + lambda*sum(D2(x)^2)``.  With unit
    observation weight, the dimensionless lambda is selected from the
    continuum fourth-order smoother approximation, ``fc = fps/(2*pi) *
    (w/lambda)**(1/4)``.  Thus ``lambda = (fps/(2*pi*fc))**4`` gives the
    requested approximately 3 Hz -3 dB cutoff at the supplied FPS.
    """
    y = np.asarray(values, dtype=np.float64).reshape(-1)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if y.size != w.size:
        raise ValueError("twist values and weights must have equal length")
    n = y.size
    if n <= 1:
        return y.copy()
    if not np.isfinite(y).all() or not np.isfinite(w).all():
        raise ValueError("twist smoothing inputs must be finite")
    w = np.maximum(w, 0.0)
    lam = (float(fps) / (2.0 * np.pi * float(cutoff_hz))) ** 4
    A = np.diag(w)
    if n >= 3:
        D2 = np.zeros((n - 2, n), dtype=np.float64)
        rows = np.arange(n - 2)
        D2[rows, rows] = 1.0
        D2[rows, rows + 1] = -2.0
        D2[rows, rows + 2] = 1.0
        A += lam * (D2.T @ D2)
    # A single observation leaves the affine nullspace of D2 underdetermined.
    # This vanishingly small ridge selects the minimum-norm deterministic fit
    # without changing observable values at practical precision.
    A.flat[:: n + 1] += 1e-12
    b = w * y
    return np.linalg.solve(A, b)


def _unwrap_observable(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Unwrap only observed angles, preserving the supplied gap mask."""
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    out = np.zeros_like(values)
    observed = weights > 1e-8
    if np.any(observed):
        out[observed] = np.unwrap(values[observed])
    return out


# Opt-in FK landmark foot model. The default remains the historical proxy
# residual and must stay bit-identical. Offsets are expressed in the solved
# proxy bone's global frame and are calibrated per clip.
FOOT_LANDMARK_MODELS = {"proxy", "fk_offsets"}
FOOT_LANDMARKS = ("heel", "big_toe", "small_toe")
FOOT_LANDMARK_HUBER_DELTA_CM = 2.0

from aimocap.retarget.mocap_skeleton import MocapSkeleton, COCO
from aimocap.retarget.foot_target_gate import (
    assess_foot_target,
    estimate_foot_gate_references,
)

class MocapIKSolver:
    def __init__(self, skel: MocapSkeleton, target_sequence: dict | None = None,
                 foot_landmark_model: str = "proxy",
                 limb_twist_anchor: str = "init",
                 two_pass_limbs: str = "all",
                 smoothed_limbs: str = "all",
                 two_pass_pass1_stride: int = 1,
                 two_pass_pass1_max_iter: int | None = None,
                 twist_limit_mode: str = "clamp",
                 soft_limit_joints=None,
                 limb_twist_pin: str = "always",
                 head_target_mode: str = "legacy",
                 redundant_dof_prior: str = "none",
                 foot_landmark_weight: float = 1.0,
                 face_position_weight: float = 1.0,
                 ankle_twist_limit_deg: float | None = None,
                 limit_projection: str = "post",
                 foot_gate_reference: str = "solved",
                 limb_hinge: str = "off",
                 fps: float | None = None):
        if head_target_mode not in HEAD_TARGET_MODES:
            raise ValueError(
                "head_target_mode must be 'legacy', 'fixed', or 'calibrated', "
                f"got {head_target_mode!r}"
            )
        if limb_twist_pin not in LIMB_TWIST_PIN_MODES:
            raise ValueError(
                "limb_twist_pin must be 'always', 'fade', or 'off', "
                f"got {limb_twist_pin!r}"
            )
        if twist_limit_mode not in TWIST_LIMIT_MODES:
            raise ValueError(
                "twist_limit_mode must be 'clamp', 'soft', 'hybrid', or 'off', "
                f"got {twist_limit_mode!r}"
            )
        if limit_projection not in LIMIT_PROJECTION_MODES:
            raise ValueError(
                "limit_projection must be 'post' or 'projected', "
                f"got {limit_projection!r}"
            )
        if foot_gate_reference not in FOOT_GATE_REFERENCE_MODES:
            raise ValueError(
                "foot_gate_reference must be 'solved' or 'observed', "
                f"got {foot_gate_reference!r}"
            )
        if limb_hinge not in {"off", "elbow_knee"}:
            raise ValueError(
                "limb_hinge must be 'off' or 'elbow_knee', "
                f"got {limb_hinge!r}"
            )
        soft_limit_joints = parse_soft_limit_joints(soft_limit_joints)
        if redundant_dof_prior not in REDUNDANT_DOF_PRIOR_MODES:
            raise ValueError(
                "redundant_dof_prior must be 'none' or 'minimal', "
                f"got {redundant_dof_prior!r}"
            )
        if limb_twist_anchor not in LIMB_TWIST_ANCHOR_MODES:
            raise ValueError(
                "limb_twist_anchor must be 'init', 'continuous', "
                "'continuous_resid', 'smoothed', or 'two_pass', got "
                f"{limb_twist_anchor!r}"
            )
        if two_pass_limbs not in TWO_PASS_LIMB_MODES:
            raise ValueError(
                "two_pass_limbs must be 'all' or 'arms', got "
                f"{two_pass_limbs!r}"
            )
        if smoothed_limbs not in SMOOTHED_LIMB_MODES:
            raise ValueError(
                "smoothed_limbs must be 'all' or 'arms', got "
                f"{smoothed_limbs!r}"
            )
        if not isinstance(two_pass_pass1_stride, (int, np.integer)) or isinstance(two_pass_pass1_stride, bool) or two_pass_pass1_stride < 1:
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
        if foot_landmark_model not in FOOT_LANDMARK_MODELS:
            raise ValueError(
                f"foot_landmark_model must be 'proxy' or 'fk_offsets', got {foot_landmark_model!r}"
            )
        if not np.isfinite(foot_landmark_weight) or foot_landmark_weight < 0.0:
            raise ValueError(
                "foot_landmark_weight must be a finite non-negative float, "
                f"got {foot_landmark_weight!r}"
            )
        if not np.isfinite(face_position_weight) or face_position_weight < 0.0:
            raise ValueError(
                "face_position_weight must be a finite non-negative float, "
                f"got {face_position_weight!r}"
            )
        if ankle_twist_limit_deg is not None:
            if (isinstance(ankle_twist_limit_deg, bool)
                    or not np.isfinite(ankle_twist_limit_deg)
                    or ankle_twist_limit_deg < 0.0):
                raise ValueError(
                    "ankle_twist_limit_deg must be None or a finite non-negative float, "
                    f"got {ankle_twist_limit_deg!r}"
                )
        self.skel = skel
        self.head_target_mode = head_target_mode
        self.num_joints = skel.num_joints
        self.foot_landmark_model = foot_landmark_model
        self.limb_twist_anchor = limb_twist_anchor
        self.two_pass_limbs = two_pass_limbs
        self.smoothed_limbs = smoothed_limbs
        self.two_pass_pass1_stride = int(two_pass_pass1_stride)
        self.two_pass_pass1_max_iter = (None if two_pass_pass1_max_iter is None
                                        else int(two_pass_pass1_max_iter))
        self.limb_twist_pin = limb_twist_pin
        self.twist_limit_mode = twist_limit_mode
        self.limit_projection = limit_projection
        self.soft_limit_joints = soft_limit_joints
        self.redundant_dof_prior = redundant_dof_prior
        self.foot_landmark_weight = float(foot_landmark_weight)
        self.face_position_weight = float(face_position_weight)
        self.ankle_twist_limit_deg = (None if ankle_twist_limit_deg is None
                                      else float(ankle_twist_limit_deg))
        self.foot_gate_reference = foot_gate_reference
        self.limb_hinge = limb_hinge
        self._fps = fps
        self._limb_twist_pin_frames: dict[tuple[str, int], tuple[float, float]] = {}
        self._last_backend_stats = {}
        self._last_solve_stats = {}
        self._limb_twist_targets: dict[tuple[int, int, int], float] = {}
        self._two_pass_phase = 1
        self._two_pass_targets: dict[tuple[int, int, int], float] = {}
        self._two_pass_anchor_diagnostics: dict = {}
        self._limb_twist_anchor_diagnostics: dict = {
            "mode": limb_twist_anchor, "fps": None, "cutoff_hz": 3.0,
            "lambda": None, "measure": "signed_twist_angle", "limbs": {},
        }
        self._foot_landmark_offsets: dict[str, np.ndarray] = {}
        self._foot_landmark_calibration: dict = {
            "requested_model": foot_landmark_model,
            "effective_model": "proxy",
            "status": "disabled" if foot_landmark_model == "proxy" else "not_calibrated",
        }
        self._foot_landmark_confidence = None

        # 3 root translation + (num_joints * 3) rotation vectors
        self.num_vars = 3 + self.num_joints * 3

        self.parents = np.array(skel.parents)
        self.rest_offsets = skel.rest_offsets.copy()

        depths = {0: 0}
        for i in range(1, self.num_joints):
            depths[i] = depths[self.parents[i]] + 1

        max_depth = max(depths.values())
        self.levels = []
        for d in range(1, max_depth + 1):
            nodes_at_d = [i for i in range(1, self.num_joints) if depths[i] == d]
            self.levels.append(np.array(nodes_at_d))

        # ── Swing-twist twist clamping setup ───────────────────────────
        # Precompute which joints have twist limits and their bone axes.
        # The bone axis is the rest offset direction (the bone's
        # longitudinal axis in parent space), which is what twist
        # rotates around.
        from aimocap.retarget.swing_twist import TWIST_LIMITS
        self._twist_limits = {}
        self._twist_joint_indices = {}
        self._twist_bone_axes = {}
        for jname, max_tw in TWIST_LIMITS.items():
            ji = skel.name_to_idx.get(jname)
            if ji is not None and ji < self.num_joints:
                axis = self.rest_offsets[ji].copy()
                n = np.linalg.norm(axis)
                if n > 1e-6:
                    if (self.ankle_twist_limit_deg is not None
                            and jname in {"ankle_l", "ankle_r"}):
                        max_tw = self.ankle_twist_limit_deg
                    axis = axis / n
                    self._twist_limits[jname] = max_tw
                    self._twist_joint_indices[jname] = ji
                    self._twist_bone_axes[jname] = axis

        # ── Axial-roll bounds for dedicated twist DOFs (VQ-FIX2) ────────
        # AXIAL_LIMITS (forearm pronation / wrist articulation) were
        # previously documented in swing_twist.py as "enforced via the
        # forearm roll residual" but never wired anywhere (dead spec).
        # Wire them into the SAME per-accepted-step clamp set as
        # TWIST_LIMITS so each joint's axial roll stays inside its
        # anatomical range (elbow <=100 deg pronation, wrist <=25 deg).
        # "child:X" resolves the axis from the child bone's rest offset
        # (e.g. the elbow's roll axis IS the forearm direction).
        from aimocap.retarget.swing_twist import AXIAL_LIMITS
        for jname, (max_deg, axis_src) in AXIAL_LIMITS.items():
            if jname in self._twist_limits:
                continue
            ji = skel.name_to_idx.get(jname)
            if ji is None or ji >= self.num_joints:
                continue
            if axis_src == "self":
                axis = self.rest_offsets[ji].copy()
            elif axis_src.startswith("child:"):
                child_name = axis_src.split(":", 1)[1]
                ci = skel.name_to_idx.get(child_name)
                if ci is None or self.parents[ci] != ji:
                    continue
                axis = self.rest_offsets[ci].copy()
            else:
                continue
            n = np.linalg.norm(axis)
            if n > 1e-6:
                self._twist_limits[jname] = max_deg
                self._twist_joint_indices[jname] = ji
                self._twist_bone_axes[jname] = axis / n

        self._twist_limit_counts = {
            name: {"frames": 0, "at_or_over_limit": 0, "at_safety_clamp": 0}
            for name in self._twist_limits
        }
        self._twist_limit_total_frames = 0

        # ── Limb hinge constraints ────────────────────────────────────
        self._hinge_joints = []
        if self.limb_hinge != "off":
            hinge_axes = self._limb_hinge_axes()
            from math import radians
            for side in ("l", "r"):
                for joint_name in [f"knee_{side}", f"elbow_{side}"]:
                    if joint_name in hinge_axes:
                        axis, _ = hinge_axes[joint_name]
                        idx = self.skel.name_to_idx.get(joint_name)
                        if idx is not None:
                            self._hinge_joints.append({
                                "joint_idx": idx,
                                "axis": axis,
                                "flex_min_rad": radians(-5),
                                "flex_max_rad": radians(160),
                            })

        # ── Rest head frame for frame-based head orientation ──────────
        # Built from clavicle positions (rig-agnostic). Left-handed formula
        # (left x up = forward) matches the data's chirality, then forced
        # to right-handed by flipping the up column. This ensures the delta
        # F_obs @ F_rest.T is a proper rotation.
        fbx_gp, _ = skel._fbx_skel.get_forward_kinematics()
        cl_name = skel.fbx_mapping.get(skel.name_to_idx.get("clavicle_l"))
        cr_name = skel.fbx_mapping.get(skel.name_to_idx.get("clavicle_r"))
        if cl_name and cr_name:
            cl_fbx = skel._fbx_skel.name_to_idx[cl_name]
            cr_fbx = skel._fbx_skel.name_to_idx[cr_name]
            x0 = fbx_gp[cl_fbx] - fbx_gp[cr_fbx]
        else:
            x0 = np.array([1.0, 0.0, 0.0])
        x0 = x0 / (np.linalg.norm(x0) + 1e-9)
        u0 = np.array([0.0, 0.0, 1.0])
        f0 = np.cross(x0, u0)
        f0 = f0 / (np.linalg.norm(f0) + 1e-9)
        u0 = np.cross(f0, x0)
        u0 = u0 / (np.linalg.norm(u0) + 1e-9)
        self._F_rest_head = np.column_stack([x0, f0, u0])
        self._F_rest_head[:, 2] *= -1  # force right-handed

        # Fixed opt-in rest face frame. The observed frame uses the same
        # landmark convention as the deterministic face offsets used by the
        # GT harness: left ear, right ear, nose, and both eyes in the head's
        # local rest coordinates. The historical frame above instead uses
        # clavicles, which is a different coordinate frame and conjugates
        # head-axis rotations. Keep both frames so legacy remains bit-exact.
        left_ear0 = np.array([-7.2, -5.0, 3.0], dtype=np.float64)
        right_ear0 = np.array([7.2, -5.0, 3.0], dtype=np.float64)
        face_points0 = np.array([
            [0.0, -10.0, 5.0],
            [-3.2, -9.2, 6.0],
            [3.2, -9.2, 6.0],
        ], dtype=np.float64)
        x_face0 = left_ear0 - right_ear0
        x_face0 /= np.linalg.norm(x_face0)
        f_face0 = face_points0.mean(axis=0) - (left_ear0 + right_ear0) / 2.0
        f_face0 -= np.dot(f_face0, x_face0) * x_face0
        f_face0 /= np.linalg.norm(f_face0)
        u_face0 = np.cross(f_face0, x_face0)
        u_face0 /= np.linalg.norm(u_face0)
        F_face_local0 = np.column_stack([x_face0, f_face0, -u_face0])
        head_proxy_i = skel.name_to_idx.get("head")
        head_fbx_name = skel.fbx_mapping.get(head_proxy_i)
        if head_fbx_name is not None:
            head_fbx_i = skel._fbx_skel.name_to_idx[head_fbx_name]
            _, fbx_rest_q = skel._fbx_skel.get_forward_kinematics()
            R_head_rest = Rotation.from_quat(fbx_rest_q[head_fbx_i]).as_matrix()
        else:
            R_head_rest = np.eye(3)
        self._F_rest_head_fixed = R_head_rest @ F_face_local0
        self._head_calibration = {
            "status": "not_requested",
            "method": "none",
            "candidate_frames": 0,
            "selected_frames": 0,
            "total_frames": 0,
            "trimmed_frames": 0,
            "offset_matrix": Rotation.identity().as_matrix().tolist(),
            "offset_yaw_pitch_roll_deg_zyx": [0.0, 0.0, 0.0],
        }
        self._head_offset = Rotation.identity()
        if self.head_target_mode == "calibrated":
            self._calibrate_head_offset(target_sequence)

        (self._foot_gate_reference_lengths,
         self._foot_gate_reference_modes,
         self._foot_gate_observed_masks) = estimate_foot_gate_references(
             skel, target_sequence)
        # Keyed by (zero-based frame, side), so a retried precompute replaces
        # its old record instead of duplicating provenance.
        self._foot_gate_records: dict[tuple[int, str], dict] = {}
        
        # For foot_gate_reference='observed' mode: track last accepted observations
        self._foot_gate_last_accepted: dict[str, tuple[int, np.ndarray]] = {}  # side -> (frame_idx, direction)


    @staticmethod
    def _face_frame_from_points(measured: dict) -> np.ndarray | None:
        """Return the RH observed face frame, or None for invalid landmarks."""
        required = ("left_ear", "right_ear", "nose")
        if not all(k in measured and np.all(np.isfinite(measured[k])) for k in required):
            return None
        la = np.asarray(measured["left_ear"], dtype=np.float64)
        ra = np.asarray(measured["right_ear"], dtype=np.float64)
        face_pts = [np.asarray(measured[k], dtype=np.float64)
                    for k in ("nose", "left_eye", "right_eye")
                    if k in measured and np.all(np.isfinite(measured[k]))]
        if len(face_pts) < 2:
            return None
        ear_delta = la - ra
        ear_norm = np.linalg.norm(ear_delta)
        if not np.isfinite(ear_norm) or ear_norm <= 5.0:
            return None
        x = ear_delta / ear_norm
        ear_mid = (la + ra) / 2.0
        f_raw = np.mean(face_pts, axis=0) - ear_mid
        f = f_raw - np.dot(f_raw, x) * x
        f_norm = np.linalg.norm(f)
        if not np.isfinite(f_norm) or f_norm <= 1e-5:
            return None
        f /= f_norm
        u = np.cross(f, x)
        u_norm = np.linalg.norm(u)
        if not np.isfinite(u_norm) or u_norm <= 1e-5:
            return None
        u /= u_norm
        frame = np.column_stack([x, f, u])
        frame[:, 2] *= -1.0
        return frame if np.linalg.det(frame) > 0.0 else None

    @staticmethod
    def _torso_frame_from_points(measured: dict) -> np.ndarray | None:
        """Build the same RH basis convention for the torso as for the face."""
        keys = ("shoulder_l", "shoulder_r", "neck", "pelvis")
        if not all(k in measured and np.all(np.isfinite(measured[k])) for k in keys):
            return None
        x_raw = np.asarray(measured["shoulder_l"], dtype=np.float64) - np.asarray(
            measured["shoulder_r"], dtype=np.float64)
        up_raw = np.asarray(measured["neck"], dtype=np.float64) - np.asarray(
            measured["pelvis"], dtype=np.float64)
        x_norm = np.linalg.norm(x_raw)
        up_norm = np.linalg.norm(up_raw)
        if x_norm <= 1e-5 or up_norm <= 1e-5:
            return None
        x = x_raw / x_norm
        up = up_raw / up_norm
        f = np.cross(x, up)
        f_norm = np.linalg.norm(f)
        if f_norm <= 1e-5:
            return None
        f /= f_norm
        up = np.cross(f, x)
        up /= np.linalg.norm(up)
        frame = np.column_stack([x, f, up])
        frame[:, 2] *= -1.0
        return frame if np.linalg.det(frame) > 0.0 else None

    def _calibrate_head_offset(self, target_sequence: dict | None) -> dict:
        """Estimate face-frame-to-torso-frame offset once per clip.

        For neutral-looking frames, R_rel = R_torso.T @ F_obs should be a
        constant anatomical face offset.  The robust mean below is the
        rotation-space mean after a deterministic median/MAD outlier trim.
        """
        identity = Rotation.identity()
        result = {
            "status": "unavailable",
            "method": "robust_rotation_mean_upright_forward",
            "formula": "R_offset = mean(R_torso.T @ F_obs); F_cal = F_obs @ R_offset.T",
            "selection": {
                "upright_max_deg": 30.0,
                "face_forward_max_deg": 60.0,
                "min_frames": 1,
            },
            "candidate_frames": 0,
            "selected_frames": 0,
            "total_frames": 0,
            "trimmed_frames": 0,
            "offset_matrix": identity.as_matrix().tolist(),
            "offset_yaw_pitch_roll_deg_zyx": [0.0, 0.0, 0.0],
        }
        if not target_sequence:
            self._head_calibration = result
            self._head_offset = identity
            return result
        n = max((len(v) for v in target_sequence.values()), default=0)
        result["total_frames"] = int(n)
        rel_rots = []
        candidate_frames = []
        cos_upright = np.cos(np.deg2rad(30.0))
        cos_forward = np.cos(np.deg2rad(60.0))
        for frame in range(n):
            measured = {name: np.asarray(values[frame], dtype=np.float64)
                        for name, values in target_sequence.items() if frame < len(values)}
            F_obs = self._face_frame_from_points(measured)
            F_torso = self._torso_frame_from_points(measured)
            if F_obs is None or F_torso is None:
                continue
            upright = abs(float(np.dot(F_torso[:, 2], np.array([0.0, 0.0, 1.0]))))
            forward = float(np.dot(F_torso[:, 1], F_obs[:, 1]))
            if upright >= cos_upright and forward >= cos_forward:
                candidate_frames.append(frame)
                rel_rots.append(Rotation.from_matrix(F_torso.T @ F_obs))
        result["candidate_frames"] = int(len(rel_rots))
        if not rel_rots:
            self._head_calibration = result
            self._head_offset = identity
            return result
        rotations = Rotation.concatenate(rel_rots)
        center = rotations.mean()
        distances = (center.inv() * rotations).magnitude()
        med = float(np.median(distances))
        mad = float(np.median(np.abs(distances - med)))
        threshold = max(np.deg2rad(15.0), med + 3.0 * 1.4826 * mad)
        keep = distances <= threshold
        if not np.any(keep):
            keep[:] = True
        offset = Rotation.concatenate([r for r, ok in zip(rel_rots, keep) if ok]).mean()
        result["selected_frames"] = int(np.sum(keep))
        result["trimmed_frames"] = int(np.sum(~keep))
        result["status"] = "calibrated"
        result["offset_matrix"] = offset.as_matrix().tolist()
        result["offset_yaw_pitch_roll_deg_zyx"] = offset.as_euler("zyx", degrees=True).tolist()
        result["trim_threshold_deg"] = float(np.degrees(threshold))
        self._head_calibration = result
        self._head_offset = offset
        return result

    def head_calibration_diagnostics(self) -> dict:
        """Return JSON-safe provenance for the calibrated head offset."""
        return dict(self._head_calibration)

    def _head_target_rotation(self, F_obs: np.ndarray) -> Rotation:
        """Map an observed face frame to the selected global head target."""
        if self.head_target_mode == "calibrated":
            # Remove the clip-constant anatomical face offset, then express
            # the corrected torso-relative frame in the rig torso rest frame.
            return Rotation.from_matrix(
                F_obs @ self._head_offset.inv().as_matrix() @ self._F_rest_head.T)
        rest_frame = (self._F_rest_head_fixed if self.head_target_mode == "fixed"
                      else self._F_rest_head)
        return Rotation.from_matrix(F_obs @ rest_frame.T)

    def foot_gate_diagnostics(self) -> dict:
        """Stable per-run observability-gate provenance for profile/audit output."""
        records = [self._foot_gate_records[k] for k in sorted(self._foot_gate_records)]
        counts = {side: sum(row["side"] == side for row in records) for side in ("l", "r")}
        condition_counts: dict[str, int] = {}
        for row in records:
            for condition in row["failed_conditions"]:
                condition_counts[condition] = condition_counts.get(condition, 0) + 1
        return {
            "reference_length_cm": dict(self._foot_gate_reference_lengths),
            "reference_mode": dict(self._foot_gate_reference_modes),
            "gated_frame_count_total": len(records),
            "gated_unique_frame_count": len({row["frame"] for row in records}),
            "gated_frame_count_by_side": counts,
            "failed_condition_counts": condition_counts,
            "gated_frames": records,
        }

    def _limb_pin_specs(self):
        """Return the eight supported proxy limb pins and their roll children."""
        primary_child = {}
        for j in range(1, self.num_joints):
            parent = int(self.parents[j])
            if parent not in primary_child:
                primary_child[parent] = j
        specs = []
        for side in ("l", "r"):
            for p_name, c_name in (
                (f"hip_{side}", f"knee_{side}"),
                (f"knee_{side}", f"ankle_{side}"),
                (f"shoulder_{side}", f"elbow_{side}"),
                (f"elbow_{side}", f"wrist_{side}"),
            ):
                p = self.skel.name_to_idx.get(p_name)
                c = self.skel.name_to_idx.get(c_name)
                if p is None or c is None:
                    continue
                g = primary_child.get(c)
                bone = self.rest_offsets[c].copy()
                norm = np.linalg.norm(bone)
                if norm <= 1e-12:
                    continue
                specs.append((side, int(p), int(c), int(g) if g is not None else None, bone / norm))
        return specs

    def _limb_hinge_axes(self):
        """Compute generic hinge axes for elbows and knees from proxy rest data only.
        
        Returns:
            dict: {
                'knee_l': (axis, sign_ok), 
                'knee_r': (axis, sign_ok),
                'elbow_l': (axis, sign_ok), 
                'elbow_r': (axis, sign_ok)
            }
            where axis is a unit 3-vector, sign_ok is True if +30 deg about axis
            moves child in anatomical flexion direction.
        """
        # World up vector
        up = np.array([0.0, 0.0, 1.0])
        
        # Lateral direction from hip positions
        hip_l = self.skel.name_to_idx.get("hip_l")
        hip_r = self.skel.name_to_idx.get("hip_r")
        lateral_lr = self.skel.rest_t[hip_r] - self.skel.rest_t[hip_l]
        lateral_lr = lateral_lr / (np.linalg.norm(lateral_lr) + 1e-12)
        
        # Forward direction (perpendicular to up and lateral)
        forward = np.cross(up, lateral_lr)
        forward = forward / (np.linalg.norm(forward) + 1e-12)
        
        # Check if forward points from ankle to toe (to ensure consistent orientation)
        ankle_l = self.skel.name_to_idx.get("ankle_l")
        toe_l = self.skel.name_to_idx.get("toe_l")
        if ankle_l is not None and toe_l is not None:
            foot_vec = self.skel.rest_t[toe_l] - self.skel.rest_t[ankle_l]
            if np.dot(foot_vec, forward) < 0:
                forward = -forward
        
        results = {}
        
        # Knee axes
        for side in ("l", "r"):
            knee_name = f"knee_{side}"
            ankle_name = f"ankle_{side}"
            
            knee_idx = self.skel.name_to_idx.get(knee_name)
            ankle_idx = self.skel.name_to_idx.get(ankle_name)
            
            if knee_idx is None or ankle_idx is None:
                continue
                
            # Shank direction (knee->ankle)
            shin_hat = self.rest_offsets[ankle_idx].copy()
            shin_norm = np.linalg.norm(shin_hat)
            if shin_norm <= 1e-12:
                continue
            shin_hat = shin_hat / shin_norm
            
            # Knee axis: lateral axis made perpendicular to shin
            lateral_component = np.dot(lateral_lr, shin_hat)
            h = lateral_lr - lateral_component * shin_hat
            h_norm = np.linalg.norm(h)
            if h_norm <= 1e-12:
                continue
            h = h / h_norm
            
            # Sign check: +30 deg rotation should move ankle backward relative to forward direction
            # Rotate shin vector about axis h by +30 deg
            from scipy.spatial.transform import Rotation
            rot_30 = Rotation.from_rotvec(h * np.deg2rad(30.0))
            child_offset = self.rest_offsets[ankle_idx].copy()
            child_rotated = rot_30.apply(child_offset)
            
            # For knee flexion, ankle should move backward (negative forward component)
            delta = child_rotated - child_offset
            flexion_sign_ok = np.dot(delta, forward) < 0
            
            # If sign check fails, flip the axis
            if not flexion_sign_ok:
                h = -h
                flexion_sign_ok = True
            
            results[knee_name] = (h, flexion_sign_ok)
        
        # Elbow axes  
        for side in ("l", "r"):
            elbow_name = f"elbow_{side}"
            wrist_name = f"wrist_{side}"
            
            elbow_idx = self.skel.name_to_idx.get(elbow_name)
            wrist_idx = self.skel.name_to_idx.get(wrist_name)
            
            if elbow_idx is None or wrist_idx is None:
                continue
                
            # Forearm direction (elbow->wrist)
            forearm_hat = self.rest_offsets[wrist_idx].copy()
            forearm_norm = np.linalg.norm(forearm_hat)
            if forearm_norm <= 1e-12:
                continue
            forearm_hat = forearm_hat / forearm_norm
            
            # Elbow axis: cross(forearm, forward) then orthogonalize to forearm
            h = np.cross(forearm_hat, forward)
            h_norm = np.linalg.norm(h)
            if h_norm <= 1e-12:
                continue
            h = h / h_norm
            
            # Ensure perpendicular to forearm
            f_component = np.dot(h, forearm_hat)
            h = h - f_component * forearm_hat
            h_norm = np.linalg.norm(h)
            if h_norm <= 1e-12:
                continue
            h = h / h_norm
            
            # Sign check: +30 deg rotation should move wrist forward relative to forward direction
            # For an arm hanging down, this is lateral axis; for A-pose it tilts with arm
            from scipy.spatial.transform import Rotation
            rot_30 = Rotation.from_rotvec(h * np.deg2rad(30.0))
            child_offset = self.rest_offsets[wrist_idx].copy()
            child_rotated = rot_30.apply(child_offset)
            
            # For elbow flexion, wrist should move forward (positive forward component)
            delta = child_rotated - child_offset
            flexion_sign_ok = np.dot(delta, forward) > 0
            
            # If sign check fails, flip the axis
            if not flexion_sign_ok:
                h = -h
                flexion_sign_ok = True
            
            results[elbow_name] = (h, flexion_sign_ok)
        
        return results

    def _limb_twist_observation(self, measured: dict, init_x: np.ndarray,
                                spec):
        """Compute one data twist using the existing bend-plane construction."""
        from aimocap.retarget.swing_twist import bend_plane_rotation
        side, p, c, g, bone = spec
        if g is None:
            return 0.0, 0.0, None
        init_root, init_local = self._state_to_local_rotations(init_x)
        _, init_global_q = self.forward_kinematics(init_root, init_local)
        init_global = Rotation.from_quat(init_global_q)
        parent_global = init_global[int(self.parents[p])]
        bone_world = np.asarray(measured[self.skel.joint_names[c]], dtype=np.float64) - np.asarray(
            measured[self.skel.joint_names[p]], dtype=np.float64)
        child_world = np.asarray(measured[self.skel.joint_names[g]], dtype=np.float64) - np.asarray(
            measured[self.skel.joint_names[c]], dtype=np.float64)
        if (not np.isfinite(bone_world).all() or not np.isfinite(child_world).all()):
            return 0.0, 0.0, None
        bn = float(np.linalg.norm(bone_world))
        gn = float(np.linalg.norm(child_world))
        if bn <= 1e-5 or gn <= 1e-5:
            return 0.0, 0.0, None
        bone_dir = bone_world / bn
        child_dir = child_world / gn
        theta = float(np.degrees(np.arccos(np.clip(np.dot(bone_dir, child_dir), -1.0, 1.0))))
        u = np.clip((theta - LIMB_TWIST_OBSERVABILITY_LOW_DEG) /
                    (LIMB_TWIST_OBSERVABILITY_HIGH_DEG -
                     LIMB_TWIST_OBSERVABILITY_LOW_DEG), 0.0, 1.0)
        observability = float(3.0 * u * u - 2.0 * u * u * u)
        desired_pl = parent_global.inv().apply(bone_world)
        child_desired_pl = parent_global.inv().apply(child_world)
        bend = bend_plane_rotation(
            self.rest_offsets[c], desired_pl,
            roll_child_rest=self.rest_offsets[g],
            roll_child_desired=child_desired_pl,
        )
        if bend is None:
            return 0.0, 0.0, theta
        # The smoothed path and its residual use signed_twist_angle.  Existing
        # init/continuous modes retain their historical measures and paths.
        from aimocap.retarget.swing_twist import signed_twist_angle
        return float(signed_twist_angle(bend, bone)), observability, theta

    def prepare_limb_twist_anchor(self, target_sequence: dict, fps: float,
                                  frame_count: int | None = None) -> dict:
        """Precompute solver-independent smoothed twist targets once per clip."""
        self._limb_twist_targets = {}
        self._limb_twist_anchor_diagnostics = {
            "mode": self.limb_twist_anchor,
            "smoothed_limbs": self.smoothed_limbs,
            "fps": float(fps),
            "cutoff_hz": 3.0,
            "lambda": float((float(fps) / (2.0 * np.pi * 3.0)) ** 4),
            "measure": "signed_twist_angle",
            "limbs": {},
        }
        if self.limb_twist_anchor != "smoothed":
            return self._limb_twist_anchor_diagnostics
        n = int(frame_count if frame_count is not None else max(
            (len(v) for v in target_sequence.values()), default=0))
        specs = self._limb_pin_specs()
        for side, p, c, g, bone in specs:
            key = (p, c)
            init_values = np.zeros(n, dtype=np.float64)
            data_values = np.zeros(n, dtype=np.float64)
            obs_values = np.zeros(n, dtype=np.float64)
            for frame in range(n):
                measured = {name: np.asarray(values[frame], dtype=np.float64)
                            for name, values in target_sequence.items()
                            if frame < len(values)}
                try:
                    init_x = self.analytic_init(measured)
                    rv = init_x[3 + p * 3:3 + p * 3 + 3]
                    from aimocap.retarget.swing_twist import signed_twist_angle
                    init_values[frame] = (float(signed_twist_angle(
                        Rotation.from_rotvec(rv), bone)) if np.isfinite(rv).all() else 0.0)
                    tau, obs, theta = self._limb_twist_observation(
                        measured, init_x, (side, p, c, g, bone))
                    data_values[frame] = tau
                    obs_values[frame] = obs if np.isfinite(tau) else 0.0
                except (KeyError, ValueError, FloatingPointError):
                    init_values[frame] = init_values[frame - 1] if frame else 0.0
                    data_values[frame] = 0.0
                    obs_values[frame] = 0.0
            if np.any(obs_values > 1e-8):
                unwrapped = _unwrap_observable(data_values, obs_values)
                smoothed = _smooth_twist_sequence(unwrapped, obs_values, fps)
            else:
                # No data-supported bend plane: preserve the old init anchor,
                # but remove its branch-switch discontinuity with the same fit.
                smoothed = _smooth_twist_sequence(init_values,
                                                   np.ones(n, dtype=np.float64), fps)
            for frame, value in enumerate(smoothed):
                self._limb_twist_targets[(frame, p, c)] = float(value)
            name = f"{side}:{p}->{c}"
            steps = np.diff(smoothed) if len(smoothed) > 1 else np.zeros(0)
            self._limb_twist_anchor_diagnostics["limbs"][name] = {
                "p": int(p), "c": int(c),
                "observable_fraction": float(np.mean(obs_values > 1e-8)) if n else 0.0,
                "max_target_step_deg": float(np.degrees(np.max(np.abs(steps)))) if steps.size else 0.0,
                "target_deg": np.degrees(smoothed).tolist(),
                "observability": obs_values.tolist(),
            }
        return self._limb_twist_anchor_diagnostics

    def limb_twist_anchor_diagnostics(self) -> dict:
        """Return stable provenance for the optional offline anchor."""
        return dict(self._limb_twist_anchor_diagnostics)


    def limb_twist_pin_diagnostics(self) -> dict:
        """Summarize fade occupancy and target steps for profile provenance."""
        by_limb = {}
        grouped = {}
        for (name, frame), (multiplier, target) in self._limb_twist_pin_frames.items():
            grouped.setdefault(name, []).append((int(frame), float(multiplier), float(target)))
        for name, rows in sorted(grouped.items()):
            rows.sort(key=lambda row: row[0])
            multipliers = np.asarray([row[1] for row in rows], dtype=np.float64)
            targets = np.asarray([row[2] for row in rows], dtype=np.float64)
            eligible = multipliers > 0.01
            steps = np.abs(np.diff(targets))
            step_mask = eligible[1:] if len(eligible) > 1 else np.zeros(0, dtype=bool)
            max_step = float(np.max(steps[step_mask])) if np.any(step_mask) else 0.0
            by_limb[name] = {
                "frames": int(len(rows)),
                "fraction_m_gt_0_5": float(np.mean(multipliers > 0.5)) if len(rows) else 0.0,

                "max_target_step_deg_when_m_gt_0_01": float(np.degrees(max_step)),
                "theta_lo_deg": LIMB_TWIST_PIN_THETA_LO_DEG,
                "theta_hi_deg": LIMB_TWIST_PIN_THETA_HI_DEG,
            }
        return {
            "mode": self.limb_twist_pin,
            "formula": "m=1-smoothstep(theta; theta_hi-20deg, theta_hi), theta=angle(target[p]->target[c], target[c]->target[g])",
            "theta_lo_deg": LIMB_TWIST_PIN_THETA_LO_DEG,
            "theta_hi_deg": LIMB_TWIST_PIN_THETA_HI_DEG,
            "limbs": by_limb,
        }

    @staticmethod
    def _two_pass_term5_twist(state: np.ndarray, parents: np.ndarray,
                              forward_kinematics, bone: np.ndarray,
                              p: int) -> float:
        """Measure Term 5 exactly: rotvec(R_pp^-1 R_p) dot bone axis."""
        owner = forward_kinematics.__self__
        root_t, local_q = owner._state_to_local_rotations(state)
        _, global_q = forward_kinematics(root_t, local_q)
        parent = int(parents[p])
        R_parent_parent = Rotation.from_quat(global_q[parent])
        R_parent = Rotation.from_quat(global_q[p])
        return float((R_parent_parent.inv() * R_parent).as_rotvec().dot(bone))

    def begin_two_pass(self, phase: int) -> None:
        """Select the pass used by the opt-in solved-state twist anchor."""
        if self.limb_twist_anchor != "two_pass":
            return
        if phase not in (1, 2):
            raise ValueError(f"two-pass phase must be 1 or 2, got {phase!r}")
        if phase == 2 and not self._two_pass_targets:
            raise ValueError("two-pass phase 2 requires prepared twist targets")
        self._two_pass_phase = int(phase)

    def _two_pass_filtered_pin_enabled(self, p_name: str) -> bool:
        """Whether pass 2 should use the filtered target for this pin."""
        return self.two_pass_limbs == "all" or p_name.startswith(("shoulder_", "elbow_"))

    def _smoothed_filtered_pin_enabled(self, p_name: str) -> bool:
        """Whether the offline smoothed target applies to this pin."""
        return self.smoothed_limbs == "all" or p_name.startswith(("shoulder_", "elbow_"))

    @staticmethod
    def _zero_phase_lowpass(values: np.ndarray, fps: float,
                            cutoff_hz: float = 3.0) -> np.ndarray:
        """Butterworth order-2 zero-phase low-pass with short-clip fallback."""
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        if values.size <= 2:
            return values.copy()
        nyquist = float(fps) * 0.5
        if not np.isfinite(fps) or fps <= 2.0 * cutoff_hz:
            raise ValueError(
                f"two-pass twist filter requires fps > {2.0 * cutoff_hz:g}, got {fps!r}"
            )
        sos = butter(2, float(cutoff_hz) / nyquist, btype="low", output="sos")
        try:
            return np.asarray(sosfiltfilt(sos, values), dtype=np.float64)
        except ValueError:
            return np.asarray(sosfiltfilt(
                sos, values, padlen=values.size - 1), dtype=np.float64)

    @staticmethod
    def _high_frequency_rms(values: np.ndarray, fps: float,
                            cutoff_hz: float = 5.0) -> float:
        """Return >5 Hz angular-velocity RMS in degrees/sec for diagnostics."""
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        if values.size <= 3:
            return 0.0
        velocity = np.gradient(values) * float(fps)
        nyquist = float(fps) * 0.5
        if nyquist <= cutoff_hz:
            return 0.0
        sos = butter(2, float(cutoff_hz) / nyquist,
                     btype="highpass", output="sos")
        try:
            high = sosfiltfilt(sos, velocity)
        except ValueError:
            high = sosfiltfilt(sos, velocity, padlen=values.size - 1)
        return float(np.degrees(np.sqrt(np.mean(np.square(high)))))

    def prepare_two_pass_anchor(self, pass1_states: list[np.ndarray],
                                fps: float,
                                frame_count: int | None = None,
                                pass1_frame_indices: list[int] | np.ndarray | None = None) -> dict:
        """Build pass-2 pin targets from pass-1 swing-twist data.

        With a stride greater than one, ``pass1_states`` contains only the
        solved frames named by ``pass1_frame_indices``.  The measured twist is
        unwrapped at those samples, interpolated on the unwrapped signal, and
        only then low-passed.  The default/all-frame path intentionally keeps
        its historical operations unchanged.
        """
        if self.limb_twist_anchor != "two_pass":
            return dict(self._two_pass_anchor_diagnostics)
        states = np.asarray(pass1_states, dtype=np.float64)
        n = int(frame_count if frame_count is not None else len(states))
        if n < 1:
            raise ValueError("two-pass anchor requires at least one frame")
        if pass1_frame_indices is None:
            sample_indices = np.arange(n, dtype=np.int64)
        else:
            sample_indices = np.asarray(pass1_frame_indices, dtype=np.int64).reshape(-1)
        if states.ndim != 2 or states.shape[1] != self.num_vars:
            raise ValueError(
                f"pass1_states must have shape ({len(sample_indices)}, {self.num_vars}), got {states.shape}"
            )
        if len(sample_indices) != len(states) or len(sample_indices) < 1:
            raise ValueError("pass1_frame_indices must match pass1_states and be non-empty")
        if np.any(sample_indices < 0) or np.any(sample_indices >= n) or np.any(np.diff(sample_indices) <= 0):
            raise ValueError("pass1_frame_indices must be strictly increasing within frame_count")
        if pass1_frame_indices is None and states.shape[0] < n:
            raise ValueError(
                f"pass1_states must have shape (>= {n}, {self.num_vars}), got {states.shape}"
            )
        states = states[:len(sample_indices)]
        sampled = len(sample_indices) != n or not np.array_equal(sample_indices, np.arange(n))
        full_indices = np.arange(n, dtype=np.int64)
        self._two_pass_targets = {}
        diagnostics = {
            "mode": "two_pass",
            "two_pass_limbs": self.two_pass_limbs,
            "pass1_stride": int(sample_indices[1] - sample_indices[0]) if len(sample_indices) > 1 else 1,
            "pass1_sampled_frames": int(len(sample_indices)),
            "pass1_frame_count": int(n),
            "fps": float(fps),
            "cutoff_hz": 3.0,
            "filter": "Butterworth order 2, sosfiltfilt zero-phase",
            "high_frequency_cutoff_hz": 5.0,
            "cutoff_rationale": (
                "3 Hz is above voluntary forearm pronation/supination main energy "
                "and below the measured V8 >5 Hz vibration band "
                "(outputs/gt_harness/VIB_twist_fix.md)."
            ),
            "measure": "dot(rotvec(R_pp^-1 R_p), bone_axis), radians",
            "target_source": "signed_twist_angle(swing_twist_decompose(pass1_relative_rotation))",
            "target_reconstruction": "pass1_swing * twist(filtered_true_twist)",
            "target_residual_conversion": "dot(rotvec(reconstructed_target), bone_axis)",
            "pass1_solve_s": self._two_pass_anchor_diagnostics.get("pass1_solve_s"),
            "pass2_solve_s": None,
            "limbs": {},
        }
        from aimocap.retarget.swing_twist import (
            signed_twist_angle,
            swing_twist_decompose,
        )
        for side, p, c, g, bone in self._limb_pin_specs():
            relative_rotations = []
            legacy_measure = []
            true_twist = []
            swings = []
            for state in states:
                root_t, local_q = self._state_to_local_rotations(state)
                _, global_q = self.forward_kinematics(root_t, local_q)
                parent = int(self.parents[p])
                relative = Rotation.from_quat(global_q[parent]).inv() * Rotation.from_quat(global_q[p])
                relative_rotations.append(relative)
                legacy_measure.append(float(relative.as_rotvec().dot(bone)))
                swing, _twist = swing_twist_decompose(relative, bone)
                swings.append(swing)
                true_twist.append(float(signed_twist_angle(relative, bone)))
            legacy_sampled = np.asarray(legacy_measure, dtype=np.float64)
            twist_sampled = np.asarray(true_twist, dtype=np.float64)
            legacy_unwrapped_sampled = (
                np.unwrap(legacy_sampled) if legacy_sampled.size > 1 else legacy_sampled.copy()
            )
            twist_unwrapped_sampled = (
                np.unwrap(twist_sampled) if twist_sampled.size > 1 else twist_sampled.copy()
            )
            if sampled:
                # Interpolate after unwrapping so a +/-pi crossing follows the
                # continuous branch rather than taking the long way around.
                legacy_unwrapped = np.interp(
                    full_indices, sample_indices, legacy_unwrapped_sampled
                )
                twist_unwrapped = np.interp(
                    full_indices, sample_indices, twist_unwrapped_sampled
                )
                if len(swings) == 1:
                    swings_all = [swings[0]] * n
                else:
                    from scipy.spatial.transform import Slerp
                    swings_all = list(Slerp(
                        sample_indices.astype(np.float64),
                        Rotation.concatenate(swings),
                    )(full_indices.astype(np.float64)))
            else:
                # Keep this branch structurally identical to the historical
                # implementation for the default stride=1 contract.
                legacy_unwrapped = legacy_unwrapped_sampled
                twist_unwrapped = twist_unwrapped_sampled
                swings_all = swings
            filtered_twist = self._zero_phase_lowpass(twist_unwrapped, fps, cutoff_hz=3.0)
            # Preserve historical wrapped raw diagnostics at stride=1.  A
            # strided pass reports the interpolated unwrapped signal because
            # that is the signal actually filtered.
            if sampled:
                legacy_raw = np.asarray(legacy_unwrapped, dtype=np.float64)
                twist_raw = np.asarray(twist_unwrapped, dtype=np.float64)
            else:
                legacy_raw = legacy_sampled
                twist_raw = twist_sampled
            target_measure = np.asarray([
                float((swings_all[frame] * Rotation.from_rotvec(
                    filtered_twist[frame] * bone)).as_rotvec().dot(bone))
                for frame in range(n)
            ], dtype=np.float64)
            for frame, value in enumerate(target_measure):
                if (self.two_pass_limbs == "all"
                        or self.skel.joint_names[p].startswith(("shoulder_", "elbow_"))):
                    self._two_pass_targets[(frame, p, c)] = float(value)
            steps = np.diff(target_measure) if target_measure.size > 1 else np.zeros(0)
            twist_steps = np.diff(filtered_twist) if filtered_twist.size > 1 else np.zeros(0)
            name = f"{side}:{p}->{c}"
            diagnostics["limbs"][name] = {
                "p": int(p), "c": int(c),
                "sampled_frames": int(len(sample_indices)),
                "pass1_stride": int(sample_indices[1] - sample_indices[0]) if len(sample_indices) > 1 else 1,
                "target_max_step_deg": float(np.degrees(np.max(np.abs(steps)))) if steps.size else 0.0,
                "true_twist_max_step_deg": float(np.degrees(np.max(np.abs(twist_steps)))) if twist_steps.size else 0.0,
                "hf_before_deg_s": self._high_frequency_rms(twist_unwrapped, fps),
                "hf_after_deg_s": self._high_frequency_rms(filtered_twist, fps),
                "legacy_measure_hf_before_deg_s": self._high_frequency_rms(legacy_unwrapped, fps),
                "legacy_measure_max_abs_deg": float(np.degrees(np.max(np.abs(legacy_raw)))) if legacy_raw.size else 0.0,
                "legacy_measure_rotvec_gt_2p8_rad_count": int(np.count_nonzero(np.array([
                    np.linalg.norm(relative.as_rotvec()) > 2.8 for relative in relative_rotations
                ]))),
                "legacy_measure_step_gt_30_deg_count": int(np.count_nonzero(
                    np.abs(np.diff(legacy_raw)) > np.deg2rad(30.0)
                )) if legacy_raw.size > 1 else 0,
                "true_twist_step_gt_30_deg_count": int(np.count_nonzero(
                    np.abs(np.diff(twist_raw)) > np.deg2rad(30.0)
                )) if twist_raw.size > 1 else 0,
                "raw_legacy_measure_deg": np.degrees(legacy_unwrapped).tolist(),
                "raw_true_twist_deg": np.degrees(twist_unwrapped).tolist(),
                "filtered_true_twist_deg": np.degrees(filtered_twist).tolist(),
                "target_deg": np.degrees(target_measure).tolist(),
            }
        self._two_pass_anchor_diagnostics = diagnostics
        return dict(diagnostics)

    def set_two_pass_timings(self, pass1_s: float, pass2_s: float) -> None:
        """Attach solve timings after the engine completes both passes."""
        if self.limb_twist_anchor != "two_pass":
            return
        self._two_pass_anchor_diagnostics["pass1_solve_s"] = float(pass1_s)
        self._two_pass_anchor_diagnostics["pass2_solve_s"] = float(pass2_s)

    def two_pass_anchor_diagnostics(self) -> dict:
        return dict(self._two_pass_anchor_diagnostics)

    def _foot_gate_transport_state(self, prev_x: np.ndarray | None,
                                   init_x: np.ndarray | None,
                                   measured: dict, frame_idx: int):
        """Return the prior/init world foot axes and local rotations for both sides."""
        if frame_idx > 0 and prev_x is not None and np.isfinite(prev_x).all():
            state = prev_x
            source = "previous_solved_frame"
        else:
            state = init_x
            source = "analytic_init"
        if state is None or not np.isfinite(state).all():
            state = self.analytic_init(measured)
            source = "analytic_init"
        root_t, local_q = self._state_to_local_rotations(state)
        _, global_q = self.forward_kinematics(root_t, local_q)
        world_axes = {}
        local_targets = {}
        for side in ("l", "r"):
            ankle_i = self.skel.name_to_idx.get(f"ankle_{side}")
            toe_i = self.skel.name_to_idx.get(f"toe_{side}")
            if ankle_i is None or toe_i is None:
                continue
            axis_local = self.rest_offsets[toe_i]
            axis_norm = float(np.linalg.norm(axis_local))
            if axis_norm <= 1e-12:
                world_axes[side] = np.zeros(3)
            else:
                axis_local = axis_local / axis_norm
                world_axes[side] = Rotation.from_quat(global_q[ankle_i]).apply(axis_local)
            local_targets[side] = local_q[ankle_i].copy()
        return source, world_axes, local_targets

    def _state_to_local_rotations(self, x: np.ndarray):
        root_t = x[0:3]
        rotvecs = x[3:].reshape(-1, 3)
        
        # Rs is (num_joints, 4), xyzw. Rotation vectors avoid Euler wrapping
        # and are locally linear around the identity.
        Rs = Rotation.from_rotvec(rotvecs).as_quat()
        return root_t, Rs

    def _state_delta(self, x: np.ndarray, ref_x: np.ndarray) -> np.ndarray:
        """Difference between two IK states, using the SO(3) log for rotations."""
        root_t, local_q = self._state_to_local_rotations(x)
        ref_root_t, ref_local_q = self._state_to_local_rotations(ref_x)
        d_root = root_t - ref_root_t
        d_rot = (Rotation.from_quat(ref_local_q).inv() *
                 Rotation.from_quat(local_q)).as_rotvec().reshape(-1)
        return np.concatenate([d_root, d_rot])

    def _foot_landmark_bone_index(self, name: str, side: str) -> int:
        proxy_name = f"ankle_{side}" if name == "heel" else f"toe_{side}"
        return int(self.skel.name_to_idx[proxy_name])

    def _foot_landmark_bound_cm(self, name: str, side: str) -> float:
        # Bound each local landmark offset from the corresponding solved proxy
        # bone by half that bone's measured anatomical length.  ``bone_lengths``
        # is the source of truth used to build ``rest_offsets``; retain the
        # offset fallback for custom skeletons that omit the diagnostic array.
        bone_i = self._foot_landmark_bone_index(name, side)
        lengths = getattr(self.skel, "bone_lengths", None)
        if lengths is not None and bone_i < len(lengths):
            length = float(lengths[bone_i])
        else:
            length = float(np.linalg.norm(self.rest_offsets[bone_i]))
        return 0.5 * length

    @staticmethod
    def _robust_landmark_median(values: list[np.ndarray]) -> tuple[np.ndarray, int]:
        x = np.asarray(values, dtype=np.float64)
        x = x[np.isfinite(x).all(axis=1)]
        if len(x) == 0:
            return np.full(3, np.nan), 0
        if len(x) < 3:
            return np.median(x, axis=0), int(len(x))
        center = np.median(x, axis=0)
        distance = np.linalg.norm(x - center, axis=1)
        med = float(np.median(distance))
        mad = float(np.median(np.abs(distance - med)))
        keep = distance <= med + 3.5 * max(1.4826 * mad, 0.5)
        return np.median(x[keep], axis=0), int(np.count_nonzero(keep))

    def _estimate_foot_landmark_offsets(self, states, frames, target_sequence):
        rows = {f"{name}_{side}": [] for side in ("l", "r")
                for name in ("heel", "big_toe", "small_toe")}
        for state, frame in zip(states, frames):
            root_t, local_q = self._state_to_local_rotations(state)
            gpos, gq = self.forward_kinematics(root_t, local_q)
            for side in ("l", "r"):
                for name in ("heel", "big_toe", "small_toe"):
                    key = f"{name}_{side}"
                    track = target_sequence.get(key)
                    if track is None or frame >= len(track):
                        continue
                    target = np.asarray(track[frame], dtype=np.float64)
                    if not np.isfinite(target).all():
                        continue
                    bi = self._foot_landmark_bone_index(name, side)
                    rows[key].append(Rotation.from_quat(gq[bi]).inv().apply(target - gpos[bi]))
        offsets, details = {}, {}
        for key, values in rows.items():
            name, side = key.rsplit("_", 1)
            estimate, used = self._robust_landmark_median(values)
            bound = self._foot_landmark_bound_cm(name, side)
            offsets[key] = estimate
            details[key] = {"used_frames": int(used), "bound_cm": float(bound),
                            "finite": bool(np.isfinite(estimate).all()),
                            "norm_cm": (float(np.linalg.norm(estimate))
                                        if np.isfinite(estimate).all() else None)}
        return offsets, details

    def calibrate_foot_landmark_offsets(self, target_sequence, confidence=None,
                                        frame_count=None, stride=5, iterations=2):
        """Run two robust, every-5th-frame calibration passes; fail closed."""
        requested = self.foot_landmark_model
        self._foot_landmark_confidence = confidence
        if requested != "fk_offsets":
            return self._foot_landmark_calibration
        n = int(frame_count if frame_count is not None else
                max((len(v) for v in target_sequence.values()), default=0))
        frames = list(range(0, n, max(1, int(stride))))
        if len(frames) < 3:
            return self._disable_foot_landmarks("insufficient_decimated_frames", frames)
        offsets = {f"{name}_{side}": np.zeros(3) for side in ("l", "r")
                   for name in ("heel", "big_toe", "small_toe")}
        convergence, details = [], {}
        try:
            for iteration in range(max(1, int(iterations))):
                self._foot_landmark_offsets = {k: v.copy() for k, v in offsets.items()}
                self.foot_landmark_model = "fk_offsets"
                states, prev, prev_prev = [], None, None
                for frame in frames:
                    measured = {k: np.asarray(v[frame]) for k, v in target_sequence.items()
                                if frame < len(v)}
                    state = self.solve_frame(measured, prev_x=prev, temporal_weight=0.03,
                                             frame_idx=frame, prev_prev_x=prev_prev,
                                             accel_weight=0.3)
                    if not np.isfinite(state).all():
                        raise ValueError(f"non-finite calibration state at frame {frame}")
                    states.append(state.copy())
                    prev_prev, prev = (prev.copy() if prev is not None else None), state
                estimated, details = self._estimate_foot_landmark_offsets(states, frames, target_sequence)
                delta = float(np.linalg.norm(np.concatenate([
                    estimated[k] - offsets[k] for k in sorted(offsets)])))
                convergence.append({"iteration": int(iteration), "aggregate_delta_cm": delta})
                for key, value in estimated.items():
                    d = details[key]
                    if (not d["finite"] or d["used_frames"] < 3 or
                            d["norm_cm"] > d["bound_cm"] + 1e-9):
                        raise ValueError(f"invalid {key} calibration: {d}")
                offsets = estimated
            self._foot_landmark_offsets = {k: v.copy() for k, v in offsets.items()}
            self.foot_landmark_model = "fk_offsets"
            self._foot_landmark_calibration = {
                "requested_model": requested, "effective_model": "fk_offsets",
                "status": "calibrated", "stride": int(stride), "sampled_frames": frames,
                "iterations": convergence,
                "offsets_cm": {k: v.tolist() for k, v in offsets.items()},
                "landmarks": details,
                "loss": {"position": "weighted FK landmark residual", "robust": "Huber IRLS",
                         "huber_delta_cm": FOOT_LANDMARK_HUBER_DELTA_CM},
                "bounds": "0.5 times corresponding solved foot/ball bone length",
            }
        except Exception as exc:
            self._disable_foot_landmarks(str(exc), frames)
        return self._foot_landmark_calibration

    def _disable_foot_landmarks(self, reason, frames):
        self.foot_landmark_model = "proxy"
        self._foot_landmark_offsets = {}
        self._foot_landmark_calibration = {
            "requested_model": "fk_offsets", "effective_model": "proxy",
            "status": "fallback_proxy", "reason": str(reason), "sampled_frames": list(frames)}
        return self._foot_landmark_calibration

    def foot_landmark_diagnostics(self):
        return dict(self._foot_landmark_calibration)
    def forward_kinematics(self, root_t: np.ndarray, local_rotations: np.ndarray):
        """Forward kinematics: local rotations -> global positions and rotations."""
        global_pos = np.zeros((self.num_joints, 3))
        global_rot = np.zeros((self.num_joints, 4))

        global_pos[0] = self.skel.rest_t[0] + root_t
        global_rot[0] = local_rotations[0]

        for level_nodes in self.levels:
            p_nodes = self.parents[level_nodes]

            R_parent = Rotation.from_quat(global_rot[p_nodes])
            R_local = Rotation.from_quat(local_rotations[level_nodes])

            global_pos[level_nodes] = (
                global_pos[p_nodes] + R_parent.apply(self.rest_offsets[level_nodes])
            )
            global_rot[level_nodes] = (R_parent * R_local).as_quat()

        return global_pos, global_rot

    def analytic_init(self, measured: dict) -> np.ndarray:
        from aimocap.retarget.swing_twist import constrained_rotation
        from aimocap.retarget.spine_chain import distribute_spine_targets
        from scipy.spatial.transform import Rotation
        spine_names = self.skel.topo.spine_chain("pelvis", "neck_01")

        hip_line = measured["hip_r"] - measured["hip_l"]
        spine_dir = measured["neck"] - measured["pelvis"]
        
        idx_hl = self.skel.name_to_idx["hip_l"]
        idx_hr = self.skel.name_to_idx["hip_r"]
        idx_sl = self.skel.name_to_idx["shoulder_l"]
        idx_sr = self.skel.name_to_idx["shoulder_r"]
        
        rest_hip = self.skel.rest_t[idx_hr] - self.skel.rest_t[idx_hl]
        rest_pelvis_mid = (self.skel.rest_t[idx_hl] + self.skel.rest_t[idx_hr]) / 2.0
        rest_neck_mid = (self.skel.rest_t[idx_sl] + self.skel.rest_t[idx_sr]) / 2.0
        rest_spine = rest_neck_mid - rest_pelvis_mid
        
        from aimocap.retarget.root_frame import root_rotation
        R_root = root_rotation(spine_dir, hip_line, rest_spine, rest_hip)
        
        shoulder_line = measured["shoulder_r"] - measured["shoulder_l"]
        rest_shoulder = self.skel.rest_t[idx_sr] - self.skel.rest_t[idx_sl]
        R_upper = root_rotation(spine_dir, shoulder_line, rest_spine, rest_shoulder)

        # Torso twist = shoulder frame relative to hip frame, expressed in
        # the hip (root) local frame.  Distributed across the spine via left-
        # SLERP (R_root -> R_upper), so the actual shoulder-vs-hip yaw is
        # spread evenly instead of dumped at the neck.  Using R_root.inv() *
        # R_upper (not R_upper * R_root.inv()) ensures R_root * R_diff = R_upper
        # exactly at frac=1.0, matching the neck branch (line 184).
        R_diff = R_root.inv() * R_upper

        # The pelvis and neck_01 targets are the measured mid-hip and mid-shoulder
        # positions, OFFSET to the actual joint positions. The pelvis joint is
        # not at mid-hip and neck_01 is not at mid-shoulder (they're offset by
        # ~2.4 cm and ~7 cm in Manny). Without the offset, no FK state can place
        # the pelvis joint at mid-hip AND the hip children at their measured
        # positions (the bone lengths don't match). The offset is the rest pose
        # difference (joint - midpoint), applied rigidly so it rotates with the
        # root frame. Bone lengths from _estimate_bone_lengths are joint-to-child
        # distances, so the corrected targets are consistent with them.
        rest_pelvis_mid = (self.skel.rest_t[idx_hl] + self.skel.rest_t[idx_hr]) / 2.0
        rest_neck_mid = (self.skel.rest_t[idx_sl] + self.skel.rest_t[idx_sr]) / 2.0
        pelvis_offset = self.skel.rest_t[0] - rest_pelvis_mid
        neck_offset = self.skel.rest_t[self.skel.name_to_idx["neck_01"]] - rest_neck_mid
        tgt_pelvis = measured["pelvis"] + R_root.apply(pelvis_offset)
        tgt_neck = measured["neck"] + R_upper.apply(neck_offset)
        
        rest_positions = np.array([self.skel.rest_t[self.skel.name_to_idx[nm]] for nm in spine_names])
        inter = distribute_spine_targets(tgt_pelvis, tgt_neck, rest_positions)
        
        target_pos = np.zeros((self.num_joints, 3))
        si = 0
        for k, nm in enumerate(spine_names):
            i = self.skel.name_to_idx[nm]
            if k == 0:
                target_pos[i] = tgt_pelvis
            elif k == len(spine_names) - 1:
                target_pos[i] = tgt_neck
            else:
                target_pos[i] = inter[si]; si += 1
            
        for i, nm in self.skel.coco_anchor.items():
            if self.skel.joint_names[i] not in ["pelvis", "neck_01"]:
                if nm in measured:
                    target_pos[i] = measured[nm]
                # If nm not in measured (e.g. 17-joint input without foot
                # keypoints), target_pos[i] stays zero — the toe fallback below
                # will synthesize from R_root.

        # Clavicle targets: the clavicles are children of neck_01 but have no
        # COCO anchor (no "clavicle" keypoint in COCO).  Without a synthesized
        # target, target_pos stays 0, which makes the init orient the clavicle
        # from the origin (garbage) and the solver pull it to the origin
        # (caved-in shoulders).  Synthesize from the MEASURED shoulder positions
        # (COCO 5/6) rather than R_upper.apply(rest_off), because the
        # triangulated data has opposite chirality from the rig — using
        # R_upper on the non-mirrored rest offset places the right clavicle
        # on the left side.  The clavicle sits on the line from neck to
        # shoulder, at the rest-proportion distance from the neck.
        neck_idx = self.skel.name_to_idx["neck_01"]
        for side in ("l", "r"):
            clav_name = f"clavicle_{side}"
            sh_name = f"shoulder_{side}"
            if (clav_name in self.skel.name_to_idx
                    and sh_name in measured
                    and np.all(np.isfinite(measured[sh_name]))):
                ci = self.skel.name_to_idx[clav_name]
                if np.allclose(target_pos[ci], 0.0):
                    # Place clavicle on the neck→shoulder line at rest proportion
                    sh_pos = measured[sh_name]
                    neck_to_shoulder = sh_pos - tgt_neck
                    rest_neck_to_shoulder = (
                        self.skel.rest_t[self.skel.name_to_idx[sh_name]]
                        - self.skel.rest_t[neck_idx])
                    rest_clav_len = np.linalg.norm(
                        self.skel.rest_t[ci] - self.skel.rest_t[neck_idx])
                    rest_shoulder_dist = np.linalg.norm(rest_neck_to_shoulder)
                    if rest_shoulder_dist > 1e-5:
                        frac = rest_clav_len / rest_shoulder_dist
                    else:
                        frac = 0.5
                    target_pos[ci] = tgt_neck + frac * neck_to_shoulder

        # Toe (ball_l/ball_r) targets: when the big-toe keypoint is measured
        # (COCO-WholeBody 17/20), the coco_anchor loop above already set the
        # target from measurement — use it, so foot orientation comes from
        # actual detected foot position, not the pelvis root frame. When the
        # big-toe is NaN (not detected) or toe_* is not coco-anchored, fall
        # back to R_root synthesis so the ankle roll still has a constraint.
        for side in ("l", "r"):
            toe_name = f"toe_{side}"
            ankle_name = f"ankle_{side}"
            if toe_name in self.skel.name_to_idx and ankle_name in self.skel.name_to_idx:
                toe_i = self.skel.name_to_idx[toe_name]
                ankle_i = self.skel.name_to_idx[ankle_name]
                toe_anchor = self.skel.coco_anchor.get(toe_i)
                needs_synth = (toe_anchor is None
                               or toe_anchor not in measured
                               or np.any(np.isnan(target_pos[toe_i])))
                if needs_synth:
                    rest_off = self.skel.rest_t[toe_i] - self.skel.rest_t[ankle_i]
                    target_pos[toe_i] = target_pos[ankle_i] + R_root.apply(rest_off)

        # Dorsiflexion clamp: prevent the toe target from landing ABOVE the
        # ankle/heel, which would dorsiflex the foot ("on heels" artifact).
        # The toe is synthesized from R_root (pelvis frame); when the torso
        # leans forward, R_root tilts and projects the toe's rest "down"
        # component forward+up -> toe above ankle -> dorsiflexion.
        # Two reference heights (prefer the more precise one):
        #   (a) heel keypoint (COCO-WholeBody 19/22) when available: the sole
        #       runs heel->toe, so toe Z <= heel Z + 1 cm keeps it flat.
        #   (b) ankle keypoint (always available, 17-joint data): the toe
        #       should not be above the ankle (rest toe is 7 cm below it).
        #       Clamp toe Z to ankle Z so the sole stays roughly flat.
        for side in ("l", "r"):
            toe_name = f"toe_{side}"
            ankle_name = f"ankle_{side}"
            heel_key = f"heel_{side}"
            if (toe_name not in self.skel.name_to_idx
                    or ankle_name not in self.skel.name_to_idx):
                continue
            toe_i = self.skel.name_to_idx[toe_name]
            ankle_i = self.skel.name_to_idx[ankle_name]
            if not np.all(np.isfinite(target_pos[toe_i])):
                continue
            if (heel_key in measured
                    and np.all(np.isfinite(measured[heel_key]))):
                # (a) heel-based clamp (133-joint data): precise.
                max_toe_z = measured[heel_key][2] + 1.0
            else:
                # (b) ankle-based clamp (17-joint data): toe can't exceed ankle.
                max_toe_z = target_pos[ankle_i][2]
            if target_pos[toe_i][2] > max_toe_z:
                target_pos[toe_i][2] = max_toe_z

        root_t = tgt_pelvis - self.skel.rest_t[0]

        global_rot = [None] * self.num_joints
        local_quats = np.zeros((self.num_joints, 4))
        global_rot[0] = R_root
        local_quats[0] = R_root.as_quat()

        # Map each parent to its primary child to orient the parent's outgoing bone
        primary_child = {}
        for j in range(1, self.num_joints):
            p = self.parents[j]
            if p not in primary_child:
                primary_child[p] = j

        ordered_parents = []
        for lvl in self.levels:
            for j in lvl:
                p = int(self.parents[j])
                if p not in ordered_parents:
                    ordered_parents.append(p)

        neck_idx = self.skel.name_to_idx["neck_01"]

        for p in ordered_parents:
            if p == 0: continue
            
            p_parent = self.parents[p]
            
            if p == neck_idx:
                R_parent = global_rot[p_parent]
                R_local_neck = R_parent.inv() * R_upper
                local_quats[p] = R_local_neck.as_quat()
                global_rot[p] = R_upper
                continue

            if self.skel.joint_names[p] in spine_names:
                # Distribute the torso twist (R_diff = R_upper . R_root^-1)
                # — NOT the full root rotation — evenly across the spine via
                # SLERP.  The pelvis is already R_root (line 156), so applying
                # R_root fractions on top over-rotates the upper spine by up
                # to R_root^(5/6) (~83 deg).  Interpolating from R_root to
                # R_upper distributes only the *difference* between the hip
                # and shoulder frames: spine_k=0 -> R_root, and the neck
                # branch (line 184) gets R_upper = R_root . R_diff exactly.
                n_spine = len(spine_names)
                spine_k = spine_names.index(self.skel.joint_names[p])
                frac = spine_k / (n_spine - 1)
                R_target = R_root * (R_diff ** frac)
                local_quats[p] = (global_rot[p_parent].inv() * R_target).as_quat()
                global_rot[p] = R_target
                continue
                
            if p not in primary_child:
                local_quats[p] = [0, 0, 0, 1]
                global_rot[p] = global_rot[p_parent]
                continue
                
            j = primary_child[p]
            rest_off = self.skel.rest_offsets[j]
            desired_world = target_pos[j] - target_pos[p]

            if np.linalg.norm(desired_world) < 1e-5:
                local_quats[p] = [0, 0, 0, 1]
                global_rot[p] = global_rot[p_parent]
                continue

            desired_pl = global_rot[p_parent].inv().apply(desired_world)

            # VQ-FIX2c: two-bone roll pinning. A pure swing leaves the
            # joint's axial roll path-dependent (the minimal rotation from
            # the A-pose to the current bone direction), which measured as
            # 85-175 deg of accumulated palm/sole roll error on the user
            # clip. For HINGE-parent joints whose grandchild bone is
            # measured, pin the roll with the grandchild direction: the
            # elbow/knee hinge axis is fixed in the parent bone's frame,
            # so the bend plane determines the parent's axial roll
            # exactly. Restricted to joints with their own COCO anchor
            # (shoulder/hip/knee): the clavicle must NOT be pinned — its
            # "roll child" is the freely-abducting upper arm, and pinning
            # it injected a constant ~136 deg roll into the arm chain
            # (measured). constrained_rotation's existing 35-deg
            # collinearity guard falls back to swing-only when the limb
            # is straight (roll unmeasurable).
            roll_cr = None
            roll_cd = None
            g = primary_child.get(j)
            if (g is not None
                    and self.skel.coco_anchor.get(p) is not None):
                g_world = target_pos[g] - target_pos[j]
                if (np.all(np.isfinite(g_world))
                        and np.linalg.norm(g_world) > 1e-5
                        and np.all(np.isfinite(self.skel.rest_offsets[g]))):
                    roll_cr = self.skel.rest_offsets[g]
                    roll_cd = global_rot[p_parent].inv().apply(g_world)

            R_local = constrained_rotation(
                rest_off, desired_pl,
                roll_child_rest=roll_cr, roll_child_desired=roll_cd)
            local_quats[p] = R_local.as_quat()
            global_rot[p] = global_rot[p_parent] * R_local

        # Orient leaves: joints with COCO anchors (e.g. head->nose) get
        # oriented toward their measured target so they don't just inherit
        # the parent's rotation.  Without this, the head always points the
        # same way as the neck regardless of where the nose is — a visible
        # tilt.  Other leaves (twist bones, etc.) keep identity.
        
        for p in range(self.num_joints):
            if p not in primary_child and p != 0:
                coco_name = self.skel.coco_anchor.get(p)
                if coco_name is not None and coco_name in measured:
                    p_parent = self.parents[p]
                    rest_dir = self.skel.rest_offsets[p]

                    # ── Safe production head mode ─────────────────────────
                    if p == self.skel.name_to_idx.get("head") and HEAD_ORIENTATION_MODE == "neck_follow":
                        local_quats[p] = [0.0, 0.0, 0.0, 1.0]
                        global_rot[p] = global_rot[p_parent]
                        continue

                    # ── Frame-based head orientation (diagnostic only) ─────
                    # Build an orthonormal head frame from face keypoints
                    # (ears for lateral, face midpoint for forward) and the
                    # rig's rest frame (clavicles for lateral). The delta
                    # R = F_obs @ F_rest.T gives real head tracking (yaw,
                    # pitch, roll) from data, without Kabsch or nose-direction.
                    #
                    # The triangulated data has opposite chirality from the
                    # rig (scalar triple product confirms this). Both frames
                    # are built with the same left-handed formula, then forced
                    # to right-handed by flipping the up column. This makes
                    # the delta a proper rotation that maps rest → observed.
                    if coco_name == "nose" and "left_ear" in measured:
                        la = measured["left_ear"]
                        ra = measured["right_ear"]
                        nose = measured["nose"]
                        le = measured.get("left_eye")
                        re = measured.get("right_eye")
                        face_pts = [p for p in [nose, le, re]
                                    if p is not None and np.all(np.isfinite(p))]
                        if (np.all(np.isfinite(la)) and np.all(np.isfinite(ra))
                                and len(face_pts) >= 2
                                and np.linalg.norm(la - ra) > 5.0):
                            # Observed head frame
                            x = la - ra  # left = ear_L - ear_R
                            x = x / np.linalg.norm(x)
                            face_mid = np.mean(face_pts, axis=0)
                            ear_mid = (la + ra) / 2.0
                            f_raw = face_mid - ear_mid
                            f = f_raw - np.dot(f_raw, x) * x  # perp to x
                            fn = np.linalg.norm(f)
                            if fn > 1e-5:
                                f = f / fn
                                u = np.cross(f, x)  # forward x left = up
                                u = u / np.linalg.norm(u)
                                F_obs = np.column_stack([x, f, u])
                                F_obs[:, 2] *= -1  # force RH
                                if np.linalg.det(F_obs) > 0:
                                    R_head_global = self._head_target_rotation(F_obs)
                                    R_neck = global_rot[p_parent]
                                    R_local = R_neck.inv() * R_head_global
                                    local_quats[p] = R_local.as_quat()
                                    global_rot[p] = global_rot[p_parent] * R_local
                                    continue
                        # Fall through to default if degenerate

                    # Compute the desired direction for this leaf joint
                    # (from parent's target to this joint's target).
                    desired_world = target_pos[p] - target_pos[p_parent]
                    desired_pl = global_rot[p_parent].inv().apply(desired_world)
                    if np.linalg.norm(desired_pl) > 1e-5:
                        # Plausibility gate: the measured anchor (e.g. nose) can
                        # be a noisy triangulation that lands far off the head's
                        # plausible direction (e.g. 23 cm ABOVE the neck when the
                        # head bone points forward).  Trusting that would swing
                        # the head ~90 deg sideways ("completely left/right").
                        # Only accept the swing if the desired direction is
                        # within ~70 deg of the rest bone direction (the head
                        # can tilt/look around, but not face backward or
                        # sideways from the neck).  Otherwise fall back to the
                        # parent's rotation (identity local) -- the orientation
                        # residual in _residuals will refine it gently.
                        cos_accept = np.dot(
                            desired_pl / np.linalg.norm(desired_pl),
                            rest_dir / np.linalg.norm(rest_dir))
                        if cos_accept > np.cos(np.deg2rad(70.0)):
                            R_local = constrained_rotation(rest_dir, desired_pl)
                            local_quats[p] = R_local.as_quat()
                            global_rot[p] = global_rot[p_parent] * R_local
                            continue
                # For leaf joints (e.g. wrist/hand), inherit the parent's
                # rotation so the hand follows the forearm instead of freezing
                # at identity.  This gives the hand the correct base orientation
                # (matching the rest pose relative to the forearm), which the
                # GPU IK can then refine.  Without this, the hand stays at
                # absolute identity, appearing frozen at rest for all frames.
                if coco_name is not None and coco_name in measured:
                    # Leaf with a measured target (e.g. wrist): orient the
                    # bone direction, and inherit parent rotation for roll.
                    p_parent = self.parents[p]
                    desired_world = target_pos[p] - target_pos[p_parent]
                    desired_pl = global_rot[p_parent].inv().apply(desired_world)
                    if np.linalg.norm(desired_pl) > 1e-5:
                        cos_accept = np.dot(
                            desired_pl / np.linalg.norm(desired_pl),
                            rest_dir / np.linalg.norm(rest_dir))
                        if cos_accept > np.cos(np.deg2rad(70.0)):
                            R_local = constrained_rotation(rest_dir, desired_pl)
                            local_quats[p] = R_local.as_quat()
                            global_rot[p] = global_rot[p_parent] * R_local
                            continue
                # Default: inherit parent rotation (identity local = follow parent)
                local_quats[p] = [0, 0, 0, 1]
                global_rot[p] = global_rot[self.parents[p]]

        rotvecs = Rotation.from_quat(local_quats).as_rotvec()
        x = np.zeros(self.num_vars)
        x[0:3] = root_t
        x[3:] = rotvecs.flatten()
        return x



    def _precompute_frame(self, measured: dict, prev_x: np.ndarray = None,
                          temporal_weight: float = 0.03, init_x: np.ndarray = None,
                          prev_lat: float = None, frame_idx: int = 0,
                          prev_prev_x: np.ndarray = None,
                          accel_weight: float = 0.0) -> dict:
        """Compute all frame-constant quantities once per solve_frame call.

        These were previously recomputed 1200+ times per frame inside
        _residuals (once per least_squares eval).  Precomputing them here
        saves ~60% of the per-frame cost with zero quality change — the
        values are identical, just computed once instead of 1200x.
        """
        from aimocap.retarget.root_frame import root_rotation
        from aimocap.retarget.spine_chain import distribute_spine_targets

        # ── Root frame (depends only on measured) ────────────────────────
        hip_line = measured["hip_r"] - measured["hip_l"]
        spine_dir = measured["neck"] - measured["pelvis"]

        idx_hl = self.skel.name_to_idx["hip_l"]
        idx_hr = self.skel.name_to_idx["hip_r"]
        idx_sl = self.skel.name_to_idx["shoulder_l"]
        idx_sr = self.skel.name_to_idx["shoulder_r"]

        rest_hip = self.skel.rest_t[idx_hr] - self.skel.rest_t[idx_hl]
        rest_pelvis_mid = (self.skel.rest_t[idx_hl] + self.skel.rest_t[idx_hr]) / 2.0
        rest_neck_mid = (self.skel.rest_t[idx_sl] + self.skel.rest_t[idx_sr]) / 2.0
        rest_spine = rest_neck_mid - rest_pelvis_mid

        R_root = root_rotation(spine_dir, hip_line, rest_spine, rest_hip)

        shoulder_line = measured["shoulder_r"] - measured["shoulder_l"]
        rest_shoulder = self.skel.rest_t[idx_sr] - self.skel.rest_t[idx_sl]
        R_upper = root_rotation(spine_dir, shoulder_line, rest_spine, rest_shoulder)
        R_diff = R_root.inv() * R_upper

        # D1 NaN-tolerance: the orientation residual terms (spine/head/limb
        # pinned) are all computed from R_root/R_upper/R_diff. When the
        # measured spine/hip/shoulder inputs are non-finite, those terms
        # would be NaN and poison the WHOLE residual (the optimizer then
        # returns garbage). Mirror the position-side w[i]=0 rule: drop the
        # orientation terms for this frame (ori_weight=0) and let the
        # position + temporal terms carry the solve. Finite inputs keep
        # ori_weight=1.0 bit-exact.
        ori_inputs_finite = (
            bool(np.all(np.isfinite(spine_dir)))
            and bool(np.all(np.isfinite(hip_line)))
            and bool(np.all(np.isfinite(shoulder_line)))
            and bool(np.all(np.isfinite(measured["pelvis"])))
            and bool(np.all(np.isfinite(measured["neck"])))
        )

        # ── Target positions (depends only on measured + rest) ───────────
        pelvis_offset = self.skel.rest_t[0] - rest_pelvis_mid
        neck_offset = self.skel.rest_t[self.skel.name_to_idx["neck_01"]] - rest_neck_mid
        tgt_pelvis = measured["pelvis"] + R_root.apply(pelvis_offset)
        tgt_neck = measured["neck"] + R_upper.apply(neck_offset)

        spine_names = self.skel.topo.spine_chain("pelvis", "neck_01")
        rest_positions = np.array([self.skel.rest_t[self.skel.name_to_idx[nm]] for nm in spine_names])
        inter = distribute_spine_targets(tgt_pelvis, tgt_neck, rest_positions)

        tgt_pos = np.zeros((self.num_joints, 3))
        si = 0
        for k, nm in enumerate(spine_names):
            i = self.skel.name_to_idx[nm]
            if k == 0:
                tgt_pos[i] = tgt_pelvis
            elif k == len(spine_names) - 1:
                tgt_pos[i] = tgt_neck
            else:
                tgt_pos[i] = inter[si]; si += 1

        for i, nm in self.skel.coco_anchor.items():
            if self.skel.joint_names[i] not in ["pelvis", "neck_01"]:
                if nm in measured:
                    tgt_pos[i] = measured[nm]

        # Clavicle synthesis
        neck_idx_r = self.skel.name_to_idx["neck_01"]
        for side in ("l", "r"):
            clav_name = f"clavicle_{side}"
            if clav_name in self.skel.name_to_idx:
                ci = self.skel.name_to_idx[clav_name]
                if np.allclose(tgt_pos[ci], 0.0):
                    rest_off = self.skel.rest_t[ci] - self.skel.rest_t[neck_idx_r]
                    tgt_pos[ci] = tgt_neck + R_upper.apply(rest_off)

        # Toe synthesis + dorsiflexion clamp
        for side in ("l", "r"):
            toe_name = f"toe_{side}"
            ankle_name = f"ankle_{side}"
            if toe_name in self.skel.name_to_idx and ankle_name in self.skel.name_to_idx:
                toe_i = self.skel.name_to_idx[toe_name]
                ankle_i = self.skel.name_to_idx[ankle_name]
                toe_anchor = self.skel.coco_anchor.get(toe_i)
                needs_synth = (toe_anchor is None
                               or toe_anchor not in measured
                               or np.any(np.isnan(tgt_pos[toe_i])))
                if needs_synth:
                    rest_off = self.skel.rest_t[toe_i] - self.skel.rest_t[ankle_i]
                    tgt_pos[toe_i] = tgt_pos[ankle_i] + R_root.apply(rest_off)

            # Dorsiflexion clamp
            heel_key = f"heel_{side}"
            if (toe_name in self.skel.name_to_idx
                    and ankle_name in self.skel.name_to_idx):
                toe_i = self.skel.name_to_idx[toe_name]
                ankle_i = self.skel.name_to_idx[ankle_name]
                if not np.all(np.isfinite(tgt_pos[toe_i])):
                    continue
                if (heel_key in measured
                        and np.all(np.isfinite(measured[heel_key]))):
                    max_toe_z = measured[heel_key][2] + 1.0
                else:
                    max_toe_z = tgt_pos[ankle_i][2]
                if tgt_pos[toe_i][2] > max_toe_z:
                    tgt_pos[toe_i][2] = max_toe_z

        # ── Weight vector w (frame-constant portion) ─────────────────────
        # Base weight: 1.0 for measured joints, 0.15 for synthesized, 0 for NaN.
        # Multiply by per-joint triangulation confidence when available —
        # this downweights noisy joints (e.g. right hip conf=0.08).
        conf = self.skel.confidence
        w = np.zeros(self.num_joints)
        for i, val in enumerate(tgt_pos):
            if np.any(np.isnan(val)):
                w[i] = 0.0
            elif i not in self.skel.coco_anchor:
                if np.allclose(val, 0.0):
                    w[i] = 0.0
                else:
                    w[i] = 0.15
            else:
                w[i] = 1.0
                # Apply per-joint confidence if available.
                if conf is not None:
                    coco_name = self.skel.coco_anchor[i]
                    coco_idx = COCO.get(coco_name)
                    if coco_idx is not None and frame_idx < conf.shape[0]:
                        c = conf[frame_idx, coco_idx]
                        if np.isfinite(c) and c > 0.05:
                            # Real confidence from RANSAC — scale the IK weight.
                            w[i] *= float(np.clip(c, 0.1, 1.0))
                        else:
                            # Zero/near-zero confidence: the 2D detector failed
                            # for this frame, but the position was gap-filled from
                            # neighboring frames.  Use a moderate default (0.3)
                            # so the IK still tries to match it, but with less
                            # authority than a high-confidence joint.
                            w[i] *= 0.3

        # VQ-FIX6: gate measured toe position only when the per-frame foot
        # observation is implausible. Other leg/chain targets keep their weights.
        gate_state_source, transported_axes, transport_local_targets = (
            self._foot_gate_transport_state(prev_x, init_x, measured, frame_idx))
        foot_target_gates = {}
        foot_transport_pins = []
        for side in ("l", "r"):
            ankle_name = f"ankle_{side}"
            toe_name = f"toe_{side}"
            measured_toe_name = f"big_toe_{side}"
            ankle_i = self.skel.name_to_idx.get(ankle_name)
            toe_i = self.skel.name_to_idx.get(toe_name)
            # Legacy 17-joint input has no measured big-toe anchor; preserve
            # its existing synthesized toe target without treating it as lost data.
            if (ankle_i is None or toe_i is None
                    or toe_i not in self.skel.coco_anchor):
                continue

            ankle_pos = measured.get(ankle_name, np.full(3, np.nan))
            toe_pos = measured.get(measured_toe_name, np.full(3, np.nan))
            mask = self._foot_gate_observed_masks.get(side)
            ankle_observed = toe_observed = None
            if mask is not None and 0 <= frame_idx < len(mask["ankle"]):
                ankle_observed = bool(mask["ankle"][frame_idx])
                toe_observed = bool(mask["big_toe"][frame_idx])
            
            # Get heel target for observed mode
            heel_target = None
            if self.foot_gate_reference == "observed":
                heel_key = f"heel_{side}"
                if heel_key in measured and np.all(np.isfinite(measured[heel_key])):
                    heel_target = np.asarray(measured[heel_key], dtype=np.float64)
            
            # Get last accepted observation for temporal consistency
            last_accepted_frame = None
            last_accepted_direction = None
            if self.foot_gate_reference == "observed" and side in self._foot_gate_last_accepted:
                last_frame, last_dir = self._foot_gate_last_accepted[side]
                last_accepted_frame = last_frame
                last_accepted_direction = last_dir
            
            result = assess_foot_target(
                ankle_pos, toe_pos, transported_axes.get(side, np.zeros(3)),
                self._foot_gate_reference_lengths.get(side),
                self._foot_gate_reference_modes.get(side, "direction_only"),
                ankle_observed=ankle_observed,
                toe_observed=toe_observed,
                foot_gate_reference=self.foot_gate_reference,
                heel_target=heel_target,
                fps=self._fps,
                last_accepted_frame=last_accepted_frame,
                last_accepted_direction=last_accepted_direction,
                current_frame_idx=frame_idx,
            )
            result.update({"frame": int(frame_idx), "side": side,
                           "transport_target_source": gate_state_source})
            foot_target_gates[side] = result
            record_key = (int(frame_idx), side)
            if not result["healthy"]:
                w[toe_i] = 0.0
                target_q = transport_local_targets.get(side)
                if target_q is not None:
                    foot_transport_pins.append({
                        "side": side,
                        "joint_idx": ankle_i,
                        "target_quaternion": target_q,
                        "weight": 1.0,
                    })
                result["transport_pin_active"] = target_q is not None
                self._foot_gate_records[record_key] = result
            else:
                result["transport_pin_active"] = False
                self._foot_gate_records.pop(record_key, None)
                # Update last accepted observation for observed mode
                if self.foot_gate_reference == "observed":
                    ankle_finite = ankle_pos.shape == (3,) and bool(np.isfinite(ankle_pos).all())
                    toe_finite = toe_pos.shape == (3,) and bool(np.isfinite(toe_pos).all())
                    if ankle_finite and toe_finite:
                        direction = toe_pos - ankle_pos
                        direction_norm = direction / float(np.linalg.norm(direction))
                        self._foot_gate_last_accepted[side] = (frame_idx, direction_norm)

        # The proxy toe position is the solver's big-toe residual. Scale only
        # this measured foot landmark; the transport pins below intentionally
        # retain their independent weight.
        if self.foot_landmark_weight != 1.0:
            for side in ("l", "r"):
                toe_i = self.skel.name_to_idx.get(f"toe_{side}")
                if toe_i is not None and toe_i in self.skel.coco_anchor:
                    w[toe_i] *= self.foot_landmark_weight

        # Opt-in FK landmark observations. The proxy model above remains the
        # only position term unless calibration completed successfully.
        # Keep this packet empty on the default path so proxy residual inputs
        # remain exactly the historical position/orientation/temporal terms.
        foot_landmark_packet = []
        if self.foot_landmark_model == "fk_offsets":
            for side in ("l", "r"):
                toe_i = self.skel.name_to_idx.get(f"toe_{side}")
                if toe_i is not None:
                    w[toe_i] = 0.0
            for side in ("l", "r"):
                healthy = bool(foot_target_gates.get(side, {}).get("healthy", True))
                for name in ("heel", "big_toe", "small_toe"):
                    key = f"{name}_{side}"
                    target = np.asarray(measured.get(key, np.full(3, np.nan)), dtype=np.float64)
                    weight = 1.0 if np.isfinite(target).all() and healthy else 0.0
                    if weight > 0.0 and self._foot_landmark_confidence is not None:
                        coco_idx = COCO.get(key)
                        conf = self._foot_landmark_confidence
                        if coco_idx is not None and frame_idx < len(conf) and coco_idx < conf.shape[1]:
                            c = float(conf[frame_idx, coco_idx])
                            weight *= float(np.clip(c if np.isfinite(c) and c > 0.05 else 0.3, 0.1, 1.0))
                    bone_i = self._foot_landmark_bone_index(name, side)
                    offset = self._foot_landmark_offsets.get(key, np.zeros(3))
                    foot_landmark_packet.append([float(bone_i), *np.nan_to_num(target),
                                                 *np.asarray(offset, dtype=float), weight])

        # ── Spine orientation targets (R_root * R_diff^frac) ─────────────
        n_spine = len(spine_names)
        spine_indices = np.array([self.skel.name_to_idx[nm] for nm in spine_names])
        R_target_quats = np.zeros((n_spine, 4))
        for k in range(n_spine):
            frac = k / (n_spine - 1) if n_spine > 1 else 0.0
            R_target = R_root * (R_diff ** frac)
            R_target_quats[k] = R_target.as_quat()  # xyzw

        # ── Lateral axis for sway damping ────────────────────────────────
        lat_axis = rest_hip / (np.linalg.norm(rest_hip) + 1e-9)

        # ── Limb twist pinned set (frame-constant: depends on tgt_pos) ───
        # VQ-FIX2d: the pin used to be SKIPPED whenever the chain's roll
        # child was "well-conditioned" (limb bent >35 deg) on the theory
        # that the bend plane measures the roll. With the arm-orientation
        # rewrite (Stage 1b) disabled there is NO measured roll (Gate 2),
        # so the skip left the parent joint's axial roll a FREE DOF at
        # bent limbs; with warm-start inheritance the LM then jumped
        # between roll basins ~180 deg apart (reproduced: 167 deg
        # single-frame arm-chain pops on the user clip, absent when the
        # solve re-initializes from analytic_init). The pin is now ALWAYS
        # active, anchored to the INIT's joint twist: the analytic init
        # (VQ-FIX2c) pins the bend plane from the measured grandchild
        # direction when the limb is bent, and falls back to swing-neutral
        # when straight — a continuous, data-derived anchor either way.
        from aimocap.retarget.swing_twist import signed_twist_angle as _sta
        init_rotvecs = (init_x[3:].reshape(-1, 3)
                        if init_x is not None
                        and np.all(np.isfinite(init_x)) else None)
        # Only construct this extra analytic-frame data for opt-in modes.  The
        # default branch below intentionally retains the historical operation
        # order and signed_twist_angle call for byte-identical init behavior.
        continuous_data = {}
        if self.limb_twist_anchor not in {"init", "two_pass"} \
                and init_x is not None \
                and np.all(np.isfinite(init_x)):
            from aimocap.retarget.swing_twist import bend_plane_rotation
            init_root, init_local = self._state_to_local_rotations(init_x)
            _, init_global_q = self.forward_kinematics(init_root, init_local)
            init_global = Rotation.from_quat(init_global_q)
            primary_child = {}
            for j in range(1, self.num_joints):
                parent = int(self.parents[j])
                if parent not in primary_child:
                    primary_child[parent] = j
            for side in ("l", "r"):
                for p_name, c_name in (
                    (f"hip_{side}", f"knee_{side}"),
                    (f"knee_{side}", f"ankle_{side}"),
                    (f"shoulder_{side}", f"elbow_{side}"),
                    (f"elbow_{side}", f"wrist_{side}"),
                ):
                    if p_name not in self.skel.name_to_idx:
                        continue
                    p = self.skel.name_to_idx[p_name]
                    c = self.skel.name_to_idx[c_name]
                    g = primary_child.get(c)
                    if g is None:
                        continue
                    bone = self.rest_offsets[c]
                    bone = bone / (np.linalg.norm(bone) + 1e-12)
                    parent_global = init_global[int(self.parents[p])]
                    bone_world = tgt_pos[c] - tgt_pos[p]
                    child_world = tgt_pos[g] - tgt_pos[c]
                    bn = np.linalg.norm(bone_world)
                    gn = np.linalg.norm(child_world)
                    if (not np.all(np.isfinite(bone_world))
                            or not np.all(np.isfinite(child_world))
                            or bn <= 1e-5 or gn <= 1e-5):
                        continue
                    bone_dir_world = bone_world / bn
                    child_dir_world = child_world / gn
                    theta = float(np.degrees(np.arccos(np.clip(
                        np.dot(bone_dir_world, child_dir_world), -1.0, 1.0))))
                    u = np.clip(
                        (theta - LIMB_TWIST_OBSERVABILITY_LOW_DEG)
                        / (LIMB_TWIST_OBSERVABILITY_HIGH_DEG
                           - LIMB_TWIST_OBSERVABILITY_LOW_DEG), 0.0, 1.0)
                    observability = float(3.0 * u * u - 2.0 * u * u * u)
                    desired_pl = parent_global.inv().apply(bone_world)
                    child_desired_pl = parent_global.inv().apply(child_world)
                    bend = bend_plane_rotation(
                        self.rest_offsets[c], desired_pl,
                        roll_child_rest=self.rest_offsets[g],
                        roll_child_desired=child_desired_pl,
                    )
                    if bend is None:
                        observability = 0.0
                    continuous_data[(p, c)] = {
                        "observability": observability,
                        "bend": bend,
                        "theta_deg": theta,
                    }

        limb_pinned_info = []
        for side in ("l", "r"):
            for p_name, c_name in (
                (f"hip_{side}", f"knee_{side}"),
                (f"knee_{side}", f"ankle_{side}"),
                (f"shoulder_{side}", f"elbow_{side}"),
                (f"elbow_{side}", f"wrist_{side}"),
            ):
                if p_name not in self.skel.name_to_idx:
                    continue
                p = self.skel.name_to_idx[p_name]
                c = self.skel.name_to_idx[c_name]
                if self.limb_twist_pin == "off":
                    continue
                pin_multiplier = 1.0
                pin_theta = None
                if self.limb_twist_pin == "fade":
                    g = next((j for j in range(1, self.num_joints)
                              if int(self.parents[j]) == int(c)), None)
                    pin_multiplier, pin_theta = limb_twist_pin_multiplier(tgt_pos, p, c, g)
                rest_off = self.skel.rest_offsets[c]
                bone = rest_off / (np.linalg.norm(rest_off) + 1e-12)
                p_parent = int(self.parents[p])

                twist_target = 0.0
                if init_rotvecs is not None:
                    rv = init_rotvecs[p]
                    if np.all(np.isfinite(rv)) and np.linalg.norm(rv) > 1e-12:
                        try:
                            # radians — the residual's twist_rad is a dot
                            # of the rotvec with the bone axis (radians).
                            twist_target = float(
                                _sta(Rotation.from_rotvec(rv), bone))
                        except Exception:
                            twist_target = 0.0

                tau_b_diag = None
                tau_ref_diag = None
                if (self.limb_twist_anchor == "two_pass"
                        and self._two_pass_phase == 2
                        and self._two_pass_filtered_pin_enabled(p_name)):
                    twist_target = float(self._two_pass_targets.get(
                        (int(frame_idx), int(p), int(c)), twist_target))
                elif self.limb_twist_anchor == "smoothed":
                    # Offline target: no previous-solver-state feedback.  In
                    # arm-only mode, retain the historical init target for
                    # legs so smoothing cannot perturb foot behavior.
                    if self._smoothed_filtered_pin_enabled(p_name):
                        twist_target = float(self._limb_twist_targets.get(
                            (int(frame_idx), int(p), int(c)), twist_target))
                    item = continuous_data.get((p, c))
                    if item is not None:
                        tau_b_diag = float(item["bend"].as_rotvec().dot(bone)) if item.get("bend") is not None else None
                        tau_ref_diag = None
                elif self.limb_twist_anchor not in {"init", "two_pass"}:
                    item = continuous_data.get((p, c))
                    if item is not None:
                        bend = item["bend"]
                        if self.limb_twist_anchor == "continuous_resid":
                            measure = lambda rotation: float(np.dot(
                                rotation.as_rotvec(), bone))
                        else:
                            measure = lambda rotation: float(_sta(rotation, bone))
                        tau_b = (measure(bend) if bend is not None else twist_target)
                        tau_b_diag = tau_b
                        if prev_x is not None and np.all(np.isfinite(prev_x)):
                            prev_rv = prev_x[3 + p * 3:3 + p * 3 + 3]
                            tau_ref = measure(Rotation.from_rotvec(prev_rv))
                        else:
                            tau_ref = twist_target
                        tau_ref_diag = tau_ref
                        delta = (tau_b - tau_ref + np.pi) % (2.0 * np.pi) - np.pi
                        twist_target = tau_ref + item["observability"] * delta

                limb_pinned_info.append({
                    "p": p, "c": c, "bone": bone, "p_parent": p_parent,
                    "twist_target": twist_target,
                    "twist_weight": 1.0,
                    "twist_pin_multiplier": pin_multiplier,
                    "twist_pin_theta_deg": pin_theta,
                    "side": side,
                    "twist_observability": (continuous_data.get((p, c), {}).get(
                        "observability", 0.0) if self.limb_twist_anchor not in {"init", "two_pass"} else None),
                    "twist_theta_deg": (continuous_data.get((p, c), {}).get(
                        "theta_deg") if self.limb_twist_anchor not in {"init", "two_pass"} else None),
                    "twist_tau_b": tau_b_diag,
                    "twist_tau_ref": tau_ref_diag,
                })
                self._limb_twist_pin_frames[(f"{side}:{p}->{c}", int(frame_idx))] = (
                    float(pin_multiplier), float(twist_target)
                )

        # ── Head orientation metadata ────────────────────────────────────
        head_i = self.skel.name_to_idx.get("head")
        neck_i = self.skel.name_to_idx.get("neck_01")
        head_ori_active = (HEAD_ORIENTATION_MODE != "neck_follow"
                           and head_i is not None and neck_i is not None
                           and "nose" in measured
                           and head_i in self.skel.coco_anchor
                           and w[head_i] > 0.0)

        # Collect face keypoints for Kabsch head orientation.
        face_kpts = {}
        for k in ("nose", "left_eye", "right_eye", "left_ear", "right_ear"):
            if k in measured and np.all(np.isfinite(measured[k])):
                face_kpts[k] = measured[k]
        use_kabsch = False  # disabled: replaced by frame-based head orientation

        # ── Frame-based head target ─────────────────────────────────────
        # Build observed head frame from face keypoints, compute delta from
        # rest frame. This gives real head tracking (yaw/pitch/roll) from
        # data without Kabsch or nose-direction.
        head_target_quat = None
        if HEAD_ORIENTATION_MODE != "neck_follow" and head_ori_active and "left_ear" in face_kpts and "right_ear" in face_kpts:
            la = face_kpts["left_ear"]
            ra = face_kpts["right_ear"]
            face_pts = [face_kpts[k] for k in ("nose", "left_eye", "right_eye")
                        if k in face_kpts]
            if len(face_pts) >= 2 and np.linalg.norm(la - ra) > 5.0:
                x = la - ra
                x = x / np.linalg.norm(x)
                face_mid = np.mean(face_pts, axis=0)
                ear_mid = (la + ra) / 2.0
                f_raw = face_mid - ear_mid
                f = f_raw - np.dot(f_raw, x) * x
                fn = np.linalg.norm(f)
                if fn > 1e-5:
                    f = f / fn
                    u = np.cross(f, x)
                    u = u / (np.linalg.norm(u) + 1e-9)
                    F_obs = np.column_stack([x, f, u])
                    F_obs[:, 2] *= -1  # force right-handed (matches data chirality)
                    if np.linalg.det(F_obs) > 0:
                        R_head = self._head_target_rotation(F_obs)
                        head_target_quat = R_head.as_quat()

        # Canonical head template (kept for backward compat).
        from aimocap.math.kabsch import CANONICAL_HEAD_TEMPLATE

        # Global clavicle references are cached only for the opt-in branch.
        # The observed failure is world-space angular HF, so regularize the
        # joint's global SO(3) while retaining the same log(R_prev^-1 R)
        # Tikhonov form and weak analytic-init/rest anchor.
        redundant_prev_global_quats = None
        redundant_init_global_quats = None
        if self.redundant_dof_prior == "minimal":
            prior_joints = [
                int(self.skel.name_to_idx[name])
                for name in ("clavicle_l", "clavicle_r")
                if name in self.skel.name_to_idx
            ]
            if prev_x is not None:
                prev_root, prev_local = self._state_to_local_rotations(prev_x)
                _, prev_global = self.forward_kinematics(prev_root, prev_local)
                redundant_prev_global_quats = prev_global
            if init_x is not None:
                init_root, init_local = self._state_to_local_rotations(init_x)
                _, init_global = self.forward_kinematics(init_root, init_local)
                redundant_init_global_quats = init_global
        else:
            prior_joints = []

        return {
            "frame_idx": int(frame_idx),
            "tgt_pos": tgt_pos,
            "w": w.copy(),
            "R_root_quat": R_root.as_quat(),
            "R_diff_quat": R_diff.as_quat(),
            "R_target_quats": R_target_quats,
            "spine_indices": spine_indices,
            "n_spine": n_spine,
            "lat_axis": lat_axis,
            "prev_lat": prev_lat,
            "init_x": init_x,
            "prev_x": prev_x,
            "prev_prev_x": prev_prev_x,
            "accel_weight": accel_weight,
            "temporal_weight": temporal_weight,
            "init_weight": 1e-3,
            # VQ-FIX1: unit-equivalent orientation weight (see ORI_WEIGHT_SCALE).
            "ori_weight": (1.0 if ori_inputs_finite else 0.0) * ORI_WEIGHT_SCALE,
            "limb_pinned_info": limb_pinned_info,
            "face_position_weight": self.face_position_weight,

            "limb_twist_anchor": self.limb_twist_anchor,
            "limb_twist_pin": self.limb_twist_pin,
            "foot_target_gates": foot_target_gates,
            "foot_transport_pins": foot_transport_pins,
            "foot_landmark_packet": np.asarray(foot_landmark_packet, dtype=np.float64),
            "foot_landmark_weight": self.foot_landmark_weight,
            "head_ori_active": head_ori_active,
            "head_idx": head_i,
            "neck_idx": neck_i,
            "pelvis_idx": self.skel.name_to_idx.get("pelvis"),
            "head_rest_dir": self.skel.rest_offsets[head_i] if head_i is not None else None,
            "nose": measured.get("nose"),
            "head_target_quat": head_target_quat,
            "head_target_mode": self.head_target_mode,
            "use_kabsch": use_kabsch,
            "face_kpts": face_kpts,
            "canonical_head": CANONICAL_HEAD_TEMPLATE,
            "num_joints": self.num_joints,
            "rest_t0": self.skel.rest_t[0].copy(),
            "rest_offsets": self.rest_offsets.copy(),
            "parents": self.parents.copy(),
            "twist_limits": self._twist_limits,
            "twist_joint_indices": self._twist_joint_indices,
            "twist_bone_axes": self._twist_bone_axes,
            "twist_limit_mode": self.twist_limit_mode,
            "limit_projection": self.limit_projection,
            "twist_soft_joints": self.soft_limit_joints,
            "twist_safety_margin_deg": TWIST_HYBRID_SAFETY_MARGIN_DEG,
            "twist_soft_band_rad": TWIST_SOFT_BAND_RAD,
            "twist_soft_k": TWIST_SOFT_K,
            # Opt-in only: the default context carries no active prior terms.
            "redundant_dof_prior": self.redundant_dof_prior,
            "redundant_dof_joints": prior_joints,
            "redundant_dof_prev_global_quats": redundant_prev_global_quats,
            "redundant_dof_init_global_quats": redundant_init_global_quats,
            "redundant_dof_kc": REDUNDANT_DOF_PRIOR_KC,
            "redundant_dof_kr": REDUNDANT_DOF_PRIOR_KR,
            "hinge_joints": self._hinge_joints if self.limb_hinge != "off" else [],
        }

    def _residuals_with_ctx(self, x: np.ndarray, ctx: dict) -> np.ndarray:
        """Hot-path residual using precomputed frame context.

        Produces identical output to _residuals(x, measured, prev_x, ...)
        but skips recomputing R_root, tgt_pos, w, R_targets, etc.
        """
        root_t, local_rotations = self._state_to_local_rotations(x)
        global_pos, global_rot_quat = self.forward_kinematics(root_t, local_rotations)

        tgt_pos = ctx["tgt_pos"]
        w = ctx["w"].copy()
        ori_weight = ctx["ori_weight"]

        # D1: NaN targets must not poison the position residual — with w=0
        # the contribution is (gpos - tgt)*0, but NaN*0 = NaN in IEEE.
        # Replace NaN targets with 0 so w=0 joints contribute exactly 0.
        # Finite targets are untouched (bit-exact).
        if not np.isfinite(tgt_pos).all():
            tgt_pos = np.where(np.isfinite(tgt_pos), tgt_pos, 0.0)

        # Head position gate (x-dependent — must stay in hot path)
        head_i = ctx["head_idx"]
        neck_i = ctx["neck_idx"]
        if (head_i is not None and neck_i is not None
                and ctx["head_ori_active"] and w[head_i] > 0.0):
            neck_pos = global_pos[neck_i]
            nose_pos = ctx["nose"]
            desired_world = nose_pos - neck_pos
            if np.linalg.norm(desired_world) > 1e-5:
                R_neck = Rotation.from_quat(global_rot_quat[neck_i])
                rest_dir = self.skel.rest_offsets[head_i]
                desired_pl = R_neck.inv().apply(desired_world)
                cos_accept = np.dot(
                    desired_pl / np.linalg.norm(desired_pl),
                    rest_dir / np.linalg.norm(rest_dir))
                if cos_accept <= np.cos(np.deg2rad(70.0)):
                    w[head_i] = 0.1

        if self.face_position_weight != 1.0 and head_i is not None:
            w[head_i] *= self.face_position_weight

        res_data = (global_pos - tgt_pos) * w[:, None]
        parts = [res_data.flatten()]

        if ori_weight > 0:
            # Spine orientation
            global_rots = Rotation.from_quat(global_rot_quat)
            n_spine = ctx["n_spine"]
            spine_indices = ctx["spine_indices"]
            R_target_quats = ctx["R_target_quats"]
            ori_res = np.zeros(n_spine * 3)
            for k in range(n_spine):
                i = spine_indices[k]
                R_target = Rotation.from_quat(R_target_quats[k])
                R_current = global_rots[i]
                ori_res[k * 3:(k + 1) * 3] = (
                    R_current * R_target.inv()).as_rotvec()
            parts.append(ori_res * ori_weight)

            # Head orientation: pin head global rotation to the frame-based
            # target computed from face keypoints (real head tracking).
            # Falls back to neck rotation (torso-follow) when face keypoints
            # are unavailable or degenerate.
            head_res = np.zeros(3)
            if ctx["head_ori_active"]:
                htq = ctx.get("head_target_quat")
                if htq is not None:
                    R_head_target = Rotation.from_quat(htq)
                else:
                    R_head_target = global_rots[neck_i]  # torso-follow fallback
                R_head_current = global_rots[head_i]
                head_res = (R_head_current * R_head_target.inv()).as_rotvec()
            parts.append(head_res * ori_weight * ctx.get("head_orientation_weight", 1.0))

            # Shoulder roll from arm triangle (Stage 3)
            # The arm triangle constrains UPPER-ARM axial roll (shoulder_* orientation).
            # This is the CORRECT use of the arm triangle — it constrains the
            # upper arm's axial roll, not forearm pronation or palm orientation.
            # We add a full SO(3) residual on the shoulder local rotation.
            shoulder_roll_targets = ctx.get("shoulder_roll_targets", {})
            shoulder_roll_active = ctx.get("shoulder_roll_active", {})
            for side in ("l", "r"):
                if not shoulder_roll_active.get(side, False):
                    continue
                targets = shoulder_roll_targets.get(side, {})
                if not targets.get("valid"):
                    continue
                arm_data = self._arm_rest.get(side)
                if not arm_data:
                    continue
                sh_i = arm_data["shoulder_i"]
                roll_axis_sh = arm_data["roll_axis_sh"]
                R_sh_parent = global_rots[self.parents[sh_i]]
                R_sh_local = R_sh_parent.inv() * global_rots[sh_i]
                # Full SO(3) residual on shoulder local rotation
                R_target_local = Rotation.from_quat(targets["target_local_quat"])
                R_err = R_sh_local * R_target_local.inv()
                r_sh = R_err.as_rotvec() * ori_weight
                parts.append(r_sh)

            # Sway amplitude + velocity
            pelvis_i = ctx["pelvis_idx"]
            neck_i_s = ctx["neck_idx"]
            if pelvis_i is not None and neck_i_s is not None:
                lat_axis = ctx["lat_axis"]
                neck_off = global_pos[neck_i_s] - global_pos[pelvis_i]
                lat = np.dot(neck_off, lat_axis)
                # SWAY_LIMIT=5cm (tightened from 8: real walking keeps the neck
                # within 5cm of the pelvis line; the measured shoulder midpoint
                # can be 40+cm off due to triangulation noise, and 8cm let too
                # much through).  SWAY_WEIGHT=60 (tripled from 20: the position
                # residual for the neck target is ~1.0 weight, so a 40cm outlier
                # costs 40 in position but only 20*(40-8)=640 in sway — but the
                # solver still compromised at 16cm.  60*(16-5)=660 makes it cost
                # as much as a 16cm position error, enough to actually clamp).
                excess = max(0.0, abs(lat) - 5.0) * np.sign(lat)
                parts.append(np.array([excess]) * 60.0)
                if ctx["prev_lat"] is not None:
                    v = lat - ctx["prev_lat"]
                    excess_v = max(0.0, abs(v) - 0.5) * np.sign(v)
                    parts.append(np.array([excess_v]) * 30.0)
                else:
                    parts.append(np.zeros(1))
            else:
                parts.append(np.zeros(1))
                if ctx["prev_lat"] is not None:
                    parts.append(np.zeros(1))

            # Limb twist
            limb_res = []
            for info in ctx["limb_pinned_info"]:
                p_parent = info["p_parent"]
                p = info["p"]
                bone = info["bone"]
                R_current = global_rots[p_parent].inv() * global_rots[p]
                rv = R_current.as_rotvec()
                twist_target = info.get("twist_target", 0.0)
                twist_weight = info.get("twist_weight", 1.0)
                pin_multiplier = info.get("twist_pin_multiplier", 1.0)
                if ctx.get("limb_twist_pin", "always") == "off":
                    continue
                if ctx.get("limb_twist_anchor") == "smoothed":
                    from aimocap.retarget.swing_twist import signed_twist_angle
                    twist_rad = float(signed_twist_angle(R_current, bone))
                    delta = (twist_rad - twist_target + np.pi) % (2.0 * np.pi) - np.pi
                    twist_error = delta * twist_weight * pin_multiplier
                else:
                    twist_rad = np.dot(rv, bone)
                    twist_error = (twist_rad - twist_target) * twist_weight * pin_multiplier
                limb_res.append(np.array([twist_error * bone[i] for i in range(3)]))
            if limb_res:
                parts.append(np.concatenate(limb_res) * ori_weight)

        # Soft swing-twist limits are part of the LM cost.  Use the same
        # local quaternion/axis pair as clamp_twist; unlike clamp mode this
        # branch never projects the state.
        limit_mode = ctx.get("twist_limit_mode", "clamp")
        if limit_mode in {"soft", "hybrid"}:
            from aimocap.retarget.swing_twist import signed_twist_angle as _signed_twist_angle
            soft_res = []
            soft_names = (set(ctx.get("twist_limits") or {})
                          if limit_mode == "soft" else set(ctx.get("twist_soft_joints") or ()))
            for jname, max_tw_deg in (ctx.get("twist_limits") or {}).items():
                if jname not in soft_names:
                    continue
                ji = ctx.get("twist_joint_indices", {}).get(jname)
                axis = ctx.get("twist_bone_axes", {}).get(jname)
                if ji is None or axis is None:
                    continue
                angle = float(_signed_twist_angle(
                    Rotation.from_rotvec(x[3 + ji * 3:3 + ji * 3 + 3]), axis))
                excess = abs(angle) - np.deg2rad(float(max_tw_deg))
                soft_res.append(float(soft_twist_hinge(excess)) * TWIST_SOFT_K)
            if soft_res:
                parts.append(np.asarray(soft_res, dtype=np.float64))

        # Limb hinge constraints
        hinge_joints = ctx.get("hinge_joints", [])
        if hinge_joints:
            hinge_parts_swing = []
            hinge_parts_range = []
            for hinge in hinge_joints:
                j = hinge["joint_idx"]
                h = hinge["axis"]
                flex_min = hinge["flex_min_rad"]
                flex_max = hinge["flex_max_rad"]
                
                # Get rotation vector for this joint
                rv = x[3+3*j:6+3*j]
                q = Rotation.from_rotvec(rv).as_quat()
                
                # Ensure positive scalar part (canonical form)
                if q[3] < 0:
                    q = -q
                
                # Extract twist component
                v = q[:3]
                w = q[3]
                proj = np.dot(v, h)
                theta = 2.0 * np.arctan2(proj, w)
                
                # Twist quaternion
                t = np.concatenate([proj * h, [w]])
                t_norm = np.linalg.norm(t)
                if t_norm > 1e-12:
                    t = t / t_norm
                
                # Swing = q * conj(t)
                q_rot = Rotation.from_quat(q)
                t_rot = Rotation.from_quat(t)
                swing = (q_rot * t_rot.inv()).as_quat()
                
                # Swing residual
                res_swing = 30.0 * Rotation.from_quat(swing).as_rotvec()
                hinge_parts_swing.append(res_swing)
                
                # Range residual
                res_range_min = max(0, flex_min - theta)
                res_range_max = max(0, theta - flex_max)
                res_range = 30.0 * np.array([res_range_min, res_range_max])
                hinge_parts_range.append(res_range)
            
            if hinge_parts_swing:
                parts.append(np.concatenate(hinge_parts_swing))
                parts.append(np.concatenate(hinge_parts_range))

        # VQ-FIX6 transport pin is a local SO(3) residual and is independent
        # of the torso-orientation validity gate. Its 30x scale matches the
        # existing limb-twist pin's orientation-residual scale.
        for pin in ctx.get("foot_transport_pins", ()):
            idx = pin["joint_idx"]
            current_local = Rotation.from_quat(local_rotations[idx])
            target_local = Rotation.from_quat(pin["target_quaternion"])
            parts.append((target_local.inv() * current_local).as_rotvec()
                         * ORI_WEIGHT_SCALE * pin.get("weight", 1.0))

        if ctx.get("foot_landmark_packet") is not None:
            packet = np.asarray(ctx["foot_landmark_packet"], dtype=np.float64)
            if packet.size:
                bone = packet[:, 0].astype(np.int64)
                pred = global_pos[bone] + Rotation.from_quat(global_rot_quat[bone]).apply(packet[:, 4:7])
                residual = (pred - packet[:, 1:4]) * packet[:, 7, None]
                norm = np.linalg.norm(residual, axis=1, keepdims=True)
                factor = np.sqrt(np.minimum(1.0, FOOT_LANDMARK_HUBER_DELTA_CM / (norm + 1e-9)))
                shaped = (residual * factor).reshape(-1)
                if self.foot_landmark_weight != 1.0:
                    shaped *= self.foot_landmark_weight
                parts.append(shaped)

        # Opt-in redundant-DOF Tikhonov prior.  Clavicle local rotations are
        # weakly/partly observable through the upper-arm endpoint, while axial
        # motion remains a null direction.  Penalize SO(3) change from the
        # previous solved state and add a weak pull to this frame's analytic
        # init (the rest-pose gauge when the analytic init is rest).
        if ctx.get("redundant_dof_prior", "none") == "minimal":
            current_global = Rotation.from_quat(global_rot_quat)
            prev_global_q = ctx.get("redundant_dof_prev_global_quats")
            init_global_q = ctx.get("redundant_dof_init_global_quats")
            prev_global = Rotation.from_quat(prev_global_q) if prev_global_q is not None else None
            init_global = Rotation.from_quat(init_global_q) if init_global_q is not None else None
            for ji in ctx.get("redundant_dof_joints", ()):
                cur = current_global[ji]
                if prev_global is not None:
                    parts.append((prev_global[ji].inv() * cur).as_rotvec()
                                 * ctx["redundant_dof_kc"])
                if init_global is not None:
                    parts.append((init_global[ji].inv() * cur).as_rotvec()
                                 * ctx["redundant_dof_kr"])

        # Temporal + init
        if ctx["prev_x"] is not None and ctx["temporal_weight"] > 0:
            parts.append(self._state_delta(x, ctx["prev_x"]) * ctx["temporal_weight"])
        if ctx["init_x"] is not None:
            parts.append(self._state_delta(x, ctx["init_x"]) * ctx["init_weight"])
        
        r = np.concatenate(parts)
        # Residual length assertion — catches silent changes when residuals
        # are added/removed in only one code path (scipy vs torch).
        n_exp = ctx.setdefault("_resid_len", r.size)
        if r.size != n_exp:
            raise RuntimeError(f"residual length changed: {r.size} != {n_exp}")
        return r

    def _residuals(self, x: np.ndarray, measured: dict,
                   prev_x: np.ndarray = None, temporal_weight: float = 0.0,
                   init_x: np.ndarray = None, init_weight: float = 1e-3,
                   ori_weight: float = 1.0, prev_lat: float = None) -> np.ndarray:
        root_t, local_rotations = self._state_to_local_rotations(x)
        global_pos, global_rot_quat = self.forward_kinematics(root_t, local_rotations)

        spine_names = self.skel.topo.spine_chain("pelvis", "neck_01")

        hip_line = measured["hip_r"] - measured["hip_l"]
        spine_dir = measured["neck"] - measured["pelvis"]

        idx_hl = self.skel.name_to_idx["hip_l"]
        idx_hr = self.skel.name_to_idx["hip_r"]
        idx_sl = self.skel.name_to_idx["shoulder_l"]
        idx_sr = self.skel.name_to_idx["shoulder_r"]

        rest_hip = self.skel.rest_t[idx_hr] - self.skel.rest_t[idx_hl]
        rest_pelvis_mid = (self.skel.rest_t[idx_hl] + self.skel.rest_t[idx_hr]) / 2.0
        rest_neck_mid = (self.skel.rest_t[idx_sl] + self.skel.rest_t[idx_sr]) / 2.0
        rest_spine = rest_neck_mid - rest_pelvis_mid

        from aimocap.retarget.root_frame import root_rotation
        R_root = root_rotation(spine_dir, hip_line, rest_spine, rest_hip)

        shoulder_line = measured["shoulder_r"] - measured["shoulder_l"]
        rest_shoulder = self.skel.rest_t[idx_sr] - self.skel.rest_t[idx_sl]
        R_upper = root_rotation(spine_dir, shoulder_line, rest_spine, rest_shoulder)

        # Torso twist in the hip local frame (same convention as analytic_init).
        R_diff = R_root.inv() * R_upper

        # The pelvis and neck_01 targets are offset to joint positions. See
        # analytic_init for the explanation of why the offset is needed.
        rest_pelvis_mid = (self.skel.rest_t[idx_hl] + self.skel.rest_t[idx_hr]) / 2.0
        rest_neck_mid = (self.skel.rest_t[idx_sl] + self.skel.rest_t[idx_sr]) / 2.0
        pelvis_offset = self.skel.rest_t[0] - rest_pelvis_mid
        neck_offset = self.skel.rest_t[self.skel.name_to_idx["neck_01"]] - rest_neck_mid
        tgt_pelvis = measured["pelvis"] + R_root.apply(pelvis_offset)
        tgt_neck = measured["neck"] + R_upper.apply(neck_offset)
        
        from aimocap.retarget.spine_chain import distribute_spine_targets
        rest_positions = np.array([self.skel.rest_t[self.skel.name_to_idx[nm]] for nm in spine_names])
        inter = distribute_spine_targets(tgt_pelvis, tgt_neck, rest_positions)
        
        tgt_pos = np.zeros((self.num_joints, 3))
        si = 0
        for k, nm in enumerate(spine_names):
            i = self.skel.name_to_idx[nm]
            if k == 0:
                tgt_pos[i] = tgt_pelvis
            elif k == len(spine_names) - 1:
                tgt_pos[i] = tgt_neck
            else:
                tgt_pos[i] = inter[si]; si += 1
            
        for i, nm in self.skel.coco_anchor.items():
            if self.skel.joint_names[i] not in ["pelvis", "neck_01"]:
                if nm in measured:
                    tgt_pos[i] = measured[nm]
                # If nm not in measured (e.g. 17-joint input without foot
                # keypoints), tgt_pos[i] stays zero — the toe fallback below
                # will synthesize from R_root.

        # Clavicle targets: synthesize from the neck (see analytic_init for
        # the explanation). Without this, tgt_pos stays 0 and the solver pulls
        # the clavicles to the origin -> caved-in shoulders.
        neck_idx_r = self.skel.name_to_idx["neck_01"]
        for side in ("l", "r"):
            clav_name = f"clavicle_{side}"
            if clav_name in self.skel.name_to_idx:
                ci = self.skel.name_to_idx[clav_name]
                if np.allclose(tgt_pos[ci], 0.0):
                    rest_off = self.skel.rest_t[ci] - self.skel.rest_t[neck_idx_r]
                    tgt_pos[ci] = tgt_neck + R_upper.apply(rest_off)

        # Toe (ball_l/ball_r) targets: when the big-toe keypoint is measured
        # (COCO-WholeBody 17/20), the coco_anchor loop above already set the
        # target from measurement — use it, so foot orientation comes from
        # actual detected foot position, not the pelvis root frame. When the
        # big-toe is NaN (not detected) or toe_* is not coco-anchored, fall
        # back to R_root synthesis so the ankle roll still has a constraint.
        for side in ("l", "r"):
            toe_name = f"toe_{side}"
            ankle_name = f"ankle_{side}"
            if toe_name in self.skel.name_to_idx and ankle_name in self.skel.name_to_idx:
                toe_i = self.skel.name_to_idx[toe_name]
                ankle_i = self.skel.name_to_idx[ankle_name]
                toe_anchor = self.skel.coco_anchor.get(toe_i)
                needs_synth = (toe_anchor is None
                               or toe_anchor not in measured
                               or np.any(np.isnan(tgt_pos[toe_i])))
                if needs_synth:
                    rest_off = self.skel.rest_t[toe_i] - self.skel.rest_t[ankle_i]
                    tgt_pos[toe_i] = tgt_pos[ankle_i] + R_root.apply(rest_off)

        # Dorsiflexion clamp (see analytic_init for the full explanation):
        # prevent the toe target from landing above the ankle/heel, which
        # dorsiflexes the foot ("on heels").  Uses the heel keypoint when
        # available (133-joint data) or falls back to the ankle Z (17-joint
        # data, where the toe is synthesized from R_root and tilts with the
        # torso).  The toe can't be above the ankle in a flat foot.
        for side in ("l", "r"):
            toe_name = f"toe_{side}"
            ankle_name = f"ankle_{side}"
            heel_key = f"heel_{side}"
            if (toe_name not in self.skel.name_to_idx
                    or ankle_name not in self.skel.name_to_idx):
                continue
            toe_i = self.skel.name_to_idx[toe_name]
            ankle_i = self.skel.name_to_idx[ankle_name]
            if not np.all(np.isfinite(tgt_pos[toe_i])):
                continue
            if (heel_key in measured
                    and np.all(np.isfinite(measured[heel_key]))):
                max_toe_z = measured[heel_key][2] + 1.0
            else:
                max_toe_z = tgt_pos[ankle_i][2]
            if tgt_pos[toe_i][2] > max_toe_z:
                tgt_pos[toe_i][2] = max_toe_z

        # Per-joint weights: 1.0 for measured (coco-anchored), 0.3 for synthetic
        # spine intermediates (they guide bend but shouldn't dominate), 0.0 for
        # joints with no real target (e.g. clavicles: not coco-anchored, not in
        # the spine chain, so tgt_pos stays 0 -> a nonzero weight would pull them
        # to the origin and cave the shoulders inward).
        w = np.zeros(self.num_joints)
        for i, val in enumerate(tgt_pos):
            if np.any(np.isnan(val)):
                w[i] = 0.0
            elif i not in self.skel.coco_anchor:
                if np.allclose(val, 0.0):
                    # No target was ever set for this joint (not spine, not
                    # coco-anchored, not synthesized). Don't pull it to the
                    # origin -- let its measured children drive its position.
                    w[i] = 0.0
                else:
                    # Spine intermediates and neck_01: moderate weight so the
                    # solver distributes the bend across spine joints instead
                    # of concentrating it at the pelvis.
                    w[i] = 0.15
            else:
                w[i] = 1.0

        if self.foot_landmark_weight != 1.0:
            for side in ("l", "r"):
                toe_i = self.skel.name_to_idx.get(f"toe_{side}")
                if toe_i is not None and toe_i in self.skel.coco_anchor:
                    w[toe_i] *= self.foot_landmark_weight

        # Head position plausibility gate: the head's POSITION target comes
        # from the measured nose (coco_anchor -> w=1.0 above), but the head
        # ORIENTATION residual below is already gated by a 70 deg check.
        # Without gating the POSITION too, the solver hard-pins the head to a
        # noisy nose position, forcing the neck/spine to swing laterally to
        # reach it -> the "head always moves left or right" artifact.  Apply
        # the same 70 deg gate: when the measured nose direction (neck->nose,
        # expressed in the neck's local frame) is >70 deg from the rest head
        # direction, down-weight the head position so the neck/spine stay
        # neutral instead of swinging.  We down-weight (0.1) rather than zero
        # so a genuinely turned head still biases the solve slightly.
        head_i = self.skel.name_to_idx.get("head")
        neck_i = self.skel.name_to_idx.get("neck_01")
        if (head_i is not None and neck_i is not None
                and "nose" in measured and head_i in self.skel.coco_anchor
                and w[head_i] > 0.0):
            neck_pos = global_pos[neck_i]
            nose_pos = measured["nose"]
            desired_world = nose_pos - neck_pos
            if np.linalg.norm(desired_world) > 1e-5:
                R_neck = Rotation.from_quat(global_rot_quat[neck_i])
                rest_dir = self.skel.rest_offsets[head_i]
                desired_pl = R_neck.inv().apply(desired_world)
                cos_accept = np.dot(
                    desired_pl / np.linalg.norm(desired_pl),
                    rest_dir / np.linalg.norm(rest_dir))
                if cos_accept <= np.cos(np.deg2rad(70.0)):
                    w[head_i] = 0.1   # implausible nose -> don't swing the neck

        if self.face_position_weight != 1.0 and head_i is not None:
            w[head_i] *= self.face_position_weight


        res_data = (global_pos - tgt_pos) * w[:, None]
        parts = [res_data.flatten()]

        # Orientation residual for spine joints: pin the global rotation of
        # each spine joint to its SLERP target (R_root * R_diff^frac).  Without
        # this, rotation about a spine bone's long axis is a null direction of
        # the position Jacobian (it moves no joint position), so the solver
        # is free to drift the twist arbitrarily.  This residual breaks that
        # gauge freedom: the solver can still adjust the spine to fit
        # positions, but it can no longer add twist for free.  ori_weight=1.0
        # means a 1-radian (~57 deg) twist drift costs as much as a 1-cm
        # position error, so ~5 deg of twist is an acceptable trade for 1 cm
        # of position — tight enough to prevent the ~83 deg over-rotation but
        # loose enough to let the spine curve.
        if ori_weight > 0:
            n_spine = len(spine_names)
            ori_res = np.zeros(n_spine * 3)
            global_rots = Rotation.from_quat(global_rot_quat)
            for k, nm in enumerate(spine_names):
                i = self.skel.name_to_idx[nm]
                frac = k / (n_spine - 1)
                R_target = R_root * (R_diff ** frac)
                R_current = global_rots[i]
                ori_res[k * 3:(k + 1) * 3] = (
                    R_current * R_target.inv()
                ).as_rotvec()
            parts.append(ori_res * ori_weight)

            # Head orientation: pin the head's global rotation toward the
            # nose direction.  The head is a leaf with a COCO anchor (nose);
            # without this, the solver only sees the head's POSITION (which
            # doesn't constrain roll about the neck->nose axis), so the head
            # can tilt sideways.  The target direction is from neck to nose,
            # and we pin the head's global rotation to match that direction
            # using the same swing-twist approach as analytic_init.
            head_i = self.skel.name_to_idx.get("head")
            neck_i = self.skel.name_to_idx.get("neck_01")
            if (head_i is not None and neck_i is not None
                    and "nose" in measured and head_i in self.skel.coco_anchor):
                neck_pos = global_pos[neck_i]
                nose_pos = measured["nose"]
                desired_world = nose_pos - neck_pos
                rest_dir = self.skel.rest_offsets[head_i]
                # Same plausibility gate as analytic_init: only apply the head
                # orientation residual when the measured nose direction is
                # within ~70 deg of the rest head direction.  A noisy nose
                # triangulation (e.g. landing 23 cm above the neck) would
                # otherwise pin the head to a sideways-facing target.  The
                # residual vector length must stay constant across solve
                # iterations, so when the gate fails we emit a zero residual
                # (no contribution) instead of skipping it.
                head_res = np.zeros(3)
                if np.linalg.norm(desired_world) > 1e-5:
                    R_neck = global_rots[neck_i]
                    desired_pl = R_neck.inv().apply(desired_world)
                    cos_accept = np.dot(
                        desired_pl / np.linalg.norm(desired_pl),
                        rest_dir / np.linalg.norm(rest_dir))
                    if cos_accept > np.cos(np.deg2rad(70.0)):
                        from aimocap.retarget.swing_twist import constrained_rotation
                        R_head_target = R_neck * constrained_rotation(rest_dir, desired_pl)
                        R_head_current = global_rots[head_i]
                        head_res = (R_head_current * R_head_target.inv()).as_rotvec()
                parts.append(head_res * ori_weight * ctx.get("head_orientation_weight", 1.0))

            # Spine lateral sway damping: the measured shoulders swing
            # laterally frame-to-frame (shoulder_l std ~9 cm on this clip),
            # which tilts spine_dir and R_root sideways -> the spine chain
            # faithfully reproduces a left-right oscillation that reads as
            # "the head keeps moving left-right".  Real human walking has
            # only ~3 deg of lateral spine bend; the measured noise drives
            # up to 10 deg, oscillating 16 times in 300 frames.  Penalize
            # the neck's lateral offset from the pelvis beyond a soft
            # threshold (~8 cm ~= 3 deg on a 153 cm spine).  The lateral
            # axis is the REST hip-line direction (the rig's anatomical
            # left-right), NOT the measured hip line (which is noisy and
            # nearly perpendicular to the neck-pelvis offset, so it would
            # measure ~0).  Using the rest hip line keeps this general
            # (works for any rig, no hardcoded world axes).  This damps the
            # oscillation at the IK level without lagging real motion the
            # way a pre-IK keypoint filter would.
            pelvis_i = self.skel.name_to_idx.get("pelvis")
            neck_i_s = self.skel.name_to_idx.get("neck_01")
            if pelvis_i is not None and neck_i_s is not None:
                lat_axis = rest_hip / (np.linalg.norm(rest_hip) + 1e-9)
                neck_off = global_pos[neck_i_s] - global_pos[pelvis_i]
                lat = np.dot(neck_off, lat_axis)
                SWAY_LIMIT = 5.0  # cm; ~3 deg lateral bend (biomechanical max)
                excess = max(0.0, abs(lat) - SWAY_LIMIT) * np.sign(lat)
                # sway_weight >> ori_weight: the sway residual is a single
                # scalar competing against ~24*3 position terms that pull
                # toward the noisy measured neck.  A high weight clamps the
                # lateral sway hard (kills the left-right oscillation).
                # WEIGHT=60: at 16cm offset, cost=60*(16-5)=660, comparable
                # to the ~16*3=48 per-joint position error the solver trades
                # against.  WEIGHT=20 was too low: 20*(16-8)=160 was dominated
                # by position terms, so the solver compromised at 16cm.
                SWAY_WEIGHT = 60.0
                parts.append(np.array([excess]) * SWAY_WEIGHT)

                # Neck lateral velocity damping: complement to the sway
                # amplitude clamp above.  The amplitude clamp (SWAY_LIMIT=5cm)
                # kills sustained sideways lean but a 4cm oscillation at 1.6 Hz
                # would pass the 5cm gate and still read as "head keeps moving
                # left-right".  Real walking has slow lateral sway (~0.3 Hz,
                # <0.5 cm/frame); the noise-driven oscillation is ~1.6 Hz
                # (>1 cm/frame).  This penalizes rapid lateral CHANGE, targeting
                # the frequency directly.  Reuses `lat` (computed above) and
                # `prev_lat` (computed once per frame in solve_frame, not per
                # residual eval — prev_x is constant across all evals within
                # one solve_frame call, so one extra FK per frame is negligible).
                if prev_lat is not None:
                    VEL_LIMIT = 0.5   # cm/frame dead-zone (~15 cm/s at 30fps)
                    v = lat - prev_lat
                    excess_v = max(0.0, abs(v) - VEL_LIMIT) * np.sign(v)
                    VEL_WEIGHT = 30.0
                    parts.append(np.array([excess_v]) * VEL_WEIGHT)
                else:
                    parts.append(np.zeros(1))
            else:
                parts.append(np.zeros(1))
                if prev_lat is not None:
                    parts.append(np.zeros(1))

            # Limb orientation: penalize twist about each limb bone's long
            # axis.  Twist about a limb bone's long axis is a null direction of
            # the position residual (it moves no joint position), so without
            # this the solver is free to drift the calf/thigh/forearm twist by
            # 60-100 deg to save a fraction of a cm elsewhere -- the visible
            # "caved-in" knee or spinning hand.  We project the bone's local
            # rotation rotvec onto its rest long axis and penalize that
            # component.  The residual is RELAXED (skipped) only when the
            # roll child is a real measurement AND is well-conditioned (the
            # roll-child direction is >35 deg from the bone -- i.e. the limb
            # is bent enough that the roll child genuinely constrains roll).
            # When the roll child is synthesized (toe from R_root) or the limb
            # is near-straight (arm bone-vs-hand <35 deg), any non-zero twist
            # is garbage and we pin it to zero.  ori_weight=1.0 makes ~5 deg
            # of drift cost as much as 1 cm of position error.
            limb_res = []
            for side in ("l", "r"):
                for p_name, c_name in (
                    (f"hip_{side}", f"knee_{side}"),     # thigh
                    (f"knee_{side}", f"ankle_{side}"),   # shin
                    (f"shoulder_{side}", f"elbow_{side}"),  # upper arm
                    (f"elbow_{side}", f"wrist_{side}"),     # forearm
                ):
                    if p_name not in self.skel.name_to_idx:
                        continue
                    p = self.skel.name_to_idx[p_name]
                    c = self.skel.name_to_idx[c_name]
                    # The roll child constrains roll ONLY if it's a real
                    # measurement AND has proxy children (intermediate joint,
                    # not a chain-end leaf).  A chain-end roll child (e.g. the
                    # wrist for the forearm) has its position fully determined
                    # by the chain, so it carries no independent roll info ->
                    # pin twist to zero.  See analytic_init for the full
                    # explanation.  When the roll child is synthesized (toe from
                    # R_root) or is a leaf, any non-zero twist is garbage.
                    rig_name_c = self.skel.fbx_mapping[c]
                    rc_name = self.skel.topo.roll_child(rig_name_c)
                    rc_well_conditioned = False
                    if rc_name and rc_name in self.skel.fbx_mapping.values():
                        rc_proxy = next(idx for idx, rn in self.skel.fbx_mapping.items()
                                        if rn == rc_name)
                        rc_coco = self.skel.coco_anchor.get(rc_proxy)
                        rc_measured = (rc_coco is not None
                                       and rc_coco in measured
                                       and np.all(np.isfinite(measured[rc_coco]))
                                       and np.linalg.norm(measured[rc_coco]) > 1e-5)
                        rc_children = [k for k in range(self.num_joints)
                                       if self.parents[k] == rc_proxy]
                        if rc_measured and rc_children:
                            # Even a measured intermediate roll child can't
                            # constrain roll when the limb is nearly straight
                            # (roll-child direction ≈ bone direction -> the
                            # perpendicular projection is noise). Check the
                            # bone-vs-rollchild angle in the current pose; if
                            # <35 deg, pin twist to zero (same threshold as
                            # constrained_rotation's near-parallel guard).
                            bone_dir = tgt_pos[c] - tgt_pos[p]
                            rc_dir = tgt_pos[rc_proxy] - tgt_pos[c]
                            bn = np.linalg.norm(bone_dir)
                            rn = np.linalg.norm(rc_dir)
                            if bn > 1e-5 and rn > 1e-5:
                                cos_ang = np.clip(
                                    np.dot(bone_dir / bn, rc_dir / rn), -1, 1)
                                rc_well_conditioned = np.degrees(
                                    np.arccos(cos_ang)) > 35.0
                    if rc_well_conditioned:
                        continue  # roll child constrains roll -> let solver refine
                    rest_off = self.skel.rest_offsets[c]
                    bone = rest_off / (np.linalg.norm(rest_off) + 1e-12)
                    p_parent = self.parents[p]
                    R_current = global_rots[p_parent].inv() * global_rots[p]
                    rv = R_current.as_rotvec()
                    twist_rad = np.dot(rv, bone)  # signed twist about bone
                    limb_res.append(np.array([twist_rad * bone[i] for i in range(3)]))
            if limb_res:
                parts.append(np.concatenate(limb_res) * ori_weight)

        if prev_x is not None and temporal_weight > 0:
            parts.append(self._state_delta(x, prev_x) * temporal_weight)
        if init_x is not None:
            parts.append(self._state_delta(x, init_x) * init_weight)
        return np.concatenate(parts)

    def _record_solve_stats(self, x: np.ndarray, ctx: dict) -> dict:
        """Record backend termination and per-frame constrained-twist diagnostics."""
        from aimocap.retarget.swing_twist import signed_twist_angle
        twists = {}
        rv = np.asarray(x[3:], dtype=np.float64).reshape(-1, 3)
        limit_mode = ctx.get("twist_limit_mode", self.twist_limit_mode)
        soft_names = set(ctx.get("twist_soft_joints") or ())
        safety_margin = float(ctx.get("twist_safety_margin_deg", TWIST_HYBRID_SAFETY_MARGIN_DEG))
        self._twist_limit_total_frames += 1
        for jname, max_tw in (ctx.get("twist_limits") or {}).items():
            ji = ctx.get("twist_joint_indices", {}).get(jname)
            axis = ctx.get("twist_bone_axes", {}).get(jname)
            if ji is None or axis is None:
                continue
            angle_deg = float(np.degrees(
                signed_twist_angle(Rotation.from_rotvec(rv[ji]), axis)))
            soft_joint = limit_mode == "hybrid" and jname in soft_names
            safety_limit = float(max_tw) + safety_margin if soft_joint else float(max_tw)
            at_or_over = bool(abs(angle_deg) >= float(max_tw) - 1e-7)
            at_safety = bool(soft_joint and abs(angle_deg) >= safety_limit - 1e-7)
            twists[jname] = {
                "angle_deg": angle_deg,
                "limit_deg": float(max_tw),
                "safety_limit_deg": safety_limit,
                "soft_joint": bool(soft_joint),
                "exceeds_limit": bool(abs(angle_deg) > float(max_tw)),
                "at_or_over_limit": at_or_over,
                "at_safety_clamp": at_safety,
            }
            counts = self._twist_limit_counts.setdefault(
                jname, {"frames": 0, "at_or_over_limit": 0, "at_safety_clamp": 0})
            counts["frames"] += 1
            counts["at_or_over_limit"] += int(at_or_over)
            counts["at_safety_clamp"] += int(at_safety)
        backend = dict(getattr(self, "_last_backend_stats", {}) or {})
        stats = {
            "frame": int(ctx.get("frame_idx", 0)),
            "backend": backend.get("backend"),
            "termination_reason": backend.get("termination_reason", "unknown"),
            "iterations": int(backend.get("iterations", 0) or 0),
            "final_cost": backend.get("final_cost"),
            "twist_limit_mode": self.twist_limit_mode,
            "soft_limit_joints": sorted(self.soft_limit_joints),
            "twist_safety_margin_deg": safety_margin,
            "redundant_dof_prior": self.redundant_dof_prior,
            "twists": twists,
        }
        self._last_solve_stats = stats
        return stats

    def reset_twist_limit_diagnostics(self) -> None:
        """Reset frame counters when a two-pass solve starts its final pass."""
        self._twist_limit_total_frames = 0
        self._twist_limit_counts = {
            name: {"frames": 0, "at_or_over_limit": 0, "at_safety_clamp": 0}
            for name in self._twist_limits
        }

    def twist_limit_diagnostics(self) -> dict:
        """Return aggregate fractions for every limited joint."""
        total = max(int(self._twist_limit_total_frames), 1)
        joints = {}
        for name in sorted(self._twist_limits):
            counts = self._twist_limit_counts.get(name, {})
            frames = int(counts.get("frames", 0))
            denom = max(frames, 1)
            joints[name] = {
                "frames": frames,
                "at_or_over_limit_frames": int(counts.get("at_or_over_limit", 0)),
                "at_or_over_limit_fraction": float(counts.get("at_or_over_limit", 0) / denom),
                "at_safety_clamp_frames": int(counts.get("at_safety_clamp", 0)),
                "at_safety_clamp_fraction": float(counts.get("at_safety_clamp", 0) / denom),
                "limit_deg": float(self._twist_limits[name]),
                "safety_limit_deg": float(self._twist_limits[name] + TWIST_HYBRID_SAFETY_MARGIN_DEG
                                             if self.twist_limit_mode == "hybrid" and name in self.soft_limit_joints
                                             else self._twist_limits[name]),
                "soft_joint": bool(self.twist_limit_mode == "hybrid" and name in self.soft_limit_joints),
            }
        return {
            "mode": self.twist_limit_mode,
            "soft_limit_joints": sorted(self.soft_limit_joints),
            "safety_margin_deg": TWIST_HYBRID_SAFETY_MARGIN_DEG,
            "total_frames": int(self._twist_limit_total_frames),
            "joints": joints,
        }

    def solve_frame(self, measured: dict, prev_x: np.ndarray = None,
                    temporal_weight: float = 0.03, frame_idx: int = 0,
                    prev_prev_x: np.ndarray = None,
                    accel_weight: float = 0.0,
                    x0: np.ndarray | None = None,
                    max_iter: int | None = None) -> np.ndarray:
        if max_iter is not None:
            if (not isinstance(max_iter, (int, np.integer)) or isinstance(max_iter, bool)
                    or max_iter < 1):
                raise ValueError(f"max_iter must be None or a positive integer, got {max_iter!r}")
        if x0 is None:
            if prev_x is None:  # re-init from measured if no prev_x
                x0 = self.analytic_init(measured)
            else:
                x0 = prev_x.copy()
        else:
            x0 = np.asarray(x0, dtype=np.float64).copy()
            if x0.shape != (self.num_vars,):
                raise ValueError(
                    f"x0 must have shape ({self.num_vars},), got {x0.shape}"
                )

        # Compute prev_lat once per frame: prev_x is constant across all
        # residual evaluations within this solve_frame call, so computing
        # the previous neck lateral offset here (one extra FK per FRAME, not
        # per eval) is negligible.  Passed to _residuals for the neck lateral
        # velocity damping term, which targets the oscillation FREQUENCY
        # (complementing the sway amplitude clamp that targets the magnitude).
        prev_lat = None
        if prev_x is not None:
            pelvis_i = self.skel.name_to_idx.get("pelvis")
            neck_i = self.skel.name_to_idx.get("neck_01")
            if pelvis_i is not None and neck_i is not None:
                idx_hr = self.skel.name_to_idx["hip_r"]
                idx_hl = self.skel.name_to_idx["hip_l"]
                rest_hip = self.skel.rest_t[idx_hr] - self.skel.rest_t[idx_hl]
                lat_axis = rest_hip / (np.linalg.norm(rest_hip) + 1e-9)
                pr, pl = self._state_to_local_rotations(prev_x)
                pgp, _ = self.forward_kinematics(pr, pl)
                prev_lat = float(np.dot(pgp[neck_i] - pgp[pelvis_i], lat_axis))

        init_x = self.analytic_init(measured)

        # ── Precompute frame context (saves ~60% per-frame time) ────────
        ctx = self._precompute_frame(
            measured, prev_x, temporal_weight, init_x, prev_lat,
            frame_idx=frame_idx, prev_prev_x=prev_prev_x,
            accel_weight=accel_weight)
        if max_iter is not None:
            ctx["max_iter"] = int(max_iter)

        # Project warm starts before the first residual evaluation in the
        # opt-in feasible-set mode. The historical post mode intentionally
        # keeps its original warm start and is therefore bit-identical.
        if self.limit_projection == "projected":
            from aimocap.retarget.ik_backend import _project_x0_numpy
            x0 = _project_x0_numpy(x0, ctx)

        # ── Backend dispatch ────────────────────────────────────────────
        from aimocap.retarget.ik_backend import dispatch_solve_frame
        x_solved = dispatch_solve_frame(self, ctx, x0)
        self._record_solve_stats(x_solved, ctx)
        return x_solved
