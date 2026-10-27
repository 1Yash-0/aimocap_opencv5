"""Shared geometry thresholds for calibrated reconstruction."""

from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class PartConfidence:
    """Per-body-part gating and weighting policy for calibrated 2D scores.

    ``min_conf`` excludes observations below the cutoff (exclusive: a score is
    kept when ``score >= min_conf``).  ``gamma`` reshapes the score into the
    weighted-DLT weight ``w = p ** gamma``: 1.0 keeps the calibrated
    probability, <1 flattens the differences, >1 trusts confident points more.
    """

    min_conf: float = 0.5
    gamma: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_conf <= 1.0:
            raise ValueError("part min_conf must be in [0, 1]")
        if not self.gamma > 0.0:
            raise ValueError("part gamma must be positive")

    def weight(self, score: float) -> float:
        """Weighted-DLT weight for one calibrated score."""
        return float(score) ** self.gamma


@dataclass(frozen=True)
class GeometryConfig:
    """One explicit geometry policy used by triangulation and diagnostics."""

    min_detection_conf: float = 0.5
    max_reprojection_error_px: float = 5.0
    hard_reprojection_error_px: float = 12.0
    min_ray_angle_deg: float = 10.0
    min_depth_m: float = 0.01
    robust_loss: str = "huber"
    f_scale: float = 10.0
    min_inliers: int = 2
    hysteresis_margin_px: float = 5.0
    w_temporal: float = 0.35
    w_geom: float = 2.0
    w_anat: float = 2.0
    # Reproduce the historical (metre/centimetre mixed) scoring numbers exactly.
    # The corrected default converts temporal displacement to centimetres and
    # the anatomical table from centimetres to metres.
    legacy_unit_behaviour: bool = False
    # Per-part score policy (COCO-WholeBody groups).  ``None`` means "use
    # min_detection_conf with gamma 1", which is the pre-P3.2 behaviour and
    # keeps every caller that only sets min_detection_conf unchanged.
    body_confidence: PartConfidence | None = None
    feet_confidence: PartConfidence | None = None
    face_confidence: PartConfidence | None = None
    hands_confidence: PartConfidence | None = None
    # Camera-trust policy.  ``"legacy"`` keeps the historical hard-consensus
    # subset search (12 px hard inlier test, hard camera-set switching) and is
    # bit-identical to pre-P3.3 output.  ``"soft"`` runs an IRLS robust solve
    # with continuous per-camera trust (leave-one-out down-weighting plus a
    # temporal EMA); the P3.1 temporal/bone scoring terms are not used because
    # they only ranked hard camera subsets.  The CLI production preset selects
    # ``"soft"``; ``--triangulation-preset legacy`` selects the old behaviour.
    camera_policy: str = "legacy"
    # Robust kernel scale ``c`` in pixels for the soft IRLS solve.  With a
    # Huber kernel a camera's trust reaches 0.5 at 2c px, so c=6 keeps the
    # historical 12 px boundary as the mask threshold.
    soft_scale_px: float = 6.0
    # Leave-one-out test: down-weight a camera when dropping it shrinks the
    # remaining cameras' aggregate residual by this factor *and* the dropped
    # camera itself is at least this factor worse than the remaining ones.
    soft_loo_ratio: float = 2.0
    # Floor of the continuous LOO down-weight (trust never becomes hard zero).
    soft_loo_floor: float = 0.05
    # Temporal EMA of per-camera trust: 0 = no memory, 1 = fully sticky.
    soft_ema_alpha: float = 0.5
    # Residual (in units of soft_scale_px) above which the EMA is bypassed so a
    # genuine occlusion can still drop the camera immediately.
    soft_fast_change_scale: float = 4.0
    # Number of IRLS refinement iterations for the soft solve (3-5 is useful).
    soft_iterations: int = 3
    # Measurement definition (not a solver threshold): a soft-trust camera
    # counts as *supporting* a reconstruction when its continuous trust is at
    # least this value.  With the Huber kernel the trust at the historical
    # 12 px hard boundary is soft_scale_px / hard_reprojection_error_px = 0.25,
    # so 0.25 makes the diagnostic support set equivalent to the legacy 12 px
    # inlier test.  Used by the raw-geometry gate and the ray-angle diagnostic;
    # it never changes the solve or the emitted points.
    soft_support_min_trust: float = 0.25
    # --- P3.8 fixes (soft camera policy only) -----------------------------
    # F1 support gate.  When enabled, a joint-frame is emitted only when at
    # least two observed cameras have ``trust >= soft_support_min_trust`` and
    # the final weighted solve is actually supported by at least two of those
    # cameras (finite residual <= 2 * soft_scale_px).  Otherwise the point is
    # NaN and the diagnostics record ``insufficient_support``.  This removes
    # depth-degenerate points whose depth was set by low-trust cameras.  Off
    # keeps the P3.7 soft output unchanged.
    soft_support_gate: bool = False
    # F2 temporal pair prior.  With 3+ valid cameras, candidate camera sets
    # (all pairs plus the full set) are scored by robust reprojection cost plus
    # ``soft_prior_w_temporal`` * distance to the previous frame's 3D point in
    # millimetres plus a small anatomical tie-break; the best set initialises
    # the IRLS weights (cameras outside the set start at
    # ``soft_prior_downweight``).  Missing previous points fall back to the
    # plain soft solve.
    soft_temporal_prior: bool = False
    # Prior weight per millimetre of distance to the previous 3D point.
    soft_prior_w_temporal: float = 0.05
    # Prior weight per unit relative bone-length deviation (metres, P3.1
    # unit contract: no centimetre mixing).
    soft_prior_w_anat: float = 1.0
    # Use a constant-velocity prediction (last two frames) instead of the
    # previous frame position when both are finite.
    soft_prior_velocity: bool = True
    # Initial IRLS weight multiplier for cameras outside the chosen set.
    soft_prior_downweight: float = 0.05
    # Lock-in escape: when the prior-chosen minimal set's robust residual is
    # above ``soft_prior_lockin_scale`` * soft_scale_px for
    # ``soft_prior_lockin_frames`` consecutive frames, the next frame re-selects
    # without the temporal prior.
    soft_prior_lockin_scale: float = 4.0
    soft_prior_lockin_frames: int = 5
    # F3 RTS spike guard.  When enabled, a measurement whose innovation exceeds
    # a MAD-based robust threshold over a short causal window (plus the
    # measurement-noise floor) is rejected; an innovation that was already
    # large on the previous frame is accepted, so a real step (or fast
    # oscillation) is followed on its second frame while an isolated one-frame
    # spike is rejected.  ``temporal_maneuver_cap`` can additionally cap the
    # process-noise inflation that a large innovation would otherwise trigger.
    # Off keeps the P3.7 smoother output unchanged.
    temporal_spike_gate: bool = False
    temporal_mad_window: int = 15
    temporal_mad_scale: float = 4.0
    temporal_mad_min_samples: int = 8
    # Fraction of the maneuver excess that is allowed to open the process
    # noise (1.0 reproduces the P3.5 behaviour).
    temporal_maneuver_cap: float = 1.0
    # Left/right swap repair (P3.4).  Opt-in per frame and camera: the arm,
    # leg and whole-hand label groups are tested for a swapped labelling and
    # the swap is accepted when the group's robust reprojection cost drops by
    # ``lr_swap_ratio``.  Requires the decision to persist for two consecutive
    # frames (or to continue the previous frame's accepted decision).
    lr_swap_fix: bool = False
    # Minimum robust-cost reduction factor for accepting a swap.
    lr_swap_ratio: float = 2.0
    # Offline 3D temporal smoother (P3.5).  ``"none"`` keeps the old output
    # bit-identical.  ``"robust_rts"`` runs a bidirectional constant-acceleration
    # Kalman/RTS smoother on the triangulated points (camera-world, metres)
    # using the P3.3 trust/residual diagnostics as measurement noise.  It is
    # offline and zero-phase, so it is not suitable for live streaming.
    temporal_filter: str = "none"
    temporal_filter_sigma_m: float = 0.012
    # Continuous white-jerk spectral density (m^2/s^5). 1000 trades jitter against
    # fast-motion speed fidelity on the Panoptic, AIST++ and OpenCap evaluations.
    temporal_filter_q: float = 1000.0
    temporal_filter_maneuver_boost: float = 100.0
    temporal_filter_fast_boost: float = 50.0
    temporal_filter_max_gap_s: float = 0.2
    # Per-point 3D confidence (P3.6).  Opt-in and only defined for the soft
    # camera policy because it consumes the continuous ``camera_trust``.  The
    # extra npz key is ``confidence3d``; with the flag off the output is
    # bit-identical to P3.5.
    confidence3d: bool = False

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_detection_conf <= 1.0:
            raise ValueError("min_detection_conf must be in [0, 1]")
        if self.max_reprojection_error_px <= 0 or self.hard_reprojection_error_px <= 0:
            raise ValueError("reprojection thresholds must be positive")
        if self.hard_reprojection_error_px < self.max_reprojection_error_px:
            raise ValueError("hard_reprojection_error_px must be >= max_reprojection_error_px")
        if self.min_ray_angle_deg < 0 or self.min_ray_angle_deg >= 180:
            raise ValueError("min_ray_angle_deg must be in [0, 180)")
        if self.min_depth_m < 0:
            raise ValueError("min_depth_m must be non-negative")
        if self.robust_loss not in {"huber", "cauchy"}:
            raise ValueError("robust_loss must be 'huber' or 'cauchy'")
        if self.min_inliers < 2:
            raise ValueError("min_inliers must be >= 2")
        if self.hysteresis_margin_px < 0.0:
            raise ValueError("hysteresis_margin_px must be non-negative")
        if self.w_temporal < 0.0:
            raise ValueError("w_temporal must be non-negative")
        if self.w_geom < 0.0:
            raise ValueError("w_geom must be non-negative")
        if self.w_anat < 0.0:
            raise ValueError("w_anat must be non-negative")
        if self.camera_policy not in {"legacy", "soft"}:
            raise ValueError("camera_policy must be 'legacy' or 'soft'")
        if self.soft_scale_px <= 0.0:
            raise ValueError("soft_scale_px must be positive")
        if self.soft_loo_ratio <= 1.0:
            raise ValueError("soft_loo_ratio must be > 1")
        if not 0.0 <= self.soft_loo_floor <= 1.0:
            raise ValueError("soft_loo_floor must be in [0, 1]")
        if not 0.0 <= self.soft_ema_alpha <= 1.0:
            raise ValueError("soft_ema_alpha must be in [0, 1]")
        if self.soft_fast_change_scale < 0.0:
            raise ValueError("soft_fast_change_scale must be non-negative")
        if self.soft_iterations < 1:
            raise ValueError("soft_iterations must be >= 1")
        if not 0.0 < self.soft_support_min_trust < 1.0:
            raise ValueError("soft_support_min_trust must be in (0, 1)")
        if self.soft_prior_w_temporal < 0.0:
            raise ValueError("soft_prior_w_temporal must be non-negative")
        if self.soft_prior_w_anat < 0.0:
            raise ValueError("soft_prior_w_anat must be non-negative")
        if not 0.0 < self.soft_prior_downweight <= 1.0:
            raise ValueError("soft_prior_downweight must be in (0, 1]")
        if self.soft_prior_lockin_scale <= 0.0:
            raise ValueError("soft_prior_lockin_scale must be positive")
        if self.soft_prior_lockin_frames < 1:
            raise ValueError("soft_prior_lockin_frames must be >= 1")
        if self.temporal_mad_window < 3:
            raise ValueError("temporal_mad_window must be >= 3")
        if self.temporal_mad_scale <= 0.0:
            raise ValueError("temporal_mad_scale must be positive")
        if self.temporal_mad_min_samples < 1:
            raise ValueError("temporal_mad_min_samples must be >= 1")
        if not 0.0 <= self.temporal_maneuver_cap <= 1.0:
            raise ValueError("temporal_maneuver_cap must be in [0, 1]")
        if (
            self.soft_support_gate
            or self.soft_temporal_prior
            or self.temporal_spike_gate
        ) and self.camera_policy != "soft":
            raise ValueError("P3.8 soft fixes require camera_policy='soft'")
        if self.lr_swap_ratio <= 1.0:
            raise ValueError("lr_swap_ratio must be > 1")
        if self.temporal_filter not in {"none", "robust_rts"}:
            raise ValueError("temporal_filter must be 'none' or 'robust_rts'")
        if self.temporal_filter_sigma_m <= 0.0:
            raise ValueError("temporal_filter_sigma_m must be positive")
        if self.temporal_filter_q <= 0.0:
            raise ValueError("temporal_filter_q must be positive")
        if self.temporal_filter_maneuver_boost < 0.0:
            raise ValueError("temporal_filter_maneuver_boost must be non-negative")
        if self.temporal_filter_fast_boost < 1.0:
            raise ValueError("temporal_filter_fast_boost must be at least 1")
        if self.temporal_filter_max_gap_s < 0.0:
            raise ValueError("temporal_filter_max_gap_s must be non-negative")
        if self.confidence3d and self.camera_policy != "soft":
            raise ValueError("confidence3d requires camera_policy='soft'")

    def confidence_for_joint(self, joint: int) -> PartConfidence:
        """Return the score policy for a COCO-WholeBody joint index."""
        if joint < 17:
            setting = self.body_confidence
        elif joint < 23:
            setting = self.feet_confidence
        elif joint < 91:
            setting = self.face_confidence
        else:
            setting = self.hands_confidence
        if setting is None:
            setting = PartConfidence(self.min_detection_conf, 1.0)
        return setting

    @classmethod
    def legacy_confidence(cls, **overrides) -> "GeometryConfig":
        """Preset reproducing P3.1: every group at 0.5 with identity weights."""
        settings: dict = {
            "body_confidence": PartConfidence(0.5, 1.0),
            "feet_confidence": PartConfidence(0.5, 1.0),
            "face_confidence": PartConfidence(0.5, 1.0),
            "hands_confidence": PartConfidence(0.5, 1.0),
        }
        settings.update(overrides)
        return cls(**settings)

    @classmethod
    def triangulation_preset(cls, name: str, **overrides) -> "GeometryConfig":
        """Return one of the two shipped geometry policy bundles.

        ``"production"`` is the default policy of the CLI ``triangulate``
        stage: soft per-camera trust with the P3.3 tuned parameters
        (scale 3 px, LOO ratio 4, EMA 0.5), the P3.5 robust RTS smoother and
        the P3.6 per-point 3D confidence.  Per-part score cutoffs are left at
        their P3.1 defaults (0.5 / gamma 1); P3.2 remains opt-in.

        ``"legacy"`` reproduces the P3.1 output exactly: hard camera subsets,
        no temporal filter, no ``confidence3d`` key.  The P3.1 metre/centimetre
        unit fix stays enabled because ``legacy_unit_behaviour`` is a separate
        compatibility flag.
        """
        if name == "production":
            settings: dict = {
                "camera_policy": "soft",
                "soft_scale_px": 3.0,
                "soft_loo_ratio": 4.0,
                "soft_ema_alpha": 0.5,
                "temporal_filter": "robust_rts",
                "confidence3d": True,
            }
        elif name == "legacy":
            settings = {
                "camera_policy": "legacy",
                "temporal_filter": "none",
                "confidence3d": False,
            }
        else:
            raise ValueError(
                f"unknown triangulation preset {name!r}; expected 'production' or 'legacy'"
            )
        settings.update(overrides)
        return cls(**settings)

    def to_dict(self) -> dict:
        return asdict(self)


__all__ = ["GeometryConfig", "PartConfidence"]
