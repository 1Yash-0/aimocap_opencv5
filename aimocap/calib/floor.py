"""Authoritative physical floor registration via surveyed rigid ChArUco board.

Phase 2.3C implementation adhering strictly to the approved Astra design.

Mathematical and coordinate invariants:
1. Physical board frame:
   - bx: Marked printed horizontal axis
   - by: Marked forward/vertical axis
   - bz: bx × by, pointing outward from the printed face
   - origin: Designated pattern point on optical plane
   - Physical floor plane in board coordinates: bz = -h

2. Floor transform derivation:
   n = R_cb @ [0, 0, 1]^T
   a = R_cb @ [1, 0, 0]^T
   o = t_cb - h * n
   y = n
   x = normalize(a - n * (n^T a))
   z = x × y
   R_cf = [x^T; y^T; z^T]
   t_cf = -R_cf @ o

3. Resulting floor_world:
   - Right-handed (det(R_cf) = +1)
   - Positive Y: Up (normal away from floor)
   - Positive Z: Backward (board marked forward is along -Z)
   - Physical floor: Y = 0
   - Board optical surface: Y = h
   - Marked board origin projected to floor: (0, 0, 0)
"""

from __future__ import annotations

import json
import math
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from aimocap.math.coords import (
    CAMERA_WORLD_FRAME,
    FLOOR_WORLD_FRAME,
    FloorTransform,
    validate_rotation,
)


class RegistrationGateError(RuntimeError):
    """Raised when an unregistered or unverified floor transform is passed to a route requiring trusted grounding."""
    pass


class FloorModelError(ValueError):
    """Raised when board specification or metrology violates physical bounds."""
    pass


class PlanarAmbiguityError(ValueError):
    """Raised when multi-view planar pose ambiguity cannot be resolved."""
    pass


@dataclass
class CharucoBoardModel:
    """Surveyed physical ChArUco board model with metrology and uncertainty metadata."""

    dictionary_name: str = "DICT_4X4_50"
    squares_x: int = 7
    squares_y: int = 5
    square_length_m: float = 0.040  # 40 mm
    marker_length_m: float = 0.030  # 30 mm
    metric_width_m: float = 0.280   # 7 * 40mm
    metric_height_m: float = 0.200  # 5 * 40mm
    designated_origin_corner_idx: int = 0
    designated_forward_axis: str = "-Z"
    optical_plane_offset_h_m: float = 0.005  # 5.0 mm measured offset from support pads to optical surface
    board_dimension_uncertainty_rel: float = 0.001  # 0.1% (threshold <= 0.2%)
    optical_offset_uncertainty_m: float = 0.0002    # 0.2 mm (threshold <= 0.5 mm)
    pattern_flatness_m: float = 0.0002              # 0.2 mm (threshold <= 0.5 mm)
    floor_levelness_deg: float = 0.10               # 0.10 deg (threshold <= 0.5 deg)
    floor_deviation_m: float = 0.001                # 1.0 mm (threshold <= 3.0 mm)
    support_pad_count: int = 3                      # Exactly three noncompressible pads
    support_pad_height_m: float = 0.005
    support_pad_locations_m: list[tuple[float, float]] = field(
        default_factory=lambda: [(0.02, 0.02), (0.26, 0.02), (0.14, 0.18)]
    )
    schema_version: str = "1.0.0"

    def __post_init__(self) -> None:
        if self.squares_x < 2 or self.squares_y < 2:
            raise FloorModelError("squares_x and squares_y must be at least 2")
        if self.square_length_m <= 0 or self.marker_length_m <= 0:
            raise FloorModelError("square_length_m and marker_length_m must be positive")
        if self.marker_length_m >= self.square_length_m:
            raise FloorModelError("marker_length_m must be strictly less than square_length_m")
        if self.optical_plane_offset_h_m <= 0:
            raise FloorModelError("optical_plane_offset_h_m must be positive")
        if self.support_pad_count != 3:
            raise FloorModelError(
                f"support_pad_count must be 3 (kinematic 3-point floor seating), got {self.support_pad_count}"
            )

    @property
    def num_inner_corners(self) -> int:
        """Total number of inner ChArUco corners on the board."""
        return (self.squares_x - 1) * (self.squares_y - 1)

    def get_object_points(self) -> np.ndarray:
        """Return 3D coordinates of all inner corners in the physical board coordinate frame.

        bx = printed horizontal axis (X)
        by = printed vertical/forward axis (Y)
        bz = outward normal (Z = 0 on optical plane, physical floor is at bz = -h)
        """
        nx = self.squares_x - 1
        ny = self.squares_y - 1
        points = np.zeros((ny * nx, 3), dtype=np.float64)
        for iy in range(ny):
            for ix in range(nx):
                idx = iy * nx + ix
                # Coordinate origin (0,0) is at corner (ix=0, iy=0)
                points[idx, 0] = float(ix + 1) * self.square_length_m
                points[idx, 1] = float(iy + 1) * self.square_length_m
                points[idx, 2] = 0.0
        # Re-center so designated origin corner is exactly at (0, 0, 0) in board frame
        origin = points[self.designated_origin_corner_idx].copy()
        points -= origin
        return points

    def to_dict(self) -> dict[str, Any]:
        """Serialize board model to JSON-compatible dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CharucoBoardModel:
        """Construct board model from dictionary."""
        # Convert support pad locations if needed
        d = dict(data)
        if "support_pad_locations_m" in d:
            d["support_pad_locations_m"] = [tuple(p) for p in d["support_pad_locations_m"]]
        return cls(**d)


@dataclass(frozen=True)
class FloorValidationProfile:
    """Frozen Initial V1 validation thresholds.

    IMPORTANT: These thresholds must not be loosened automatically.
    """

    # Board and metrology
    max_board_dimension_uncertainty_rel: float = 0.002    # <= 0.2%
    max_optical_offset_uncertainty_m: float = 0.0005       # <= 0.5 mm
    max_pattern_flatness_m: float = 0.0005                 # <= 0.5 mm
    max_floor_levelness_deg: float = 0.5                   # <= 0.5 degrees
    max_floor_deviation_m: float = 0.003                   # <= 3.0 mm over declared footprint

    # Detection and geometry
    max_reprojection_rms_px: float = 0.7                   # <= 0.7 px
    max_reprojection_p95_px: float = 1.5                   # <= 1.5 px
    max_reprojection_residual_px: float = 3.0              # <= 3.0 px
    min_geometric_inlier_ratio: float = 0.80               # >= 80% inliers
    min_corners_per_observation: int = 20                  # >= 20 corners
    min_extent_coverage_x: float = 0.60                    # >= 60% coverage along bx
    min_extent_coverage_y: float = 0.60                    # >= 60% coverage along by

    # Cross-camera consensus
    max_cross_camera_translation_disagreement_m: float = 0.005  # <= 5.0 mm
    max_cross_camera_rotation_disagreement_deg: float = 0.5    # <= 0.5 deg

    # Repeatability (lift/reseat twice)
    max_repeatability_normal_change_deg: float = 0.2       # <= 0.2 deg
    max_repeatability_height_change_m: float = 0.003       # <= 3.0 mm
    max_repeatability_horizontal_origin_m: float = 0.005   # <= 5.0 mm
    max_repeatability_heading_change_deg: float = 0.5      # <= 0.5 deg

    # Spatial validation
    max_spatial_normal_disagreement_deg: float = 0.2       # <= 0.2 deg
    max_spatial_floor_contact_disagreement_m: float = 0.005 # <= 5.0 mm

    # Uncertainty
    max_height_uncertainty_95_m: float = 0.005             # <= 5.0 mm at footprint boundary
    max_heading_uncertainty_deg: float = 0.5               # <= 0.5 deg

    # Protocol requirements
    min_primary_observation_sets: int = 10                 # >= 10 observation sets
    min_useful_camera_views: int = 2                       # >= 2 cameras per hold


@dataclass
class CameraObservation:
    """Corner detections from one camera on one frame."""

    camera_name: str
    corner_ids: np.ndarray          # Shape (M,), int32
    corner_coords_px: np.ndarray    # Shape (M, 2), float64


@dataclass
class ObservationSet:
    """Synchronized multi-camera detections for a single stationary board hold frame."""

    frame_idx: int
    observations: dict[str, CameraObservation]  # camera_name -> CameraObservation


@dataclass
class HoldCapture:
    """One stationary placement hold containing multiple synchronized observation sets."""

    hold_type: str  # "primary", "repeatability_1", "repeatability_2", "spatial_1", "spatial_2"
    hold_id: str
    observation_sets: list[ObservationSet]
    target_location: str = "primary_origin"


@dataclass
class GateOutcome:
    """Individual validation gate check outcome."""

    gate_name: str
    passed: bool
    measured_value: float | str
    threshold: float | str
    message: str
    reason_code: str


@dataclass
class BoardPoseResult:
    """Estimated rigid pose of the physical board in camera_world frame."""

    R_cb: np.ndarray  # (3, 3) rotation matrix: p_camera = R_cb @ p_board + t_cb
    t_cb: np.ndarray  # (3,) translation vector
    reprojection_residuals_px: dict[str, np.ndarray]
    per_camera_rms: dict[str, float]
    overall_rms: float
    p95_residual: float
    max_residual: float
    inlier_ratio: float
    cross_camera_translation_disagreement_m: float
    cross_camera_rotation_disagreement_deg: float
    disambiguation_status: str


@dataclass
class FloorRegistrationResult:
    """Complete authoritative floor registration result with audit provenance and gate outcomes."""

    floor_transform: FloorTransform
    grounding_status: str     # "registered" or "unregistered"
    validation_status: str    # "passed", "failed", "missing", "stale"
    failure_reasons: list[str]
    gate_outcomes: list[GateOutcome]
    primary_pose: BoardPoseResult | None
    repeatability_results: list[dict[str, Any]]
    spatial_validation_results: list[dict[str, Any]]
    uncertainty_results: dict[str, Any]
    provenance: dict[str, Any]

    @property
    def is_registered(self) -> bool:
        return self.grounding_status == "registered" and self.validation_status == "passed"


def require_grounded_floor(transform: FloorTransform) -> None:
    """Fail-closed assertion: enforce that a FloorTransform is authoritative and verified.

    Raises RegistrationGateError if transform is not registered or validation did not pass.
    Blocks downstream grounded contact / foot-lock / IK route.
    """
    if not isinstance(transform, FloorTransform):
        raise RegistrationGateError(f"Expected FloorTransform instance, got {type(transform)}")
    if not transform.is_registered:
        reasons_str = ", ".join(transform.failure_reasons) if transform.failure_reasons else "not physically measured"
        raise RegistrationGateError(
            f"FloorTransform is rejected for grounded pipeline: grounding_status='{transform.grounding_status}', "
            f"validation_status='{transform.validation_status}'. Reasons: [{reasons_str}]. "
            "Pipeline cannot proceed down the trusted grounded contact/IK route."
        )


def compute_floor_transform_from_board(
    R_cb: np.ndarray,
    t_cb: np.ndarray,
    h_m: float,
    *,
    epoch_id: str = "",
    source: str = "physical_charuco_floor_v1",
    validation_status: str = "passed",
    failure_reasons: list[str] | None = None,
) -> FloorTransform:
    """Compute rigid camera_world -> floor_world transform from board pose in camera_world.

    Follows the approved equations:
        n = R_cb @ [0, 0, 1]^T
        a = R_cb @ [1, 0, 0]^T
        o = t_cb - h * n
        y = n
        x = normalize(a - n * (n^T a))
        z = x × y
        R_cf = [x^T; y^T; z^T]
        t_cf = -R_cf @ o

    Forward:
        p_floor = R_cf @ p_camera + t_cf
    Inverse:
        p_camera = R_cf.T @ (p_floor - t_cf)
    """
    R_cb = validate_rotation(R_cb)
    t_cb = np.asarray(t_cb, dtype=np.float64).reshape(3)
    h = float(h_m)
    if h <= 0:
        raise ValueError(f"optical offset h_m must be positive, got {h}")

    # n: outward optical surface normal in camera_world
    n = R_cb @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    n = n / np.linalg.norm(n)

    # a: marked horizontal axis in camera_world
    a = R_cb @ np.array([1.0, 0.0, 0.0], dtype=np.float64)

    # o: physical floor origin in camera_world (projected downward along -n by h)
    o = t_cb - h * n

    # Floor frame basis vectors:
    # y = n (pointing away from floor)
    y = n.copy()

    # x = normalize(a - n (n^T a))
    x_proj = a - n * float(np.dot(n, a))
    norm_x = float(np.linalg.norm(x_proj))
    if norm_x < 1e-8:
        raise ValueError("Degenerate board horizontal axis: cannot project onto floor plane")
    x = x_proj / norm_x

    # z = x × y (right-handed convention: positive Z backward, marked board forward is along -Z)
    z = np.cross(x, y)
    norm_z = float(np.linalg.norm(z))
    if norm_z < 1e-8:
        raise ValueError("Degenerate cross product for floor basis")
    z = z / norm_z

    R_cf = np.vstack([x, y, z])
    R_cf = validate_rotation(R_cf)

    t_cf = -R_cf @ o

    grounding_status = "registered" if validation_status == "passed" else "unregistered"

    return FloorTransform(
        rotation=R_cf,
        translation=t_cf,
        source_frame=CAMERA_WORLD_FRAME,
        target_frame=FLOOR_WORLD_FRAME,
        units="meters",
        transform_version=1,
        grounding_status=grounding_status,
        source=source,
        validation_status=validation_status,
        failure_reasons=failure_reasons or [],
        epoch_id=epoch_id,
    )


def _solve_single_camera_planar_poses(
    object_points: np.ndarray,
    image_points: np.ndarray,
    K: np.ndarray,
    dist_coeffs: np.ndarray | None,
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Solve planar pose using cv2.solvePnPGeneric returning all candidate solutions and reprojections."""
    dist = np.zeros(5, dtype=np.float64) if dist_coeffs is None else np.asarray(dist_coeffs, dtype=np.float64).flatten()
    obj = np.ascontiguousarray(object_points, dtype=np.float32).reshape(-1, 3)
    img = np.ascontiguousarray(image_points, dtype=np.float32).reshape(-1, 2)
    K_f = np.ascontiguousarray(K, dtype=np.float64)

    # Use solvePnPGeneric with SOLVEPNP_IPPE for planar targets to return both ambiguous solutions
    try:
        ret, rvecs, tvecs, errors = cv2.solvePnPGeneric(
            obj,
            img,
            K_f,
            dist,
            flags=cv2.SOLVEPNP_IPPE,
        )
    except cv2.error:
        # Fallback to standard iterative solvePnP if IPPE fails (e.g. non-planar or degenerate)
        success, rvec, tvec = cv2.solvePnP(obj, img, K_f, dist, flags=cv2.SOLVEPNP_ITERATIVE)
        if not success:
            return []
        proj, _ = cv2.projectPoints(obj, rvec, tvec, K_f, dist)
        err = float(np.mean(np.linalg.norm(img - proj.reshape(-1, 2), axis=1)))
        R, _ = cv2.Rodrigues(rvec)
        return [(R, tvec.flatten(), err)]

    solutions: list[tuple[np.ndarray, np.ndarray, float]] = []
    for i in range(len(rvecs)):
        R, _ = cv2.Rodrigues(rvecs[i])
        t = tvecs[i].flatten().astype(np.float64)
        err = float(errors[i, 0]) if errors is not None and errors.size > i else 0.0
        solutions.append((R, t, err))
    return solutions


def solve_board_pose_multiview(
    board_model: CharucoBoardModel,
    observation_set: ObservationSet,
    K_dict: dict[str, np.ndarray],
    extrinsics_dict: dict[str, tuple[np.ndarray, np.ndarray]],
    dist_dict: dict[str, np.ndarray] | None = None,
    profile: FloorValidationProfile = FloorValidationProfile(),
) -> BoardPoseResult:
    """Solve the multi-view board pose in camera_world, resolving planar ambiguity.

    For each camera:
        X_cam = R_c @ X_cw + t_c  =>  X_cw = R_c^T @ (X_cam - t_c)
    If camera observes board as R_cam_b, t_cam_b:
        X_cam = R_cam_b @ X_board + t_cam_b
        X_cw = R_c^T @ R_cam_b @ X_board + R_c^T @ (t_cam_b - t_c)
    Hence board pose in camera_world is:
        R_cb = R_c^T @ R_cam_b
        t_cb = R_c^T @ (t_cam_b - t_c)
    """
    all_obj_pts = board_model.get_object_points()
    dist_dict = dist_dict or {}

    contributing_cameras = []
    candidate_solutions_cw: dict[str, list[tuple[np.ndarray, np.ndarray, float]]] = {}

    for cam_name, obs in observation_set.observations.items():
        if cam_name not in K_dict or cam_name not in extrinsics_dict:
            continue
        if len(obs.corner_ids) < 4:
            continue

        K = K_dict[cam_name]
        R_c, t_c = extrinsics_dict[cam_name]
        R_c = validate_rotation(R_c)
        t_c = np.asarray(t_c, dtype=np.float64).reshape(3)
        dist = dist_dict.get(cam_name, np.zeros(5))

        obj_subset = all_obj_pts[obs.corner_ids]
        img_subset = obs.corner_coords_px

        cam_solutions = _solve_single_camera_planar_poses(obj_subset, img_subset, K, dist)
        if not cam_solutions:
            continue

        cw_solutions = []
        for R_cam_b, t_cam_b, err in cam_solutions:
            R_cb = R_c.T @ R_cam_b
            t_cb = R_c.T @ (t_cam_b - t_c)
            cw_solutions.append((R_cb, t_cb, err))

        candidate_solutions_cw[cam_name] = cw_solutions
        contributing_cameras.append(cam_name)

    if len(contributing_cameras) < profile.min_useful_camera_views:
        raise PlanarAmbiguityError(
            f"Need at least {profile.min_useful_camera_views} contributing views to resolve ambiguity; "
            f"got {len(contributing_cameras)}"
        )

    # Disambiguate across multiple cameras:
    # Find the combination of solutions across cameras that has mutual consensus in camera_world
    # The physical board pose will agree across views; spurious planar ambiguity solutions diverge.
    best_combo = None
    min_disagreement = float("inf")

    # Generate candidate combinations across cameras
    import itertools
    camera_names = sorted(candidate_solutions_cw.keys())
    solution_indices_per_cam = [range(len(candidate_solutions_cw[c])) for c in camera_names]

    for combo in itertools.product(*solution_indices_per_cam):
        poses = [candidate_solutions_cw[camera_names[i]][combo[i]] for i in range(len(camera_names))]
        # Compute pairwise rotation and translation differences in camera_world
        max_trans_diff = 0.0
        max_rot_diff_deg = 0.0

        for i in range(len(poses)):
            for j in range(i + 1, len(poses)):
                R_i, t_i, _ = poses[i]
                R_j, t_j, _ = poses[j]
                t_diff = float(np.linalg.norm(t_i - t_j))
                max_trans_diff = max(max_trans_diff, t_diff)

                R_rel = R_i @ R_j.T
                rot_vec = Rotation.from_matrix(R_rel).as_rotvec()
                angle_deg = float(np.degrees(np.linalg.norm(rot_vec)))
                max_rot_diff_deg = max(max_rot_diff_deg, angle_deg)

        total_score = max_trans_diff + (max_rot_diff_deg / 100.0)
        if total_score < min_disagreement:
            min_disagreement = total_score
            best_combo = (combo, poses, max_trans_diff, max_rot_diff_deg)

    if best_combo is None:
        raise PlanarAmbiguityError("Failed to find any valid pose combination across views")

    _, selected_poses, cross_cam_trans_m, cross_cam_rot_deg = best_combo

    # Initial consensus pose: average translation, geodesic rotation mean
    init_t_cb = np.mean([p[1] for p in selected_poses], axis=0)
    quats = [Rotation.from_matrix(p[0]).as_quat() for p in selected_poses]
    init_rot = Rotation.from_quat(np.mean(quats, axis=0) / np.linalg.norm(np.mean(quats, axis=0)))
    init_rvec_cb = init_rot.as_rotvec()

    # Joint non-linear multi-view refinement of R_cb, t_cb in camera_world with Huber robust loss
    def _residual_func(params: np.ndarray) -> np.ndarray:
        r_cb = params[:3]
        t_cb_curr = params[3:6]
        R_cb_curr = Rotation.from_rotvec(r_cb).as_matrix()

        residuals = []
        for cam_name in camera_names:
            obs = observation_set.observations[cam_name]
            K = K_dict[cam_name]
            dist = dist_dict.get(cam_name, np.zeros(5))
            R_c, t_c = extrinsics_dict[cam_name]

            # In camera frame: R_cam_b = R_c @ R_cb_curr, t_cam_b = R_c @ t_cb_curr + t_c
            R_cam_b = R_c @ R_cb_curr
            t_cam_b = R_c @ t_cb_curr + t_c
            rvec_cam, _ = cv2.Rodrigues(R_cam_b)

            obj_subset = all_obj_pts[obs.corner_ids]
            img_subset = obs.corner_coords_px

            proj, _ = cv2.projectPoints(
                obj_subset.astype(np.float32),
                rvec_cam,
                t_cam_b.reshape(3, 1),
                K,
                dist,
            )
            res = (img_subset - proj.reshape(-1, 2)).flatten()
            residuals.append(res)
        return np.concatenate(residuals)

    init_params = np.concatenate([init_rvec_cb, init_t_cb])
    opt_res = least_squares(_residual_func, init_params, loss="huber", f_scale=1.5)

    refined_rvec_cb = opt_res.x[:3]
    refined_t_cb = opt_res.x[3:6]
    refined_R_cb = Rotation.from_rotvec(refined_rvec_cb).as_matrix()
    refined_R_cb = validate_rotation(refined_R_cb)

    # Compute final residuals and RMS per camera
    per_cam_residuals: dict[str, np.ndarray] = {}
    per_cam_rms: dict[str, float] = {}
    all_res_magnitudes = []

    for cam_name in camera_names:
        obs = observation_set.observations[cam_name]
        K = K_dict[cam_name]
        dist = dist_dict.get(cam_name, np.zeros(5))
        R_c, t_c = extrinsics_dict[cam_name]

        R_cam_b = R_c @ refined_R_cb
        t_cam_b = R_c @ refined_t_cb + t_c
        rvec_cam, _ = cv2.Rodrigues(R_cam_b)

        obj_subset = all_obj_pts[obs.corner_ids]
        img_subset = obs.corner_coords_px

        proj, _ = cv2.projectPoints(
            obj_subset.astype(np.float32),
            rvec_cam,
            t_cam_b.reshape(3, 1),
            K,
            dist,
        )
        diff = img_subset - proj.reshape(-1, 2)
        mags = np.linalg.norm(diff, axis=1)
        per_cam_residuals[cam_name] = mags
        per_cam_rms[cam_name] = float(np.sqrt(np.mean(mags ** 2)))
        all_res_magnitudes.extend(mags.tolist())

    res_array = np.array(all_res_magnitudes)
    overall_rms = float(np.sqrt(np.mean(res_array ** 2))) if res_array.size else 0.0
    p95_res = float(np.percentile(res_array, 95)) if res_array.size else 0.0
    max_res = float(np.max(res_array)) if res_array.size else 0.0
    inliers = np.count_nonzero(res_array <= profile.max_reprojection_residual_px)
    inlier_ratio = float(inliers / res_array.size) if res_array.size else 0.0

    return BoardPoseResult(
        R_cb=refined_R_cb,
        t_cb=refined_t_cb,
        reprojection_residuals_px=per_cam_residuals,
        per_camera_rms=per_cam_rms,
        overall_rms=overall_rms,
        p95_residual=p95_res,
        max_residual=max_res,
        inlier_ratio=inlier_ratio,
        cross_camera_translation_disagreement_m=cross_cam_trans_m,
        cross_camera_rotation_disagreement_deg=cross_cam_rot_deg,
        disambiguation_status="resolved",
    )


def evaluate_floor_registration(
    board_model: CharucoBoardModel,
    primary_hold: HoldCapture,
    repeatability_holds: list[HoldCapture],
    spatial_holds: list[HoldCapture],
    K_dict: dict[str, np.ndarray],
    extrinsics_dict: dict[str, tuple[np.ndarray, np.ndarray]],
    dist_dict: dict[str, np.ndarray] | None = None,
    profile: FloorValidationProfile = FloorValidationProfile(),
    *,
    epoch_id: str = "",
    rig_hash: str = "default_rig",
    capture_hashes: dict[str, str] | None = None,
) -> FloorRegistrationResult:
    """Execute all frozen Initial V1 registration gates over multi-hold physical capture.

    Gates checked:
    1. Board metrology (dimension uncertainty <= 0.2%, offset uncertainty <= 0.5 mm, flatness <= 0.5 mm, levelness <= 0.5 deg, deviation <= 3 mm)
    2. Primary hold protocol (>= 10 observation sets, >= 2 useful camera views per hold, every production camera contributes)
    3. Detection and geometry (per-camera RMS <= 0.7 px, P95 <= 1.5 px, max residual <= 3.0 px, inliers >= 80%, corners >= 20, extent coverage >= 60%)
    4. Cross-camera consensus (translation disagreement <= 5 mm, rotation disagreement <= 0.5 deg)
    5. Repeatability holds (normal change <= 0.2 deg, height change <= 3 mm, horizontal origin <= 5 mm, heading <= 0.5 deg)
    6. Spatial validation holds (normal disagreement <= 0.2 deg, floor contact disagreement <= 5 mm)
    7. Rigid transform contract (orthonormal, det=+1, distance preservation)
    8. Boundary height uncertainty (<= 5 mm) and heading uncertainty (<= 0.5 deg)

    Fail-closed: If ANY gate fails, grounding_status = "unregistered" and validation_status = "failed".
    """
    gate_outcomes: list[GateOutcome] = []
    failure_reasons: list[str] = []
    capture_hashes = capture_hashes or {}

    def _add_gate(
        name: str,
        passed: bool,
        measured: float | str,
        threshold: float | str,
        message: str,
        reason_code: str,
    ) -> None:
        outcome = GateOutcome(
            name,
            bool(passed),
            float(measured) if isinstance(measured, (int, float, np.number)) else str(measured),
            float(threshold) if isinstance(threshold, (int, float, np.number)) else str(threshold),
            str(message),
            str(reason_code),
        )
        gate_outcomes.append(outcome)
        if not passed:
            failure_reasons.append(reason_code)

    # -------------------------------------------------------------
    # 1. Metrology & Board Model Gates
    # -------------------------------------------------------------
    _add_gate(
        "gate_board_dimension_uncertainty",
        board_model.board_dimension_uncertainty_rel <= profile.max_board_dimension_uncertainty_rel,
        board_model.board_dimension_uncertainty_rel,
        profile.max_board_dimension_uncertainty_rel,
        f"Board dimension uncertainty {board_model.board_dimension_uncertainty_rel*100:.3f}% (max {profile.max_board_dimension_uncertainty_rel*100:.3f}%)",
        "GATE_BOARD_DIMENSION_UNCERTAINTY_EXCEEDED",
    )
    _add_gate(
        "gate_optical_offset_uncertainty",
        board_model.optical_offset_uncertainty_m <= profile.max_optical_offset_uncertainty_m,
        board_model.optical_offset_uncertainty_m,
        profile.max_optical_offset_uncertainty_m,
        f"Optical offset uncertainty {board_model.optical_offset_uncertainty_m*1000:.2f} mm (max {profile.max_optical_offset_uncertainty_m*1000:.2f} mm)",
        "GATE_OPTICAL_OFFSET_UNCERTAINTY_EXCEEDED",
    )
    _add_gate(
        "gate_pattern_flatness",
        board_model.pattern_flatness_m <= profile.max_pattern_flatness_m,
        board_model.pattern_flatness_m,
        profile.max_pattern_flatness_m,
        f"Pattern flatness {board_model.pattern_flatness_m*1000:.2f} mm (max {profile.max_pattern_flatness_m*1000:.2f} mm)",
        "GATE_PATTERN_FLATNESS_EXCEEDED",
    )
    _add_gate(
        "gate_floor_levelness",
        board_model.floor_levelness_deg <= profile.max_floor_levelness_deg,
        board_model.floor_levelness_deg,
        profile.max_floor_levelness_deg,
        f"Floor levelness {board_model.floor_levelness_deg:.2f}° (max {profile.max_floor_levelness_deg:.2f}°)",
        "GATE_FLOOR_LEVELNESS_EXCEEDED",
    )
    _add_gate(
        "gate_floor_deviation",
        board_model.floor_deviation_m <= profile.max_floor_deviation_m,
        board_model.floor_deviation_m,
        profile.max_floor_deviation_m,
        f"Floor deviation {board_model.floor_deviation_m*1000:.2f} mm (max {profile.max_floor_deviation_m*1000:.2f} mm)",
        "GATE_FLOOR_DEVIATION_EXCEEDED",
    )

    # -------------------------------------------------------------
    # 2. Primary Hold Protocol & Geometry Gates
    # -------------------------------------------------------------
    num_obs_sets = len(primary_hold.observation_sets)
    _add_gate(
        "gate_primary_observation_count",
        num_obs_sets >= profile.min_primary_observation_sets,
        num_obs_sets,
        profile.min_primary_observation_sets,
        f"Primary hold has {num_obs_sets} observation sets (min {profile.min_primary_observation_sets})",
        "GATE_PRIMARY_OBSERVATION_COUNT_DEFICIT",
    )

    # Solve primary pose across all observation sets in the primary hold
    primary_poses: list[BoardPoseResult] = []
    contributing_production_cameras: set[str] = set()

    for obs_set in primary_hold.observation_sets:
        try:
            pose_res = solve_board_pose_multiview(
                board_model, obs_set, K_dict, extrinsics_dict, dist_dict, profile
            )
            primary_poses.append(pose_res)
            for cam_name, obs in obs_set.observations.items():
                if len(obs.corner_ids) >= profile.min_corners_per_observation:
                    contributing_production_cameras.add(cam_name)
        except Exception as exc:
            failure_reasons.append(f"PRIMARY_OBS_SOLVE_FAILED: {exc}")

    # Production camera coverage gate: every production camera in extrinsics_dict must contribute
    all_production_cameras = set(extrinsics_dict.keys())
    missing_cams = all_production_cameras - contributing_production_cameras
    _add_gate(
        "gate_production_camera_coverage",
        len(missing_cams) == 0,
        len(contributing_production_cameras),
        len(all_production_cameras),
        f"Production cameras contributing: {sorted(contributing_production_cameras)} (missing {sorted(missing_cams)})",
        "GATE_PRODUCTION_CAMERA_COVERAGE_INCOMPLETE",
    )

    if not primary_poses:
        # Cannot proceed further without primary pose
        return FloorRegistrationResult(
            floor_transform=FloorTransform.identity(
                grounding_status="unregistered",
                validation_status="failed",
                failure_reasons=failure_reasons,
                epoch_id=epoch_id,
            ),
            grounding_status="unregistered",
            validation_status="failed",
            failure_reasons=failure_reasons,
            gate_outcomes=gate_outcomes,
            primary_pose=None,
            repeatability_results=[],
            spatial_validation_results=[],
            uncertainty_results={},
            provenance={"epoch_id": epoch_id, "rig_hash": rig_hash},
        )

    # Consensus primary pose: average over accepted observation frames
    mean_R_cb = Rotation.from_quat(
        np.mean([Rotation.from_matrix(p.R_cb).as_quat() for p in primary_poses], axis=0)
    ).as_matrix()
    mean_R_cb = validate_rotation(mean_R_cb)
    mean_t_cb = np.mean([p.t_cb for p in primary_poses], axis=0)

    # Overall primary pose metrics
    max_cam_rms = max(max(p.per_camera_rms.values()) for p in primary_poses)
    overall_p95 = float(np.mean([p.p95_residual for p in primary_poses]))
    overall_max_res = max(p.max_residual for p in primary_poses)
    mean_inlier_ratio = float(np.mean([p.inlier_ratio for p in primary_poses]))
    max_cross_trans = max(p.cross_camera_translation_disagreement_m for p in primary_poses)
    max_cross_rot = max(p.cross_camera_rotation_disagreement_deg for p in primary_poses)

    _add_gate(
        "gate_reprojection_rms",
        max_cam_rms <= profile.max_reprojection_rms_px,
        max_cam_rms,
        profile.max_reprojection_rms_px,
        f"Max per-camera RMS {max_cam_rms:.3f} px (threshold <= {profile.max_reprojection_rms_px} px)",
        "GATE_REPROJECTION_RMS_EXCEEDED",
    )
    _add_gate(
        "gate_reprojection_p95",
        overall_p95 <= profile.max_reprojection_p95_px,
        overall_p95,
        profile.max_reprojection_p95_px,
        f"P95 reprojection {overall_p95:.3f} px (threshold <= {profile.max_reprojection_p95_px} px)",
        "GATE_REPROJECTION_P95_EXCEEDED",
    )
    _add_gate(
        "gate_reprojection_max_residual",
        overall_max_res <= profile.max_reprojection_residual_px,
        overall_max_res,
        profile.max_reprojection_residual_px,
        f"Max residual {overall_max_res:.3f} px (threshold <= {profile.max_reprojection_residual_px} px)",
        "GATE_REPROJECTION_MAX_RESIDUAL_EXCEEDED",
    )
    _add_gate(
        "gate_geometric_inliers",
        mean_inlier_ratio >= profile.min_geometric_inlier_ratio,
        mean_inlier_ratio,
        profile.min_geometric_inlier_ratio,
        f"Geometric inlier ratio {mean_inlier_ratio*100:.1f}% (min {profile.min_geometric_inlier_ratio*100:.1f}%)",
        "GATE_GEOMETRIC_INLIERS_DEFICIT",
    )
    _add_gate(
        "gate_cross_camera_translation",
        max_cross_trans <= profile.max_cross_camera_translation_disagreement_m,
        max_cross_trans,
        profile.max_cross_camera_translation_disagreement_m,
        f"Cross-camera translation disagreement {max_cross_trans*1000:.2f} mm (max {profile.max_cross_camera_translation_disagreement_m*1000:.2f} mm)",
        "GATE_CROSS_CAMERA_TRANSLATION_DISAGREEMENT_EXCEEDED",
    )
    _add_gate(
        "gate_cross_camera_rotation",
        max_cross_rot <= profile.max_cross_camera_rotation_disagreement_deg,
        max_cross_rot,
        profile.max_cross_camera_rotation_disagreement_deg,
        f"Cross-camera rotation disagreement {max_cross_rot:.2f}° (max {profile.max_cross_camera_rotation_disagreement_deg:.2f}°)",
        "GATE_CROSS_CAMERA_ROTATION_DISAGREEMENT_EXCEEDED",
    )

    # Pattern extent coverage check across primary hold
    all_obj_pts = board_model.get_object_points()
    detected_corner_ids = set()
    for obs_set in primary_hold.observation_sets:
        for obs in obs_set.observations.values():
            detected_corner_ids.update(obs.corner_ids.tolist())

    if detected_corner_ids:
        det_pts = all_obj_pts[list(detected_corner_ids)]
        cov_x = (np.max(det_pts[:, 0]) - np.min(det_pts[:, 0])) / board_model.metric_width_m
        cov_y = (np.max(det_pts[:, 1]) - np.min(det_pts[:, 1])) / board_model.metric_height_m
    else:
        cov_x, cov_y = 0.0, 0.0

    _add_gate(
        "gate_pattern_extent_coverage_x",
        cov_x >= profile.min_extent_coverage_x,
        cov_x,
        profile.min_extent_coverage_x,
        f"Pattern extent coverage X: {cov_x*100:.1f}% (min {profile.min_extent_coverage_x*100:.1f}%)",
        "GATE_PATTERN_EXTENT_COVERAGE_X_DEFICIT",
    )
    _add_gate(
        "gate_pattern_extent_coverage_y",
        cov_y >= profile.min_extent_coverage_y,
        cov_y,
        profile.min_extent_coverage_y,
        f"Pattern extent coverage Y: {cov_y*100:.1f}% (min {profile.min_extent_coverage_y*100:.1f}%)",
        "GATE_PATTERN_EXTENT_COVERAGE_Y_DEFICIT",
    )

    # Primary consensus floor normal and origin
    n_primary = mean_R_cb @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    n_primary /= np.linalg.norm(n_primary)
    o_primary = mean_t_cb - board_model.optical_plane_offset_h_m * n_primary
    heading_primary = mean_R_cb @ np.array([0.0, 1.0, 0.0], dtype=np.float64)

    # -------------------------------------------------------------
    # 3. Repeatability Holds (lift/reseat twice at marked pose)
    # -------------------------------------------------------------
    rep_results: list[dict[str, Any]] = []
    _add_gate(
        "gate_repeatability_hold_count",
        len(repeatability_holds) >= 2,
        len(repeatability_holds),
        2,
        f"Repeatability hold count: {len(repeatability_holds)} (required 2)",
        "GATE_REPEATABILITY_HOLD_COUNT_DEFICIT",
    )

    for r_idx, r_hold in enumerate(repeatability_holds):
        try:
            r_poses = [
                solve_board_pose_multiview(board_model, os, K_dict, extrinsics_dict, dist_dict, profile)
                for os in r_hold.observation_sets
            ]
            r_R_cb = Rotation.from_quat(
                np.mean([Rotation.from_matrix(p.R_cb).as_quat() for p in r_poses], axis=0)
            ).as_matrix()
            r_t_cb = np.mean([p.t_cb for p in r_poses], axis=0)
            r_n = r_R_cb @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
            r_n /= np.linalg.norm(r_n)
            r_o = r_t_cb - board_model.optical_plane_offset_h_m * r_n
            r_heading = r_R_cb @ np.array([0.0, 1.0, 0.0], dtype=np.float64)

            # Check repeatability relative to primary hold
            normal_angle_deg = float(np.degrees(np.arccos(np.clip(np.dot(n_primary, r_n), -1.0, 1.0))))
            height_change_m = float(abs(np.dot(n_primary, r_o - o_primary)))
            # Horizontal origin reproduction (vector component perpendicular to normal)
            diff_vec = r_o - o_primary
            horiz_diff_vec = diff_vec - np.dot(diff_vec, n_primary) * n_primary
            horiz_origin_diff_m = float(np.linalg.norm(horiz_diff_vec))

            # Heading change
            h1 = heading_primary - np.dot(heading_primary, n_primary) * n_primary
            h2 = r_heading - np.dot(r_heading, r_n) * r_n
            h1 /= np.linalg.norm(h1)
            h2 /= np.linalg.norm(h2)
            heading_change_deg = float(np.degrees(np.arccos(np.clip(np.dot(h1, h2), -1.0, 1.0))))

            rep_data = {
                "hold_id": r_hold.hold_id,
                "normal_angle_deg": normal_angle_deg,
                "height_change_m": height_change_m,
                "horiz_origin_diff_m": horiz_origin_diff_m,
                "heading_change_deg": heading_change_deg,
            }
            rep_results.append(rep_data)

            _add_gate(
                f"gate_repeatability_{r_idx+1}_normal",
                normal_angle_deg <= profile.max_repeatability_normal_change_deg,
                normal_angle_deg,
                profile.max_repeatability_normal_change_deg,
                f"Repeatability {r_idx+1} normal change {normal_angle_deg:.3f}° (max {profile.max_repeatability_normal_change_deg:.3f}°)",
                "GATE_REPEATABILITY_NORMAL_EXCEEDED",
            )
            _add_gate(
                f"gate_repeatability_{r_idx+1}_height",
                height_change_m <= profile.max_repeatability_height_change_m,
                height_change_m,
                profile.max_repeatability_height_change_m,
                f"Repeatability {r_idx+1} height change {height_change_m*1000:.2f} mm (max {profile.max_repeatability_height_change_m*1000:.2f} mm)",
                "GATE_REPEATABILITY_HEIGHT_EXCEEDED",
            )
            _add_gate(
                f"gate_repeatability_{r_idx+1}_horizontal_origin",
                horiz_origin_diff_m <= profile.max_repeatability_horizontal_origin_m,
                horiz_origin_diff_m,
                profile.max_repeatability_horizontal_origin_m,
                f"Repeatability {r_idx+1} horizontal origin diff {horiz_origin_diff_m*1000:.2f} mm (max {profile.max_repeatability_horizontal_origin_m*1000:.2f} mm)",
                "GATE_REPEATABILITY_HORIZONTAL_ORIGIN_EXCEEDED",
            )
            _add_gate(
                f"gate_repeatability_{r_idx+1}_heading",
                heading_change_deg <= profile.max_repeatability_heading_change_deg,
                heading_change_deg,
                profile.max_repeatability_heading_change_deg,
                f"Repeatability {r_idx+1} heading change {heading_change_deg:.3f}° (max {profile.max_repeatability_heading_change_deg:.3f}°)",
                "GATE_REPEATABILITY_HEADING_EXCEEDED",
            )
        except Exception as exc:
            _add_gate(
                f"gate_repeatability_{r_idx+1}_failed",
                False,
                str(exc),
                "success",
                f"Repeatability hold {r_idx+1} failed to solve: {exc}",
                "GATE_REPEATABILITY_SOLVE_FAILED",
            )

    # -------------------------------------------------------------
    # 4. Spatial Validation Holds (two placements toward footprint edges)
    # -------------------------------------------------------------
    spatial_results: list[dict[str, Any]] = []
    _add_gate(
        "gate_spatial_validation_hold_count",
        len(spatial_holds) >= 2,
        len(spatial_holds),
        2,
        f"Spatial validation hold count: {len(spatial_holds)} (required 2)",
        "GATE_SPATIAL_VALIDATION_HOLD_COUNT_DEFICIT",
    )

    for s_idx, s_hold in enumerate(spatial_holds):
        try:
            s_poses = [
                solve_board_pose_multiview(board_model, os, K_dict, extrinsics_dict, dist_dict, profile)
                for os in s_hold.observation_sets
            ]
            s_R_cb = Rotation.from_quat(
                np.mean([Rotation.from_matrix(p.R_cb).as_quat() for p in s_poses], axis=0)
            ).as_matrix()
            s_t_cb = np.mean([p.t_cb for p in s_poses], axis=0)
            s_n = s_R_cb @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
            s_n /= np.linalg.norm(s_n)
            s_o = s_t_cb - board_model.optical_plane_offset_h_m * s_n

            # Spatial validation: normal disagreement with primary floor normal
            normal_disagreement_deg = float(np.degrees(np.arccos(np.clip(np.dot(n_primary, s_n), -1.0, 1.0))))

            # Floor contact location disagreement: distance of spatial placement origin from primary floor plane
            # Signed distance: dot(n_primary, s_o - o_primary)
            floor_contact_disagreement_m = float(abs(np.dot(n_primary, s_o - o_primary)))

            s_data = {
                "hold_id": s_hold.hold_id,
                "normal_disagreement_deg": normal_disagreement_deg,
                "floor_contact_disagreement_m": floor_contact_disagreement_m,
            }
            spatial_results.append(s_data)

            _add_gate(
                f"gate_spatial_{s_idx+1}_normal",
                normal_disagreement_deg <= profile.max_spatial_normal_disagreement_deg,
                normal_disagreement_deg,
                profile.max_spatial_normal_disagreement_deg,
                f"Spatial hold {s_idx+1} normal disagreement {normal_disagreement_deg:.3f}° (max {profile.max_spatial_normal_disagreement_deg:.3f}°)",
                "GATE_SPATIAL_NORMAL_DISAGREEMENT_EXCEEDED",
            )
            _add_gate(
                f"gate_spatial_{s_idx+1}_floor_contact",
                floor_contact_disagreement_m <= profile.max_spatial_floor_contact_disagreement_m,
                floor_contact_disagreement_m,
                profile.max_spatial_floor_contact_disagreement_m,
                f"Spatial hold {s_idx+1} floor contact disagreement {floor_contact_disagreement_m*1000:.2f} mm (max {profile.max_spatial_floor_contact_disagreement_m*1000:.2f} mm)",
                "GATE_SPATIAL_FLOOR_CONTACT_DISAGREEMENT_EXCEEDED",
            )
        except Exception as exc:
            _add_gate(
                f"gate_spatial_{s_idx+1}_failed",
                False,
                str(exc),
                "success",
                f"Spatial validation hold {s_idx+1} failed to solve: {exc}",
                "GATE_SPATIAL_SOLVE_FAILED",
            )

    # -------------------------------------------------------------
    # 5. Uncertainty Gate (Footprint Boundary Propagation)
    # -------------------------------------------------------------
    # Footprint boundary radius (e.g. 2.0 meters)
    footprint_radius_m = 2.0
    # Uncertainty in plane height at boundary: sigma_h = sqrt(sigma_origin^2 + (R * tan(sigma_angle))^2)
    normal_sigma_rad = math.radians(max([p.cross_camera_rotation_disagreement_deg for p in primary_poses] + [0.05]))
    origin_sigma_m = max([p.cross_camera_translation_disagreement_m for p in primary_poses] + [0.001])
    # 95% confidence interval (~ 2.0 * sigma)
    predicted_height_uncertainty_95_m = 2.0 * math.sqrt(origin_sigma_m ** 2 + (footprint_radius_m * math.tan(normal_sigma_rad)) ** 2)
    heading_uncertainty_deg = math.degrees(normal_sigma_rad)

    _add_gate(
        "gate_boundary_height_uncertainty",
        predicted_height_uncertainty_95_m <= profile.max_height_uncertainty_95_m,
        predicted_height_uncertainty_95_m,
        profile.max_height_uncertainty_95_m,
        f"Predicted 95% height uncertainty at footprint boundary: {predicted_height_uncertainty_95_m*1000:.2f} mm (max {profile.max_height_uncertainty_95_m*1000:.2f} mm)",
        "GATE_UNCERTAINTY_HEIGHT_EXCEEDED",
    )
    _add_gate(
        "gate_heading_uncertainty",
        heading_uncertainty_deg <= profile.max_heading_uncertainty_deg,
        heading_uncertainty_deg,
        profile.max_heading_uncertainty_deg,
        f"Predicted heading uncertainty: {heading_uncertainty_deg:.3f}° (max {profile.max_heading_uncertainty_deg:.3f}°)",
        "GATE_UNCERTAINTY_HEADING_EXCEEDED",
    )

    uncertainty_results = {
        "footprint_radius_m": footprint_radius_m,
        "predicted_height_uncertainty_95_m": predicted_height_uncertainty_95_m,
        "heading_uncertainty_deg": heading_uncertainty_deg,
    }

    # -------------------------------------------------------------
    # 6. Compute Floor Transform and Check Rigid Properties
    # -------------------------------------------------------------
    all_passed = len(failure_reasons) == 0
    validation_status = "passed" if all_passed else "failed"

    floor_transform = compute_floor_transform_from_board(
        mean_R_cb,
        mean_t_cb,
        board_model.optical_plane_offset_h_m,
        epoch_id=epoch_id,
        source="physical_charuco_floor_v1",
        validation_status=validation_status,
        failure_reasons=failure_reasons,
    )

    # Rigid transform contract checks
    R = floor_transform.rotation
    is_orthonormal = bool(np.allclose(R.T @ R, np.eye(3), atol=1e-6))
    det_val = float(np.linalg.det(R))
    is_proper = bool(np.isclose(det_val, 1.0, atol=1e-6))
    is_finite = bool(np.all(np.isfinite(R)) and np.all(np.isfinite(floor_transform.translation)))

    _add_gate(
        "gate_rigid_rotation_orthonormal",
        is_orthonormal,
        "orthonormal" if is_orthonormal else "non_orthonormal",
        "orthonormal",
        "Floor transform rotation matrix is orthonormal (R^T R = I)",
        "GATE_RIGID_ROTATION_NOT_ORTHONORMAL",
    )
    _add_gate(
        "gate_rigid_rotation_determinant",
        is_proper,
        det_val,
        1.0,
        f"Floor transform rotation matrix determinant is +1 (det={det_val:.6f})",
        "GATE_RIGID_ROTATION_DET_NOT_ONE",
    )
    _add_gate(
        "gate_rigid_transform_finite",
        is_finite,
        "finite" if is_finite else "non_finite",
        "finite",
        "Floor transform values are finite",
        "GATE_RIGID_TRANSFORM_NON_FINITE",
    )

    # Re-evaluate all_passed after rigid transform checks
    final_failure_reasons = [g.reason_code for g in gate_outcomes if not g.passed]
    final_passed = len(final_failure_reasons) == 0
    final_validation_status = "passed" if final_passed else "failed"
    final_grounding_status = "registered" if final_passed else "unregistered"

    floor_transform.grounding_status = final_grounding_status
    floor_transform.validation_status = final_validation_status
    floor_transform.failure_reasons = final_failure_reasons

    # -------------------------------------------------------------
    # 7. Complete Audit Provenance Metadata
    # -------------------------------------------------------------
    provenance = {
        "protocol_version": "2.3C_v1",
        "schema_version": board_model.schema_version,
        "solver_version": "1.0.0",
        "validation_version": "1.0.0",
        "registration_id": str(uuid.uuid4()),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "epoch_id": epoch_id or str(uuid.uuid4()),
        "rig_hash": rig_hash,
        "capture_hashes": capture_hashes,
        "board_model": board_model.to_dict(),
        "primary_hold": {
            "num_observation_sets": num_obs_sets,
            "contributing_cameras": sorted(contributing_production_cameras),
            "reprojection_rms_px": max_cam_rms,
            "reprojection_p95_px": overall_p95,
            "max_residual_px": overall_max_res,
            "inlier_ratio": mean_inlier_ratio,
        },
        "repeatability": rep_results,
        "spatial_validation": spatial_results,
        "uncertainty": uncertainty_results,
        "gate_outcomes": [asdict(g) for g in gate_outcomes],
        "final_outcome": {
            "grounding_status": final_grounding_status,
            "validation_status": final_validation_status,
            "failure_reasons": final_failure_reasons,
        },
    }

    primary_summary_pose = primary_poses[0] if primary_poses else None

    return FloorRegistrationResult(
        floor_transform=floor_transform,
        grounding_status=final_grounding_status,
        validation_status=final_validation_status,
        failure_reasons=final_failure_reasons,
        gate_outcomes=gate_outcomes,
        primary_pose=primary_summary_pose,
        repeatability_results=rep_results,
        spatial_validation_results=spatial_results,
        uncertainty_results=uncertainty_results,
        provenance=provenance,
    )


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def save_floor_registration(path: str | Path, result: FloorRegistrationResult) -> None:
    """Save full floor registration record and provenance to JSON."""
    dst = Path(path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "transform": result.floor_transform.to_dict(),
        "grounding_status": result.grounding_status,
        "validation_status": result.validation_status,
        "failure_reasons": result.failure_reasons,
        "gate_outcomes": [asdict(g) for g in result.gate_outcomes],
        "repeatability_results": result.repeatability_results,
        "spatial_validation_results": result.spatial_validation_results,
        "uncertainty_results": result.uncertainty_results,
        "provenance": result.provenance,
    }
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=_json_default)


def load_floor_registration(path: str | Path) -> dict[str, Any]:
    """Load floor registration record from JSON."""
    src = Path(path)
    with open(src, "r", encoding="utf-8") as f:
        return json.load(f)
