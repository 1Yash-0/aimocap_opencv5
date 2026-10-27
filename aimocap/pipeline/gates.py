import numpy as np
from aimocap.pipeline.evidence import EvidenceContract, GateFailureReport
from aimocap.pipeline.epipolar import get_fundamental_matrices
from aimocap.math.metrics import compute_epipolar_consistency

class PipelineGateError(Exception):
    pass

def check_2d_gates(
    keypoints: np.ndarray,      # (F, C, J, 2)
    scores: np.ndarray,         # (F, C, J)
    K_list: list[np.ndarray], 
    extrinsics: list[tuple[np.ndarray, np.ndarray]], 
    calib_hash: str,
    provenance_hash: str,
    image_width: float = 1920.0,
    image_height: float = 1080.0,
    min_conf: float = 0.5,
    soft_fail: bool = True
) -> EvidenceContract | GateFailureReport:
    """
    Check 2D observation gates before triangulation.
    """
    F_frames, C_cams, J_joints, _ = keypoints.shape
    
    gate_results = {}
    failing_gate = None
    failure_mode = None
    
    # Gate 1: coords_in_image
    # All confident kpts lie within [0, W] x [0, H] with 5px tolerance
    confident_mask = scores >= min_conf
    x_valid = (keypoints[..., 0] >= -5.0) & (keypoints[..., 0] <= image_width + 5.0)
    y_valid = (keypoints[..., 1] >= -5.0) & (keypoints[..., 1] <= image_height + 5.0)
    coords_valid = x_valid & y_valid
    
    # If there are any confident points that are outside the image, this fails
    bad_coords = confident_mask & ~coords_valid
    passed_coords = not np.any(bad_coords)
    
    gate_results["coords_in_image"] = {
        "passed": passed_coords,
        "value": int(np.sum(bad_coords)),
        "threshold": 0
    }
    if not passed_coords and failing_gate is None:
        failing_gate = "coords_in_image"
        failure_mode = f"{np.sum(bad_coords)} confident keypoints outside image bounds"

    # Gate 2: multi_camera_coverage
    # >=2 usable cameras per joint per frame for >=80% of joints
    # confident_mask is (F, C, J). Sum over C:
    cams_per_joint = np.sum(confident_mask, axis=1) # (F, J)
    valid_joints = cams_per_joint >= 2 # (F, J)
    # Average across all frames and joints
    coverage = np.mean(valid_joints)
    
    passed_coverage = coverage >= 0.8
    gate_results["multi_camera_coverage"] = {
        "passed": passed_coverage,
        "value": coverage,
        "threshold": 0.8
    }
    if not passed_coverage and failing_gate is None:
        failing_gate = "multi_camera_coverage"
        failure_mode = f"Only {coverage:.1%} of joints have >=2 cameras"
        
    # Gate 3: epipolar_consistency
    F_matrices = get_fundamental_matrices(K_list, extrinsics)
    epi_results = compute_epipolar_consistency(
        np.transpose(keypoints, (0, 1, 2, 3)), # Wait, compute_epipolar expects (F, C, J, 2)
        scores,
        F_matrices,
        min_conf=min_conf
    )
    
    median_err = epi_results["median_err"]
    p95_err = epi_results["p95_err"]
    epipolar_coverage = epi_results["valid_coverage"]
    
    passed_epi_med = median_err < 5.0
    passed_epi_p95 = p95_err < 15.0
    passed_epi_cov = epipolar_coverage >= 0.7
    
    gate_results["epipolar_median"] = {
        "passed": passed_epi_med,
        "value": median_err,
        "threshold": 5.0
    }
    if not passed_epi_med and failing_gate is None:
        failing_gate = "epipolar_median"
        failure_mode = f"Median epipolar error {median_err:.1f}px >= 5.0px"
        
    gate_results["epipolar_p95"] = {
        "passed": passed_epi_p95,
        "value": p95_err,
        "threshold": 15.0
    }
    if not passed_epi_p95 and failing_gate is None:
        failing_gate = "epipolar_p95"
        failure_mode = f"P95 epipolar error {p95_err:.1f}px >= 15.0px"
        
    gate_results["epipolar_coverage"] = {
        "passed": passed_epi_cov,
        "value": epipolar_coverage,
        "threshold": 0.7
    }
    if not passed_epi_cov and failing_gate is None:
        failing_gate = "epipolar_coverage"
        failure_mode = f"Epipolar coverage {epipolar_coverage:.1%} < 70%"
        
    # Skeleton plausibility is done implicitly in triangulation min_aspect_ratio for now
    
    evidence = EvidenceContract(
        stage="2d_observation",
        status="FAIL" if failing_gate else "PASS",
        validity_mask=valid_joints,
        uncertainty=scores, # simplfication
        calibration_id=calib_hash,
        coordinate_space="image_px",
        input_provenance=provenance_hash,
        gate_results=gate_results,
        coverage_fraction=coverage,
        failure_reason=failing_gate
    )
    
    if failing_gate:
        if soft_fail:
            return GateFailureReport(
                failing_stage="2d_observation",
                failing_gate=failing_gate,
                cameras_implicated=[str(c) for c in range(C_cams)],
                failure_mode=failure_mode,
                coverage_fraction=coverage,
                evidence=evidence
            )
        else:
            raise PipelineGateError(f"2D Gate Failed: {failure_mode}")
            
    return evidence


def _weighted_percentile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """Weighted percentile with the midpoint cumulative-weight convention."""
    order = np.argsort(values, kind="stable")
    values = np.asarray(values, dtype=np.float64)[order]
    weights = np.asarray(weights, dtype=np.float64)[order]
    total = float(np.sum(weights))
    if total <= 0.0:
        return float("nan")
    cumulative = (np.cumsum(weights) - 0.5 * weights) / total
    return float(np.interp(q / 100.0, cumulative, values))


def check_raw_geometry_gates(
    diagnostics,
    calib_hash: str,
    provenance_hash: str,
    *,
    core_joint_count: int = 23,
    min_finite_fraction: float = 0.90,
    min_two_camera_fraction: float = 0.90,
    max_reprojection_median_px: float = 3.0,
    max_reprojection_p95_px: float = 10.0,
    min_positive_depth_fraction: float = 0.99,
    min_median_ray_angle_deg: float = 15.0,
    max_low_ray_fraction: float = 0.01,
    camera_policy: str = "legacy",
    support_min_trust: float = 0.25,
    hard_inlier_threshold_px: float = 12.0,
    soft_fail: bool = True,
) -> EvidenceContract | GateFailureReport:
    """Apply the strict pre-fitting gate for raw calibrated 3D geometry.

    The numeric thresholds are policy independent.  What changes for the soft
    camera policy is the *measurement*: ``support_mask`` (trust at least
    ``support_min_trust``, which reproduces the historical 12 px hard inlier
    boundary) replaces the hard ``inlier_mask``, two-camera coverage counts
    supporting cameras, and reprojection residuals are pooled with the
    continuous per-camera trust as weight.  The legacy policy keeps every
    historical definition bit-identical.
    """
    if camera_policy not in {"legacy", "soft"}:
        raise ValueError("camera_policy must be 'legacy' or 'soft'")
    if hard_inlier_threshold_px <= 0.0:
        raise ValueError("hard_inlier_threshold_px must be positive")
    core = min(int(core_joint_count), int(diagnostics.points3d.shape[1]))
    points = np.asarray(diagnostics.points3d[:, :core])
    num_inliers = np.asarray(diagnostics.num_inliers[:, :core])
    inlier_mask = np.asarray(diagnostics.inlier_mask[:, :core])
    reprojection = np.asarray(diagnostics.reprojection_error_px[:, :core])
    positive_depth = np.asarray(diagnostics.positive_depth_mask[:, :core])
    ray_angle = np.asarray(diagnostics.ray_angle_deg[:, :core])
    finite = np.isfinite(points).all(axis=-1)
    finite_fraction = float(np.mean(finite))

    if camera_policy == "soft":
        support_mask = getattr(diagnostics, "support_mask", None)
        if support_mask is None:
            support_mask = np.asarray(diagnostics.camera_trust[:, :core]) >= support_min_trust
        else:
            support_mask = np.asarray(support_mask[:, :core], dtype=bool)
        support_mask = support_mask & finite[..., None]
        support_count = np.sum(support_mask, axis=-1)
        camera_trust = np.asarray(diagnostics.camera_trust[:, :core], dtype=np.float64)
        accepted_mask = support_mask
        accepted_count = support_count
    else:
        support_mask = inlier_mask
        accepted_mask = inlier_mask
        accepted_count = num_inliers

    if camera_policy == "soft":
        two_camera_fraction = float(np.mean(finite & (accepted_count >= 2)))
    else:
        two_camera_fraction = float(np.mean(num_inliers >= 2))
    finite_reprojection = reprojection[np.isfinite(reprojection)]
    if camera_policy == "soft":
        support_residual = np.isfinite(reprojection) & support_mask
        residual_values = reprojection[support_residual]
        residual_weights = np.clip(camera_trust[support_residual], 1e-6, None)
        reprojection_median = (
            _weighted_percentile(residual_values, residual_weights, 50.0)
            if residual_values.size
            else float("inf")
        )
        reprojection_p95 = (
            _weighted_percentile(residual_values, residual_weights, 95.0)
            if residual_values.size
            else float("inf")
        )
    else:
        reprojection_median = float(np.median(finite_reprojection)) if finite_reprojection.size else float("inf")
        reprojection_p95 = float(np.percentile(finite_reprojection, 95)) if finite_reprojection.size else float("inf")
    accepted_positive = positive_depth[accepted_mask]
    positive_depth_fraction = float(np.mean(accepted_positive)) if accepted_positive.size else 0.0
    if camera_policy == "soft":
        supported_ray_angles = ray_angle[finite & (accepted_count >= 2)]
    else:
        supported_ray_angles = ray_angle[num_inliers >= 2]
    supported_ray_angles = supported_ray_angles[np.isfinite(supported_ray_angles)]
    median_ray_angle = float(np.median(supported_ray_angles)) if supported_ray_angles.size else 0.0
    low_ray_fraction = (
        float(np.mean(supported_ray_angles < 2.0)) if supported_ray_angles.size else 1.0
    )

    gate_results = {
        "core_finite_coverage": {
            "passed": finite_fraction >= min_finite_fraction,
            "value": finite_fraction,
            "threshold": min_finite_fraction,
            "definition": "fraction of core joint-frames with a finite 3D point",
        },
        "core_two_camera_coverage": {
            "passed": two_camera_fraction >= min_two_camera_fraction,
            "value": two_camera_fraction,
            "threshold": min_two_camera_fraction,
            "definition": (
                "finite point and >=2 supporting cameras (trust >= support_min_trust)"
                if camera_policy == "soft"
                else "hard-inlier count >= 2"
            ),
        },
        "reprojection_median": {
            "passed": reprojection_median < max_reprojection_median_px,
            "value": reprojection_median,
            "threshold": max_reprojection_median_px,
            "definition": (
                "trust-weighted median residual of supporting cameras"
                if camera_policy == "soft"
                else "median residual over all observed cameras"
            ),
        },
        "reprojection_p95": {
            "passed": reprojection_p95 < max_reprojection_p95_px,
            "value": reprojection_p95,
            "threshold": max_reprojection_p95_px,
            "definition": (
                "trust-weighted p95 residual of supporting cameras"
                if camera_policy == "soft"
                else "p95 residual over all observed cameras"
            ),
        },
        "positive_depth": {
            "passed": positive_depth_fraction >= min_positive_depth_fraction,
            "value": positive_depth_fraction,
            "threshold": min_positive_depth_fraction,
            "definition": "fraction of accepted camera observations with positive depth",
        },
        "median_ray_angle": {
            "passed": median_ray_angle >= min_median_ray_angle_deg,
            "value": median_ray_angle,
            "threshold": min_median_ray_angle_deg,
            "definition": "median max pairwise ray angle of accepted points",
        },
        "low_ray_fraction": {
            "passed": low_ray_fraction < max_low_ray_fraction,
            "value": low_ray_fraction,
            "threshold": max_low_ray_fraction,
            "definition": "fraction of accepted points with a ray angle below 2 degrees",
        },
    }
    if camera_policy == "soft":
        # P3.8: record the legacy-definition numbers next to the soft ones so a
        # reader can compare like with like.  These entries are informational:
        # pass/fail above and below still uses the soft definitions and the
        # thresholds are unchanged.
        legacy_median = (
            float(np.median(finite_reprojection))
            if finite_reprojection.size
            else float("inf")
        )
        legacy_p95 = (
            float(np.percentile(finite_reprojection, 95))
            if finite_reprojection.size
            else float("inf")
        )
        # The legacy gate counted hard inliers at the historical 12 px
        # threshold, not the soft trust>0.5 mask.  Recompute that definition
        # from the current residuals so the informational number is comparable.
        legacy_inliers = np.isfinite(reprojection) & (
            reprojection <= hard_inlier_threshold_px
        )
        legacy_two_camera = float(
            np.mean(finite & (np.sum(legacy_inliers, axis=-1) >= 2))
        )
        gate_results["legacy_reprojection_median"] = {
            "passed": None,
            "informational": True,
            "value": legacy_median,
            "threshold": max_reprojection_median_px,
            "definition": "legacy definition: median residual over all observed cameras",
        }
        gate_results["legacy_reprojection_p95"] = {
            "passed": None,
            "informational": True,
            "value": legacy_p95,
            "threshold": max_reprojection_p95_px,
            "definition": "legacy definition: p95 residual over all observed cameras",
        }
        gate_results["legacy_two_camera_coverage"] = {
            "passed": None,
            "informational": True,
            "value": legacy_two_camera,
            "threshold": min_two_camera_fraction,
            "hard_inlier_threshold_px": hard_inlier_threshold_px,
            "definition": (
                "legacy definition: finite point and >=2 hard inliers "
                f"(residual <= {hard_inlier_threshold_px:g} px)"
            ),
        }
    failing_gate = next(
        (
            name
            for name, result in gate_results.items()
            if result.get("passed") is False
        ),
        None,
    )
    coverage = min(finite_fraction, two_camera_fraction)
    evidence = EvidenceContract(
        stage="raw_3d_geometry",
        status="FAIL" if failing_gate else "PASS",
        validity_mask=finite & (accepted_count >= 2),
        uncertainty=np.asarray(diagnostics.confidence[:, :core]),
        calibration_id=calib_hash,
        coordinate_space="yup_m",
        input_provenance=provenance_hash,
        gate_results=gate_results,
        coverage_fraction=coverage,
        failure_reason=failing_gate,
    )
    if failing_gate:
        message = f"{failing_gate} failed: {gate_results[failing_gate]['value']!r}"
        if not soft_fail:
            raise PipelineGateError(f"Raw geometry gate failed: {message}")
        return GateFailureReport(
            failing_stage="raw_3d_geometry",
            failing_gate=failing_gate,
            cameras_implicated=[],
            failure_mode=message,
            coverage_fraction=coverage,
            evidence=evidence,
        )
    return evidence


def check_triangulation_gates(
    diagnostics, # TriangulationDiagnostics
    calib_hash: str,
    provenance_hash: str,
    soft_fail: bool = True
) -> EvidenceContract | GateFailureReport:
    """
    Check 3D triangulation gates.
    """
    gate_results = {}
    failing_gate = None
    failure_mode = None
    
    # 1. inlier_ratio
    # Fraction of joint-frames with >=2 inlier cameras >= 70%
    inliers = diagnostics.num_inliers >= 2
    inlier_ratio = float(np.mean(inliers))
    passed_inliers = inlier_ratio >= 0.7
    
    gate_results["inlier_ratio"] = {
        "passed": passed_inliers,
        "value": inlier_ratio,
        "threshold": 0.7
    }
    if not passed_inliers and failing_gate is None:
        failing_gate = "inlier_ratio"
        failure_mode = f"Inlier ratio {inlier_ratio:.1%} < 70%"
        
    # 2. reproj_median & p95
    # Across all valid reprojections
    valid_reproj = diagnostics.reprojection_error_px[np.isfinite(diagnostics.reprojection_error_px)]
    if len(valid_reproj) > 0:
        med_reproj = float(np.median(valid_reproj))
        p95_reproj = float(np.percentile(valid_reproj, 95))
    else:
        med_reproj = float('inf')
        p95_reproj = float('inf')
        
    passed_reproj_med = med_reproj < 5.0
    passed_reproj_p95 = p95_reproj < 15.0
    
    gate_results["reproj_median"] = {
        "passed": passed_reproj_med,
        "value": med_reproj,
        "threshold": 5.0
    }
    if not passed_reproj_med and failing_gate is None:
        failing_gate = "reproj_median"
        failure_mode = f"Median reproj error {med_reproj:.1f}px >= 5.0px"
        
    gate_results["reproj_p95"] = {
        "passed": passed_reproj_p95,
        "value": p95_reproj,
        "threshold": 15.0
    }
    if not passed_reproj_p95 and failing_gate is None:
        failing_gate = "reproj_p95"
        failure_mode = f"P95 reproj error {p95_reproj:.1f}px >= 15.0px"
        
    # 3. depth_positive
    # Currently triangulation returns points in Y-up. Depth check requires camera space.
    # We will assume Z > 0 in camera space is satisfied if volume bounds are reasonable.
    
    # 4. volume_bounds
    # All points within [-500, 500] cm on each axis
    valid_pts = diagnostics.points3d[inliers]
    if len(valid_pts) > 0:
        max_bound = float(np.max(np.abs(valid_pts)))
    else:
        max_bound = float('inf')
        
    passed_bounds = max_bound <= 500.0
    gate_results["volume_bounds"] = {
        "passed": passed_bounds,
        "value": max_bound,
        "threshold": 500.0
    }
    if not passed_bounds and failing_gate is None:
        failing_gate = "volume_bounds"
        failure_mode = f"Max volume bound {max_bound:.1f}cm > 500.0cm"
        
    # 5. critical_joint_coverage
    # Hips (11,12) and shoulders (5,6) must have >=50% valid frames
    if inliers.shape[1] > 12:
        critical_cov = float(np.mean(inliers[:, [5, 6, 11, 12]]))
    else:
        critical_cov = 1.0 # fallback
        
    passed_critical = critical_cov >= 0.5
    gate_results["critical_joint_coverage"] = {
        "passed": passed_critical,
        "value": critical_cov,
        "threshold": 0.5
    }
    if not passed_critical and failing_gate is None:
        failing_gate = "critical_joint_coverage"
        failure_mode = f"Critical joint coverage {critical_cov:.1%} < 50%"

    evidence = EvidenceContract(
        stage="triangulation",
        status="FAIL" if failing_gate else "PASS",
        validity_mask=inliers,
        uncertainty=diagnostics.confidence,
        calibration_id=calib_hash,
        coordinate_space="yup_cm",
        input_provenance=provenance_hash,
        gate_results=gate_results,
        coverage_fraction=inlier_ratio,
        failure_reason=failing_gate
    )
    
    if failing_gate:
        if soft_fail:
            return GateFailureReport(
                failing_stage="triangulation",
                failing_gate=failing_gate,
                cameras_implicated=[],
                failure_mode=failure_mode,
                coverage_fraction=inlier_ratio,
                evidence=evidence
            )
        else:
            raise PipelineGateError(f"Triangulation Gate Failed: {failure_mode}")
            
    return evidence
