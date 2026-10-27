"""Phase 2.3B test suite: Explicit Floor-World Transform Infrastructure.

Covers:
- Test A: Identity transform returns identical points
- Test B: Known rotation produces exact expected coordinates
- Test C: Known translation produces exact expected coordinates
- Test D: Complete rotation + translation verified against manual calculation
- Test E: Invertible round-trip (camera -> floor -> camera) on complex points
- Test F: Pairwise Euclidean distance and bone length preservation
- Test G: Canonical (T, 133, 3) shape preservation
- Test H: Diagnostics arrays remain aligned and untouched
- Test I: Canonical reconstruction non-mutation contract
- Test J: Metadata serialization and deserialization round-trip
- Test K: Frame mismatch and double-transformation rejection
- Test L: Unregistered identity state contract
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from aimocap.math.coords import (
    CAMERA_WORLD_FRAME,
    FLOOR_WORLD_FRAME,
    FloorTransform,
    FrameMismatchError,
    camera_extrinsics_to_floor,
    transform_sequence_from_floor,
    transform_sequence_to_floor,
    validate_rotation,
)


# ── Test A: Identity Transform ───────────────────────────────────────────────

def test_a_identity_transform_preserves_points():
    """Test A: T = identity must return identical points."""
    transform = FloorTransform.identity()
    points = np.array([
        [1.0, 2.0, 3.0],
        [-4.5, 0.0, 10.2],
        [0.0, -1.0, -2.5],
    ])
    transformed = transform.transform_points(points)
    np.testing.assert_allclose(transformed, points, atol=1e-12)


# ── Test B: Known Rotation ───────────────────────────────────────────────────

def test_b_known_rotation():
    """Test B: A known +90 degree rotation about Z produces exact expected coordinates."""
    # Rotation of +90 deg about Z: (x, y, z) -> (-y, x, z)
    # R = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    R_z90 = np.array([
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    transform = FloorTransform(rotation=R_z90, translation=np.zeros(3))

    points = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 5.0],
        [3.0, 4.0, 1.0],
    ])
    expected = np.array([
        [0.0, 1.0, 0.0],
        [-2.0, 0.0, 0.0],
        [0.0, 0.0, 5.0],
        [-4.0, 3.0, 1.0],
    ])
    transformed = transform.transform_points(points)
    np.testing.assert_allclose(transformed, expected, atol=1e-12)


# ── Test C: Known Translation ────────────────────────────────────────────────

def test_c_known_translation():
    """Test C: p_floor = p_camera + t for identity rotation."""
    t = np.array([10.5, -3.2, 100.0])
    transform = FloorTransform(rotation=np.eye(3), translation=t)

    points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 2.0, 3.0],
        [-10.0, 5.0, -50.0],
    ])
    expected = points + t
    transformed = transform.transform_points(points)
    np.testing.assert_allclose(transformed, expected, atol=1e-12)


# ── Test D: Rotation + Translation ───────────────────────────────────────────

def test_d_rotation_and_translation():
    """Test D: Complete transform p_floor = R @ p_camera + t against manual calculation."""
    # Rotate +90 deg about X: (x, y, z) -> (x, -z, y)
    # R = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
    R_x90 = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ])
    t = np.array([2.0, -5.0, 10.0])
    transform = FloorTransform(rotation=R_x90, translation=t)

    p_camera = np.array([3.0, 4.0, 7.0])
    # Expected: R @ p + t = [3, -7, 4] + [2, -5, 10] = [5, -12, 14]
    expected = np.array([5.0, -12.0, 14.0])
    transformed = transform.transform_points(p_camera)
    np.testing.assert_allclose(transformed, expected, atol=1e-12)


# ── Test E: Inverse Round Trip ───────────────────────────────────────────────

def test_e_inverse_round_trip():
    """Test E: camera -> floor -> camera reproduces original points within numerical tolerance."""
    # Nontrivial 3D rotation and translation
    rot = Rotation.from_euler("xyz", [27.5, -43.2, 115.8], degrees=True).as_matrix()
    trans = np.array([-14.2, 88.6, -105.3])
    transform = FloorTransform(rotation=rot, translation=trans)

    # Test cases: origin, negative coordinates, large values, multi-frame shape (10, 25, 3)
    rng = np.random.default_rng(12345)
    points = rng.uniform(-500.0, 500.0, size=(10, 25, 3))
    points[0, 0] = [0.0, 0.0, 0.0]  # Origin

    floor_points = transform.transform_points(points)
    recovered_points = transform.inverse_transform_points(floor_points)

    np.testing.assert_allclose(recovered_points, points, atol=1e-11)

    # Also test via inverse() transform object
    inv_transform = transform.inverse()
    recovered_via_inv = inv_transform.transform_points(floor_points)
    np.testing.assert_allclose(recovered_via_inv, points, atol=1e-11)


# ── Test F: Distance and Bone Length Preservation ───────────────────────────

def test_f_distance_and_bone_length_preservation():
    """Test F: Rigid transform strictly preserves Euclidean distances and skeleton bone lengths."""
    rot = Rotation.from_euler("zyx", [-35.0, 52.0, 18.0], degrees=True).as_matrix()
    trans = np.array([12.0, -34.0, 56.0])
    transform = FloorTransform(rotation=rot, translation=trans)

    # Define synthetic limbs (e.g. hip to knee, knee to ankle)
    hip = np.array([10.0, -5.0, 95.0])
    knee = np.array([12.0, -2.0, 50.0])
    ankle = np.array([11.0, 0.0, 8.0])

    thigh_length_orig = np.linalg.norm(hip - knee)
    shin_length_orig = np.linalg.norm(knee - ankle)

    skeleton = np.stack([hip, knee, ankle])
    transformed_skeleton = transform.transform_points(skeleton)

    thigh_length_trans = np.linalg.norm(transformed_skeleton[0] - transformed_skeleton[1])
    shin_length_trans = np.linalg.norm(transformed_skeleton[1] - transformed_skeleton[2])

    np.testing.assert_allclose(thigh_length_trans, thigh_length_orig, atol=1e-11)
    np.testing.assert_allclose(shin_length_trans, shin_length_orig, atol=1e-11)


# ── Test G: Canonical (T, 133, 3) Shape ──────────────────────────────────────

def test_g_canonical_133_shape_preservation():
    """Test G: (T, 133, 3) sequence maintains identical shape through the transform."""
    transform = FloorTransform(
        rotation=Rotation.from_euler("y", 45, degrees=True).as_matrix(),
        translation=np.array([1.0, 2.0, 3.0]),
    )
    t_frames = 15
    camera_world = np.ones((t_frames, 133, 3), dtype=np.float64)

    floor_world = transform.transform_points(camera_world)
    assert floor_world.shape == (t_frames, 133, 3)
    assert floor_world.dtype == np.float64


# ── Test H: Diagnostics Untouched ───────────────────────────────────────────

def test_h_diagnostics_untouched():
    """Test H: Diagnostics arrays are unaffected and remain aligned with original observations."""
    t_frames, j_joints, c_cams = 10, 133, 3
    points3d = np.ones((t_frames, j_joints, 3), dtype=np.float64)

    # Simulated diagnostic arrays
    confidence = np.full((t_frames, j_joints), 0.85, dtype=np.float32)
    inlier_mask = np.ones((t_frames, j_joints, c_cams), dtype=bool)
    reprojection_error_px = np.full((t_frames, j_joints, c_cams), 1.42, dtype=np.float32)
    ray_angle_deg = np.full((t_frames, j_joints), 45.0, dtype=np.float32)

    confidence_orig = confidence.copy()
    inlier_orig = inlier_mask.copy()
    reproj_orig = reprojection_error_px.copy()
    ray_orig = ray_angle_deg.copy()

    transform = FloorTransform(
        rotation=Rotation.from_euler("z", 30, degrees=True).as_matrix(),
        translation=np.array([5.0, -10.0, 2.0]),
    )

    # Transform only affects 3D coordinates
    transformed_points = transform_sequence_to_floor(points3d, transform)
    assert transformed_points.shape == points3d.shape

    # Verify diagnostic arrays remain identical
    np.testing.assert_array_equal(confidence, confidence_orig)
    np.testing.assert_array_equal(inlier_mask, inlier_orig)
    np.testing.assert_array_equal(reprojection_error_px, reproj_orig)
    np.testing.assert_array_equal(ray_angle_deg, ray_orig)


# ── Test I: Canonical Non-Mutation Contract ──────────────────────────────────

def test_i_canonical_non_mutation():
    """Test I: Applying floor transform does NOT modify the canonical skeleton3d input array."""
    rng = np.random.default_rng(999)
    skeleton3d = rng.normal(0.0, 50.0, size=(20, 133, 3))
    original_copy = skeleton3d.copy()

    transform = FloorTransform(
        rotation=Rotation.from_euler("xz", [15.0, -25.0], degrees=True).as_matrix(),
        translation=np.array([-50.0, 120.0, 30.0]),
    )

    skeleton3d_floor = transform.transform_points(skeleton3d)

    # Original array must be bitwise identical
    np.testing.assert_array_equal(skeleton3d, original_copy)
    # Output must be different from input
    assert not np.allclose(skeleton3d_floor, skeleton3d)


# ── Test J: Metadata Round Trip ─────────────────────────────────────────────

def test_j_metadata_serialization_round_trip(tmp_path: Path):
    """Test J: Serialize FloorTransform to dict and JSON file and recover exact properties."""
    rot = Rotation.from_euler("xyz", [12.0, 34.0, 56.0], degrees=True).as_matrix()
    trans = np.array([1.23, -4.56, 7.89])
    transform = FloorTransform(
        rotation=rot,
        translation=trans,
        source_frame=CAMERA_WORLD_FRAME,
        target_frame=FLOOR_WORLD_FRAME,
        units="meters",
        transform_version=1,
        grounding_status="registered",
        source="physical_floor_reference",
    )

    # 1. Dict round-trip
    d = transform.to_dict()
    recovered_from_dict = FloorTransform.from_dict(d)
    np.testing.assert_allclose(recovered_from_dict.rotation, rot, atol=1e-12)
    np.testing.assert_allclose(recovered_from_dict.translation, trans, atol=1e-12)
    assert recovered_from_dict.source_frame == CAMERA_WORLD_FRAME
    assert recovered_from_dict.target_frame == FLOOR_WORLD_FRAME
    assert recovered_from_dict.units == "meters"
    assert recovered_from_dict.transform_version == 1
    assert recovered_from_dict.grounding_status == "registered"
    assert recovered_from_dict.source == "physical_floor_reference"
    assert recovered_from_dict.is_registered is True

    # 2. File JSON round-trip
    json_path = tmp_path / "floor_transform.json"
    transform.to_json(json_path)
    recovered_from_json = FloorTransform.from_json(json_path)
    np.testing.assert_allclose(recovered_from_json.rotation, rot, atol=1e-12)
    np.testing.assert_allclose(recovered_from_json.translation, trans, atol=1e-12)
    assert recovered_from_json.grounding_status == "registered"


# ── Test K: Frame Mismatch / Double-Transform Protection ─────────────────────

def test_k_frame_mismatch_and_double_transform_protection():
    """Test K: Attempting to apply a camera->floor transform to an input already in floor_world fails loudly."""
    transform = FloorTransform.identity(
        source_frame=CAMERA_WORLD_FRAME,
        target_frame=FLOOR_WORLD_FRAME,
    )
    points = np.zeros((5, 133, 3))

    # Valid application
    out = transform.transform_points(points, current_frame=CAMERA_WORLD_FRAME)
    assert out.shape == points.shape

    # Double-transform attempt: input is already in FLOOR_WORLD_FRAME
    with pytest.raises(FrameMismatchError, match="Double-transform rejected"):
        transform.transform_points(out, current_frame=FLOOR_WORLD_FRAME)

    # Unknown frame mismatch attempt
    with pytest.raises(FrameMismatchError, match="Frame mismatch"):
        transform.transform_points(points, current_frame="some_arbitrary_frame")

    # Inverse transform mismatch: cannot invert points that claim to be in CAMERA_WORLD_FRAME
    with pytest.raises(FrameMismatchError, match="Inverse transform rejected"):
        transform.inverse_transform_points(points, current_frame=CAMERA_WORLD_FRAME)


# ── Test L: Unregistered Identity Contract ──────────────────────────────────

def test_l_unregistered_identity_contract():
    """Test L: Identity transform with grounding_status='unregistered' is never reported as physically grounded."""
    unregistered_tf = FloorTransform.identity(grounding_status="unregistered", source="none")
    np.testing.assert_allclose(unregistered_tf.rotation, np.eye(3))
    np.testing.assert_allclose(unregistered_tf.translation, np.zeros(3))
    assert unregistered_tf.grounding_status == "unregistered"
    assert unregistered_tf.source == "none"
    assert unregistered_tf.is_registered is False

    # Check validation: invalid status rejected
    with pytest.raises(ValueError, match="grounding_status must be"):
        FloorTransform(grounding_status="fabricated_status")

    # Check validation: non-orthonormal matrix rejected
    with pytest.raises(ValueError, match="rotation is not orthonormal"):
        FloorTransform(rotation=np.array([[2.0, 0, 0], [0, 1, 0], [0, 0, 1]]))


# ── Camera Extrinsics Conversion (Explicit Utility) ──────────────────────────

def test_camera_extrinsics_to_floor_preserves_projection():
    """Verify camera_extrinsics_to_floor rewrites extrinsics to yield identical projections in floor-world."""
    rot = Rotation.from_euler("zy", [45.0, -30.0], degrees=True).as_matrix()
    trans = np.array([2.5, -1.0, 0.5])
    floor_tf = FloorTransform(rotation=rot, translation=trans)

    R_cam = Rotation.from_euler("x", 20.0, degrees=True).as_matrix()
    t_cam = np.array([[0.5], [1.2], [3.0]])

    p_camera_world = np.array([
        [1.0, -0.5, 4.0],
        [-2.0, 1.5, 6.0],
    ])

    # Project in camera world: X_c = R_cam @ X_cw + t_cam
    p_cam_space_orig = (R_cam @ p_camera_world.T + t_cam).T

    # Transform points to floor world
    p_floor_world = floor_tf.transform_points(p_camera_world)

    # Transform extrinsics to floor world
    R_cam_floor, t_cam_floor = camera_extrinsics_to_floor(R_cam, t_cam, floor_tf)

    # Project in floor world: X_c = R_cam_floor @ X_fw + t_cam_floor
    p_cam_space_floor = (R_cam_floor @ p_floor_world.T + t_cam_floor).T

    np.testing.assert_allclose(p_cam_space_floor, p_cam_space_orig, atol=1e-11)
