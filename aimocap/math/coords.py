"""Coordinate system definitions and conversions.

This module defines the internal canonical 3D coordinate system for the pipeline:
**Y-up, right-handed, meters**.

Conventions:
- X: Right
- Y: Up
- Z: Back (towards the viewer)

This matches glTF, Three.js, and Blender. It minimizes conversions at the export stage.
OpenCV's native 3D space (from triangulation) is Y-down, Z-forward (right-handed).
We convert from OpenCV space to our internal space immediately after triangulation.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


OPENCV_COORDINATE_SYSTEM = "opencv_right_handed_y_down_z_forward"
INTERNAL_COORDINATE_SYSTEM = "internal_y_up_right_handed_z_back"
CAMERA_WORLD_FRAME = "camera_world"
FLOOR_WORLD_FRAME = "floor_world"


class FrameMismatchError(ValueError):
    """Raised when applying a transform to an array in an unexpected frame."""
    pass


def validate_rotation(rotation: np.ndarray, atol: float = 1e-6) -> np.ndarray:
    """Validate and return a proper 3x3 rotation matrix."""
    R = np.asarray(rotation, dtype=np.float64)
    if R.shape != (3, 3) or not np.isfinite(R).all():
        raise ValueError("rotation must be a finite 3x3 matrix")
    if not np.allclose(R.T @ R, np.eye(3), atol=atol):
        raise ValueError("rotation is not orthonormal")
    if not np.isclose(np.linalg.det(R), 1.0, atol=atol):
        raise ValueError("rotation must have determinant +1")
    return R


class DisplayTransform:
    """Rigid transform from OpenCV world coordinates to display coordinates.

    ``X_display = rotation @ X_opencv + translation``.  The transform is
    intentionally explicit so world-frame changes do not get mixed into
    camera extrinsics or hidden in a plotting/exporter call.
    """

    def __init__(self, rotation: np.ndarray | None = None, translation: np.ndarray | None = None):
        self.rotation = validate_rotation(np.eye(3) if rotation is None else rotation)
        self.translation = np.zeros(3, dtype=np.float64) if translation is None else np.asarray(translation, dtype=np.float64).reshape(3)
        if not np.isfinite(self.translation).all():
            raise ValueError("translation must be finite")


def camera_center(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Return the world-space camera center for ``Xc = R Xw + t``."""
    R = validate_rotation(rotation)
    t = np.asarray(translation, dtype=np.float64).reshape(3)
    return -R.T @ t


def camera_extrinsics_to_display(
    rotation: np.ndarray,
    translation: np.ndarray,
    display: DisplayTransform,
) -> tuple[np.ndarray, np.ndarray]:
    """Rewrite world-to-camera extrinsics after a display-world transform."""
    R = validate_rotation(rotation)
    t = np.asarray(translation, dtype=np.float64).reshape(3, 1)
    R_display = display.rotation
    R_out = R @ R_display.T
    t_out = t - R_out @ display.translation.reshape(3, 1)
    return R_out, t_out


def transform_world_points(points: np.ndarray, display: DisplayTransform) -> np.ndarray:
    """Transform OpenCV-world points into display coordinates."""
    pts = np.asarray(points, dtype=np.float64)
    return pts @ display.rotation.T + display.translation


def inverse_transform_world_points(points: np.ndarray, display: DisplayTransform) -> np.ndarray:
    """Transform display coordinates back into OpenCV-world coordinates."""
    pts = np.asarray(points, dtype=np.float64)
    return (pts - display.translation) @ display.rotation

def opencv_to_internal(points_3d: np.ndarray) -> np.ndarray:
    """Convert points from OpenCV space to the internal Y-up, right-handed space.
    
    OpenCV: X right, Y down, Z forward
    Internal: X right, Y up, Z back
    
    Args:
        points_3d: Array of shape (..., 3) containing 3D points in OpenCV space.
        
    Returns:
        Array of the same shape in internal coordinate space.
    """
    out = np.copy(points_3d)
    out[..., 1] = -out[..., 1]  # Y up
    out[..., 2] = -out[..., 2]  # Z back
    return out

def internal_to_opencv(points_3d: np.ndarray) -> np.ndarray:
    """Convert points from the internal Y-up space back to OpenCV space.
    
    This is useful for reprojection (rendering 3D points back to 2D cameras).
    """
    out = np.copy(points_3d)
    out[..., 1] = -out[..., 1]  # Y down
    out[..., 2] = -out[..., 2]  # Z forward
    return out


class FloorTransform:
    """Rigid transform from canonical camera-world coordinates to derived floor-world coordinates.

    Mathematical contract:
        p_floor = R_cf @ p_camera + t_cf
        p_camera = R_cf.T @ (p_floor - t_cf)

    where:
    - R_cf is a 3x3 proper rotation matrix (R^T R = I, det(R) = +1)
    - t_cf is a 3-vector translation
    - Metric scale and distances are strictly preserved (pure Euclidean isometry).
    """

    def __init__(
        self,
        rotation: np.ndarray | None = None,
        translation: np.ndarray | None = None,
        *,
        source_frame: str = CAMERA_WORLD_FRAME,
        target_frame: str = FLOOR_WORLD_FRAME,
        units: str = "meters",
        transform_version: int = 1,
        grounding_status: str = "unregistered",
        source: str = "unregistered",
        validation_status: str | None = None,
        failure_reasons: list[str] | None = None,
        epoch_id: str = "",
    ):
        self.rotation = validate_rotation(np.eye(3) if rotation is None else rotation)
        self.translation = (
            np.zeros(3, dtype=np.float64)
            if translation is None
            else np.asarray(translation, dtype=np.float64).reshape(3)
        )
        if not np.isfinite(self.translation).all():
            raise ValueError("translation must be finite")

        if validation_status is None:
            validation_status = "passed" if grounding_status == "registered" else "missing"

        if grounding_status not in ("unregistered", "registered"):
            raise ValueError(
                f"grounding_status must be 'unregistered' or 'registered', got {grounding_status!r}"
            )
        if validation_status not in ("missing", "passed", "failed", "stale"):
            raise ValueError(
                f"validation_status must be 'missing', 'passed', 'failed', or 'stale', got {validation_status!r}"
            )

        self.source_frame = str(source_frame)
        self.target_frame = str(target_frame)
        self.units = str(units)
        self.transform_version = int(transform_version)
        self.grounding_status = str(grounding_status)
        self.source = str(source)
        self.validation_status = str(validation_status)
        self.failure_reasons = list(failure_reasons or [])
        self.epoch_id = str(epoch_id)

    @property
    def is_registered(self) -> bool:
        """Return True only if this transform represents an authoritative physical floor measurement."""
        return self.grounding_status == "registered" and self.validation_status == "passed"

    @classmethod
    def identity(
        cls,
        *,
        source_frame: str = CAMERA_WORLD_FRAME,
        target_frame: str = FLOOR_WORLD_FRAME,
        units: str = "meters",
        grounding_status: str = "unregistered",
        source: str = "none",
        validation_status: str = "missing",
        failure_reasons: list[str] | None = None,
        epoch_id: str = "",
    ) -> FloorTransform:
        """Create an identity transform.

        Note: By default grounding_status is 'unregistered'. An identity transform
        does NOT imply the capture is physically floor-grounded; it explicitly records
        that no floor transform has yet been measured.
        """
        return cls(
            rotation=np.eye(3),
            translation=np.zeros(3),
            source_frame=source_frame,
            target_frame=target_frame,
            units=units,
            grounding_status=grounding_status,
            source=source,
            validation_status=validation_status,
            failure_reasons=failure_reasons,
            epoch_id=epoch_id,
        )

    def _validate_frame_tag(
        self,
        current_frame: str | None,
        *,
        inverse: bool,
        noun: str,
    ) -> None:
        """Shared frame-tag guard for all transform_* methods.

        Raises FrameMismatchError on a double-transform (tag already names the
        destination frame) or on a tag matching neither frame endpoint.
        `noun` carries the subject+verb so error wording stays exact, e.g.
        "input points are", "velocity is".
        """
        if current_frame is None:
            return
        src, dst = (
            (self.target_frame, self.source_frame)
            if inverse
            else (self.source_frame, self.target_frame)
        )
        direction = "inverse transform" if inverse else "transform"
        side = "target" if inverse else "source"
        if current_frame == dst:
            raise FrameMismatchError(
                f"{'Inverse transform' if inverse else 'Double-transform'} rejected: "
                f"{noun} already in {side} frame '{dst}'"
            )
        if current_frame != src:
            raise FrameMismatchError(
                f"Frame mismatch: {direction} expects {side} frame '{src}', got '{current_frame}'"
            )

    def transform_points(
        self,
        points: np.ndarray,
        *,
        current_frame: str | None = None,
    ) -> np.ndarray:
        """Transform points from source_frame (camera_world) to target_frame (floor_world).

        p_floor = p_camera @ R^T + t
        Preserves arbitrary leading dimensions (..., 3) and is strictly non-mutating.
        """
        self._validate_frame_tag(current_frame, inverse=False, noun="input points are")

        pts = np.asarray(points, dtype=np.float64)
        if pts.shape[-1] != 3:
            raise ValueError(f"points last dimension must be 3, got shape {pts.shape}")

        return pts @ self.rotation.T + self.translation

    def inverse_transform_points(
        self,
        points: np.ndarray,
        *,
        current_frame: str | None = None,
    ) -> np.ndarray:
        """Transform points from target_frame (floor_world) back to source_frame (camera_world).

        p_camera = (p_floor - t) @ R
        Preserves arbitrary leading dimensions (..., 3) and is strictly non-mutating.
        """
        self._validate_frame_tag(current_frame, inverse=True, noun="input points are")

        pts = np.asarray(points, dtype=np.float64)
        if pts.shape[-1] != 3:
            raise ValueError(f"points last dimension must be 3, got shape {pts.shape}")

        return (pts - self.translation) @ self.rotation

    def transform_velocity(
        self,
        velocity: np.ndarray,
        *,
        current_frame: str | None = None,
    ) -> np.ndarray:
        """Transform velocity vectors from camera_world to floor_world (pure rotation, no translation)."""
        self._validate_frame_tag(current_frame, inverse=False, noun="velocity is")

        vel = np.asarray(velocity, dtype=np.float64)
        if vel.shape[-1] != 3:
            raise ValueError(f"velocity last dimension must be 3, got shape {vel.shape}")
        return vel @ self.rotation.T

    def inverse_transform_velocity(
        self,
        velocity: np.ndarray,
        *,
        current_frame: str | None = None,
    ) -> np.ndarray:
        """Transform velocity vectors from floor_world back to camera_world (pure rotation)."""
        self._validate_frame_tag(current_frame, inverse=True, noun="velocity is")

        vel = np.asarray(velocity, dtype=np.float64)
        if vel.shape[-1] != 3:
            raise ValueError(f"velocity last dimension must be 3, got shape {vel.shape}")
        return vel @ self.rotation

    def transform_acceleration(
        self,
        acceleration: np.ndarray,
        *,
        current_frame: str | None = None,
    ) -> np.ndarray:
        """Transform acceleration vectors from camera_world to floor_world (pure rotation, no translation)."""
        return self.transform_velocity(acceleration, current_frame=current_frame)

    def inverse_transform_acceleration(
        self,
        acceleration: np.ndarray,
        *,
        current_frame: str | None = None,
    ) -> np.ndarray:
        """Transform acceleration vectors from floor_world back to camera_world (pure rotation)."""
        return self.inverse_transform_velocity(acceleration, current_frame=current_frame)

    def transform_covariance(
        self,
        covariance: np.ndarray,
        *,
        current_frame: str | None = None,
    ) -> np.ndarray:
        """Transform 3x3 covariance matrix from camera_world to floor_world: Sigma_floor = R @ Sigma @ R.T."""
        self._validate_frame_tag(current_frame, inverse=False, noun="covariance is")

        cov = np.asarray(covariance, dtype=np.float64)
        if cov.shape[-2:] != (3, 3):
            raise ValueError(f"covariance must have shape (..., 3, 3), got {cov.shape}")
        return self.rotation @ cov @ self.rotation.T

    def inverse_transform_covariance(
        self,
        covariance: np.ndarray,
        *,
        current_frame: str | None = None,
    ) -> np.ndarray:
        """Transform 3x3 covariance matrix from floor_world back to camera_world: Sigma_camera = R.T @ Sigma @ R."""
        self._validate_frame_tag(current_frame, inverse=True, noun="covariance is")

        cov = np.asarray(covariance, dtype=np.float64)
        if cov.shape[-2:] != (3, 3):
            raise ValueError(f"covariance must have shape (..., 3, 3), got {cov.shape}")
        return self.rotation.T @ cov @ self.rotation

    def inverse(self) -> FloorTransform:
        """Return the exact mathematical inverse FloorTransform."""
        return FloorTransform(
            rotation=self.rotation.T,
            translation=-self.rotation.T @ self.translation,
            source_frame=self.target_frame,
            target_frame=self.source_frame,
            units=self.units,
            transform_version=self.transform_version,
            grounding_status=self.grounding_status,
            source=self.source,
            validation_status=self.validation_status,
            failure_reasons=self.failure_reasons,
            epoch_id=self.epoch_id,
        )

    def to_dict(self) -> dict:
        """Return a JSON-serializable dictionary."""
        return {
            "transform_version": self.transform_version,
            "source_frame": self.source_frame,
            "target_frame": self.target_frame,
            "units": self.units,
            "grounding_status": self.grounding_status,
            "source": self.source,
            "validation_status": self.validation_status,
            "failure_reasons": list(self.failure_reasons),
            "epoch_id": self.epoch_id,
            "rotation": self.rotation.tolist(),
            "translation": self.translation.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> FloorTransform:
        """Construct a FloorTransform from a dictionary."""
        return cls(
            rotation=np.asarray(data.get("rotation", np.eye(3)), dtype=np.float64),
            translation=np.asarray(data.get("translation", np.zeros(3)), dtype=np.float64),
            source_frame=data.get("source_frame", CAMERA_WORLD_FRAME),
            target_frame=data.get("target_frame", FLOOR_WORLD_FRAME),
            units=data.get("units", "meters"),
            transform_version=int(data.get("transform_version", 1)),
            grounding_status=data.get("grounding_status", "unregistered"),
            source=data.get("source", "unregistered"),
            validation_status=data.get("validation_status", "missing"),
            failure_reasons=data.get("failure_reasons", []),
            epoch_id=data.get("epoch_id", ""),
        )

    def to_json(self, path: str | Path) -> None:
        """Save the transform to a JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: str | Path) -> FloorTransform:
        """Load the transform from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def __repr__(self) -> str:
        return (
            f"FloorTransform(version={self.transform_version}, "
            f"from='{self.source_frame}' -> to='{self.target_frame}', "
            f"status='{self.grounding_status}', validation='{self.validation_status}', "
            f"source='{self.source}', units='{self.units}')"
        )


def transform_sequence_to_floor(
    points: np.ndarray,
    transform: FloorTransform,
    *,
    current_frame: str | None = None,
) -> np.ndarray:
    """Transform point array (..., 3) from camera-world to floor-world."""
    return transform.transform_points(points, current_frame=current_frame)


def transform_sequence_from_floor(
    points: np.ndarray,
    transform: FloorTransform,
    *,
    current_frame: str | None = None,
) -> np.ndarray:
    """Transform point array (..., 3) from floor-world back to camera-world."""
    return transform.inverse_transform_points(points, current_frame=current_frame)


def camera_extrinsics_to_floor(
    rotation: np.ndarray,
    translation: np.ndarray,
    transform: FloorTransform,
) -> tuple[np.ndarray, np.ndarray]:
    """Rewrite world-to-camera extrinsics after applying a camera-world -> floor-world transform.

    In camera-world:
        X_cam = R_c @ X_cw + t_c

    In floor-world:
        X_fw = R_cf @ X_cw + t_cf  =>  X_cw = R_cf.T @ (X_fw - t_cf)

    Substituting into camera projection:
        X_cam = (R_c @ R_cf.T) @ X_fw + (t_c - (R_c @ R_cf.T) @ t_cf)
    """
    R_c = validate_rotation(rotation)
    t_c = np.asarray(translation, dtype=np.float64).reshape(3, 1)
    R_cf = transform.rotation
    t_cf = transform.translation.reshape(3, 1)

    R_out = R_c @ R_cf.T
    t_out = t_c - R_out @ t_cf
    return R_out, t_out


def transform_velocity_to_floor(
    velocity: np.ndarray,
    transform: FloorTransform,
    *,
    current_frame: str | None = None,
) -> np.ndarray:
    """Transform velocity vectors (..., 3) from camera-world to floor-world (pure rotation)."""
    return transform.transform_velocity(velocity, current_frame=current_frame)


def transform_velocity_from_floor(
    velocity: np.ndarray,
    transform: FloorTransform,
    *,
    current_frame: str | None = None,
) -> np.ndarray:
    """Transform velocity vectors (..., 3) from floor-world back to camera-world (pure rotation)."""
    return transform.inverse_transform_velocity(velocity, current_frame=current_frame)


def transform_covariance_to_floor(
    covariance: np.ndarray,
    transform: FloorTransform,
    *,
    current_frame: str | None = None,
) -> np.ndarray:
    """Transform covariance matrices (..., 3, 3) from camera-world to floor-world."""
    return transform.transform_covariance(covariance, current_frame=current_frame)


def transform_covariance_from_floor(
    covariance: np.ndarray,
    transform: FloorTransform,
    *,
    current_frame: str | None = None,
) -> np.ndarray:
    """Transform covariance matrices (..., 3, 3) from floor-world back to camera-world."""
    return transform.inverse_transform_covariance(covariance, current_frame=current_frame)

