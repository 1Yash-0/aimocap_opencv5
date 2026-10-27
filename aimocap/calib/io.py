"""Input/output for camera calibration data."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from aimocap.math.coords import OPENCV_COORDINATE_SYSTEM, validate_rotation


CALIBRATION_PROJECTION_CONVENTION = "world_to_camera"


def _unit_scale_to_meters(units: str | None) -> tuple[str, float] | tuple[None, None]:
    """Return a canonical source-unit label and conversion factor to metres."""
    if units is None:
        return None, None
    aliases = {
        "m": ("meters", 1.0),
        "meter": ("meters", 1.0),
        "meters": ("meters", 1.0),
        "cm": ("cm", 0.01),
        "centimeter": ("cm", 0.01),
        "centimeters": ("cm", 0.01),
        "mm": ("mm", 0.001),
        "millimeter": ("mm", 0.001),
        "millimeters": ("mm", 0.001),
    }
    return aliases.get(str(units).strip().lower(), (None, None))


class CalibrationBundle:
    """Calibration arrays plus the metadata required to interpret them."""

    def __init__(self, K_list, extrinsics, dist_coeffs, metadata):
        self.K_list = K_list
        self.extrinsics = extrinsics
        self.dist_coeffs = dist_coeffs
        self.metadata = metadata

    @property
    def camera_names(self) -> list[str]:
        return list(self.metadata.get("camera_names", []))


def _load_calibration_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or not isinstance(data.get("cameras"), list):
        raise ValueError("calibration JSON must contain a cameras list")
    return data


def _camera_name(cam: dict, index: int) -> str:
    return str(cam.get("name", cam.get("node", cam.get("id", index))))


def _canonical_camera_name(value: str) -> str:
    """Apply only the documented Panoptic ``hd_`` filename convention."""
    stem = Path(str(value)).stem
    return stem[3:] if stem.startswith("hd_") else stem


def _resolve_camera_names(
    calibration_names: list[str],
    requested_names: list[str] | None,
    camera_map: dict[str, str] | None,
) -> tuple[list[int], list[str]]:
    if requested_names is None:
        return list(range(len(calibration_names))), list(calibration_names)

    by_exact = {name: i for i, name in enumerate(calibration_names)}
    by_canonical: dict[str, list[int]] = {}
    for i, name in enumerate(calibration_names):
        by_canonical.setdefault(_canonical_camera_name(name), []).append(i)

    selected_indices: list[int] = []
    selected_names: list[str] = []
    for requested in requested_names:
        mapped = (camera_map or {}).get(str(requested), str(requested))
        if mapped in by_exact:
            index = by_exact[mapped]
        else:
            candidates = by_canonical.get(_canonical_camera_name(mapped), [])
            if len(candidates) != 1:
                raise ValueError(
                    f"camera {requested!r} has no unique calibration match; "
                    "provide --camera-map input=calibration explicitly"
                )
            index = candidates[0]
        if index in selected_indices:
            raise ValueError(f"camera mapping is not one-to-one: {requested!r}")
        selected_indices.append(index)
        selected_names.append(calibration_names[index])
    return selected_indices, selected_names


def _scale_intrinsics(
    K: np.ndarray,
    calibration_size: tuple[int, int] | None,
    observation_size: tuple[int, int] | None,
    camera_name: str,
) -> np.ndarray:
    if observation_size is None or calibration_size is None:
        return K
    cw, ch = map(float, calibration_size)
    ow, oh = map(float, observation_size)
    if min(cw, ch, ow, oh) <= 0:
        raise ValueError(f"invalid image resolution for camera {camera_name}")
    cal_ar = cw / ch
    obs_ar = ow / oh
    if not np.isclose(cal_ar, obs_ar, rtol=0.0, atol=1e-6):
        raise ValueError(
            f"camera {camera_name} changes aspect ratio from {calibration_size} "
            f"to {observation_size}; provide an explicit crop/resize transform"
        )
    out = K.copy()
    out[0, :] *= ow / cw
    out[1, :] *= oh / ch
    return out


def load_calibration_bundle(
    path: str | Path,
    camera_names: list[str] | None = None,
    scale_factor: float = 1.0,
    *,
    require_metadata: bool = False,
    camera_map: dict[str, str] | None = None,
    observation_image_sizes: list[tuple[int, int]] | np.ndarray | None = None,
) -> CalibrationBundle:
    """Load calibration, distortion, camera names, and geometry metadata."""
    data = _load_calibration_json(path)
    cameras = list(data["cameras"])
    names = [_camera_name(cam, i) for i, cam in enumerate(cameras)]
    selected_indices, selected_names = _resolve_camera_names(names, camera_names, camera_map)
    cameras = [cameras[i] for i in selected_indices]
    names = selected_names

    # Official Panoptic calibration files use RX+t, centimetre translations,
    # OpenCV distorted pixels, and per-camera ``resolution`` fields even
    # though they do not repeat AIMoCap's metadata keys.
    is_panoptic = "calibDataSource" in data and all("R" in cam and "t" in cam for cam in cameras)
    inferred = {
        "projection_convention": "world_to_camera" if is_panoptic else None,
        "camera_coordinate_system": OPENCV_COORDINATE_SYSTEM if is_panoptic else None,
        "world_coordinate_system": OPENCV_COORDINATE_SYSTEM if is_panoptic else None,
        "units": "cm" if is_panoptic else None,
        "distortion_treatment": "distorted_pixels" if is_panoptic else None,
        "accuracy_class": "production" if is_panoptic else None,
        "calibration_source": "cmu_panoptic" if is_panoptic else None,
        "scale_source": "panoptic_studio" if is_panoptic else None,
    }
    required = (
        "projection_convention", "camera_coordinate_system", "world_coordinate_system",
        "units", "distortion_treatment",
    )
    missing = [key for key in required if not data.get(key) and not inferred.get(key)]
    if require_metadata and missing:
        raise ValueError(f"calibration metadata missing required fields: {', '.join(missing)}")
    if require_metadata:
        missing_resolution = [
            names[i] for i, cam in enumerate(cameras)
            if cam.get("resolution", cam.get("image_size")) is None
        ]
        if missing_resolution:
            raise ValueError(
                "calibration resolution missing for camera(s): "
                + ", ".join(missing_resolution)
            )

    source_units = data.get("units") or inferred["units"]
    canonical_source_units, unit_scale_to_meters = _unit_scale_to_meters(source_units)
    if require_metadata and unit_scale_to_meters is None:
        raise ValueError(f"unsupported or unknown calibration units: {source_units!r}")

    K_list = []
    extrinsics = []
    dist_coeffs = []
    effective_sizes = []
    for i, cam in enumerate(cameras):
        K = np.asarray(cam["K"], dtype=np.float64)
        calibration_size = cam.get("resolution", cam.get("image_size"))
        calibration_size = tuple(map(int, calibration_size)) if calibration_size is not None else None
        observed_size = None
        if observation_image_sizes is not None:
            observed_size = tuple(map(int, observation_image_sizes[i]))
        K = _scale_intrinsics(K, calibration_size, observed_size, names[i])
        R = np.asarray(cam["R"], dtype=np.float64)
        t = np.asarray(cam["t"], dtype=np.float64).reshape(3, 1)
        if unit_scale_to_meters is not None:
            t = t * unit_scale_to_meters
        t = t * scale_factor
        validate_rotation(R)
        if K.shape != (3, 3) or not np.isfinite(K).all():
            raise ValueError(f"invalid K for camera {names[i]}")
        dist = np.asarray(cam.get("dist", cam.get("distCoef", np.zeros(5))), dtype=np.float64).reshape(-1)
        if dist.size not in (4, 5, 8, 12, 14):
            raise ValueError(f"unsupported distortion coefficient count for camera {names[i]}: {dist.size}")
        K_list.append(K)
        extrinsics.append((R, t))
        dist_coeffs.append(dist)
        effective_sizes.append(observed_size or calibration_size)

    metadata = dict(data)
    metadata["camera_names"] = names
    metadata["projection_convention"] = data.get("projection_convention") or inferred["projection_convention"] or "unknown"
    metadata["camera_coordinate_system"] = data.get("camera_coordinate_system") or inferred["camera_coordinate_system"] or "unknown"
    metadata["world_coordinate_system"] = data.get("world_coordinate_system") or inferred["world_coordinate_system"] or "unknown"
    metadata["source_units"] = canonical_source_units or str(source_units or "unknown")
    metadata["units"] = "meters" if unit_scale_to_meters is not None else "unknown"
    metadata["geometry_units"] = metadata["units"]
    metadata["unit_scale_to_meters"] = unit_scale_to_meters
    metadata["accuracy_class"] = data.get("accuracy_class") or inferred["accuracy_class"] or "unknown"
    metadata["calibration_source"] = data.get("calibration_source") or inferred["calibration_source"] or "unknown"
    metadata["scale_source"] = data.get("scale_source") or inferred["scale_source"] or ("checkerboard" if metadata["calibration_source"] == "checkerboard" else "unknown")
    metadata["distortion_model"] = data.get("distortion_model", "opencv_radtan")
    metadata["distortion_treatment"] = data.get("distortion_treatment") or inferred["distortion_treatment"] or "unknown"
    metadata["effective_image_sizes"] = effective_sizes
    metadata["resolved_camera_map"] = dict(zip(camera_names or names, names))
    metadata["is_panoptic"] = is_panoptic
    return CalibrationBundle(K_list, extrinsics, dist_coeffs, metadata)


def save_calibration(
    path: str | Path,
    K_list: list[np.ndarray],
    extrinsics: list[tuple[np.ndarray, np.ndarray]],
    image_sizes: list[tuple[int, int]] | None = None,
    dist_coeffs: list[np.ndarray] | None = None,
    camera_names: list[str] | None = None,
    *,
    units: str = "unknown",
    calibration_source: str = "unknown",
    accuracy_class: str = "unknown",
    scale_source: str | None = None,
    projection_convention: str = CALIBRATION_PROJECTION_CONVENTION,
    coordinate_system: str = OPENCV_COORDINATE_SYSTEM,
    distortion_model: str = "opencv_radtan",
    distortion_treatment: str = "distorted_pixels",
    provenance_manifest: dict | None = None,
) -> None:
    """
    Save N cameras to a JSON file.
    
    Args:
        path: Output path.
        K_list: N (3,3) intrinsic matrices.
        extrinsics: N tuples of (3,3) Rotation, (3,1) translation.
        image_sizes: N tuples of (width, height).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if camera_names is not None and len(camera_names) != len(K_list):
        raise ValueError("camera_names length must match K_list")
    if dist_coeffs is not None and len(dist_coeffs) != len(K_list):
        raise ValueError("dist_coeffs length must match K_list")

    if scale_source is None:
        if calibration_source in ("checkerboard", "charuco"):
            scale_source = calibration_source
        elif accuracy_class == "production":
            scale_source = "metric_known"
        elif accuracy_class == "exploratory" or units == "arbitrary":
            scale_source = "unscaled"
        else:
            scale_source = "unknown"

    cameras = []
    for i in range(len(K_list)):
        validate_rotation(extrinsics[i][0])
        dist = np.zeros(5, dtype=np.float64) if dist_coeffs is None else np.asarray(dist_coeffs[i], dtype=np.float64)
        name = camera_names[i] if camera_names is not None else f"cam_{i:02d}"
        cam_data = {
            "id": i,
            "name": name,
            "K": K_list[i].tolist(),
            "R": extrinsics[i][0].tolist(),
            "t": extrinsics[i][1].flatten().tolist(),
            "dist": dist.tolist(),
            "distCoef": dist.tolist(),
        }
        if image_sizes is not None and i < len(image_sizes):
            cam_data["image_size"] = list(image_sizes[i])
            cam_data["resolution"] = list(image_sizes[i])
            
        cameras.append(cam_data)
        
    data = {
        "projection_convention": projection_convention,
        "camera_coordinate_system": coordinate_system,
        "world_coordinate_system": coordinate_system,
        "coordinate_system": coordinate_system,
        "units": units,
        "distortion_model": distortion_model,
        "distortion_treatment": distortion_treatment,
        "calibration_source": calibration_source,
        "accuracy_class": accuracy_class,
        "scale_source": scale_source,
        "camera_names": [cam["name"] for cam in cameras],
        "cameras": cameras
    }
    if provenance_manifest is not None:
        data["provenance_manifest"] = provenance_manifest
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_calibration(
    path: str | Path,
    camera_names: list[str] | None = None,
    scale_factor: float = 1.0
) -> tuple[list[np.ndarray], list[tuple[np.ndarray, np.ndarray]]]:
    """
    Load calibration data from JSON.
    
    Args:
        path: Path to the calibration JSON file.
        camera_names: Optional list of camera names to load in that exact order.
                      If provided, matches the camera's JSON name against the elements.
        scale_factor: Additional multiplier applied after the loader converts
                      declared source units to metres.
    
    Returns:
        K_list: N (3,3) intrinsic matrices.
        extrinsics: N tuples of (3,3) Rotation, (3,1) translation.
    """
    bundle = load_calibration_bundle(path, camera_names=camera_names, scale_factor=scale_factor)
    return bundle.K_list, bundle.extrinsics

