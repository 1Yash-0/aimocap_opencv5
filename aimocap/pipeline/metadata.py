from dataclasses import dataclass, field
import json
from pathlib import Path


@dataclass
class SequenceMetadata:
    """Explicit tracking of the geometric and semantic provenance of a sequence.
    
    This fulfills the input contract requirement: "Never let Stage 6b consume an array 
    whose coordinate space, units, or provenance are implicit."
    """
    # Raw source
    sequence_id: str = ""
    framerate: float = 30.0
    num_frames: int = 0
    camera_names: list[str] = field(default_factory=list)
    resolution: list[int] = field(default_factory=list)  # [width, height]
    
    # Calibration hashes and units
    calibration_hash: str = ""
    calibration_source: str = ""
    metric_scale: float = 1.0  # Must always be 1.0 (meters) inside the pipeline
    original_calibration_units: str = "unknown"  # e.g., "cm", "mm", "m"
    
    # 2D Detection
    detector_version: str = ""
    pose_model_version: str = ""
    joint_convention: str = "coco-wholebody"
    coordinate_space: str = "unknown"
    projection_convention: str = "world_to_camera"
    camera_coordinate_system: str = "unknown"
    world_coordinate_system: str = "unknown"
    units: str = "unknown"
    distortion_treatment: str = "unknown"
    sync_status: str = "unknown"
    provenance_manifest: str = ""
    
    # Provenance
    input_video_hashes: dict[str, str] = field(default_factory=dict)
    
    def save(self, path: Path | str) -> None:
        path = Path(path)
        with open(path, "w") as f:
            # dataclass to dict natively or manually
            data = {
                "sequence_id": self.sequence_id,
                "framerate": self.framerate,
                "num_frames": self.num_frames,
                "camera_names": self.camera_names,
                "resolution": self.resolution,
                "calibration_hash": self.calibration_hash,
                "calibration_source": self.calibration_source,
                "metric_scale": self.metric_scale,
                "original_calibration_units": self.original_calibration_units,
                "detector_version": self.detector_version,
                "pose_model_version": self.pose_model_version,
                "joint_convention": self.joint_convention,
                "coordinate_space": self.coordinate_space,
                "projection_convention": self.projection_convention,
                "camera_coordinate_system": self.camera_coordinate_system,
                "world_coordinate_system": self.world_coordinate_system,
                "units": self.units,
                "distortion_treatment": self.distortion_treatment,
                "sync_status": self.sync_status,
                "provenance_manifest": self.provenance_manifest,
                "input_video_hashes": self.input_video_hashes
            }
            json.dump(data, f, indent=2)
            
    @classmethod
    def load(cls, path: Path | str) -> "SequenceMetadata":
        path = Path(path)
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)
