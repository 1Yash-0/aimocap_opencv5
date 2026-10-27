import json
import numpy as np
from dataclasses import dataclass, asdict
from typing import Literal, Optional, Any
from pathlib import Path


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        if isinstance(obj, (np.float16, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.bool_)):
            return bool(obj)
        return super().default(obj)


@dataclass
class EvidenceContract:
    stage: str                   # "2d_observation" | "triangulation" | "3d_geometry"
    status: Literal["PASS", "FAIL", "UNSAFE_OVERRIDE"]
    validity_mask: np.ndarray    # bool mask, usually (F, J) or (F, J, C)
    uncertainty: np.ndarray      # per-joint confidence / reprojection px
    calibration_id: str          # sha256 of calibration file
    coordinate_space: str        # "image_px_1920x1080" | "opencv_cm" | "yup_cm"
    input_provenance: str        # sha256 of input artifact
    gate_results: dict[str, Any] # {gate_name: {passed, value, threshold}}
    coverage_fraction: float     # finite valid joints / total possible
    failure_reason: Optional[str] = None # first gate that failed

    def save_sidecar(self, target_file: Path | str):
        """Save this evidence contract as a JSON sidecar to the given target file."""
        target = Path(target_file)
        sidecar = target.parent / f"{target.stem}_evidence.json"
        
        data = asdict(self)
        with open(sidecar, "w") as f:
            json.dump(data, f, cls=NumpyEncoder, indent=2)

    @classmethod
    def load_sidecar(cls, target_file: Path | str) -> "EvidenceContract":
        target = Path(target_file)
        sidecar = target.parent / f"{target.stem}_evidence.json"
        
        with open(sidecar, "r") as f:
            data = json.load(f)
            
        data['validity_mask'] = np.array(data['validity_mask'], dtype=bool)
        data['uncertainty'] = np.array(data['uncertainty'], dtype=np.float32)
        
        return cls(**data)


@dataclass
class GateFailureReport:
    """Soft failure report returned instead of a processed artifact when a pipeline gate fails."""
    failing_stage: str
    failing_gate: str
    cameras_implicated: list[str]
    failure_mode: str
    coverage_fraction: float
    evidence: EvidenceContract

    def summary(self) -> str:
        return (
            f"PIPELINE GATE FAILED at {self.failing_stage} stage.\n"
            f"Gate '{self.failing_gate}' failed ({self.failure_mode}).\n"
            f"Cameras implicated: {self.cameras_implicated}\n"
            f"Valid coverage: {self.coverage_fraction:.1%}"
        )

