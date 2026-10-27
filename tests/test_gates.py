import numpy as np
import pytest
from pathlib import Path
import json
from aimocap.pipeline.runner import run_evidence_gated_triangulation
from aimocap.pipeline.gates import GateFailureReport

CALIB_FILE = Path(__file__).parent.parent / "data" / "panoptic" / "171204_pose1" / "calibration_171204_pose1.json"
pytestmark = pytest.mark.skipif(not CALIB_FILE.exists(), reason="Panoptic calibration not present")

# We need a dummy calibration to run the tests.
# Since we generated fixtures from actual `00_11, 00_12, 00_23`, we should load that calibration.
def get_panoptic_calibration():
    # Load calibration 
    base_dir = Path(__file__).parent.parent
    calib_file = base_dir / "data" / "panoptic" / "171204_pose1" / "calibration_171204_pose1.json"
    with open(calib_file, "r") as f:
        calib_data = json.load(f)
        
    K_list = []
    extrinsics = []
    camera_names = ["00_11", "00_12", "00_23"]
    
    # The JSON structure is {"cameras": [{"name": "00_11", "K": ...}, ...]}
    cameras_list = calib_data["cameras"]
    calib_dict = {cam["name"]: cam for cam in cameras_list}
    
    for c_name in camera_names:
        c_data = calib_dict[c_name]
        K = np.array(c_data['K'], dtype=np.float64)
        R = np.array(c_data['R'], dtype=np.float64)
        t = np.array(c_data['t'], dtype=np.float64).reshape(3, 1) # Must be (3, 1)
        # Panoptic is defined as [R | t].
        K_list.append(K)
        extrinsics.append((R, t))
        
    return K_list, extrinsics, camera_names

def load_fixture(name: str):
    p = Path(__file__).parent / "fixtures" / name
    data = np.load(p)
    return data['kpts'], data['scores']

def test_corrupted_obs_fails_before_triangulation(tmp_path):
    K_list, extrinsics, camera_names = get_panoptic_calibration()
    kpts, scores = load_fixture("corrupted_obs.npz")
    
    # Transpose kpts for runner which expects (F, C, J, 2)
    # Wait, runner expects (F, C, J, 2). The fixture is (F, C, J, 2)
    
    report = run_evidence_gated_triangulation(
        keypoints=kpts,
        scores=scores,
        K_list=K_list,
        extrinsics=extrinsics,
        calib_hash="dummy",
        provenance_hash="dummy",
        camera_names=camera_names,
        output_dir=tmp_path,
        soft_fail=True
    )
    
    assert isinstance(report, GateFailureReport)
    assert report.failing_stage == "2d_observation"
    # Should fail either coords_in_image or epipolar
    assert "coords_in_image" in report.failing_gate or "epipolar" in report.failing_gate
    assert not (tmp_path / "triangulated.npz").exists()


def test_valid_obs_passes_all_gates(tmp_path):
    K_list, extrinsics, camera_names = get_panoptic_calibration()
    kpts, scores = load_fixture("valid_obs_f0_f59.npz")
    
    result = run_evidence_gated_triangulation(
        keypoints=kpts,
        scores=scores,
        K_list=K_list,
        extrinsics=extrinsics,
        calib_hash="dummy",
        provenance_hash="dummy",
        camera_names=camera_names,
        output_dir=tmp_path,
        soft_fail=True
    )
    
    assert not isinstance(result, GateFailureReport), f"Failed with {result.summary() if isinstance(result, GateFailureReport) else ''}"
    assert (tmp_path / "triangulated.npz").exists()
    assert (tmp_path / "triangulated_evidence.json").exists()


def test_synthetic_translation_is_caught(tmp_path):
    K_list, extrinsics, camera_names = get_panoptic_calibration()
    kpts, scores = load_fixture("synthetic_translation.npz")
    
    report = run_evidence_gated_triangulation(
        keypoints=kpts,
        scores=scores,
        K_list=K_list,
        extrinsics=extrinsics,
        calib_hash="dummy",
        provenance_hash="dummy",
        camera_names=camera_names,
        output_dir=tmp_path,
        soft_fail=True
    )
    
    assert isinstance(report, GateFailureReport)
    assert report.failing_stage == "2d_observation"
    assert "epipolar" in report.failing_gate or "coords_in_image" in report.failing_gate
    
def test_synthetic_axis_swap_is_caught(tmp_path):
    K_list, extrinsics, camera_names = get_panoptic_calibration()
    kpts, scores = load_fixture("synthetic_axis_swap.npz")
    
    report = run_evidence_gated_triangulation(
        keypoints=kpts,
        scores=scores,
        K_list=K_list,
        extrinsics=extrinsics,
        calib_hash="dummy",
        provenance_hash="dummy",
        camera_names=camera_names,
        output_dir=tmp_path,
        soft_fail=True
    )
    
    assert isinstance(report, GateFailureReport)
    assert report.failing_stage == "2d_observation"
    assert "epipolar" in report.failing_gate or "coords_in_image" in report.failing_gate
    
def test_synthetic_time_shift_is_caught(tmp_path):
    K_list, extrinsics, camera_names = get_panoptic_calibration()
    kpts, scores = load_fixture("synthetic_time_shift.npz")
    
    report = run_evidence_gated_triangulation(
        keypoints=kpts,
        scores=scores,
        K_list=K_list,
        extrinsics=extrinsics,
        calib_hash="dummy",
        provenance_hash="dummy",
        camera_names=camera_names,
        output_dir=tmp_path,
        soft_fail=True
    )
    
    assert isinstance(report, GateFailureReport)
    assert report.failing_stage == "2d_observation"
    assert "epipolar" in report.failing_gate or "coords_in_image" in report.failing_gate
