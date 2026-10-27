"""Tests for the dataset-verified Panoptic frame-sync contract."""

from __future__ import annotations

import numpy as np
import pytest

from aimocap.pipeline.panoptic_sync import (
    PANOPTIC_FRAME_SYNC_STATUS,
    PanopticSyncError,
    build_panoptic_sync_manifest,
    is_panoptic_sync,
    normalize_camera_id,
)
from aimocap.pipeline.sync_contract import (
    SyncContractError,
    validate_sync_contract,
)


def test_normalize_camera_id_accepts_hd_forms():
    assert normalize_camera_id("00_26") == "00_26"
    assert normalize_camera_id("hd_00_26") == "00_26"
    assert normalize_camera_id("hd_00_26.mp4") == "00_26"
    with pytest.raises(PanopticSyncError):
        normalize_camera_id("garbage")
    with pytest.raises(PanopticSyncError):
        normalize_camera_id("01_26")  # VGA panel not supported for frame sync


def test_manifest_status_is_distinct_and_reference_relative(tmp_path):
    (tmp_path / "hdPose3d_stage1_coco19").mkdir()
    manifest = build_panoptic_sync_manifest(
        tmp_path,
        "171204_pose1",
        29.97002997002997,
        [("hd_00_01.mp4", "v1.mp4"), ("hd_00_26.mp4", "v26.mp4")],
        reference_camera="00_01",
        verify_gt=False,
    )
    assert manifest["status"] == PANOPTIC_FRAME_SYNC_STATUS
    assert manifest["reference_id"] == "00_01"
    # Reference camera must carry a zero offset relative to itself.
    assert manifest["cameras"]["00_01"]["offset_sec"] == 0.0
    assert "offset_sec" in manifest["cameras"]["00_26"]
    assert manifest["source_paths"]["00_26"] == "v26.mp4"
    assert is_panoptic_sync(manifest)
    assert not is_panoptic_sync({"status": "VERIFIED_AUDIO_CLAP"})


def test_manifest_trimmed_camera_gets_correct_offset(tmp_path):
    (tmp_path / "hdPose3d_stage1_coco19").mkdir()
    fps = 30.0
    manifest = build_panoptic_sync_manifest(
        tmp_path, "171204_pose1", fps,
        [("00_01", "full.mp4", 5), ("00_02", "full2.mp4", 81), ("00_30", "clip.mp4", 23312)],
        reference_camera="00_01", verify_gt=False,
    )
    # Reference (00_01, global start 5) is at offset 0 relative to itself.
    assert manifest["cameras"]["00_01"]["offset_sec"] == 0.0
    # 00_30 starts at global 23312; relative to reference global start 5, its local
    # timeline leads the reference by (23312-5)=23307 frames -> positive frame offset.
    assert manifest["cameras"]["00_30"]["frame_offset"] == pytest.approx(5.0 - 23312.0)
    assert manifest["cameras"]["00_30"]["offset_sec"] == pytest.approx((5.0 - 23312.0) / fps)
    assert manifest["cameras"]["00_02"]["frame_offset"] == pytest.approx(5.0 - 81.0)


def test_manifest_rejects_negative_start(tmp_path):
    (tmp_path / "hdPose3d_stage1_coco19").mkdir()
    with pytest.raises(PanopticSyncError):
        build_panoptic_sync_manifest(
            tmp_path, "seq", 30.0, [("00_01", "a.mp4", -1)], verify_gt=False
        )


def test_manifest_bad_tuple_arity(tmp_path):
    (tmp_path / "hdPose3d_stage1_coco19").mkdir()
    with pytest.raises(PanopticSyncError):
        build_panoptic_sync_manifest(
            tmp_path, "seq", 30.0, [("00_01", "a.mp4", 0, "extra")], verify_gt=False
        )


def test_manifest_requires_gt_dir(tmp_path):
    with pytest.raises(PanopticSyncError):
        build_panoptic_sync_manifest(
            tmp_path, "seq", 30.0, [("00_01", "a.mp4")], verify_gt=True
        )


def test_manifest_duplicate_cameras_rejected(tmp_path):
    (tmp_path / "hdPose3d_stage1_coco19").mkdir()
    with pytest.raises(PanopticSyncError, match="duplicate"):
        build_panoptic_sync_manifest(
            tmp_path, "seq", 30.0,
            [("00_01", "a.mp4"), ("hd_00_01.mp4", "b.mp4")],
            verify_gt=False,
        )


def test_gt_frame_probe_flags_missing(tmp_path):
    gt_dir = tmp_path / "hdPose3d_stage1_coco19"
    gt_dir.mkdir()
    (gt_dir / "body3DScene_00000000.json").write_text("{}")
    manifest = build_panoptic_sync_manifest(
        tmp_path, "seq", 30.0, [("00_01", "a.mp4")],
        verify_gt=True, probe_frames=[0, 5],
    )
    probe = manifest["provenance"]["gt_frame_probe"]
    assert probe["num_present"] == 1
    assert probe["missing"] == [5]
    assert probe["all_present"] is False


def test_panoptic_status_satisfies_sync_contract_validator():
    # A Panoptic frame-sync artifact decodes the requested global timeline
    # exactly; the timestamp contract must accept the Panoptic status.
    source = np.array([[0.0, 0.0], [1.0 / 30, 1.0 / 30]], dtype=np.float64)
    requested = source.copy()
    result = validate_sync_contract(
        PANOPTIC_FRAME_SYNC_STATUS, source, requested, fps=30.0
    )
    assert result["status"] == "VERIFIED"


def test_unverified_status_still_rejected():
    with pytest.raises(SyncContractError):
        validate_sync_contract(
            "UNVERIFIED_EQUAL_FRAME_INDEX",
            np.zeros((2, 2)), np.zeros((2, 2)), fps=30.0,
        )
