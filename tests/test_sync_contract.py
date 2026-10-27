"""Regression tests for the strict capture synchronization contract."""

from __future__ import annotations

import numpy as np
import pytest

from aimocap.pipeline.sync_contract import SyncContractError, validate_sync_contract


def test_sync_contract_accepts_finite_aligned_timestamps():
    source = np.array([[0.0, 0.1], [1.0, 1.1]], dtype=np.float64)
    requested = source + 0.005

    result = validate_sync_contract(
        "VERIFIED",
        source,
        requested,
        fps=30.0,
        max_error_s=0.02,
    )

    assert result["status"] == "VERIFIED"
    assert result["max_abs_error_s"] == pytest.approx(0.005)
    assert result["num_samples"] == 4


def test_sync_contract_rejects_unverified_or_missing_timing():
    with pytest.raises(SyncContractError, match="sync status must be one of"):
        validate_sync_contract(
            "UNVERIFIED_EQUAL_FRAME_INDEX",
            np.zeros((2, 2)),
            np.zeros((2, 2)),
            fps=30.0,
        )

    with pytest.raises(SyncContractError, match="both source_timestamps"):
        validate_sync_contract("VERIFIED", None, np.zeros((2, 2)), fps=30.0)


def test_sync_contract_rejects_decode_timestamp_error():
    source = np.array([[0.0, 0.2]], dtype=np.float64)
    requested = np.array([[0.0, 0.2 + 0.05]], dtype=np.float64)

    with pytest.raises(SyncContractError, match="timestamp error"):
        validate_sync_contract(
            "VERIFIED",
            source,
            requested,
            fps=30.0,
            max_error_s=0.02,
        )
