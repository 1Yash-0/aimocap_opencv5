"""Map raw pose-model keypoint scores to calibrated probabilities.

A calibrated score p means: "probability that this 2D keypoint is within
``error_threshold_px`` (20 px) of the true position". It lets downstream
thresholds (tracker visibility, triangulation min confidence) keep the same
meaning across pose models whose raw score scales differ (CIGPose ~0..1.5,
RTMW ~3..7).

The RTMW-x table was fitted on AIST++ (gBR_sFM_cAll_d04_mBR5_ch06, cams
c02/c05/c08) and validated on Panoptic 171204_pose1 (ECE 0.095); see
outputs/phase1_2d/score_calib_20261017/REPORT.md. Feet, face and hand
keypoints have no ground truth in those datasets and reuse the ankle, head
and wrist calibration respectively (an explicit, documented assumption).

Recommended triangulation cutoff for calibrated scores is p >= 0.9
(removes ~36% of bad points, loses ~2% of good points). It is NOT applied
automatically; triangulation settings are unchanged.

When the opt-in ``part_refine`` pass is enabled, refined hand/face points are
mapped by ``calibrate_refined_part_scores`` using
``calibration/part_refine_score_calibration.json``.  That table is fitted on
refined clip1 (Panoptic 171204_pose1) and tested only on clip2
(171026_pose3); see outputs/phase1b/s2_calibration_20261020/REPORT.md.

S2c replaces the refined-hand score with cross-model agreement:
``d = ||final H4 point - RTMW body point|| / hand_size_px`` and
``p = f(d)`` from the monotone ``hand_agreement`` section of the same JSON.
The older hand isotonic groups stay in the file for reference.  Points that
fell back to RTMW (``hand_source == 0``) keep their body-calibrated score.
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

_CALIB_DIR = Path(__file__).resolve().parent / "calibration"
RTMW_X_CALIBRATION = _CALIB_DIR / "rtmw_x_score_calibration.json"

PART_REFINE_CALIBRATION = _CALIB_DIR / "part_refine_score_calibration.json"

# Non-body keypoints reuse the nearest body group (documented assumption).
_NON_BODY_GROUPS = ((range(17, 23), "ankles"), (range(23, 91), "head"), (range(91, 133), "wrists"))


def _load_table(path: str | Path) -> dict:
    path = Path(path)
    table = json.loads(path.read_text(encoding="utf-8"))
    table["_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    table["_file"] = path.name
    return table


@lru_cache(maxsize=4)
def load_calibration(path: str | Path = RTMW_X_CALIBRATION) -> dict:
    return _load_table(path)


@lru_cache(maxsize=4)
def load_part_calibration(path: str | Path = PART_REFINE_CALIBRATION) -> dict:
    return _load_table(path)


def _predict(score: np.ndarray, model: dict) -> np.ndarray:
    params = model["parameters"]
    if model["selected_method"] == "isotonic":
        bx, by = np.asarray(params["x"], float), np.asarray(params["y"], float)
        p = np.interp(score, bx, by, left=by[0], right=by[-1])
    else:
        z = (score - params["mean"]) / params["std"]
        p = 1.0 / (1.0 + np.exp(-np.clip(params["intercept"] + params["slope"] * z, -40.0, 40.0)))
    return np.clip(p, 0.0, 1.0)


def _group_of_index(table: dict) -> list[str]:
    groups = ["all"] * 133
    for k, g in table["body_joint_index_to_group"].items():
        groups[int(k)] = g
    for idx, g in _NON_BODY_GROUPS:
        for i in idx:
            groups[i] = g
    return groups


def calibrate_scores(raw_scores: np.ndarray, table: dict | None = None) -> np.ndarray:
    """Calibrate raw scores of shape (..., 133). NaN stays NaN. Deterministic."""
    table = load_calibration() if table is None else table
    raw = np.asarray(raw_scores, dtype=np.float64)
    if raw.shape[-1] != 133:
        raise ValueError(f"expected last dimension 133, got {raw.shape}")
    out = np.full(raw.shape, np.nan, dtype=np.float64)
    groups = _group_of_index(table)
    for g in set(groups):
        cols = [i for i, x in enumerate(groups) if x == g]
        vals = raw[..., cols]
        finite = np.isfinite(vals)
        res = np.full(vals.shape, np.nan)
        res[finite] = _predict(vals[finite], table["groups"][g])
        out[..., cols] = res
    return out


def _part_group_columns(table: dict) -> dict[str, dict[str, list[int]]]:
    groups = {"hand": {}, "face": {}}
    for index, group in table["part_joint_index_to_group"].items():
        part = "hand" if group.startswith("hand_") else "face"
        groups[part].setdefault(group, []).append(int(index))
    return groups


def calibrate_refined_part_scores(
    scores: np.ndarray,
    *,
    hand_scores_raw: np.ndarray | None = None,
    hand_source: np.ndarray | None = None,
    face_scores_raw: np.ndarray | None = None,
    face_source: np.ndarray | None = None,
    table: dict | None = None,
) -> np.ndarray:
    """Replace refined hand/face scores with calibrated probabilities.

    Only points with ``hand_source == 1`` or ``face_source == 1`` and a finite
    model-native score are changed; every other entry keeps its body-calibrated
    value.  The raw model scores stay in ``hand_scores_raw``/``face_scores_raw``.
    NaN inputs never raise and never invent a probability.
    """
    table = load_part_calibration() if table is None else table
    out = np.array(scores, dtype=np.float64, copy=True)
    if out.ndim < 1 or out.shape[-1] != 133:
        raise ValueError(f"expected last dimension 133, got {out.shape}")
    for part, raw, source in (
        ("hand", hand_scores_raw, hand_source),
        ("face", face_scores_raw, face_source),
    ):
        if raw is None and source is None:
            continue
        if raw is None or source is None:
            raise ValueError(f"{part} calibration needs both raw scores and source mask")
        raw = np.asarray(raw, dtype=np.float64)
        source = np.asarray(source)
        if raw.shape != out.shape or source.shape != out.shape:
            raise ValueError(
                f"{part} shapes must match scores {out.shape}, got {raw.shape} and {source.shape}"
            )
        for group, cols in _part_group_columns(table)[part].items():
            cols = np.asarray(cols, dtype=int)
            values = raw[..., cols]
            mask = (source[..., cols] == 1) & np.isfinite(values)
            if not mask.any():
                continue
            current = out[..., cols]
            current[mask] = _predict(values[mask], table["groups"][group])
            out[..., cols] = current
    return out


def hand_agreement_probability(
    d: np.ndarray,
    table: dict | None = None,
) -> np.ndarray:
    """Map cross-model hand disagreement ``d`` to a good-point probability.

    ``d`` is non-negative; the fitted map is monotone non-increasing.  NaN
    stays NaN.  The parameters live in the ``hand_agreement`` section of
    ``part_refine_score_calibration.json``; no threshold or scale is
    hard-coded here.
    """
    table = load_part_calibration() if table is None else table
    section = table.get("hand_agreement")
    if not section:
        raise KeyError("part calibration table has no 'hand_agreement' section")
    method = section["selected_method"]
    parameters = section["parameters"]
    values = np.asarray(d, dtype=np.float64)
    out = np.full(values.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(values)
    if not finite.any():
        return out
    eps = float(parameters["eps"])
    log_d = np.log(np.maximum(values[finite], 0.0) + eps)
    if method == "logistic":
        z = (log_d - parameters["mean"]) / parameters["std"]
        probability = 1.0 / (
            1.0 + np.exp(-np.clip(parameters["intercept"] + parameters["slope"] * z, -40.0, 40.0))
        )
    elif method == "isotonic":
        probability = np.interp(
            -log_d,
            np.asarray(parameters["x"], dtype=np.float64),
            np.asarray(parameters["y"], dtype=np.float64),
            left=float(np.asarray(parameters["y"], dtype=np.float64)[0]),
            right=float(np.asarray(parameters["y"], dtype=np.float64)[-1]),
        )
    else:
        raise ValueError(f"unknown hand_agreement method: {method!r}")
    out[finite] = np.clip(probability, 0.0, 1.0)
    return out


def calibrate_hand_agreement_scores(
    scores: np.ndarray,
    hand_agreement_d: np.ndarray,
    hand_source: np.ndarray,
    *,
    table: dict | None = None,
) -> np.ndarray:
    """Replace refined-hand scores with ``f(d)`` and keep RTMW fallbacks.

    For ``hand_source == 1`` and finite ``d`` the score becomes
    ``hand_agreement_probability(d)``.  Every other entry (including
    ``hand_source == 0``) keeps its incoming body-calibrated score, so points
    that fell back to RTMW never inherit the hand model's top trust.  Shape
    is ``(..., 133)``; non-hand indices are untouched.
    """
    table = load_part_calibration() if table is None else table
    out = np.array(scores, dtype=np.float64, copy=True)
    distances = np.asarray(hand_agreement_d, dtype=np.float64)
    source = np.asarray(hand_source)
    if out.ndim < 1 or out.shape[-1] != 133:
        raise ValueError(f"expected last dimension 133, got {out.shape}")
    if distances.shape != out.shape or source.shape != out.shape:
        raise ValueError(
            f"hand agreement shapes must match scores {out.shape}, "
            f"got d={distances.shape} and source={source.shape}"
        )
    for base in (91, 112):
        selection = slice(base, base + 21)
        values = distances[..., selection]
        mask = (source[..., selection] == 1) & np.isfinite(values)
        if not mask.any():
            continue
        current = out[..., selection]
        current[mask] = hand_agreement_probability(values[mask], table=table)
        out[..., selection] = current
    return out


__all__ = [
    "PART_REFINE_CALIBRATION",
    "RTMW_X_CALIBRATION",
    "calibrate_hand_agreement_scores",
    "calibrate_refined_part_scores",
    "calibrate_scores",
    "hand_agreement_probability",
    "load_calibration",
    "load_part_calibration",
]
