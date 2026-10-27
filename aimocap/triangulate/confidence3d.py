"""Per-point 3D confidence from runtime observability signals (P3.6).

The confidence is the probability that a triangulated joint is "good", where
good means a 3D error below 30 mm for body joints and 15 mm for hand joints.
It is computed from calibration/runtime signals only (no ground truth):

* effective number of trusted cameras (sum of the P3.3 soft camera trust),
* robust reprojection residual in pixels, normalised by the observed 2D body
  height so that a person at a different image scale is treated fairly,
* maximum ray angle between trusted cameras,
* magnitude of the P3.5 smoother correction (smoother innovation),
* mean calibrated 2D score over the trusted cameras.

The model is a sign-constrained logistic regression.  The coefficients are
stored in ``confidence3d.json`` next to this module.  The sign constraints
make the output monotone: more trusted cameras / wider ray angle / higher 2D
score can only increase the confidence, and a larger residual / smoother
innovation can only decrease it.  It was fitted once on clip1 (Panoptic
171204_pose1) and evaluated on held-out clip2 and clip3.

The module is deliberately independent of the triangulation engine so it can
also be used by offline evaluation and by phase-5 IK weights.
"""

from __future__ import annotations

import json
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import numpy as np

MODEL_VERSION = 1
MODEL_FILENAME = "confidence3d.json"
BODY_JOINTS = 17
HAND_START = 91
DEFAULT_RESIDUAL_PX = 30.0
DEFAULT_BODY_HEIGHT_PX = 200.0
MIN_BODY_HEIGHT_PX = 20.0

FEATURE_NAMES: tuple[str, ...] = (
    "log_trust_sum",
    "log_residual_rel",
    "ray_angle_frac",
    "log_smoother_innovation_mm",
    "mean_2d_score",
)

# "increasing" features can only raise the confidence, "decreasing" only lower
# it.  The fit enforces these directions with box bounds.
SIGN_CONSTRAINTS: dict[str, str] = {
    "log_trust_sum": "increasing",
    "log_residual_rel": "decreasing",
    "ray_angle_frac": "increasing",
    "log_smoother_innovation_mm": "decreasing",
    "mean_2d_score": "increasing",
}


def model_path() -> Path:
    """Path of the shipped fitted model."""
    return Path(__file__).with_name(MODEL_FILENAME)


def _nanmedian(values: np.ndarray, axis: int) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(values, axis=axis)


def _nanmean(values: np.ndarray, axis: int) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(values, axis=axis)


def _nanmax(values: np.ndarray, axis: int) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmax(values, axis=axis)


def _nanmin(values: np.ndarray, axis: int) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmin(values, axis=axis)


def _sigmoid(logit: np.ndarray) -> np.ndarray:
    logit = np.clip(np.asarray(logit, dtype=np.float64), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-logit))


def load_model(path: str | Path | None = None) -> dict[str, Any]:
    """Load and validate one fitted model JSON."""
    model_file = Path(path) if path is not None else model_path()
    payload = json.loads(model_file.read_text(encoding="utf-8"))
    if int(payload.get("version", -1)) != MODEL_VERSION:
        raise ValueError(
            f"unsupported confidence3d model version {payload.get('version')!r} "
            f"in {model_file}"
        )
    names = tuple(payload.get("feature_names", ()))
    if names != FEATURE_NAMES:
        raise ValueError(
            f"confidence3d feature names {names!r} do not match {FEATURE_NAMES!r}"
        )
    coefficients = np.asarray(payload.get("coefficients", []), dtype=np.float64)
    mean = np.asarray(payload.get("feature_mean", []), dtype=np.float64)
    scale = np.asarray(payload.get("feature_scale", []), dtype=np.float64)
    if coefficients.shape != (len(FEATURE_NAMES),):
        raise ValueError("confidence3d coefficients must match the feature count")
    if mean.shape != (len(FEATURE_NAMES),) or scale.shape != (len(FEATURE_NAMES),):
        raise ValueError("confidence3d feature_mean/feature_scale must match the feature count")
    if not np.isfinite(coefficients).all() or not np.isfinite(mean).all():
        raise ValueError("confidence3d model contains non-finite parameters")
    if not np.isfinite(scale).all() or np.any(scale <= 0.0):
        raise ValueError("confidence3d feature_scale must be finite and positive")
    for name, value in zip(FEATURE_NAMES, coefficients):
        if SIGN_CONSTRAINTS[name] == "increasing" and float(value) < 0.0:
            raise ValueError(f"confidence3d coefficient for {name} violates its sign constraint")
        if SIGN_CONSTRAINTS[name] == "decreasing" and float(value) > 0.0:
            raise ValueError(f"confidence3d coefficient for {name} violates its sign constraint")
    payload["coefficients_array"] = coefficients
    payload["feature_mean_array"] = mean
    payload["feature_scale_array"] = scale
    return payload


@lru_cache(maxsize=1)
def default_model() -> dict[str, Any]:
    """Cached model used by the triangulation engine."""
    return load_model()


def body_height_px(
    keypoints: np.ndarray, scores: np.ndarray, min_score: float = 0.2
) -> np.ndarray:
    """Per-frame 2D body-height proxy in pixels from the body keypoints.

    The proxy is the larger of the observed x and y extents of joints 0..16 in
    any camera (median over cameras).  It is only a scale normaliser for the
    reprojection residual, not a calibrated bounding box.
    """
    points = np.asarray(keypoints, dtype=np.float64)
    confidence = np.asarray(scores, dtype=np.float64)
    if points.ndim != 4 or points.shape[-1] != 2:
        raise ValueError("keypoints must have shape (frames, cameras, joints, 2)")
    if confidence.shape != points.shape[:3]:
        raise ValueError("scores must have shape (frames, cameras, joints)")
    frames, cameras, joints = confidence.shape
    if frames == 0 or cameras == 0 or joints == 0:
        return np.full(frames, DEFAULT_BODY_HEIGHT_PX, dtype=np.float64)
    body_joints = min(BODY_JOINTS, joints)
    body = points[:, :, :body_joints, :]
    valid = (confidence[:, :, :body_joints] >= float(min_score)) & np.isfinite(body).all(axis=-1)
    xs = np.where(valid, body[..., 0], np.nan)
    ys = np.where(valid, body[..., 1], np.nan)
    x_extent = _nanmax(xs, axis=2) - _nanmin(xs, axis=2)
    y_extent = _nanmax(ys, axis=2) - _nanmin(ys, axis=2)
    height = np.nanmax(np.stack([x_extent, y_extent], axis=-1), axis=-1)
    scale = _nanmedian(height, axis=1)
    bad = ~np.isfinite(scale) | (scale < MIN_BODY_HEIGHT_PX)
    return np.where(bad, DEFAULT_BODY_HEIGHT_PX, scale)


def build_features(
    camera_trust: np.ndarray,
    reprojection_error_px: np.ndarray,
    ray_angle_deg: np.ndarray,
    *,
    keypoints: np.ndarray | None = None,
    scores: np.ndarray | None = None,
    innovation_m: np.ndarray | None = None,
    points_finite: np.ndarray | None = None,
    default_residual_px: float = DEFAULT_RESIDUAL_PX,
) -> np.ndarray:
    """Build the ``(frames, joints, 5)`` confidence feature tensor.

    Parameters are the P3.3/P3.5 diagnostics in ``(frames, joints, cameras)``
    layout plus the original 2D keypoints/scores in ``(frames, cameras,
    joints, ...)`` layout.  ``innovation_m`` is ``(frames, joints)`` and holds
    the smoother correction magnitude in metres (NaN where not defined).
    ``points_finite`` optionally masks invalid 3D points; their features are
    replaced by neutral values and their confidence is forced to zero by
    :func:`predict_confidence3d`.
    """
    trust = np.asarray(camera_trust, dtype=np.float64)
    if trust.ndim != 3:
        raise ValueError("camera_trust must have shape (frames, joints, cameras)")
    frames, joints, cameras = trust.shape
    error = np.asarray(reprojection_error_px, dtype=np.float64)
    if error.shape != trust.shape:
        raise ValueError("reprojection_error_px must match camera_trust shape")
    ray = np.asarray(ray_angle_deg, dtype=np.float64)
    if ray.shape != (frames, joints):
        raise ValueError("ray_angle_deg must have shape (frames, joints)")

    finite_error = np.isfinite(error)
    trusted = (trust > 0.5) & finite_error
    residual = _nanmedian(np.where(trusted, error, np.nan), axis=2)
    fallback = _nanmedian(np.where(finite_error, error, np.nan), axis=2)
    residual = np.where(np.isfinite(residual), residual, fallback)
    residual = np.where(np.isfinite(residual), residual, float(default_residual_px))
    residual = np.maximum(residual, 0.0)

    if keypoints is not None and scores is not None:
        score_values = np.asarray(scores, dtype=np.float64)
        height = body_height_px(keypoints, score_values)
    else:
        score_values = None
        height = np.full(frames, DEFAULT_BODY_HEIGHT_PX, dtype=np.float64)
    height = np.maximum(height, MIN_BODY_HEIGHT_PX)
    residual_rel = residual / (height[:, None] / 100.0)

    if score_values is not None:
        if score_values.shape != (frames, cameras, joints):
            raise ValueError("scores must have shape (frames, cameras, joints)")
        score_2d = np.transpose(score_values, (0, 2, 1))
        finite_score = np.isfinite(score_2d)
        trusted_score = (trust > 0.5) & finite_score
        mean_score = _nanmean(np.where(trusted_score, score_2d, np.nan), axis=2)
        score_fallback = _nanmean(np.where(finite_score, score_2d, np.nan), axis=2)
        mean_score = np.where(np.isfinite(mean_score), mean_score, score_fallback)
    else:
        mean_score = np.zeros((frames, joints), dtype=np.float64)
    mean_score = np.clip(np.nan_to_num(mean_score, nan=0.0), 0.0, 1.0)

    trust_sum = np.sum(np.nan_to_num(trust, nan=0.0), axis=2)
    if innovation_m is None:
        innovation = np.zeros((frames, joints), dtype=np.float64)
    else:
        innovation = np.asarray(innovation_m, dtype=np.float64)
        if innovation.shape != (frames, joints):
            raise ValueError("innovation_m must have shape (frames, joints)")
        innovation = np.nan_to_num(innovation, nan=0.0, posinf=0.0, neginf=0.0)
        innovation = np.maximum(innovation, 0.0) * 1000.0

    features = np.stack(
        [
            np.log1p(trust_sum),
            np.log1p(residual_rel),
            np.clip(np.nan_to_num(ray, nan=0.0) / 90.0, 0.0, 1.0),
            np.log1p(innovation),
            mean_score,
        ],
        axis=-1,
    )
    if points_finite is not None:
        valid = np.asarray(points_finite, dtype=bool)
        if valid.shape != (frames, joints):
            raise ValueError("points_finite must have shape (frames, joints)")
        neutral = np.array(
            [0.0, np.log1p(float(default_residual_px)), 0.0, 0.0, 0.0],
            dtype=np.float64,
        )
        features = np.where(valid[..., None], features, neutral)
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


def predict_confidence3d(
    features: np.ndarray,
    model: Mapping[str, Any] | None = None,
    *,
    valid: np.ndarray | None = None,
    num_joints: int | None = None,
) -> np.ndarray:
    """Map features to a calibrated ``[0, 1]`` confidence.

    ``num_joints`` supplies the joint axis for the hand intercept when the
    feature tensor is flattened to ``(samples, 5)``.
    """
    fitted = default_model() if model is None else model
    values = np.asarray(features, dtype=np.float64)
    if values.shape[-1] != len(FEATURE_NAMES):
        raise ValueError(f"features must end in {len(FEATURE_NAMES)} values")
    mean = np.asarray(fitted["feature_mean"], dtype=np.float64)
    scale = np.asarray(fitted["feature_scale"], dtype=np.float64)
    coefficients = np.asarray(fitted["coefficients"], dtype=np.float64)
    values = np.where(np.isfinite(values), values, mean)
    standardised = np.clip((values - mean) / scale, -8.0, 8.0)
    logit = float(fitted.get("intercept", 0.0)) + standardised @ coefficients
    joint_count = values.shape[-2] if num_joints is None and values.ndim >= 2 else num_joints
    if joint_count is not None and int(joint_count) > HAND_START:
        is_hand = np.zeros(int(joint_count), dtype=np.float64)
        is_hand[HAND_START:] = 1.0
        logit = logit + float(fitted.get("hand_offset", 0.0)) * is_hand
    probability = _sigmoid(logit)
    if valid is not None:
        mask = np.asarray(valid, dtype=bool)
        if mask.shape != probability.shape:
            raise ValueError("valid must match the confidence shape")
        probability = np.where(mask, probability, 0.0)
    return np.clip(probability, 0.0, 1.0).astype(np.float32)


def fit_confidence3d(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    is_hand: np.ndarray | None = None,
    l2: float = 0.02,
    sample_mask: np.ndarray | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fit the sign-constrained logistic model and return a JSON-ready dict.

    ``features`` is ``(frames, joints, 5)`` and ``labels`` is the boolean
    "good" label with shape ``(frames, joints)``.  Only finite feature rows and
    finite labels are used.  ``is_hand`` is a per-joint boolean vector; the
    hand intercept lets the model calibrate body and hands separately while
    keeping one monotone feature set.
    """
    from scipy.optimize import minimize

    values = np.asarray(features, dtype=np.float64)
    target = np.asarray(labels, dtype=float)
    if values.shape[:-1] != target.shape:
        raise ValueError("features and labels must share the frame/joint shape")
    if values.shape[-1] != len(FEATURE_NAMES):
        raise ValueError(f"features must end in {len(FEATURE_NAMES)} values")
    usable = np.isfinite(values).all(axis=-1) & np.isfinite(target)
    if sample_mask is not None:
        mask = np.asarray(sample_mask, dtype=bool)
        if mask.shape != target.shape:
            raise ValueError("sample_mask must match the label shape")
        usable &= mask
    rows = values[usable]
    y = target[usable]
    if rows.size == 0:
        raise ValueError("no usable rows to fit the confidence model")
    if np.unique(y).size < 2:
        raise ValueError("labels need both good and bad examples")

    if is_hand is None:
        hand_joint = np.zeros(values.shape[1], dtype=np.float64)
        hand_joint[HAND_START:] = 1.0
    else:
        hand_joint = np.asarray(is_hand, dtype=np.float64).reshape(-1)
        if hand_joint.shape != (values.shape[1],):
            raise ValueError("is_hand must have one value per joint")
    hand_rows = np.broadcast_to(hand_joint, target.shape)[usable]

    mean = rows.mean(axis=0)
    scale = rows.std(axis=0)
    scale = np.where(scale > 1e-9, scale, 1.0)
    standardised = (rows - mean) / scale

    def objective(parameters: np.ndarray) -> float:
        logit = parameters[0] + parameters[1] * hand_rows + standardised @ parameters[2:]
        return float(
            np.mean(np.logaddexp(0.0, logit) - y * logit)
            + float(l2) * np.sum(parameters[2:] ** 2)
        )

    bounds: list[tuple[float | None, float | None]] = [(None, None), (None, None)]
    for name in FEATURE_NAMES:
        bounds.append((0.0, None) if SIGN_CONSTRAINTS[name] == "increasing" else (None, 0.0))
    positive_rate = float(np.clip(y.mean(), 1e-3, 1.0 - 1e-3))
    start = np.zeros(2 + len(FEATURE_NAMES), dtype=np.float64)
    start[0] = float(np.log(positive_rate / (1.0 - positive_rate)))
    result = minimize(objective, start, method="L-BFGS-B", bounds=bounds)

    coefficients = np.asarray(result.x[2:], dtype=np.float64)
    for index, name in enumerate(FEATURE_NAMES):
        if SIGN_CONSTRAINTS[name] == "increasing":
            coefficients[index] = max(float(coefficients[index]), 0.0)
        else:
            coefficients[index] = min(float(coefficients[index]), 0.0)

    hand_mask = hand_rows > 0.5
    body_mask = ~hand_mask
    payload: dict[str, Any] = {
        "version": MODEL_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "sign_constraints": dict(SIGN_CONSTRAINTS),
        "intercept": float(result.x[0]),
        "hand_offset": float(result.x[1]),
        "coefficients": [float(value) for value in coefficients],
        "feature_mean": [float(value) for value in mean],
        "feature_scale": [float(value) for value in scale],
        "fit": {
            "samples": int(rows.shape[0]),
            "samples_body": int(body_mask.sum()),
            "samples_hands": int(hand_mask.sum()),
            "good_rate_body": float(y[body_mask].mean()) if body_mask.any() else None,
            "good_rate_hands": float(y[hand_mask].mean()) if hand_mask.any() else None,
            "l2": float(l2),
            "optimizer_success": bool(result.success),
            "optimizer_nit": int(getattr(result, "nit", 0)),
            "objective": float(result.fun),
        },
    }
    if metadata:
        payload["fit"].update(dict(metadata))
    return payload


def write_model(model: Mapping[str, Any], path: str | Path | None = None) -> Path:
    """Write a fitted model (without the in-memory numpy copies)."""
    target = Path(path) if path is not None else model_path()
    clean = {
        key: value
        for key, value in model.items()
        if not key.endswith("_array")
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(clean, indent=2, sort_keys=True) + "\n", encoding="ascii")
    return target


__all__ = [
    "FEATURE_NAMES",
    "MODEL_VERSION",
    "SIGN_CONSTRAINTS",
    "body_height_px",
    "build_features",
    "default_model",
    "fit_confidence3d",
    "load_model",
    "model_path",
    "predict_confidence3d",
    "write_model",
]
