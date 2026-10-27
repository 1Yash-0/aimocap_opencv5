"""Focused regression tests for the promoted NumPy-only C3h operation."""
from __future__ import annotations

import numpy as np

from aimocap.retarget.rigid_segments import (
    RIGIDITY_SIGMA_LIMIT_CM,
    build_template,
    rigid_denoise_targets,
)


def _proper_motion(seed: int, frames: int, template: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(seed)
    result = []
    for _ in range(frames):
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        if np.linalg.det(q) < 0.0:
            q[:, -1] *= -1.0
        result.append(template @ q + rng.normal(0.0, 4.0, size=3))
    return np.asarray(result, dtype=np.float64)


def _tracks(points: np.ndarray) -> dict[str, np.ndarray]:
    return {f"p{i}": points[:, i].copy() for i in range(points.shape[1])}


def test_chirality_selection_handles_mirror_and_keeps_rigid_prediction():
    base = np.asarray(
        [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [0.0, 5.0, 0.0],
         [0.0, 0.0, 5.0], [3.0, 4.0, 5.0]], dtype=np.float64
    )
    for seed, source in enumerate((base, base * np.asarray([1.0, 1.0, -1.0]))):
        clean = _proper_motion(seed, 80, source)
        noisy = clean + np.random.default_rng(seed + 10).normal(0.0, 0.05, clean.shape)
        template, diag = build_template(_tracks(noisy), tuple(f"p{i}" for i in range(5)))
        assert template is not None
        assert diag["selected_chirality"] in {"mds", "mirror"}
        # The selected template must explain all frames as one rigid segment.
        output, diagnostics = rigid_denoise_targets(
            _tracks(noisy), {"segment": tuple(f"p{i}" for i in range(5))}
        )
        assert diagnostics["segments"]["segment"]["rigidity_gate_pass"]
        predicted = np.stack([output[f"p{i}"] for i in range(5)], axis=1)
        assert np.percentile(np.linalg.norm(predicted - clean, axis=2), 95) < 0.6


def test_gap_fill_replaces_missing_points_only_after_valid_three_point_fit():
    base = np.asarray(
        [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [0.0, 5.0, 0.0],
         [0.0, 0.0, 5.0], [3.0, 4.0, 5.0]], dtype=np.float64
    )
    clean = _proper_motion(21, 30, base)
    observed = clean.copy()
    observed[4, 3:] = np.nan
    observed[7, :2] = np.nan
    output, diagnostics = rigid_denoise_targets(_tracks(observed), {"segment": tuple(f"p{i}" for i in range(5))})
    assert np.isfinite(np.stack(list(output.values()), axis=1)).all()
    row = diagnostics["segments"]["segment"]
    assert row["frames_rigid_consistent"] == 30
    assert row["all_point_replacements"] == 150
    assert np.array_equal(output["p0"][0], observed[0, 0]) is False or np.array_equal(output["p0"][0], clean[0, 0])


def test_rigidity_gate_fails_closed_and_preserves_input():
    base = np.asarray(
        [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [0.0, 5.0, 0.0],
         [0.0, 0.0, 5.0], [3.0, 4.0, 5.0]], dtype=np.float64
    )
    observed = _proper_motion(33, 40, base)
    observed[:, 4] += np.random.default_rng(34).normal(0.0, 3.0, size=(40, 3))
    source = _tracks(observed)
    output, diagnostics = rigid_denoise_targets(source, {"segment": tuple(source)})
    row = diagnostics["segments"]["segment"]
    assert row["rigidity_sigma_cm"] >= RIGIDITY_SIGMA_LIMIT_CM
    assert not row["rigidity_gate_pass"]
    for name in source:
        assert np.array_equal(output[name], source[name], equal_nan=True)


def test_rigid_segments_are_bitwise_deterministic():
    base = np.asarray(
        [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [0.0, 5.0, 0.0],
         [0.0, 0.0, 5.0], [3.0, 4.0, 5.0]], dtype=np.float64
    )
    source = _tracks(_proper_motion(44, 35, base))
    first, first_diag = rigid_denoise_targets(source, {"segment": tuple(source)})
    second, second_diag = rigid_denoise_targets(source, {"segment": tuple(source)})
    for name in source:
        assert first[name].tobytes() == second[name].tobytes()
    assert first_diag == second_diag


def test_bad_shapes_fail_closed_without_mutating_track_values():
    source = {"p0": np.zeros((4, 2), dtype=np.float64), "p1": np.zeros((4, 3), dtype=np.float64)}
    output, diagnostics = rigid_denoise_targets(source, {"segment": ("p0", "p1", "p2")})
    assert not diagnostics["enabled"]
    assert diagnostics["reason"]
    assert output["p0"].shape == (4, 2)
    assert np.array_equal(output["p1"], source["p1"])
