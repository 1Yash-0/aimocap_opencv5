import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aimocap.pose.score_calibration import calibrate_scores, load_calibration


def test_calibrated_scores_are_probabilities_and_monotonic():
    raw = np.tile(np.linspace(-2.0, 12.0, 300)[:, None], (1, 133))
    p = calibrate_scores(raw)
    assert p.shape == raw.shape
    assert np.all((p >= 0.0) & (p <= 1.0))
    assert np.all(np.diff(p, axis=0) >= -1e-12)


def test_nan_stays_nan_and_deterministic():
    raw = np.random.default_rng(0).uniform(2, 7, size=(5, 133))
    raw[2, 10] = np.nan
    a, b = calibrate_scores(raw), calibrate_scores(raw)
    assert np.isnan(a[2, 10]) and np.array_equal(a, b, equal_nan=True)


def test_matches_experiment_mapping():
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        import phase1_score_calibration as exp
    except ImportError:
        pytest.skip("scripts/phase1_score_calibration.py is not part of this repository")

    table = load_calibration()
    raw = np.random.default_rng(1).uniform(0, 8, size=(1000, 133))
    p = calibrate_scores(raw, table)
    for k, g in table["body_joint_index_to_group"].items():
        ref = exp.predict_calibration(raw[:, int(k)], table["groups"][g])
        assert np.max(np.abs(ref - p[:, int(k)])) < 1e-9
