# tests/test_confidence_calibration.py
"""P1.B: reliability-bin computation determinism (CPU-only, synthetic)."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _bins(s, e):
    dec = np.quantile(s, np.linspace(0, 1, 11))
    out = []
    for i in range(10):
        m = (s >= dec[i]) & (s <= dec[i + 1] if i == 9 else s < dec[i + 1])
        if m.sum():
            out.append((round(float(dec[i]), 3), round(float(np.median(e[m])), 3), int(m.sum())))
    return out


def test_monotone_scores_produce_ordered_bins():
    s = np.linspace(0.1, 1.0, 100)
    e = 50.0 - 40.0 * s  # higher score -> lower error
    bins = _bins(s, e)
    assert len(bins) == 10
    assert bins[0][1] > bins[-1][1]


def test_confidently_wrong_rate_known_value():
    s = np.array([0.9, 0.9, 0.2, 0.2])
    e = np.array([50.0, 5.0, 50.0, 5.0])
    assert float(((s > 0.5) & (e > 10.0)).mean()) == 0.25
