import numpy as np

from aimocap.retarget.engine import _compose_p2_targets
from aimocap.retarget.observation_conditioning import condition_target_tracks
from aimocap.retarget.rigid_segments import RIGID_SEGMENTS


def test_p2_conditions_raw_face_when_rigid_gate_rejects():
    rng = np.random.default_rng(1234)
    frames = 40
    raw = {
        name: rng.normal(size=(frames, 3)).astype(np.float64)
        for names in RIGID_SEGMENTS.values() for name in names
    }
    raw.update({
        "hip_l": rng.normal(size=(frames, 3)),
        "hip_r": rng.normal(size=(frames, 3)),
        "shoulder_l": rng.normal(size=(frames, 3)),
        "shoulder_r": rng.normal(size=(frames, 3)),
    })
    raw["nose"] *= 100.0

    p2, diag = _compose_p2_targets(raw, fps=30.0)
    assert diag["c3h"]["segments"]["head_face"]["rigidity_gate_pass"] is False
    expected, _ = condition_target_tracks(
        {name: raw[name] for name in RIGID_SEGMENTS["head_face"]}, fps=30.0
    )
    for name in RIGID_SEGMENTS["head_face"]:
        np.testing.assert_array_equal(p2[name], expected[name])
