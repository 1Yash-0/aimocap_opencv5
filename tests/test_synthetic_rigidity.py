"""Test that the synthetic ground-truth data is rigid under the solver's proxy
topology.

The synthetic data in outputs/true_synthetic.npz is supposed to be generated
from the SAME skeleton the IK solver uses (the proxy MocapSkeleton). If it's
generated from the FBX rig instead, the topology differs (e.g. FBX parents
shoulders under clavicle as a sibling of neck_01; the proxy parents shoulders
under neck_01). A neck_01 bend then moves the shoulders in the proxy but NOT
in the FBX, making the data non-rigid and unfittable.

Rigidity check: in the proxy topology, the shoulder line and hip line are both
perpendicular to the spine direction in rest, and both rotate WITH the spine.
So angle(spine_dir, shoulder_line) should stay near 90° for all frames.
If the data was generated from the FBX topology (where neck_01 bend doesn't
move the shoulders), this angle blows out toward 180°.
"""
import numpy as np
import pytest

from aimocap.retarget.ik_probe import load_slice


from tests._assets import requires_true_synthetic
def _spine_shoulder_angles(pts3d: np.ndarray) -> np.ndarray:
    """Per-frame angle (deg) between spine_dir and shoulder_line."""
    F = pts3d.shape[0]
    angles = np.zeros(F)
    for f in range(F):
        pelvis = (pts3d[f, 11] + pts3d[f, 12]) / 2.0   # COCO pelvis_l, pelvis_r
        neck = (pts3d[f, 5] + pts3d[f, 6]) / 2.0        # COCO shoulder_l, shoulder_r
        spine_dir = neck - pelvis
        shoulder_line = pts3d[f, 6] - pts3d[f, 5]
        cos_a = np.dot(spine_dir, shoulder_line) / (
            np.linalg.norm(spine_dir) * np.linalg.norm(shoulder_line)
        )
        # Full angle: 0° = parallel, 90° = perpendicular, 180° = anti-parallel.
        # In rest this is 90°. Do NOT use abs() — that collapses 176° (the bug)
        # to 4° and makes the test pass when it shouldn't.
        angles[f] = np.degrees(np.arccos(np.clip(cos_a, -1, 1)))
    return angles


@requires_true_synthetic
def test_synthetic_data_is_rigid_under_proxy_topology():
    """Spine-shoulder angle must stay near 90° (perpendicular) for all frames.

    In rest the angle is 90.001°. With small per-joint bends (~0.2 rad) applied
    through the proxy topology (shoulders as children of neck_01, which is in
    the spine chain), the shoulders rotate with the spine so the angle stays
    near 90°. Tolerance 30° (60°-120°) is generous: the worst-case 3×0.2 rad
    cumulative bend only shifts this by a few degrees. Data generated from the
    FBX topology (neck_01 bend doesn't move shoulders) produces ~176°.
    """
    pts3d = load_slice()
    angles = _spine_shoulder_angles(pts3d)
    worst = angles.max()
    assert worst < 120.0, (
        f"Synthetic data is non-rigid: spine-shoulder angle reaches {worst:.1f}° "
        f"(should be ~90°). Per-frame: min={angles.min():.1f}° "
        f"mean={angles.mean():.1f}° max={angles.max():.1f}°. "
        f"This means the data was likely generated from a different topology "
        f"than the solver uses."
    )
