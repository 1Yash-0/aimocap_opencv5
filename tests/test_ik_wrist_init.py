# tests/test_ik_wrist_init.py
"""WP0.2: latent `wrist_targets_init` regression tests (CPU-only).

Root cause (see WP0.2 report): `MocapIKSolver.analytic_init` contained a
wrist-twist hook reading the name `wrist_targets_init`, which no code in the
repository ever assigned. Execution tracing proved the reference unreachable:
the leaf-orient block above it (same target_pos, same parent rotation, same
rest direction) computes the identical 70-degree gate and `continue`s
whenever the gate passes — so the hook's own duplicate gate can never pass
on arrival. The dead reference was removed; these tests pin that outcome.
"""
import ast
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aimocap.retarget.mocap_ik import MocapIKSolver
from aimocap.retarget.mocap_skeleton import MocapSkeleton

TARGET_FILE = ROOT / "aimocap" / "retarget" / "mocap_ik.py"


from tests._assets import requires_manny_fbx
def test_no_undefined_wrist_targets_reference():
    """Fail if `wrist_targets_init` is read without ever being assigned."""
    tree = ast.parse(TARGET_FILE.read_text(encoding="utf-8"))
    loaded = {n.id for n in ast.walk(tree)
              if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    assigned = {n.id for n in ast.walk(tree)
                if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del))}
    assert "wrist_targets_init" not in loaded, (
        "dead undefined reference reintroduced in mocap_ik.py")


def _rest_aligned_measured(skel: MocapSkeleton) -> dict:
    """Measured targets placed exactly on the proxy rest geometry.

    Every desired bone direction then equals its rest direction, so every
    gated leaf branch is exercised.
    """
    proxy_of_coco: dict[str, int] = {}
    for idx, coco_name in skel.coco_anchor.items():
        proxy_of_coco.setdefault(coco_name, idx)
    offset = np.array([0.0, 0.0, 100.0])
    measured = {name: np.asarray(skel.rest_t[idx], dtype=float) + offset
                for name, idx in proxy_of_coco.items()}
    for name in ("nose", "left_eye", "right_eye", "left_ear", "right_ear",
                 "heel_l", "heel_r", "big_toe_l", "big_toe_r"):
        if name not in measured:
            measured[name] = measured.get("neck", np.zeros(3)).copy()
    return measured


def _solver() -> MocapIKSolver:
    pts3d = np.zeros((1, 133, 3), dtype=np.float64)
    pts3d[0, :, 1] = np.linspace(0, 150, 133)
    weights = np.ones((1, 133), dtype=np.float64)
    return MocapIKSolver(MocapSkeleton(pts3d, weights))


@requires_manny_fbx
def test_analytic_init_aligned_leaf_output_finite():
    solver = _solver()
    x = solver.analytic_init(_rest_aligned_measured(solver.skel))
    assert isinstance(x, np.ndarray)
    assert x.shape == (solver.num_vars,)
    assert np.all(np.isfinite(x)), "NaN found in analytic_init output"


@requires_manny_fbx
def test_analytic_init_is_deterministic():
    solver = _solver()
    measured = _rest_aligned_measured(solver.skel)
    a = solver.analytic_init(measured)
    b = solver.analytic_init(measured)
    assert np.array_equal(a, b)
