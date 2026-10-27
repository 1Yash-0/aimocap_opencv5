"""Regression tests for the opt-in Stage 4 replant anchor policy."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from aimocap.retarget.fbx_rig import Skeleton
from aimocap.retarget import foot_lock

ROOT = Path(__file__).resolve().parents[1]
REF_PATH = ROOT / "outputs" / "gt_harness" / "cache" / "foot_lock_pre_replant_ref.py"


from tests._assets import requires_manny_fbx
def _load_reference():
    spec = importlib.util.spec_from_file_location("foot_lock_pre_replant_ref", REF_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rest_animation(skel: Skeleton, frames: int, root_x: np.ndarray | None = None):
    rest = Rotation.from_quat(np.asarray(skel.rest_rotations)).as_euler("xyz")
    animation = np.zeros((frames, 3 + 3 * skel.num_joints), dtype=float)
    animation[:, 3:] = rest.reshape(-1)
    if root_x is not None:
        animation[:, 0] = root_x
    return animation


def _contact_patch(monkeypatch, frames: int):
    mask = np.ones(frames, dtype=bool)
    strength, segments = foot_lock._build_contact_strength(mask, blend_frames=5)
    contacts = {
        side: {
            "mask": mask.copy(),
            "strength": strength.copy(),
            "segments": list(segments),
            "ground_z": 0.0,
            "height_band": 3.0,
            "speed_threshold": 0.0,
        }
        for side in ("left", "right")
    }

    def replacement(*args, **kwargs):
        return contacts

    monkeypatch.setattr(foot_lock, "detect_target_foot_contacts", replacement)


def _fk(skel: Skeleton, animation: np.ndarray) -> np.ndarray:
    eulers = animation[:, 3:].reshape(len(animation), skel.num_joints, 3)
    quats = Rotation.from_euler("xyz", eulers.reshape(-1, 3)).as_quat()
    quats = quats.reshape(len(animation), skel.num_joints, 4)
    return foot_lock._fk_sequence(skel, animation[:, :3], quats)[0]


@requires_manny_fbx
def test_default_policy_matches_pre_replant_reference_bit_for_bit(monkeypatch):
    skel = Skeleton(str(ROOT / "Manny.FBX"))
    frames = 24
    animation = _rest_animation(skel, frames, np.linspace(0.0, 4.0, frames))
    source = np.zeros((frames, 133, 3), dtype=float)
    _contact_patch(monkeypatch, frames)

    current, current_diag = foot_lock.apply_foot_lock(
        skel, animation.copy(), source, fps=30.0)
    reference = _load_reference()
    expected, _ = reference.apply_foot_lock(
        skel, animation.copy(), source, fps=30.0)
    assert np.array_equal(current, expected)
    assert current_diag["anchor_policy"] == "fixed"


@requires_manny_fbx
def test_replant_tracks_sliding_pivot_while_fixed_keeps_stale_anchor(monkeypatch):
    skel = Skeleton(str(ROOT / "Manny.FBX"))
    frames = 180
    root_x = np.linspace(0.0, 40.0, frames)
    animation = _rest_animation(skel, frames, root_x)
    source = np.zeros((frames, 133, 3), dtype=float)
    _contact_patch(monkeypatch, frames)

    fixed, fixed_diag = foot_lock.apply_foot_lock(
        skel, animation.copy(), source, fps=30.0, anchor_policy="fixed")
    replanted, replanted_diag = foot_lock.apply_foot_lock(
        skel, animation.copy(), source, fps=30.0, anchor_policy="replant")
    before = _fk(skel, animation)
    fixed_pos = _fk(skel, fixed)
    replant_pos = _fk(skel, replanted)
    foot = skel.name_to_idx["foot_l"]
    error_fixed = np.linalg.norm(fixed_pos[:, foot] - before[:, foot], axis=1)
    error_replant = np.linalg.norm(replant_pos[:, foot] - before[:, foot], axis=1)

    assert fixed_diag["anchor_policy"] == "fixed"
    assert replanted_diag["replant_count"] > 1
    assert float(np.percentile(error_fixed, 95)) > 10.0
    assert float(np.percentile(error_replant, 95)) < 3.0
    assert np.isfinite(replanted).all()


@requires_manny_fbx
def test_replant_does_not_worsen_a_truly_planted_noisy_foot(monkeypatch):
    skel = Skeleton(str(ROOT / "Manny.FBX"))
    frames = 120
    noise = 0.35 * np.sin(np.linspace(0.0, 18.0 * np.pi, frames))
    animation = _rest_animation(skel, frames, noise)
    source = np.zeros((frames, 133, 3), dtype=float)
    _contact_patch(monkeypatch, frames)

    fixed, _ = foot_lock.apply_foot_lock(
        skel, animation.copy(), source, fps=30.0, anchor_policy="fixed")
    replanted, diag = foot_lock.apply_foot_lock(
        skel, animation.copy(), source, fps=30.0, anchor_policy="replant")
    fixed_pos = _fk(skel, fixed)
    replant_pos = _fk(skel, replanted)
    foot = skel.name_to_idx["foot_l"]
    fixed_slide = np.linalg.norm(np.diff(fixed_pos[:, foot], axis=0), axis=1)
    replant_slide = np.linalg.norm(np.diff(replant_pos[:, foot], axis=0), axis=1)

    assert diag["replant_count"] == 0
    assert float(np.percentile(replant_slide, 95)) <= float(np.percentile(fixed_slide, 95)) + 0.1
    assert np.isfinite(replanted).all()


def test_unreachable_replant_target_falls_back_and_stays_finite():
    target, weight, flagged = foot_lock._reachable_lock_target(
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 50.0]),
        np.array([1000.0, 0.0, 0.0]),
        1.0,
        30.0,
        30.0,
    )
    assert flagged
    assert 0.0 <= weight < 1.0
    assert np.isfinite(target).all()
    assert np.linalg.norm(target) <= 60.0
