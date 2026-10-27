"""WP-A: --profile instrumentation must be zero-overhead when off and must
write a complete profile JSON when on, with bit-identical solved output."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aimocap.retarget.engine import retarget_to_fbx

MANNY = ROOT / "Manny.FBX"


from tests._assets import requires_manny_fbx
@pytest.fixture(scope="module")
def healthy_npz(tmp_path_factory):
    """(12, 133, 3) m-skeleton, all core tracks finite (passes WP2.B gate)."""
    rng = np.random.default_rng(7)
    pts = rng.normal(0.0, 0.4, size=(12, 133, 3))
    pts[:, :, 1] += 1.0
    p = tmp_path_factory.mktemp("prof") / "in.npz"
    np.savez(p, skeleton3d=pts, units=np.asarray("m"),
             confidence=np.ones((12, 133), dtype=np.float32))
    return p


def _run(npz, out_bvh, profile, pipeline_profile=None):
    kwargs = {}
    if pipeline_profile is not None:
        kwargs["pipeline_profile"] = pipeline_profile
    ret = retarget_to_fbx(
        str(npz), str(MANNY), str(out_bvh),
        mocap_target_stabilization=False,
        leg_stabilization=False,
        foot_lock=False,
        export_fbx=False,
        profile=profile,
        **kwargs,
    )
    return np.load(str(out_bvh).replace(".bvh", "_solved.npy"))


@requires_manny_fbx
def test_profile_off_writes_no_profile_json_and_is_identical(healthy_npz, tmp_path):
    out = tmp_path / "off.bvh"
    arr = _run(healthy_npz, out, profile=False)
    assert not (tmp_path / "off.profile.json").exists()
    assert arr.shape[0] == 12
    assert np.isfinite(arr).all()


@requires_manny_fbx
def test_profile_on_writes_complete_json(healthy_npz, tmp_path):
    out = tmp_path / "on.bvh"
    arr = _run(healthy_npz, out, profile=True)
    prof_path = tmp_path / "on.profile.json"
    assert prof_path.exists()
    doc = json.loads(prof_path.read_text())
    for key in ("stage_wall_s_cumulative", "stage_wall_s_per_block",
                "solve_frame_ms", "total_s", "pipeline_profile",
                "effective_options", "module_hashes"):
        assert key in doc
    assert doc["pipeline_profile"] == "v7"
    assert doc["effective_options"]["target_conditioning"] == "legacy"
    assert len(doc["module_hashes"]) >= 5
    # Required stage marks present (stages disabled -> some marks absent).
    for stage in ("extract_mocap_points", "fill_missing_target_tracks",
                  "solve_loop", "bvh_save"):
        assert stage in doc["stage_wall_s_cumulative"]
    assert doc["total_s"] > 0.0
    if doc["solve_frame_ms"]:
        assert doc["solve_frame_ms"]["n_frames_timed"] == 12


@requires_manny_fbx
def test_profile_does_not_change_solved_output(healthy_npz, tmp_path):
    a = _run(healthy_npz, tmp_path / "a.bvh", profile=False)
    b = _run(healthy_npz, tmp_path / "b.bvh", profile=True)
    assert np.array_equal(a, b), "profile instrumentation changed the output"


@requires_manny_fbx
def test_v7_implicit_and_explicit_profile_paths_are_bit_identical(healthy_npz, tmp_path):
    implicit = _run(healthy_npz, tmp_path / "implicit.bvh", profile=False)
    explicit = _run(healthy_npz, tmp_path / "explicit.bvh", profile=False, pipeline_profile="v7")
    assert implicit.tobytes() == explicit.tobytes()


def test_v8_preset_selects_p1_and_disables_stages_by_default():
    from aimocap.retarget.engine import _effective_pipeline_options

    assert _effective_pipeline_options("v8", None, None, None) == ("p1", False, False)
    assert _effective_pipeline_options("v8", "legacy", True, True) == ("legacy", True, True)
    assert _effective_pipeline_options("v7", None, None, None) == ("legacy", True, True)


def test_v8_implicit_fps_reads_npz_fps(tmp_path):
    npz = tmp_path / "with_fps.npz"
    np.savez(npz, fps=np.asarray(29.97002997002997))
    from aimocap.cli import _resolve_retarget_fps

    assert _resolve_retarget_fps(npz, "v8", None) == pytest.approx(29.97002997002997)


def test_v8_implicit_fps_derives_from_timestamps(tmp_path):
    npz = tmp_path / "with_timestamps.npz"
    timestamps = np.arange(450, dtype=np.float64) / 29.97002997002997
    np.savez(npz, timestamps=timestamps)
    from aimocap.cli import _resolve_retarget_fps

    assert _resolve_retarget_fps(npz, "v8", None) == pytest.approx(29.97002997002997)


def test_v8_implicit_fps_fails_closed_without_metadata(tmp_path):
    npz = tmp_path / "without_fps.npz"
    np.savez(npz, skeleton3d=np.zeros((1, 17, 3)))
    from aimocap.cli import _resolve_retarget_fps

    with pytest.raises(ValueError, match=r"provide --fps explicitly"):
        _resolve_retarget_fps(npz, "v8", None)


def test_v7_implicit_fps_remains_30(tmp_path):
    npz = tmp_path / "v7.npz"
    np.savez(npz, fps=np.asarray(29.97002997002997))
    from aimocap.cli import _resolve_retarget_fps

    assert _resolve_retarget_fps(npz, "v7", None) == 30.0

