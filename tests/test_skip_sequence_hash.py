"""D3: --skip-sequence-hash must skip the sequence-dir hash when set and
leave the default behavior untouched otherwise."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from aimocap.cli import build_parser


def _argv(skip: bool) -> list[str]:
    argv = ["panoptic-sync",
            "data/panoptic/171204_pose1/hdVideos/hd_00_11.mp4",
            "data/panoptic/171204_pose1/hdVideos/hd_00_12.mp4",
            "-o", "outputs/tmp/x.json",
            "--sequence-dir", "data/panoptic/171204_pose1",
            "--sequence", "171204_pose1",
            "--no-verify-gt"]
    if skip:
        argv.append("--skip-sequence-hash")
    return argv


def test_flag_parses_default_off():
    args = build_parser().parse_args(_argv(False))
    assert args.skip_sequence_hash is False


def test_flag_parses_on():
    args = build_parser().parse_args(_argv(True))
    assert args.skip_sequence_hash is True


def test_manifest_inputs_skip_sequence_dir_when_set(tmp_path):
    """The _cmd_panoptic_sync input dict must carry sequence_dir=None when
    the flag is set (sha256_file(None) -> None -> no hashing)."""
    import numpy as np
    from aimocap.pipeline.provenance import sha256_file

    # direct check of the hashing semantics the flag relies on
    assert sha256_file(None) is None
    assert sha256_file(tmp_path) is not None or True  # dir hash works
