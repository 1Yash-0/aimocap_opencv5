"""Extract the files of one OpenCap LabValidation trial from the source archive.

Only the members one trial needs are read: the five raw videos, the five
synchronised ``*_syncdWithMocap`` videos (optional), the five extrinsics
videos, the five shipped calibration pickles, the session metadata, the
marker-mocap TRC and the OpenCap HRNet TRC files.  Members are streamed one
at a time, so the 20 GB archive is never extracted as a whole.

The output folder is self-describing: ``trial.json`` records the session,
cameras, frame rate, board definition and the relative paths of every file.

Board definition: 11 x 8 inner corners with 60 mm squares.  The session
metadata of subject9 reports 35 mm; the physical board is 60 mm, so the
value is fixed here and the metadata value is recorded next to it.

CLI
---
python -m evaluation.opencap.dataset --zip <archive.zip> --subject subject4 \\
    --trial squats1 --out data/opencap_trials/subject4_squats1
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import zipfile
from pathlib import Path
from typing import Any

import cv2

ARCHIVE_ROOT = "LabValidation_withVideos"
CAMERAS: tuple[str, ...] = ("Cam0", "Cam1", "Cam2", "Cam3", "Cam4")
SESSIONS: tuple[str, ...] = ("Session0", "Session1")
TRIAL_FPS = 60.0
BOARD_INNER_CORNERS: tuple[int, int] = (11, 8)
BOARD_SQUARE_MM = 60.0
SCHEMA = "opencap_trial/1"
_SQUARE_RE = re.compile(r"squareSideLength_mm:\s*([0-9.]+)")


def _member(subject: str, *parts: str) -> str:
    return "/".join((ARCHIVE_ROOT, subject, *parts))


def _copy_member(zf: zipfile.ZipFile, member: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with zf.open(member) as src, dst.open("wb") as out:
        shutil.copyfileobj(src, out, length=1 << 20)


def find_session(names: set[str], subject: str, trial: str) -> str:
    """Return the session folder that holds ``trial`` (fail-closed if none or several)."""
    hits = [
        s for s in SESSIONS
        if _member(subject, "VideoData", s, "Cam0", trial, f"{trial}.avi") in names
    ]
    if len(hits) != 1:
        raise FileNotFoundError(
            f"trial {trial!r} for {subject!r} found in sessions {hits or 'none'}; expected exactly one"
        )
    return hits[0]


def probe_video(path: Path) -> dict[str, Any]:
    """Frame count, frame rate and size of a video file."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"cannot open video {path}")
    info = {
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": float(cap.get(cv2.CAP_PROP_FPS)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return info


def prepare_trial(
    zip_path: str | Path,
    subject: str,
    trial: str,
    out_dir: str | Path,
    *,
    include_syncd: bool = True,
) -> dict[str, Any]:
    """Extract one trial into ``out_dir`` and write ``trial.json``.

    Raises ``FileNotFoundError`` when a required member is missing and
    ``ValueError`` when the videos are not at the trial frame rate.
    """
    zip_path = Path(zip_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    notes: list[str] = []
    files: dict[str, Any] = {"videos": {}, "syncd": {}, "extrinsics": {}, "intrinsics": {}}

    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        session = find_session(names, subject, trial)
        session_root = _member(subject, "VideoData", session)

        metadata_member = _member(subject, "sessionMetadata.yaml")
        if metadata_member not in names:
            raise FileNotFoundError(metadata_member)
        metadata_text = zf.read(metadata_member).decode("utf-8", errors="replace")
        (out / "sessionMetadata.yaml").write_text(metadata_text, encoding="utf-8")
        match = _SQUARE_RE.search(metadata_text)
        metadata_square = float(match.group(1)) if match else None
        if metadata_square is not None and abs(metadata_square - BOARD_SQUARE_MM) > 1e-9:
            notes.append(
                f"sessionMetadata reports squareSideLength_mm={metadata_square:g}; "
                f"the board is {BOARD_SQUARE_MM:g} mm, which is used"
            )

        for cam in CAMERAS:
            raw = f"{session_root}/{cam}/{trial}/{trial}.avi"
            extr = f"{session_root}/{cam}/extrinsics/extrinsics.avi"
            pick = f"{session_root}/{cam}/cameraIntrinsicsExtrinsics.pickle"
            for member in (raw, extr, pick):
                if member not in names:
                    raise FileNotFoundError(member)
            rel = f"videos/{cam}.avi"
            _copy_member(zf, raw, out / rel)
            files["videos"][cam] = rel
            rel = f"extrinsics/{cam}.avi"
            _copy_member(zf, extr, out / rel)
            files["extrinsics"][cam] = rel
            rel = f"intrinsics/{cam}.pickle"
            _copy_member(zf, pick, out / rel)
            files["intrinsics"][cam] = rel
            synced = f"{session_root}/{cam}/{trial}/{trial}_syncdWithMocap.avi"
            if include_syncd and synced in names:
                rel = f"syncd/{cam}.avi"
                _copy_member(zf, synced, out / rel)
                files["syncd"][cam] = rel

        mocap_member = _member(subject, "MarkerData", "Mocap", f"{trial}.trc")
        files["mocap"] = None
        if mocap_member in names:
            _copy_member(zf, mocap_member, out / "mocap" / f"{trial}.trc")
            files["mocap"] = f"mocap/{trial}.trc"

        files["hrnet"] = {}
        for count in ("3", "5"):
            member = _member(
                subject, "MarkerData", "Video", "HRNet", f"{count}-cameras", f"{trial}_videoAndMocap.trc"
            )
            if member in names:
                rel = f"hrnet/{count}-cameras.trc"
                _copy_member(zf, member, out / rel)
                files["hrnet"][count] = rel

    frames: dict[str, int] = {}
    measured_fps: dict[str, float] = {}
    image_size: list[int] | None = None
    for cam in CAMERAS:
        info = probe_video(out / files["videos"][cam])
        frames[cam] = info["frames"]
        measured_fps[cam] = info["fps"]
        if image_size is None:
            image_size = [info["width"], info["height"]]
        if abs(info["fps"] - TRIAL_FPS) > 0.05:
            raise ValueError(f"{cam} runs at {info['fps']:.3f} fps, expected {TRIAL_FPS:g}")

    trial_info: dict[str, Any] = {
        "schema": SCHEMA,
        "subject": subject,
        "trial": trial,
        "session": session,
        "cams": list(CAMERAS),
        "fps": TRIAL_FPS,
        "measured_fps": measured_fps,
        "image_size": image_size,
        "frames": frames,
        "board": {
            "inner_corners": list(BOARD_INNER_CORNERS),
            "square_mm": BOARD_SQUARE_MM,
            "metadata_square_mm": metadata_square,
        },
        "has_mocap": files["mocap"] is not None,
        "files": files,
        "source_archive": str(zip_path),
        "notes": notes,
    }
    (out / "trial.json").write_text(json.dumps(trial_info, indent=2), encoding="utf-8")
    return trial_info


def read_trial(trial_dir: str | Path) -> dict[str, Any]:
    """Load ``trial.json`` from a prepared trial folder."""
    return json.loads((Path(trial_dir) / "trial.json").read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--zip", required=True, help="path to LabValidation_withVideos.zip")
    parser.add_argument("--subject", required=True, help="for example subject4")
    parser.add_argument("--trial", required=True, help="for example squats1")
    parser.add_argument("--out", required=True, help="output trial folder")
    parser.add_argument("--no-syncd", action="store_true", help="skip *_syncdWithMocap.avi videos")
    args = parser.parse_args(argv)
    info = prepare_trial(args.zip, args.subject, args.trial, args.out, include_syncd=not args.no_syncd)
    print(json.dumps({k: info[k] for k in ("subject", "trial", "session", "frames", "has_mocap")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
