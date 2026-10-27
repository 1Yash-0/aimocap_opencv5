"""Download the ONNX models used by the default pipeline into ``models/``.

Models (all distributed by their authors, see THIRD_PARTY_NOTICES.md):

- RTMW-x whole-body pose, 384x288 (OpenMMLab MMPose)  -> models/rtmw/
- RTMPose-m hand pose, 256x256 (OpenMMLab MMPose)     -> models/rtmpose_hand/
- YOLOX-Nano person detector (cigpose-onnx release)   -> models/yolox_nano.onnx

The archives are downloaded once, the ONNX files are extracted to the paths the
pipeline expects, and their SHA-256 digests are printed (and checked where a
reference digest is known).

Usage::

    python scripts/fetch_models.py            # all models
    python scripts/fetch_models.py --list     # show what would be fetched
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
CACHE_DIR = MODELS_DIR / ".cache"


@dataclass(frozen=True)
class ModelSpec:
    """One model file and the archive it is extracted from."""

    name: str
    url: str
    member_suffix: str  # suffix of the archive member to extract
    target: Path  # destination relative to MODELS_DIR
    sha256: str | None = None  # reference digest of the extracted file, if known


MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        name="rtmw-x",
        url=(
            "https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/"
            "rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.zip"
        ),
        member_suffix="end2end.onnx",
        target=Path("rtmw/rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.onnx"),
    ),
    ModelSpec(
        name="rtmpose-m-hand",
        url=(
            "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
            "rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.zip"
        ),
        member_suffix="end2end.onnx",
        target=Path(
            "rtmpose_hand/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320/end2end.onnx"
        ),
        sha256="39e858936bca0f94c09847d4e70b68a51d6c0adac61f36b457fcadb54621cd29",
    ),
    ModelSpec(
        name="yolox-nano",
        url="https://github.com/namas191297/cigpose-onnx/releases/download/v1.0.0/cigpose_models.zip",
        member_suffix="yolox_nano.onnx",
        target=Path("yolox_nano.onnx"),
        sha256="c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d",
    ),
)


def sha256_of(path: Path) -> str:
    """Return the hex SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, dest: Path) -> None:
    """Download ``url`` to ``dest`` with a simple progress line."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as response, tmp.open("wb") as out:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        while chunk := response.read(1 << 20):
            out.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 1e6:7.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    print()
    tmp.replace(dest)


def fetch(spec: ModelSpec) -> bool:
    """Make sure ``spec.target`` exists; return True if it is present and verified."""
    target = MODELS_DIR / spec.target
    if not target.exists():
        archive = CACHE_DIR / Path(spec.url).name
        if not archive.exists():
            print(f"{spec.name}: downloading {spec.url}")
            download(spec.url, archive)
        with zipfile.ZipFile(archive) as zf:
            members = [m for m in zf.namelist() if m.endswith(spec.member_suffix)]
            if len(members) != 1:
                print(f"{spec.name}: expected one '{spec.member_suffix}' in {archive.name}, found {members}")
                return False
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(members[0]) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
    digest = sha256_of(target)
    if spec.sha256 and digest != spec.sha256:
        print(f"{spec.name}: SHA-256 mismatch for {target} ({digest})")
        return False
    status = "verified" if spec.sha256 else "sha256"
    print(f"{spec.name}: {target.relative_to(PROJECT_ROOT)} ({status} {digest})")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--list", action="store_true", help="list the models and exit")
    args = parser.parse_args(argv)
    if args.list:
        for spec in MODELS:
            print(f"{spec.name:16s} -> models/{spec.target.as_posix()}\n{'':16s}    {spec.url}")
        return 0
    ok = all([fetch(spec) for spec in MODELS])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
