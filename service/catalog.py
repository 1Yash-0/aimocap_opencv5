"""Catalogue of prepared trials and precomputed showcase runs."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")

# Files a results page may serve, with their media types.
RESULT_FILES: dict[str, str] = {
    "overlay_2d.mp4": "video/mp4",
    "skeleton_vs_mocap.mp4": "video/mp4",
    "character.mp4": "video/mp4",
    "motion.bvh": "text/plain",
    "skeleton3d.npz": "application/octet-stream",
    "metrics.json": "application/json",
    "calibration.json": "application/json",
    "calibration_check.json": "application/json",
    "sync.json": "application/json",
    "trial.json": "application/json",
    "events.jsonl": "application/x-ndjson",
}

# Movement types by trial prefix, in the order the results page lists them.
MOVEMENTS: dict[str, str] = {"squats": "Squats", "walking": "Walking", "sts": "Sit to stand", "dj": "Drop jump"}


def read_json(path: Path) -> dict[str, Any] | None:
    """Read a JSON object, returning None if the file is missing or invalid."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def movement_key(trial: str) -> str:
    """Movement type from an OpenCap trial name: ``squats1`` and ``squats2`` give ``squats``."""
    return re.sub(r"\d+$", "", trial.lower())


def movement_label(trial: str) -> str:
    """Human-readable movement name, for example ``STS1`` gives ``Sit to stand``."""
    key = movement_key(trial)
    return MOVEMENTS.get(key, key.capitalize() or trial)


def movement_rank(key: str) -> tuple[int, str]:
    """Sort key that lists known movements in their page order, then any others alphabetically."""
    order = list(MOVEMENTS)
    return (order.index(key) if key in MOVEMENTS else len(order), key)


@dataclass(frozen=True)
class Trial:
    """A prepared trial that can be run, optionally with a showcase result."""

    name: str
    subject: str
    trial: str
    cams: tuple[str, ...]
    trial_dir: Path
    showcase_dir: Path | None
    # Marker mocap paths from trial.json metadata, relative to the trial folder, in order of preference.
    mocap_rels: tuple[str, ...] = ()

    @property
    def subject_label(self) -> str:
        return self.subject.replace("subject", "Subject ")

    @property
    def movement_key(self) -> str:
        return movement_key(self.trial)

    @property
    def movement(self) -> str:
        return movement_label(self.trial)

    @property
    def title(self) -> str:
        return f"{self.subject_label}, {self.movement.lower()}"

    def mocap_file(self) -> Path | None:
        """Marker motion capture (.trc) of this trial: the first listed file that exists inside the trial folder."""
        base = self.trial_dir.resolve()
        for rel in self.mocap_rels:
            path = (base / rel).resolve()
            if path.is_relative_to(base) and path.is_file():
                return path
        return None


class Catalog:
    """Looks up trials in the data directory and results in the showcase directory."""

    def __init__(self, data_dir: Path, showcase_dir: Path) -> None:
        self.data_dir = data_dir
        self.showcase_dir = showcase_dir

    def trials(self) -> list[Trial]:
        names: set[str] = set()
        for root in (self.data_dir, self.showcase_dir):
            if root.is_dir():
                names.update(p.name for p in root.iterdir() if p.is_dir() and NAME_RE.match(p.name))
        found = [t for t in (self.get(name) for name in sorted(names)) if t is not None]
        return found

    def get(self, name: str) -> Trial | None:
        if not NAME_RE.match(name):
            return None
        trial_dir = self.data_dir / name
        showcase = self.showcase_dir / name
        data_meta = read_json(trial_dir / "trial.json")
        showcase_meta = read_json(showcase / "trial.json")
        meta = data_meta or showcase_meta
        if meta is None:
            return None
        return Trial(
            name=name,
            subject=str(meta.get("subject", name)),
            trial=str(meta.get("trial", "")),
            cams=tuple(meta.get("cams", ())),
            trial_dir=trial_dir,
            showcase_dir=showcase if (showcase / "metrics.json").is_file() else None,
            mocap_rels=tuple(rel for rel in (_mocap_rel(data_meta), _mocap_rel(showcase_meta)) if rel),
        )

    def runnable(self, trial: Trial) -> bool:
        return (trial.trial_dir / "trial.json").is_file()


def _mocap_rel(meta: dict[str, Any] | None) -> str | None:
    """The ``files.mocap`` path of a trial.json, relative to the trial folder."""
    files = meta.get("files") if meta else None
    value = files.get("mocap") if isinstance(files, dict) else None
    return str(value) if value else None


def safe_file(base: Path, filename: str) -> Path | None:
    """Resolve an allow-listed result file inside ``base``; None if not allowed or absent."""
    if filename not in RESULT_FILES:
        return None
    base = base.resolve()
    candidate = (base / filename).resolve()
    if not candidate.is_relative_to(base) or not candidate.is_file():
        return None
    return candidate
