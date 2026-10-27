"""Service configuration, read once from environment variables."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else default


def _float(name: str, default: float) -> float:
    value = os.environ.get(name, "").strip()
    try:
        return float(value) if value else default
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {value!r}") from exc


@dataclass(frozen=True)
class Settings:
    """Runtime settings of the web service."""

    data_dir: Path
    runs_dir: Path
    showcase_dir: Path
    log_dir: Path
    repo_root: Path
    password: str | None
    dev_mode: bool
    s3_bucket: str | None
    cloudwatch: bool
    hourly_usd: float
    run_timeout_s: float
    python: str

    @classmethod
    def from_env(cls) -> "Settings":
        repo_root = _path("AIMOCAP_REPO_ROOT", REPO_ROOT)
        return cls(
            data_dir=_path("AIMOCAP_DATA_DIR", repo_root / "data" / "opencap_trials"),
            runs_dir=_path("AIMOCAP_RUNS_DIR", repo_root / "runs" / "service"),
            showcase_dir=_path("AIMOCAP_SHOWCASE_DIR", repo_root / "showcase"),
            log_dir=_path("AIMOCAP_LOG_DIR", repo_root / "runs" / "logs"),
            repo_root=repo_root,
            password=os.environ.get("AIMOCAP_PASSWORD") or None,
            dev_mode=os.environ.get("AIMOCAP_DEV", "0") == "1",
            s3_bucket=os.environ.get("AIMOCAP_S3_BUCKET") or None,
            cloudwatch=os.environ.get("AIMOCAP_CLOUDWATCH", "0") == "1",
            hourly_usd=_float("AIMOCAP_HOURLY_USD", 0.0),
            run_timeout_s=_float("AIMOCAP_RUN_TIMEOUT_S", 1200.0),
            python=os.environ.get("AIMOCAP_PYTHON", sys.executable),
        )

    def validate(self) -> None:
        """Refuse to serve without a password unless development mode is on."""
        if not self.password and not self.dev_mode:
            raise RuntimeError(
                "AIMOCAP_PASSWORD is not set. Set it, or set AIMOCAP_DEV=1 for local development."
            )
