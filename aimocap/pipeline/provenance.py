"""Reproducibility metadata for every public pipeline artifact."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping


def sha256_file(path: str | Path | None) -> str | None:
    if path is None:
        return None
    candidate = Path(path)
    if candidate.is_dir():
        digest = hashlib.sha256()
        for child in sorted(p for p in candidate.rglob("*") if p.is_file()):
            digest.update(str(child.relative_to(candidate)).encode("utf-8"))
            file_hash = sha256_file(child)
            digest.update(str(file_hash).encode("ascii"))
        return digest.hexdigest()
    if not candidate.is_file():
        return None
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args: str, cwd: Path) -> str:
    result = subprocess.run(
        ["git", *args], cwd=cwd, text=True, capture_output=True, check=False
    )
    return (result.stdout + result.stderr).strip()


def _dependency_lock_hash(root: Path) -> str:
    material: list[str] = []
    for name in ("pyproject.toml", "requirements.txt", "requirements-lock.txt"):
        path = root / name
        if path.exists():
            material.append(path.read_text(encoding="utf-8", errors="replace"))
    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    material.append(freeze.stdout)
    return hashlib.sha256("\n".join(material).encode("utf-8")).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    try:
        return value.item()
    except AttributeError:
        return value


def build_manifest(
    *,
    root: str | Path,
    command: str,
    inputs: Mapping[str, str | Path | None],
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a self-contained provenance manifest for one pipeline action."""
    root_path = Path(root).resolve()
    input_hashes = {str(name): sha256_file(path) for name, path in inputs.items()}
    config_values = _jsonable(dict(config or {}))
    config_hash = hashlib.sha256(
        json.dumps(config_values, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()
    input_hash = hashlib.sha256(
        json.dumps(input_hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    calibration_hash = input_hashes.get("calibration")
    return {
        "AIMOCAP_SOURCE_COMMIT": _git("rev-parse", "HEAD", cwd=root_path),
        "AIMOCAP_CONFIG_HASH": config_hash,
        "AIMOCAP_INPUT_HASH": input_hash,
        "AIMOCAP_CALIBRATION_HASH": calibration_hash,
        "git_commit": _git("rev-parse", "HEAD", cwd=root_path),
        "git_branch": _git("branch", "--show-current", cwd=root_path),
        "git_status": _git("status", "--porcelain", cwd=root_path),
        "python_version": sys.version,
        "dependency_lock_hash": _dependency_lock_hash(root_path),
        "input_hashes": input_hashes,
        "exact_command": command,
        "all_config_values": config_values,
    }


def print_startup(manifest: Mapping[str, Any]) -> None:
    """Print the four stable startup identifiers required by diagnostics."""
    for key in (
        "AIMOCAP_SOURCE_COMMIT",
        "AIMOCAP_CONFIG_HASH",
        "AIMOCAP_INPUT_HASH",
        "AIMOCAP_CALIBRATION_HASH",
    ):
        print(f"{key}={manifest.get(key)}")


def write_manifest(path: str | Path, manifest: Mapping[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(_jsonable(dict(manifest)), indent=2, default=str), encoding="utf-8")
    return destination


def sidecar_path(output: str | Path) -> Path:
    destination = Path(output)
    return destination.with_suffix(destination.suffix + ".manifest.json")


def write_sidecar(output: str | Path, manifest: Mapping[str, Any]) -> Path:
    return write_manifest(sidecar_path(output), manifest)


def add_npz_manifest(path: str | Path, manifest: Mapping[str, Any]) -> None:
    """Embed provenance in an existing NPZ without dropping its arrays."""
    destination = Path(path)
    import numpy as np

    with np.load(destination, allow_pickle=True) as data:
        payload = {key: data[key] for key in data.files}
    payload["provenance_manifest"] = np.asarray(json.dumps(_jsonable(dict(manifest)), sort_keys=True))
    np.savez(destination, **payload)


__all__ = [
    "add_npz_manifest",
    "build_manifest",
    "print_startup",
    "sha256_file",
    "sidecar_path",
    "write_manifest",
    "write_sidecar",
]
