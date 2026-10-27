"""Live pipeline runs: a persistent run store and a single background worker."""

from __future__ import annotations

import json
import os
import re
import secrets
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from service.catalog import RESULT_FILES, Trial
from service.telemetry import log_event, publish_run_metrics, read_events, upload_run

RUN_ID_RE = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{4}$")


@dataclass(frozen=True)
class Preset:
    """A fixed, safe set of pipeline options a visitor may start."""

    name: str
    label: str
    description: str
    cams: tuple[str, ...]
    part_refine: str
    window_seconds: float
    engine: str


PRESETS: dict[str, Preset] = {
    "quick": Preset(
        name="quick",
        label="Quick",
        description="3 cameras, body keypoints, 2 seconds of video, OpenCV 5 DNN on the instance CPU",
        cams=("Cam0", "Cam2", "Cam4"),
        part_refine="off",
        window_seconds=2.0,
        engine="cv2dnn",
    )
}


@dataclass
class Run:
    """State of one run, persisted as ``run.json`` in the run directory."""

    id: str
    trial: str
    preset: str
    status: str = "queued"  # queued | running | completed | failed
    created: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    started: str | None = None
    finished: str | None = None
    seconds: float | None = None
    returncode: int | None = None
    error: str | None = None

    @property
    def active(self) -> bool:
        return self.status in ("queued", "running")


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(2)


def build_command(python: str, trial: Trial, preset: Preset, out_dir: Path) -> list[str]:
    """Command line of the trial runner for a preset (list form, no shell)."""
    cmd = [
        python, "-m", "evaluation.opencap.run_trial",
        "--trial-dir", str(trial.trial_dir),
        "--out", str(out_dir),
        "--cams", ",".join(preset.cams),
        "--engine", preset.engine,
        "--part-refine", preset.part_refine,
        "--window-seconds", f"{preset.window_seconds:g}",
    ]
    if trial.showcase_dir is not None and (trial.showcase_dir / "sync.json").is_file():
        cmd += ["--sync-file", str(trial.showcase_dir / "sync.json")]
    return cmd


class RunManager:
    """Keeps run records on disk and executes at most one run at a time."""

    def __init__(
        self,
        runs_dir: Path,
        repo_root: Path,
        python: str,
        timeout_s: float,
        s3_bucket: str | None = None,
        cloudwatch: bool = False,
        command_builder: Callable[[str, Trial, Preset, Path], list[str]] = build_command,
    ) -> None:
        self.runs_dir = runs_dir
        self.repo_root = repo_root
        self.python = python
        self.timeout_s = timeout_s
        self.s3_bucket = s3_bucket
        self.cloudwatch = cloudwatch
        self.command_builder = command_builder
        self._lock = threading.Lock()
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self._recover()

    # -- store ---------------------------------------------------------------

    def run_dir(self, run_id: str) -> Path:
        return self.runs_dir / run_id

    def _save(self, run: Run) -> None:
        path = self.run_dir(run.id) / "run.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(run), indent=2), encoding="utf-8")
        tmp.replace(path)

    def get(self, run_id: str) -> Run | None:
        if not RUN_ID_RE.match(run_id):
            return None
        try:
            data = json.loads((self.run_dir(run_id) / "run.json").read_text(encoding="utf-8"))
            return Run(**data)
        except (OSError, ValueError, TypeError):
            return None

    def list(self) -> list[Run]:
        runs = [self.get(p.name) for p in self.runs_dir.iterdir() if p.is_dir()]
        return sorted((r for r in runs if r is not None), key=lambda r: r.id, reverse=True)

    def active(self) -> Run | None:
        return next((r for r in self.list() if r.active), None)

    def _recover(self) -> None:
        """Runs left running by a previous process cannot finish; mark them failed."""
        for run in self.list():
            if run.active:
                run.status = "failed"
                run.error = "interrupted by a service restart"
                self._save(run)

    # -- execution -----------------------------------------------------------

    def start(self, trial: Trial, preset: Preset) -> Run:
        """Create and launch a run; raises RuntimeError if another run is active."""
        with self._lock:
            if self.active() is not None:
                raise RuntimeError("Another run is in progress. Please wait for it to finish.")
            run = Run(id=new_run_id(), trial=trial.name, preset=preset.name)
            self._save(run)
        thread = threading.Thread(target=self._execute, args=(run, trial, preset), daemon=True)
        thread.start()
        return run

    def _execute(self, run: Run, trial: Trial, preset: Preset) -> None:
        out_dir = self.run_dir(run.id)
        cmd = self.command_builder(self.python, trial, preset, out_dir)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONUNBUFFERED="1")
        run.status = "running"
        run.started = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self._save(run)
        log_event("run started", run=run.id, trial=run.trial, preset=run.preset)
        t0 = time.monotonic()
        try:
            with (out_dir / "run.log").open("w", encoding="utf-8") as log:
                proc = subprocess.Popen(cmd, cwd=self.repo_root, env=env, stdout=log, stderr=subprocess.STDOUT)
                try:
                    run.returncode = proc.wait(timeout=self.timeout_s)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                    run.returncode = proc.returncode
                    run.error = f"timed out after {self.timeout_s:.0f} s"
        except OSError as exc:
            run.error = f"could not start the pipeline: {exc}"
        run.seconds = round(time.monotonic() - t0, 1)
        run.finished = datetime.now(timezone.utc).isoformat(timespec="seconds")
        failed = run.error is not None or run.returncode != 0
        if failed and run.error is None:
            run.error = f"pipeline exited with code {run.returncode}"
        run.status = "failed" if failed else "completed"
        self._save(run)
        log_event("run finished", run=run.id, status=run.status, seconds=run.seconds, error=run.error)
        if self.cloudwatch:
            publish_run_metrics(read_events(out_dir / "events.jsonl"), run.seconds, failed)
        if self.s3_bucket and not failed:
            upload_run(out_dir, self.s3_bucket, list(RESULT_FILES))

    # -- views ---------------------------------------------------------------

    def status_payload(self, run: Run) -> dict[str, Any]:
        """JSON for the live status endpoint: run state, stages and log tail."""
        out_dir = self.run_dir(run.id)
        stages: dict[str, dict[str, Any]] = {}
        for event in read_events(out_dir / "events.jsonl"):
            name = str(event.get("stage"))
            stages[name] = {"stage": name, "status": event.get("status"), "seconds": event.get("seconds"),
                            "message": event.get("message") or ""}
        elapsed = None
        if run.started:
            start = datetime.fromisoformat(run.started)
            end = datetime.fromisoformat(run.finished) if run.finished else datetime.now(timezone.utc)
            elapsed = round((end - start).total_seconds(), 1)
        return {
            "run": asdict(run),
            "elapsed": elapsed,
            "stages": list(stages.values()),
            "log_tail": tail(out_dir / "run.log", 40),
        }


def tail(path: Path, lines: int) -> list[str]:
    try:
        content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    return content[-lines:]
