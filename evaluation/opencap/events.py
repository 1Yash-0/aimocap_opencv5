"""Append-only JSON Lines log of evaluation stages.

Every stage writes one ``started`` record and then exactly one terminal record:
``completed`` with the elapsed seconds, or ``failed`` with the exception text.
Each record is flushed as it is written, so a crashed run still leaves a
complete account of the stages that ran.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Any

STATUSES = ("started", "completed", "failed")


class EventLog:
    """Writes stage records to a ``.jsonl`` file, one JSON object per line."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a", encoding="utf-8")

    def emit(
        self,
        stage: str,
        status: str,
        seconds: float | None = None,
        message: str = "",
    ) -> dict[str, Any]:
        """Write one record and return it."""
        if status not in STATUSES:
            raise ValueError(f"unknown event status {status!r}")
        record: dict[str, Any] = {
            "ts": datetime.now().astimezone().isoformat(timespec="milliseconds"),
            "stage": stage,
            "status": status,
            "seconds": None if seconds is None else round(float(seconds), 3),
            "message": message,
        }
        self._handle.write(json.dumps(record) + "\n")
        self._handle.flush()
        return record

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Log ``started``, then ``completed`` or ``failed``; the error is re-raised."""
        self.emit(name, "started")
        t0 = time.perf_counter()
        try:
            yield
        except BaseException as exc:
            self.emit(name, "failed", time.perf_counter() - t0, f"{type(exc).__name__}: {exc}")
            raise
        self.emit(name, "completed", time.perf_counter() - t0)

    def close(self) -> None:
        """Close the underlying file."""
        self._handle.close()

    def __enter__(self) -> EventLog:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
