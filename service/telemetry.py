"""Structured logging, optional CloudWatch metrics and optional S3 upload."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOGGER_NAME = "aimocap.service"


class JsonFormatter(logging.Formatter):
    """One JSON object per log line, convenient for CloudWatch Logs Insights."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "msg": record.getMessage(),
        }
        extra = getattr(record, "fields", None)
        if isinstance(extra, dict):
            payload.update(extra)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def setup_logging(log_dir: Path) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    formatter = JsonFormatter()
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_dir / "service.log", encoding="utf-8")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    except OSError:
        logger.warning("cannot write service.log", extra={"fields": {"log_dir": str(log_dir)}})
    return logger


def log_event(message: str, **fields: Any) -> None:
    logging.getLogger(LOGGER_NAME).info(message, extra={"fields": fields})


def read_events(path: Path) -> list[dict[str, Any]]:
    """Parse ``events.jsonl``; incomplete trailing lines are ignored."""
    events = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return events
    for line in lines:
        try:
            item = json.loads(line)
        except ValueError:
            continue
        if isinstance(item, dict):
            events.append(item)
    return events


def publish_run_metrics(events: list[dict[str, Any]], run_seconds: float, failed: bool) -> None:
    """Send per-stage and per-run timings to CloudWatch (namespace ``AIMoCap``)."""
    try:
        import boto3

        data = [
            {"MetricName": "RunSeconds", "Value": float(run_seconds), "Unit": "Seconds"},
            {"MetricName": "RunFailed", "Value": 1.0 if failed else 0.0, "Unit": "Count"},
        ]
        for event in events:
            if event.get("status") == "completed" and event.get("seconds") is not None:
                data.append(
                    {
                        "MetricName": "StageSeconds",
                        "Dimensions": [{"Name": "Stage", "Value": str(event.get("stage"))}],
                        "Value": float(event["seconds"]),
                        "Unit": "Seconds",
                    }
                )
        client = boto3.client("cloudwatch")
        for start in range(0, len(data), 20):
            client.put_metric_data(Namespace="AIMoCap", MetricData=data[start : start + 20])
    except Exception as exc:  # telemetry must never break a run
        log_event("cloudwatch publish failed", error=str(exc))


def upload_run(run_dir: Path, bucket: str, files: list[str]) -> None:
    """Copy the result files of a run to ``s3://bucket/runs/<id>/``."""
    try:
        import boto3

        client = boto3.client("s3")
        for name in files:
            path = run_dir / name
            if path.is_file():
                client.upload_file(str(path), bucket, f"runs/{run_dir.name}/{name}")
    except Exception as exc:
        log_event("s3 upload failed", run=run_dir.name, error=str(exc))
