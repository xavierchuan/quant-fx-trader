"""Append execution metrics to CSV for monitoring."""

from __future__ import annotations

import csv
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class ExecutionMetric:
    event: str
    order_id: str
    symbol: str
    latency_ms: Optional[float]
    status: str
    timestamp: str


class MetricsLogger:
    def __init__(self, path: Optional[str] = None):
        metrics_path = path or os.environ.get("EXECUTION_METRICS_PATH", "metrics/execution.csv")
        self.path = Path(metrics_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with self.path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(ExecutionMetric.__annotations__.keys()))
                writer.writeheader()

    def log(self, metric: ExecutionMetric) -> None:
        with self.path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(ExecutionMetric.__annotations__.keys()))
            writer.writerow(asdict(metric))


def log_event(event: str, order_id: str, symbol: str, status: str, start_ts: datetime, end_ts: datetime) -> None:
    latency_ms = (end_ts - start_ts).total_seconds() * 1000.0
    logger = MetricsLogger()
    logger.log(
        ExecutionMetric(
            event=event,
            order_id=order_id,
            symbol=symbol,
            latency_ms=latency_ms,
            status=status,
            timestamp=end_ts.isoformat(),
        )
    )
