"""Simple JSONL order store for audit/replay."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from QuantTrader.execution.adapter import OrderParams


class OrderStore:
    def __init__(self, path: str = "results/execution/orders.log"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, order_id: str, params: OrderParams) -> None:
        record = {"order_id": order_id, **asdict(params)}
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

    def load(self) -> Iterable[dict]:
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                yield json.loads(line)
