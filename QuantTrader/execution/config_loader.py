"""Helpers to load execution adapter configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def load_oanda_config(path: Optional[str] = None) -> Dict[str, Any]:
    cfg_path = Path(path or "QuantTrader/config/execution_oanda.yaml")
    if not cfg_path.exists():
        raise FileNotFoundError(f"OANDA config not found: {cfg_path}")
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    return {
        "account_id": data.get("account_id"),
        "base_url": data.get("base_url", "https://api-fxpractice.oanda.com/v3"),
        "timeout_ms": data.get("timeout_ms", 10000),
        "retry_backoff": data.get("retry_backoff", 1.0),
        "max_retries": data.get("max_retries", 3),
        "metrics_path": data.get("metrics_path"),
        "error_log": data.get("error_log", "results/execution/errors.log"),
    }
