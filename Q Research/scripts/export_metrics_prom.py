#!/usr/bin/env python3
"""
Export latest risk metrics row in Prometheus exposition format.

Usage:
  python scripts/export_metrics_prom.py --csv results/risk/metrics.csv --job risk_sim
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export risk metrics to Prometheus format.")
    parser.add_argument("--csv", default="results/risk/metrics.csv")
    parser.add_argument("--job", default="risk_sim")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.csv)
    if not path.exists():
        raise SystemExit(f"metrics CSV not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise SystemExit("metrics CSV is empty.")
    latest = df.tail(1).iloc[0]
    run_id = latest.get("run_id", "unknown")
    def value(key: str, default: float = 0.0) -> float:
        val = latest.get(key, default)
        if isinstance(val, str) and not val:
            return default
        try:
            if pd.isna(val):
                return default
        except TypeError:
            pass
        return val

    metrics = {
        "risk_rejects": value("rejects", 0),
        "risk_kills": value("kills", 0),
        "risk_latency_ms_avg": value("latency_ms_avg"),
        "risk_latency_ms_p95": value("latency_ms_p95"),
        "risk_total_pnl": value("total_pnl"),
        "risk_max_gross_notional": value("max_gross_notional"),
        "risk_max_symbol_exposure": value("max_symbol_exposure"),
        "risk_max_drawdown_pct": value("max_drawdown_pct"),
        "risk_live_sharpe_30d": value("rolling_sharpe_30d"),
        "risk_live_drawdown_pct": value("live_drawdown_pct"),
        "risk_live_latency_ms_p95": value("live_latency_ms_p95"),
        "risk_slippage_bps": value("slippage_bps"),
    }
    labels = f'run="{run_id}",job="{args.job}"'
    for name, value in metrics.items():
        print(f'{name}{{{labels}}} {value}')


if __name__ == "__main__":
    main()
