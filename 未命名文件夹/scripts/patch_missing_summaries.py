#!/usr/bin/env python3
"""
Generate minimal summary.json and placeholder trades for runs that lack metadata.
"""

from __future__ import annotations

import argparse
import csv
import json
import yaml
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
TRADES_DIR = ROOT / "data" / "outputs" / "trades"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Patch runs missing summary.json artifacts.")
    parser.add_argument("--runs", required=True, help="Comma-separated run IDs (e.g. 20251106_142210,20251106_142422)")
    parser.add_argument("--symbol", default="EURUSD", help="Fallback symbol when unknown.")
    parser.add_argument("--csv-path", default=None, help="Optional CSV path override (defaults to data/raw/<symbol>_H1.csv).")
    return parser.parse_args()


def ensure_placeholder_trades(run_id: str, symbol: str) -> str:
    TRADES_DIR.mkdir(parents=True, exist_ok=True)
    path = TRADES_DIR / f"trades_PLACEHOLDER_{run_id}.csv"
    if path.exists():
        return str(path.as_posix())
    timestamp = datetime.strptime(run_id, "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)
    row = [
        timestamp.isoformat(),
        timestamp.isoformat(),
        symbol,
        "long",
        "0",
        "1.0",
        "1.0",
        "0.0",
        "default",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["ts_entry", "ts_exit", "symbol", "direction", "qty", "price_entry", "exit", "pnl", "strategy"])
        writer.writerow(row)
    return str(path.as_posix())


def load_performance(run_dir: Path) -> Dict[str, float]:
    perf_path = run_dir / "performance.yml"
    if not perf_path.exists():
        return {}
    data = yaml.safe_load(perf_path.read_text(encoding="utf-8")) or {}
    mapping = {
        "annualized_return": "ann_return",
        "volatility": "ann_vol",
        "sharpe_ratio": "sharpe",
        "max_drawdown": "max_drawdown",
        "total_return": "total_return",
    }
    metrics: Dict[str, float] = {}
    for src, dest in mapping.items():
        value = data.get(src)
        if value is not None:
            metrics[dest] = value
    return metrics


def write_summary(run_id: str, symbol: str, csv_path: str) -> None:
    run_dir = RESULTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "summary.json"
    metrics_path = run_dir / "metrics.json"
    equity_path = run_dir / "equity_curve.csv"
    trades_path = ensure_placeholder_trades(run_id, symbol)

    perf_metrics = load_performance(run_dir)
    timestamp = datetime.strptime(run_id, "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc).isoformat()

    summary = {
        "run_id": run_id,
        "timestamp": timestamp,
        "symbol": symbol,
        "csv_path": csv_path,
        "parameters": {"note": "Placeholder summary generated via patch_missing_summaries.py"},
        "metrics": perf_metrics,
        "data_report": {"severity": "unknown", "messages": []},
        "artifacts": {
            "equity": str(equity_path.relative_to(ROOT)) if equity_path.exists() else "",
            "trades": trades_path,
            "trade_stats": "",
        },
    }

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    metrics_path.write_text(json.dumps(perf_metrics, indent=2), encoding="utf-8")
    print(f"Patched summary for {run_id}")


def main() -> None:
    args = parse_args()
    runs: List[str] = [item.strip() for item in args.runs.split(",") if item.strip()]
    if not runs:
        raise SystemExit("No runs provided.")
    csv_path = args.csv_path or f"data/raw/{args.symbol}_H1.csv"
    for run_id in runs:
        summary_file = RESULTS_DIR / run_id / "summary.json"
        if summary_file.exists():
            print(f"{run_id}: summary already exists, skipping.")
            continue
        write_summary(run_id, args.symbol, csv_path)


if __name__ == "__main__":
    main()
