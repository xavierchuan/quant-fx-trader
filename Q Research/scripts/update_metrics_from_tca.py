#!/usr/bin/env python3
"""
Append a metrics row using TCA summary + KPI overrides.

Example:
  python scripts/update_metrics_from_tca.py \
    --tca QuantTrader/results/execution/tca_summary.json \
    --run-id 20251112_parallel_demo_live \
    --status pass --latency-avg 28 --latency-p95 45 \
    --total-pnl 27.3 --max-exposure 500000 --max-drawdown 0.02 \
    --rolling-sharpe 1.5 --live-drawdown 0.03 \
    --live-latency-p95 45 --slippage-bps 1.2
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Append metrics row from TCA summary.")
    parser.add_argument("--tca", required=True, help="Path to tca_summary.json")
    parser.add_argument("--metrics", default="results/risk/metrics.csv", help="Metrics CSV path")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--status", default="pass")
    parser.add_argument("--latency-avg", type=float, required=True)
    parser.add_argument("--latency-p95", type=float, required=True)
    parser.add_argument("--total-pnl", type=float, required=True)
    parser.add_argument("--max-exposure", type=float, required=True)
    parser.add_argument("--max-drawdown", type=float, required=True)
    parser.add_argument("--rolling-sharpe", type=float, required=True)
    parser.add_argument("--live-drawdown", type=float, required=True)
    parser.add_argument("--live-latency-p95", type=float, required=True)
    parser.add_argument("--slippage-bps", type=float, required=True)
    parser.add_argument("--rejects", type=int, default=0)
    parser.add_argument("--kills", type=int, default=0)
    parser.add_argument("--max-gross-notional", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tca_path = Path(args.tca)
    if not tca_path.exists():
        raise SystemExit(f"TCA summary not found: {tca_path}")
    with tca_path.open("r", encoding="utf-8") as f:
        tca = json.load(f)

    timestamp = datetime.now(timezone.utc).isoformat()
    row = {
        "timestamp": timestamp,
        "run_id": args.run_id,
        "rejects": args.rejects,
        "kills": args.kills,
        "status": args.status,
        "latency_ms_avg": args.latency_avg,
        "latency_ms_p95": args.latency_p95,
        "total_pnl": args.total_pnl,
        "max_gross_notional": args.max_gross_notional or args.max_exposure,
        "max_symbol_exposure": args.max_exposure,
        "max_drawdown_pct": args.max_drawdown,
        "rolling_sharpe_30d": args.rolling_sharpe,
        "live_drawdown_pct": args.live_drawdown,
        "live_latency_ms_p95": args.live_latency_p95,
        "slippage_bps": args.slippage_bps,
        "paper_trade_count": tca.get("paper_trade_count"),
        "paper_total_pnl": tca.get("paper_total_pnl"),
        "live_trade_count": tca.get("live_trade_count"),
        "live_total_pnl": tca.get("live_total_pnl"),
        "pnl_diff_mean": tca.get("pnl_diff_mean"),
        "pnl_diff_std": tca.get("pnl_diff_std"),
    }

    metrics_path = Path(args.metrics)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not metrics_path.exists()
    with metrics_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"Appended metrics row for run={args.run_id}")


if __name__ == "__main__":
    main()
