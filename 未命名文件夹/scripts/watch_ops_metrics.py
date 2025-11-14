#!/usr/bin/env python3
"""
Run multiple metric validators (risk, latency, pnl) in a single command.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch aggregated risk/ops metrics.")
    parser.add_argument("--csv", default="results/risk/metrics.csv", help="Metrics CSV path.")
    parser.add_argument("--max-rejects", type=int, default=0)
    parser.add_argument("--max-latency-ms", type=float, default=500.0)
    parser.add_argument("--min-pnl", type=float, default=-3000.0)
    parser.add_argument("--max-exposure", type=float, default=2_000_000.0)
    parser.add_argument("--max-drawdown", type=float, default=0.1)
    parser.add_argument("--min-live-sharpe", type=float, default=1.4)
    parser.add_argument("--max-live-drawdown", type=float, default=0.05)
    parser.add_argument("--max-live-latency-ms", type=float, default=500.0)
    parser.add_argument("--max-slippage-bps", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"metrics CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if df.empty:
        raise SystemExit("metrics CSV is empty.")
    latest = df.tail(1).iloc[0]
    rejects = latest.get("rejects", 0)
    status = str(latest.get("status", "unknown")).lower()
    latency = latest.get("latency_ms_avg", 0.0)
    pnl = latest.get("total_pnl", 0.0)
    max_exposure = latest.get("max_symbol_exposure", 0.0)
    drawdown = latest.get("max_drawdown_pct", 0.0)
    live_sharpe = latest.get("rolling_sharpe_30d", float("nan"))
    live_drawdown = latest.get("live_drawdown_pct", float("nan"))
    live_latency = latest.get("live_latency_ms_p95", float("nan"))
    slippage_bps = latest.get("slippage_bps", float("nan"))

    errors = []
    if rejects > args.max_rejects or status != "pass":
        errors.append(f"Rejects/status violation (rejects={rejects}, status={status})")
    if latency > args.max_latency_ms:
        errors.append(f"Latency {latency:.1f}ms > threshold {args.max_latency_ms}")
    if pnl < args.min_pnl:
        errors.append(f"Total PnL {pnl:.2f} < min {args.min_pnl}")
    if max_exposure > args.max_exposure:
        errors.append(f"Exposure {max_exposure:.2f} > max {args.max_exposure}")
    if drawdown > args.max_drawdown:
        errors.append(f"Drawdown {drawdown:.3f} > max {args.max_drawdown}")
    if not pd.isna(live_sharpe) and live_sharpe < args.min_live_sharpe:
        errors.append(f"Live Sharpe {live_sharpe:.2f} < min {args.min_live_sharpe}")
    if not pd.isna(live_drawdown) and live_drawdown > args.max_live_drawdown:
        errors.append(f"Live drawdown {live_drawdown:.3f} > max {args.max_live_drawdown}")
    if not pd.isna(live_latency) and live_latency > args.max_live_latency_ms:
        errors.append(f"Live latency p95 {live_latency:.1f}ms > max {args.max_live_latency_ms}")
    if not pd.isna(slippage_bps) and slippage_bps > args.max_slippage_bps:
        errors.append(f"Slippage {slippage_bps:.2f}bps > max {args.max_slippage_bps}")
    print(
        f"[watch_ops_metrics] run={latest.get('run_id')} status={status} "
        f"rejects={rejects} latency_avg={latency} pnl={pnl} exposure={max_exposure} drawdown={drawdown} "
        f"live_sharpe={live_sharpe} live_drawdown={live_drawdown} live_latency_p95={live_latency} slippage_bps={slippage_bps}"
    )
    if errors:
        raise SystemExit("; ".join(errors))


if __name__ == "__main__":
    main()
