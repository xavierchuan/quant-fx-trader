#!/usr/bin/env python3
"""Summarize risk events logged by simulate_execution or live adapters."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
import os
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate risk event logs")
    parser.add_argument("--log", default="results/risk/events.jsonl", help="Path to risk events JSONL")
    parser.add_argument("--out", default="results/risk/report.csv", help="CSV summary output")
    parser.add_argument("--run-id", help="Run identifier for metrics logging (defaults to $RUN)")
    parser.add_argument("--metrics-path", default="results/risk/metrics.csv", help="Aggregate metrics CSV")
    parser.add_argument(
        "--status",
        default="unknown",
        help="Execution status recorded in metrics (e.g., pending/pass/fail).",
    )
    parser.add_argument("--skip-report", action="store_true", help="Do not write the detailed CSV report.")
    parser.add_argument("--skip-metrics", action="store_true", help="Do not append to metrics CSV.")
    return parser.parse_args()


def load_events(path: Path) -> list[dict]:
    if not path.exists():
        return []
    events = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def main() -> None:
    args = parse_args()
    path = Path(args.log)
    events = load_events(path)
    if events:
        df = pd.DataFrame(events)
        counts = Counter(df["event"].fillna("unknown"))
        print("Event counts:")
        for event, count in counts.items():
            print(f"  {event}: {count}")
    else:
        print(f"No events found in {path}")
        df = pd.DataFrame(columns=["event", "ts", "symbol", "strategy", "reason"])
        counts = Counter()
    if not args.skip_report:
        df.to_csv(args.out, index=False)
        if events:
            print(f"Detailed report saved to {args.out}")
        else:
            print(f"Empty report written to {args.out}")

    run_id = args.run_id or os.environ.get("RUN")
    if not run_id:
        print("Run ID not provided; skipping metrics append.")
        return
    if args.skip_metrics:
        print("skip-metrics enabled; metrics append skipped.")
        return
    reject_count = counts.get("reject", 0)
    kill_count = counts.get("kill_switch", 0)
    exec_dir = Path(f"results/execution/{run_id}")
    append_metrics(Path(args.metrics_path), run_id, reject_count, kill_count, status=args.status, exec_dir=exec_dir)


def append_metrics(path: Path, run_id: str, rejects: int, kills: int, status: str = "unknown", exec_dir: Path | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    timestamp = datetime.now(timezone.utc).isoformat()
    latency_avg = latency_p95 = total_pnl = max_drawdown = 0.0
    max_gross = max_symbol = 0.0
    if exec_dir and (exec_dir / "fills.csv").exists():
        df = pd.read_csv(exec_dir / "fills.csv")
        if "adapter_latency_ms" in df.columns and not df["adapter_latency_ms"].dropna().empty:
            series = df["adapter_latency_ms"].fillna(0.0)
            latency_avg = float(series.mean())
            latency_p95 = float(series.quantile(0.95))
        if "pnl" in df.columns:
            total_pnl = float(df["pnl"].sum())
    if exec_dir and (exec_dir / "sim_results.json").exists():
        data = json.loads((exec_dir / "sim_results.json").read_text(encoding="utf-8"))
        max_gross = float(data.get("max_gross_notional", 0.0))
        symbol_peaks = data.get("max_symbol_exposure", {})
        if isinstance(symbol_peaks, dict) and symbol_peaks:
            max_symbol = float(max(abs(v) for v in symbol_peaks.values()))
        max_drawdown = float(data.get("max_drawdown_pct", 0.0))
    with path.open("a", encoding="utf-8") as fh:
        if is_new:
            fh.write("timestamp,run_id,rejects,kills,status,latency_ms_avg,latency_ms_p95,total_pnl,max_gross_notional,max_symbol_exposure,max_drawdown_pct\n")
        fh.write(f"{timestamp},{run_id},{rejects},{kills},{status},{latency_avg},{latency_p95},{total_pnl},{max_gross},{max_symbol},{max_drawdown}\n")


if __name__ == "__main__":
    main()
