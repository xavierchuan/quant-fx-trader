#!/usr/bin/env python3
"""
Watchdog script for risk metrics.

If the latest entry in results/risk/metrics.csv has status=fail or rejects>0,
exit with non-zero status (can be used in cron/CI).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check latest risk metrics entry.")
    parser.add_argument("--csv", default="results/risk/metrics.csv", help="Metrics CSV path.")
    parser.add_argument("--allow-rejects", type=int, default=0, help="Maximum rejects allowed before alert.")
    parser.add_argument("--allow-status", choices=["pass", "any"], default="pass", help="Expected status for latest run.")
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
    rejects = int(latest.get("rejects", 0))
    status = str(latest.get("status", "unknown")).lower()

    ok = True
    messages = []
    if rejects > args.allow_rejects:
        ok = False
        messages.append(f"Rejects={rejects} exceed allow_rejects={args.allow_rejects}")
    if args.allow_status == "pass" and status != "pass":
        ok = False
        messages.append(f"Status={status}")

    print(
        f"[watch_risk_metrics] latest run={latest.get('run_id')} "
        f"status={status} rejects={rejects} kills={latest.get('kills')}"
    )
    if not ok:
        raise SystemExit(" ; ".join(messages))


if __name__ == "__main__":
    main()
