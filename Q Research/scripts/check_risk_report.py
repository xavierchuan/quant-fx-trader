#!/usr/bin/env python3
"""Fail CI if risk report exceeds thresholds."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate risk report thresholds")
    parser.add_argument("--report", default="results/risk/report.csv", help="CSV produced by risk_report.py")
    parser.add_argument("--max-rejects", type=int, default=0)
    parser.add_argument("--max-kill", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = Path(args.report)
    if not report.exists():
        raise SystemExit(f"Report not found: {report}")
    rejects = 0
    kills = 0
    with report.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row.get("event") == "reject":
                rejects += 1
            elif row.get("event") == "kill_switch":
                kills += 1
    if rejects > args.max_rejects or kills > args.max_kill:
        raise SystemExit(
            f"Risk report exceeded thresholds: rejects={rejects} (max {args.max_rejects}), "
            f"kill_switch={kills} (max {args.max_kill})"
        )
    print(f"Risk report OK (rejects={rejects}, kill_switch={kills})")


if __name__ == "__main__":
    main()
