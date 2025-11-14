#!/usr/bin/env python3
"""
Validate a backtest result folder produced by run_once/grid/batch.

Checks:
  - summary.json / metrics.json exist
  - core KPI fields present and not None
  - data report metadata available + referenced file存在（可选）
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

BASE_DIR = Path(__file__).resolve().parents[1]

REQUIRED_FILES = ["summary.json", "metrics.json"]
KPI_KEYS = [
    "final_equity",
    "ann_return",
    "ann_vol",
    "sharpe",
    "sortino",
    "calmar",
    "max_drawdown",
    "max_drawdown_duration_bars",
    "recovery_time_bars",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate contents of a results/<run_id> directory.")
    parser.add_argument("path", help="Path to results/<run_id> directory.")
    parser.add_argument(
        "--require-data-report",
        action="store_true",
        help="Fail if data_report metadata or referenced file is missing.",
    )
    return parser.parse_args()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def main() -> None:
    args = parse_args()
    run_path = Path(args.path).expanduser().resolve()
    if not run_path.exists():
        print(f"[ERROR] Run directory not found: {run_path}", file=sys.stderr)
        sys.exit(1)

    errors: List[str] = []

    files = {}
    for name in REQUIRED_FILES:
        file_path = run_path / name
        if not file_path.exists():
            errors.append(f"Missing required file: {file_path}")
        else:
            files[name] = load_json(file_path)

    summary = files.get("summary.json") or {}
    metrics = files.get("metrics.json") or summary.get("metrics") or {}

    for key in KPI_KEYS:
        if key not in metrics or metrics[key] is None:
            errors.append(f"Metric '{key}' missing or null in metrics.json")

    if args.require_data_report:
        data_meta = summary.get("data_report")
        if not data_meta:
            errors.append("data_report metadata missing in summary.json")
        else:
            if data_meta.get("severity") is None:
                errors.append("data_report.severity missing in summary.json")
        report_path = metrics.get("data_report")
        if not report_path:
            errors.append("metrics.data_report missing (expected relative path to JSON)")
        else:
            report_path_obj = Path(report_path)
            report_full = report_path_obj if report_path_obj.is_absolute() else (BASE_DIR / report_path_obj).resolve()
            if not report_full.exists():
                errors.append(f"Referenced data report not found: {report_path}")

    if errors:
        print("[FAIL] Result validation failed:")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)

    print(f"[OK] Result folder {run_path.name} passed validation.")


if __name__ == "__main__":
    main()
