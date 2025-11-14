#!/usr/bin/env python3
"""
Aggregate data-quality reports (results/data_quality/*.json) into a tabular summary.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict


DEFAULT_DIR = Path("results/data_quality")
DEFAULT_OUTPUT = Path("metrics/data_quality_summary.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate data-quality reports.")
    parser.add_argument("--input", default=str(DEFAULT_DIR), help="Directory containing data_quality JSON reports.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="CSV file to write summary (default metrics/data_quality_summary.csv).")
    parser.add_argument("--print", action="store_true", help="Print summary to stdout.")
    return parser.parse_args()


def load_reports(report_dir: Path) -> List[Dict]:
    rows: List[Dict] = []
    if not report_dir.exists():
        return rows
    for path in sorted(report_dir.glob("*.json")):
        try:
            data = json.load(path.open("r", encoding="utf-8"))
        except Exception:
            continue
        manifest = data.get("manifest") or {}
        dataset_path = data.get("dataset_path") or manifest.get("path")
        rows.append({
            "generated_at": data.get("generated_at"),
            "dataset_path": dataset_path,
            "symbol": infer_symbol(path.name, dataset_path),
            "severity": data.get("severity"),
            "gap_ratio": data.get("gap_ratio"),
            "duplicate_timestamps": data.get("duplicate_timestamps"),
            "null_max": max((data.get("null_counts") or {}).values() or [0]),
            "outlier_columns": ", ".join((data.get("numeric_outliers") or {}).keys()),
            "hash": manifest.get("sha256"),
            "report_file": str(path),
        })
    return rows


def infer_symbol(filename: str, dataset_path: str | None) -> str:
    if dataset_path:
        stem = Path(dataset_path).stem
        parts = stem.split("_")
        if parts:
            return parts[0]
    if "_" in filename:
        return filename.split("_")[1]
    return "UNKNOWN"


def write_csv(rows: List[Dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    headers = ["generated_at", "dataset_path", "symbol", "severity", "gap_ratio", "duplicate_timestamps", "null_max", "outlier_columns", "hash", "report_file"]
    with output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def print_table(rows: List[Dict]) -> None:
    if not rows:
        print("No reports found.")
        return
    print(f"{'Generated':25} {'Symbol':8} {'Severity':7} {'Gap%':7} {'Dup':5} {'Hash':64}")
    for row in rows:
        gap = f"{row['gap_ratio']:.4f}" if isinstance(row["gap_ratio"], (int, float)) else "n/a"
        print(
            f"{row['generated_at'][:23] if row['generated_at'] else '':25} "
            f"{row['symbol']:8} {row['severity']:7} {gap:7} "
            f"{row['duplicate_timestamps']!s:5} {row['hash'] or ''}"
        )


def main() -> None:
    args = parse_args()
    report_dir = Path(args.input)
    rows = load_reports(report_dir)
    if args.print:
        print_table(rows)
    output_path = Path(args.output)
    write_csv(rows, output_path)
    print(f"Wrote summary to {output_path} ({len(rows)} rows).")


if __name__ == "__main__":
    main()
