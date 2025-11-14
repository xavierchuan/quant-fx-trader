#!/usr/bin/env python3
"""
Backfill results/risk/metrics.csv using historical execution runs.

For each run under results/execution/<run_id>/, the script inspects
sim_results.json, derives reject/kill counts, infers status (pass/fail),
and appends any missing records to results/risk/metrics.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
EXECUTION_DIR = ROOT / "results" / "execution"
METRICS_PATH = ROOT / "results" / "risk" / "metrics.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill risk metrics from historical execution runs.")
    parser.add_argument(
        "--runs",
        type=str,
        help="Comma-separated run IDs to backfill. Defaults to all directories in results/execution/.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-write entries even if run_id already exists in metrics.csv (otherwise skipped).",
    )
    return parser.parse_args()


def list_runs(filter_ids: Optional[List[str]]) -> List[str]:
    if filter_ids:
        return filter_ids
    if not EXECUTION_DIR.exists():
        return []
    return sorted([p.name for p in EXECUTION_DIR.iterdir() if p.is_dir()])


def load_sim_results(run_id: str) -> Optional[Dict]:
    sim_path = EXECUTION_DIR / run_id / "sim_results.json"
    if not sim_path.exists():
        return None
    return json.loads(sim_path.read_text(encoding="utf-8"))


def infer_timestamp(run_id: str) -> str:
    summary_path = ROOT / "results" / run_id / "summary.json"
    if summary_path.exists():
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
            if "created_at" in data:
                return data["created_at"]
        except json.JSONDecodeError:
            pass
    sim_path = EXECUTION_DIR / run_id / "sim_results.json"
    if sim_path.exists():
        return datetime.fromtimestamp(sim_path.stat().st_mtime, tz=timezone.utc).isoformat()
    return datetime.now(timezone.utc).isoformat()


def summarize_run(run_id: str) -> Optional[Dict[str, str]]:
    payload = load_sim_results(run_id)
    if payload is None:
        return None
    rejects = len(payload.get("rejects", []))
    kills = len(payload.get("kill_switch_events", []))
    status = "pass" if rejects == 0 and kills == 0 else "fail"
    return {
        "timestamp": infer_timestamp(run_id),
        "run_id": run_id,
        "rejects": str(rejects),
        "kills": str(kills),
        "status": status,
    }


def load_existing() -> Dict[str, Dict[str, str]]:
    if not METRICS_PATH.exists():
        return {}
    with METRICS_PATH.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return {row["run_id"]: row for row in reader if row.get("run_id")}


def append_rows(rows: List[Dict[str, str]], replace: bool = False) -> None:
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing = load_existing()
    if replace:
        for row in rows:
            existing[row["run_id"]] = row
        with METRICS_PATH.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["timestamp", "run_id", "rejects", "kills", "status"])
            writer.writeheader()
            for row in sorted(existing.values(), key=lambda r: r["timestamp"]):
                writer.writerow(row)
    else:
        new_file = not METRICS_PATH.exists()
        with METRICS_PATH.open("a", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["timestamp", "run_id", "rejects", "kills", "status"])
            if new_file:
                writer.writeheader()
            for row in rows:
                writer.writerow(row)


def main() -> None:
    args = parse_args()
    runs = list_runs(args.runs.split(",") if args.runs else None)
    if not runs:
        print("No runs to process.")
        return
    existing = load_existing()
    new_rows: List[Dict[str, str]] = []
    replacements: List[Dict[str, str]] = []
    for run_id in runs:
        record = summarize_run(run_id)
        if not record:
            print(f"[skip] sim_results.json missing for {run_id}")
            continue
        if run_id in existing and not args.force:
            print(f"[skip] {run_id} already in metrics.csv")
            continue
        if args.force and run_id in existing:
            replacements.append(record)
        else:
            new_rows.append(record)
    if replacements:
        append_rows(replacements, replace=True)
        print(f"Replaced {len(replacements)} entries (force mode).")
    if new_rows:
        append_rows(new_rows, replace=False)
        print(f"Appended {len(new_rows)} new entries.")
    if not replacements and not new_rows:
        print("No changes made.")


if __name__ == "__main__":
    main()
