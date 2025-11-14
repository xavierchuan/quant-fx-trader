#!/usr/bin/env python3
"""
Automated OANDA ingestion with retries, logging, and manifest refresh.

- Accepts direct CLI arguments (symbol/granularity/days/target count/output).
- Or accepts a schedule YAML describing multiple jobs:
    - symbol: EUR_USD
    - granularity: H1
    - days: 365
    - target_count: 8000
    - output: data/raw/EURUSD_H1.csv  (optional; defaults to auto path)

Each ingestion:
  * retries on failure with exponential backoff
  * appends a JSON line to metrics/ingest.log
  * updates metrics/ingest_status.json with latest state per (symbol, granularity)
  * triggers build_dataset_manifest.py to refresh hashes.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from scripts.get_candles import get_candles, default_out_csv, normalize_to_oanda  # noqa

METRICS_DIR = PROJECT_ROOT / "metrics"
METRICS_DIR.mkdir(exist_ok=True)
INGEST_LOG = METRICS_DIR / "ingest.log"
INGEST_STATUS = METRICS_DIR / "ingest_status.json"
INGEST_METRICS = METRICS_DIR / "ingest_metrics.csv"

MANIFEST_SCRIPT = PROJECT_ROOT / "scripts" / "build_dataset_manifest.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest OANDA candles into data/raw.")
    parser.add_argument("--symbol", help="Instrument symbol (e.g., EUR_USD)")
    parser.add_argument("--granularity", default="H1", help="OANDA granularity (default H1)")
    parser.add_argument("--days", type=int, default=365, help="Lookback days for ingestion")
    parser.add_argument("--target-count", type=int, default=None, help="Optional target bar count")
    parser.add_argument("--output", help="Explicit CSV output path")
    parser.add_argument("--retries", type=int, default=3, help="Number of retries on failure")
    parser.add_argument("--backoff", type=float, default=15.0, help="Base backoff seconds between retries")
    parser.add_argument("--schedule", help="YAML file listing ingestion jobs")
    parser.add_argument("--manifest-output", default="data/_manifest.json", help="Manifest path to refresh")
    return parser.parse_args()


def load_schedule(path: str) -> List[Dict]:
    schedule_path = PROJECT_ROOT / path
    if not schedule_path.exists():
        raise FileNotFoundError(f"Schedule file not found: {schedule_path}")
    with schedule_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or []
    if not isinstance(data, list):
        raise ValueError("Schedule YAML must be a list of jobs.")
    return data


def resolve_output(symbol: str, granularity: str, explicit: Optional[str]) -> Path:
    if explicit:
        return (PROJECT_ROOT / explicit).resolve()
    norm = normalize_to_oanda(symbol)
    return Path(default_out_csv(str(PROJECT_ROOT), norm, granularity))


def log_ingest(entry: Dict) -> None:
    line = json.dumps(entry, ensure_ascii=False)
    with INGEST_LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")

    existing = {}
    if INGEST_STATUS.exists():
        try:
            existing = json.load(INGEST_STATUS.open("r", encoding="utf-8")) or {}
        except Exception:
            existing = {}
    key = f"{entry['symbol']}::{entry['granularity']}"
    existing[key] = entry
    with INGEST_STATUS.open("w", encoding="utf-8") as fh:
        json.dump(existing, fh, indent=2, ensure_ascii=False)

    append_ingest_metric(entry)


def append_ingest_metric(entry: Dict) -> None:
    headers = ["timestamp", "symbol", "granularity", "status", "rows", "duration_sec"]
    INGEST_METRICS.parent.mkdir(exist_ok=True)
    write_header = not INGEST_METRICS.exists()
    with INGEST_METRICS.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if write_header:
            writer.writerow(headers)
        writer.writerow([
            entry.get("timestamp"),
            entry.get("symbol"),
            entry.get("granularity"),
            entry.get("status"),
            entry.get("rows"),
            entry.get("duration_sec"),
        ])


def run_job(job: Dict, retries: int, backoff: float) -> bool:
    symbol = job["symbol"]
    granularity = job.get("granularity", "H1")
    days = int(job.get("days", 365))
    target_count = job.get("target_count")
    target_count = int(target_count) if target_count else None
    output_path = resolve_output(symbol, granularity, job.get("output"))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    attempt = 0
    start_time = time.time()
    errors = []
    while attempt <= retries:
        attempt += 1
        try:
            logger.info(f"[INGEST] {symbol} {granularity} attempt {attempt}/{retries+1} (days={days})")
            df = get_candles(symbol=symbol, granularity=granularity, start_days_ago=days, target_count=target_count)
            if df.empty:
                raise RuntimeError("Fetched DataFrame is empty.")
            df.to_csv(output_path, index=False)
            duration = time.time() - start_time
            entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "symbol": symbol,
                "granularity": granularity,
                "rows": int(len(df)),
                "output": str(output_path.relative_to(PROJECT_ROOT)),
                "status": "success",
                "duration_sec": round(duration, 2),
            }
            log_ingest(entry)
            logger.info(f"[INGEST] Success {symbol} {granularity}: {len(df)} rows -> {output_path}")
            return True
        except Exception as exc:
            err_msg = f"{type(exc).__name__}: {exc}"
            errors.append(err_msg)
            logger.error(f"[INGEST] Failed {symbol} {granularity} attempt {attempt}: {err_msg}")
            if attempt <= retries:
                sleep_time = backoff * attempt
                logger.info(f"[INGEST] Sleeping {sleep_time:.1f}s before retry...")
                time.sleep(sleep_time)

    duration = time.time() - start_time
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol,
        "granularity": granularity,
        "rows": None,
        "output": str(output_path.relative_to(PROJECT_ROOT)),
        "status": "failure",
        "duration_sec": round(duration, 2),
        "errors": errors[-retries:],
    }
    log_ingest(entry)
    return False


def refresh_manifest(manifest_output: str) -> None:
    logger.info("[MANIFEST] Refreshing dataset manifest…")
    cmd = [
        sys.executable,
        str(MANIFEST_SCRIPT),
        "--dirs",
        "data/raw",
        "data/derived",
        "--output",
        manifest_output,
    ]
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    args = parse_args()
    jobs: List[Dict]
    if args.schedule:
        schedule_jobs = load_schedule(args.schedule)
        jobs = schedule_jobs
    elif args.symbol:
        jobs = [{
            "symbol": args.symbol,
            "granularity": args.granularity,
            "days": args.days,
            "target_count": args.target_count,
            "output": args.output,
        }]
    else:
        raise ValueError("Either --symbol or --schedule must be provided.")

    successes = 0
    for job in jobs:
        if run_job(job, args.retries, args.backoff):
            successes += 1

    if successes > 0:
        refresh_manifest(args.manifest_output)
    else:
        logger.warning("No successful ingestions; manifest not updated.")


if __name__ == "__main__":
    main()
