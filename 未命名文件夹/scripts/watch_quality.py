#!/usr/bin/env python3
"""
Scan recent data-quality reports and raise alerts when severity >= warn.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Dict
from urllib import request, error

REPORT_DIR = Path("results/data_quality")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch data-quality reports.")
    parser.add_argument("--hours", type=float, default=24.0, help="Lookback window in hours (default 24).")
    parser.add_argument("--report-dir", default=str(REPORT_DIR), help="Directory with data quality JSON files.")
    parser.add_argument("--webhook-url", help="Optional webhook URL for POST notifications.")
    return parser.parse_args()


def load_recent_reports(report_dir: Path, min_ts: datetime) -> List[Dict]:
    rows: List[Dict] = []
    if not report_dir.exists():
        return rows
    for path in report_dir.glob("*.json"):
        try:
            data = json.load(path.open("r", encoding="utf-8"))
        except Exception:
            continue
        ts_raw = data.get("generated_at")
        if not ts_raw:
            continue
        ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
        if ts >= min_ts:
            data["__file"] = str(path)
            rows.append(data)
    return rows


def notify(webhook: str, message: str) -> None:
    payload = json.dumps({"text": message}).encode("utf-8")
    req = request.Request(webhook, data=payload, headers={"Content-Type": "application/json"})
    try:
        request.urlopen(req, timeout=10)
    except error.URLError as exc:
        print(f"Failed to deliver webhook notification: {exc}")


def main() -> None:
    args = parse_args()
    cutoff = datetime.now(timezone.utc) - timedelta(hours=args.hours)
    reports = load_recent_reports(Path(args.report_dir), cutoff)
    alerts = [
        r for r in reports
        if r.get("severity") in {"warn", "error"}
    ]
    if not alerts:
        print("No alerts in the selected window.")
        return
    lines = []
    for r in alerts:
        manifest = r.get("manifest") or {}
        lines.append(
            f"{r.get('generated_at')} | {manifest.get('path')} | severity={r.get('severity')} "
            f"gap_ratio={r.get('gap_ratio')} file={r.get('__file')}"
        )
    message = "\n".join(["Data quality alerts detected:"] + lines)
    print(message)
    if args.webhook_url:
        notify(args.webhook_url, message)
    raise SystemExit(1)


if __name__ == "__main__":
    main()
