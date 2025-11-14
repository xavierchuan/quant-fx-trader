#!/usr/bin/env python3
"""
Validate dataset quality (missing data, duplicate timestamps, gaps, outliers).

Example:
    python scripts/validate_dataset.py --path data/raw/EURUSD_H1.csv
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger


TIME_COLUMNS = ["ts", "time", "timestamp", "datetime", "date"]
DEFAULT_MANIFEST = "data/_manifest.json"
REPORT_DIR = Path("results/data_quality")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate dataset quality.")
    parser.add_argument("--path", required=True, help="Dataset file path (relative or absolute).")
    parser.add_argument(
        "--manifest",
        default=DEFAULT_MANIFEST,
        help="Optional manifest JSON to enrich report.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional explicit path for JSON report.",
    )
    parser.add_argument(
        "--max-outlier-z",
        type=float,
        default=5.0,
        help="Z-score threshold to flag numeric outliers.",
    )
    return parser.parse_args()


def detect_time_column(df: pd.DataFrame) -> Optional[str]:
    for col in TIME_COLUMNS:
        if col in df.columns:
            return col
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col
    return None


def infer_interval_seconds(series: pd.Series) -> Optional[float]:
    ts = pd.to_datetime(series, utc=True, errors="coerce").dropna()
    if ts.size < 3:
        return None
    diffs = np.diff(ts.values.astype("datetime64[ns]").astype(np.int64) // 1_000_000_000)
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return None
    return float(np.median(diffs))


def load_manifest_entry(manifest_path: Path, dataset_path: Path) -> Optional[Dict]:
    if not manifest_path.exists():
        return None
    try:
        with manifest_path.open("r", encoding="utf-8") as fh:
            manifest = json.load(fh)
    except Exception as exc:
        logger.warning(f"Failed to parse manifest {manifest_path}: {exc}")
        return None
    rel = str(dataset_path)
    rel_alt = str(dataset_path.resolve().relative_to(Path.cwd()))
    for entry in manifest.get("files", []):
        if entry.get("path") in (rel, rel_alt):
            return entry
    return None


def summarize_numeric_outliers(df: pd.DataFrame, threshold: float) -> Dict[str, int]:
    outliers: Dict[str, int] = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if numeric_cols.empty:
        return outliers
    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty:
            continue
        zscores = np.abs((series - series.mean()) / (series.std(ddof=1) or 1))
        count = int((zscores > threshold).sum())
        if count:
            outliers[col] = count
    return outliers


def compute_report(path: Path, manifest_entry: Optional[Dict], z_threshold: float) -> Dict:
    df = pd.read_csv(path)
    total_rows = int(df.shape[0])
    null_counts = df.isna().sum().to_dict()

    time_col = detect_time_column(df)
    duplicates = 0
    gap_ratio = 0.0
    inferred_interval = None
    if time_col:
        ts = pd.to_datetime(df[time_col], utc=True, errors="coerce")
        duplicates = int(ts.duplicated().sum())
        valid_ts = ts.dropna().sort_values()
        inferred_interval = infer_interval_seconds(valid_ts)
        if inferred_interval:
            diffs = np.diff(valid_ts.values.astype("datetime64[ns]").astype(np.int64) // 1_000_000_000)
            expected = inferred_interval
            gaps = diffs[diffs > expected * 1.5]
            gap_ratio = float(len(gaps) / max(len(diffs), 1))
    outliers = summarize_numeric_outliers(df, z_threshold)

    messages = []
    severity = "pass"
    if total_rows == 0:
        severity = "error"
        messages.append("Dataset is empty.")
    if duplicates > 0:
        severity = "error"
        messages.append(f"Found {duplicates} duplicate timestamps in column '{time_col}'.")
    if gap_ratio > 0.01:
        severity = "error"
        messages.append(f"Timestamp gaps exceed 1% of intervals (ratio={gap_ratio:.4f}).")
    elif gap_ratio > 0:
        severity = "warn"
        messages.append(f"Non-zero timestamp gaps detected (ratio={gap_ratio:.4f}).")
    null_ratio = max((count / total_rows) if total_rows else 0 for count in null_counts.values() or [0])
    if null_ratio > 0.05:
        severity = "error"
        messages.append(f"Columns contain >5% nulls (max ratio={null_ratio:.4f}).")
    elif null_ratio > 0:
        severity = "warn"
        messages.append(f"Columns contain nulls (max ratio={null_ratio:.4f}).")
    if outliers:
        if severity == "pass":
            severity = "warn"
        messages.append(f"Detected numeric outliers (threshold z>{z_threshold}).")

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(path),
        "total_rows": total_rows,
        "time_column": time_col,
        "inferred_interval_seconds": inferred_interval,
        "duplicate_timestamps": duplicates,
        "gap_ratio": gap_ratio,
        "null_counts": null_counts,
        "numeric_outliers": outliers,
        "severity": severity,
        "messages": messages,
    }
    if manifest_entry:
        report["manifest"] = {
            "path": manifest_entry.get("path"),
            "sha256": manifest_entry.get("sha256"),
            "rows": manifest_entry.get("rows"),
            "time_start": manifest_entry.get("time_start"),
            "time_end": manifest_entry.get("time_end"),
        }
    return report


def save_report(report: Dict, output_path: Optional[Path], dataset_path: Path) -> Path:
    if output_path is None:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = dataset_path.stem
        output_path = REPORT_DIR / f"{ts}_{name}.json"
    else:
        output_path = output_path.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    return output_path


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.path).expanduser().resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    manifest_entry = None
    if args.manifest:
        manifest_entry = load_manifest_entry(Path(args.manifest), dataset_path)

    report = compute_report(dataset_path, manifest_entry, args.max_outlier_z)
    output_path = save_report(report, Path(args.output) if args.output else None, dataset_path)
    logger.info(f"Validation severity: {report['severity']}")
    logger.info(f"Report written to {output_path}")


if __name__ == "__main__":
    main()
