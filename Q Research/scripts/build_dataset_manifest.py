#!/usr/bin/env python3
"""
Generate a manifest describing datasets stored under `data/`.

For each supported file (CSV/Parquet/Feather) this script records:
  - relative path
  - file size, mtime, and SHA256 checksum
  - row count, columns, dtypes
  - detected time range (based on common timestamp columns)

Usage:
    python scripts/build_dataset_manifest.py \
        --dirs data/raw data/derived \
        --output data/_manifest.json
"""

from __future__ import annotations

import argparse
import json
import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd
from loguru import logger


SUPPORTED_EXTS = {".csv", ".parquet", ".feather"}
TIME_COLUMNS = ["ts", "time", "timestamp", "datetime", "date"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build dataset manifest.")
    parser.add_argument(
        "--dirs",
        nargs="+",
        default=["data/raw", "data/derived"],
        help="Directories to scan for datasets.",
    )
    parser.add_argument(
        "--output",
        default="data/_manifest.json",
        help="Path to write manifest JSON.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of rows to load when summarizing large files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan files but do not write manifest.",
    )
    return parser.parse_args()


def sha256sum(path: Path, chunk_size: int = 1 << 20) -> str:
    """Compute SHA256 hash for a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def load_dataframe(path: Path, limit: Optional[int]) -> pd.DataFrame:
    """Load a dataframe from supported formats with optional row limit."""
    if path.suffix == ".csv":
        return pd.read_csv(path, nrows=limit)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
        return df.head(limit) if limit else df
    if path.suffix == ".feather":
        df = pd.read_feather(path)
        return df.head(limit) if limit else df
    raise ValueError(f"Unsupported file format: {path}")


def detect_time_range(df: pd.DataFrame) -> tuple[Optional[str], Optional[str]]:
    """Try to detect timestamp column and return ISO8601 min/max."""
    col = next((c for c in TIME_COLUMNS if c in df.columns), None)
    if col is None:
        for candidate in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[candidate]):
                col = candidate
                break
    if col is None:
        return None, None
    ts = pd.to_datetime(df[col], utc=True, errors="coerce").dropna()
    if ts.empty:
        return None, None
    return ts.min().isoformat(), ts.max().isoformat()


def summarize_file(path: Path, repo_root: Path, limit: Optional[int]) -> dict:
    """Collect metadata for the given dataset file."""
    rel_path = path.relative_to(repo_root)
    stat = path.stat()
    size = stat.st_size
    mtime = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
    checksum = sha256sum(path)

    try:
        df = load_dataframe(path, limit)
    except Exception as exc:
        logger.error(f"Failed to load {rel_path}: {exc}")
        df = None

    rows = int(df.shape[0]) if df is not None else None
    columns: List[str] = df.columns.tolist() if df is not None else []
    dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()} if df is not None else {}
    t_start, t_end = detect_time_range(df) if df is not None else (None, None)

    return {
        "path": str(rel_path),
        "extension": path.suffix,
        "size_bytes": size,
        "modified_at": mtime,
        "sha256": checksum,
        "rows": rows,
        "columns": columns,
        "dtypes": dtypes,
        "time_start": t_start,
        "time_end": t_end,
    }


def iter_dataset_files(dirs: Iterable[str], repo_root: Path) -> List[Path]:
    files: List[Path] = []
    for d in dirs:
        target = (repo_root / d).resolve()
        if not target.exists():
            logger.warning(f"Skip missing directory: {target}")
            continue
        for path in target.rglob("*"):
            if path.is_file() and path.suffix in SUPPORTED_EXTS:
                files.append(path)
    return files


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    files = iter_dataset_files(args.dirs, repo_root)
    if not files:
        logger.warning("No dataset files found.")
    logger.info(f"Found {len(files)} dataset files.")

    entries = []
    for path in files:
        logger.info(f"Summarizing {path}")
        entry = summarize_file(path, repo_root, args.limit)
        entries.append(entry)

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": args.dirs,
        "file_count": len(entries),
        "files": entries,
    }

    if args.dry_run:
        logger.info("Dry-run enabled; manifest not written.")
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        return

    output_path = (repo_root / args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    logger.info(f"Manifest written to {output_path}")


if __name__ == "__main__":
    main()
