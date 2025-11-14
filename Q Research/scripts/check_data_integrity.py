#!/usr/bin/env python3
"""
CI helper that ensures watched datasets keep their expected signatures and pass validation.

Fail conditions:
  1. Dataset listed in baseline JSON is missing.
  2. Manifest entry hash differs from the baseline (meaning data changed without approval).
  3. Validation severity == "error".
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from loguru import logger

ROOT = Path(__file__).resolve().parents[1]

sys.path.append(str(ROOT))

from scripts.validate_dataset import compute_report, DEFAULT_MANIFEST  # noqa


def load_manifest(manifest_path: Path) -> dict:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check dataset hashes & validation severity.")
    parser.add_argument("--baseline", default="data/signature_baseline.json", help="Baseline hash file.")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST, help="Manifest JSON to compare against.")
    parser.add_argument("--max-outlier-z", type=float, default=5.0, help="Z-score threshold used for validation.")
    return parser.parse_args()


def load_baseline_entries(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Baseline file missing: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict):
        return [{"path": k, "sha256": v, "allow_drift": False} for k, v in data.items()]
    if isinstance(data, list):
        return data
    raise ValueError("Baseline must be a dict or list of entries.")


def main() -> None:
    args = parse_args()
    baseline_path = (ROOT / args.baseline).resolve()
    manifest_path = (ROOT / args.manifest).resolve()

    baseline_entries = load_baseline_entries(baseline_path)

    manifest = load_manifest(manifest_path)
    manifest_index = {entry["path"]: entry for entry in manifest.get("files", [])}

    errors = []
    warnings = []
    for item in baseline_entries:
        rel_path = item["path"]
        expected_hash = item.get("sha256")
        allow_drift = bool(item.get("allow_drift", False))
        dataset_path = (ROOT / rel_path).resolve()
        manifest_entry = manifest_index.get(rel_path)
        label = rel_path
        if manifest_entry is None:
            errors.append(f"{label}: missing from manifest {manifest_path}")
            continue
        actual_hash = manifest_entry.get("sha256")
        if actual_hash != expected_hash:
            message = (
                f"{label}: hash drift detected (expected {expected_hash}, actual {actual_hash}). "
                "Update the baseline with justification if intentional."
            )
            if allow_drift:
                warnings.append(message)
            else:
                errors.append(message)
        if not dataset_path.exists():
            errors.append(f"{label}: dataset file not found on disk")
            continue
        report = compute_report(dataset_path, manifest_entry, args.max_outlier_z)
        severity = report.get("severity")
        msg = f"{label}: validation severity={severity} ({'; '.join(report.get('messages', []))})"
        if severity == "error":
            errors.append(msg)
        elif severity == "warn":
            warnings.append(msg)

    for warn in warnings:
        logger.warning(warn)

    if errors:
        logger.error("Data integrity check failed:")
        for err in errors:
            logger.error(" - {}", err)
        sys.exit(1)

    logger.info(f"Data integrity checks passed for {len(baseline_entries)} dataset(s).")


if __name__ == "__main__":
    main()
