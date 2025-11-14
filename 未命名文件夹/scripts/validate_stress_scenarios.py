#!/usr/bin/env python3
"""Validate config/stress_scenarios.yaml definitions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
import sys

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from scripts.scenario_utils import ALLOWED_FIELDS, load_scenarios, validate_data

DEFAULT_PATH = Path("config/stress_scenarios.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate stress scenario catalog")
    parser.add_argument(
        "--path",
        default=str(DEFAULT_PATH),
        help="Path to stress_scenarios YAML (default: config/stress_scenarios.yaml)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.path).expanduser()
    try:
        raw = load_scenarios(path)
    except Exception as exc:
        print(f"Validation failed: {exc}")
        sys.exit(1)
    errors = validate_data(raw)
    if errors:
        print("Validation failed:")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)
    print(f"{path} OK ({len(ALLOWED_FIELDS)} allowed fields validated, {len(raw)} scenarios).")


if __name__ == "__main__":
    main()
