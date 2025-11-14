"""Helpers for loading and validating stress scenarios."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

ALLOWED_FIELDS = {
    "description",
    "notes",
    "stress_cost_spread_mult",
    "stress_cost_comm_mult",
    "stress_slippage_mult",
    "stress_price_vol_mult",
    "stress_skip_trade_pct",
    "return_scale",
    "block_size",
}

NUMERIC_FIELDS = {
    "stress_cost_spread_mult": (0.0, None),
    "stress_cost_comm_mult": (0.0, None),
    "stress_slippage_mult": (0.0, None),
    "stress_price_vol_mult": (0.0, None),
    "stress_skip_trade_pct": (0.0, 1.0),
    "return_scale": (0.0, None),
}

INT_FIELDS = {
    "block_size": (1, None),
}


def _validate_mapping(name: str, config: Any) -> List[str]:
    errors: List[str] = []
    if not isinstance(config, dict):
        errors.append(f"Scenario '{name}' must be a mapping of field -> value.")
        return errors
    unknown = set(config.keys()) - ALLOWED_FIELDS
    if unknown:
        errors.append(f"Scenario '{name}' has unknown fields: {sorted(unknown)}")
    for field, (min_val, max_val) in NUMERIC_FIELDS.items():
        if field in config and config[field] is not None:
            try:
                value = float(config[field])
            except (TypeError, ValueError):
                errors.append(f"Scenario '{name}' field '{field}' must be numeric.")
                continue
            if min_val is not None and value < min_val:
                errors.append(f"Scenario '{name}' field '{field}' must be >= {min_val}. Got {value}.")
            if max_val is not None and value > max_val:
                errors.append(f"Scenario '{name}' field '{field}' must be <= {max_val}. Got {value}.")
    for field, (min_val, max_val) in INT_FIELDS.items():
        if field in config and config[field] is not None:
            try:
                value = int(config[field])
            except (TypeError, ValueError):
                errors.append(f"Scenario '{name}' field '{field}' must be an integer.")
                continue
            if min_val is not None and value < min_val:
                errors.append(f"Scenario '{name}' field '{field}' must be >= {min_val}. Got {value}.")
            if max_val is not None and value > max_val:
                errors.append(f"Scenario '{name}' field '{field}' must be <= {max_val}. Got {value}.")
    return errors


def validate_data(data: Any) -> List[str]:
    if not isinstance(data, dict):
        return ["Scenario file must contain a mapping of scenario_name -> config"]
    errors: List[str] = []
    for name, config in data.items():
        errors.extend(_validate_mapping(name, config))
    return errors


def load_scenarios(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Scenario file not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    errors = validate_data(raw)
    if errors:
        raise ValueError("Invalid scenario definitions:\n" + "\n".join(errors))
    return {name: dict(config or {}) for name, config in raw.items()}


def get_scenario(name: str, path: Path) -> Dict[str, Any]:
    scenarios = load_scenarios(path)
    if name not in scenarios:
        raise KeyError(f"Scenario '{name}' not found in {path}")
    return scenarios[name]


def apply_defaults(values: Dict[str, Any], defaults: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(defaults)
    merged.update({k: v for k, v in values.items() if v is not None})
    return merged

