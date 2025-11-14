#!/usr/bin/env python3
"""
Rolling walk-forward runner that slices a dataset into train/test windows
and executes the upgraded backtest pipeline for each slice.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml
from loguru import logger

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from core.backtest.strategy_engine import parse_strategy_specs  # type: ignore
from scripts import validate_dataset as dq  # type: ignore
from scripts.backtest_strategy import run_once  # type: ignore

TIME_COLUMNS = ["ts", "time", "timestamp", "datetime", "date"]
DEFAULT_MANIFEST = "data/_manifest.json"
DEFAULT_RESULTS = "results"

RUN_ONCE_PARAMS = set(inspect.signature(run_once).parameters.keys())

KEY_MAP = {
    "csv": "csv_path",
    "cash": "initial_cash",
    "qty": "qty",
    "account_ccy": "account_ccy",
    "fast": "fast_win",
    "slow": "slow_win",
    "spread": "spread_pips",
    "slip": "slippage_pips",
    "comm": "commission_per_million",
    "sl": "stop_loss_pips",
    "tp": "take_profit_pips",
    "atr_sl": "atr_sl",
    "atr_tp": "atr_tp",
    "atr_window": "atr_window",
    "rsi_period": "rsi_period",
    "rsi_long_thresh": "rsi_long_thresh",
    "rsi_short_thresh": "rsi_short_thresh",
    "enable_trailing": "enable_trailing",
    "trailing_enable_atr_mult": "trailing_enable_atr_mult",
    "trailing_atr_mult": "trailing_atr_mult",
    "long_only_above_slow": "long_only_above_slow",
    "slope_lookback": "slope_lookback",
    "cooldown": "cooldown",
    "allow_short": "allow_short",
    "short_only_below_slow": "short_only_below_slow",
    "risk_per_trade_pct": "risk_per_trade_pct",
    "max_drawdown_pct": "max_drawdown_pct",
    "max_position_units": "max_position_units",
    "regime_ema_window": "regime_ema_window",
    "regime_slope_min": "regime_slope_min",
    "regime_atr_min": "regime_atr_min",
    "regime_atr_percentile_min": "regime_atr_percentile_min",
    "regime_atr_percentile_window": "regime_atr_percentile_window",
    "regime_trend_min_bars": "regime_trend_min_bars",
    "strategies": "strategies",
    "htf_factor": "htf_factor",
    "htf_ema_window": "htf_ema_window",
    "htf_rsi_period": "htf_rsi_period",
    "cost_profiles": "cost_profiles",
    "slippage_model": "slippage_model",
    "strategy_mode": "strategy_mode",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run walk-forward analysis across rolling windows.")
    parser.add_argument("--config", required=True, help="YAML config with base strategy parameters.")
    parser.add_argument("--csv", help="Override CSV path (defaults to config csv_path).")
    parser.add_argument("--train-bars", type=int, default=3000, help="Number of bars in each training window.")
    parser.add_argument("--test-bars", type=int, default=1000, help="Number of bars in each test window.")
    parser.add_argument(
        "--step-bars",
        type=int,
        default=None,
        help="Step size between windows (defaults to test-bars).",
    )
    parser.add_argument("--max-windows", type=int, default=None, help="Optional cap on number of windows.")
    parser.add_argument(
        "--output-root",
        default=DEFAULT_RESULTS,
        help="Directory where aggregated walk-forward artifacts will be stored.",
    )
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST, help="Manifest path for optional validation.")
    parser.add_argument("--label", default=None, help="Optional label recorded in summary.json.")
    parser.add_argument(
        "--sharpe-threshold",
        type=float,
        default=1.0,
        help="Minimum Sharpe to mark a window as pass.",
    )
    parser.add_argument(
        "--max-dd-threshold",
        type=float,
        default=0.1,
        help="Maximum allowed drawdown magnitude (positive value).",
    )
    parser.add_argument(
        "--validate-base-data",
        action="store_true",
        help="Run data validation once on the source CSV before slicing.",
    )
    parser.add_argument(
        "--keep-train-csv",
        action="store_true",
        help="Export the train slices alongside test slices for auditing (default: only test).",
    )
    return parser.parse_args()


def normalize_params(raw: Dict[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    for key, value in (raw or {}).items():
        canon_key = KEY_MAP.get(key, key)
        if canon_key in RUN_ONCE_PARAMS:
            normalized[canon_key] = value
    return normalized


def load_config(cfg_path: Path) -> Dict[str, Any]:
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    cfg = normalize_params(raw)
    strategies = cfg.get("strategies")
    if strategies:
        cfg["strategies"] = parse_strategy_specs(strategies)
    return cfg


def detect_time_column(df: pd.DataFrame) -> str:
    for col in TIME_COLUMNS:
        if col in df.columns:
            return col
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col
    raise ValueError(f"No timestamp column found in dataset; expected any of {TIME_COLUMNS}")


def prepare_dataset(csv_path: Path) -> Tuple[pd.DataFrame, str]:
    df = pd.read_csv(csv_path)
    time_col = detect_time_column(df)
    df["ts"] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    return df, time_col


def compute_windows(
    df: pd.DataFrame,
    train: int,
    test: int,
    step: int,
    max_windows: Optional[int] = None,
) -> List[Tuple[int, slice, slice]]:
    total = len(df)
    if train <= 0 or test <= 0:
        raise ValueError("train-bars and test-bars must be positive.")
    if train + test > total:
        raise ValueError(f"Dataset length ({total}) insufficient for a single window of train+test={train + test}.")
    windows: List[Tuple[int, slice, slice]] = []
    idx = 0
    win_id = 0
    while idx + train + test <= total:
        train_slice = slice(idx, idx + train)
        test_slice = slice(idx + train, idx + train + test)
        windows.append((win_id, train_slice, test_slice))
        win_id += 1
        if max_windows is not None and win_id >= max_windows:
            break
        idx += step
    return windows


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def params_fingerprint(params: Dict[str, Any]) -> str:
    ignore = {"csv_path", "results_dir", "manifest_path", "validate_data", "write_summary"}
    filtered = {k: v for k, v in params.items() if k not in ignore}
    blob = json.dumps(filtered, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def export_slice(df: pd.DataFrame, indices: slice, path: Path) -> None:
    subset = df.iloc[indices]
    subset.to_csv(path, index=False)


def maybe_validate_dataset(csv_path: Path, manifest: Path) -> Optional[Dict[str, Any]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset for validation not found: {csv_path}")
    manifest_entry = dq.load_manifest_entry(Path(manifest), csv_path) if manifest else None
    report = dq.compute_report(csv_path, manifest_entry, z_threshold=5.0)
    severity = report.get("severity")
    logger.info(
        "Base dataset validation: severity=%s rows=%s gap_ratio=%.6f",
        severity,
        report.get("total_rows"),
        report.get("gap_ratio", 0.0),
    )
    if severity == "error":
        raise RuntimeError(f"Dataset validation failed for {csv_path}: {report.get('messages')}")
    return report


def build_summary(stats: List[Dict[str, Any]]) -> Dict[str, Any]:
    df = pd.DataFrame(stats)
    aggregates: Dict[str, Dict[str, float]] = {}
    for metric in ["sharpe", "ann_return", "ann_vol", "max_drawdown", "sortino", "calmar"]:
        if metric in df.columns and not df[metric].dropna().empty:
            series = df[metric].dropna()
            aggregates[metric] = {
                "mean": float(series.mean()),
                "median": float(series.median()),
                "std": float(series.std(ddof=0)),
                "p05": float(series.quantile(0.05)),
                "p95": float(series.quantile(0.95)),
            }
    summary = {
        "windows": len(stats),
        "aggregates": aggregates,
        "run_ids": [row.get("run_id") for row in stats],
        "passes": int((df["status"] == "pass").sum()) if "status" in df.columns else None,
        "fails": int((df["status"] == "fail").sum()) if "status" in df.columns else None,
    }
    return summary


def main():
    args = parse_args()
    cfg_path = Path(args.config).expanduser().resolve()
    cfg = load_config(cfg_path)
    csv_path = Path(args.csv or cfg.get("csv_path") or cfg.get("csv", "")).expanduser()
    if not csv_path:
        raise ValueError("CSV path must be provided via --csv or config file.")
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df, _ = prepare_dataset(csv_path)
    train = args.train_bars
    test = args.test_bars
    step = args.step_bars or test
    windows = compute_windows(df, train, test, step, args.max_windows)
    if not windows:
        raise RuntimeError("No walk-forward windows could be generated with the provided parameters.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = args.label or cfg_path.stem
    session_dir = Path(args.output_root).expanduser().resolve() / f"walkforward_{label}_{timestamp}"
    wf_dir = session_dir / "walkforward"
    wf_dir.mkdir(parents=True, exist_ok=True)

    base_validation_report = None
    if args.validate_base_data:
        base_validation_report = maybe_validate_dataset(csv_path, Path(args.manifest).expanduser().resolve())

    base_params = copy.deepcopy(cfg)
    base_params["validate_data"] = False  # slices derive from validated dataset
    base_params.setdefault("symbol", label.upper())

    stats: List[Dict[str, Any]] = []
    for win_id, train_slice, test_slice in windows:
        test_csv_path = wf_dir / f"window_{win_id:03d}_test.csv"
        export_slice(df, test_slice, test_csv_path)
        if args.keep_train_csv:
            train_csv_path = wf_dir / f"window_{win_id:03d}_train.csv"
            export_slice(df, train_slice, train_csv_path)

        params = copy.deepcopy(base_params)
        params["csv_path"] = str(test_csv_path)
        params["results_dir"] = str(session_dir)
        params["manifest_path"] = args.manifest

        logger.info(
            "Walk-forward window {win} | train={train} bars test={test} bars ({start} → {end})",
            win=win_id,
            train=train,
            test=test,
            start=df.iloc[test_slice.start]["ts"],
            end=df.iloc[test_slice.stop - 1]["ts"],
        )

        result = run_once(**params)

        record = {
            "window": win_id,
            "train_rows": train_slice.stop - train_slice.start,
            "test_rows": test_slice.stop - test_slice.start,
            "train_start": df.iloc[train_slice.start]["ts"].isoformat(),
            "train_end": df.iloc[train_slice.stop - 1]["ts"].isoformat(),
            "test_start": df.iloc[test_slice.start]["ts"].isoformat(),
            "test_end": df.iloc[test_slice.stop - 1]["ts"].isoformat(),
            "run_id": result.get("run_id"),
            "summary_path": result.get("summary_path"),
            "sharpe": result.get("sharpe"),
            "ann_return": result.get("ann_return"),
            "ann_vol": result.get("ann_vol"),
            "max_drawdown": result.get("max_drawdown"),
            "sortino": result.get("sortino"),
            "calmar": result.get("calmar"),
            "final_equity": result.get("final_equity"),
            "trades": result.get("trades"),
            "data_hash": file_sha256(test_csv_path),
            "data_path": str(test_csv_path.relative_to(BASE_DIR)) if test_csv_path.is_relative_to(BASE_DIR) else str(test_csv_path),
            "param_fingerprint": params_fingerprint(params),
        }
        max_dd = abs(record.get("max_drawdown") or 0.0)
        sharpe = record.get("sharpe") or 0.0
        record["status"] = "pass" if sharpe >= args.sharpe_threshold and max_dd <= args.max_dd_threshold else "fail"
        stats.append(record)

    metrics_csv = wf_dir / "metrics.csv"
    pd.DataFrame(stats).to_csv(metrics_csv, index=False)

    summary = {
        "label": label,
        "session_dir": str(session_dir.relative_to(BASE_DIR)) if session_dir.is_relative_to(BASE_DIR) else str(session_dir),
        "source_csv": str(csv_path.relative_to(BASE_DIR)) if csv_path.is_relative_to(BASE_DIR) else str(csv_path),
        "train_bars": train,
        "test_bars": test,
        "step_bars": step,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sharpe_threshold": args.sharpe_threshold,
        "max_dd_threshold": args.max_dd_threshold,
        "base_validation": base_validation_report,
    }
    summary.update(build_summary(stats))
    summary_path = wf_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info("Walk-forward run complete: %s", summary_path)
    logger.info("Metrics CSV saved to %s", metrics_csv)


if __name__ == "__main__":
    main()
