#!/usr/bin/env python3
"""
Monte Carlo / stress test for an existing backtest run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from loguru import logger

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from metrics.perf import compute_metrics
from scripts.scenario_utils import get_scenario


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Monte Carlo stress on a backtest result.")
    parser.add_argument("--run", help="Path to results/<run_id> directory.")
    parser.add_argument("--equity", help="Explicit equity CSV (ts,equity). Overrides --run artifacts.")
    parser.add_argument("--iterations", type=int, default=500, help="Number of bootstrap iterations.")
    parser.add_argument("--method", choices=["bootstrap", "block"], default="bootstrap", help="Resampling method.")
    parser.add_argument("--block-size", type=int, default=None, help="Block size for block bootstrap (overrides scenario).")
    parser.add_argument("--return-scale", type=float, default=None, help="Scale factor applied to resampled returns (overrides scenario).")
    parser.add_argument("--ruin-threshold", type=float, default=0.8, help="Final equity / initial equity threshold to count as ruin.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--scenario", default=None, help="Optional stress scenario label stored in outputs.")
    parser.add_argument(
        "--scenario-file",
        default="config/stress_scenarios.yaml",
        help="Scenario definitions file (default: config/stress_scenarios.yaml).",
    )
    return parser.parse_args()


def load_equity_series(run_path: Path | None, explicit_csv: str | None) -> pd.Series:
    if explicit_csv:
        path = Path(explicit_csv).expanduser()
    else:
        if not run_path:
            raise ValueError("Either --run or --equity must be provided.")
        summary_file = run_path / "summary.json"
        if not summary_file.exists():
            raise FileNotFoundError(f"summary.json not found in {run_path}")
        summary = json.load(summary_file.open("r", encoding="utf-8"))
        artifacts = summary.get("artifacts") or {}
        equity_path = artifacts.get("equity")
        if not equity_path:
            raise FileNotFoundError("Equity artifact missing in summary; rerun backtest after upgrading.")
        path = (BASE_DIR / equity_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Equity file not found: {path}")
    df = pd.read_csv(path)
    if "equity" not in df.columns:
        raise ValueError(f"Equity CSV missing 'equity' column: {path}")
    return df["equity"].astype(float)


def block_bootstrap(returns: np.ndarray, size: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    if returns.size == 0:
        raise ValueError("Cannot run block bootstrap on an empty return series.")
    if block_size <= 0:
        raise ValueError("Block size must be a positive integer.")
    if block_size > len(returns):
        raise ValueError(f"Block size {block_size} exceeds return series length {len(returns)}.")
    out = []
    while len(out) < size:
        start = rng.integers(0, len(returns) - block_size + 1)
        block = returns[start : start + block_size]
        out.extend(block.tolist())
    return np.array(out[:size])


def run_iteration(returns: np.ndarray, args, rng: np.random.Generator) -> dict:
    if returns.size == 0:
        raise ValueError("Return series is empty; cannot run Monte Carlo.")
    if args.method == "bootstrap":
        draw = rng.choice(returns, size=returns.size, replace=True)
    else:
        draw = block_bootstrap(returns, returns.size, args.block_size, rng)
    draw = draw * args.return_scale
    equity = np.cumprod(1.0 + draw)
    equity_series = list(enumerate(equity, start=1))
    metrics = compute_metrics(equity_series)
    return metrics


def summarize(metrics_list: List[dict], ruin_threshold: float, initial_equity: float) -> dict:
    df = pd.DataFrame(metrics_list)
    summary = {}
    for column in ["sharpe", "sortino", "calmar", "ann_return", "max_drawdown"]:
        if column in df.columns:
            summary[column] = {
                "mean": float(df[column].mean()),
                "std": float(df[column].std()),
                "p05": float(df[column].quantile(0.05)),
                "p50": float(df[column].quantile(0.5)),
                "p95": float(df[column].quantile(0.95)),
            }
    ruin = (df["final_equity"] <= initial_equity * ruin_threshold).mean() if "final_equity" in df else None
    summary["p_ruin"] = float(ruin) if ruin is not None else None
    return summary


def _load_scenario(args: argparse.Namespace) -> Optional[dict]:
    if not args.scenario:
        return None
    scenario_path = Path(args.scenario_file).expanduser()
    try:
        scenario = get_scenario(args.scenario, scenario_path)
    except KeyError as exc:
        raise ValueError(f"Scenario '{args.scenario}' not found in {scenario_path}") from exc
    return scenario


def _apply_scenario(args: argparse.Namespace, scenario_cfg: Optional[dict]) -> None:
    if not scenario_cfg:
        args.return_scale = args.return_scale if args.return_scale is not None else 1.0
        args.block_size = args.block_size if args.block_size is not None else 20
        return
    if args.return_scale is None and scenario_cfg.get("return_scale") is not None:
        args.return_scale = float(scenario_cfg["return_scale"])
    if args.block_size is None and scenario_cfg.get("block_size") is not None:
        args.block_size = int(scenario_cfg["block_size"])
    args.return_scale = args.return_scale if args.return_scale is not None else scenario_cfg.get("return_scale", 1.0)
    args.block_size = args.block_size if args.block_size is not None else scenario_cfg.get("block_size", 20)


def main():
    args = parse_args()
    scenario_cfg = _load_scenario(args)
    _apply_scenario(args, scenario_cfg)
    run_path = Path(args.run).expanduser().resolve() if args.run else None
    equity = load_equity_series(run_path, args.equity)
    returns = np.diff(equity.values) / equity.values[:-1]
    if returns.size == 0:
        raise ValueError("Equity series must contain at least two points.")
    initial_equity = float(equity.iloc[0])
    rng = np.random.default_rng(args.seed)
    metrics_list: List[dict] = []

    logger.info(
        "Running Monte Carlo | method={method} iterations={iterations} scenario={scenario} seed={seed}",
        method=args.method,
        iterations=args.iterations,
        scenario=args.scenario or "default",
        seed=args.seed,
    )
    for _ in range(args.iterations):
        metrics = run_iteration(returns, args, rng)
        metrics_list.append(metrics)

    summary = summarize(metrics_list, args.ruin_threshold, initial_equity)
    summary["iterations"] = args.iterations
    summary["method"] = args.method
    summary["return_scale"] = args.return_scale
    summary["scenario"] = args.scenario or "default"
    summary["seed"] = args.seed
    summary["scenario_overrides"] = scenario_cfg

    output_dir = run_path / "stress" if run_path else BASE_DIR / "results" / "stress"
    output_dir.mkdir(parents=True, exist_ok=True)
    iterations_csv = output_dir / "mc_iterations.csv"
    pd.DataFrame(metrics_list).to_csv(iterations_csv, index=False)
    summary_json = output_dir / "mc_summary.json"
    with summary_json.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    logger.info(f"Monte Carlo summary saved to {summary_json}")


if __name__ == "__main__":
    main()
