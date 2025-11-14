from __future__ import annotations

"""Batch backtests for multiple configs/symbols."""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
import yaml
from loguru import logger

sys.path.append(str(Path(__file__).resolve().parents[1]))
from core.backtest.strategy_engine import parse_strategy_specs  # type: ignore
from scripts.backtest_strategy import run_once  # type: ignore
from scripts.scenario_utils import load_scenarios


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
    "strategies": "strategies",
    "htf_factor": "htf_factor",
    "htf_ema_window": "htf_ema_window",
    "htf_rsi_period": "htf_rsi_period",
    "cost_profiles": "cost_profiles",
    "slippage_model": "slippage_model",
    "strategy_mode": "strategy_mode",
    "strategy_vote_threshold": "strategy_vote_threshold",
    "stress_cost_spread_mult": "stress_cost_spread_mult",
    "stress_cost_comm_mult": "stress_cost_comm_mult",
    "stress_slippage_mult": "stress_slippage_mult",
    "stress_price_vol_mult": "stress_price_vol_mult",
    "stress_skip_trade_pct": "stress_skip_trade_pct",
}

SCENARIO_FIELDS = [
    "stress_cost_spread_mult",
    "stress_cost_comm_mult",
    "stress_slippage_mult",
    "stress_price_vol_mult",
    "stress_skip_trade_pct",
]

PARAM_COLUMNS = [
    "fast_win",
    "slow_win",
    "atr_sl",
    "atr_tp",
    "atr_window",
    "cooldown",
    "allow_short",
    "long_only_above_slow",
    "short_only_below_slow",
    "slope_lookback",
    "strategy_vote_threshold",
    "spread_pips",
    "slippage_pips",
    "commission_per_million",
    "risk_per_trade_pct",
]


def normalize_params(raw: Dict[str, Any]) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    for k, v in (raw or {}).items():
        key = KEY_MAP.get(k, k)
        params[key] = v
    return params


def load_params(cfg_path: Path) -> Dict[str, Any]:
    with cfg_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    params = normalize_params(raw)
    params["symbol"] = params.get("symbol", cfg_path.stem.upper())
    params["config_name"] = cfg_path.name
    return params


def run_job(
    params: Dict[str, Any],
    scenario_name: Optional[str],
    scenario_cfg: Optional[Dict[str, Any]],
    extra_cols: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    label = params.get("label")
    config_name = params.get("config_name")
    job_params = {k: v for k, v in params.items() if k not in ("label", "config_name")}
    if isinstance(job_params.get("strategies"), (list, dict)):
        job_params["strategies"] = parse_strategy_specs(job_params["strategies"])
    symbol = job_params.get("symbol", "UNKNOWN").upper()
    logger.info(f"Running {symbol} ({label or config_name})")
    result = run_once(**job_params)
    data_validation = result.get("data_validation") or {}
    summary = {
        "symbol": symbol,
        "label": label,
        "config_name": config_name,
        "sharpe": result.get("sharpe"),
        "ann_return": result.get("ann_return"),
        "ann_vol": result.get("ann_vol"),
        "max_drawdown": result.get("max_drawdown"),
        "trades": result.get("trades"),
        "final_equity": result.get("final_equity"),
        "sortino": result.get("sortino"),
        "calmar": result.get("calmar"),
        "run_id": result.get("run_id"),
        "summary_path": result.get("summary_path"),
        "data_severity": data_validation.get("severity"),
        "strategy_mode": job_params.get("strategy_mode"),
        "stress_cost_spread_mult": job_params.get("stress_cost_spread_mult"),
        "stress_cost_comm_mult": job_params.get("stress_cost_comm_mult"),
        "stress_slippage_mult": job_params.get("stress_slippage_mult"),
        "stress_price_vol_mult": job_params.get("stress_price_vol_mult"),
        "stress_skip_trade_pct": job_params.get("stress_skip_trade_pct"),
        "scenario": scenario_name,
        "scenario_overrides": scenario_cfg,
    }
    for col in PARAM_COLUMNS:
        value = job_params.get(col)
        if isinstance(value, (int, float, bool, str)) or value is None:
            summary[col] = value
    if extra_cols:
        for col in extra_cols:
            if col in summary:
                continue
            if col in job_params:
                summary[col] = job_params.get(col)
            elif col in result:
                summary[col] = result.get(col)
    return summary


def load_jobs(symbols: List[str], config_dir: Path, schedule_path: Optional[str]) -> List[Dict[str, Any]]:
    jobs: List[Dict[str, Any]] = []
    if schedule_path:
        schedule_file = Path(schedule_path).expanduser()
        data = yaml.safe_load(schedule_file.open("r", encoding="utf-8")) or []
        if not isinstance(data, list):
            raise ValueError("Schedule YAML must be a list of job definitions")
        for idx, raw in enumerate(data):
            job = normalize_params(dict(raw or {}))
            job["label"] = job.get("label") or f"job_{idx}"
            if "symbol" not in job:
                raise ValueError(f"Job {job['label']} missing 'symbol'")
            jobs.append(job)
    else:
        for sym in symbols:
            cfg_path = config_dir / f"{sym.lower()}_regime.yaml"
            if not cfg_path.exists():
                logger.error(f"Config not found for {sym}: {cfg_path}")
                continue
            jobs.append(load_params(cfg_path))
    return jobs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", type=str, default="EURUSD", help="Comma separated symbols (ignored if --schedule provided)")
    parser.add_argument("--config-dir", type=str, default="config", help="Config directory")
    parser.add_argument("--out", type=str, default="data/results", help="Output dir for summaries")
    parser.add_argument("--schedule", type=str, help="YAML schedule of jobs")
    parser.add_argument("--scenario", type=str, default=None, help="Default stress scenario for all jobs")
    parser.add_argument(
        "--scenario-file",
        type=str,
        default="config/stress_scenarios.yaml",
        help="Scenario definition file (default: config/stress_scenarios.yaml)",
    )
    parser.add_argument(
        "--extra-cols",
        type=str,
        default="",
        help="Comma-separated list of additional columns to include in the output (copied from params/result).",
    )
    args = parser.parse_args()

    config_dir = Path(args.config_dir)
    symbols = [s.strip().upper() for s in args.symbols.split(",")]
    jobs = load_jobs(symbols, config_dir, args.schedule)
    if not jobs:
        logger.error("No jobs to run")
        sys.exit(1)

    scenario_path = Path(args.scenario_file).expanduser()
    scenario_required = bool(args.scenario) or any(job.get("scenario") for job in jobs)
    scenarios = load_scenarios(scenario_path) if scenario_required else {}

    extra_cols = [c.strip() for c in (args.extra_cols or "").split(",") if c.strip()]
    rows = []
    for job in jobs:
        scenario_name = job.get("scenario") or args.scenario
        scenario_cfg = None
        if scenario_name:
            if scenario_name not in scenarios:
                raise ValueError(f"Scenario '{scenario_name}' not found in {scenario_path}")
            scenario_cfg = scenarios[scenario_name]
        job_copy = dict(job)
        job_params = {k: v for k, v in job_copy.items() if k not in ("scenario",)}
        if scenario_cfg:
            for field in SCENARIO_FIELDS:
                if field in scenario_cfg and field not in job_params:
                    job_params[field] = scenario_cfg[field]
        try:
            rows.append(run_job(job_params, scenario_name, scenario_cfg, extra_cols=extra_cols))
        except Exception as exc:
            label = job.get("label") or job.get("symbol")
            logger.exception(f"Backtest failed for job {label}: {exc}")

    if not rows:
        logger.error("All backtests failed")
        sys.exit(1)

    df = pd.DataFrame(rows)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"batch_backtests_{ts}.csv"
    df.to_csv(csv_path, index=False)
    json_path = out_dir / f"batch_backtests_{ts}.json"
    json_path.write_text(df.to_json(orient="records", indent=2), encoding="utf-8")
    stats = {
        "timestamp": ts,
        "job_count": len(rows),
        "symbols": sorted(df["symbol"].unique()),
        "sharpe_max": df["sharpe"].max(),
        "sharpe_min": df["sharpe"].min(),
    }
    (out_dir / f"batch_backtests_{ts}_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(df.to_string(index=False))
    print(f"Saved summary to {csv_path} / {json_path}")


if __name__ == "__main__":
    main()
