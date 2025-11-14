#!/usr/bin/env python3
"""Replay orders through ExecutionAdapter + RiskEngine with per-strategy configs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yaml

BASE_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = BASE_DIR.parent
for path in (BASE_DIR, ROOT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from QuantTrader.core.risk.risk_engine import RiskEngine, RiskLimits
from QuantTrader.execution.adapter import MockAdapter, OrderParams
from QuantTrader.execution.paper_adapter import PaperAdapter


def infer_symbol_from_path(path: Path) -> Optional[str]:
    stem = path.stem
    for token in stem.split("_"):
        clean = "".join(ch for ch in token if ch.isalpha())
        if not clean:
            continue
        if clean.lower() in {"trade", "trades"}:
            continue
        if 3 <= len(clean) <= 10:
            return clean.upper()
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate execution with risk checks.")
    parser.add_argument("--orders", help="CSV with columns ts,symbol,side,qty,price,notional,strategy(optional)")
    parser.add_argument("--trades-csv", help="Optional trades.csv (ts_entry,symbol,direction,qty,price_entry,pnl,strategy)")
    parser.add_argument("--symbol", help="Fallback symbol when trades CSV lacks column")
    parser.add_argument("--risk-config", help="JSON risk config (single engine)")
    parser.add_argument("--risk-limits-yaml", help="YAML mapping strategies -> limits")
    parser.add_argument("--run-id", help="Run identifier (defaults to timestamp)")
    parser.add_argument("--output", default=None, help="Summary output path (auto from run-id if omitted)")
    parser.add_argument("--risk-log", default="results/risk/events.jsonl", help="Risk event log path")
    parser.add_argument("--adapter", choices=["mock", "paper"], default="mock", help="Execution adapter to use.")
    parser.add_argument("--paper-latency-ms", type=float, default=50.0, help="Paper adapter latency (ms).")
    parser.add_argument("--paper-slippage-pips", type=float, default=0.1, help="Paper adapter slippage (pips).")
    return parser.parse_args()


def load_risk_engine(config_path: Path) -> RiskEngine:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    limits = RiskLimits(
        max_position_notional=cfg["max_position_notional"],
        max_gross_leverage=cfg["max_gross_leverage"],
        max_daily_loss=cfg["max_daily_loss"],
        max_drawdown=cfg["max_drawdown"],
    )
    return RiskEngine(limits=limits, starting_equity=cfg["starting_equity"])


def load_strategy_engines(yaml_path: Path) -> Dict[str, RiskEngine]:
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    engines: Dict[str, RiskEngine] = {}
    global_cfg = data.get("global", {})
    global_limits = global_cfg.get("limits", {})
    default_engine = RiskEngine(
        limits=RiskLimits(
            max_position_notional=global_limits.get("max_position_notional", 0),
            max_gross_leverage=global_limits.get("max_gross_leverage", 0),
            max_daily_loss=global_limits.get("max_daily_loss", 0),
            max_drawdown=global_limits.get("max_drawdown", 0),
        ),
        starting_equity=global_cfg.get("starting_equity", 0),
    )
    engines["default"] = default_engine
    for name, cfg in (data.get("strategies") or {}).items():
        limits_cfg = cfg.get("limits", {})
        engines[name] = RiskEngine(
            limits=RiskLimits(
                max_position_notional=limits_cfg.get("max_position_notional", global_limits.get("max_position_notional", 0)),
                max_gross_leverage=limits_cfg.get("max_gross_leverage", global_limits.get("max_gross_leverage", 0)),
                max_daily_loss=limits_cfg.get("max_daily_loss", global_limits.get("max_daily_loss", 0)),
                max_drawdown=limits_cfg.get("max_drawdown", global_limits.get("max_drawdown", 0)),
            ),
            starting_equity=cfg.get("starting_equity", global_cfg.get("starting_equity", 0)),
        )
    return engines


def append_event(path: Path, event: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(event) + "\n")


def load_orders(args: argparse.Namespace) -> pd.DataFrame:
    if args.trades_csv:
        trades_path = Path(args.trades_csv)
        trades = pd.read_csv(trades_path)

        if "direction" not in trades.columns and "side" in trades.columns:
            trades["direction"] = trades["side"]
        if "price_entry" not in trades.columns:
            if "entry_price" in trades.columns:
                trades["price_entry"] = trades["entry_price"]
            elif "entry" in trades.columns:
                trades["price_entry"] = trades["entry"]

        if "symbol" not in trades.columns:
            inferred = infer_symbol_from_path(trades_path)
            symbol = args.symbol or inferred
            if symbol:
                trades["symbol"] = symbol
            else:
                raise ValueError("trades_csv missing 'symbol' and no --symbol fallback provided")

        required = {"ts_entry", "ts_exit", "symbol", "direction", "qty", "price_entry"}
        missing = required - set(trades.columns)
        if missing:
            raise ValueError(f"trades_csv missing columns: {missing}")

        trades["direction"] = trades["direction"].str.lower()
        trades["price_entry"] = trades["price_entry"].astype(float)
        trades["qty"] = trades["qty"].astype(float)

        records: List[Dict] = []
        for _, row in trades.iterrows():
            direction = str(row["direction"]).lower()
            symbol = row["symbol"]
            qty = float(row["qty"])
            price_entry = float(row["price_entry"])
            notional = qty * price_entry
            strategy = row.get("strategy") if isinstance(row.get("strategy"), str) else "default"
            pnl_value = float(row["pnl"]) if not pd.isna(row.get("pnl")) else 0.0

            ts_entry = row.get("ts_entry")
            ts_exit = row.get("ts_exit")
            exit_price = row.get("exit")

            if pd.notna(ts_entry):
                side = "buy" if direction == "long" else "sell"
                records.append(
                    {
                        "ts": ts_entry,
                        "symbol": symbol,
                        "side": side,
                        "qty": qty,
                        "price": price_entry,
                        "notional": notional,
                        "pnl": 0.0,
                        "strategy": strategy,
                    }
                )

            if pd.notna(ts_exit):
                side = "sell" if direction == "long" else "buy"
                price_use = float(exit_price) if exit_price and not pd.isna(exit_price) else price_entry
                records.append(
                    {
                        "ts": ts_exit,
                        "symbol": symbol,
                        "side": side,
                        "qty": qty,
                        "price": price_use,
                        "notional": notional,
                        "pnl": pnl_value,
                        "strategy": strategy,
                    }
                )

        if not records:
            raise ValueError("No usable rows found in trades CSV.")
        return pd.DataFrame.from_records(records)
    if not args.orders:
        raise ValueError("Either --orders or --trades-csv must be provided")
    return pd.read_csv(Path(args.orders))


def simulate() -> None:
    args = parse_args()
    orders_df = load_orders(args)

    if args.risk_limits_yaml:
        engines = load_strategy_engines(Path(args.risk_limits_yaml))
        default_engine = engines.get("default")
    elif args.risk_config:
        engine = load_risk_engine(Path(args.risk_config))
        engines = {"default": engine}
        default_engine = engine
    else:
        raise ValueError("Provide either --risk-config or --risk-limits-yaml")

    latency_ms = args.paper_latency_ms if args.adapter == "paper" else 0.0
    if args.adapter == "paper":
        adapter = PaperAdapter(latency_ms=latency_ms, slippage_pips=args.paper_slippage_pips)
    else:
        adapter = MockAdapter()

    symbol_exposure_peaks: Dict[str, float] = {}
    max_gross_notional = 0.0
    total_pnl_sum = 0.0
    run_id = args.run_id or pd.Timestamp.now().strftime("exec_%Y%m%d_%H%M%S")
    run_dir = Path(f"results/execution/{run_id}")
    run_dir.mkdir(parents=True, exist_ok=True)
    risk_log_path = Path(args.risk_log)

    fills: List[Dict] = []
    rejects: List[Dict] = []
    kill_events: List[Dict] = []

    for _, row in orders_df.iterrows():
        strategy = row.get("strategy", "default")
        risk_engine = engines.get(strategy, default_engine)
        ok, reason = risk_engine.evaluate_order(row["symbol"], row["side"], row["notional"])
        if not ok:
            reject = {"ts": row["ts"], "symbol": row["symbol"], "strategy": strategy, "reason": reason}
            rejects.append(reject)
            append_event(risk_log_path, {"event": "reject", **reject})
            continue
        ack = adapter.submit(
            OrderParams(
                symbol=row["symbol"],
                side=row["side"],
                quantity=row["qty"],
                price=row.get("price"),
                metadata={"strategy": strategy},
            )
        )
        pnl = row.get("pnl", 0.0)
        risk_engine.record_fill(row["symbol"], row["side"], row["notional"], pnl)
        exposures = risk_engine.state.exposures
        current_gross = sum(abs(v) for v in exposures.values())
        max_gross_notional = max(max_gross_notional, current_gross)
        for sym, val in exposures.items():
            peak = symbol_exposure_peaks.get(sym, 0.0)
            symbol_exposure_peaks[sym] = max(peak, abs(val))
        total_pnl_sum += pnl
        ok_loss, reason_loss = risk_engine.check_loss_limits()
        if not ok_loss:
            event = {"event": "kill_switch", "strategy": strategy, "reason": reason_loss}
            kill_events.append(event)
            append_event(risk_log_path, event)
        lat = getattr(ack, "latency_ms", latency_ms)
        fills.append({
            "order_id": ack.order_id,
            "ts": row["ts"],
            "symbol": row["symbol"],
            "pnl": pnl,
            "strategy": strategy,
            "adapter_latency_ms": lat,
        })

    summary = {
        "fills": fills,
        "rejects": rejects,
        "kill_switch_events": kill_events,
        "run_id": run_id,
        "max_symbol_exposure": symbol_exposure_peaks,
        "max_gross_notional": max_gross_notional,
        "total_pnl": total_pnl_sum,
        "max_drawdown_pct": risk_engine.max_drawdown_pct(),
    }
    out_path = Path(args.output) if args.output else run_dir / "sim_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame(fills).to_csv(run_dir / "fills.csv", index=False)
    pd.DataFrame(rejects).to_csv(run_dir / "rejects.csv", index=False)
    print(f"Simulation summary saved to {out_path} (run_id={run_id})")


if __name__ == "__main__":
    simulate()
