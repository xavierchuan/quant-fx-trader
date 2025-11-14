"""
Paper trading driver that wires live OANDA pricing into the strategy engine.
"""

from __future__ import annotations

import argparse
import signal
import time
from queue import Empty, Queue

import pandas as pd
from loguru import logger
import yaml

import os
import sys

TRADER_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(TRADER_ROOT)
RESEARCH_ROOT = os.path.join(REPO_ROOT, "QuantResearch")
sys.path.extend([TRADER_ROOT, REPO_ROOT, RESEARCH_ROOT])

from data.oanda_stream import OandaPricingStream
from core.oanda_execution import OandaExecution
from core.events import TickEvent, OrderEvent
from QuantResearch.core.backtest.strategy_engine import (
    StrategyEngine,
    StrategySpec,
    parse_strategy_specs,
    _coerce_fx_rates,
    _merge_fx_rates,
)
from shared.utils.config import OANDA_ACCOUNT_ID, OANDA_TOKEN


class BarAggregator:
    """
    Aggregate tick data into fixed timeframe OHLC bars.
    """

    def __init__(self, symbol: str, timeframe: str):
        self.symbol = symbol.replace("_", "").upper()
        tf = timeframe.lower() if isinstance(timeframe, str) else timeframe
        self.timeframe = pd.to_timedelta(tf)
        if self.timeframe <= pd.Timedelta(0):
            raise ValueError(f"Invalid timeframe: {timeframe}")
        self.current_bucket: pd.Timestamp | None = None
        self.open = self.high = self.low = self.close = None
        self.last_ts: pd.Timestamp | None = None

    def update(self, tick: TickEvent) -> dict | None:
        ts = pd.Timestamp(tick.ts)
        bucket = ts.floor(self.timeframe)
        mid = (tick.bid + tick.ask) / 2.0
        if self.current_bucket is None:
            self._start_bar(bucket, mid, ts)
            return None
        if bucket != self.current_bucket:
            finished = self._build_bar()
            self._start_bar(bucket, mid, ts)
            return finished
        self._update_bar(mid, ts)
        return None

    def flush(self) -> dict | None:
        if self.current_bucket is None:
            return None
        return self._build_bar()

    def _start_bar(self, bucket: pd.Timestamp, price: float, ts: pd.Timestamp) -> None:
        self.current_bucket = bucket
        self.open = self.high = self.low = self.close = price
        self.last_ts = ts

    def _update_bar(self, price: float, ts: pd.Timestamp) -> None:
        self.high = max(self.high, price)
        self.low = min(self.low, price)
        self.close = price
        self.last_ts = ts

    def _build_bar(self) -> dict:
        bar = {
            "symbol": self.symbol,
            "ts": self.last_ts,
            "open": float(self.open),
            "high": float(self.high),
            "low": float(self.low),
            "close": float(self.close),
            "volume": 0,
        }
        return bar


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_engine(cfg: dict, args, execution_handler):
    symbol = args.symbol or cfg.get("symbol", "EURUSD")
    account_ccy = cfg.get("account_ccy", "USD")
    fast_win = int(cfg.get("fast", 50))
    slow_win = int(cfg.get("slow", 200))
    spread = float(cfg.get("spread", 1.0))
    slip = float(cfg.get("slip", 0.2))
    comm = float(cfg.get("comm", 2.0))
    qty = float(cfg.get("qty", 10_000))
    initial_cash = float(cfg.get("cash", 100_000))
    stop_loss_pips = cfg.get("sl", 50)
    take_profit_pips = cfg.get("tp")
    atr_sl = cfg.get("atr_sl")
    atr_tp = cfg.get("atr_tp")
    atr_window = int(cfg.get("atr_window", 14))
    regime_ema_window = int(cfg.get("regime_ema_window", 200))
    regime_slope_min = cfg.get("regime_slope_min")
    if regime_slope_min is not None:
        regime_slope_min = float(regime_slope_min)
    regime_atr_min = cfg.get("regime_atr_min")
    if regime_atr_min is not None:
        regime_atr_min = float(regime_atr_min)
    rsi_period = int(cfg.get("rsi_period", 14))
    rsi_long_thresh = cfg.get("rsi_long_thresh")
    if rsi_long_thresh is not None:
        rsi_long_thresh = float(rsi_long_thresh)
    rsi_short_thresh = cfg.get("rsi_short_thresh")
    if rsi_short_thresh is not None:
        rsi_short_thresh = float(rsi_short_thresh)
    enable_trailing = bool(cfg.get("enable_trailing", False))
    trailing_enable_atr_mult = float(cfg.get("trailing_enable_atr_mult", 1.0))
    trailing_atr_mult = float(cfg.get("trailing_atr_mult", 0.5))
    long_only_above_slow = bool(cfg.get("long_only_above_slow", False))
    slope_lookback = int(cfg.get("slope_lookback", 0))
    cooldown = int(cfg.get("cooldown", 0))
    allow_short = bool(cfg.get("allow_short", True))
    short_only_below_slow = bool(cfg.get("short_only_below_slow", False))
    risk_per_trade_pct = cfg.get("risk_per_trade_pct")
    max_drawdown_pct = cfg.get("max_drawdown_pct")
    max_position_units = cfg.get("max_position_units")
    htf_factor = int(cfg.get("htf_factor", 4))
    htf_ema_window = cfg.get("htf_ema_window")
    if htf_ema_window is not None:
        htf_ema_window = int(htf_ema_window)
    htf_rsi_period = cfg.get("htf_rsi_period")
    if htf_rsi_period is not None:
        htf_rsi_period = int(htf_rsi_period)

    cfg_fx_rates = _coerce_fx_rates(cfg.get("fx_rates"))
    cli_fx_rates = _coerce_fx_rates(args.fx_rate)
    fx_rates = _merge_fx_rates(cfg_fx_rates, cli_fx_rates)

    strategy_specs = parse_strategy_specs(cfg.get("strategies"))

    engine = StrategyEngine(
        symbol=symbol,
        fast_win=fast_win,
        slow_win=slow_win,
        spread_pips=spread,
        commission_per_million=comm,
        slippage_pips=slip,
        stop_loss_pips=stop_loss_pips,
        take_profit_pips=take_profit_pips,
        atr_sl=atr_sl,
        atr_tp=atr_tp,
        atr_window=atr_window,
        regime_ema_window=regime_ema_window,
        regime_slope_min=regime_slope_min,
        regime_atr_min=regime_atr_min,
        rsi_period=rsi_period,
        rsi_long_thresh=rsi_long_thresh,
        rsi_short_thresh=rsi_short_thresh,
        enable_trailing=enable_trailing,
        trailing_enable_atr_mult=trailing_enable_atr_mult,
        trailing_atr_mult=trailing_atr_mult,
        long_only_above_slow=long_only_above_slow,
        slope_lookback=slope_lookback,
        cooldown=cooldown,
        qty=qty,
        account_ccy=account_ccy,
        fx_rates=fx_rates,
        strategy_specs=strategy_specs,
        allow_short=allow_short,
        short_only_below_slow=short_only_below_slow,
        risk_per_trade_pct=risk_per_trade_pct,
        max_drawdown_pct=max_drawdown_pct,
        max_position_units=max_position_units,
        htf_factor=htf_factor,
        htf_ema_window=htf_ema_window,
        htf_rsi_period=htf_rsi_period,
        execution_handler=execution_handler,
    )
    engine.set_initial_cash(initial_cash)
    return engine, symbol, initial_cash


def main():
    parser = argparse.ArgumentParser(description="OANDA paper trading driver")
    parser.add_argument("--config", required=True, help="策略配置 YAML")
    parser.add_argument("--symbol", default=None, help="覆盖配置中的交易品种")
    parser.add_argument("--timeframe", default="60s", help="K线时间粒度，默认 60s")
    parser.add_argument("--environment", default="practice", choices=["practice", "live"], help="OANDA 环境")
    parser.add_argument("--fx-rate", action="append", default=None, help="额外汇率，示例 GBPUSD=1.27")
    parser.add_argument("--max-bars", type=int, default=None, help="最多生成多少根 bar 后自动停止")
    parser.add_argument("--log-heartbeat", action="store_true", help="打印 OANDA 心跳信息")
    args = parser.parse_args()

    cfg = load_config(args.config)
    token = OANDA_TOKEN
    account_id = OANDA_ACCOUNT_ID
    if not token or not account_id:
        raise RuntimeError("OANDA_TOKEN 或 OANDA_ACCOUNT_ID 未在环境变量中设置")

    order_queue: Queue = Queue()
    execution = OandaExecution(order_queue, account_id=account_id, access_token=token, environment=args.environment)
    engine, symbol, initial_cash = build_engine(cfg, args, execution.on_event)
    aggregator = BarAggregator(symbol, args.timeframe)

    tick_queue: Queue = Queue()
    stream = OandaPricingStream(
        tick_queue,
        account_id=account_id,
        instruments=[symbol],
        access_token=token,
        environment=args.environment,
        log_heartbeat=args.log_heartbeat,
    )

    stop_flag = False

    def handle_sigterm(signum, frame):
        nonlocal stop_flag
        stop_flag = True

    signal.signal(signal.SIGINT, handle_sigterm)
    signal.signal(signal.SIGTERM, handle_sigterm)

    logger.info(f"[Paper] Starting pricing stream for {symbol} ({args.timeframe})")
    stream.start()

    bars_processed = 0
    try:
        while not stop_flag:
            try:
                tick = tick_queue.get(timeout=1.0)
            except Empty:
                continue
            if not isinstance(tick, TickEvent):
                continue
            bar = aggregator.update(tick)
            if bar:
                engine.handle_bar(bar)
                bars_processed += 1
                if args.max_bars and bars_processed >= args.max_bars:
                    logger.info("[Paper] Reached max bar limit, stopping.")
                    break
    finally:
        stream.stop()

    # flush last partially built bar
    final_bar = aggregator.flush()
    if final_bar:
        engine.handle_bar(final_bar)

    engine.finalize()
    suffix = engine.compute_suffix()
    engine.export_outputs(
        fast_win=int(cfg.get("fast", 50)),
        slow_win=int(cfg.get("slow", 200)),
        suffix=suffix,
    )
    result = engine.summary(
        fast_win=int(cfg.get("fast", 50)),
        slow_win=int(cfg.get("slow", 200)),
        suffix=suffix,
    )
    final_equity = result["final_equity"] if result["final_equity"] is not None else engine.cash
    ret_pct = (final_equity / initial_cash - 1.0) * 100.0
    logger.info(f"[Paper] Bars processed: {engine.bar_count}, Trades executed: {engine.trade_count}")
    logger.info(f"[Paper] Final equity: {final_equity:.2f} ({ret_pct:.2f}%)")

    # Drain any fills left in queue
    fills = []
    while True:
        try:
            fill = order_queue.get_nowait()
        except Empty:
            break
        else:
            fills.append(fill)
    if fills:
        for fill in fills:
            logger.info(f"[Paper] Fill received: {fill}")


if __name__ == "__main__":
    main()
