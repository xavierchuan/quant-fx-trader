from __future__ import annotations

import argparse
import itertools
import os
import sys
from copy import deepcopy
from pathlib import Path
from queue import Empty, Queue
from typing import Dict, Iterable, List, Optional

import pandas as pd
import yaml
from loguru import logger

# 允许直接 import 项目内模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.backtest.strategy_engine import StrategyEngine, parse_strategy_specs, _coerce_fx_rates  # noqa: E402
from data.csv_feed import CSVFeed  # noqa: E402
from metrics.perf import trade_stats  # noqa: E402


BASE_CONFIG_PATH = Path("config/optimized_eurusd_v2_with_rsi.yaml")
DEFAULT_SYMBOL = "EURUSD"
DEFAULT_CSV = Path("data/raw/EURUSD_H1.csv")

# 重点围绕 ATR / RSI / cooldown 进行调参
PARAM_GRID = {
    "fast": [20],
    "slow": [120],
    "atr_sl": [1.0, 1.3, 1.6],
    "atr_tp": [None, 3.0, 4.5],
    "rsi_long_thresh": [60],
    "rsi_short_thresh": [40],
    "cooldown": [12, 24, 36],
    "trailing_enable_atr_mult": [0.5],
    "trailing_atr_mult": [0.5],
    "htf_factor": [4],
    "size_tier_mode": ["base"],
    "boll_window": [32, 48, 64],
    "boll_enter_z": [1.0, 1.3, 1.6],
    "boll_exit_z": [0.2, 0.4],
    "boll_allow_short": [False],
}

SIZE_TIER_PRESETS = {
    "base": {
        "base_size_mult": 1.0,
        "size_tiers": [{"size_mult": 1.0}],
    },
    "balanced": {
        "base_size_mult": 1.0,
        "size_tiers": [
            {"name": "strong", "min_atr_pct": 0.55, "min_trend_bars": 6, "size_mult": 1.3},
            {"name": "base", "size_mult": 1.0},
        ],
    },
    "aggressive": {
        "base_size_mult": 1.0,
        "size_tiers": [
            {"name": "strong", "min_atr_pct": 0.6, "min_trend_bars": 8, "size_mult": 1.6},
            {"name": "mid", "min_atr_pct": 0.45, "min_trend_strength": 0.00008, "size_mult": 1.2},
        ],
    },
}


def _apply_size_tier_mode(cfg: Dict[str, object], mode: Optional[str]) -> None:
    if not mode:
        return
    preset = SIZE_TIER_PRESETS.get(mode)
    if not preset:
        logger.warning(f"[GRID] 未知 size_tier 模式: {mode}")
        return
    strategies = cfg.get("strategies")
    if not isinstance(strategies, list):
        return
    for strat in strategies:
        if isinstance(strat, dict) and strat.get("name") == "regime_sma":
            params = strat.setdefault("params", {})
            tiers = preset.get("size_tiers")
            if tiers is not None:
                params["size_tiers"] = deepcopy(tiers)
            if "base_size_mult" in preset:
                params["base_size_mult"] = preset["base_size_mult"]


def _apply_bollinger_params(cfg: Dict[str, object], overrides: Dict[str, object]) -> None:
    strategies = cfg.get("strategies")
    if not isinstance(strategies, list):
        return
    for strat in strategies:
        if isinstance(strat, dict) and strat.get("name") == "bollinger_mean_revert":
            params = strat.setdefault("params", {})
            if overrides.get("window") is not None:
                params["window"] = int(overrides["window"])
            if overrides.get("enter_z") is not None:
                params["enter_z"] = float(overrides["enter_z"])
            if overrides.get("exit_z") is not None:
                params["exit_z"] = float(overrides["exit_z"])
            if overrides.get("allow_short") is not None:
                params["allow_short"] = bool(overrides["allow_short"])


def _to_float(value: Optional[object]) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, str) and value.lower() in {"none", "null", ""}:
        return None
    return float(value)


def load_base_config(path: Path = BASE_CONFIG_PATH) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def evaluate_config(base_cfg: Dict[str, object], overrides: Dict[str, object], symbol: str) -> Optional[Dict[str, object]]:
    cfg = deepcopy(base_cfg)
    local_overrides = dict(overrides)
    size_mode = local_overrides.pop("size_tier_mode", None)
    boll_window = local_overrides.pop("boll_window", None)
    boll_enter = local_overrides.pop("boll_enter_z", None)
    boll_exit = local_overrides.pop("boll_exit_z", None)
    boll_allow_short = local_overrides.pop("boll_allow_short", None)
    cfg.update(local_overrides)
    if size_mode:
        _apply_size_tier_mode(cfg, size_mode)
    if any(v is not None for v in [boll_window, boll_enter, boll_exit, boll_allow_short]):
        _apply_bollinger_params(
            cfg,
            {
                "window": boll_window,
                "enter_z": boll_enter,
                "exit_z": boll_exit,
                "allow_short": boll_allow_short,
            },
        )

    symbol = cfg.get("symbol", DEFAULT_SYMBOL)
    csv_path = Path(cfg.get("csv", DEFAULT_CSV))
    if not csv_path.exists():
        logger.error(f"CSV 路径不存在: {csv_path}")
        return None

    initial_cash = float(cfg.get("cash", 100_000))
    qty = float(cfg.get("qty", 10_000))
    account_ccy = cfg.get("account_ccy", "USD")
    fast = int(cfg.get("fast", 20))
    slow = int(cfg.get("slow", 150))
    spread = float(cfg.get("spread", 1.0))
    slip = float(cfg.get("slip", 0.2))
    comm = float(cfg.get("comm", 2.0))
    stop_loss_pips = _to_float(cfg.get("sl"))
    take_profit_pips = _to_float(cfg.get("tp"))
    atr_sl = _to_float(cfg.get("atr_sl"))
    atr_tp = _to_float(cfg.get("atr_tp"))
    atr_window = int(cfg.get("atr_window", 14))
    rsi_period = int(cfg.get("rsi_period", 14))
    rsi_long = _to_float(cfg.get("rsi_long_thresh"))
    rsi_short = _to_float(cfg.get("rsi_short_thresh"))
    enable_trailing = bool(cfg.get("enable_trailing", False))
    trailing_enable = float(cfg.get("trailing_enable_atr_mult", 1.0))
    trailing_mult = float(cfg.get("trailing_atr_mult", 0.5))
    slope_lookback = int(cfg.get("slope_lookback", 0))
    cooldown = int(cfg.get("cooldown", 0))
    allow_short = bool(cfg.get("allow_short", True))
    long_only_above_slow = bool(cfg.get("long_only_above_slow", False))
    short_only_below_slow = bool(cfg.get("short_only_below_slow", False))
    risk_per_trade_pct = _to_float(cfg.get("risk_per_trade_pct"))
    max_drawdown_pct = _to_float(cfg.get("max_drawdown_pct"))
    max_position_units = _to_float(cfg.get("max_position_units"))

    cfg_fx_rates = _coerce_fx_rates(cfg.get("fx_rates"))
    fx_rates = cfg_fx_rates if cfg_fx_rates else None

    strategy_specs = parse_strategy_specs(cfg.get("strategies"))

    engine = StrategyEngine(
        symbol=symbol,
        fast_win=fast,
        slow_win=slow,
        spread_pips=spread,
        commission_per_million=comm,
        slippage_pips=slip,
        stop_loss_pips=stop_loss_pips,
        take_profit_pips=take_profit_pips,
        atr_sl=atr_sl,
        atr_tp=atr_tp,
        atr_window=atr_window,
        rsi_period=rsi_period,
        rsi_long_thresh=rsi_long,
        rsi_short_thresh=rsi_short,
        enable_trailing=enable_trailing,
        trailing_enable_atr_mult=trailing_enable,
        trailing_atr_mult=trailing_mult,
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
    )
    engine.set_initial_cash(initial_cash)

    q: Queue = Queue()
    feed = CSVFeed(q, path=str(csv_path), symbol=symbol)
    feed.start()

    try:
        while True:
            try:
                event = q.get(timeout=0.05)
            except Empty:
                if hasattr(feed, "pump"):
                    feed.pump(n=100)
                if getattr(feed, "finished", False):
                    break
                continue

            if event.get("type") != "bar":
                continue
            engine.handle_bar(event)
    finally:
        engine.finalize()

    summary = engine.summary(fast, slow)
    stats = trade_stats(engine.trade_log) if engine.trade_log else {}

    final_equity = summary.get("final_equity", engine.cash)
    ret_pct = (final_equity / initial_cash - 1.0) if initial_cash else None

    return {
        "params": overrides,
        "summary": summary,
        "stats": stats,
        "final_equity": final_equity,
        "return_pct": ret_pct,
        "symbol": symbol,
    }


def param_product(grid: Dict[str, Iterable[object]]) -> Iterable[Dict[str, object]]:
    keys = list(grid.keys())
    for combo in itertools.product(*(grid[k] for k in keys)):
        params = dict(zip(keys, combo))
        fast_v = params.get("fast")
        slow_v = params.get("slow")
        if fast_v is not None and slow_v is not None:
            if float(fast_v) >= float(slow_v):
                continue
        long_v = params.get("rsi_long_thresh")
        short_v = params.get("rsi_short_thresh")
        if long_v is not None and short_v is not None and long_v <= short_v:
            continue
        yield params


def run_grid(base_cfg: Dict[str, object], save_suffix: Optional[str] = None, param_grid: Optional[Dict[str, List[object]]] = None) -> Path:
    # 降低日志噪音
    logger.remove()
    logger.add(sys.stderr, level="WARNING")

    results: List[Dict[str, object]] = []
    grid_def = deepcopy(param_grid or PARAM_GRID)
    combos = list(param_product(grid_def))
    total = len(combos)
    print(f"将测试 {total} 种参数组合")

    suffix = f"_{save_suffix}" if save_suffix else ""
    out_dir = Path("data/grid")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_filename = f"grid_rsi_trailing_diagnostics{suffix}.csv"

    def fmt_pct(value: Optional[float]) -> str:
        return f"{value:.2%}" if value is not None else "NA"

    def fmt_float(value: Optional[float], digits: int = 3) -> str:
        return f"{value:.{digits}f}" if value is not None else "NA"

    for idx, params in enumerate(combos, 1):
        print(f"\n[{idx}/{total}] 评估参数: {params}")
        res = evaluate_config(base_cfg, params, base_cfg.get("symbol", DEFAULT_SYMBOL))
        if not res:
            print("  -> 运行失败")
            continue
        summary = res["summary"]
        stats = res["stats"]
        sharpe = summary.get("sharpe")
        win_rate = stats.get("win_rate")
        exp = stats.get("expectancy")
        trades = summary.get("trades")
        ret_pct = res["return_pct"]
        dd = summary.get("max_drawdown")
        print(
            "  -> Sharpe={}  回撤={}  胜率={}  期望={}  交易数={}".format(
                fmt_float(sharpe),
                fmt_pct(dd),
                fmt_pct(win_rate),
                fmt_float(exp, digits=2),
                trades if trades is not None else "NA",
            )
        )
        res_record = {
            "sharpe": sharpe,
            "ann_return": summary.get("ann_return"),
            "ann_vol": summary.get("ann_vol"),
            "max_drawdown": summary.get("max_drawdown"),
            "trades": trades,
            "return_pct": ret_pct,
            "win_rate": win_rate,
            "rr": stats.get("rr"),
            "expectancy": exp,
            "median_hold": stats.get("median_hold"),
            "symbol": res.get("symbol", base_cfg.get("symbol", DEFAULT_SYMBOL)),
            **params,
            "source_file": out_filename,
        }
        results.append(res_record)

    if not results:
        print("\n未得到任何有效结果。")
        return

    df = pd.DataFrame(results)
    df["sharpe"] = pd.to_numeric(df["sharpe"], errors="coerce")
    df["expectancy"] = pd.to_numeric(df["expectancy"], errors="coerce")
    df = df.sort_values(by="sharpe", ascending=False)
    out_path = out_dir / out_filename
    df.to_csv(out_path, index=False)

    print(f"\n已保存全部结果 -> {out_path}")
    top_n = df.head(5)
    print("\nTop 5 组合:")
    for _, row in top_n.iterrows():
        win_rate_val = row.get("win_rate")
        win_rate_val = None if pd.isna(win_rate_val) else win_rate_val
        cooldown_val = row.get("cooldown")
        cooldown_disp = int(cooldown_val) if cooldown_val is not None and not pd.isna(cooldown_val) else "NA"
        atr_sl_disp = fmt_float(row.get("atr_sl"), digits=2)
        atr_tp_val = row.get("atr_tp")
        atr_tp_disp = "None" if atr_tp_val is None or (isinstance(atr_tp_val, float) and pd.isna(atr_tp_val)) else fmt_float(atr_tp_val, digits=2)
        rsi_short = row.get("rsi_short_thresh")
        rsi_long = row.get("rsi_long_thresh")
        fast_val = row.get("fast")
        slow_val = row.get("slow")
        print(
            "  Sharpe={}  回撤={}  胜率={}  fast/slow={}/{}  cooldown={}  atr_sl={}  atr_tp={}  RSI=({}/{})".format(
                fmt_float(row.get("sharpe")),
                fmt_pct(row.get("max_drawdown")),
                fmt_pct(win_rate_val),
                fmt_float(fast_val, digits=0) if fast_val is not None and not pd.isna(fast_val) else "NA",
                fmt_float(slow_val, digits=0) if slow_val is not None and not pd.isna(slow_val) else "NA",
                cooldown_disp,
                atr_sl_disp,
                atr_tp_disp,
                fmt_float(rsi_short, digits=1) if rsi_short is not None and not pd.isna(rsi_short) else "NA",
                fmt_float(rsi_long, digits=1) if rsi_long is not None and not pd.isna(rsi_long) else "NA",
            )
        )

    usdjpy_mask = (
        df["symbol"].astype(str).str.upper().eq("USDJPY")
        & df["sharpe"].gt(1.5)
        & df["expectancy"].gt(2.0)
    )
    top_usdjpy = df.loc[usdjpy_mask].head(3)
    if not top_usdjpy.empty:
        print("\nUSDJPY Sharpe>1.5 & Expectancy>$2 (Top 3):")
        for _, row in top_usdjpy.iterrows():
            print(
                "  Sharpe={:.3f}  Expectancy=${:.2f}  Trades={}  atr_sl={}  atr_tp={}  cooldown={}".format(
                    row["sharpe"],
                    row["expectancy"],
                    row.get("trades", "NA"),
                    fmt_float(row.get("atr_sl"), digits=2),
                    "None" if pd.isna(row.get("atr_tp")) or row.get("atr_tp") is None else fmt_float(row.get("atr_tp"), digits=2),
                    int(row.get("cooldown")) if row.get("cooldown") is not None and not pd.isna(row.get("cooldown")) else "NA",
                )
            )
        best_out = out_dir / "grid_usdjpy_top3.csv"
        top_usdjpy.to_csv(best_out, index=False)
        print(f"\n已保存 USDJPY 筛选结果 -> {best_out}")
    else:
        print("\nUSDJPY 暂无满足 Sharpe>1.5 & Expectancy>$2 的组合。")

    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grid search for ATR/RSI/trailing parameters.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(BASE_CONFIG_PATH),
        help="YAML 配置路径（默认使用 optimized_eurusd_v2_with_rsi.yaml）",
    )
    parser.add_argument("--symbol", type=str, default=None, help="覆盖配置中的 symbol（可选）")
    parser.add_argument("--csv", type=str, default=None, help="覆盖配置中的 csv 路径（可选）")
    parser.add_argument("--suffix", type=str, default=None, help="输出文件名后缀（默认取 symbol）")
    parser.add_argument("--atr-sl", type=str, default=None, help="自定义 atr_sl 列表，例如 '1.0,1.3,1.6'")
    parser.add_argument("--atr-tp", type=str, default=None, help="自定义 atr_tp 列表，例如 'None,3.0,4.0'")
    parser.add_argument("--cooldown-list", type=str, default=None, help="自定义 cooldown 列表，例如 '12,24,36'")
    parser.add_argument("--htf-factor-list", type=str, default=None, help="自定义 htf_factor 列表，例如 '2,4,6'")
    parser.add_argument("--size-tier-mode-list", type=str, default=None, help="size_tier 模式列表，例如 'base,aggressive'")
    parser.add_argument("--boll-window-list", type=str, default=None, help="Bollinger 窗口列表，例如 '32,48,64'")
    parser.add_argument("--boll-enter-list", type=str, default=None, help="Bollinger 入场 Z 值列表，例如 '1.0,1.3'")
    parser.add_argument("--boll-exit-list", type=str, default=None, help="Bollinger 退出 Z 值列表，例如 '0.2,0.4'")
    parser.add_argument("--boll-allow-short", type=str, default=None, help="Bollinger 是否允许做空，示例 'true,false'")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg_path = Path(args.config)
    base_cfg = load_base_config(cfg_path)
    if args.symbol:
        base_cfg["symbol"] = args.symbol
    if args.csv:
        base_cfg["csv"] = args.csv

    suffix = args.suffix or base_cfg.get("symbol")

    grid_override = deepcopy(PARAM_GRID)

    def _parse_float_list(raw: Optional[str]) -> Optional[List[Optional[float]]]:
        if raw is None:
            return None
        values: List[Optional[float]] = []
        for token in raw.split(","):
            token = token.strip()
            if not token:
                continue
            if token.lower() in {"none", "null"}:
                values.append(None)
            else:
                values.append(float(token))
        return values or None

    atr_sl_list = _parse_float_list(args.atr_sl)
    atr_tp_list = _parse_float_list(args.atr_tp)
    cooldown_list = None
    if args.cooldown_list:
        cooldown_list = [int(item.strip()) for item in args.cooldown_list.split(",") if item.strip()]
    htf_factor_list = None
    if args.htf_factor_list:
        htf_factor_list = [int(item.strip()) for item in args.htf_factor_list.split(",") if item.strip()]
    size_mode_list = None
    if args.size_tier_mode_list:
        size_mode_list = [item.strip() for item in args.size_tier_mode_list.split(",") if item.strip()]
    boll_window_list = None
    if args.boll_window_list:
        boll_window_list = [int(item.strip()) for item in args.boll_window_list.split(",") if item.strip()]
    boll_enter_list = _parse_float_list(args.boll_enter_list)
    boll_exit_list = _parse_float_list(args.boll_exit_list)
    boll_allow_short_list = None
    if args.boll_allow_short:
        mapping = {"true": True, "false": False, "1": True, "0": False}
        boll_allow_short_list = [
            mapping.get(item.strip().lower(), item.strip().lower() in {"true", "1"}) for item in args.boll_allow_short.split(",") if item.strip()
        ]

    if atr_sl_list:
        grid_override["atr_sl"] = atr_sl_list
    if atr_tp_list:
        grid_override["atr_tp"] = atr_tp_list
    if cooldown_list:
        grid_override["cooldown"] = cooldown_list
    if htf_factor_list:
        grid_override["htf_factor"] = htf_factor_list
    if size_mode_list:
        grid_override["size_tier_mode"] = size_mode_list
    if boll_window_list:
        grid_override["boll_window"] = boll_window_list
    if boll_enter_list:
        grid_override["boll_enter_z"] = boll_enter_list
    if boll_exit_list:
        grid_override["boll_exit_z"] = boll_exit_list
    if boll_allow_short_list:
        grid_override["boll_allow_short"] = boll_allow_short_list

    out_csv = run_grid(base_cfg, suffix, grid_override)
    print(f"结果已写入: {out_csv}")
