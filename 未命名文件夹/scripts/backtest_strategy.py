from __future__ import annotations

import os
import sys
from pathlib import Path
from datetime import datetime, timezone

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import pandas as pd
import yaml
import numpy as np
from queue import Empty, Queue
from typing import Any, List, Optional
from loguru import logger

from core.backtest.strategy_engine import (
    FXRateProvider,
    StrategyEngine,
    StrategySpec,
    parse_strategy_specs,
    _coerce_fx_rates,
    _merge_fx_rates,
)
from data.csv_feed import CSVFeed  # 你的CSVFeed
from scripts.validate_dataset import (
    DEFAULT_MANIFEST,
    compute_report,
    load_manifest_entry,
)

# ===== FX 元数据与换算工具 =====
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
DERIVED_DATA_DIR = os.path.join(DATA_DIR, "derived")
OUTPUT_DIR = os.path.join(DATA_DIR, "outputs")
EQUITY_DIR = os.path.join(OUTPUT_DIR, "equity")
TRADES_DIR = os.path.join(OUTPUT_DIR, "trades")
STATS_DIR = os.path.join(OUTPUT_DIR, "stats")
DATA_REPORT_DIR = os.path.join(STATS_DIR, "data_reports")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
GRID_DIR = os.path.join(DATA_DIR, "grid")
PARAMS_DIR = os.path.join(DATA_DIR, "params")

for _dir in [RAW_DATA_DIR, DERIVED_DATA_DIR, EQUITY_DIR, TRADES_DIR, STATS_DIR, DATA_REPORT_DIR, RESULTS_DIR, GRID_DIR, PARAMS_DIR]:
    os.makedirs(_dir, exist_ok=True)


def _load_manifest_entry(csv_path: Path, manifest_path: Optional[str]) -> Optional[dict]:
    if not manifest_path:
        manifest_path = os.path.join(DATA_DIR, "_manifest.json")
    manifest_file = Path(manifest_path)
    if not manifest_file.exists():
        return None
    try:
        entry = load_manifest_entry(manifest_file, csv_path)
        return entry
    except Exception as exc:
        logger.warning(f"Failed to read manifest entry for {csv_path}: {exc}")
        return None


def _validate_input_dataset(csv_path: str, manifest_path: Optional[str] = DEFAULT_MANIFEST) -> dict:
    dataset_path = Path(csv_path).expanduser().resolve()
    manifest_entry = _load_manifest_entry(dataset_path, manifest_path)
    report = compute_report(dataset_path, manifest_entry, z_threshold=5.0)
    severity = report.get("severity")
    gap_ratio = report.get("gap_ratio")
    gap_ratio_str = f"{gap_ratio:.4f}" if isinstance(gap_ratio, (int, float)) else "n/a"
    logger.info(
        f"Data validation severity={severity} "
        f"duplicates={report.get('duplicate_timestamps')} gap_ratio={gap_ratio_str}"
    )
    if severity == "error":
        raise RuntimeError(
            f"Dataset validation failed for {dataset_path}. Messages: {report.get('messages')}"
        )
    return report


def _relpath_or_abs(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    try:
        return str(Path(path).resolve().relative_to(BASE_DIR))
    except Exception:
        return str(path)


def _load_structured_data(path: Optional[str]):
    if not path:
        return None
    file_path = Path(path).expanduser()
    if not file_path.exists():
        raise FileNotFoundError(f"Config file not found: {file_path}")
    with file_path.open("r", encoding="utf-8") as fh:
        if file_path.suffix.lower() in (".yaml", ".yml"):
            return yaml.safe_load(fh)
        return json.load(fh)


def _write_data_report(report: Optional[dict], symbol: str, fast_win: int, slow_win: int, suffix: str) -> Optional[str]:
    if not report:
        return None
    report_path = Path(DATA_REPORT_DIR) / f"data_{symbol}_H1_{fast_win}x{slow_win}_{suffix}.json"
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    try:
        return str(report_path.relative_to(BASE_DIR))
    except ValueError:
        return str(report_path)


def _prepare_run_dir(enabled: bool, results_dir: Optional[str]) -> tuple[Optional[str], Optional[Path]]:
    if not enabled or not results_dir:
        return None, None
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_path = Path(results_dir) / run_id
    run_path.mkdir(parents=True, exist_ok=True)
    return run_id, run_path


def _write_run_summary(run_path: Path, summary: dict) -> str:
    summary_path = run_path / "summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    metrics_path = run_path / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(summary.get("metrics", {}), fh, indent=2, ensure_ascii=False)
    return str(summary_path)


def run_once(
    symbol: str = "EURUSD",
    csv_path: str = os.path.join(RAW_DATA_DIR, "EURUSD_H1.csv"),
    initial_cash: float = 100000.0,
    qty: int = 10_000,
    account_ccy: str = "USD",
    fx_rates: FXRateProvider = None,
    fast_win: int = 20,      # 改为更敏感的短期均线
    slow_win: int = 100,     # 改为中期均线
    spread_pips: float = 2.0,
    commission_per_million: float = 0.25,
    slippage_pips: float = 0.3,
    stop_loss_pips: float = 50,
    take_profit_pips: float | None = None,
    atr_sl: float | None = 1.5,      # 默认使用1.5倍ATR止损
    atr_tp: float | None = 3.0,      # 默认使用3倍ATR止盈
    atr_window: int = 14,            # ATR窗口保持14天
    # RSI & trailing defaults
    rsi_period: int = 14,
    rsi_long_thresh: Optional[float] = None,
    rsi_short_thresh: Optional[float] = None,
    enable_trailing: bool = False,
    trailing_enable_atr_mult: float = 1.0,
    trailing_atr_mult: float = 0.5,
    htf_factor: int = 4,
    htf_ema_window: Optional[int] = None,
    htf_rsi_period: Optional[int] = None,
    regime_ema_window: int = 200,
    regime_slope_min: Optional[float] = None,
    regime_atr_min: Optional[float] = None,
    regime_atr_percentile_min: Optional[float] = None,
    regime_atr_percentile_window: int = 500,
    regime_trend_min_bars: int = 0,
    long_only_above_slow: bool = False,
    slope_lookback: int = 0,
    cooldown: int = 0,
    allow_short: bool = True,
    short_only_below_slow: bool = False,
    strategies: Optional[List[StrategySpec]] = None,
    cost_profiles: Optional[Any] = None,
    slippage_model: Optional[Any] = None,
    strategy_mode: str = "first_hit",
    strategy_vote_threshold: float = 0.0,
    stress_cost_spread_mult: float = 1.0,
    stress_cost_comm_mult: float = 1.0,
    stress_slippage_mult: float = 1.0,
    stress_price_vol_mult: float = 1.0,
    stress_skip_trade_pct: float = 0.0,
    risk_per_trade_pct: Optional[float] = None,
    max_drawdown_pct: Optional[float] = None,
    max_position_units: Optional[float] = None,
    skip_outlier_entries: bool = False,
    validate_data: bool = True,
    manifest_path: Optional[str] = DEFAULT_MANIFEST,
    results_dir: Optional[str] = RESULTS_DIR,
    write_summary: bool = True,
):
    os.makedirs(EQUITY_DIR, exist_ok=True)
    os.makedirs(TRADES_DIR, exist_ok=True)
    os.makedirs(STATS_DIR, exist_ok=True)

    run_id, run_dir = _prepare_run_dir(write_summary, results_dir)

    data_report = None
    if validate_data and csv_path:
        try:
            data_report = _validate_input_dataset(csv_path, manifest_path)
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(f"Data validation error: {exc}") from exc

    engine = StrategyEngine(
        symbol=symbol,
        fast_win=fast_win,
        slow_win=slow_win,
        spread_pips=spread_pips,
        commission_per_million=commission_per_million,
        slippage_pips=slippage_pips,
        stop_loss_pips=stop_loss_pips,
        take_profit_pips=take_profit_pips,
        atr_sl=atr_sl,
        atr_tp=atr_tp,
        atr_window=atr_window,
        regime_ema_window=regime_ema_window,
        regime_slope_min=regime_slope_min,
        regime_atr_min=regime_atr_min,
        regime_atr_percentile_min=regime_atr_percentile_min,
        regime_atr_percentile_window=regime_atr_percentile_window,
        regime_trend_min_bars=regime_trend_min_bars,
        rsi_period=rsi_period,
        rsi_long_thresh=rsi_long_thresh,
        rsi_short_thresh=rsi_short_thresh,
        enable_trailing=enable_trailing,
        trailing_enable_atr_mult=trailing_enable_atr_mult,
        trailing_atr_mult=trailing_atr_mult,
        htf_factor=htf_factor,
        htf_ema_window=htf_ema_window,
        htf_rsi_period=htf_rsi_period,
        long_only_above_slow=long_only_above_slow,
        slope_lookback=slope_lookback,
        cooldown=cooldown,
        qty=qty,
        account_ccy=account_ccy,
        fx_rates=fx_rates,
        strategy_specs=strategies,
        cost_profiles=cost_profiles,
        slippage_model=slippage_model,
        strategy_combine_mode=strategy_mode,
        strategy_vote_threshold=strategy_vote_threshold,
        stress_cost_spread_mult=stress_cost_spread_mult,
        stress_cost_comm_mult=stress_cost_comm_mult,
        stress_slippage_mult=stress_slippage_mult,
        stress_price_vol_mult=stress_price_vol_mult,
        stress_skip_trade_pct=stress_skip_trade_pct,
        skip_outlier_bars=skip_outlier_entries,
        allow_short=allow_short,
        short_only_below_slow=short_only_below_slow,
        risk_per_trade_pct=risk_per_trade_pct,
        max_drawdown_pct=max_drawdown_pct,
        max_position_units=max_position_units,
        output_dirs={
            "equity": EQUITY_DIR,
            "trades": TRADES_DIR,
            "stats": STATS_DIR,
        },
    )
    engine.set_initial_cash(initial_cash)

    q = Queue()
    data = CSVFeed(q, path=str(csv_path), symbol=symbol)
    logger.info(f"Using CSV: {csv_path} for {symbol}")
    data.start()

    while True:
        try:
            ev = q.get(timeout=0.05)
        except Empty:
            if hasattr(data, "pump"):
                data.pump(n=50)
            if getattr(data, "finished", False):
                break
            continue
        if ev.get("type") != "bar":
            continue
        engine.handle_bar(ev)

    engine.finalize()

    suffix = engine.compute_suffix()
    output_files = engine.export_outputs(fast_win, slow_win, suffix)
    data_report_path = _write_data_report(data_report, symbol, fast_win, slow_win, suffix)
    result = engine.summary(fast_win, slow_win, suffix)

    final_equity = result["final_equity"] if result["final_equity"] is not None else engine.cash
    ret_pct = (final_equity / initial_cash - 1.0) * 100.0
    logger.info(f"Bars processed: {engine.bar_count}, Trades executed: {engine.trade_count}")
    logger.info(f"策略最终权益: {final_equity:.2f}，累计收益: {ret_pct:.4f}%")
    if all(result.get(k) is not None for k in ("sharpe", "ann_return", "ann_vol", "max_drawdown")):
        logger.info(
            f"Sharpe={result['sharpe']:.3f}  AnnRet={result['ann_return']*100:.2f}%  "
            f"AnnVol={result['ann_vol']*100:.2f}%  MaxDD={result['max_drawdown']*100:.2f}%"
        )

    data_summary = None
    if data_report:
        manifest_info = (data_report.get("manifest") or {})
        data_summary = {
            "severity": data_report.get("severity"),
            "gap_ratio": data_report.get("gap_ratio"),
            "duplicate_timestamps": data_report.get("duplicate_timestamps"),
            "hash": manifest_info.get("sha256"),
            "path": manifest_info.get("path"),
        }
        logger.info(
            "Data signature: severity={} hash={}",
            data_summary["severity"],
            data_summary["hash"],
        )

    if data_report_path:
        result["data_report"] = data_report_path
        result["data_validation"] = {
            "severity": data_summary["severity"] if data_summary else None,
            "messages": data_report.get("messages") if data_report else None,
        }

    if run_dir:
        param_snapshot = {
            "fast_win": fast_win,
            "slow_win": slow_win,
            "spread_pips": spread_pips,
            "commission_per_million": commission_per_million,
            "slippage_pips": slippage_pips,
            "stop_loss_pips": stop_loss_pips,
            "take_profit_pips": take_profit_pips,
            "atr_sl": atr_sl,
            "atr_tp": atr_tp,
            "atr_window": atr_window,
            "rsi_period": rsi_period,
            "regime_ema_window": regime_ema_window,
            "skip_outlier_entries": skip_outlier_entries,
            "strategy_mode": strategy_mode,
            "strategy_vote_threshold": strategy_vote_threshold,
            "stress_cost_spread_mult": stress_cost_spread_mult,
            "stress_cost_comm_mult": stress_cost_comm_mult,
            "stress_slippage_mult": stress_slippage_mult,
            "stress_price_vol_mult": stress_price_vol_mult,
            "stress_skip_trade_pct": stress_skip_trade_pct,
        }
        summary = {
            "run_id": run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "csv_path": os.path.relpath(csv_path, BASE_DIR) if csv_path else None,
            "parameters": param_snapshot,
            "metrics": result,
            "data_report": data_summary,
            "artifacts": {
                "equity": _relpath_or_abs(output_files.get("equity")),
                "trades": _relpath_or_abs(output_files.get("trades")),
                "trade_stats": _relpath_or_abs(output_files.get("trade_stats")),
            },
        }
        summary_path = _write_run_summary(run_dir, summary)
        result["run_id"] = run_id
        result["summary_path"] = summary_path

    return result

def main(**kwargs):
    """
    Backwards-compatible wrapper for legacy callers that imported
    scripts.backtest_strategy.main. It simply proxies to run_once().
    """
    return run_once(**kwargs)

def grid_search(symbol="EURUSD",
                csv_path=None,
                qty=10_000,
                initial_cash=100000.0,
                account_ccy="USD",
                fx_rates: FXRateProvider = None,
                # 单值默认；若传入 *_list 则以列表为准
                spread=1.0,
                slip=0.2,
                comm=2.0,
                atr_window=14,
                skip_outlier_entries: bool = False,
                # 维度开关；传 None 使用默认网格
                fast_list=None,
                slow_list=None,
                atr_sl_list=None,
                atr_tp_list=None,
                long_only_list=None,
                cooldown_list=None,
                slope_list=None,
                spread_list=None,
                slip_list=None,
                comm_list=None):
    """
    多维参数网格搜索。
    - 若 *_list 为 None，则采用合理的默认网格；否则使用传入列表。
    - 结果会输出：
        data/grid/grid_{symbol}_H1_ATR.csv
        data/grid/grid_top10_by_sharpe_{symbol}.csv
        data/params/best_params_grid_{symbol}.json
    """
    import pandas as pd
    import json

    # --- 默认网格（可被参数列表覆盖） ---
    fast_list = fast_list or [20, 30, 50]
    slow_list = slow_list or [100, 150, 200]
    atr_sl_list = atr_sl_list or [1.5, 2.0]         # 止损倍数
    atr_tp_list = atr_tp_list or [None, 2.0, 3.0]   # 含不设止盈
    long_only_list = long_only_list or [False, True]
    cooldown_list = cooldown_list or [0, 6, 12, 24]
    slope_list = slope_list or [0, 3]
    spread_list = spread_list or [spread]
    slip_list = slip_list or [slip]
    comm_list = comm_list or [comm]

    rows = []
    total = 0
    for f in fast_list:
        for s in slow_list:
            if f >= s:
                continue
            for k in atr_sl_list:
                for m in atr_tp_list:
                    for lo in long_only_list:
                        for cd in cooldown_list:
                            for slp in slope_list:
                                for sp in spread_list:
                                    for sp_slip in slip_list:
                                        for cm in comm_list:
                                            total += 1
                                            logger.info(
                                                f"[GRID] sym={symbol} fast={f} slow={s} "
                                                f"SL=ATR×{k} TP={'None' if m is None else 'ATR×'+str(m)} "
                                                f"ABOVE={lo} CD={cd} SLOPE={slp} "
                                                f"spread={sp} slip={sp_slip} comm={cm}"
                                            )
                                            res = run_once(
                                                symbol=symbol,
                                                csv_path=csv_path,
                                                fast_win=int(f), slow_win=int(s),
                                                spread_pips=float(sp),
                                                commission_per_million=float(cm),
                                                slippage_pips=float(sp_slip),
                                                # 关闭固定 pips，启用 ATR
                                                stop_loss_pips=None,
                                                take_profit_pips=None,
                                                atr_sl=float(k) if k is not None else None,
                                                atr_tp=float(m) if m is not None else None,
                                                atr_window=int(atr_window),
                                                qty=int(qty),
                                                initial_cash=float(initial_cash),
                                                account_ccy=str(account_ccy),
                                                fx_rates=fx_rates,
                                                long_only_above_slow=bool(lo),
                                                cooldown=int(cd),
                                                slope_lookback=int(slp),
                                                skip_outlier_entries=skip_outlier_entries,
                                                write_summary=False,
                                            )
                                            # 把当前维度也写入结果，便于回看
                                            res.update({
                                                "symbol": symbol,
                                                "spread": float(sp),
                                                "slip": float(sp_slip),
                                                "comm": float(cm),
                                                "long_only_above_slow": bool(lo),
                                                "cooldown": int(cd),
                                                "slope_lookback": int(slp),
                                            })
                                            rows.append(res)

    df = pd.DataFrame(rows)
    os.makedirs(GRID_DIR, exist_ok=True)
    os.makedirs(PARAMS_DIR, exist_ok=True)
    out = os.path.join(GRID_DIR, f"grid_{symbol}_H1_ATR.csv")
    df.to_csv(out, index=False)
    logger.info(f"[GRID] 扫描完成（组合数={total}），已保存: {out}")

    try:
        if df.empty:
            logger.warning("[GRID] 无结果，跳过排名/保存。")
            return
        # 确保 sharpe 可排序
        df["sharpe"] = pd.to_numeric(df["sharpe"], errors="coerce")
        df_sorted = df.sort_values("sharpe", ascending=False, na_position="last")

        logger.info("\n[GRID] Top 10 by Sharpe:\n" + df_sorted.head(10).to_string(index=False))

        # 保存最优参数
        best = df_sorted.iloc[0].to_dict()
        best_path = os.path.join(PARAMS_DIR, f"best_params_grid_{symbol}.json")
        with open(best_path, "w", encoding="utf-8") as f:
            json.dump(best, f, ensure_ascii=False, indent=2)
        logger.info(f"[GRID] 最优参数已保存: {best_path}")

        # 保存 Top-10
        top10_path = os.path.join(GRID_DIR, f"grid_top10_by_sharpe_{symbol}.csv")
        df_sorted.head(10).to_csv(top10_path, index=False)
        logger.info(f"[GRID] Top 10 已保存: {top10_path}")
    except Exception as e:
        logger.warning(f"[GRID] 排序/保存失败: {e}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", action="store_true", help="启用参数网格扫描（含 ATR）")
    ap.add_argument("--symbol", type=str, default="EURUSD", help="交易品种")
    ap.add_argument("--csv", type=str, default=os.path.join(RAW_DATA_DIR, "EURUSD_H1.csv"), help="CSV 路径")
    ap.add_argument("--fast", type=int, default=50, help="快速均线窗口")
    ap.add_argument("--slow", type=int, default=200, help="慢速均线窗口（应 > fast）")
    ap.add_argument("--qty", type=int, default=10_000, help="下单数量（名义）")
    ap.add_argument("--cash", type=float, default=100000.0, help="初始资金")
    ap.add_argument("--account-ccy", type=str, default="USD", help="账户结算货币（默认 USD）")
    ap.add_argument("--spread", type=float, default=1.0, help="点差（pips）")
    ap.add_argument("--slip", type=float, default=0.2, help="滑点（pips）")
    ap.add_argument("--comm", type=float, default=2.0, help="佣金（$ per $1,000,000 名义）")
    ap.add_argument("--sl", type=float, default=50.0, help="止损（pips）")
    ap.add_argument("--tp", type=float, default=None, help="止盈（pips，可空）")
    ap.add_argument("--atr-sl", type=float, default=None, help="ATR 止损倍数（k_SL），例如 2.0 表示 2×ATR")
    ap.add_argument("--atr-tp", type=float, default=None, help="ATR 止盈倍数（m_TP），例如 3.0 表示 3×ATR；缺省表示不用 ATR 止盈")
    ap.add_argument("--atr-window", type=int, default=14, help="ATR 窗口（默认 14）")
    ap.add_argument("--regime-ema-window", dest="regime_ema_window", type=int, default=200, help="Regime 过滤使用的 EMA 窗口长度")
    ap.add_argument("--regime-slope-min", dest="regime_slope_min", type=float, default=None, help="EMA 斜率阈值（价格单位）判定趋势 regime")
    ap.add_argument("--regime-atr-min", dest="regime_atr_min", type=float, default=None, help="ATR 下限，用于判定趋势 regime")
    ap.add_argument("--regime-atr-percentile-min", dest="regime_atr_percentile_min", type=float, default=None, help="ATR 百分位下限（0-1），用来过滤低波动段")
    ap.add_argument("--regime-atr-percentile-window", dest="regime_atr_percentile_window", type=int, default=500, help="ATR 百分位计算窗口长度（条数）")
    ap.add_argument("--regime-trend-min-bars", dest="regime_trend_min_bars", type=int, default=0, help="趋势 regime 需要至少持续多少根 K 才允许入场")
    ap.add_argument("--htf-factor", dest="htf_factor", type=int, default=4, help="高时间框聚合倍数（例如 4 表示 4 根低频合成一根高频）")
    ap.add_argument("--htf-ema-window", dest="htf_ema_window", type=int, default=None, help="高时间框 EMA 窗口")
    ap.add_argument("--htf-rsi-period", dest="htf_rsi_period", type=int, default=None, help="高时间框 RSI 周期")
    ap.add_argument("--rsi-period", type=int, default=14, help="RSI 窗口（默认 14）")
    ap.add_argument("--rsi-long-thresh", type=float, default=None, help="做多入场最低 RSI（例如 55）")
    ap.add_argument("--rsi-short-thresh", type=float, default=None, help="做空入场最高 RSI（例如 45）")
    ap.add_argument("--enable-trailing", action="store_true", help="启用基于 ATR 的 trailing stop")
    ap.add_argument("--trailing-enable-atr-mult", type=float, default=1.0, help="盈利达到多少倍 entry_atr 时启用 trailing（默认1.0）")
    ap.add_argument("--trailing-atr-mult", type=float, default=0.5, help="trailing 步长，按 curr_atr 的倍数移动止损（默认0.5）")
    ap.add_argument("--long-only-above-slow", action="store_true", help="仅当 close > SMA_slow 时允许做多")
    ap.add_argument("--slope-lookback", type=int, default=0, help="fast SMA 斜率确认（>0 开启, 单位=bar）")
    ap.add_argument("--cooldown", type=int, default=0, help="平仓后冷却 N 根bar 才允许再次进场")
    ap.add_argument("--config", type=str, default=None, help="YAML 配置路径（命令行显式参数将覆盖配置）")
    ap.add_argument(
        "--fx-rate",
        action="append",
        default=None,
        help="额外换汇报价（可重复），格式示例：GBPUSD=1.27 或 EUR/JPY=161.3",
    )
    ap.add_argument("--no-short", action="store_true", help="禁用做空信号")
    ap.add_argument("--short-only-below-slow", action="store_true", help="仅当 close < SMA_slow 时允许做空")
    ap.add_argument("--skip-outlier-entries", action="store_true", help="标记为 outlier 的 bar 上禁止开新仓位")
    ap.add_argument("--strategy-mode", choices=["first_hit", "weighted"], default="first_hit", help="多策略组合模式（first_hit 或 weighted）")
    ap.add_argument("--strategy-vote-threshold", type=float, default=0.0, help="weighted 模式下投票阈值（默认 0）")
    ap.add_argument("--cost-profile-file", type=str, help="JSON/YAML 文件路径，定义成本/点差 profile")
    ap.add_argument("--slippage-model-file", type=str, help="JSON/YAML 文件路径，定义滑点模型")
    ap.add_argument("--stress-cost-spread-mult", type=float, default=1.0, help="压力测试：点差乘子（默认1）")
    ap.add_argument("--stress-cost-comm-mult", type=float, default=1.0, help="压力测试：佣金乘子（默认1）")
    ap.add_argument("--stress-slippage-mult", type=float, default=1.0, help="压力测试：滑点乘子（默认1）")
    ap.add_argument("--stress-price-vol-mult", type=float, default=1.0, help="压力测试：高低点范围乘子（默认1）")
    ap.add_argument("--stress-skip-trade-pct", type=float, default=0.0, help="压力测试：随机跳过交易的概率（0-1）")
    ap.add_argument("--risk-percent", type=float, default=None, help="每笔风险占当前权益比例（如 0.01 表示 1%）")
    ap.add_argument("--max-drawdown", type=float, default=None, help="最大允许回撤（小数，如 0.2 表示 20%），超出后停止开仓")
    ap.add_argument("--max-units", type=float, default=None, help="仓位上限（基准货币单位）")
    ap.add_argument(
        "--use-best",
        action="store_true",
        help="从 data/params/best_params_grid_{symbol}.json 读取最优参数并运行（命令行显式参数仍可覆盖）"
    )
    args = ap.parse_args()
    cli_fx_rates = _coerce_fx_rates(args.fx_rate)

    if args.grid:
        grid_search(
            symbol=args.symbol,
            csv_path=args.csv,
            qty=args.qty,
            initial_cash=args.cash,
            account_ccy=args.account_ccy,
            fx_rates=cli_fx_rates,
            spread=args.spread,
            slip=args.slip,
            comm=args.comm,
            atr_window=args.atr_window,
            skip_outlier_entries=args.skip_outlier_entries,
        )
    else:
        # ---- 加载 YAML 配置并与命令行合并（命令行显式参数优先） ----
        cfg = {}
        if args.config:
            with open(args.config, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
                # 允许大小写/短名对齐 argparse 名称
                key_map = {
                    "symbol": "symbol",
                    "csv": "csv_path",
                    "cash": "cash",
                    "qty": "qty",
                    "account_ccy": "account_ccy",
                    "fast": "fast",
                    "slow": "slow",
                    "spread": "spread",
                    "slip": "slip",
                    "comm": "comm",
                    "sl": "sl",
                    "tp": "tp",
                    "atr_sl": "atr_sl",
                    "atr_tp": "atr_tp",
                    "atr_window": "atr_window",
                    "regime_ema_window": "regime_ema_window",
                    "regime_slope_min": "regime_slope_min",
                    "regime_atr_min": "regime_atr_min",
                    "regime_atr_percentile_min": "regime_atr_percentile_min",
                    "regime_atr_percentile_window": "regime_atr_percentile_window",
                    "regime_trend_min_bars": "regime_trend_min_bars",
                    "htf_factor": "htf_factor",
                    "htf_ema_window": "htf_ema_window",
                    "htf_rsi_period": "htf_rsi_period",
                        "rsi_period": "rsi_period",
                        "rsi_long_thresh": "rsi_long_thresh",
                        "rsi_short_thresh": "rsi_short_thresh",
                        "enable_trailing": "enable_trailing",
                        "trailing_enable_atr_mult": "trailing_enable_atr_mult",
                        "trailing_atr_mult": "trailing_atr_mult",
                    "long_only_above_slow": "long_only_above_slow",
                    "slope_lookback": "slope_lookback",
                    "cooldown": "cooldown",
                    "fx_rates": "fx_rates",
                    "allow_short": "allow_short",
                    "short_only_below_slow": "short_only_below_slow",
                    "risk_per_trade_pct": "risk_per_trade_pct",
                    "max_drawdown_pct": "max_drawdown_pct",
                    "max_position_units": "max_position_units",
                    "skip_outlier_entries": "skip_outlier_entries",
                    "strategies": "strategies",
                    "cost_profiles": "cost_profiles",
                    "slippage_model": "slippage_model",
                }
                # 规范化键名
                norm = {}
                for k, v in raw.items():
                    kk = k.strip()
                    if kk in key_map:
                        norm[key_map[kk]] = v
                    else:
                        norm[kk] = v
                cfg = norm
        else:
            cfg = {}
        cfg_fx_rates = _coerce_fx_rates(cfg.get("fx_rates")) if cfg else None
        cfg_strategies = parse_strategy_specs(cfg.get("strategies")) if cfg else None
        cfg_cost_profiles = cfg.get("cost_profiles") if cfg else None
        cfg_slippage_model = cfg.get("slippage_model") if cfg else None
        if args.cost_profile_file:
            cfg_cost_profiles = _load_structured_data(args.cost_profile_file)
        if args.slippage_model_file:
            cfg_slippage_model = _load_structured_data(args.slippage_model_file)
        # [PATCH B START] 载入 best_params_grid.json（若 --use-best），并做类型规范化
        best_cfg = {}
        if args.use_best:
            try:
                import json, math
                os.makedirs(PARAMS_DIR, exist_ok=True)
                best_path = os.path.join(PARAMS_DIR, f"best_params_grid_{args.symbol}.json")
                with open(best_path, "r", encoding="utf-8") as f:
                    best = json.load(f) or {}

                def _is_nan(x):
                    return isinstance(x, float) and math.isnan(x)

                def _to_int_or_none(x):
                    if x is None or _is_nan(x):
                        return None
                    if isinstance(x, (int, np.integer)):
                        return int(x)
                    if isinstance(x, (float, np.floating)):
                        return int(round(float(x)))
                    # 其他类型尝试转
                    try:
                        return int(float(x))
                    except Exception:
                        return None

                def _to_float_or_none(x):
                    if x is None or _is_nan(x):
                        return None
                    if isinstance(x, (int, float, np.integer, np.floating)):
                        return float(x)
                    try:
                        v = float(x)
                        return v if not math.isnan(v) else None
                    except Exception:
                        return None

                # 将网格结果列名映射为参数名，并做规范化
                best_cfg = {
                    "fast": _to_int_or_none(best.get("fast")),
                    "slow": _to_int_or_none(best.get("slow")),
                    "atr_sl": _to_float_or_none(best.get("atr_sl")),
                    "atr_tp": _to_float_or_none(best.get("atr_tp")),   # NaN -> None
                    "atr_window": _to_int_or_none(best.get("atr_window")),
                }
                # 去掉 None 的键，避免覆盖有效默认值
                best_cfg = {k: v for k, v in best_cfg.items() if v is not None}

                logger.info(f"[BEST] 已载入 {args.symbol} 最优参数: {best_cfg}")
            except Exception as e:
                logger.warning(f"[BEST] 读取最优参数失败，忽略 --use-best：{e}")

        # 合并 best_cfg 到 cfg（优先级：命令行 > best_cfg > cfg > 默认）
        for k, v in (best_cfg or {}).items():
            if k not in cfg:
                cfg[k] = v
        # [PATCH B END]
            


        # 构造 run_once 的最终参数（先用 cfg 的，若命令行显式传入则覆盖）
        def override(val, default, cfg_val):
            """
            如果命令行传入值 != argparse 的 default，说明用户显式设置 => 用命令行；否则用 cfg；再否则用 default
            """
            if val != default:
                return val
            return cfg_val if (cfg_val is not None) else default

        # 取 argparse 默认值（用于判断是否显式覆盖）
        defaults = vars(ap.parse_args([]))  # 空参解析拿到默认表

        kwargs = dict(
            symbol = override(args.symbol, defaults["symbol"], cfg.get("symbol")),
            csv_path = override(args.csv, defaults["csv"], cfg.get("csv_path")),
            initial_cash = override(args.cash, defaults["cash"], cfg.get("cash")),
            qty = override(args.qty, defaults["qty"], cfg.get("qty")),
            account_ccy = override(args.account_ccy, defaults["account_ccy"], cfg.get("account_ccy")),
            fast_win = override(args.fast, defaults["fast"], cfg.get("fast")),
            slow_win = override(args.slow, defaults["slow"], cfg.get("slow")),
            spread_pips = override(args.spread, defaults["spread"], cfg.get("spread")),
            commission_per_million = override(args.comm, defaults["comm"], cfg.get("comm")),
            slippage_pips = override(args.slip, defaults["slip"], cfg.get("slip")),
            stop_loss_pips = override(args.sl, defaults["sl"], cfg.get("sl")),
            take_profit_pips = override(args.tp, defaults["tp"], cfg.get("tp")),
            atr_sl = override(args.atr_sl, defaults["atr_sl"], cfg.get("atr_sl")),
            atr_tp = override(args.atr_tp, defaults["atr_tp"], cfg.get("atr_tp")),
            atr_window = override(args.atr_window, defaults["atr_window"], cfg.get("atr_window")),
            regime_ema_window = override(args.regime_ema_window, defaults["regime_ema_window"], cfg.get("regime_ema_window")),
            regime_slope_min = override(args.regime_slope_min, defaults["regime_slope_min"], cfg.get("regime_slope_min")),
            regime_atr_min = override(args.regime_atr_min, defaults["regime_atr_min"], cfg.get("regime_atr_min")),
            regime_atr_percentile_min = override(args.regime_atr_percentile_min, defaults["regime_atr_percentile_min"], cfg.get("regime_atr_percentile_min")),
            regime_atr_percentile_window = override(args.regime_atr_percentile_window, defaults["regime_atr_percentile_window"], cfg.get("regime_atr_percentile_window")),
            regime_trend_min_bars = override(args.regime_trend_min_bars, defaults["regime_trend_min_bars"], cfg.get("regime_trend_min_bars")),
            htf_factor = override(args.htf_factor, defaults["htf_factor"], cfg.get("htf_factor")),
            htf_ema_window = override(args.htf_ema_window, defaults["htf_ema_window"], cfg.get("htf_ema_window")),
            htf_rsi_period = override(args.htf_rsi_period, defaults["htf_rsi_period"], cfg.get("htf_rsi_period")),
            rsi_period = override(args.rsi_period, defaults["rsi_period"], cfg.get("rsi_period")),
            rsi_long_thresh = override(args.rsi_long_thresh, defaults["rsi_long_thresh"], cfg.get("rsi_long_thresh")),
            rsi_short_thresh = override(args.rsi_short_thresh, defaults["rsi_short_thresh"], cfg.get("rsi_short_thresh")),
            enable_trailing = override(args.enable_trailing, defaults["enable_trailing"], cfg.get("enable_trailing")),
            trailing_enable_atr_mult = override(args.trailing_enable_atr_mult, defaults["trailing_enable_atr_mult"], cfg.get("trailing_enable_atr_mult")),
            trailing_atr_mult = override(args.trailing_atr_mult, defaults["trailing_atr_mult"], cfg.get("trailing_atr_mult")),
            long_only_above_slow = override(args.long_only_above_slow, defaults["long_only_above_slow"], cfg.get("long_only_above_slow")),
            slope_lookback = override(args.slope_lookback, defaults["slope_lookback"], cfg.get("slope_lookback")),
            cooldown = override(args.cooldown, defaults["cooldown"], cfg.get("cooldown")),
            short_only_below_slow = override(args.short_only_below_slow, defaults["short_only_below_slow"], cfg.get("short_only_below_slow")),
            risk_per_trade_pct = override(args.risk_percent, defaults["risk_percent"], cfg.get("risk_per_trade_pct")),
            max_drawdown_pct = override(args.max_drawdown, defaults["max_drawdown"], cfg.get("max_drawdown_pct")),
            max_position_units = override(args.max_units, defaults["max_units"], cfg.get("max_position_units")),
            skip_outlier_entries = override(args.skip_outlier_entries, defaults["skip_outlier_entries"], cfg.get("skip_outlier_entries")),
            cost_profiles = cfg_cost_profiles,
            slippage_model = cfg_slippage_model,
            strategy_mode = override(args.strategy_mode, defaults["strategy_mode"], cfg.get("strategy_mode")),
            strategy_vote_threshold = override(args.strategy_vote_threshold, defaults["strategy_vote_threshold"], cfg.get("strategy_vote_threshold")),
            stress_cost_spread_mult = override(args.stress_cost_spread_mult, defaults["stress_cost_spread_mult"], cfg.get("stress_cost_spread_mult")),
            stress_cost_comm_mult = override(args.stress_cost_comm_mult, defaults["stress_cost_comm_mult"], cfg.get("stress_cost_comm_mult")),
            stress_slippage_mult = override(args.stress_slippage_mult, defaults["stress_slippage_mult"], cfg.get("stress_slippage_mult")),
            stress_price_vol_mult = override(args.stress_price_vol_mult, defaults["stress_price_vol_mult"], cfg.get("stress_price_vol_mult")),
            stress_skip_trade_pct = override(args.stress_skip_trade_pct, defaults["stress_skip_trade_pct"], cfg.get("stress_skip_trade_pct")),
        )
        # [PATCH C START] 关键窗口参数安全转为 int
        try:
            kwargs["fast_win"] = int(kwargs["fast_win"])
            kwargs["slow_win"] = int(kwargs["slow_win"])
            kwargs["atr_window"] = int(kwargs["atr_window"])
            kwargs["rsi_period"] = int(kwargs.get("rsi_period", 14))
        except Exception as e:
            logger.error(f"参数类型转换错误，请检查 fast/slow/atr_window：{e}")
            raise
        # [PATCH C END]

        kwargs["short_only_below_slow"] = bool(kwargs["short_only_below_slow"])
        if kwargs["risk_per_trade_pct"] is not None:
            kwargs["risk_per_trade_pct"] = float(kwargs["risk_per_trade_pct"])
        if kwargs["max_drawdown_pct"] is not None:
            kwargs["max_drawdown_pct"] = float(kwargs["max_drawdown_pct"])
        if kwargs["max_position_units"] is not None:
            kwargs["max_position_units"] = float(kwargs["max_position_units"])
        # RSI / trailing types
        if kwargs.get("rsi_long_thresh") is not None:
            kwargs["rsi_long_thresh"] = float(kwargs["rsi_long_thresh"])
        if kwargs.get("rsi_short_thresh") is not None:
            kwargs["rsi_short_thresh"] = float(kwargs["rsi_short_thresh"])
        kwargs["enable_trailing"] = bool(kwargs.get("enable_trailing", False))
        kwargs["trailing_enable_atr_mult"] = float(kwargs.get("trailing_enable_atr_mult", 1.0))
        kwargs["trailing_atr_mult"] = float(kwargs.get("trailing_atr_mult", 0.5))
        kwargs["regime_ema_window"] = int(kwargs.get("regime_ema_window") or 0)
        if kwargs.get("regime_slope_min") is not None:
            kwargs["regime_slope_min"] = float(kwargs["regime_slope_min"])
        if kwargs.get("regime_atr_min") is not None:
            kwargs["regime_atr_min"] = float(kwargs["regime_atr_min"])
        if kwargs.get("regime_atr_percentile_min") is not None:
            kwargs["regime_atr_percentile_min"] = float(kwargs["regime_atr_percentile_min"])
        kwargs["regime_atr_percentile_window"] = int(kwargs.get("regime_atr_percentile_window") or 0)
        kwargs["regime_trend_min_bars"] = int(kwargs.get("regime_trend_min_bars") or 0)
        kwargs["htf_factor"] = int(kwargs.get("htf_factor") or 1)
        if kwargs.get("htf_ema_window") is not None:
            kwargs["htf_ema_window"] = int(kwargs["htf_ema_window"])
        if kwargs.get("htf_rsi_period") is not None:
            kwargs["htf_rsi_period"] = int(kwargs["htf_rsi_period"])

        if args.no_short != defaults["no_short"]:
            allow_short = not args.no_short
        else:
            cfg_allow = cfg.get("allow_short") if cfg else None
            allow_short = bool(cfg_allow) if cfg_allow is not None else True
        kwargs["allow_short"] = allow_short

        kwargs["fx_rates"] = _merge_fx_rates(cfg_fx_rates, cli_fx_rates)
        kwargs["strategies"] = cfg_strategies

        run_once(**kwargs)
