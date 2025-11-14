#!/usr/bin/env python3
"""
Train an XGBoost classifier on USDJPY H1 and export model artifacts.

Artifacts are written to: QuantResearch/artifacts/models/usdjpy_h1_xgb/<ts>/
  - model.json           (xgboost Booster)
  - feature_list.json    (ordered feature names)
  - thresholds.json      (p_long, p_exit, val/test metrics)
  - meta.json            (dataset/costs/params/seed/etc.)

Also updates: QuantResearch/artifacts/models/usdjpy_h1_xgb_latest.json
  {"model_dir": "QuantResearch/artifacts/models/usdjpy_h1_xgb/<ts>"}
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train XGB (long-only) on USDJPY H1")
    p.add_argument("--csv", default="QuantResearch/data/raw/USDJPY_H1.csv")
    p.add_argument("--symbol", default="USDJPY")
    p.add_argument("--horizon", type=int, default=6)
    p.add_argument("--train-ratio", type=float, default=0.6)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    # Unified costs (match backtest/YAML)
    p.add_argument("--spread-pips", type=float, default=2.0)
    p.add_argument("--slip-pips", type=float, default=0.3)
    p.add_argument("--comm-per-million", type=float, default=0.25)
    # Model params (conservative defaults)
    p.add_argument("--max-depth", type=int, default=4)
    p.add_argument("--n-estimators", type=int, default=300)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--min-child-weight", type=float, default=1.0)
    p.add_argument("--reg-lambda", type=float, default=1.0)
    # Output base dir
    p.add_argument("--out", default="QuantResearch/artifacts/models/usdjpy_h1_xgb")
    return p.parse_args()


def _pip_value(symbol: str) -> float:
    return 0.01 if symbol.upper().endswith("JPY") else 0.0001


def compute_cost_return(symbol: str, price: pd.Series, spread_pips: float, slip_pips: float, comm_per_million: float) -> pd.Series:
    pip = _pip_value(symbol)
    # Approx trade cost (fractional): spread + 2*slippage in price terms + commission fraction
    frac_px = (spread_pips + 2.0 * slip_pips) * pip / price
    frac_comm = (comm_per_million / 1_000_000.0)
    return frac_px + frac_comm


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.rolling(period).mean()
    roll_down = down.rolling(period).mean()
    rs = roll_up / roll_down
    return 100 - (100 / (1 + rs))


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    ts = pd.to_datetime(df["time"], utc=True)
    hour = ts.dt.hour.astype(float)
    dow = ts.dt.dayofweek.astype(float)
    df["hour_sin"], df["hour_cos"] = np.sin(2 * np.pi * hour / 24.0), np.cos(2 * np.pi * hour / 24.0)
    df["dow_sin"], df["dow_cos"] = np.sin(2 * np.pi * dow / 7.0), np.cos(2 * np.pi * dow / 7.0)
    return df


def build_features(df: pd.DataFrame, fast: int = 20, slow: int = 80, rsi_p: int = 14, atr_p: int = 14) -> Tuple[pd.DataFrame, List[str]]:
    out = df.copy()
    out = add_time_features(out)
    out["ret_1"] = out["close"].pct_change(1)
    out["ret_3"] = out["close"].pct_change(3)
    out["ret_6"] = out["close"].pct_change(6)
    out["vol_24"] = out["close"].pct_change(1).rolling(24).std()
    sma_f = out["close"].rolling(fast).mean()
    sma_s = out["close"].rolling(slow).mean()
    out["sma_diff"] = (sma_f - sma_s) / out["close"]
    out["rsi"] = rsi(out["close"], rsi_p)
    out["atr_norm"] = atr(out["high"], out["low"], out["close"], atr_p) / out["close"]
    feats = [
        "ret_1", "ret_3", "ret_6", "vol_24",
        "sma_diff", "rsi", "atr_norm",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    ]
    return out, feats


def forward_return(close: pd.Series, horizon: int) -> pd.Series:
    return close.shift(-horizon) / close - 1.0


def pick_thresholds(p_val: np.ndarray, fwd_val: np.ndarray, cost_val: np.ndarray) -> Dict[str, float]:
    best = {"thr_long": 0.6, "thr_exit": 0.5, "sharpe": -np.inf, "trades": 0}
    for thr in np.linspace(0.55, 0.8, 11):
        for thr_exit in np.linspace(0.45, 0.6, 16):
            chosen = p_val >= thr
            if not np.any(chosen):
                continue
            net = (fwd_val - cost_val)[chosen]
            if net.size < 50:
                continue
            mu = float(np.nanmean(net))
            sd = float(np.nanstd(net, ddof=1))
            sharpe = (mu / sd) * math.sqrt(252 * 24) if sd > 0 else -np.inf
            if sharpe > best["sharpe"]:
                best = {"thr_long": float(thr), "thr_exit": float(thr_exit), "sharpe": sharpe, "trades": int(net.size)}
    return best


def main() -> None:
    try:
        import xgboost as xgb  # requires xgboost==1.7.6 per requirements
    except Exception as exc:
        raise SystemExit("xgboost is required. Please install xgboost==1.7.6.") from exc

    args = parse_args()
    np.random.seed(args.seed)

    df = pd.read_csv(args.csv)
    if "time" not in df.columns:
        raise SystemExit("CSV must include 'time' column")
    df = df[["time", "open", "high", "low", "close", "volume"]].copy()

    df_feat, feat_list = build_features(df)
    fwd = forward_return(df_feat["close"], args.horizon)
    costs = compute_cost_return(args.symbol, df_feat["close"], args.spread_pips, args.slip_pips, args.comm_per_million)
    # Label with cost margin
    y = pd.Series(np.where(fwd > costs, 1, np.where(fwd < -costs, 0, np.nan)), index=df_feat.index)

    data = df_feat.assign(y=y, cost=costs).dropna(subset=feat_list + ["y"]).reset_index(drop=True)
    X = data[feat_list].to_numpy()
    y_arr = data["y"].astype(int).to_numpy()
    fwd_arr = fwd.loc[data.index].to_numpy()
    cost_arr = data["cost"].to_numpy()

    n = len(data)
    i_train = int(n * args.train_ratio)
    i_val = int(n * (args.train_ratio + args.val_ratio))
    if i_val >= n:
        i_val = n - max(1, n // 10)

    X_train, y_train = X[:i_train], y_arr[:i_train]
    X_val, y_val = X[i_train:i_val], y_arr[i_train:i_val]
    X_test, y_test = X[i_val:], y_arr[i_val:]
    fwd_val, cost_val = fwd_arr[i_train:i_val], cost_arr[i_train:i_val]
    fwd_test, cost_test = fwd_arr[i_val:], cost_arr[i_val:]

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feat_list)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feat_list)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feat_list)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": args.max_depth,
        "eta": args.learning_rate,
        "lambda": args.reg_lambda,
        "min_child_weight": args.min_child_weight,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "seed": args.seed,
    }
    evals = [(dtrain, "train"), (dval, "val")]
    booster = xgb.train(params, dtrain, num_boost_round=args.n_estimators, evals=evals, early_stopping_rounds=50, verbose_eval=False)

    p_val = booster.predict(dval)
    th = pick_thresholds(p_val, fwd_val, cost_val)
    p_test = booster.predict(dtest)
    chosen = p_test >= th["thr_long"]
    net = (fwd_test - cost_test)[chosen]
    test_trades = int(net.size)
    mu = float(np.nanmean(net)) if net.size else 0.0
    sd = float(np.nanstd(net, ddof=1)) if net.size > 1 else 0.0
    test_sharpe = (mu / sd) * math.sqrt(252 * 24) if sd > 0 else 0.0

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    model_dir = Path(args.out).with_suffix("") / ts
    model_dir.mkdir(parents=True, exist_ok=True)
    # Save artifacts
    booster.save_model(str(model_dir / "model.json"))
    (model_dir / "feature_list.json").write_text(json.dumps(feat_list, indent=2), encoding="utf-8")
    thresholds = {
        "p_long": float(th["thr_long"]),
        "p_exit": float(th["thr_exit"]),
        "val_sharpe": float(th["sharpe"]),
        "val_trades": int(th["trades"]),
        "test_sharpe": float(test_sharpe),
        "test_trades": int(test_trades),
    }
    (model_dir / "thresholds.json").write_text(json.dumps(thresholds, indent=2), encoding="utf-8")
    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "symbol": args.symbol,
        "csv": args.csv,
        "horizon": args.horizon,
        "splits": {"train": int(i_train), "val": int(i_val - i_train), "test": int(n - i_val)},
        "seed": int(args.seed),
        "features": feat_list,
        "xgb_params": params,
        "best_iteration": int(getattr(booster, "best_iteration", 0) or 0),
        "thresholds": thresholds,
        "costs": {"spread_pips": args.spread_pips, "slip_pips": args.slip_pips, "comm_per_million": args.comm_per_million},
        "git_commit": None,
    }
    (model_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    latest_path = Path(args.out).with_suffix("").parent / "usdjpy_h1_xgb_latest.json"
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    latest_path.write_text(json.dumps({"model_dir": str(model_dir)}, indent=2), encoding="utf-8")
    print(f"Saved model artifacts to {model_dir}")


if __name__ == "__main__":
    main()
