from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd
from loguru import logger

METRIC_COLS = [
    "sharpe",
    "ann_return",
    "ann_vol",
    "max_drawdown",
    "expectancy",
    "trades",
    "win_rate",
    "return_pct",
    "rr",
    "median_hold",
    "symbol",
    "source_file",
]


def main():
    parser = argparse.ArgumentParser(description="Compare zero-cost vs real-cost grid results.")
    parser.add_argument("--zero", required=True, help="CSV from zero-cost grid run.")
    parser.add_argument("--real", required=True, help="CSV from real-cost grid run.")
    parser.add_argument("--out", default="data/grid/cost_comparison.csv", help="Output CSV path.")
    parser.add_argument("--top", type=int, default=20, help="Print top-N rows with smallest Sharpe delta.")
    args = parser.parse_args()

    df_zero = pd.read_csv(args.zero)
    df_real = pd.read_csv(args.real)

    shared_cols = [c for c in df_zero.columns if c in df_real.columns]
    if not shared_cols:
        raise ValueError("No overlapping columns between zero and real CSVs.")

    param_cols: List[str] = [c for c in shared_cols if c not in METRIC_COLS]
    if not param_cols:
        raise ValueError("Unable to infer parameter columns for join; ensure CSVs contain metrics from METRIC_COLS.")

    z = df_zero.rename(columns={col: f"{col}_zero" for col in METRIC_COLS if col in df_zero.columns})
    r = df_real.rename(columns={col: f"{col}_real" for col in METRIC_COLS if col in df_real.columns})

    merged = z.merge(r, on=param_cols, how="inner", suffixes=("_zero", "_real"))
    if merged.empty:
        raise RuntimeError("Join result is empty; ensure both CSVs share the same parameter combinations.")

    if "sharpe_zero" in merged.columns and "sharpe_real" in merged.columns:
        merged["delta_sharpe"] = merged["sharpe_real"] - merged["sharpe_zero"]
    if "ann_return_zero" in merged.columns and "ann_return_real" in merged.columns:
        merged["delta_ann_return"] = merged["ann_return_real"] - merged["ann_return_zero"]
    if "expectancy_zero" in merged.columns and "expectancy_real" in merged.columns:
        merged["delta_expectancy"] = merged["expectancy_real"] - merged["expectancy_zero"]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    logger.info(f"Cost comparison saved to {out_path} (rows={len(merged)})")

    if "delta_sharpe" in merged.columns:
        top_df = merged.sort_values("delta_sharpe", ascending=False).head(args.top)
        print(top_df[param_cols + ["sharpe_zero", "sharpe_real", "delta_sharpe"]].to_string(index=False))


if __name__ == "__main__":
    main()
