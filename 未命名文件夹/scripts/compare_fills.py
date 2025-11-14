#!/usr/bin/env python3
"""
Compare paper vs. live fills to produce a lightweight TCA summary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare paper vs live fills.")
    parser.add_argument("--paper", required=True, help="Path to paper fills CSV.")
    parser.add_argument("--live", required=True, help="Path to live fills CSV.")
    parser.add_argument("--out", default="results/execution/tca_summary.json", help="Output JSON path.")
    return parser.parse_args()


def load(path: str) -> pd.DataFrame:
    csv = Path(path)
    if not csv.exists():
        raise SystemExit(f"fills CSV not found: {csv}")
    df = pd.read_csv(csv)
    if df.empty:
        raise SystemExit(f"{csv} is empty")
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"])
    return df


def summary(df: pd.DataFrame, label: str) -> dict:
    pnl = df["pnl"] if "pnl" in df.columns else pd.Series(dtype=float)
    latency = df["adapter_latency_ms"] if "adapter_latency_ms" in df.columns else pd.Series(dtype=float)
    return {
        f"{label}_trade_count": int(len(df)),
        f"{label}_total_pnl": float(pnl.sum()) if not pnl.empty else None,
        f"{label}_avg_pnl": float(pnl.mean()) if not pnl.empty else None,
        f"{label}_avg_latency_ms": float(latency.mean()) if not latency.empty else None,
    }


def pnl_gap(paper: pd.DataFrame, live: pd.DataFrame) -> dict:
    cols = []
    if "ts" in paper.columns and "ts" in live.columns:
        merged = pd.merge(
            paper[["ts", "pnl"]].rename(columns={"pnl": "paper_pnl"}),
            live[["ts", "pnl"]].rename(columns={"pnl": "live_pnl"}),
            on="ts",
            how="outer",
        )
    else:
        merged = pd.DataFrame({"paper_pnl": paper.get("pnl"), "live_pnl": live.get("pnl")})
    merged = merged.fillna(0.0)
    merged["pnl_diff"] = merged["live_pnl"] - merged["paper_pnl"]
    return {
        "pnl_diff_mean": float(merged["pnl_diff"].mean()),
        "pnl_diff_std": float(merged["pnl_diff"].std(ddof=0)),
    }


def main() -> None:
    args = parse_args()
    paper = load(args.paper)
    live = load(args.live)

    report = {"paper_path": args.paper, "live_path": args.live}
    report.update(summary(paper, "paper"))
    report.update(summary(live, "live"))
    report.update(pnl_gap(paper, live))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    print(json.dumps(report, indent=2, default=float))


if __name__ == "__main__":
    main()
