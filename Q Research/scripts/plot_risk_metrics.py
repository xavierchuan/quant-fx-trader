#!/usr/bin/env python3
"""
Quick visualization/report for results/risk/metrics.csv.

Usage:
  python scripts/plot_risk_metrics.py --csv results/risk/metrics.csv --out charts/risk_metrics.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot risk metrics history (reject counts, status).")
    parser.add_argument("--csv", default="results/risk/metrics.csv", help="Path to metrics CSV.")
    parser.add_argument("--out", default="charts/risk_metrics.png", help="Output image path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"metrics CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if df.empty:
        raise SystemExit("metrics CSV is empty.")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.sort_values("timestamp", inplace=True)
    df["status"] = df["status"].fillna("unknown").str.lower()

    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(df["timestamp"], df["rejects"], marker="o", label="Rejects")
    ax1.set_ylabel("Reject count")
    ax1.set_xlabel("Timestamp")
    ax1.set_title("Risk simulation rejects over time")

    fail_mask = df["status"] == "fail"
    ax1.scatter(df.loc[fail_mask, "timestamp"], df.loc[fail_mask, "rejects"], color="red", label="Fail", zorder=5)
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    status_numeric = df["status"].map({"pass": 1, "fail": 0}).fillna(0.5)
    ax2.plot(df["timestamp"], status_numeric, color="gray", alpha=0.3, label="Status (1=pass,0=fail)")
    ax2.set_ylim(-0.1, 1.1)
    ax2.set_yticks([0, 0.5, 1])
    ax2.set_yticklabels(["fail", "unknown", "pass"])

    fig.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved risk metrics plot to {out_path}")


if __name__ == "__main__":
    main()
