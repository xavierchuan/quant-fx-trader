#!/usr/bin/env python3
"""Generate Phase 2 diagnostics charts (batch heatmap, Monte Carlo box, walk-forward timeline, equity plots)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

DEFAULT_OUT = Path("charts")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot diagnostics from batch/Monte Carlo/walk-forward outputs.")
    parser.add_argument("--batch-csv", help="Path to batch_backtests_*.csv")
    parser.add_argument("--walkforward-csv", help="Path to walkforward/metrics.csv")
    parser.add_argument("--mc-summary", help="Path to stress/mc_summary.json")
    parser.add_argument("--mc-iterations", help="Path to stress/mc_iterations.csv")
    parser.add_argument("--equity-csv", help="Path to equity curve CSV (ts,equity)")
    parser.add_argument("--underwater-csv", help="Path to underwater/drawdown CSV (ts,drawdown)")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output directory (default charts/)")
    parser.add_argument("--x-col", default="fast_win", help="Batch heatmap X-axis column")
    parser.add_argument("--y-col", default="slow_win", help="Batch heatmap Y-axis column")
    parser.add_argument("--metric", default="sharpe", help="Batch heatmap metric column")
    parser.add_argument(
        "--facet-scenario",
        action="store_true",
        help="If set, creates one heatmap per scenario column when available.",
    )
    parser.add_argument("--format", default="png", choices=["png", "pdf"], help="Image format")
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_heatmap(
    batch_csv: Path,
    out_dir: Path,
    x_col: str,
    y_col: str,
    metric: str,
    fmt: str,
    facet: bool,
) -> Tuple[Dict[str, Optional[str]], Dict[str, Dict]]:
    outputs: Dict[str, Optional[str]] = {}
    data: Dict[str, Dict] = {}
    if not batch_csv:
        return outputs, data
    df = pd.read_csv(batch_csv)
    if not {x_col, y_col, metric}.issubset(df.columns):
        logger.warning("Batch CSV missing columns required for heatmap (%s, %s, %s)", x_col, y_col, metric)
        return outputs, data

    def _render(sub_df: pd.DataFrame, suffix: str) -> Optional[Path]:
        pivot = sub_df.pivot_table(index=y_col, columns=x_col, values=metric, aggfunc="mean")
        if pivot.empty:
            return None
        fig, ax = plt.subplots(figsize=(6, 4))
        c = ax.imshow(pivot.values, origin="lower", aspect="auto", cmap="viridis")
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticklabels(pivot.index)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f"{metric.title()} heatmap ({suffix})")
        fig.colorbar(c, ax=ax, label=metric)
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                value = pivot.values[i, j]
                if not np.isnan(value):
                    ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="white", fontsize=8)
        filename = f"heatmap_{metric}_{suffix}.{fmt}" if suffix else f"heatmap_{metric}.{fmt}"
        out_path = out_dir / filename
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        data[suffix or "all"] = pivot.to_dict()
        return out_path

    if facet and "scenario" in df.columns:
        for scenario, sub in df.groupby("scenario"):
            outputs[scenario] = str(_render(sub, scenario) or "")
    else:
        outputs["all"] = str(_render(df, "all") or "")
    return outputs, data


def plot_mc_box(itr_csv: Path, summary_json: Path, out_dir: Path, fmt: str) -> Tuple[Optional[str], Dict[str, float]]:
    if not itr_csv or not summary_json:
        return None, {}
    if not itr_csv.exists() or not summary_json.exists():
        logger.warning("Monte Carlo files missing: %s or %s", itr_csv, summary_json)
        return None, {}
    df = pd.read_csv(itr_csv)
    if "sharpe" not in df.columns:
        logger.warning("Monte Carlo iterations missing 'sharpe' column; skipping box plot")
        return None, {}
    series = df["sharpe"].dropna()
    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    scenario = summary.get("scenario", "unknown")
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.boxplot(series, labels=[scenario])
    ax.set_ylabel("Sharpe")
    ax.set_title("Monte Carlo Sharpe distribution")
    out_path = out_dir / f"monte_carlo_box.{fmt}"
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    logger.info("Saved Monte Carlo box plot to %s", out_path)
    stats = {
        "mean": float(series.mean()) if not series.empty else None,
        "p05": float(series.quantile(0.05)) if not series.empty else None,
        "p50": float(series.quantile(0.5)) if not series.empty else None,
        "p95": float(series.quantile(0.95)) if not series.empty else None,
        "count": int(series.size),
    }
    return str(out_path), stats


def plot_walkforward(wf_csv: Path, out_dir: Path, fmt: str) -> Tuple[Optional[str], list]:
    if not wf_csv:
        return None, []
    if not wf_csv.exists():
        logger.warning("Walk-forward metrics file missing: %s", wf_csv)
        return None, []
    df = pd.read_csv(wf_csv)
    if "window" not in df.columns or "sharpe" not in df.columns:
        logger.warning("Walk-forward metrics missing 'window' or 'sharpe'; skipping timeline")
        return None, []
    status = df.get("status", pd.Series(["unknown"] * len(df)))
    colors = status.map({"pass": "#2ca02c", "fail": "#d62728"}).fillna("#1f77b4")
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.scatter(df["window"], df["sharpe"], c=colors)
    ax.plot(df["window"], df["sharpe"], color="#cccccc", linewidth=1, alpha=0.5)
    ax.set_xlabel("Window")
    ax.set_ylabel("Sharpe")
    ax.set_title("Walk-forward Sharpe timeline")
    out_path = out_dir / f"walkforward_timeline.{fmt}"
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    logger.info("Saved walk-forward timeline to %s", out_path)
    return str(out_path), df[["window", "sharpe", *([ "status"] if "status" in df.columns else [])]].to_dict(orient="records")


def _resolve_column(df: pd.DataFrame, candidates) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def plot_equity(equity_csv: Path, out_dir: Path, fmt: str) -> Optional[str]:
    if not equity_csv:
        return None
    if not equity_csv.exists():
        logger.warning("Equity CSV missing: %s", equity_csv)
        return None
    df = pd.read_csv(equity_csv)
    ts_col = _resolve_column(df, ["ts", "timestamp", "time", "datetime"])
    equity_col = _resolve_column(df, ["equity", "capital", "balance"])
    if not ts_col or not equity_col:
        logger.warning(
            "Equity CSV missing timestamp/equity columns (looked for %s / %s)",
            ["ts", "timestamp", "time", "datetime"],
            ["equity", "capital", "balance"],
        )
        return None
    df[ts_col] = pd.to_datetime(df[ts_col])
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(df[ts_col], df[equity_col], color="#1f77b4")
    ax.set_xlabel("Time")
    ax.set_ylabel("Equity")
    ax.set_title("Equity Curve")
    fig.autofmt_xdate()
    out_path = out_dir / f"equity_curve.{fmt}"
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    logger.info("Saved equity curve to %s", out_path)
    return str(out_path)


def plot_underwater(underwater_csv: Path, out_dir: Path, fmt: str) -> Optional[str]:
    if not underwater_csv:
        return None
    if not underwater_csv.exists():
        logger.warning("Underwater CSV missing: %s", underwater_csv)
        return None
    df = pd.read_csv(underwater_csv)
    required = {"ts", "drawdown"}
    if not required.issubset(df.columns):
        logger.warning("Underwater CSV missing columns %s", required)
        return None
    df["ts"] = pd.to_datetime(df["ts"])
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.fill_between(df["ts"], df["drawdown"], color="#d62728", alpha=0.6)
    ax.set_xlabel("Time")
    ax.set_ylabel("Drawdown")
    ax.set_title("Underwater Curve")
    fig.autofmt_xdate()
    out_path = out_dir / f"underwater_curve.{fmt}"
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    logger.info("Saved underwater curve to %s", out_path)
    return str(out_path)


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(Path(args.out).expanduser()) / Path(
        f"diagnostics_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    )
    ensure_dir(out_dir)

    artifacts: Dict[str, Optional[str]] = {}
    data_dump: Dict[str, Dict] = {}

    if args.batch_csv:
        path = Path(args.batch_csv).expanduser()
        heatmap_paths, heatmap_data = plot_heatmap(
            path, out_dir, args.x_col, args.y_col, args.metric, args.format, args.facet_scenario
        )
        artifacts["batch_heatmap"] = heatmap_paths.get("all")
        if args.facet_scenario and "scenario" in heatmap_paths:
            artifacts["batch_heatmap_scenarios"] = heatmap_paths
        data_dump["batch_heatmap"] = heatmap_data
    if args.mc_iterations and args.mc_summary:
        itr = Path(args.mc_iterations).expanduser()
        summary = Path(args.mc_summary).expanduser()
        box_path, mc_stats = plot_mc_box(itr, summary, out_dir, args.format)
        artifacts["mc_boxplot"] = box_path
        data_dump["monte_carlo"] = mc_stats
    if args.walkforward_csv:
        wf = Path(args.walkforward_csv).expanduser()
        timeline_path, wf_records = plot_walkforward(wf, out_dir, args.format)
        artifacts["walkforward_timeline"] = timeline_path
        data_dump["walkforward"] = wf_records
    if args.equity_csv:
        equity_path = plot_equity(Path(args.equity_csv).expanduser(), out_dir, args.format)
        artifacts["equity_curve"] = equity_path
    if args.underwater_csv:
        underwater_path = plot_underwater(Path(args.underwater_csv).expanduser(), out_dir, args.format)
        artifacts["underwater_curve"] = underwater_path

    meta = {
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "inputs": {
            "batch_csv": args.batch_csv,
            "walkforward_csv": args.walkforward_csv,
            "mc_summary": args.mc_summary,
            "mc_iterations": args.mc_iterations,
            "equity_csv": args.equity_csv,
            "underwater_csv": args.underwater_csv,
        },
        "artifacts": artifacts,
    }
    meta_path = out_dir / "diagnostics_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("Diagnostics metadata written to %s", meta_path)

    data_path = out_dir / "diagnostics_data.json"
    data_path.write_text(json.dumps(data_dump, indent=2, default=str), encoding="utf-8")
    logger.info("Diagnostics data written to %s", data_path)


if __name__ == "__main__":
    main()
