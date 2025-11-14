from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml
from loguru import logger


DEFAULT_SESSIONS: List[Dict[str, Any]] = [
    {"name": "asia_open", "weekdays": [0, 1, 2, 3, 4], "start_hour": 0, "end_hour": 7},
    {"name": "europe", "weekdays": [0, 1, 2, 3, 4], "start_hour": 7, "end_hour": 13},
    {"name": "us_session", "weekdays": [0, 1, 2, 3, 4], "start_hour": 13, "end_hour": 22},
]


def _load_sessions(path: str | None) -> List[Dict[str, Any]]:
    if not path:
        return DEFAULT_SESSIONS
    fp = Path(path)
    if not fp.exists():
        raise FileNotFoundError(f"Session file not found: {path}")
    if fp.suffix.lower() in {".yml", ".yaml"}:
        data = yaml.safe_load(fp.read_text(encoding="utf-8")) or {}
    else:
        data = json.loads(fp.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        sessions = data.get("sessions") or data.get("profiles")
    else:
        sessions = data
    if not isinstance(sessions, list):
        raise ValueError("session file must contain a list of session definitions")
    return sessions


def _match_session(row: pd.Series, session: Dict[str, Any]) -> bool:
    hour = row["hour"]
    weekday = row["weekday"]
    weekdays = session.get("weekdays")
    if weekdays and weekday not in weekdays:
        return False
    start = session.get("start_hour")
    end = session.get("end_hour")
    if start is None and end is None:
        return True
    start = 0 if start is None else float(start)
    end = 24 if end is None else float(end)
    if start < end:
        return start <= hour < end
    return hour >= start or hour < end


def main():
    parser = argparse.ArgumentParser(description="Aggregate spread/slippage samples into cost profiles.")
    parser.add_argument("--input", required=True, help="CSV with columns ts,spread_pips,slip_pips (plus optional fields).")
    parser.add_argument("--symbol", required=True, help="Symbol name, e.g. USDJPY.")
    parser.add_argument("--out", default="data/cost_profiles/profile.yaml", help="Output YAML path.")
    parser.add_argument("--sessions", help="Optional YAML/JSON file describing session windows.")
    parser.add_argument("--min-samples", type=int, default=20, help="Minimum rows required for a session to be emitted.")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if "ts" not in df.columns:
        raise ValueError("input CSV must contain 'ts' column (timestamp).")
    if "spread_pips" not in df.columns or "slip_pips" not in df.columns:
        raise ValueError("input CSV must contain 'spread_pips' and 'slip_pips'.")

    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"])
    df["hour"] = df["ts"].dt.hour + df["ts"].dt.minute / 60.0
    df["weekday"] = df["ts"].dt.weekday

    sessions = _load_sessions(args.sessions)
    profiles: List[Dict[str, Any]] = []
    for session in sessions:
        mask = df.apply(lambda row: _match_session(row, session), axis=1)
        subset = df.loc[mask]
        if len(subset) < args.min_samples:
            logger.warning(f"Session {session.get('name')} skipped (samples={len(subset)} < {args.min_samples}).")
            continue
        profile = {
            "name": session.get("name", f"session_{len(profiles)}"),
            "weekdays": session.get("weekdays"),
            "start_hour": session.get("start_hour"),
            "end_hour": session.get("end_hour"),
            "spread": round(subset["spread_pips"].mean(), 4),
            "slip": round(subset["slip_pips"].mean(), 4),
            "comm": session.get("comm"),
            "samples": int(len(subset)),
            "spread_p95": round(subset["spread_pips"].quantile(0.95), 4),
            "slip_p95": round(subset["slip_pips"].quantile(0.95), 4),
        }
        if session.get("priority") is not None:
            profile["priority"] = session["priority"]
        profiles.append(profile)

    if not profiles:
        raise RuntimeError("No sessions met the minimum sample requirement; nothing to write.")

    default_profile = min(profiles, key=lambda p: p.get("priority", float("inf")))
    default_profile["default"] = True

    payload = {
        "symbol": args.symbol.upper(),
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "source": os.path.abspath(args.input),
        "profiles": profiles,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(payload, fh, allow_unicode=True, sort_keys=False)
    logger.info(f"Cost profile saved to {out_path}")


if __name__ == "__main__":
    main()
