#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

RUN_ID=${RUN:-$(date +"risk_%Y%m%d_%H%M%S")}
RISK_LOG="results/risk/events.jsonl"

# 每次运行前清理旧的风险事件，避免历史数据干扰统计
mkdir -p "$(dirname "$RISK_LOG")"
: > "$RISK_LOG"

# 自动发现 trades.csv（优先使用显式 TRADES，未设置则读取 summary.json 中的 artifacts.trades）
if [ -z "${TRADES:-}" ]; then
  SUMMARY="results/${RUN_ID}/summary.json"
  if [ -f "$SUMMARY" ]; then
    TRADES=$(python - <<'PY' "$SUMMARY"
import json, sys
with open(sys.argv[1], encoding="utf-8") as fh:
    summary = json.load(fh)
print(summary.get("artifacts", {}).get("trades", ""))
PY
)
  fi
fi

# 如果 summary 中给的是相对路径，补全为绝对路径
if [ -n "${TRADES:-}" ] && [ ! -f "$TRADES" ] && [ -f "$PWD/$TRADES" ]; then
  TRADES="$PWD/$TRADES"
fi

if [ -z "${TRADES:-}" ] || [ ! -f "$TRADES" ]; then
  echo "Trades CSV not found. Provide TRADES env or ensure results/${RUN_ID}/summary.json.artifacts.trades 指向有效文件。" >&2
  exit 1
fi

python scripts/simulate_execution.py --trades-csv "$TRADES" \
  --risk-limits-yaml ../QuantTrader/config/risk_limits_sim.yaml \
  --adapter paper \
  --paper-latency-ms 25 \
  --paper-slippage-pips 0.05 \
  --run-id "$RUN_ID" \
  --risk-log "$RISK_LOG"
python scripts/risk_report.py --log "$RISK_LOG" --out results/risk/report.csv --run-id "$RUN_ID" --skip-metrics
if python scripts/check_risk_report.py --report results/risk/report.csv --max-rejects 0 --max-kill 0; then
  STATUS="pass"
else
  STATUS="fail"
fi
python scripts/risk_report.py --log "$RISK_LOG" --out results/risk/report.csv --run-id "$RUN_ID" --status "$STATUS" --skip-report
if [ -s "$RISK_LOG" ]; then
  cp "$RISK_LOG" "results/risk/events_${RUN_ID}.jsonl"
fi
if [ "$STATUS" != "pass" ]; then
  exit 1
fi
