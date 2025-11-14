#!/usr/bin/env bash
set -euo pipefail
if [ -z "${SLACK_RISK_WEBHOOK:-}" ]; then
  echo "SLACK_RISK_WEBHOOK not set" >&2
  exit 1
fi
cd "$(dirname "$0")/.."
set +e
output=$(python scripts/watch_ops_metrics.py --csv results/risk/metrics.csv 2>&1)
status=$?
set -e
export OUTPUT="$output"
if [ $status -ne 0 ]; then
  payload=$(python - <<'PY'
import json, os
msg = os.environ['OUTPUT']
print(json.dumps({"text": f"[Risk Alert] {msg}"}))
PY
)
  curl -s -X POST -H 'Content-type: application/json' --data "$payload" "$SLACK_RISK_WEBHOOK"
  echo "$output"
  exit 1
else
  echo "$output"
fi
