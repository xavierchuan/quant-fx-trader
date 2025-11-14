#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Auto-select newest artifacts unless explicit paths supplied.
TARGET=${1:-latest}
if [ "$TARGET" = "latest" ]; then
  BATCH=$(ls -t data/results/batch_backtests_*.csv 2>/dev/null | head -n1 || true)
  WALKFORWARD=$(ls -t results/*/walkforward/metrics.csv 2>/dev/null | head -n1 || true)
  MC_SUMMARY=$(ls -t results/*/stress/mc_summary.json 2>/dev/null | head -n1 || true)
  MC_ITER=$(ls -t results/*/stress/mc_iterations.csv 2>/dev/null | head -n1 || true)
  EQUITY=$(ls -t results/*/equity.csv 2>/dev/null | head -n1 || true)
  UNDERWATER=$(ls -t results/*/stats/underwater.csv 2>/dev/null | head -n1 || true)
fi
BATCH=${BATCH:-tests/fixtures/batch_results_sample.csv}
WALKFORWARD=${WALKFORWARD:-tests/fixtures/walkforward_metrics_sample.csv}
MC_SUMMARY=${MC_SUMMARY:-tests/fixtures/mc_summary_sample.json}
MC_ITER=${MC_ITER:-tests/fixtures/mc_iterations_sample.csv}
EQUITY=${EQUITY:-tests/fixtures/equity_sample.csv}
UNDERWATER=${UNDERWATER:-tests/fixtures/underwater_sample.csv}

OUT_DIR="charts/ci/diagnostics_$(date +%Y%m%d_%H%M%S)"

python scripts/plot_backtest_diagnostics.py \
  --batch-csv "$BATCH" \
  --walkforward-csv "$WALKFORWARD" \
  --mc-summary "$MC_SUMMARY" \
  --mc-iterations "$MC_ITER" \
  --equity-csv "$EQUITY" \
  --underwater-csv "$UNDERWATER" \
  --facet-scenario \
  --out "$OUT_DIR" \
  --format png

echo "Diagnostics artifacts saved to $OUT_DIR"
