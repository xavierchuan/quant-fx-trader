# FX Backtest & Execution Stack

End-to-end foreign-exchange research and trading toolkit that links feature engineering, backtesting, ML-driven signal generation, and OANDA execution into one repo.

## Highlights

- Unified `StrategyEngine` (see `QuantResearch/core/backtest/strategy_engine.py`) powers historical backtests, walk-forward studies, paper trading, and the live runner so signals behave identically across environments.
- Strategy registry ships with SMA/ATR trend, Bollinger & band mean-revert, breakout momentum, and an XGBoost probability model (`QuantResearch/strategies/*`), allowing multi-strategy voting through YAML configs such as `QuantTrader/config/usdjpy_multi_strategy.yaml`.
- Research workflows enforce data-manifest validation, risk sims, KPI summaries (`results/<run_id>/summary.json`), and promotion of vetted artifacts into `QuantTrader/artifacts/` before they are allowed to reach trading.
- Runtime layer contains async OANDA data/execution handlers, event-driven risk checks, and pluggable multi-strategy allocation for both paper (`scripts/paper_trade.py`) and live trading (`scripts/live_trade.py`).
- Monitoring stack (Pushgateway + Prometheus + Grafana) ships ready-to-import risk dashboards (`monitoring/grafana/*.json`), custom drilldown plugins (logs/traces/profiles/metrics), and Slack/pushgateway hooks for diagnostics automation.

## Repository Layout

- `QuantResearch/` – Research code, datasets, strategy implementations, notebooks/scripts, docs, artifacts, and test suites.
- `QuantTrader/` – Trading runtime with execution/risk/data engines, configs, logging, and artifact promotion targets.
- `monitoring/` – Dockerized observability stack plus Grafana dashboards & plugins for metrics/logs/traces/profiles.
- `shared/` – Cross-cutting helpers (`shared/utils/config.py` loads OANDA/Slack/Pushgateway secrets from `.env`).
- `results/` – Canonical run outputs uploaded with PRs (e.g., walk-forward summaries) for auditing.
- `metrics/` – Lightweight operational CSVs (e.g., execution latencies) that can be pushed to Prometheus.

## Quick Start

1. **Clone & create a virtual environment**

   ```bash
   git clone <your fork url>
   cd FX_Backtest
   python -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r QuantResearch/requirements.txt
   pip install -r QuantTrader/requirements.txt
   ```

   Python 3.10+ is recommended for `pandas`/`xgboost` compatibility.

2. **Configure secrets**

   ```bash
   cp .env.demo .env
   # edit .env with your OANDA practice/live credentials + webhook URLs
   source .env
   ```

   All scripts that touch OANDA import from `shared.utils.config`, so missing env vars fail fast.

3. **Prepare data**

   - Drop raw CSVs (e.g., `USDJPY_H1.csv`) under `QuantResearch/data/raw/`.
   - Rebuild the manifest + integrity reports any time data changes:

     ```bash
     cd QuantResearch
     python scripts/build_dataset_manifest.py --dirs data/raw data/derived --output data/_manifest.json
     python scripts/check_data_integrity.py
     ```

4. **Run a backtest**

   ```bash
   python QuantResearch/scripts/backtest_strategy.py \
     --csv QuantResearch/data/raw/USDJPY_H1.csv \
     --symbol USDJPY \
     --fast 20 --slow 80 \
     --strategies QuantTrader/config/usdjpy_multi_strategy.yaml
   ```

   The script validates the dataset, runs the engine, and writes KPIs plus `equity/`, `trades/`, and `stats/` artifacts under `QuantResearch/data/outputs/`.

5. **Train or refresh the XGBoost signal**

   ```bash
   python QuantResearch/scripts/train_xgb_usdjpy.py \
     --csv QuantResearch/data/raw/USDJPY_H1.csv \
     --symbol USDJPY \
     --out QuantResearch/artifacts/models/usdjpy_h1_xgb
   ```

   This exports `model.json`, feature lists, thresholds, and updates `usdjpy_h1_xgb_latest.json` so trading configs can point to the latest model.

6. **Run walk-forward analysis (optional gating)**

   ```bash
   python QuantResearch/scripts/run_walkforward.py \
     --config QuantTrader/config/usdjpy_multi_strategy.yaml \
     --csv QuantResearch/data/raw/USDJPY_H1.csv \
     --train-bars 4000 --test-bars 1000 \
     --output-root QuantResearch/results \
     --label usdjpy_xgb
   ```

   Each window produces metrics and a `summary.json` under `QuantResearch/results/<run_id>/`. Reference these run IDs in PRs.

7. **Promote artifacts to the trader**

   After validating a run, sync configs/params into `QuantTrader/artifacts/` (see `QuantTrader/artifacts/README.md`):

   ```bash
   cp QuantTrader/config/usdjpy_multi_strategy.yaml QuantTrader/artifacts/config/
   cp QuantResearch/artifacts/models/usdjpy_h1_xgb_latest.json QuantTrader/artifacts/params/
   ```

8. **Paper trading or live execution**

   - Paper (uses live pricing -> StrategyEngine -> simulated fills):

     ```bash
     python QuantTrader/scripts/paper_trade.py \
       --config QuantTrader/config/usdjpy_multi_strategy.yaml \
       --symbol USDJPY \
       --timeframe 60s
     ```

   - Live example (direct OANDA handler + RSI strategy template, see `QuantTrader/scripts/live_trade.py`):

     ```bash
     python QuantTrader/scripts/live_trade.py
     ```

   Customize the risk manager, strategy, and execution handler before pointing to a funded account.

9. **Spin up monitoring (optional but recommended)**

   ```bash
   docker compose up -d
   ```

   This launches Pushgateway (`:9091`), Prometheus (`:9090`), and Grafana (`:3000`). Import `monitoring/grafana/risk_metrics_dashboard.json` and enable the bundled drilldown plugins for logs/traces/profiles/metrics exploration.

## Common Workflows

- **Data quality gating:** `python QuantResearch/scripts/watch_quality.py` or the CI-friendly `scripts/watch_risk_metrics.py` push metrics to Slack/Pushgateway before PRs merge.
- **Batch experiments:** `python QuantResearch/scripts/run_batch_backtests.py --config config/eurusd_grid.yaml` sweeps parameter grids and streams metrics under `results/<batch>/`.
- **Stress testing:** `python QuantResearch/scripts/validate_stress_scenarios.py --config ...` replays adverse cost scenarios to validate drawdown budgets.
- **Risk sims:** `RUN=<run_id> ./QuantResearch/scripts/run_risk_sim.sh && ./QuantResearch/bin/backfill_risk.sh` keep `results/risk/metrics.csv` aligned with latest runs.

## Monitoring & Diagnostics

- `QuantResearch/scripts/export_metrics_prom.py` streams aggregated KPIs to Pushgateway (`PUSHGATEWAY_URL`).
- `QuantResearch/scripts/notify_risk_metrics.sh` wraps `watch_risk_metrics.py` to send Slack alerts using `SLACK_RISK_WEBHOOK`.
- Grafana plugins under `monitoring/grafana/plugins/grafana-*-app/` document the queryless drilldown experiences for logs (Loki), metrics (Prometheus), traces (Tempo), and profiles (Pyroscope).
- `monitoring/grafana/risk_metrics_dashboard.json` visualizes walk-forward pass rates, tail risk, exposure, and per-strategy attribution. Load it after Grafana boots (`admin/admin` by default).

## Testing & Validation

- Unit tests: `pytest QuantResearch/tests QuantTrader/tests`.
- Strategy registry coverage: `QuantResearch/tests/test_strategy_registry.py` ensures new strategies register correctly; add fixtures before contributing.
- Result validation: `python QuantResearch/scripts/validate_results.py QuantResearch/results/<run_id>` checks KPI completeness + data references.
- Data feed/execution smoke tests: `python QuantTrader/tests/test_execution_adapters.py` mocks OANDA flows.

## Extending the Stack

1. Implement a new research strategy under `QuantResearch/strategies/` and decorate it with `@register("my_strategy")`.
2. Reference it inside a config YAML (e.g., `usdjpy_multi_strategy.yaml`) with weights/params.
3. Add risk rules in `QuantTrader/core/risk/` if the position sizing model needs to change.
4. Document any new process in `QuantResearch/docs/` or module-level READMEs so CI reviewers have breadcrumbs.

## Related Docs

- `QuantResearch/README.md` – data submission rules, risk/diagnostics workflow.
- `QuantTrader/artifacts/README.md` – promotion checklist for configs/params.
- `monitoring/grafana/plugins/*/README.md` – upstream plugin instructions.

## License

No open-source license is declared yet. Keep the repository private or add a LICENSE file before publishing.

