# QuantTrader Artifacts

This directory stores the research outputs that the trading runtime consumes.

## Structure
- `config/`: strategy configuration snapshots exported from QuantResearch (YAML).
- `params/`: optimized parameter JSONs (`best_params_*.json`).

## Manual sync process
1. In `QuantResearch/`, run optimization/backtest scripts to produce updated configs/params.
2. Copy the vetted files into this directory:
   - `cp QuantResearch/config/<strategy>.yaml QuantTrader/artifacts/config/`
   - `cp QuantResearch/data/params/best_*.json QuantTrader/artifacts/params/`
3. Commit the new artifacts (or upload to storage) alongside the trading release.
4. Runner processes load configs from `artifacts/config/` and parameters from `artifacts/params/` to ensure live trading uses the approved research snapshot.

Automating this sync (e.g., via CI) is recommended once the promotion flow stabilizes.
