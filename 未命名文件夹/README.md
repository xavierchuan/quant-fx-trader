# QuantResearch

## 数据与回测提交须知

- 任何修改 `data/raw`、`data/derived`、`results/data_quality` 时，务必重新运行：
  ```bash
  python scripts/build_dataset_manifest.py --dirs data/raw data/derived --output data/_manifest.json
  python scripts/check_data_integrity.py
  ```
- 如果 `data/signature_baseline.json` 中的哈希需要更新，请在 PR 描述中写明原因、影响范围，并附上新的 `results/data_quality/*.json` 报告路径。
- 回测脚本会在 `results/<run_id>/summary.json` 记录本次运行的 KPI + 数据签名，提交代码时请一并引用该 run_id 便于审核。
- 提交前可执行 `python scripts/validate_results.py results/<run_id>`，快速检查 KPI 字段、数据报告引用是否完整。

## 风控与 Diagnostics 流程

1. **风险仿真守门**
   ```bash
   cd QuantResearch
   RUN=<run_id> ./scripts/run_risk_sim.sh
   ./bin/backfill_risk.sh   # 确保 results/risk/metrics.csv 同步最新 run
   ```
   `run_risk_sim.sh` 默认使用 `QuantTrader/config/risk_limits_sim.yaml`（宽松额度，仅用于数据 gating）。在 PR 描述中标注对应 run_id，并说明若需要引用实盘限额，应改用 `QuantTrader/config/risk_limits.yaml`。
   > CI 会在 diagnostics workflow 中调用 `python scripts/watch_risk_metrics.py`；若 metrics 中仍存在 fail/run 缺失，PR 会直接失败，因此务必在本地先修复再推送。

2. **诊断图表**
   ```bash
   ./scripts/run_ci_diagnostics.sh   # 自动挑选最新 batch/walkforward/MC 输出
   ```
   图表保存在 `charts/ci/diagnostics_<timestamp>/`，CI 可上传该目录作为 artifact 供审阅。

3. **Slack 告警（可选）**
   - 在本地或 CI 中设置 `SLACK_RISK_WEBHOOK=https://hooks.slack.com/services/...`，即可运行 `./scripts/notify_risk_metrics.sh`；脚本会调用 `watch_risk_metrics.py` 并在发现 fail 时发送告警。
   - 生产服务器可将同一命令写入 cron（示例：`*/30 * * * * cd /path/to/QuantResearch && source .env && ./scripts/notify_risk_metrics.sh`）；请参考 `.env.example` 填写 webhook。

4. **Prometheus 推送（Phase 4）**
   ```bash
   python scripts/export_metrics_prom.py | curl --data-binary @- http://pushgateway:9091/metrics/job/risk_sim
   ```
   CI 已支持该脚本；如需手动推送，请先在 `.env` 中配置 `PUSHGATEWAY_URL`。

更多运维细节见 `docs/runbook_paper_risk.md` 与 `docs/runbook_ops.md`。
