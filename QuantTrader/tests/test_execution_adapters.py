import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from QuantTrader.execution.adapter import OrderParams
from QuantTrader.execution.oanda_adapter import OandaAdapter
from QuantTrader.execution.order_store import OrderStore
from QuantTrader.execution.paper_adapter import PaperAdapter


class ExecutionAdaptersTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        os.environ["EXECUTION_METRICS_PATH"] = str(
            (tempfile.NamedTemporaryFile(delete=False, dir=self.tmpdir.name).name)
        )

    def tearDown(self):
        self.tmpdir.cleanup()
        os.environ.pop("EXECUTION_METRICS_PATH", None)

    def test_oanda_adapter_submit_uses_order_store(self):
        store_path = f"{self.tmpdir.name}/orders.log"
        adapter = OandaAdapter(
            account_id="ACC",
            token="TOKEN",
            order_store=OrderStore(store_path),
            base_url="https://example.com",
            metrics_path=os.environ["EXECUTION_METRICS_PATH"],
        )
        stub_response = {
            "orderCreateTransaction": {
                "id": "123",
                "time": "2024-01-01T00:00:00.000000Z",
            }
        }
        adapter._request = MagicMock(return_value=stub_response)  # type: ignore
        order = OrderParams(symbol="EUR_USD", side="buy", quantity=1000)
        ack = adapter.submit(order)
        self.assertEqual(ack.order_id, "123")
        self.assertTrue(Path(store_path).exists())

    def test_paper_adapter_generates_ids_and_logs_equity(self):
        store_path = f"{self.tmpdir.name}/paper.log"
        equity_path = f"{self.tmpdir.name}/equity.csv"
        adapter = PaperAdapter(
            latency_ms=1,
            slippage_pips=0.0,
            order_store=OrderStore(store_path),
            equity_log_path=equity_path,
        )
        order = OrderParams(symbol="EURUSD", side="buy", quantity=1000, price=1.1)
        ack = adapter.submit(order)
        self.assertTrue(ack.order_id.startswith("PAPER-"))
        cancel_ack = adapter.cancel(ack.order_id)
        self.assertEqual(cancel_ack.status, "cancelled")
        self.assertTrue(Path(equity_path).exists())


if __name__ == "__main__":
    unittest.main()
