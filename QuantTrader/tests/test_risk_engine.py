import unittest

from QuantTrader.core.risk.risk_engine import RiskEngine, RiskLimits


class RiskEngineTest(unittest.TestCase):
    def setUp(self):
        limits = RiskLimits(
            max_position_notional=100000,
            max_gross_leverage=2.0,
            max_daily_loss=5000,
            max_drawdown=0.1,
        )
        self.engine = RiskEngine(limits=limits, starting_equity=50000)

    def test_exposure_limit(self):
        ok, _ = self.engine.evaluate_order("EURUSD", "buy", 90000)
        self.assertTrue(ok)
        self.engine.record_fill("EURUSD", "buy", 90000, pnl=0)
        ok, reason = self.engine.evaluate_order("EURUSD", "buy", 20000)
        self.assertFalse(ok)
        self.assertEqual(reason, "symbol_exposure_limit:EURUSD")

    def test_daily_loss_limit(self):
        ok, _ = self.engine.check_loss_limits()
        self.assertTrue(ok)
        self.engine.record_fill("USDJPY", "sell", 50000, pnl=-6000)
        ok, reason = self.engine.check_loss_limits()
        self.assertFalse(ok)
        self.assertEqual(reason, "daily_loss_limit")


if __name__ == "__main__":
    unittest.main()
