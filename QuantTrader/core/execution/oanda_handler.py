from typing import Optional
from datetime import datetime
from loguru import logger
from oandapyV20 import API
import oandapyV20.endpoints.orders as orders
import csv
from pathlib import Path

from .base import ExecutionHandler, OrderEvent, FillEvent

class OANDAExecutionHandler(ExecutionHandler):
    """实盘环境下的 OANDA 执行实现。"""

    def __init__(self, account_id: str, access_token: str, environment: str = "practice", fills_path: Optional[str] = None):
        super().__init__()
        self.client = API(access_token=access_token, environment=environment)
        self.account_id = account_id
        default_dir = Path("results/execution/live") if environment == "live" else Path("results/execution/paper")
        default_dir.mkdir(parents=True, exist_ok=True)
        self._fills_csv = Path(fills_path) if fills_path else default_dir / "fills.csv"
        if not self._fills_csv.exists():
            self._init_csv()

    def _init_csv(self) -> None:
        with self._fills_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["order_id", "ts", "symbol", "pnl", "adapter_latency_ms", "direction", "price", "quantity"],
            )
            writer.writeheader()

    async def execute_order(self, order: OrderEvent) -> None:
        # 将 OrderEvent 转换为 OANDA 下单请求
        instrument = order.instrument.replace("_", "_")
        units = int(order.quantity) if order.direction.upper() in ("BUY", "LONG") else -int(order.quantity)

        order_body = {
            "instrument": instrument,
            "units": str(units),
            "timeInForce": "FOK",
            "positionFill": "DEFAULT",
        }
        if order.price:
            order_body["type"] = "LIMIT"
            order_body["price"] = f"{order.price:.5f}"
            order_body["timeInForce"] = "GTC"
        else:
            order_body["type"] = "MARKET"

        payload = {"order": order_body}
        req = orders.OrderCreate(accountID=self.account_id, data=payload)
        try:
            resp = self.client.request(req)
        except Exception as exc:
            logger.error(f"[OANDA] Order submission failed for {instrument}: {exc}")
            order.status = "REJECTED"
            await self._notify_order(order)
            return

        order.status = "SUBMITTED"
        await self._notify_order(order)

        fill_txn = resp.get("orderFillTransaction")
        if not fill_txn:
            logger.warning(f"[OANDA] Order accepted but no fill: {resp}")
            return

        try:
            price = float(fill_txn["price"])
            filled_units = abs(float(fill_txn["units"]))
            commission = float(fill_txn.get("commission", 0))
            ts = datetime.fromisoformat(fill_txn["time"].replace("Z", "+00:00"))
            side = "BUY" if float(fill_txn["units"]) > 0 else "SELL"
        except Exception as exc:
            logger.error(f"[OANDA] Unable to parse fill transaction: {fill_txn} ({exc})")
            return

        fill = FillEvent(
            instrument=order.instrument,
            direction=side,
            quantity=filled_units,
            price=price,
            timestamp=ts,
            commission=abs(commission),
            order_id=order.order_id
        )

        order.status = "FILLED"
        await self._notify_order(order)
        await self._notify_fill(fill)
        self.fills.append(fill)
        self.update_position(fill)
        self._append_fill_csv(fill)

    def _append_fill_csv(self, fill: FillEvent) -> None:
        record = {
            "order_id": fill.order_id or "",
            "ts": fill.timestamp.isoformat(),
            "symbol": fill.instrument,
            "pnl": 0.0,
            "adapter_latency_ms": None,
            "direction": fill.direction,
            "price": fill.price,
            "quantity": fill.quantity,
        }
        with self._fills_csv.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=record.keys())
            writer.writerow(record)

    async def cancel_order(self, order_id: str) -> bool:
        # OANDA 取消需要调用 OrderCancel 或交易 API；简单实现为更新状态
        if order_id in self.orders:
            order = self.orders[order_id]
            order.status = "CANCELLED"
            await self._notify_order(order)
            return True
        return False
