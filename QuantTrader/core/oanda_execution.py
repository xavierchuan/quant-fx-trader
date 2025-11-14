"""
OANDA 执行适配器：把 OrderEvent 转换为 OANDA API 下单，并回写 FillEvent。
"""

from __future__ import annotations

from queue import Queue

import pandas as pd
from loguru import logger
from oandapyV20 import API
import oandapyV20.endpoints.orders as orders

from .events import FillEvent, OrderEvent


def _normalize_instrument(symbol: str) -> str:
    s = symbol.upper().replace(" ", "").replace("/", "_").replace("-", "_")
    if "_" in s and len(s) == 7:
        return s
    stripped = s.replace("_", "")
    if len(stripped) == 6:
        return f"{stripped[:3]}_{stripped[3:]}"
    return s


class OandaExecution:
    """
    把 OrderEvent 翻译为 OANDA 订单；成交后投递 FillEvent。
    """

    def __init__(
        self,
        q: Queue,
        account_id: str,
        access_token: str,
        environment: str = "practice",
    ) -> None:
        self.q = q
        self.account_id = account_id
        self.client = API(access_token=access_token, environment=environment)

    def on_event(self, ev) -> None:
        if not isinstance(ev, OrderEvent):
            return
        instrument = _normalize_instrument(ev.symbol)
        units = ev.size if ev.side == "BUY" else -ev.size
        order_body = {
            "instrument": instrument,
            "units": str(int(units)),
            "timeInForce": "FOK",
            "positionFill": "DEFAULT",
        }
        if ev.price is None:
            order_body["type"] = "MARKET"
        else:
            order_body["type"] = "LIMIT"
            order_body["timeInForce"] = "GTC"
            order_body["price"] = f"{ev.price:.5f}"

        payload = {"order": order_body}
        req = orders.OrderCreate(accountID=self.account_id, data=payload)
        try:
            resp = self.client.request(req)
        except Exception as exc:
            logger.error(f"[OANDA] Order submission failed for {instrument}: {exc}")
            return

        fill_txn = resp.get("orderFillTransaction")
        if not fill_txn:
            logger.warning(f"[OANDA] Order accepted but no fill: {resp}")
            return

        try:
            price = float(fill_txn["price"])
            filled_units = abs(float(fill_txn["units"]))
            commission = float(fill_txn.get("commission", 0))
            ts = pd.to_datetime(fill_txn["time"]).to_pydatetime()
            side = "BUY" if float(fill_txn["units"]) > 0 else "SELL"
        except Exception as exc:
            logger.error(f"[OANDA] Unable to parse fill transaction: {fill_txn} ({exc})")
            return

        self.q.put(
            FillEvent(
                ts=ts,
                symbol=ev.symbol,
                side=side,
                size=filled_units,
                price=price,
                commission=abs(commission),
            )
        )
