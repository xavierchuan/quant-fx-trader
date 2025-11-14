"""Paper trading adapter that simulates fills with configurable latency/slippage."""

from __future__ import annotations

import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict

from .adapter import CancelAck, ExecutionAdapter, OrderAck, OrderParams, PositionState
from .metrics_logger import log_event
from .order_store import OrderStore


class PaperAdapter(ExecutionAdapter):
    def __init__(
        self,
        latency_ms: float = 50.0,
        slippage_pips: float = 0.1,
        order_store: OrderStore | None = None,
        starting_equity: float = 100000.0,
        equity_log_path: str = "results/execution/paper_equity.csv",
    ):
        self.latency_ms = latency_ms
        self.slippage_pips = slippage_pips
        self.order_store = order_store or OrderStore("results/execution/paper_orders.log")
        self.positions: Dict[str, PositionState] = {}
        self.last_price: Dict[str, float] = {}
        self.cash = starting_equity
        self.equity_path = Path(equity_log_path)
        self.equity_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.equity_path.exists():
            self.equity_path.write_text("ts,equity\n", encoding="utf-8")
        self._order_seq = 0

    def _next_id(self) -> str:
        self._order_seq += 1
        return f"PAPER-{self._order_seq}"

    def submit(self, order: OrderParams) -> OrderAck:
        start = datetime.utcnow()
        time.sleep(self.latency_ms / 1000.0)
        order_id = self._next_id()
        fill_price = self._fill_price(order)
        self._apply_fill(order, fill_price)
        self.order_store.append(order_id, order)
        end = datetime.utcnow()
        log_event("submit", order_id, order.symbol, "accepted", start, end)
        self._record_equity(end)
        return OrderAck(order_id=order_id, status="accepted", timestamp=end)

    def cancel(self, order_id: str) -> CancelAck:
        start = datetime.utcnow()
        time.sleep(self.latency_ms / 2000.0)
        end = datetime.utcnow()
        log_event("cancel", order_id, "", "cancelled", start, end)
        return CancelAck(order_id=order_id, status="cancelled", timestamp=end)

    def sync_positions(self) -> dict[str, PositionState]:
        return self.positions

    def heartbeat(self) -> bool:
        return True

    def _pip_value(self, symbol: str) -> float:
        return 0.01 if symbol.endswith("JPY") else 0.0001

    def _fill_price(self, order: OrderParams) -> float:
        base_price = order.price or self.last_price.get(order.symbol, 1.0)
        slip = self.slippage_pips * self._pip_value(order.symbol)
        direction = 1 if order.side.lower() == "buy" else -1
        return base_price + direction * slip * random.choice([1, -1])

    def _apply_fill(self, order: OrderParams, price: float) -> None:
        qty = order.quantity if order.side.lower() == "buy" else -order.quantity
        pos = self.positions.get(order.symbol)
        if pos is None:
            pos = PositionState(symbol=order.symbol, quantity=0.0, avg_price=price, unrealized_pnl=0.0)
        total_qty = pos.quantity + qty
        if total_qty == 0:
            realized = (price - pos.avg_price) * (-qty)  # closing position
            self.cash += realized
            self.positions.pop(order.symbol, None)
        else:
            if pos.quantity == 0 or (pos.quantity > 0 and qty > 0) or (pos.quantity < 0 and qty < 0):
                avg = ((pos.quantity * pos.avg_price) + (qty * price)) / total_qty
                pos = PositionState(symbol=order.symbol, quantity=total_qty, avg_price=avg, unrealized_pnl=0.0)
                self.positions[order.symbol] = pos
            else:
                realized = (pos.avg_price - price) * qty * -1
                self.cash += realized
                pos = PositionState(symbol=order.symbol, quantity=total_qty, avg_price=pos.avg_price, unrealized_pnl=0.0)
                if total_qty == 0:
                    self.positions.pop(order.symbol, None)
                else:
                    self.positions[order.symbol] = pos
        notional = price * order.quantity
        if qty > 0:
            self.cash -= notional
        else:
            self.cash += notional
        self.last_price[order.symbol] = price

    def _record_equity(self, timestamp: datetime) -> None:
        equity = self.cash
        for symbol, pos in self.positions.items():
            mark = self.last_price.get(symbol, pos.avg_price)
            equity += pos.quantity * mark
        with self.equity_path.open("a", encoding="utf-8") as fh:
            fh.write(f"{timestamp.isoformat()},{equity}\n")
