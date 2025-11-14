"""Execution adapter interfaces and data models for Phase 3."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional


@dataclass
class OrderParams:
    symbol: str
    side: str  # "buy" or "sell"
    quantity: float
    price: Optional[float] = None
    tif: str = "GTC"
    client_order_id: Optional[str] = None
    metadata: Optional[Dict[str, str]] = None


@dataclass
class OrderAck:
    order_id: str
    status: str  # accepted/rejected
    timestamp: datetime
    reason: Optional[str] = None


@dataclass
class CancelAck:
    order_id: str
    status: str
    timestamp: datetime
    reason: Optional[str] = None


@dataclass
class FillEvent:
    order_id: str
    fill_id: str
    symbol: str
    side: str
    quantity: float
    price: float
    timestamp: datetime


@dataclass
class PositionState:
    symbol: str
    quantity: float
    avg_price: float
    unrealized_pnl: float = 0.0


class ExecutionAdapter(abc.ABC):
    """Abstract interface for broker/order routing adapters."""

    @abc.abstractmethod
    def submit(self, order: OrderParams) -> OrderAck:
        """Submit a new order to the venue."""

    @abc.abstractmethod
    def cancel(self, order_id: str) -> CancelAck:
        """Cancel an existing order."""

    @abc.abstractmethod
    def sync_positions(self) -> Dict[str, PositionState]:
        """Return latest position snapshot."""

    @abc.abstractmethod
    def heartbeat(self) -> bool:
        """Quick connectivity check."""


class MockAdapter(ExecutionAdapter):
    """MVP mock adapter used in simulation/testing pipelines."""

    def __init__(self):
        self._order_counter = 0
        self._orders: Dict[str, OrderParams] = {}

    def _next_id(self) -> str:
        self._order_counter += 1
        return f"SIM-{self._order_counter}"

    def submit(self, order: OrderParams) -> OrderAck:
        order_id = self._next_id()
        self._orders[order_id] = order
        return OrderAck(order_id=order_id, status="accepted", timestamp=datetime.utcnow())

    def cancel(self, order_id: str) -> CancelAck:
        status = "cancelled" if order_id in self._orders else "not_found"
        self._orders.pop(order_id, None)
        return CancelAck(order_id=order_id, status=status, timestamp=datetime.utcnow())

    def sync_positions(self) -> Dict[str, PositionState]:
        return {}

    def heartbeat(self) -> bool:
        return True
