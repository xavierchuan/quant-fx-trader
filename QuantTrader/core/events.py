# fx_backtest/core/events.py
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Optional

@dataclass(frozen=True)
class TickEvent:
    ts: datetime
    symbol: str
    bid: float
    ask: float

@dataclass(frozen=True)
class SignalEvent:
    ts: datetime
    symbol: str
    direction: Literal["LONG", "SHORT", "EXIT"]
    size: float  # 单位：合约单位/手，随你定义

@dataclass(frozen=True)
class OrderEvent:
    ts: datetime
    symbol: str
    side: Literal["BUY", "SELL"]
    size: float
    price: Optional[float] = None  # 市价可为 None；限价时填价格

@dataclass(frozen=True)
class FillEvent:
    ts: datetime
    symbol: str
    side: Literal["BUY", "SELL"]
    size: float
    price: float
    commission: float