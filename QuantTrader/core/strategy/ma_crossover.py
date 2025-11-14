from typing import List, Optional
import pandas as pd
from datetime import datetime

from .base import Strategy, SignalEvent
from .base import Position
from ..data.base import MarketDataEvent

class MACrossoverStrategy(Strategy):
    """
    简单移动平均交叉策略
    快线上穿慢线做多，下穿做空
    """
    def __init__(self, instrument: str, fast_period: int = 50, slow_period: int = 200, position_size: float = 1.0):
        super().__init__(instrument, position_size)
        self.fast_period = fast_period
        self.slow_period = slow_period

    async def on_data(self, event: MarketDataEvent) -> Optional[SignalEvent]:
        if self.historical_data is None:
            return None
        # 更新收盘价
        close = event.data.get('close') or event.data.get('mid')
        self.historical_data.loc[event.timestamp] = {
            'open': event.data.get('open', close),
            'high': event.data.get('high', close),
            'low': event.data.get('low', close),
            'close': close
        }
        if len(self.historical_data) < self.slow_period:
            return None

        fast = self.historical_data['close'].rolling(self.fast_period).mean()
        slow = self.historical_data['close'].rolling(self.slow_period).mean()

        if fast.iloc[-2] <= slow.iloc[-2] and fast.iloc[-1] > slow.iloc[-1]:
            return SignalEvent(
                instrument=self.instrument,
                timestamp=event.timestamp,
                signal_type="LONG",
                direction="BUY",
                strength=self.position_size
            )
        if fast.iloc[-2] >= slow.iloc[-2] and fast.iloc[-1] < slow.iloc[-1]:
            return SignalEvent(
                instrument=self.instrument,
                timestamp=event.timestamp,
                signal_type="SHORT",
                direction="SELL",
                strength=self.position_size
            )
        return None

    async def calculate_signals(self, data: pd.DataFrame) -> List[SignalEvent]:
        signals = []
        self.historical_data = data.copy()
        fast = data['close'].rolling(self.fast_period).mean()
        slow = data['close'].rolling(self.slow_period).mean()
        for i in range(self.slow_period, len(data)):
            ts = data.index[i]
            if fast.iloc[i-1] <= slow.iloc[i-1] and fast.iloc[i] > slow.iloc[i]:
                signals.append(SignalEvent(instrument=self.instrument, timestamp=ts, signal_type="LONG", direction="BUY", strength=self.position_size))
            if fast.iloc[i-1] >= slow.iloc[i-1] and fast.iloc[i] < slow.iloc[i]:
                signals.append(SignalEvent(instrument=self.instrument, timestamp=ts, signal_type="SHORT", direction="SELL", strength=self.position_size))
        return signals
