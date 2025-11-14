from typing import List, Optional
import pandas as pd
from datetime import datetime

from .base import Strategy, SignalEvent
from ..data.base import MarketDataEvent

class MomentumStrategy(Strategy):
    """
    简单动量策略：当价格高于N日均线时做多，低于时做空
    """
    def __init__(self, instrument: str, lookback: int = 20, position_size: float = 1.0):
        super().__init__(instrument, position_size)
        self.lookback = lookback

    async def on_data(self, event: MarketDataEvent) -> Optional[SignalEvent]:
        if self.historical_data is None:
            return None
        close = event.data.get('close') or event.data.get('mid')
        self.historical_data.loc[event.timestamp] = {
            'open': event.data.get('open', close),
            'high': event.data.get('high', close),
            'low': event.data.get('low', close),
            'close': close
        }
        if len(self.historical_data) < self.lookback:
            return None
        ma = self.historical_data['close'].rolling(self.lookback).mean()
        if close > ma.iloc[-1] and self.can_open_position():
            return SignalEvent(self.instrument, event.timestamp, "LONG", "BUY", strength=self.position_size)
        if close < ma.iloc[-1] and self.can_open_position():
            return SignalEvent(self.instrument, event.timestamp, "SHORT", "SELL", strength=self.position_size)
        return None

    async def calculate_signals(self, data: pd.DataFrame) -> List[SignalEvent]:
        signals = []
        self.historical_data = data.copy()
        ma = data['close'].rolling(self.lookback).mean()
        for i in range(self.lookback, len(data)):
            ts = data.index[i]
            price = data['close'].iloc[i]
            if price > ma.iloc[i-1]:
                signals.append(SignalEvent(self.instrument, ts, "LONG", "BUY", strength=self.position_size))
            elif price < ma.iloc[i-1]:
                signals.append(SignalEvent(self.instrument, ts, "SHORT", "SELL", strength=self.position_size))
        return signals
