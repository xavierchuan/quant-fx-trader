from typing import List, Optional
import pandas as pd
import numpy as np
from datetime import datetime

from ..data.base import MarketDataEvent
from .base import Strategy, SignalEvent

class RSIMeanReversionStrategy(Strategy):
    """
    RSI均值回归策略
    当RSI超买时做空，超卖时做多
    """
    
    def __init__(
        self,
        instrument: str,
        position_size: float = 1.0,
        max_positions: int = 1,
        rsi_period: int = 14,
        overbought: float = 70.0,
        oversold: float = 30.0,
        stop_loss_atr: float = 2.0,
        atr_period: int = 14
    ):
        super().__init__(instrument, position_size, max_positions)
        self.rsi_period = rsi_period
        self.overbought = overbought
        self.oversold = oversold
        self.stop_loss_atr = stop_loss_atr
        self.atr_period = atr_period
        self.last_rsi = None
        self.last_atr = None
        
    @staticmethod
    def calculate_rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """计算RSI指标"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def calculate_atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
        """计算ATR指标"""
        high = data['high']
        low = data['low']
        close = data['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        return tr.rolling(window=period).mean()
    
    async def on_data(self, event: MarketDataEvent) -> Optional[SignalEvent]:
        """
        处理实时市场数据
        
        Args:
            event: 市场数据事件
            
        Returns:
            如果触发信号则返回SignalEvent，否则返回None
        """
        if self.historical_data is None:
            return None
            
        # 更新数据
        current_price = event.data['mid']
        self.historical_data.loc[event.timestamp] = current_price
        
        # 计算指标
        close_prices = self.historical_data['close']
        rsi = self.calculate_rsi(close_prices, self.rsi_period).iloc[-1]
        atr = self.calculate_atr(self.historical_data, self.atr_period).iloc[-1]
        
        self.last_rsi = rsi
        self.last_atr = atr
        
        # 生成信号
        if self.can_open_position():
            if rsi > self.overbought:
                return SignalEvent(
                    instrument=self.instrument,
                    timestamp=event.timestamp,
                    signal_type="SHORT",
                    direction="SELL",
                    strength=self.position_size,
                    stop_loss=current_price + self.stop_loss_atr * atr
                )
            elif rsi < self.oversold:
                return SignalEvent(
                    instrument=self.instrument,
                    timestamp=event.timestamp,
                    signal_type="LONG",
                    direction="BUY",
                    strength=self.position_size,
                    stop_loss=current_price - self.stop_loss_atr * atr
                )
        
        return None
    
    async def calculate_signals(self, data: pd.DataFrame) -> List[SignalEvent]:
        """
        基于历史数据计算交易信号
        
        Args:
            data: 历史市场数据
            
        Returns:
            交易信号列表
        """
        signals = []
        self.historical_data = data.copy()
        
        # 计算指标
        close_prices = data['close']
        rsi = self.calculate_rsi(close_prices, self.rsi_period)
        atr = self.calculate_atr(data, self.atr_period)
        
        # 生成信号
        for i in range(self.rsi_period, len(data)):
            timestamp = data.index[i]
            current_price = close_prices[i]
            current_rsi = rsi[i]
            current_atr = atr[i]
            
            if current_rsi > self.overbought:
                signals.append(SignalEvent(
                    instrument=self.instrument,
                    timestamp=timestamp,
                    signal_type="SHORT",
                    direction="SELL",
                    strength=self.position_size,
                    stop_loss=current_price + self.stop_loss_atr * current_atr
                ))
            elif current_rsi < self.oversold:
                signals.append(SignalEvent(
                    instrument=self.instrument,
                    timestamp=timestamp,
                    signal_type="LONG",
                    direction="BUY",
                    strength=self.position_size,
                    stop_loss=current_price - self.stop_loss_atr * current_atr
                ))
                
        return signals