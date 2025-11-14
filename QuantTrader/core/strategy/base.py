from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd

from ..data.base import MarketDataEvent

class Position:
    """持仓类，表示当前市场头寸"""
    
    def __init__(
        self,
        instrument: str,
        direction: str,  # "LONG" or "SHORT"
        size: float,
        entry_price: float,
        entry_time: datetime,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ):
        self.instrument = instrument
        self.direction = direction
        self.size = size
        self.entry_price = entry_price
        self.entry_time = entry_time
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0

class SignalEvent:
    """交易信号事件"""
    
    def __init__(
        self,
        instrument: str,
        timestamp: datetime,
        signal_type: str,  # "LONG", "SHORT", "EXIT"
        direction: str,
        strength: float = 1.0,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ):
        self.event_type = "SIGNAL"
        self.instrument = instrument
        self.timestamp = timestamp
        self.signal_type = signal_type
        self.direction = direction
        self.strength = strength
        self.stop_loss = stop_loss
        self.take_profit = take_profit

class Strategy(ABC):
    """
    策略基类
    定义了策略开发的标准接口
    """
    
    def __init__(
        self,
        instrument: str,
        position_size: float = 1.0,
        max_positions: int = 1
    ):
        self.instrument = instrument
        self.position_size = position_size
        self.max_positions = max_positions
        self.positions: List[Position] = []
        self.historical_data: Optional[pd.DataFrame] = None
        
    @abstractmethod
    async def on_data(self, event: MarketDataEvent) -> Optional[SignalEvent]:
        """
        处理市场数据更新
        
        Args:
            event: 市场数据事件
            
        Returns:
            如果产生交易信号，返回SignalEvent；否则返回None
        """
        pass
        
    @abstractmethod
    async def calculate_signals(self, data: pd.DataFrame) -> List[SignalEvent]:
        """
        基于历史数据计算交易信号
        
        Args:
            data: 历史市场数据
            
        Returns:
            交易信号列表
        """
        pass
    
    def update_position(self, position: Position, current_price: float) -> None:
        """
        更新持仓的未实现盈亏
        
        Args:
            position: 需要更新的持仓
            current_price: 当前市场价格
        """
        if position.direction == "LONG":
            position.unrealized_pnl = (current_price - position.entry_price) * position.size
        else:
            position.unrealized_pnl = (position.entry_price - current_price) * position.size
            
    def can_open_position(self) -> bool:
        """检查是否可以开新仓位"""
        return len(self.positions) < self.max_positions
        
    def get_position_value(self) -> float:
        """获取当前持仓的总价值"""
        return sum(abs(pos.unrealized_pnl) for pos in self.positions)
        
    def get_total_pnl(self) -> float:
        """获取总盈亏（已实现 + 未实现）"""
        unrealized = sum(pos.unrealized_pnl for pos in self.positions)
        realized = sum(pos.realized_pnl for pos in self.positions)
        return realized + unrealized