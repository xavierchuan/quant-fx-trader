from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from datetime import datetime
import asyncio
from loguru import logger

from ..strategy.base import SignalEvent, Position

class OrderEvent:
    """订单事件"""
    
    def __init__(
        self,
        instrument: str,
        order_type: str,  # "MARKET", "LIMIT", "STOP"
        direction: str,   # "BUY", "SELL"
        quantity: float,
        timestamp: datetime,
        price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        order_id: Optional[str] = None
    ):
        self.event_type = "ORDER"
        self.instrument = instrument
        self.order_type = order_type
        self.direction = direction
        self.quantity = quantity
        self.timestamp = timestamp
        self.price = price
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.order_id = order_id
        self.status = "CREATED"  # CREATED, SUBMITTED, FILLED, CANCELLED, REJECTED

class FillEvent:
    """成交事件"""
    
    def __init__(
        self,
        instrument: str,
        direction: str,
        quantity: float,
        price: float,
        timestamp: datetime,
        commission: float = 0.0,
        order_id: Optional[str] = None
    ):
        self.event_type = "FILL"
        self.instrument = instrument
        self.direction = direction
        self.quantity = quantity
        self.price = price
        self.timestamp = timestamp
        self.commission = commission
        self.order_id = order_id

class ExecutionHandler(ABC):
    """
    执行处理器基类
    负责订单执行和管理
    """
    
    def __init__(self):
        self.orders: Dict[str, OrderEvent] = {}  # order_id -> OrderEvent
        self.positions: Dict[str, Position] = {}  # instrument -> Position
        self.fills: List[FillEvent] = []
        self._order_callbacks = []
        self._fill_callbacks = []
        
    async def process_signal(self, signal: SignalEvent, price: Optional[float] = None) -> OrderEvent:
        """
        处理交易信号并创建订单
        
        Args:
            signal: 交易信号
            
        Returns:
            创建的订单事件
        """
        order = OrderEvent(
            instrument=signal.instrument,
            order_type="MARKET",  # 默认为市价单
            direction="BUY" if signal.signal_type == "LONG" else "SELL",
            quantity=abs(signal.strength),
            timestamp=signal.timestamp,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            price=price
        )
        
        # 生成订单ID
        order.order_id = f"{order.instrument}_{order.timestamp.strftime('%Y%m%d_%H%M%S')}"
        self.orders[order.order_id] = order
        
        # 执行订单
        try:
            await self.execute_order(order)
        except Exception as e:
            logger.error(f"订单执行失败: {str(e)}")
            order.status = "REJECTED"
            
        return order
    
    @abstractmethod
    async def execute_order(self, order: OrderEvent) -> None:
        """
        执行订单
        
        Args:
            order: 要执行的订单
        """
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """
        取消订单
        
        Args:
            order_id: 要取消的订单ID
            
        Returns:
            是否成功取消
        """
        pass
    
    def add_order_callback(self, callback):
        """添加订单状态更新回调"""
        self._order_callbacks.append(callback)
        
    def add_fill_callback(self, callback):
        """添加成交更新回调"""
        self._fill_callbacks.append(callback)
        
    async def _notify_order(self, order: OrderEvent):
        """通知订单状态更新"""
        for callback in self._order_callbacks:
            await callback(order)
            
    async def _notify_fill(self, fill: FillEvent):
        """通知成交更新"""
        for callback in self._fill_callbacks:
            await callback(fill)
            
    def get_position(self, instrument: str) -> Optional[Position]:
        """获取某个品种的持仓"""
        return self.positions.get(instrument)
        
    def get_all_positions(self) -> List[Position]:
        """获取所有持仓"""
        return list(self.positions.values())
        
    def update_position(self, fill: FillEvent) -> None:
        """根据成交更新持仓"""
        instrument = fill.instrument
        position = self.positions.get(instrument)
        
        if position is None:
            # 新建仓位
            position = Position(
                instrument=instrument,
                direction=fill.direction,
                size=fill.quantity,
                entry_price=fill.price,
                entry_time=fill.timestamp
            )
            self.positions[instrument] = position
        else:
            # 更新现有仓位
            if fill.direction == position.direction:
                # 同向加仓
                new_size = position.size + fill.quantity
                position.entry_price = (position.entry_price * position.size + 
                                     fill.price * fill.quantity) / new_size
                position.size = new_size
            else:
                # 反向减仓
                position.size -= fill.quantity
                if position.size <= 0:
                    # 清仓
                    del self.positions[instrument]
