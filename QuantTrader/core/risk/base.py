from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd

from ..strategy.base import Position, SignalEvent

class RiskEvent:
    """风险事件"""
    
    def __init__(
        self,
        event_type: str,  # "RISK_LIMIT", "STOP_LOSS", "MARGIN_CALL" etc.
        instrument: str,
        timestamp: datetime,
        message: str,
        severity: str = "WARNING",  # "INFO", "WARNING", "CRITICAL"
        data: Optional[Dict[str, Any]] = None
    ):
        self.event_type = event_type
        self.instrument = instrument
        self.timestamp = timestamp
        self.message = message
        self.severity = severity
        self.data = data or {}

class PositionSizer(ABC):
    """
    仓位管理器基类
    负责计算每笔交易的具体仓位大小
    """
    
    @abstractmethod
    def calculate_position_size(
        self,
        signal: SignalEvent,
        portfolio_value: float,
        risk_per_trade: float
    ) -> float:
        """
        计算交易仓位大小
        
        Args:
            signal: 交易信号
            portfolio_value: 当前组合总价值
            risk_per_trade: 每笔交易的风险比例
            
        Returns:
            建议的仓位大小
        """
        pass

class RiskManager(ABC):
    """
    风险管理器基类
    负责风险控制和监控
    """
    
    def __init__(
        self,
        max_position_size: float,
        max_portfolio_risk: float,
        max_drawdown: float
    ):
        self.max_position_size = max_position_size
        self.max_portfolio_risk = max_portfolio_risk
        self.max_drawdown = max_drawdown
        self.current_drawdown = 0.0
        self.peak_value = 0.0
        
    @abstractmethod
    async def check_signal(self, signal: SignalEvent) -> bool:
        """
        检查交易信号是否符合风险控制要求
        
        Args:
            signal: 交易信号
            
        Returns:
            True if signal is acceptable, False otherwise
        """
        pass
        
    @abstractmethod
    async def check_position(self, position: Position) -> List[RiskEvent]:
        """
        检查持仓的风险状况
        
        Args:
            position: 当前持仓
            
        Returns:
            风险事件列表
        """
        pass
        
    def update_drawdown(self, portfolio_value: float) -> Optional[RiskEvent]:
        """
        更新和检查回撤状况
        
        Args:
            portfolio_value: 当前组合价值
            
        Returns:
            如果超过最大回撤限制，返回风险事件
        """
        if portfolio_value > self.peak_value:
            self.peak_value = portfolio_value
            self.current_drawdown = 0.0
        else:
            self.current_drawdown = (self.peak_value - portfolio_value) / self.peak_value
            
        if self.current_drawdown > self.max_drawdown:
            return RiskEvent(
                event_type="MAX_DRAWDOWN_BREACH",
                instrument="PORTFOLIO",
                timestamp=datetime.now(),
                message=f"Maximum drawdown breached: {self.current_drawdown:.2%}",
                severity="CRITICAL",
                data={"drawdown": self.current_drawdown}
            )
        return None

class SimpleRiskManager(RiskManager):
    """
    简单风险管理器实现
    实现基本的风险控制功能
    """
    
    async def check_signal(self, signal: SignalEvent) -> bool:
        """检查交易信号"""
        # 实现基本的信号检查逻辑
        if not signal.stop_loss:
            return False  # 要求必须有止损
        return True
        
    async def check_position(self, position: Position) -> List[RiskEvent]:
        """检查持仓风险"""
        events = []
        
        # 检查持仓规模
        if abs(position.size) > self.max_position_size:
            events.append(RiskEvent(
                event_type="POSITION_SIZE_LIMIT",
                instrument=position.instrument,
                timestamp=datetime.now(),
                message=f"Position size {position.size} exceeds limit {self.max_position_size}",
                severity="WARNING"
            ))
            
        # 检查止损
        if not position.stop_loss:
            events.append(RiskEvent(
                event_type="MISSING_STOP_LOSS",
                instrument=position.instrument,
                timestamp=datetime.now(),
                message="Position has no stop loss",
                severity="WARNING"
            ))
            
        return events