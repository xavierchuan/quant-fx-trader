from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd

class DataFeed(ABC):
    """
    数据源的基础抽象类
    定义了获取市场数据的标准接口
    """
    
    def __init__(self, instrument: str, timeframe: str):
        """
        初始化数据源
        
        Args:
            instrument: 交易品种 (例如: "EUR_USD")
            timeframe: 时间周期 (例如: "H1", "D")
        """
        self.instrument = instrument
        self.timeframe = timeframe
        
    @abstractmethod
    async def get_historical_data(
        self, 
        start: datetime, 
        end: datetime,
        **kwargs
    ) -> pd.DataFrame:
        """
        获取历史数据
        
        Args:
            start: 开始时间
            end: 结束时间
            **kwargs: 额外参数
            
        Returns:
            包含历史数据的DataFrame，至少应该包含以下列：
            - datetime: 时间戳
            - open: 开盘价
            - high: 最高价
            - low: 最低价
            - close: 收盘价
            - volume: 成交量（如果可用）
        """
        pass
    
    @abstractmethod
    async def get_latest_data(self) -> Dict[str, Any]:
        """
        获取最新的市场数据
        
        Returns:
            包含最新市场数据的字典
        """
        pass
    
    @abstractmethod
    async def subscribe(self, callback) -> None:
        """
        订阅实时数据更新
        
        Args:
            callback: 处理实时数据的回调函数
        """
        pass
    
    @abstractmethod
    async def unsubscribe(self) -> None:
        """
        取消订阅实时数据
        """
        pass

class DataProcessor:
    """
    数据处理器基类
    用于对原始市场数据进行预处理和计算指标
    """
    
    def __init__(self):
        self.indicators = {}
        
    def add_indicator(self, name: str, func, **params):
        """
        添加技术指标计算
        
        Args:
            name: 指标名称
            func: 计算指标的函数
            **params: 指标参数
        """
        self.indicators[name] = {
            'function': func,
            'params': params
        }
        
    def process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        处理数据并计算所有已注册的指标
        
        Args:
            data: 原始市场数据
            
        Returns:
            添加了技术指标的DataFrame
        """
        result = data.copy()
        for name, indicator in self.indicators.items():
            try:
                result[name] = indicator['function'](
                    data, 
                    **indicator['params']
                )
            except Exception as e:
                print(f"计算指标 {name} 时发生错误: {str(e)}")
        return result

class MarketDataEvent:
    """
    市场数据事件类
    用于在系统各层之间传递市场数据更新
    """
    
    def __init__(
        self,
        instrument: str,
        timestamp: datetime,
        data: Dict[str, Any],
        event_type: str = "MARKET_DATA"
    ):
        self.event_type = event_type
        self.instrument = instrument
        self.timestamp = timestamp
        self.data = data