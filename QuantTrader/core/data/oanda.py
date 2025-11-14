from __future__ import annotations

import asyncio
import time
from datetime import datetime
from typing import Dict, Any, Optional, Sequence
import pandas as pd
from queue import Queue
import threading
from loguru import logger

from oandapyV20 import API
try:
    from oandapyV20.endpoints.pricing import PricingStream as OandaPricingStream
except ImportError:  # pragma: no cover
    OandaPricingStream = None  # type: ignore

from .base import DataFeed, MarketDataEvent

def _normalize_instrument(symbol: str) -> str:
    """规范化交易品种名称"""
    s = symbol.upper().replace(" ", "").replace("/", "_").replace("-", "_")
    if "_" in s and len(s) == 7:
        return s
    stripped = s.replace("_", "")
    if len(stripped) == 6:
        return f"{stripped[:3]}_{stripped[3:]}"
    return s

class OANDADataFeed(DataFeed):
    """
    OANDA数据源实现
    提供实时和历史市场数据
    """
    
    def __init__(
        self,
        instrument: str,
        timeframe: str,
        account_id: str,
        access_token: str,
        environment: str = "practice",
        reconnect_wait: float = 5.0,
        log_heartbeat: bool = False
    ):
        super().__init__(instrument, timeframe)
        self.account_id = account_id
        self.access_token = access_token
        self.environment = environment
        self.reconnect_wait = reconnect_wait
        self.log_heartbeat = log_heartbeat
        
        self.client = API(access_token=access_token, environment=environment)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._callback = None
        self._latest_data = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        
    async def get_historical_data(
        self, 
        start: datetime, 
        end: datetime,
        **kwargs
    ) -> pd.DataFrame:
        """获取历史数据
        
        Args:
            start: 开始时间
            end: 结束时间
            **kwargs: 额外参数，支持：
                - granularity: str, 时间周期 (如 "H1", "D")
                - count: int, 返回的K线数量
                
        Returns:
            DataFrame包含以下列：datetime, open, high, low, close, volume
        """
        from oandapyV20 import API
        import oandapyV20.endpoints.instruments as instruments

        granularity = kwargs.get("granularity", self.timeframe)
        price = kwargs.get("price", "M")

        # OANDA 的单次请求有最大返回数限制（例如 5000 candles），对长区间需要分段请求
        MAX_CANDLES = 5000

        # 估算每个candle的时间长度（秒），支持常见的granularity
        def granularity_seconds(g: str) -> int:
            if g.endswith('H'):
                return int(g[:-1]) * 3600
            if g.endswith('D'):
                return int(g[:-1]) * 86400 if g[:-1].isdigit() else 86400
            if g.endswith('M') and len(g) > 1 and g[0].isdigit():
                # 例如 M1, M5 (分钟)
                return int(g[1:]) * 60 if g[0] == 'M' else 30
            # 默认按小时处理
            return 3600

        step_seconds = granularity_seconds(granularity) * MAX_CANDLES

        all_data = []
        current_start = pd.to_datetime(start)
        end_ts = pd.to_datetime(end)

        while current_start < end_ts:
            current_end = current_start + pd.Timedelta(seconds=step_seconds)
            if current_end > end_ts:
                current_end = end_ts

            params = {
                "from": current_start.strftime("%Y-%m-%dT%H:%M:%S.000000Z"),
                "to": current_end.strftime("%Y-%m-%dT%H:%M:%S.000000Z"),
                "granularity": granularity,
                "price": price
            }

            request = instruments.InstrumentsCandles(
                instrument=self.instrument,
                params=params
            )

            try:
                response = self.client.request(request)
                candles = response.get("candles", [])

                for candle in candles:
                    if candle.get("complete"):
                        all_data.append({
                            "datetime": pd.to_datetime(candle["time"]),
                            "open": float(candle["mid"]["o"]),
                            "high": float(candle["mid"]["h"]),
                            "low": float(candle["mid"]["l"]),
                            "close": float(candle["mid"]["c"]),
                            "volume": int(candle.get("volume", 0))
                        })

            except Exception as e:
                logger.error(f"获取历史数据失败: {str(e)}")
                raise

            # 推进起点
            current_start = current_end

        if not all_data:
            return pd.DataFrame()

        df = pd.DataFrame(all_data)
        df.drop_duplicates(subset=["datetime"], inplace=True)
        df.sort_values(by="datetime", inplace=True)
        df.set_index("datetime", inplace=True)
        return df
        
    async def get_latest_data(self) -> Dict[str, Any]:
        """获取最新数据"""
        return self._latest_data if self._latest_data else {}
        
    async def subscribe(self, callback) -> None:
        """订阅实时数据"""
        self._callback = callback
        self._loop = asyncio.get_running_loop()
        if not self._thread or not self._thread.is_alive():
            self._start_stream()
            
    async def unsubscribe(self) -> None:
        """取消订阅"""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        self._loop = None
        logger.info("[OANDA] Pricing stream stopped.")
        
    def _start_stream(self) -> None:
        """启动价格流"""
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run_stream, daemon=True)
        self._thread.start()
        logger.info(
            f"[OANDA] Pricing stream started for {self.instrument} (account={self.account_id})"
        )
        
    def _run_stream(self) -> None:
        """运行价格流"""
        if OandaPricingStream is None:
            raise RuntimeError(
                "oandapyV20.endpoints.pricing.PricingStream is unavailable. "
                "Ensure oandapyV20 is installed."
            )
        params = {"instruments": self.instrument}
        while not self._stop.is_set():
            request = OandaPricingStream(
                accountID=self.account_id, 
                params=params
            )
            try:
                for msg in self.client.request(request):
                    if self._stop.is_set():
                        break
                    self._handle_msg(msg)
            except Exception as exc:
                if self._stop.is_set():
                    break
                logger.warning(
                    f"[OANDA] Pricing stream error: {exc}. "
                    f"Reconnecting in {self.reconnect_wait}s"
                )
                time.sleep(self.reconnect_wait)
                
    def _handle_msg(self, msg: dict) -> None:
        """处理价格消息"""
        msg_type = msg.get("type")
        if msg_type == "HEARTBEAT":
            if self.log_heartbeat:
                logger.debug(f"[OANDA] Heartbeat {msg.get('time')}")
            return
            
        if msg_type != "PRICE":
            logger.debug(f"[OANDA] Skip message type={msg_type}")
            return
            
        try:
            bids = msg.get("bids")
            asks = msg.get("asks")
            if not bids or not asks:
                return
                
            bid = float(bids[0]["price"])
            ask = float(asks[0]["price"])
            timestamp = pd.to_datetime(msg["time"]).to_pydatetime()
            
            self._latest_data = {
                "bid": bid,
                "ask": ask,
                "timestamp": timestamp,
                "mid": (bid + ask) / 2
            }
            
            if self._callback and self._loop:
                event = MarketDataEvent(
                    instrument=self.instrument,
                    timestamp=timestamp,
                    data=self._latest_data
                )
                try:
                    coro = self._callback(event)
                    if asyncio.iscoroutine(coro):
                        asyncio.run_coroutine_threadsafe(coro, self._loop)
                    else:
                        self._loop.call_soon_threadsafe(self._callback, event)
                except RuntimeError as exc:
                    logger.warning(f"[OANDA] Failed to dispatch callback: {exc}")
            elif self._callback and not self._loop:
                logger.warning("[OANDA] Callback set but event loop missing; dropping tick")
                
        except Exception as exc:
            logger.warning(f"[OANDA] Malformed price message: {msg} ({exc})")
