"""
OANDA 定价流适配器，用于从模拟/实盘账户拉取实时报价并投递 TickEvent。
"""

from __future__ import annotations

import threading
import time
from queue import Queue
from typing import Iterable, List, Optional, Sequence

import pandas as pd
from loguru import logger
from oandapyV20 import API
try:
    from oandapyV20.endpoints.pricing import PricingStream
except ImportError:
    PricingStream = None

from core.events import TickEvent


def _normalize_instrument(symbol: str) -> str:
    s = symbol.upper().replace(" ", "").replace("/", "_").replace("-", "_")
    if "_" in s and len(s) == 7:
        return s
    stripped = s.replace("_", "")
    if len(stripped) == 6:
        return f"{stripped[:3]}_{stripped[3:]}"
    return s


class OandaPricingStream:
    """
    使用 OANDA Pricing Stream API 推送 TickEvent。
    """

    def __init__(
        self,
        q: Queue,
        account_id: str,
        instruments: Sequence[str],
        access_token: str,
        environment: str = "practice",
        reconnect_wait: float = 5.0,
        log_heartbeat: bool = False,
    ) -> None:
        self.q = q
        self.account_id = account_id
        self.instruments = [_normalize_instrument(sym) for sym in instruments]
        self.log_heartbeat = log_heartbeat
        self.client = API(access_token=access_token, environment=environment)
        self.reconnect_wait = reconnect_wait
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def start(self) -> None:
        if PricingStream is None:
            raise RuntimeError(
                "oandapyV20 未暴露 PricingStream（需 0.7.2+）。请执行 `pip install --upgrade oandapyV20` 后重试。"
            )
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info(
            f"[OANDA] Pricing stream started for {','.join(self.instruments)} (account={self.account_id})"
        )

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        logger.info("[OANDA] Pricing stream stopped.")

    def _run(self) -> None:
        if PricingStream is None:
            logger.error("无法启动价格流：缺少 PricingStream 类")
            return
        params = {"instruments": ",".join(self.instruments)}
        while not self._stop.is_set():
            request = PricingStream(accountID=self.account_id, params=params)
            try:
                for msg in self.client.request(request):
                    if self._stop.is_set():
                        break
                    self._handle_msg(msg)
            except Exception as exc:
                if self._stop.is_set():
                    break
                logger.warning(f"[OANDA] Pricing stream error: {exc}. Reconnecting in {self.reconnect_wait}s")
                time.sleep(self.reconnect_wait)

    def _handle_msg(self, msg: dict) -> None:
        msg_type = msg.get("type")
        if msg_type == "HEARTBEAT":
            if self.log_heartbeat:
                logger.debug(f"[OANDA] Heartbeat {msg.get('time')}")
            return
        if msg_type != "PRICE":
            logger.debug(f"[OANDA] Skip message type={msg_type}")
            return
        try:
            symbol = msg["instrument"].replace("_", "")
            bids = msg.get("bids")
            asks = msg.get("asks")
            if not bids or not asks:
                return
            bid = float(bids[0]["price"])
            ask = float(asks[0]["price"])
            ts = pd.Timestamp(msg["time"]).floor("us").to_pydatetime()
        except Exception as exc:
            logger.warning(f"[OANDA] Malformed price message: {msg} ({exc})")
            return
        self.q.put(TickEvent(ts=ts, symbol=symbol, bid=bid, ask=ask))
