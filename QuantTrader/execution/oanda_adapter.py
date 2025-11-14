"""OANDA REST adapter (mock/stub for Phase 3)."""

from __future__ import annotations

import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import requests

from .adapter import CancelAck, ExecutionAdapter, OrderAck, OrderParams, PositionState
from .config_loader import load_oanda_config
from .metrics_logger import MetricsLogger, log_event
from .order_store import OrderStore


class OandaAdapter(ExecutionAdapter):
    BASE_URL = "https://api-fxpractice.oanda.com/v3"

    def __init__(
        self,
        account_id: Optional[str] = None,
        token: Optional[str] = None,
        order_store: Optional[OrderStore] = None,
        base_url: Optional[str] = None,
        timeout_ms: int = 10000,
        retry_backoff: float = 1.0,
        max_retries: int = 3,
        metrics_path: Optional[str] = None,
        error_log: str = "results/execution/errors.log",
    ) -> None:
        self.account_id = account_id or os.environ.get("OANDA_ACCOUNT_ID", "")
        self.token = token or os.environ.get("OANDA_TOKEN", "")
        self.base_url = base_url or self.BASE_URL
        self.timeout_ms = timeout_ms
        self.retry_backoff = retry_backoff
        self.max_retries = max_retries
        self.metrics_logger = MetricsLogger(metrics_path)
        self.error_log = Path(error_log)
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"})
        self.order_store = order_store or OrderStore()

    def _endpoint(self, path: str) -> str:
        return f"{self.base_url}{path}"

    @classmethod
    def from_config(cls, path: Optional[str] = None, order_store: Optional[OrderStore] = None) -> "OandaAdapter":
        cfg = load_oanda_config(path)
        return cls(order_store=order_store, **cfg)

    def _log_error(self, context: str, message: str) -> None:
        self.error_log.parent.mkdir(parents=True, exist_ok=True)
        with self.error_log.open("a", encoding="utf-8") as fh:
            fh.write(f"[{datetime.utcnow().isoformat()}] {context}: {message}\n")

    def _request(self, method: str, url: str, **kwargs):
        backoff = self.retry_backoff
        for attempt in range(self.max_retries):
            try:
                resp = self.session.request(method, url, timeout=self.timeout_ms / 1000.0, **kwargs)
            except requests.RequestException as exc:
                self._log_error("network", str(exc))
                time.sleep(backoff)
                backoff *= 2
                continue
            if resp.status_code >= 500:
                self._log_error("server_error", resp.text)
                time.sleep(backoff)
                backoff *= 2
                continue
            if resp.status_code >= 400:
                self._log_error("client_error", resp.text)
                resp.raise_for_status()
            resp.raise_for_status()
            return resp.json()
        raise RuntimeError(f"Failed {method} {url} after {self.max_retries} retries")

    def submit(self, order: OrderParams) -> OrderAck:
        start = datetime.utcnow()
        payload = {
            "order": {
                "instrument": order.symbol,
                "units": int(order.quantity if order.side.lower() == "buy" else -order.quantity),
                "type": "MARKET" if order.price is None else "LIMIT",
                "timeInForce": order.tif.upper(),
            }
        }
        if order.price is not None:
            payload["order"]["price"] = f"{order.price:.5f}"
        if order.client_order_id:
            payload["order"]["clientExtensions"] = {"clientOrderID": order.client_order_id}
        url = self._endpoint(f"/accounts/{self.account_id}/orders")
        data = self._request("POST", url, json=payload)
        order_id = data.get("orderCreateTransaction", {}).get("id", "")
        if order_id:
            self.order_store.append(order_id, order)
        end = datetime.utcnow()
        log_event("submit", order_id or "", order.symbol, "accepted", start, end)
        ts = data.get("time")
        timestamp = datetime.fromisoformat(ts.replace("Z", "+00:00")) if ts else datetime.utcnow()
        return OrderAck(order_id=order_id, status="accepted", timestamp=timestamp)

    def cancel(self, order_id: str) -> CancelAck:
        start = datetime.utcnow()
        url = self._endpoint(f"/accounts/{self.account_id}/orders/{order_id}/cancel")
        data = self._request("PUT", url)
        end = datetime.utcnow()
        log_event("cancel", order_id, "", "cancelled", start, end)
        ts = data.get("time")
        timestamp = datetime.fromisoformat(ts.replace("Z", "+00:00")) if ts else datetime.utcnow()
        return CancelAck(order_id=order_id, status="cancelled", timestamp=timestamp)

    def sync_positions(self) -> Dict[str, PositionState]:
        url = self._endpoint(f"/accounts/{self.account_id}/positions")
        data = self._request("GET", url)
        positions = {}
        for pos in data.get("positions", []):
            symbol = pos["instrument"]
            net = float(pos["netUnrealizedPL"])
            qty = float(pos.get("long", {}).get("units", 0)) + float(pos.get("short", {}).get("units", 0))
            avg_price = float(pos.get("avgPrice", 0))
            positions[symbol] = PositionState(symbol=symbol, quantity=qty, avg_price=avg_price, unrealized_pnl=net)
        return positions

    def heartbeat(self) -> bool:
        url = self._endpoint("/accounts")
        try:
            self._request("GET", url)
            return True
        except Exception:
            return False
