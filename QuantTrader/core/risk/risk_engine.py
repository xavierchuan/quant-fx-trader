"""Lightweight risk engine enforcing exposure, leverage, and loss caps."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass
class RiskLimits:
    max_position_notional: float
    max_gross_leverage: float
    max_daily_loss: float
    max_drawdown: float


@dataclass
class RiskState:
    equity: float = 0.0
    peak_equity: float = 0.0
    min_equity: float = float("inf")
    realized_pnl: float = 0.0
    gross_notional: float = 0.0
    exposures: Dict[str, float] = field(default_factory=dict)


class RiskViolation(Exception):
    """Raised when orders violate limits."""


class RiskEngine:
    def __init__(self, limits: RiskLimits, starting_equity: float):
        self.limits = limits
        self.state = RiskState(equity=starting_equity, peak_equity=starting_equity, min_equity=starting_equity)

    def evaluate_order(self, symbol: str, side: str, notional: float) -> Tuple[bool, str]:
        exposure = self.state.exposures.get(symbol, 0.0)
        proposed = exposure + (notional if side.lower() == "buy" else -notional)
        if abs(proposed) > self.limits.max_position_notional:
            return False, f"symbol_exposure_limit:{symbol}"

        gross = self.state.gross_notional + abs(notional)
        leverage = gross / self.state.equity if self.state.equity else float("inf")
        if leverage > self.limits.max_gross_leverage:
            return False, "gross_leverage_limit"
        return True, "ok"

    def record_fill(self, symbol: str, side: str, notional: float, pnl: float) -> None:
        delta = notional if side.lower() == "buy" else -notional
        self.state.exposures[symbol] = self.state.exposures.get(symbol, 0.0) + delta
        self.state.gross_notional = sum(abs(v) for v in self.state.exposures.values())

        self.state.realized_pnl += pnl
        self.state.equity += pnl
        self.state.peak_equity = max(self.state.peak_equity, self.state.equity)
        self.state.min_equity = min(self.state.min_equity, self.state.equity)

    def check_loss_limits(self) -> Tuple[bool, str]:
        if -self.state.realized_pnl > self.limits.max_daily_loss:
            return False, "daily_loss_limit"
        drawdown = (self.state.equity - self.state.peak_equity) / self.state.peak_equity if self.state.peak_equity else 0.0
        if drawdown < -self.limits.max_drawdown:
            return False, "drawdown_limit"
        return True, "ok"

    def max_drawdown_pct(self) -> float:
        if not self.state.peak_equity:
            return 0.0
        trough = self.state.min_equity if self.state.min_equity != float("inf") else self.state.equity
        return abs((trough - self.state.peak_equity) / self.state.peak_equity)
