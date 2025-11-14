"""Simple moving-average crossover strategy registered for StrategyEngine combos."""

from __future__ import annotations

from typing import Dict, Any

from . import register
from .base import Strategy


@register("ma_crossover")
class MovingAverageCrossover(Strategy):
    """
    Emits ENTER/EXIT signals when a fast SMA crosses a slow SMA.

    State inputs expected from StrategyEngine:
      - sma_fast / sma_slow
      - bar_idx (int)
      - position_units (float)
      - default_qty (float)
    """

    def __init__(
        self,
        size_mult: float = 1.0,
        cooldown_bars: int = 0,
        exit_buffer_pct: float = 0.0,
        allow_short: bool = True,
    ) -> None:
        super().__init__(
            size_mult=size_mult,
            cooldown_bars=cooldown_bars,
            exit_buffer_pct=exit_buffer_pct,
            allow_short=allow_short,
        )
        self.size_mult = float(size_mult)
        self.cooldown_bars = int(max(0, cooldown_bars))
        self.exit_buffer_pct = float(max(0.0, exit_buffer_pct))
        self.allow_short = bool(allow_short)

        self._prev_fast: float | None = None
        self._prev_slow: float | None = None
        self._block_until: int = 0

    def on_bar(self, state: Dict[str, Any]) -> Dict[str, Any]:
        fast = state.get("sma_fast")
        slow = state.get("sma_slow")
        bar_idx = int(state.get("bar_idx", 0) or 0)
        position = float(state.get("position_units", 0.0) or 0.0)

        if fast is None or slow is None:
            return {"action": "HOLD"}

        prev_fast = self._prev_fast
        prev_slow = self._prev_slow
        self._prev_fast = fast
        self._prev_slow = slow

        if prev_fast is None or prev_slow is None:
            return {"action": "HOLD"}

        if bar_idx < self._block_until:
            return {"action": "HOLD"}

        buffer = self.exit_buffer_pct
        size = self._position_size(state)

        crossed_up = prev_fast <= prev_slow and fast > slow
        crossed_down = prev_fast >= prev_slow and fast < slow

        if crossed_up:
            self._block_until = bar_idx + self.cooldown_bars
            return {"action": "ENTER_LONG", "size": size}

        if crossed_down and self.allow_short:
            self._block_until = bar_idx + self.cooldown_bars
            return {"action": "ENTER_SHORT", "size": size}

        slow_with_buffer = slow * (1.0 + buffer)
        slow_lower = slow * (1.0 - buffer)

        if position > 0 and fast < slow_lower:
            return {"action": "EXIT_LONG"}

        if position < 0 and (fast > slow_with_buffer or not self.allow_short):
            return {"action": "EXIT_SHORT"}

        return {"action": "HOLD"}

    def _position_size(self, state: Dict[str, Any]) -> float | None:
        default_qty = state.get("default_qty")
        if default_qty is None:
            return None
        return float(default_qty) * self.size_mult
