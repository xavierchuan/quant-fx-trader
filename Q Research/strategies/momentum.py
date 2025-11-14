"""Basic momentum breakout strategy usable inside StrategyEngine combos."""

from __future__ import annotations

from typing import Dict, Any, Sequence

from . import register
from .base import Strategy


@register("momentum_breakout")
class MomentumBreakout(Strategy):
    """
    Uses rate-of-change over a configurable lookback to enter in the dominant direction.

    Parameters
    ----------
    lookback : int
        Number of bars between comparisons.
    enter_threshold : float
        Minimum absolute return (%) to trigger an entry (e.g. 0.002 = 20 bps).
    exit_threshold : float
        Momentum magnitude below which existing positions are flattened.
    size_mult : float
        Multiplier applied to StrategyEngine default_qty when sizing trades.
    allow_short : bool
        Whether to take short trades when momentum turns negative.
    cooldown_bars : int
        Minimum bars between successive entries.
    """

    def __init__(
        self,
        lookback: int = 24,
        enter_threshold: float = 0.0015,
        exit_threshold: float = 0.0005,
        size_mult: float = 1.0,
        allow_short: bool = True,
        cooldown_bars: int = 0,
    ) -> None:
        super().__init__(
            lookback=lookback,
            enter_threshold=enter_threshold,
            exit_threshold=exit_threshold,
            size_mult=size_mult,
            allow_short=allow_short,
            cooldown_bars=cooldown_bars,
        )
        self.lookback = max(1, int(lookback))
        self.enter_threshold = float(enter_threshold)
        self.exit_threshold = float(exit_threshold)
        self.size_mult = float(size_mult)
        self.allow_short = bool(allow_short)
        self.cooldown_bars = int(max(0, cooldown_bars))

        self._block_until: int = 0

    def on_bar(self, state: Dict[str, Any]) -> Dict[str, Any]:
        closes = state.get("close_history")
        bar_idx = int(state.get("bar_idx", 0) or 0)
        position = float(state.get("position_units", 0.0) or 0.0)

        if not self._has_enough_history(closes):
            return {"action": "HOLD"}

        roc = self._rate_of_change(closes)

        if position > 0 and roc < self.exit_threshold:
            return {"action": "EXIT_LONG"}
        if position < 0 and roc > -self.exit_threshold:
            return {"action": "EXIT_SHORT"}

        if bar_idx < self._block_until:
            return {"action": "HOLD"}

        size = self._position_size(state)
        if roc >= self.enter_threshold:
            self._block_until = bar_idx + self.cooldown_bars
            return {"action": "ENTER_LONG", "size": size}
        if roc <= -self.enter_threshold and self.allow_short:
            self._block_until = bar_idx + self.cooldown_bars
            return {"action": "ENTER_SHORT", "size": size}

        return {"action": "HOLD"}

    def _has_enough_history(self, closes: Any) -> bool:
        if closes is None:
            return False
        if isinstance(closes, Sequence):
            return len(closes) > self.lookback
        return False

    def _rate_of_change(self, closes: Sequence[float]) -> float:
        recent = float(closes[-1])
        past = float(closes[-(self.lookback + 1)])
        if past == 0:
            return 0.0
        return (recent - past) / past

    def _position_size(self, state: Dict[str, Any]) -> float | None:
        default_qty = state.get("default_qty")
        if default_qty is None:
            return None
        return float(default_qty) * self.size_mult
