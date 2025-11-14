from __future__ import annotations

import numpy as np

from . import register
from .base import Strategy


@register("bollinger_mean_revert")
class BollingerMeanRevert(Strategy):
    """
    Simple Bollinger-band based mean reversion strategy skeleton.
    - Enters long when price falls `enter_z` standard deviations below the mean.
    - Enters short symmetrically (if allow_short).
    - Exits when price reverts back within `exit_z` standard deviations.
    This is meant to be combined with other sleeves in portfolio tests.
    """

    def __init__(
        self,
        window: int = 50,
        num_std: float = 2.0,
        enter_z: float = 1.0,
        exit_z: float = 0.2,
        allow_short: bool = True,
        cooldown: int = 0,
    ) -> None:
        super().__init__(
            window=window,
            num_std=num_std,
            enter_z=enter_z,
            exit_z=exit_z,
            allow_short=allow_short,
            cooldown=cooldown,
        )
        self.window = int(window)
        self.num_std = float(num_std)
        self.enter_z = float(enter_z)
        self.exit_z = float(exit_z)
        self.allow_short = bool(allow_short)
        self.cooldown = int(cooldown or 0)
        self._next_entry_bar = 0

    def on_bar(self, state: dict) -> dict:
        closes = state.get("close_history")
        bar_idx = state.get("bar_idx", 0)
        position = state.get("position", 0)
        if closes is None or len(closes) < self.window:
            return {"action": "HOLD"}

        window_data = np.array(closes[-self.window:], dtype=float)
        mean = window_data.mean()
        std = window_data.std(ddof=0)
        if std == 0:
            return {"action": "HOLD"}

        price = float(window_data[-1])
        z_score = (price - mean) / std

        # Enforce cooldown between fresh entries
        if bar_idx < self._next_entry_bar and position == 0:
            return {"action": "HOLD"}

        if position == 0:
            if z_score <= -self.enter_z:
                self._next_entry_bar = bar_idx + self.cooldown
                return {"action": "ENTER_LONG"}
            if self.allow_short and z_score >= self.enter_z:
                self._next_entry_bar = bar_idx + self.cooldown
                return {"action": "ENTER_SHORT"}
        elif position > 0:
            if z_score >= -self.exit_z:
                return {"action": "EXIT_LONG"}
        else:  # position < 0
            if z_score <= self.exit_z:
                return {"action": "EXIT_SHORT"}

        return {"action": "HOLD"}
