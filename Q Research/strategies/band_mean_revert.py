from __future__ import annotations

from . import register
from .base import Strategy


@register("band_mean_revert")
class BandMeanRevert(Strategy):
    """
    简单区间/均值回归策略：
    - 使用慢均线 +/- ATR*mult 作为区间带；
    - 当价格跌破下带且 RSI 低于阈值时做多；
    - 当价格突破上带且 RSI 高于阈值时做空（可选）。
    - 价格回到均值或 RSI 归中时离场。
    """

    def __init__(
        self,
        band_atr_mult: float = 1.5,
        rsi_long: float = 35.0,
        rsi_short: float = 65.0,
        exit_rsi_mid: float = 50.0,
        allow_short: bool = True,
    ) -> None:
        super().__init__(
            band_atr_mult=band_atr_mult,
            rsi_long=rsi_long,
            rsi_short=rsi_short,
            exit_rsi_mid=exit_rsi_mid,
            allow_short=allow_short,
        )
        self.band_atr_mult = float(band_atr_mult)
        self.rsi_long = float(rsi_long)
        self.rsi_short = float(rsi_short)
        self.exit_rsi_mid = float(exit_rsi_mid)
        self.allow_short = bool(allow_short)

    def on_bar(self, state: dict) -> dict:
        close = state.get("close")
        sma_slow = state.get("sma_slow")
        atr = state.get("curr_atr")
        rsi = state.get("rsi")
        position = state.get("position", 0)

        if close is None or sma_slow is None or atr is None or rsi is None:
            return {"action": "HOLD"}

        upper = sma_slow + self.band_atr_mult * atr
        lower = sma_slow - self.band_atr_mult * atr

        if position == 0:
            if close <= lower and rsi <= self.rsi_long:
                return {"action": "ENTER_LONG"}
            if self.allow_short and close >= upper and rsi >= self.rsi_short:
                return {"action": "ENTER_SHORT"}
        elif position > 0:
            if close >= sma_slow or rsi >= self.exit_rsi_mid:
                return {"action": "EXIT_LONG"}
        elif position < 0:
            if close <= sma_slow or rsi <= self.exit_rsi_mid:
                return {"action": "EXIT_SHORT"}

        return {"action": "HOLD"}
