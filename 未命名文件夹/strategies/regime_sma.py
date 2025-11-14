# strategies/regime_sma.py

from __future__ import annotations

from datetime import datetime

from . import register
from .base import Strategy
from .sma_atr import SmaAtr


@register("regime_sma")
class RegimeSMAStrategy(Strategy):
    """
    Wraps the SMA+ATR strategy with a simple regime filter.

    - 在趋势 regime（由 StrategyEngine 提供）下，沿用 SmaAtr 信号。
    - 在震荡 regime 下，可选择保持空仓或使用简单的 RSI 均值回归。
    """

    def __init__(
        self,
        trend_params: dict | None = None,
        range_mode: str = "flat",
        range_rsi_high: float = 75.0,
        range_rsi_low: float = 25.0,
        trend_min_bars: int = 0,
        atr_percentile_min: float | None = None,
        size_tiers: list | None = None,
        base_size_mult: float = 1.0,
        htf_alignment: bool = False,
        htf_rsi_range: tuple[float, float] | list | None = None,
        risk_rules: list | None = None,
    ) -> None:
        super().__init__()
        self.range_mode = (range_mode or "flat").lower()
        self.range_rsi_high = float(range_rsi_high)
        self.range_rsi_low = float(range_rsi_low)
        self.range_exit_mid = (self.range_rsi_high + self.range_rsi_low) / 2.0
        self.trend_min_bars = int(trend_min_bars or 0)
        self.atr_percentile_min = atr_percentile_min if atr_percentile_min is None else float(atr_percentile_min)
        self.base_size_mult = float(base_size_mult or 1.0)
        self.size_tiers = self._normalize_tiers(size_tiers)
        self.htf_alignment = bool(htf_alignment)
        if htf_rsi_range:
            lo, hi = htf_rsi_range
            self.htf_rsi_min = float(lo)
            self.htf_rsi_max = float(hi)
        else:
            self.htf_rsi_min = 30.0
            self.htf_rsi_max = 70.0
        self.risk_rules = risk_rules or []

        trend_params = trend_params or {}
        self.trend_strategy = SmaAtr(**trend_params)

    def on_bar(self, state: dict):
        cooldown = self._check_risk_rules(state)
        if cooldown:
            return {"action": "HOLD", "cooldown_bars": cooldown}

        regime_label = state.get("regime_label", "unknown")
        trend_streak = int(state.get("regime_trend_bars", 0) or 0)
        atr_percentile = state.get("atr_percentile")
        base_signal = self.trend_strategy.on_bar(state) or {}
        action = base_signal.get("action", "HOLD")

        if regime_label == "trend":
            if self.trend_min_bars and trend_streak < self.trend_min_bars:
                regime_label = "range"
            elif self.atr_percentile_min is not None:
                if atr_percentile is None or atr_percentile < self.atr_percentile_min:
                    regime_label = "range"
            if regime_label == "trend" and action.startswith("ENTER"):
                if not self._passes_htf_filter(action, state):
                    regime_label = "range"
                else:
                    sized = dict(base_signal)
                    sized["size"] = self._position_size(action, state)
                    return sized
            elif regime_label == "trend":
                return base_signal

        # 在趋势 regime，直接沿用趋势策略的信号
        if regime_label == "trend":
            return base_signal

        # 非趋势 regime：允许趋势策略发出的平仓指令生效，但屏蔽入场
        if action.startswith("EXIT"):
            return base_signal

        # 根据 range_mode 决定行为
        range_action = self._range_signal(state)
        if range_action:
            return {"action": range_action}

        # 默认保持空仓
        if state.get("position"):
            # 持仓状态下交给风控（risk exit）或趋势策略的平仓指令处理
            return {"action": "HOLD"}
        return {"action": "HOLD"}

    def _range_signal(self, state: dict) -> str | None:
        if self.range_mode != "mean_revert":
            return None
        rsi = state.get("rsi")
        if rsi is None:
            return None

        position = state.get("position", 0)
        if position == 0:
            if rsi >= self.range_rsi_high:
                return "ENTER_SHORT"
            if rsi <= self.range_rsi_low:
                return "ENTER_LONG"
        elif position > 0 and rsi >= self.range_exit_mid:
            return "EXIT_LONG"
        elif position < 0 and rsi <= self.range_exit_mid:
            return "EXIT_SHORT"
        return None

    def _passes_htf_filter(self, action: str, state: dict) -> bool:
        if not self.htf_alignment:
            return True
        htf_ema = state.get("htf_ema")
        if htf_ema is None:
            return False
        close = state.get("close")
        if close is None:
            return False
        if action == "ENTER_LONG" and close < htf_ema:
            return False
        if action == "ENTER_SHORT" and close > htf_ema:
            return False
        htf_rsi = state.get("htf_rsi")
        if htf_rsi is not None:
            if htf_rsi < self.htf_rsi_min or htf_rsi > self.htf_rsi_max:
                return False
        return True

    def _normalize_tiers(self, tiers: list | None) -> list:
        if not tiers:
            return [{"size_mult": self.base_size_mult}]
        normalized = []
        for tier in tiers:
            if not isinstance(tier, dict):
                continue
            entry = tier.copy()
            entry["size_mult"] = float(entry.get("size_mult", 1.0))
            normalized.append(entry)
        if not normalized:
            normalized.append({"size_mult": self.base_size_mult})
        normalized.sort(key=lambda t: t.get("size_mult", 0), reverse=True)
        return normalized

    def _position_size(self, action: str, state: dict) -> float:
        base_qty = float(state.get("default_qty") or 0.0)
        if base_qty <= 0:
            return 0.0
        atr_pct = state.get("atr_percentile")
        trend_strength = state.get("trend_strength")
        streak = int(state.get("regime_trend_bars", 0) or 0)
        for tier in self.size_tiers:
            if self._tier_matches(tier, atr_pct, trend_strength, streak, action):
                return base_qty * tier.get("size_mult", 1.0)
        return base_qty * self.base_size_mult

    def _tier_matches(self, tier: dict, atr_pct, trend_strength, streak: int, action: str) -> bool:
        min_atr = tier.get("min_atr_pct")
        max_atr = tier.get("max_atr_pct")
        min_strength = tier.get("min_trend_strength")
        min_streak = tier.get("min_trend_bars")
        allow_short = tier.get("allow_short")
        allow_long = tier.get("allow_long")
        if min_atr is not None:
            if atr_pct is None or atr_pct < float(min_atr):
                return False
        if max_atr is not None and atr_pct is not None:
            if atr_pct > float(max_atr):
                return False
        if min_strength is not None:
            if trend_strength is None or trend_strength < float(min_strength):
                return False
        if min_streak is not None and streak < int(min_streak):
            return False
        if action == "ENTER_LONG" and allow_long is False:
            return False
        if action == "ENTER_SHORT" and allow_short is False:
            return False
        return True

    def _check_risk_rules(self, state: dict) -> int:
        if not self.risk_rules:
            return 0
        atr_pct = state.get("atr_percentile")
        ts = self._to_datetime(state.get("ts"))
        for rule in self.risk_rules:
            rtype = rule.get("type")
            if rtype == "atr_percentile":
                min_v = rule.get("min")
                max_v = rule.get("max")
                triggered = False
                if min_v is not None:
                    if atr_pct is None or atr_pct < float(min_v):
                        triggered = True
                if max_v is not None and atr_pct is not None and atr_pct > float(max_v):
                    triggered = True
                if triggered:
                    return int(rule.get("cooldown_bars", 0) or 0)
            elif rtype == "calendar":
                if ts is None:
                    continue
                dates = rule.get("dates") or []
                date_str = ts.strftime("%Y-%m-%d")
                if date_str in dates:
                    return int(rule.get("cooldown_bars", 0) or 0)
                windows = rule.get("windows") or []
                for window in windows:
                    start = self._to_datetime(window.get("start"))
                    end = self._to_datetime(window.get("end"))
                    if start and end and start <= ts <= end:
                        return int(rule.get("cooldown_bars", 0) or 0)
            elif rtype == "time_window":
                start = self._to_datetime(rule.get("start"))
                end = self._to_datetime(rule.get("end"))
                if ts and start and end and start <= ts <= end:
                    return int(rule.get("cooldown_bars", 0) or 0)
        return 0

    def _to_datetime(self, value):
        if value is None:
            return None
        if hasattr(value, "to_pydatetime"):
            try:
                return value.to_pydatetime()
            except Exception:
                pass
        if isinstance(value, datetime):
            return value
        text = str(value)
        if not text:
            return None
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00"))
        except Exception:
            return None
