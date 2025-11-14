# strategies/sma_atr.py
from .base import Strategy
from . import register

@register("sma_atr")
class SmaAtr(Strategy):
    """
    参数（与 runner 对齐）：
    - long_only_above_slow: bool
    - allow_short: bool
    - short_only_below_slow: bool
    - slope_lookback: int
    - cooldown: int
    - fast_win: int
    - slow_win: int
    - atr_sl / atr_tp / atr_window（仅用于记录，不在策略层计算）
    """
    def on_bar(self, state):
        c = state["close"]
        position = state["position"]
        rsi = state.get("rsi")
        sma_fast = state.get("sma_fast")
        sma_slow = state.get("sma_slow")
        bar_idx = state["bar_idx"]
        next_entry_bar_idx_long = state.get("next_entry_bar_idx_long", 0)
        next_entry_bar_idx_short = state.get("next_entry_bar_idx_short", 0)
        sma_fast_hist = state["sma_fast_hist"]  # deque，最近值在右侧

        fast_win = self.params.get("fast_win")
        slow_win = self.params.get("slow_win")
        long_only_above_slow = self.params.get("long_only_above_slow", False)
        slope_lookback = self.params.get("slope_lookback", 0)
        cooldown = self.params.get("cooldown", 0)
        allow_short = self.params.get("allow_short", True)
        short_only_below_slow = self.params.get("short_only_below_slow", False)
        rsi_long_thresh = self.params.get("rsi_long_thresh")
        rsi_short_thresh = self.params.get("rsi_short_thresh")

        # 均线就绪才判断
        if sma_fast is None or sma_slow is None:
            return {"action": "HOLD"}

        go_long = (position == 0) and (sma_fast > sma_slow)
        exit_long = (position == 1) and (sma_fast < sma_slow)

        # 仅做多需在慢均线上方
        if long_only_above_slow and go_long:
            if not (c > sma_slow):
                go_long = False

        # fast 斜率确认
        if slope_lookback and go_long:
            if len(sma_fast_hist) > slope_lookback:
                # 最近一个值与 L 根前比较
                if not (sma_fast_hist[-1] > sma_fast_hist[-1 - slope_lookback]):
                    go_long = False
            else:
                go_long = False

        # 冷却
        if cooldown and go_long:
            if bar_idx < next_entry_bar_idx_long:
                go_long = False

        # RSI 过滤（如果配置了阈值且 RSI 可用）
        if go_long and rsi_long_thresh is not None:
            if rsi is None:
                go_long = False
            else:
                if not (rsi > float(rsi_long_thresh)):
                    go_long = False

        go_short = False
        exit_short = False
        if allow_short:
            go_short = (position == 0) and (sma_fast < sma_slow)
            exit_short = (position == -1) and (sma_fast > sma_slow)

            if short_only_below_slow and go_short:
                if not (c < sma_slow):
                    go_short = False

            if slope_lookback and go_short:
                if len(sma_fast_hist) > slope_lookback:
                    if not (sma_fast_hist[-1] < sma_fast_hist[-1 - slope_lookback]):
                        go_short = False
                else:
                    go_short = False

            if cooldown and go_short:
                if bar_idx < next_entry_bar_idx_short:
                    go_short = False

            # RSI 过滤（空头侧）
            if go_short and rsi_short_thresh is not None:
                if rsi is None:
                    go_short = False
                else:
                    if not (rsi < float(rsi_short_thresh)):
                        go_short = False

        if exit_long:
            return {"action": "EXIT_LONG"}
        if exit_short:
            return {"action": "EXIT_SHORT"}
        if go_long:
            return {"action": "ENTER_LONG"}
        if go_short:
            return {"action": "ENTER_SHORT"}
        return {"action": "HOLD"}
