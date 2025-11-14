# FX_BACKTEST/strategies/ma_cross.py
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import deque
from core.events import TickEvent, SignalEvent

class MACross:
    def __init__(self, q, symbol: str, short: int = 20, long: int = 50, size: float = 10000.0):
        assert short < long
        self.q = q
        self.symbol = symbol
        self.short_n, self.long_n = short, long
        self.short_win, self.long_win = deque(maxlen=short), deque(maxlen=long)
        self.pos = 0.0          # 当前方向（>0 多 / <0 空 / =0 空仓）
        self.size = size

    def on_event(self, ev):
        if isinstance(ev, TickEvent) and ev.symbol == self.symbol:
            mid = (ev.bid + ev.ask) / 2.0
            self.short_win.append(mid)
            self.long_win.append(mid)
            if len(self.long_win) < self.long_n:
                return
            sma_s = sum(self.short_win) / len(self.short_win)
            sma_l = sum(self.long_win) / len(self.long_win)
            # 交叉信号
            if self.pos <= 0 and sma_s > sma_l:   # 金叉 -> 做多
                self.q.put(SignalEvent(ev.ts, self.symbol, "LONG", self.size))
                self.pos = 1
            elif self.pos >= 0 and sma_s < sma_l: # 死叉 -> 做空
                self.q.put(SignalEvent(ev.ts, self.symbol, "SHORT", self.size))
                self.pos = -1