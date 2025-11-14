"""XGBoost-based signal strategy (long-only v1).

Loads a trained Booster + feature list + thresholds and emits ENTER_LONG/EXIT_LONG
decisions based on predicted probability compared to configured thresholds.

Registration name: "xgb_signal"
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger

from . import register
from .base import Strategy


def _safe_get(d: Dict[str, Any], key: str, default=None):
    v = d.get(key, default)
    return v if v is not None else default


def _rsi_from_series(arr: np.ndarray, period: int = 14) -> float | None:
    if arr.size < period + 1:
        return None
    diffs = np.diff(arr)
    gains = diffs[diffs > 0]
    losses = -diffs[diffs < 0]
    avg_gain = gains.mean() if gains.size > 0 else 0.0
    avg_loss = losses.mean() if losses.size > 0 else 0.0
    if avg_loss == 0.0 and avg_gain == 0.0:
        return 50.0
    if avg_loss == 0.0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


@register("xgb_signal")
class XGBSignal(Strategy):
    def __init__(
        self,
        model_dir: Optional[str] = None,
        latest_ptr: str = "QuantResearch/artifacts/models/usdjpy_h1_xgb_latest.json",
        prob_long: Optional[float] = None,
        prob_exit: Optional[float] = None,
        size_mult: float = 1.0,
        cooldown_bars: int = 0,
        min_atr_pct: Optional[float] = None,
        low_atr_pct: Optional[float] = None,
        prob_long_low: Optional[float] = None,
        cooldown_low: Optional[int] = None,
        min_vol_24: Optional[float] = None,
        atr_relax_pct: Optional[float] = None,
        prob_long_relaxed: Optional[float] = None,
        cooldown_relaxed: Optional[int] = None,
        debug_log_hits: bool = False,
    ) -> None:
        super().__init__()
        # Resolve model directory
        if not model_dir:
            ptr = Path(latest_ptr)
            if not ptr.exists():
                raise RuntimeError(f"latest.json not found: {ptr}")
            latest = json.loads(ptr.read_text(encoding="utf-8"))
            model_dir = latest.get("model_dir")
            if not model_dir:
                raise RuntimeError("latest.json missing 'model_dir'")
        self.model_dir = Path(model_dir)
        # Load artifacts
        self.feature_list: List[str] = json.loads((self.model_dir / "feature_list.json").read_text(encoding="utf-8"))
        thr = json.loads((self.model_dir / "thresholds.json").read_text(encoding="utf-8"))
        self.p_long = float(prob_long) if prob_long is not None else float(thr.get("p_long", 0.6))
        self.p_exit = float(prob_exit) if prob_exit is not None else float(thr.get("p_exit", 0.5))
        try:
            import xgboost as xgb
        except Exception as exc:
            raise RuntimeError("xgboost is required at runtime for xgb_signal.") from exc
        self._xgb = xgb
        self._booster = xgb.Booster()
        self._booster.load_model(str(self.model_dir / "model.json"))

        self.size_mult = float(size_mult)
        self.cooldown_bars = int(max(0, cooldown_bars))
        self.cooldown_relaxed = int(max(0, cooldown_relaxed)) if cooldown_relaxed is not None else None
        self.min_atr_pct = float(min_atr_pct) if min_atr_pct is not None else None
        self.low_atr_pct = float(low_atr_pct) if low_atr_pct is not None else None
        self.prob_long_low = float(prob_long_low) if prob_long_low is not None else None
        self.cooldown_low = int(max(0, cooldown_low)) if cooldown_low is not None else None
        self.min_vol_24 = float(min_vol_24) if min_vol_24 is not None else None
        self.atr_relax_pct = float(atr_relax_pct) if atr_relax_pct is not None else None
        self.p_long_relaxed = float(prob_long_relaxed) if prob_long_relaxed is not None else None
        self.debug_log_hits = bool(debug_log_hits)
        self._block_until: int = 0
        self._debug_max = 0.0
        self._debug_none = 0

    def _note_feature_miss(self, reason: str) -> None:
        self._debug_none += 1
        if self.debug_log_hits and self._debug_none <= 10:
            logger.warning(f"[xgb_signal] feature unavailable ({reason})")

    def _features_from_state(self, state: Dict[str, Any]) -> tuple[Optional[np.ndarray], Optional[float]]:
        # Close history for returns/volatility
        ch = state.get("close_history")
        if ch is None:
            self._note_feature_miss("close_history missing")
            return None, None
        closes = np.asarray(ch, dtype=float)
        if closes.size < 80:  # need at least slow window context
            self._note_feature_miss("insufficient history")
            return None, None
        close = float(state.get("close", closes[-1]))

        # Returns & rolling vol
        def pct_change(arr: np.ndarray, k: int) -> float | None:
            if arr.size <= k:
                self._note_feature_miss(f"ret_{k} insufficient")
                return None, None
            a, b = arr[-k - 1], arr[-1]
            return (b - a) / a if a else None

        ret_1 = pct_change(closes, 1)
        ret_3 = pct_change(closes, 3)
        ret_6 = pct_change(closes, 6)
        vol_24 = None
        if closes.size >= 25:
            rets = np.diff(closes[-25:]) / closes[-25:-1]
            vol_24 = float(np.std(rets)) if rets.size else None

        # SMA diff
        sma_fast = _safe_get(state, "sma_fast")
        sma_slow = _safe_get(state, "sma_slow")
        if sma_fast is None or sma_slow is None:
            sma_fast = float(np.mean(closes[-20:])) if closes.size >= 20 else None
            sma_slow = float(np.mean(closes[-80:])) if closes.size >= 80 else None
        if sma_fast is None or sma_slow is None:
            self._note_feature_miss("sma missing")
            return None, None
        sma_diff = (float(sma_fast) - float(sma_slow)) / close if close else 0.0

        # RSI (prefer engine state, else compute)
        rsi_val = state.get("rsi")
        if rsi_val is None:
            r = _rsi_from_series(closes, 14)
            rsi_val = r if r is not None else 50.0

        # ATR normalized
        curr_atr = state.get("curr_atr")
        atr_norm = float(curr_atr) / close if (curr_atr is not None and close) else 0.0

        # Time features from ts
        ts = state.get("ts")
        if ts is None:
            self._note_feature_miss("timestamp missing")
            return None, None
        try:
            import pandas as pd
            ts_pd = pd.Timestamp(ts)
            hour = float(ts_pd.hour)
            dow = float(ts_pd.dayofweek)
        except Exception:
            self._note_feature_miss("timestamp parse")
            return None, None
        hour_sin, hour_cos = np.sin(2 * np.pi * hour / 24.0), np.cos(2 * np.pi * hour / 24.0)
        dow_sin, dow_cos = np.sin(2 * np.pi * dow / 7.0), np.cos(2 * np.pi * dow / 7.0)

        feat_map = {
            "ret_1": ret_1,
            "ret_3": ret_3,
            "ret_6": ret_6,
            "vol_24": vol_24,
            "sma_diff": sma_diff,
            "rsi": float(rsi_val),
            "atr_norm": atr_norm,
            "hour_sin": float(hour_sin),
            "hour_cos": float(hour_cos),
            "dow_sin": float(dow_sin),
            "dow_cos": float(dow_cos),
        }
        vec = []
        for name in self.feature_list:
            val = feat_map.get(name)
            if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
                self._note_feature_miss(f"feature {name} invalid")
                return None, None
            vec.append(float(val))
        return np.asarray(vec, dtype=float), float(vol_24) if vol_24 is not None else None

    def on_bar(self, state: Dict[str, Any]) -> Dict[str, Any]:
        bar_idx = int(state.get("bar_idx", 0) or 0)
        position_units = float(state.get("position_units", 0.0) or 0.0)
        default_qty = float(state.get("default_qty", 0.0) or 0.0)
        close = float(state.get("close", 0.0) or 0.0)

        atr_pct = state.get("atr_percentile")
        if self.min_atr_pct is not None:
            if atr_pct is None or float(atr_pct) < self.min_atr_pct:
                return {"action": "HOLD"}

        # Determine per-bar entry threshold / cooldown after ATR gating
        effective_prob_long = self.p_long
        effective_cooldown = self.cooldown_bars
        if (
            self.low_atr_pct is not None
            and atr_pct is not None
            and float(atr_pct) < self.low_atr_pct
        ):
            if self.prob_long_low is not None:
                effective_prob_long = self.prob_long_low
            if self.cooldown_low is not None:
                effective_cooldown = self.cooldown_low
        if (
            self.atr_relax_pct is not None
            and atr_pct is not None
            and float(atr_pct) >= self.atr_relax_pct
        ):
            if self.p_long_relaxed is not None:
                effective_prob_long = self.p_long_relaxed
            if self.cooldown_relaxed is not None:
                effective_cooldown = self.cooldown_relaxed

        # Cooldown gate (updated per bar)
        if bar_idx < self._block_until:
            return {"action": "HOLD"}

        feats, vol_24 = self._features_from_state(state)
        if feats is None:
            return {"action": "HOLD"}
        if self.min_vol_24 is not None:
            if vol_24 is None or vol_24 < self.min_vol_24:
                return {"action": "HOLD"}

        dmat = self._xgb.DMatrix(feats.reshape(1, -1), feature_names=self.feature_list)
        p_up = float(self._booster.predict(dmat)[0])
        if p_up > self._debug_max:
            self._debug_max = p_up
            if self.debug_log_hits:
                logger.info(
                    "[xgb_signal] new max prob %.4f (ts=%s close=%.5f position=%s)",
                    p_up,
                    state.get("ts"),
                    close,
                    position_units,
                )

        # Long-only logic
        if position_units == 0.0:
            if p_up >= effective_prob_long and default_qty > 0.0:
                self._block_until = bar_idx + effective_cooldown
                if self.debug_log_hits:
                    logger.info(
                        "[xgb_signal] ENTER signal p=%.4f (thr=%.4f) ts=%s",
                        p_up,
                        effective_prob_long,
                        state.get("ts"),
                    )
                return {"action": "ENTER_LONG", "size": default_qty * self.size_mult}
            return {"action": "HOLD"}
        else:
            if p_up < self.p_exit:
                if self.debug_log_hits:
                    logger.info(
                        "[xgb_signal] EXIT signal p=%.4f (thr=%.4f) ts=%s",
                        p_up,
                        self.p_exit,
                        state.get("ts"),
                    )
                return {"action": "EXIT_LONG"}
            return {"action": "HOLD"}
