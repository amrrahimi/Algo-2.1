"""Turns raw strategy triggers into full trade plans (entry / SL / TP1-3)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import indicators as ind
from .levels import choose_sl, choose_tps, levels_at, pivot_flags
from .signals import Signal
from .strategies import STRATEGIES


class Analyzer:
    def __init__(self, strategy_key: str, ml_filter=None, use_ml: bool = True):
        self.strategy = STRATEGIES.get(strategy_key, STRATEGIES["ai_confluence"])
        self.ml = ml_filter
        self.use_ml = use_ml and ml_filter is not None

    def prepare(self, df: pd.DataFrame):
        sig = self.strategy.compute(df)
        a = ind.atr(df, 14)
        ph, pl = pivot_flags(df)
        return sig, a, ph, pl

    def plan(self, df, i, side, strength, a, ph, pl, symbol="", timeframe="") -> Signal | None:
        atr_val = float(a.iloc[i])
        if not np.isfinite(atr_val) or atr_val <= 0:
            return None
        entry = float(df["close"].iloc[i])
        sl, sl_note = choose_sl(df, i, side, entry, atr_val, pl, ph)
        lv = levels_at(df, i, ph, pl, atr_val)
        tps, tp_notes = choose_tps(side, entry, sl, lv, atr_val)
        return Signal(symbol=symbol, timeframe=timeframe, strategy=self.strategy.name, side=side,
                      entry=entry, sl=float(sl), tps=[float(t) for t in tps],
                      confidence=float(strength),
                      reasons=[f"SL {sl_note}", "TPs: " + ", ".join(tp_notes)],
                      bar_time=float(df.index[i].timestamp()))

    def latest_signal(self, df: pd.DataFrame, symbol: str, timeframe: str):
        """Evaluate the last CLOSED candle of df. Returns (Signal|None, sigframe)."""
        sig, a, ph, pl = self.prepare(df)
        i = len(df) - 1
        row = sig.iloc[i]
        side = "long" if row["long"] else "short" if row["short"] else None
        if side is None:
            return None, sig
        s = self.plan(df, i, side, float(row["strength"]), a, ph, pl, symbol, timeframe)
        if s is None:
            return None, sig
        if self.use_ml:
            p_up, acc = self.ml.prob_up((symbol, timeframe), df)
            if p_up is not None:
                p_side = p_up if side == "long" else 1 - p_up
                s.ml_prob = p_side
                s.reasons.append(f"ML {p_side:.0%} (holdout acc {acc:.0%})")
                if p_side < 0.45:          # ML veto
                    s.reasons.append("ML veto")
                    s.confidence = min(s.confidence, p_side)
                    return s, sig
                s.confidence = 0.5 * s.confidence + 0.5 * min(1.0, p_side + 0.1)
        return s, sig
