"""Trading strategies.

Each strategy returns a DataFrame aligned to the candles with columns
`long`, `short` (bool, signal on that closed bar) and `strength` (0..1).
All strategies stack noise filters (ATR activity, volume, trend/ADX or
efficiency ratio) to avoid trading chop.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import indicators as ind


def _b(s: pd.Series) -> pd.Series:
    return s.fillna(False).astype(bool)


def _cross_up(a, b):
    return _b((a > b) & (a.shift() <= b.shift()))


def _cross_dn(a, b):
    return _b((a < b) & (a.shift() >= b.shift()))


class Strategy:
    key = "base"
    name = "Base"
    modes = ("scalp", "daily", "long")
    description = ""

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    @staticmethod
    def _out(df, long, short, strength):
        return pd.DataFrame({"long": _b(long), "short": _b(short),
                             "strength": pd.Series(strength, index=df.index).fillna(0).clip(0, 1)},
                            index=df.index)


class UTBotTrend(Strategy):
    key, name = "ut_bot_trend", "UT Bot + EMA200 Trend"
    modes = ("scalp", "daily")
    description = "UT Bot ATR-trailing-stop flips, only with the EMA200 trend and ADX > 18."

    def compute(self, df):
        c = df["close"]
        _, buy, sell = ind.ut_bot(df, key=1.5, n=10)
        e200 = ind.ema(c, 200)
        adx, _, _ = ind.adx(df)
        kslope = ind.slope(ind.kalman_price(c), 5)
        ok = ind.noise_ok(df) & (adx > 18)
        long = buy & (c > e200) & ok
        short = sell & (c < e200) & ok
        strength = 0.55 + 0.2 * (adx > 25) + 0.15 * (np.sign(kslope) == np.where(long, 1, -1))
        return self._out(df, long, short, strength)


class RangeFilterBreakout(Strategy):
    key, name = "range_filter", "Range Filter Breakout"
    modes = ("scalp", "daily")
    description = "Range Filter direction change with volume above average (noise-filtered trend)."

    def compute(self, df):
        c = df["close"]
        filt, d = ind.range_filter(c, 50, 2.5)
        vol_ok = df["volume"] > ind.sma(df["volume"], 20)
        e200 = ind.ema(c, 200)
        ok = ind.noise_ok(df) & vol_ok
        long = (d == 1) & (d.shift() != 1) & (c > filt) & (c > e200) & ok
        short = (d == -1) & (d.shift() != -1) & (c < filt) & (c < e200) & ok
        er = ind.efficiency_ratio(c)
        return self._out(df, long, short, 0.55 + 0.4 * er)


class SqueezeBreakout(Strategy):
    key, name = "squeeze", "Squeeze Momentum Breakout"
    modes = ("scalp", "daily", "long")
    description = "Volatility squeeze release (Bollinger exits Keltner) in direction of rising momentum."

    def compute(self, df):
        c = df["close"]
        val, sqz = ind.squeeze_momentum(df)
        fired = _b(sqz.shift(1)) & ~sqz
        recent = fired | _b(fired.shift(1))
        e50 = ind.ema(c, 50)
        ok = ind.noise_ok(df)
        long = recent & (val > 0) & (val > val.shift()) & (c > e50) & ok
        short = recent & (val < 0) & (val < val.shift()) & (c < e50) & ok
        long &= ~_b(long.shift(1))
        short &= ~_b(short.shift(1))
        mag = (val.abs() / ind.atr(df)).clip(0, 2) / 2
        return self._out(df, long, short, 0.55 + 0.35 * mag)


class SupertrendADXMACD(Strategy):
    key, name = "supertrend_adx", "Supertrend + ADX + MACD"
    modes = ("daily", "long")
    description = "Supertrend direction confirmed by ADX > 20 and MACD histogram turning."

    def compute(self, df):
        c = df["close"]
        _, d = ind.supertrend(df, 10, 3.0)
        adx, _, _ = ind.adx(df)
        _, _, h = ind.macd(c)
        flip = d != d.shift()
        ok = ind.noise_ok(df) & (adx > 20)
        long = (d == 1) & (h > 0) & (flip | (h.shift() <= 0)) & ok
        short = (d == -1) & (h < 0) & (flip | (h.shift() >= 0)) & ok
        return self._out(df, long, short, 0.5 + (adx.clip(20, 45) - 20) / 50)


class WaveTrendReversal(Strategy):
    key, name = "wavetrend", "WaveTrend Reversal + RSI"
    modes = ("scalp", "daily")
    description = "WaveTrend cross in oversold/overbought zones, confirmed by RSI and candle direction."

    def compute(self, df):
        c = df["close"]
        wt1, wt2 = ind.wavetrend(df)
        r = ind.rsi(c)
        ok = ind.noise_ok(df)
        long = _cross_up(wt1, wt2) & (wt1 < -53) & (r < 45) & (c > df["open"]) & ok
        short = _cross_dn(wt1, wt2) & (wt1 > 53) & (r > 55) & (c < df["open"]) & ok
        depth = (wt1.abs().clip(53, 90) - 53) / 74
        return self._out(df, long, short, 0.55 + depth)


class IchimokuTrend(Strategy):
    key, name = "ichimoku", "Ichimoku Cloud Trend"
    modes = ("daily", "long")
    description = "Tenkan/Kijun cross above/below the cloud with Chikou confirmation."

    def compute(self, df):
        c = df["close"]
        t, k, a, b = ind.ichimoku(df)
        top, bot = np.maximum(a, b), np.minimum(a, b)
        ok = ind.noise_ok(df)
        long = _cross_up(t, k) & (c > top) & (c > c.shift(26)) & ok
        short = _cross_dn(t, k) & (c < bot) & (c < c.shift(26)) & ok
        return self._out(df, long, short, 0.65 + 0.25 * ind.efficiency_ratio(c))


class EMADonchian(Strategy):
    key, name = "ema_donchian", "EMA 50/200 + Donchian Breakout"
    modes = ("daily", "long")
    description = "Turtle-style 20-bar breakout only in the EMA 50/200 trend direction."

    def compute(self, df):
        c = df["close"]
        e50, e200 = ind.ema(c, 50), ind.ema(c, 200)
        hi, lo = ind.donchian(df, 20)
        ok = ind.noise_ok(df)
        long = (e50 > e200) & (c > hi.shift()) & (c.shift() <= hi.shift(2)) & ok
        short = (e50 < e200) & (c < lo.shift()) & (c.shift() >= lo.shift(2)) & ok
        return self._out(df, long, short, 0.55 + 0.4 * ind.efficiency_ratio(c, 40))


class VWAPPullback(Strategy):
    key, name = "vwap_pullback", "VWAP Pullback"
    modes = ("scalp",)
    description = "Trend pullback that taps session VWAP and rejects it."

    def compute(self, df):
        c, o = df["close"], df["open"]
        vw = ind.vwap(df)
        e50, e200 = ind.ema(c, 50), ind.ema(c, 200)
        r = ind.rsi(c)
        a = ind.atr(df)
        ok = ind.noise_ok(df)
        long = (e50 > e200) & (df["low"] <= vw + 0.1 * a) & (c > vw) & (c > o) & r.between(40, 62) & ok
        short = (e50 < e200) & (df["high"] >= vw - 0.1 * a) & (c < vw) & (c < o) & r.between(38, 60) & ok
        long &= ~_b(long.shift(1))
        short &= ~_b(short.shift(1))
        return self._out(df, long, short, 0.6)


class AIConfluence(Strategy):
    key, name = "ai_confluence", "AI Confluence (all indicators + ML)"
    modes = ("scalp", "daily", "long")
    description = ("Weighted vote of 11 trend/momentum/Algo indicators, regime-filtered by ADX and "
                   "efficiency ratio. Pair with the ML filter for highest precision.")
    threshold = 0.55

    def votes(self, df) -> pd.Series:
        c = df["close"]
        _, st_dir = ind.supertrend(df)
        ut_stop, _, _ = ind.ut_bot(df, 1.5, 10)
        _, rf_dir = ind.range_filter(c, 50, 2.5)
        e20, e50, e200 = ind.ema(c, 20), ind.ema(c, 50), ind.ema(c, 200)
        _, _, hist = ind.macd(c)
        r = ind.rsi(c)
        wt1, wt2 = ind.wavetrend(df)
        _, _, sa, sb = ind.ichimoku(df)
        sq, _ = ind.squeeze_momentum(df)
        kal = ind.kalman_price(c)
        h = ind.hma(c, 55)
        stack = np.sign(e20 - e50) * 0.5 + np.sign(e50 - e200) * 0.5
        cloud = np.where(c > np.maximum(sa, sb), 1, np.where(c < np.minimum(sa, sb), -1, 0))
        parts = [
            (st_dir, 1.0), (np.sign(c - ut_stop), 1.0), (rf_dir, 1.0), (stack, 1.0),
            (np.sign(hist), 0.7), (np.sign(r - 50), 0.5), (np.sign(wt1 - wt2), 0.5),
            (pd.Series(cloud, index=df.index), 0.8),
            (np.sign(sq) * (np.sign(sq.diff()) == np.sign(sq)), 0.7),
            (np.sign(ind.slope(kal, 3)), 1.0), (np.sign(h.diff()), 0.6),
        ]
        total = sum(w for _, w in parts)
        score = sum(pd.Series(v, index=df.index).fillna(0) * w for v, w in parts) / total
        return score

    def compute(self, df):
        c = df["close"]
        score = self.votes(df)
        adx, _, _ = ind.adx(df)
        er = ind.efficiency_ratio(c)
        regime = (adx > 18) & (er > 0.2)
        ok = ind.noise_ok(df) & regime
        thr = self.threshold
        long = (score >= thr) & (score.shift() < thr) & ok
        short = (score <= -thr) & (score.shift() > -thr) & ok
        # map |score| in [thr, 1] to [0.6, 0.9] (+ up to 0.1 for a clean trend) so a valid
        # trigger isn't silently dropped by the default 60% minimum confidence
        over = ((score.abs() - thr) / (1 - thr)).clip(0, 1)
        strength = 0.6 + 0.3 * over + 0.1 * er.clip(0, 1)
        return self._out(df, long, short, strength)


STRATEGIES = {s.key: s for s in (AIConfluence(), UTBotTrend(), RangeFilterBreakout(), SqueezeBreakout(),
                                 SupertrendADXMACD(), WaveTrendReversal(), IchimokuTrend(),
                                 EMADonchian(), VWAPPullback())}


def strategies_for_mode(mode: str):
    return [s for s in STRATEGIES.values() if mode in s.modes]
