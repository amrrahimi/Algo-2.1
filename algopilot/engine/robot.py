"""The robot: scans charts, plans trades, notifies, optionally auto-trades,
and manages open positions. Runs in a background thread; talks to the UI
only through the `events` callback object (Qt-agnostic)."""
from __future__ import annotations

import threading
import time
import traceback
from datetime import datetime, timezone

from ..brokers.paper import PaperBroker
from ..core import indicators as ind
from ..core.analyzer import Analyzer
from ..core.ml import MLFilter
from ..core.risk import position_size, sl_safe_with_leverage
from ..data.feed import make_feed


class NullEvents:
    def log(self, msg): print(msg)
    def chart(self, symbol, payload): pass
    def signal(self, sig): pass
    def positions(self, positions): pass
    def trade_event(self, ev): pass
    def account(self, info): pass
    def status(self, text): pass


class Robot:
    def __init__(self, settings, db, events=None):
        self.db = db
        self.events = events or NullEvents()
        self.ml = MLFilter()
        self.settings = settings.copy()
        self.scanning = False
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._lock = threading.RLock()
        self._seen = {}
        self._hl = {}                       # symbol -> (bar_ts, high, low, checked_at)
        self._exec_lock = threading.Lock()
        self._thread = None
        self.feed = None
        self.broker = None
        self.last_df = {}
        self._build()

    # ------------------------------------------------------------ config
    @staticmethod
    def _heavy_key(s):
        c = s.creds
        return (s.data_exchange, s.market_type, s.account, c.exchange, c.api_key, c.secret,
                c.password, c.testnet, s.exchange_side_sl)

    def needs_rebuild(self, settings) -> bool:
        return self._heavy_key(self.settings) != self._heavy_key(settings)

    def _make(self, s):
        """Create (feed, broker) for settings. Network calls happen here: never under the lock."""
        if s.account == "live":
            from ..brokers.live import LiveBroker
            broker = LiveBroker(self.db, s.market_type, s.creds, s.exchange_side_sl, self.events.log)
            # prices for SL/TP management and signals must come from the venue we trade on
            feed = make_feed(s.creds.exchange, s.market_type, client=broker.ex)
            if s.data_exchange != s.creds.exchange or s.creds.testnet:
                self.events.log(f"Live account: using {s.creds.exchange}"
                                f"{' TESTNET' if s.creds.testnet else ''} prices for charts and signals.")
        else:
            feed = make_feed(s.data_exchange, s.market_type)
            broker = PaperBroker(self.db, s.market_type, s.paper_start_balance, log=self.events.log)
        return feed, broker

    def _build(self):
        s = self.settings
        self.feed, self.broker = self._make(s)
        self.analyzer = Analyzer(s.strategy, self.ml, s.use_ml)

    def reconfigure(self, settings):
        """Apply new settings. May block on the network (call from a worker thread when
        switching accounts). Raises if the new broker can't be created; the old one stays."""
        new = settings.copy()
        if self.needs_rebuild(new):
            feed, broker = self._make(new)
            with self._lock:
                self.settings, self.feed, self.broker = new, feed, broker
                self.analyzer = Analyzer(new.strategy, self.ml, new.use_ml)
                self.last_df.clear()
                self._hl.clear()
        else:
            with self._lock:
                old = self.settings
                self.settings = new
                self.analyzer = Analyzer(new.strategy, self.ml, new.use_ml)
                if old.timeframe != new.timeframe:
                    self.last_df.clear()
                    self._hl.clear()
        self._wake.set()

    # ------------------------------------------------------------ control
    def start_engine(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="robot", daemon=True)
        self._thread.start()

    def shutdown(self):
        self._stop.set()
        self._wake.set()
        if self._thread:
            self._thread.join(timeout=5)

    def set_scanning(self, on: bool):
        self.scanning = on
        self.events.status("Robot running" if on else "Robot stopped (monitoring positions)")
        self._wake.set()

    # ------------------------------------------------------------ loop
    def _run(self):
        while not self._stop.is_set():
            t0 = time.time()
            try:
                self.tick()
            except Exception as e:
                self.events.log(f"Engine error: {e}\n{traceback.format_exc(limit=2)}")
            wait = max(1.0, self.settings.poll_seconds - (time.time() - t0))
            self._wake.wait(wait)
            self._wake.clear()

    def _range_since_last(self, sym, df):
        """High/low traded since the previous tick (None on first look)."""
        cur_ts, h, l = df.index[-1], float(df["high"].iloc[-1]), float(df["low"].iloc[-1])
        prev = self._hl.get(sym)
        now = time.time()
        self._hl[sym] = (cur_ts, h, l, now)
        if prev is None:
            return None, None, None
        p_ts, p_h, p_l, p_at = prev
        newer = df[df.index > p_ts]
        his, los = list(newer["high"]), list(newer["low"])
        if p_ts in df.index:
            if df.at[p_ts, "high"] > p_h:
                his.append(float(df.at[p_ts, "high"]))
            if df.at[p_ts, "low"] < p_l:
                los.append(float(df.at[p_ts, "low"]))
        return (max(his) if his else None), (min(los) if los else None), p_at

    def tick(self):
        with self._lock:
            s, feed, broker, analyzer = self.settings, self.feed, self.broker, self.analyzer
        try:
            for ev in broker.sync():
                self.events.trade_event(ev)
        except Exception as e:
            self.events.log(f"Position sync error: {e}")
        open_syms = [p.symbol for p in broker.open_positions()]
        symbols = list(dict.fromkeys(s.symbols + open_syms))
        for sym in symbols:
            if self._stop.is_set():
                return
            try:
                df = feed.fetch_ohlcv(sym, s.timeframe, 600)
            except Exception as e:
                self.events.log(f"{sym}: data error: {e}")
                if sym in open_syms:          # still protect the open position
                    try:
                        px = feed.fetch_price(sym, s.timeframe)
                        for ev in broker.on_price(sym, px):
                            self.events.trade_event(ev)
                    except Exception as e2:
                        self.events.log(f"{sym}: price error, position NOT checked: {e2}")
                continue
            price = float(df["close"].iloc[-1])
            hi, lo, since = self._range_since_last(sym, df)
            for ev in broker.on_price(sym, price, hi, lo, since):
                self.events.trade_event(ev)
            if len(df) < 250:
                self.events.log(f"{sym}: not enough history ({len(df)} bars)")
                continue
            closed = df.iloc[:-1]           # last candle is still forming
            sig, sigframe = None, None
            if sym in s.symbols:
                try:
                    sig, sigframe = analyzer.latest_signal(closed, sym, s.timeframe)
                except Exception as e:
                    self.events.log(f"{sym}: analysis error: {e}")
                if sig and self.scanning:
                    self._handle_signal(sig, price)
            self.last_df[sym] = df
            self.events.chart(sym, self._chart_payload(sym, df, sigframe, s))
        self.events.positions(broker.open_positions())
        try:
            self.events.account({"balance": broker.balance(), "equity": broker.equity(),
                                 "account": s.account,
                                 "testnet": bool(s.account == "live" and s.creds.testnet)})
        except Exception as e:
            self.events.log(f"Balance error: {e}")

    # ------------------------------------------------------------ signals
    def _handle_signal(self, sig, price):
        key = (sig.symbol, sig.timeframe, sig.bar_time, sig.side)
        if key in self._seen:
            return
        self._seen[key] = time.time()
        if len(self._seen) > 2000:          # keep memory bounded
            cutoff = time.time() - 3 * 86400
            self._seen = {k: t for k, t in self._seen.items() if t >= cutoff}
        s = self.settings
        if "ML veto" in sig.reasons or sig.confidence < s.min_confidence:
            self.events.log(f"{sig.symbol} {sig.side} filtered (confidence {sig.confidence:.0%})")
            return
        self.db.add_signal(sig)
        self.events.signal(sig)
        if s.trade_mode == "auto":
            try:
                self.execute_signal(sig, price, auto=True)
            except Exception as e:
                self.events.log(f"Auto-trade skipped for {sig.symbol}: {e}")

    def daily_loss_hit(self) -> bool:
        s = self.settings
        start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
        realized = self.db.realized_since(self.broker.account, start)
        eq = (self.broker.equity() or 0) - realized     # equity at the start of the day (approx.)
        if eq <= 0:
            return realized < 0
        return realized < 0 and -realized / eq * 100 >= s.max_daily_loss_pct

    def execute_signal(self, sig, price=None, auto=False):
        """Open a trade from a signal (robot auto-trade or user click). Serialized so two
        threads can't both pass the max-positions / one-per-coin checks."""
        with self._exec_lock:
            return self._execute(sig, price, auto)

    def _execute(self, sig, price=None, auto=False):
        with self._lock:
            s, broker, feed = self.settings, self.broker, self.feed
        if sig.timeframe and sig.timeframe != s.timeframe:
            raise RuntimeError(f"signal is for {sig.timeframe} but the robot now runs {s.timeframe}")
        if s.market_type == "spot" and sig.side == "short":
            raise RuntimeError("short signals can't be traded on spot")
        if len(broker.open_positions()) >= s.max_positions:
            raise RuntimeError(f"max open positions ({s.max_positions}) reached")
        if any(p.symbol == sig.symbol for p in broker.open_positions()):
            raise RuntimeError(f"already in a {sig.symbol} position")
        if self.daily_loss_hit():
            raise RuntimeError("daily loss limit reached, trading paused until tomorrow (UTC)")
        price = price or feed.fetch_price(sig.symbol, s.timeframe)
        # signal must still be valid at the current price
        if (sig.side == "long" and price <= sig.sl) or (sig.side == "short" and price >= sig.sl):
            raise RuntimeError("price already beyond stop loss, signal expired")
        if (sig.side == "long" and price >= sig.tps[0]) or (sig.side == "short" and price <= sig.tps[0]):
            raise RuntimeError("price already passed TP1, signal expired")
        lev = s.leverage if s.market_type == "futures" else 1
        if not sl_safe_with_leverage(sig.side, price, sig.sl, lev):
            raise RuntimeError(f"stop loss is beyond liquidation at {lev}x, lower leverage")
        qty = position_size(broker.equity() if s.market_type == "futures" else broker.balance(),
                            s.risk_pct, price, sig.sl, s.market_type, lev, fee=0.0006,
                            free_balance=broker.balance())
        if qty <= 0:
            raise RuntimeError("position size is zero (check balance)")
        pos = broker.open_from_signal(sig, qty, lev, price)
        self.db.mark_signal(sig.id)
        self.events.trade_event({"type": "opened", "pos": pos, "reason": "auto" if auto else "manual",
                                 "price": pos.entry, "qty": pos.qty, "pnl": 0.0})
        self.events.positions(broker.open_positions())
        return pos

    def close_position(self, pos_id: str):
        from ..brokers.base import ExecError
        with self._lock:
            broker, feed, tf = self.broker, self.feed, self.settings.timeframe
        pos = broker.positions.get(pos_id)
        if not pos:
            return
        price = feed.fetch_price(pos.symbol, tf)
        try:
            for ev in broker.close(pos_id, price, "manual"):
                self.events.trade_event(ev)
        except ExecError as e:
            raise RuntimeError(f"close order failed, position is still open: {e}") from e
        self.events.positions(broker.open_positions())

    # ------------------------------------------------------------ chart data
    def _chart_payload(self, sym, df, sigframe, s):
        c = df["close"]
        st_line, st_dir = ind.supertrend(df)
        payload = {"symbol": sym, "timeframe": s.timeframe, "df": df,
                   "ema20": ind.ema(c, 20), "ema50": ind.ema(c, 50), "ema200": ind.ema(c, 200),
                   "supertrend": st_line, "st_dir": st_dir, "rsi": ind.rsi(c),
                   "markers": [], "positions": [p for p in self.broker.open_positions() if p.symbol == sym]}
        if sigframe is not None:
            n = len(sigframe)
            for i in range(max(0, n - 400), n):
                if sigframe["long"].iloc[i]:
                    payload["markers"].append((i, "long", float(df["low"].iloc[i])))
                elif sigframe["short"].iloc[i]:
                    payload["markers"].append((i, "short", float(df["high"].iloc[i])))
        return payload
