# v2.1 bug-fix release

## Live trading (critical)
- Failed close orders no longer mark the position closed; state is unchanged and the close is retried.
- Positions are reconciled with the exchange every tick (exchange stop hit, manual close, liquidation,
  partial reduce). Real stop fill price is used; no more fake TP profits after an exchange stop.
- Open positions are loaded per exchange (switching Binance -> Bybit no longer mixes positions).
- Live PnL is net of entry and exit fees (fees in base coin converted; spot qty reduced by coin fee).
- Isolated margin is requested; a trade is refused if the leverage can't be set.
- Exchange-side stop: new stop placed before the old is cancelled; stop cancelled on full close,
  restored if the close order fails.
- Partial TPs below the exchange minimum are skipped instead of being booked as filled.
- Testnet: Binance uses Demo Trading (enable_demo_trading), Bybit uses testnet; the live/testnet
  account uses its own venue's prices for charts, signals and SL/TP.

## Logic
- ML model now retrains every 25 new bars (counted by time; window length is fixed).
- ML holdout is purged by the label horizon and the edge uses a lower confidence bound.
- Demo data: price for manual open/close comes from the same series as the chart.
- Paper account: wicks between polls trigger SL/TP (stop first, gap fills at price).
- If candles fail to load, open positions are still checked with the ticker price.
- OHLCV pagination respects each exchange's per-request limit (KuCoin futures 200).
- AI Confluence strength rescaled so valid triggers aren't silently filtered at 60%.
- SL reason reports when the swing stop was clamped to the ATR range.
- Position size includes fees and is capped by free margin; risk is based on equity (futures).
- Daily loss limit uses start-of-day equity and includes partial exits of open trades.

## Backtest
- Gaps through the stop fill at the bar open; notional cap applied; intra-trade drawdown counted;
  min-confidence and leverage applied like the robot; time exits no longer double counted.

## App / security
- API secret/passphrase stored in the OS keychain (keyring), settings file written with 0600.
- Settings values are type-checked and clamped on load.
- Switching to a live account connects in the background (no UI freeze); on failure the old account is kept.
- Manual open and auto-trade are serialized (max positions can't be exceeded); signal memory is bounded.
- New regression tests: tests/test_fixes.py (fake exchange, no network).
