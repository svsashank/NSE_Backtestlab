"""
core/milt25_engine.py — MILT 25 strategy backtester.

Event-driven, per-stock weekly breakout system. Fundamentally different
shape from core/backtest_engine.py's ranked top-N rebalance model:
  - No cross-sectional ranking cut (ROC_12M is only used as a tie-breaker
    when more Buy signals fire in one week than there are free slots).
  - Positions are opened/closed independently on their own signals, not
    forced to a fixed rebalance calendar.
  - No trimming, no rebalancing of existing winners — a position runs
    until one of its three exit rules fires.

Universe/filter: MCap > config['min_mcap'] (Cr) only. All other legacy
screener filters (ADV, RSI, volatility, SMA buffer, CMF) are intentionally
NOT applied — MILT 25 is a standalone system with its own entry logic.

Data note: the OHLCV store only has daily Close/High/Low/Volume (no Open).
"Monday Open execution" is approximated using the Close of the next
trading day after the Friday signal (documented limitation).

Weekly bars are built by resampling daily Close/High/Low to W-FRI:
  weekly_close = last daily close of the week (Friday, or last available
                 trading day if Friday is a holiday)
  weekly_high  = max daily high over the week
  weekly_low   = min daily low over the week
"""

import pandas as pd
import numpy as np


# ── Weekly indicator computation ─────────────────────────────────────────────
def _weekly_indicators(daily_close, daily_high, daily_low, config):
    """Resample to weekly (W-FRI) and compute all MILT 25 indicators.
    Returns a dict of DataFrames (index=weekly Friday dates, columns=tickers).
    """
    bb_period   = config.get('bb_period', 20)
    bb_std      = config.get('bb_std', 3.7)
    exit_ma     = config.get('exit_ma_period', 23)
    atr_period  = config.get('atr_period', 14)
    atr_mult    = config.get('atr_multiplier', 1.8)
    roc_period  = config.get('roc_period_weeks', 52)

    w_close = daily_close.resample('W-FRI').last()
    w_high  = daily_high.resample('W-FRI').max()
    w_low   = daily_low.resample('W-FRI').min()
    w_high  = w_high.combine_first(w_close)   # guard against all-NaN weeks for thin tickers
    w_low   = w_low.combine_first(w_close)

    # Bollinger upper band
    basis  = w_close.rolling(bb_period).mean()
    stdev  = w_close.rolling(bb_period).std()
    bb_upper = basis + bb_std * stdev

    # Exit MA
    exit_sma = w_close.rolling(exit_ma).mean()

    # Weekly ATR (Wilder-style true range on weekly bars, simple rolling mean)
    prev_close = w_close.shift(1)
    tr1 = (w_high - w_low).abs()
    tr2 = (w_high - prev_close).abs()
    tr3 = (w_low - prev_close).abs()
    tr = np.maximum(np.maximum(tr1, tr2), tr3)   # elementwise max, same shape/index as w_close
    atr = tr.rolling(atr_period).mean()

    roc_12m = (w_close - w_close.shift(roc_period)) / w_close.shift(roc_period) * 100

    return {
        'w_close': w_close, 'w_high': w_high, 'w_low': w_low,
        'bb_upper': bb_upper, 'exit_sma': exit_sma, 'atr': atr,
        'roc_12m': roc_12m,
    }


def _next_trading_day_price(daily_close, all_days, friday_date):
    """Approximate 'Monday Open' execution price: Close of the next trading
    day strictly after `friday_date`. Returns (exec_date, price_row) or
    (None, None) if no further trading days exist."""
    later = all_days[all_days > friday_date]
    if len(later) == 0:
        return None, None
    exec_date = later[0]
    return exec_date, daily_close.loc[exec_date]


# ── Main backtest loop ───────────────────────────────────────────────────────
def run_milt25_backtest(raw_fields, config, mcap_data, start_date, end_date,
                        initial_capital, verbose=True):
    """
    raw_fields: {'Close': df, 'High': df, 'Low': df, 'Volume': df} — daily,
                date-indexed, ticker columns (from core.history_store.load_history)
    mcap_data:  {ticker: mcap_in_cr} — latest snapshot, used as a static
                universe filter (point-in-time MCap history not stored;
                same simplification core/backtest_engine.py already makes)
    Returns (portfolio_df, trades_df, snapshots_df) — same shape/columns
    convention as core.backtest_engine.run_backtest() so run_backtest.py's
    existing Supabase push logic works unchanged.
    """
    daily_close = raw_fields['Close']
    daily_high  = raw_fields.get('High', daily_close)
    daily_low   = raw_fields.get('Low',  daily_close)
    all_days    = daily_close.index

    min_mcap        = config.get('min_mcap', 1000)
    max_positions   = config.get('portfolio_size', 25)
    alloc_pct       = config.get('alloc_pct', 0.04)
    hard_stop_pct   = config.get('hard_stop_pct', 0.20)

    eligible_universe = {t for t, m in mcap_data.items()
                         if m is not None and m > min_mcap and t in daily_close.columns}

    if verbose:
        print(f"MILT 25: {len(eligible_universe)} tickers eligible (MCap > ₹{min_mcap} Cr)")

    ind = _weekly_indicators(daily_close, daily_high, daily_low, config)
    w_close, w_high, bb_upper, exit_sma, atr, roc_12m = (
        ind['w_close'], ind['w_high'], ind['bb_upper'], ind['exit_sma'],
        ind['atr'], ind['roc_12m']
    )

    weekly_dates = w_close.index
    weekly_dates = weekly_dates[(weekly_dates >= pd.Timestamp(start_date)) &
                                (weekly_dates <= pd.Timestamp(end_date))]

    capital  = float(initial_capital)
    holdings = {}   # ticker -> {'shares': int, 'entry_price': float, 'highest_high': float}
    trade_log, portfolio_values, snapshots = [], [], []

    for i, friday in enumerate(weekly_dates):
        label = friday.strftime('%d %b %Y')

        # Update running highest-high-since-entry for all current holdings
        for ticker, pos in holdings.items():
            h = w_high[ticker].get(friday, np.nan) if ticker in w_high.columns else np.nan
            if pd.notna(h):
                pos['highest_high'] = max(pos['highest_high'], h)

        # ── Exit checks (evaluated on Friday close) ──────────────────────────
        to_sell = []
        for ticker, pos in holdings.items():
            if ticker not in w_close.columns:
                continue
            c = w_close[ticker].get(friday, np.nan)
            if pd.isna(c):
                continue
            hard_stop = pos['entry_price'] * (1 - hard_stop_pct)
            ma23      = exit_sma[ticker].get(friday, np.nan)
            a         = atr[ticker].get(friday, np.nan)
            trail     = (pos['highest_high'] - config.get('atr_multiplier', 1.8) * a
                        if pd.notna(a) else np.nan)

            reason = None
            if c < hard_stop:
                reason = 'STOP_LOSS'
            elif pd.notna(ma23) and c < ma23:
                reason = 'MA_EXIT'
            elif pd.notna(trail) and c < trail:
                reason = 'ATR_TRAIL'
            if reason:
                to_sell.append((ticker, reason))

        # Execute sells at next trading day's close (Monday Open proxy)
        for ticker, reason in to_sell:
            exec_date, price_row = _next_trading_day_price(daily_close, all_days, friday)
            if exec_date is None:
                continue
            p = price_row.get(ticker, np.nan)
            if pd.isna(p):
                p = w_close[ticker].get(friday, np.nan)  # fallback to signal-week close
            if pd.notna(p):
                pos = holdings[ticker]
                capital += pos['shares'] * p * (1 - config.get('cost_sell', 0.001))
                trade_log.append({'date': exec_date, 'ticker': ticker, 'action': f'SELL_{reason}',
                                  'price': p, 'shares': pos['shares']})
            del holdings[ticker]

        # ── Entry signals (evaluated on Friday close) ────────────────────────
        candidates = []
        for ticker in eligible_universe:
            if ticker in holdings or ticker not in w_close.columns:
                continue
            c  = w_close[ticker].get(friday, np.nan)
            bb = bb_upper[ticker].get(friday, np.nan)
            if pd.notna(c) and pd.notna(bb) and c > bb:
                r = roc_12m[ticker].get(friday, np.nan)
                candidates.append((ticker, r if pd.notna(r) else -1e9))

        free_slots = max_positions - len(holdings)
        if free_slots > 0 and candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            to_buy = [t for t, _ in candidates[:free_slots]]
        else:
            to_buy = []

        # Mark-to-market equity (for 4%-of-current-equity sizing) using
        # Friday close, before executing this week's new buys
        equity_now = capital + sum(
            holdings[t]['shares'] * w_close[t].get(friday, holdings[t]['entry_price'])
            for t in holdings
        )

        for ticker in to_buy:
            exec_date, price_row = _next_trading_day_price(daily_close, all_days, friday)
            if exec_date is None:
                continue
            p = price_row.get(ticker, np.nan)
            if pd.isna(p):
                continue
            alloc = equity_now * alloc_pct
            shares = int(alloc * (1 - config.get('cost_buy', 0.001)) / p)
            if shares <= 0:
                continue
            cost = shares * p * (1 + config.get('cost_buy', 0.001))
            if cost > capital:
                continue
            capital -= cost
            holdings[ticker] = {'shares': shares, 'entry_price': p, 'highest_high': p}
            trade_log.append({'date': exec_date, 'ticker': ticker, 'action': 'BUY',
                              'price': p, 'shares': shares})

        # ── Snapshot at Friday close (mark-to-market, post this week's trades) ──
        portfolio_value = capital + sum(
            holdings[t]['shares'] * w_close[t].get(friday, holdings[t]['entry_price'])
            for t in holdings
        )
        portfolio_values.append({'date': friday, 'value': portfolio_value})
        snapshots.append({
            'date': friday,
            'portfolio_value': portfolio_value,
            'stocks_screened': len(eligible_universe),
            'slots_used': len(holdings),
            'cash_slots': max_positions - len(holdings),
            'in_cash': len(holdings) == 0,
            'cash_balance': capital,
            'top_picks': list(holdings.keys()),
            'n_buys': len(to_buy),
            'n_sells': len(to_sell),
        })

        if verbose and (i % 26 == 0 or i == len(weekly_dates) - 1):
            print(f'{label} [{i+1}/{len(weekly_dates)}] positions={len(holdings)} '
                  f'buys={len(to_buy)} sells={len(to_sell)} value={portfolio_value:,.0f}')

    portfolio_df = pd.DataFrame(portfolio_values).set_index('date')
    trades_df    = pd.DataFrame(trade_log)
    snapshots_df = pd.DataFrame(snapshots).set_index('date')

    if verbose:
        print(f"\nMILT 25 complete: {len(snapshots_df)} weekly periods, "
              f"{len(trades_df)} trades, final value {portfolio_df['value'].iloc[-1]:,.0f}")

    return portfolio_df, trades_df, snapshots_df
