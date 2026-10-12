"""
core/trade_metrics.py — Churn, holding-period, whipsaw and tax metrics.

Computed post-hoc from the trade log + equity curves produced by
core.backtest_engine.run_backtest(). Nothing here changes the simulation;
it only measures it, so every strategy is judged on the same yardstick.

Metrics
-------
daily_cagr / daily_max_drawdown / daily_ann_vol / daily_sharpe
    From the daily mark-to-market curve (portfolio_df.attrs['daily']).
    Use these to compare across rebalance frequencies — the per-rebalance
    curve under-samples drawdowns for monthly runs.
turnover_annual
    One-way turnover: total sell-side value (SELL + TRIM) per year divided
    by average portfolio value. 1.0 = the whole book replaced once a year.
n_buys, n_sells
avg_hold_days / median_hold_days
    Calendar days from BUY to full exit, over closed positions.
pct_held_over_1y
    Share of closed positions held > 365 days (LTCG eligible).
whipsaw_reentries / whipsaw_rate
    BUYs of a ticker that was SOLD within the previous `whipsaw_days`
    calendar days (default 90). Rate = re-entries / total BUYs.
tax_paid_total / after_tax_cagr
    Indian listed-equity capital-gains tax at CURRENT rates applied to the
    whole history for comparability: STCG 20% (held <= 365 days), LTCG
    12.5% above a Rs 1.25 lakh annual exemption. Netted per financial year
    (Apr–Mar); ST losses offset ST then LT gains, LT losses offset LT gains
    only; unused losses carry forward. Tax is assumed paid at each FY end,
    so its cost includes the growth it forgoes afterwards. Unrealised gains
    at the end are not taxed. Simplification: ignores grandfathering,
    surcharge/cess, and STT (STT is inside cost_buy/cost_sell).
"""

import numpy as np
import pandas as pd

STCG_RATE = 0.20
LTCG_RATE = 0.125
LTCG_EXEMPT = 125_000.0


def _fy_end(d):
    """Indian financial year end (31 Mar) for a date."""
    y = d.year + (1 if d.month > 3 else 0)
    return pd.Timestamp(year=y, month=3, day=31)


def daily_stats(daily, risk_free_rate=0.0):
    if daily is None or len(daily) < 3:
        return {}
    v = daily.astype(float)
    r = v.pct_change().dropna()
    yrs = (v.index[-1] - v.index[0]).days / 365.25
    cagr = (v.iloc[-1] / v.iloc[0]) ** (1 / yrs) - 1 if yrs > 0 else np.nan
    vol = r.std() * np.sqrt(252)
    dd = (v / v.cummax() - 1).min()
    return {
        'daily_cagr': cagr,
        'daily_ann_vol': vol,
        'daily_sharpe': (cagr - risk_free_rate / 100) / vol if vol > 0 else np.nan,
        'daily_max_drawdown': dd,
        'calmar': cagr / abs(dd) if dd < 0 else np.nan,
    }


def trade_stats(trades_df, portfolio_df, cost_buy=0.0, cost_sell=0.0, whipsaw_days=90):
    out = {}
    if trades_df is None or trades_df.empty:
        return out

    t = trades_df.copy()
    t['date'] = pd.to_datetime(t['date'])
    t = t[t['action'].isin(['BUY', 'SELL', 'TRIM', 'SELL_CASH', 'SELL_CAP'])].sort_values('date', kind='stable')
    sells = t[t['action'] != 'BUY']

    values = portfolio_df['value'].astype(float)
    yrs = max((values.index[-1] - values.index[0]).days / 365.25, 1e-9)
    sell_value = float((sells['price'] * sells['shares']).sum())
    out['turnover_annual'] = sell_value / float(values.mean()) / yrs
    out['n_buys'] = int((t['action'] == 'BUY').sum())
    out['n_sells'] = int((t['action'].isin(['SELL', 'SELL_CASH', 'SELL_CAP'])).sum())

    # ── Walk positions: holding periods, whipsaws, realised gains ────────────
    open_pos = {}           # ticker -> {'shares', 'cost', 'entry'}
    last_exit = {}          # ticker -> date of last full exit
    holds, reentries = [], 0
    realised = []           # (date, gain, is_long_term)

    for _, row in t.iterrows():
        tk, act, d, px, sh = row['ticker'], row['action'], row['date'], float(row['price']), float(row['shares'])
        if act == 'BUY':
            le = last_exit.get(tk)
            if le is not None and (d - le).days <= whipsaw_days:
                reentries += 1
            open_pos[tk] = {'shares': sh, 'cost': px * (1 + cost_buy), 'entry': d}
            continue
        pos = open_pos.get(tk)
        if pos is None:
            continue
        sh = min(sh, pos['shares'])
        gain = sh * (px * (1 - cost_sell) - pos['cost'])
        realised.append((d, gain, (d - pos['entry']).days > 365))
        pos['shares'] -= sh
        if act != 'TRIM' or pos['shares'] <= 0:
            holds.append((d - pos['entry']).days)
            last_exit[tk] = d
            del open_pos[tk]

    out['whipsaw_reentries'] = reentries
    out['whipsaw_rate'] = reentries / out['n_buys'] if out['n_buys'] else np.nan
    if holds:
        out['avg_hold_days'] = float(np.mean(holds))
        out['median_hold_days'] = float(np.median(holds))
        out['pct_held_over_1y'] = float(np.mean(np.array(holds) > 365))

    # ── Tax by financial year ────────────────────────────────────────────────
    by_fy = {}
    for d, g, lt in realised:
        fy = _fy_end(d)
        st_g, lt_g = by_fy.get(fy, (0.0, 0.0))
        by_fy[fy] = (st_g + (0 if lt else g), lt_g + (g if lt else 0))

    st_cf = lt_cf = 0.0     # carried-forward losses (positive numbers)
    taxes = {}
    for fy in sorted(by_fy):
        st, lt = by_fy[fy]
        st -= st_cf; st_cf = 0.0
        lt -= lt_cf; lt_cf = 0.0
        if st < 0:              # ST loss offsets LT gain
            if lt > 0:
                use = min(-st, lt); lt -= use; st += use
            st_cf = -st if st < 0 else 0.0
            st = max(st, 0.0)
        if lt < 0:
            lt_cf = -lt; lt = 0.0
        taxes[fy] = STCG_RATE * st + LTCG_RATE * max(lt - LTCG_EXEMPT, 0.0)

    end_val = float(values.iloc[-1])
    lost = 0.0
    for fy, tax in taxes.items():
        if tax <= 0:
            continue
        pay = min(fy, values.index[-1])
        v_at = float(values[values.index <= pay].iloc[-1]) if (values.index <= pay).any() else float(values.iloc[0])
        lost += tax * (end_val / v_at)          # tax plus the growth it forgoes
    out['tax_paid_total'] = float(sum(taxes.values()))
    after = max(end_val - lost, 1e-9)
    out['after_tax_cagr'] = (after / float(values.iloc[0])) ** (1 / yrs) - 1
    out['tax_drag_cagr'] = None   # filled by caller once pre-tax CAGR is known
    return out


def all_metrics(portfolio_df, trades_df, config):
    daily = portfolio_df.attrs.get('daily')
    m = daily_stats(daily, config.get('risk_free_rate', 0.0))
    m.update(trade_stats(trades_df, portfolio_df,
                         config.get('cost_buy', 0.0), config.get('cost_sell', 0.0),
                         int(config.get('whipsaw_days', 90))))
    values = portfolio_df['value'].astype(float)
    yrs = (values.index[-1] - values.index[0]).days / 365.25
    pre = (values.iloc[-1] / values.iloc[0]) ** (1 / yrs) - 1 if yrs > 0 else np.nan
    if 'after_tax_cagr' in m:
        m['tax_drag_cagr'] = pre - m['after_tax_cagr']
    return m
