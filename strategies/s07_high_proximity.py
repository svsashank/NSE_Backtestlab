"""
Strategy 07 — 52-Week-High Proximity Blend
==========================================
Same filter funnel + near-miss/hold-zone logic as Strategy 03 (the live
screener), but the RANK is a blend that tilts toward stocks approaching or
breaching their 52-week high, and away from stocks drifting off it.

Intent: own names that are getting closer to / breaking through their
52W high; leave names that are moving away from it — without adding churn.
Grounding: George & Hwang (2004), nearness to the 52W high predicts returns
(anchoring on the old high delays price discovery until it breaks).

Score (each component is a cross-sectional PERCENTILE, 0–1, so the
weights mean the same thing regardless of universe size):

    score = w_sma  · pct(SMA21 / SMA200)            trend strength (baseline)
          + w_near · pct(−dist)                      how close to the 52W high
          + w_appr · pct(−slope(dist, N days))       approaching (+) vs leaving (−)

    dist  = (52W high − close) / ATR(14)
            Distance measured in ATRs, so a 5% gap in a quiet large-cap and a
            5% gap in a volatile small-cap aren't treated as the same thing.
            0 = sitting at / making a new high.
    slope = OLS slope of dist over the last `hp_slope_days` sessions.
            Negative = gap closing (approaching the high) → scores higher.

    w_sma = 1 − w_near − w_appr

Anti-churn: the blended score is averaged over the last `hp_smooth_days`
sessions before ranking, so one noisy day can't flip a hold into a sell.
The top-25 hold zone from s03 still applies.

Tunable via BT_PARAMS (all optional):
    hp_w_near       weight on nearness            (default 0.30)
    hp_w_appr       weight on approach slope      (default 0.10)
    hp_slope_days   slope lookback, sessions      (default 20)
    hp_smooth_days  score smoothing, sessions     (default 5)
    hp_atr_period   ATR period                    (default 14)
    hold_zone_size  sell only when outside top-N  (default 25)
    trim_band       trim only when > slot·(1+x)   (default 0 = engine default)
    run_label       tag for diagnostic_output filename
"""

import numpy as np
import pandas as pd

STRATEGY_ID   = "s07_high_proximity"
STRATEGY_NAME = "52W-High Proximity Blend"

P = {
    "hp_w_near": 0.30,
    "hp_w_appr": 0.10,
    "hp_slope_days": 20,
    "hp_smooth_days": 5,
    "hp_atr_period": 14,
}


def get_config_overrides():
    o = {
        "max_volatility": 0.80,   # same as s03 / live screener
        "cmf_threshold" : 0.05,
        "hold_zone_size": 25,
        "retention_rank": 0,
    }
    o.update(P)
    return o


def configure(config):
    """Called by run_backtest.py after BT_PARAMS overrides are applied."""
    for k in P:
        if k in config and config[k] is not None:
            P[k] = config[k]
    for k in ("hp_slope_days", "hp_smooth_days", "hp_atr_period"):
        P[k] = int(P[k])
    w_near, w_appr = float(P["hp_w_near"]), float(P["hp_w_appr"])
    if w_near < 0 or w_appr < 0 or w_near + w_appr > 1:
        raise ValueError(f"Invalid weights: w_near={w_near}, w_appr={w_appr}")
    print(f"   s07 weights: sma={1 - w_near - w_appr:.2f} near={w_near:.2f} appr={w_appr:.2f} | "
          f"slope={P['hp_slope_days']}d smooth={P['hp_smooth_days']}d atr={P['hp_atr_period']}")


def _pct(df):
    """Row-wise percentile rank (0–1); missing values get a neutral 0.5."""
    return df.rank(axis=1, pct=True, na_option="keep").fillna(0.5)


def _rolling_slope(df, n):
    """OLS slope of each column over a trailing n-row window (NaN if incomplete)."""
    x = np.arange(n, dtype=float)
    xc = x - x.mean()
    denom = (xc ** 2).sum()
    vals = df.values
    out = np.full(vals.shape, np.nan)
    for i in range(n - 1, len(vals)):
        w = vals[i - n + 1:i + 1]
        out[i] = (xc @ (w - w.mean(axis=0))) / denom
    return pd.DataFrame(out, index=df.index, columns=df.columns)


def components(ind, idx, tickers, n_days=None):
    """Return dict of daily component frames for the last `n_days` sessions
    ending at idx (exposed separately so diagnostics can inspect them)."""
    slope_n, smooth_n, atr_n = P["hp_slope_days"], P["hp_smooth_days"], P["hp_atr_period"]
    n_days = n_days or smooth_n
    need = n_days + slope_n + atr_n * 3 + 5
    lo = max(0, idx - need + 1)
    sl = slice(lo, idx + 1)

    cols = [t for t in tickers if t in ind["close"].columns]
    close = ind["close"].iloc[sl][cols]
    high  = ind["high"].iloc[sl][cols]
    low   = ind["low"].iloc[sl][cols]
    h52   = ind["high_52w"].iloc[sl][cols]
    ratio = ind["rank_score"].iloc[sl][cols]

    prev = close.shift(1)
    tr = np.maximum(high - low, np.maximum((high - prev).abs(), (low - prev).abs()))
    atr = tr.ewm(alpha=1 / atr_n, min_periods=atr_n).mean()

    dist = ((h52 - close) / atr.replace(0, np.nan)).clip(lower=0)
    slope = _rolling_slope(dist, slope_n)

    tail = slice(-n_days, None)
    return {"ratio": ratio.iloc[tail], "dist": dist.iloc[tail], "slope": slope.iloc[tail]}


def rank_fn(ind, idx, tickers):
    w_near, w_appr = float(P["hp_w_near"]), float(P["hp_w_appr"])
    w_sma = 1.0 - w_near - w_appr

    c = components(ind, idx, tickers)
    score = (w_sma  * _pct(c["ratio"])
           + w_near * _pct(-c["dist"])
           + w_appr * _pct(-c["slope"]))

    # No SMA200 yet → not rankable (screener already marks it invalid)
    score = score.where(c["ratio"].notna())
    smoothed = score.mean(axis=0, skipna=True)
    return smoothed.reindex(tickers)


def diagnostics(ind, dates):
    """How much does nearness add beyond the SMA ratio? Spearman rank
    correlation between SMA ratio and −dist (and −slope) across stocks
    within 25% of their 52W high, averaged over the rebalance dates."""
    rho_near, rho_appr = [], []
    for d in dates:
        idx = ind["close"].index.get_indexer([d], method="ffill")[0]
        if idx < 0:
            continue
        close, h52 = ind["close"].iloc[idx], ind["high_52w"].iloc[idx]
        elig = (close >= h52 * 0.75) & ind["rank_score"].iloc[idx].notna()
        tick = elig[elig].index.tolist()
        if len(tick) < 30:
            continue
        c = components(ind, idx, tick, n_days=1)
        df = pd.DataFrame({"r": c["ratio"].iloc[-1], "n": -c["dist"].iloc[-1],
                           "a": -c["slope"].iloc[-1]}).dropna()
        if len(df) < 30:
            continue
        rk = df.rank()
        rho_near.append(rk["r"].corr(rk["n"]))
        rho_appr.append(rk["r"].corr(rk["a"]))
    return {
        "spearman_sma_vs_near_mean": float(np.mean(rho_near)) if rho_near else None,
        "spearman_sma_vs_appr_mean": float(np.mean(rho_appr)) if rho_appr else None,
        "corr_dates": len(rho_near),
    }
