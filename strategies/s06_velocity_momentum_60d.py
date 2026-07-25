"""
Strategy 06 — Velocity Momentum (60-day lookback)
====================================================
Same filter funnel + near-miss/hold-zone logic as Strategy 03
(Near-Miss Momentum), but ranks stocks by RANK VELOCITY instead of
raw SMA21/SMA200 level — mirrors the `rank_velocity` feature added to
the live NSE_1000Cr_Momentum screener.

Rank velocity definition (matches live screener):
    1. Take each ticker's SMA21/SMA200 ratio ("rank_score") over the
       trailing 60 trading days ending at the current screen date
       (variant of s05, testing whether a longer window behaves more
       like durable trend acceleration and less like short-term noise).
    2. Convert each day's cross-sectional rank_score into a PERCENTILE
       RANK (0-1) across all tickers with valid data that day.
       (Percentile, not raw rank, so universe size drift doesn't bias
       the slope — same principle as the live velocity metric.)
    3. Fit a Theil-Sen slope of percentile rank vs. day-index (0..59)
       per ticker. This is robust to single-day outliers/gaps, same
       estimator used in rank_momentum.py.
    4. Rank the portfolio by this slope, descending — i.e. stocks
       climbing the rank table fastest win, regardless of where they
       currently sit in absolute terms.

Note: this is a deliberate deviation from the live 20-run velocity
window, to test whether a slower, less noisy slope estimate performs
better as a primary rank than the 20-day version (s05).

Parameters: identical filter thresholds to Strategy 03 (Near-Miss
Momentum) so the only variable being tested is the ranking metric.
"""

import numpy as np
import pandas as pd

STRATEGY_ID   = "s06_velocity_momentum_60d"
STRATEGY_NAME = "Velocity Momentum (60d)"

LOOKBACK_DAYS = 60   # trading days — 3x the live window, testing a slower slope


def get_config_overrides():
    return {
        "max_volatility": 0.80,   # 80% (vs 75% baseline) — same as s03
        "cmf_threshold" : 0.05,   # 0.05 (vs 0.1 baseline) — same as s03
        "hold_zone_size": 25,     # sell only when stock exits top 25
        "retention_rank": 0,      # disable legacy retention, use hold zone
    }


def rank_fn(ind, idx, tickers):
    """Rank by Theil-Sen slope of trailing 20-day percentile rank."""
    from scipy.stats import theilslopes

    rank_score = ind["rank_score"]  # daily DataFrame, sma_short/sma_long

    lo = max(0, idx - LOOKBACK_DAYS + 1)
    window = rank_score.iloc[lo:idx + 1]

    # Not enough history yet — fall back to raw SMA ratio so early
    # backtest periods still produce a portfolio instead of empty.
    if len(window) < 5:
        sma_s = ind["sma_short"].iloc[idx]
        sma_l = ind["sma_long"].iloc[idx].replace(0, float("nan"))
        return (sma_s / sma_l).reindex(tickers)

    # Cross-sectional percentile rank each day (0-1), NaNs excluded.
    pct_rank = window.rank(axis=1, pct=True, na_option="keep")

    n_days = len(pct_rank)
    x = np.arange(n_days)

    slopes = {}
    for t in tickers:
        y = pct_rank[t].values if t in pct_rank.columns else None
        if y is None:
            slopes[t] = np.nan
            continue
        mask = ~np.isnan(y)
        if mask.sum() < 5:
            slopes[t] = np.nan
            continue
        try:
            slope, _, _, _ = theilslopes(y[mask], x[mask])
            slopes[t] = slope
        except Exception:
            slopes[t] = np.nan

    result = pd.Series(slopes).reindex(tickers)

    # Any ticker still NaN (e.g. brand new listing) ranks last, not
    # dropped — matches "missing runs = missing data, not rank crash".
    if result.isna().any():
        result = result.fillna(result.min(skipna=True) - 1e-9)

    return result
