"""
Strategy 01 — SMA Momentum (Baseline)
======================================
Identical to the live NSE_1000Cr_Momentum screener.

Filters  : MCap > 1000Cr, ADV > 10Cr, Volatility < 75%, RSI14 > 50,
           Price within 25% of 52W high, Price > SMA21*(1-5% buffer), CMF > 0.1
Rank by  : SMA21 / SMA200  (higher = stronger trend)
Rebalance: weekly or monthly (set via BT_PARAMS)

no_trim  : if True, winning positions are NOT trimmed back to equal weight
           at each rebalance — they run freely until they exit the funnel.
"""

STRATEGY_ID   = "s01_sma_momentum"
STRATEGY_NAME = "SMA Momentum"

def get_config_overrides():
    return {}   # no_trim is passed directly from GUI via BT_PARAMS

def rank_fn(ind, idx, tickers):
    """Rank by SMA21/SMA200 ratio. Higher = stronger trend."""
    import pandas as pd
    sma_s = ind["sma_short"].iloc[idx]
    sma_l = ind["sma_long"].iloc[idx].replace(0, float("nan"))
    return (sma_s / sma_l).reindex(tickers)
