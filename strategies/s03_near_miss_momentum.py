"""
Strategy 03 — Near-Miss Momentum
==================================
Same filter funnel as Strategy 01 (SMA Momentum) with two key improvements:

1. Near-miss promotion: stocks in the top 50 by rank that fail exactly ONE
   filter by at most 10% of its threshold are promoted to fill Top 15 slots.
   Higher-ranked near-misses always beat lower-ranked strict passes.

2. Hold-zone anti-whipsaw: a stock is only sold when it falls out of the
   top 25 eligible stocks (same near-miss rules). Normal rank fluctuation
   within the top 25 does not trigger a sell.

Parameters (from live screener config):
    MCap   > 1000 Cr
    ADV    > 10 Cr
    Vol    < 80%  (0.8)
    RSI14  > 50
    52W high within 25%
    SMA21 buffer 5%
    CMF    > 0.05
    Portfolio size: 15
    Hold zone size: 25
"""

STRATEGY_ID   = "s03_near_miss_momentum"
STRATEGY_NAME = "Near-Miss Momentum"


def get_config_overrides():
    return {
        "max_volatility": 0.80,   # 80% (vs 75% baseline)
        "cmf_threshold" : 0.05,   # 0.05 (vs 0.1 baseline)
        "hold_zone_size": 25,     # sell only when stock exits top 25
        "retention_rank": 0,      # disable legacy retention, use hold zone
    }


def rank_fn(ind, idx, tickers):
    """Rank by SMA21/SMA200 ratio — identical to the live screener."""
    import pandas as pd
    sma_s = ind["sma_short"].iloc[idx]
    sma_l = ind["sma_long"].iloc[idx].replace(0, float("nan"))
    return (sma_s / sma_l).reindex(tickers)
