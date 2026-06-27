"""
Strategy 02 — RSI Composite Rank
==================================
Same filter funnel as Strategy 01 EXCEPT the RSI>50 filter is removed.
Rank by  : mean( RSI_14, RSI_22, RSI_72 )  — multi-timeframe momentum strength.
           Short (14d) = recent burst, Mid (22d) = swing, Long (72d) = trend.
           Higher composite = stronger momentum across all horizons.

Rationale: SMA21/SMA200 rewards stocks that are already well above their
long-run average but may be extended. RSI composite rewards stocks with
sustained buying pressure across timeframes without the trend-extension bias.
The RSI>50 filter is removed because the rank score itself is RSI-based —
there is no need to double-count it as both a filter and a rank.
"""

STRATEGY_ID   = "s02_rsi_composite"
STRATEGY_NAME = "RSI Composite Rank (14/22/72)"

def get_config_overrides():
    return {
        "skip_filters": ["rsi"],     # remove RSI>50 gate; rank handles it
        "rsi_period":      14,
        "rsi_period_mid":  22,
        "rsi_period_long": 72,
    }

def rank_fn(ind, idx, tickers):
    """
    Rank by equal-weight average of RSI_14, RSI_22, RSI_72.
    Falls back gracefully if any period is missing (uses available ones).
    """
    import pandas as pd
    import numpy as np
    components = []
    for key in ("rsi", "rsi_mid", "rsi_long"):
        if key in ind:
            components.append(ind[key].iloc[idx])
    if not components:
        raise KeyError("No RSI components found in ind — check config has rsi_period_mid/long")
    stacked = pd.concat(components, axis=1)
    return stacked.mean(axis=1).reindex(tickers)
