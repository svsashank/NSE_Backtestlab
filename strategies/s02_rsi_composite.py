"""
Strategy 02 — RSI Composite Rank
==================================
Same filter funnel as Strategy 01 EXCEPT the RSI>50 filter is removed.
Rank by  : mean( RSI_14, RSI_22, RSI_72 )  — multi-timeframe momentum strength.
           Short (14d) = recent burst, Mid (22d) = swing, Long (72d) = trend.

no_trim  : if True, winning positions are NOT trimmed back to equal weight
           at each rebalance — they run freely until they exit the funnel.
"""

STRATEGY_ID   = "s02_rsi_composite"
STRATEGY_NAME = "RSI Composite"

def get_config_overrides():
    return {
        "skip_filters": ["rsi"],
        "rsi_period":      14,
        "rsi_period_mid":  22,
        "rsi_period_long": 72,
    }

def rank_fn(ind, idx, tickers):
    import pandas as pd
    components = []
    for key in ("rsi", "rsi_mid", "rsi_long"):
        if key in ind:
            components.append(ind[key].iloc[idx])
    if not components:
        raise KeyError("No RSI components found in ind")
    return pd.concat(components, axis=1).mean(axis=1).reindex(tickers)
