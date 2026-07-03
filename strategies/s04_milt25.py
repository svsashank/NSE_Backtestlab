"""
strategies/s04_milt25.py — MILT 25: weekly Bollinger breakout with
ATR trailing stop. Event-driven, no cross-sectional ranking (ROC_12M
used only as a tie-breaker when Buy signals exceed free slots).

Routed separately in run_backtest.py to core.milt25_engine — this
strategy does NOT use core.screener_engine / core.backtest_engine's
ranked top-N rebalance model.
"""

STRATEGY_ID   = "s04_milt25"
STRATEGY_NAME = "MILT 25 (Weekly BB Breakout + ATR Trail)"


def get_config_overrides():
    return {
        "portfolio_size": 25,       # max open positions
        "min_mcap": 1000,           # ₹ Cr — only filter applied to the universe
        "alloc_pct": 0.04,          # 4% of current equity per new position
        "hard_stop_pct": 0.20,      # -20% fixed stop from entry
        "bb_period": 20,
        "bb_std": 3.7,
        "exit_ma_period": 23,
        "atr_period": 14,
        "atr_multiplier": 1.8,
        "roc_period_weeks": 52,
        "cost_buy": 0.001,
        "cost_sell": 0.001,
    }


def rank_fn(ind, idx, tickers):
    # Not used — milt25_engine.py does its own ROC_12M tie-break internally.
    # Present only so load_strategy()/run_backtest.py's generic strategy
    # interface doesn't break if something calls it.
    import pandas as pd
    return pd.Series(0.0, index=tickers)
