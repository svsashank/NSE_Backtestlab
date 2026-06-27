# NSE Backtest Lab

Independent backtest lab for NSE strategies. Reads OHLCV from the same
Supabase Storage as NSE_1000Cr_Momentum — no data duplication.

## Adding a new strategy

1. Create `strategies/sXX_yourname.py` with:
   - `STRATEGY_ID = "sXX_yourname"`
   - `STRATEGY_NAME = "Human readable name"`
   - `get_config_overrides()` → dict of config keys to change from base
   - `rank_fn(ind, idx, tickers)` → pd.Series of rank scores

2. Trigger the workflow with `strategy_id = "sXX_yourname"`. Done.

## Strategies

| ID | Name | Rank By | Filters removed |
|---|---|---|---|
| s01_sma_momentum | SMA Momentum (Baseline) | SMA21/SMA200 | None |
| s02_rsi_composite | RSI Composite Rank (14/22/72) | avg(RSI14, RSI22, RSI72) | RSI>50 |

## Config (config_base.json)

Shared params: MCap>1000Cr, ADV>10Cr, Vol<75%, RSI14>50, within 25% of 52W high,
price>SMA21*(1-5% buffer), CMF>0.1. Portfolio size: 15. Costs: 0.1% each side.

## Results table: `backtest_results` in Supabase

Columns: strategy_id, strategy_name, rebalance_freq, date_range_start/end,
initial_capital, params, performance, equity_curve, trades, snapshots.
