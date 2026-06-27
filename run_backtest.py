"""
run_backtest.py — Backtest Lab runner (GitHub Actions).

Reads OHLCV from Supabase Storage (same bucket as NSE_1000Cr_Momentum).
Dispatches to the strategy module named in STRATEGY env var.
Writes results to `backtest_results` table in Supabase.

Each strategy lives in strategies/sXX_name.py and defines:
  STRATEGY_ID          str   unique identifier
  STRATEGY_NAME        str   human-readable name
  get_config_overrides()     returns dict of config keys to override
  rank_fn(ind, idx, tickers) returns rank score pd.Series

To add a new strategy: create strategies/sXX_newname.py, done.
"""

import os, sys, json, time, math, importlib, warnings
from datetime import datetime
import numpy as np
import pandas as pd
from supabase import create_client

from core.history_store import load_history, fields_to_raw_multiindex
from core.indicators    import compute_indicators
from core.backtest_engine import get_rebalance_dates, run_backtest, compute_performance_stats

warnings.filterwarnings("ignore")

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]

CONFIG_FILE = os.path.join(os.path.dirname(__file__), "config_base.json")
with open(CONFIG_FILE) as f:
    BASE_CONFIG = json.load(f)

UNIVERSE_NAME = BASE_CONFIG["universe_name"]


def load_strategy(strategy_id):
    """Dynamically import the strategy module by ID."""
    import glob, os
    strategy_files = glob.glob(os.path.join(os.path.dirname(__file__), "strategies", "*.py"))
    for fpath in strategy_files:
        mod_name = os.path.basename(fpath).replace(".py", "")
        if mod_name == "__init__":
            continue
        mod = importlib.import_module(f"strategies.{mod_name}")
        if getattr(mod, "STRATEGY_ID", None) == strategy_id:
            return mod
    raise ValueError(f"Strategy '{strategy_id}' not found. "
                     f"Available: {[os.path.basename(f).replace('.py','') for f in strategy_files if 'init' not in f]}")


def clean(val):
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
        return None
    if isinstance(val, np.integer):  return int(val)
    if isinstance(val, np.floating):
        v = float(val)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(val, pd.Timestamp): return str(val.date())
    if isinstance(val, list):         return [clean(x) for x in val]
    return val


def df_to_records(df):
    if df is None or df.empty:
        return []
    return [{k: clean(v) for k, v in row.items()} for _, row in df.reset_index().iterrows()]


def main():
    t0 = time.time()
    print("="*60)
    print("  NSE BACKTEST LAB — GitHub Actions")
    print(f"  {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    print("="*60)

    # ── Parse BT_PARAMS ──────────────────────────────────────────────────────
    params = {}
    params_raw = os.environ.get("BT_PARAMS", "")
    if params_raw:
        try:
            decoded = json.loads(params_raw)
            params  = json.loads(decoded) if isinstance(decoded, str) else decoded
        except Exception as e:
            print(f"⚠ BT_PARAMS parse error: {e}")

    strategy_id    = params.get("strategy_id") or os.environ.get("STRATEGY", "s01_sma_momentum")
    rebalance_type = params.get("rebalance_type", "monthly")
    initial_capital= float(params.get("initial_capital", 1_000_000))
    start_date     = params.get("start_date") or None
    end_date       = params.get("end_date")   or None

    print(f"Strategy : {strategy_id}")
    print(f"Rebalance: {rebalance_type} | Capital: {initial_capital:,.0f}")

    # ── Load strategy ─────────────────────────────────────────────────────────
    strategy = load_strategy(strategy_id)
    print(f"Loaded   : {strategy.STRATEGY_NAME}")

    config = dict(BASE_CONFIG)
    config.update(strategy.get_config_overrides())
    config["rank_fn"] = strategy.rank_fn       # inject callable — not serialisable

    # Allow GUI to override numeric params (on top of strategy overrides)
    int_keys = ["portfolio_size", "sma_short", "sma_long", "min_stocks_to_invest", "retention_rank"]
    skip_keys = {"strategy_id", "rebalance_type", "initial_capital", "start_date", "end_date"}
    for k, v in params.items():
        if k in skip_keys or v is None or v == "":
            continue
        if k in config and k not in ("rank_fn", "skip_filters", "anchors"):
            config[k] = int(v) if k in int_keys else float(v)

    # ── Load OHLCV history ────────────────────────────────────────────────────
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    print("\nLoading OHLCV from Supabase Storage...")
    history = load_history(supabase, UNIVERSE_NAME)
    if history is None:
        raise RuntimeError(f"No stored history for universe '{UNIVERSE_NAME}'. Run refresh_history.py in NSE_1000Cr_Momentum first.")

    raw     = fields_to_raw_multiindex(history)
    tickers = history["Close"].columns.tolist()
    print(f"   {len(tickers)} tickers, {history['Close'].index[0].date()} → {history['Close'].index[-1].date()}")

    # MCap passthrough: use a large constant so mcap filter never cuts backtest stocks
    # (point-in-time mcap history not yet stored)
    mcap_data = {t: max(config["min_mcap"] * 10, 1e6) for t in tickers}

    # ── Compute indicators ────────────────────────────────────────────────────
    print("\nComputing indicators...")
    ind = compute_indicators(raw, mcap_data, tickers, config)

    # ── Date window ───────────────────────────────────────────────────────────
    full_start = ind["close"].index[0]
    full_end   = ind["close"].index[-1]
    bt_start   = max(pd.Timestamp(start_date) if start_date else full_start, full_start)
    bt_end     = min(pd.Timestamp(end_date)   if end_date   else full_end,   full_end)

    print(f"\nBacktest: {bt_start.date()} → {bt_end.date()} ({rebalance_type})")
    rebalance_dates = get_rebalance_dates(bt_start, bt_end, ind["close"].index, rebalance_type)
    print(f"{len(rebalance_dates)} rebalance dates")

    if len(rebalance_dates) < 2:
        raise RuntimeError("Fewer than 2 rebalance dates — widen the date range.")

    # ── Run backtest ──────────────────────────────────────────────────────────
    portfolio_df, trades_df, snapshots_df = run_backtest(
        ind, config, rebalance_dates, initial_capital, verbose=True
    )

    stats = compute_performance_stats(portfolio_df, rebalance_type=rebalance_type,
                                      risk_free_rate=config.get("risk_free_rate", 0.0))

    print("\nPerformance:")
    for k, v in stats.items():
        print(f"   {k}: {v:.4f}" if isinstance(v, float) else f"   {k}: {v}")

    # ── Push to Supabase ──────────────────────────────────────────────────────
    row = {
        "strategy_id"    : strategy_id,
        "strategy_name"  : strategy.STRATEGY_NAME,
        "rebalance_freq" : rebalance_type,
        "date_range_start": str(bt_start.date()),
        "date_range_end"  : str(bt_end.date()),
        "initial_capital" : initial_capital,
        "params"          : {k: clean(v) for k, v in config.items()
                             if not k.startswith("_") and k != "rank_fn" and not callable(v)},
        "performance"     : {k: clean(v) for k, v in stats.items()},
        "equity_curve"    : df_to_records(portfolio_df.reset_index().rename(columns={"index": "date"})),
        "trades"          : df_to_records(trades_df),
        "snapshots"       : df_to_records(snapshots_df.reset_index().drop(columns=["top_picks"], errors="ignore")),
    }

    print("\nPushing to Supabase...")
    resp   = supabase.table("backtest_results").insert(row).execute()
    run_id = resp.data[0]["id"] if resp.data else None
    print(f"✅ backtest_results → id: {run_id}")
    print(f"\nDone in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
