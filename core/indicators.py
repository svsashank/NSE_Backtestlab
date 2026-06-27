"""
core/indicators.py — Shared, market-agnostic indicator computation.

Extended vs NSE_1000Cr_Momentum to support multi-period RSI for backtest
strategies. Computes rsi (period = rsi_period), rsi_mid (rsi_period_mid),
rsi_long (rsi_period_long). If the extra periods are not in config,
rsi_mid and rsi_long are omitted from the output dict.

All other indicators are identical to the source repo.
"""

import numpy as np
import pandas as pd


def _rsi(close, period):
    delta    = close.diff()
    avg_gain = delta.clip(lower=0).ewm(com=period-1, min_periods=period).mean()
    avg_loss = (-delta.clip(upper=0)).ewm(com=period-1, min_periods=period).mean()
    return 100 - (100 / (1 + avg_gain / avg_loss.replace(0, np.nan)))


def compute_indicators(raw_data, mcap_data, screen_tickers, config):
    SMA_SHORT    = config["sma_short"]
    SMA_LONG     = config["sma_long"]
    RSI_PERIOD   = config["rsi_period"]
    VOL_LOOKBACK = config["vol_lookback"]
    ADV_PERIOD   = config["adv_period"]
    CMF_PERIOD   = config["cmf_period"]
    ADV_DIVISOR  = config["adv_divisor"]

    available = [t for t in screen_tickers if t in raw_data["Close"].columns]
    print(f"   {len(available)} tickers in data ({len(screen_tickers)-len(available)} missing)")

    close  = raw_data["Close"][available].copy().astype(float)
    volume = raw_data["Volume"][available].copy().astype(float)
    high   = raw_data["High"][available].copy().astype(float)
    low    = raw_data["Low"][available].copy().astype(float)
    print(f"   Shape: {close.shape}")

    close  = close.ffill(limit=3)
    high   = high.ffill(limit=3)
    low    = low.ffill(limit=3)
    volume = volume.fillna(0)

    print("   [1/8] SMA21 / SMA200...", end=" ", flush=True)
    sma_short = close.rolling(SMA_SHORT, min_periods=SMA_SHORT).mean()
    sma_long  = close.rolling(SMA_LONG,  min_periods=SMA_LONG).mean()
    print("✓")

    print("   [2/8] Rank score (SMA ratio)...", end=" ", flush=True)
    rank_score = sma_short / sma_long.replace(0, np.nan)
    print("✓")

    print("   [3/8] RSI (14)...", end=" ", flush=True)
    rsi = _rsi(close, RSI_PERIOD)
    print("✓")

    # Optional multi-period RSI — only computed if periods are in config
    rsi_mid  = None
    rsi_long = None
    if "rsi_period_mid" in config:
        print(f"   [3b] RSI ({config['rsi_period_mid']})...", end=" ", flush=True)
        rsi_mid = _rsi(close, config["rsi_period_mid"])
        print("✓")
    if "rsi_period_long" in config:
        print(f"   [3c] RSI ({config['rsi_period_long']})...", end=" ", flush=True)
        rsi_long = _rsi(close, config["rsi_period_long"])
        print("✓")

    print("   [4/8] Annualised volatility...", end=" ", flush=True)
    log_ret = np.log(close / close.shift(1))
    ann_vol = log_ret.rolling(VOL_LOOKBACK, min_periods=VOL_LOOKBACK).std() * np.sqrt(252)
    print("✓")

    print("   [5/8] ADV...", end=" ", flush=True)
    adv = ((volume * close) / ADV_DIVISOR).rolling(ADV_PERIOD, min_periods=ADV_PERIOD).mean()
    print("✓")

    print("   [6/8] 52W high...", end=" ", flush=True)
    high_52w = high.rolling(252, min_periods=100).max()
    print("✓")

    print("   [7/8] CMF...", end=" ", flush=True)
    mfm = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    circuit_mfm = np.sign(close - close.shift(1))
    mfm = mfm.where(mfm.notna(), circuit_mfm)
    mfv = mfm * volume
    cmf = mfv.rolling(CMF_PERIOD, min_periods=CMF_PERIOD).sum() / \
          volume.rolling(CMF_PERIOD, min_periods=CMF_PERIOD).sum().replace(0, np.nan)
    print("✓")

    print("   [8/8] MCap matrix...", end=" ", flush=True)
    if isinstance(mcap_data, pd.DataFrame):
        mcap_mat = mcap_data.reindex(columns=close.columns)
    else:
        mcap_arr = np.array([float(mcap_data.get(t, 0)) for t in close.columns], dtype=float)
        mcap_arr[mcap_arr == 0] = np.nan
        mcap_mat = pd.DataFrame(
            np.tile(mcap_arr[np.newaxis, :], (len(close), 1)),
            index=close.index, columns=close.columns
        )
    print("✓")

    out = dict(
        close=close, volume=volume, high=high, low=low,
        sma_short=sma_short, sma_long=sma_long, rank_score=rank_score,
        rsi=rsi, ann_vol=ann_vol, adv=adv, high_52w=high_52w, cmf=cmf, mcap=mcap_mat
    )
    if rsi_mid  is not None: out["rsi_mid"]  = rsi_mid
    if rsi_long is not None: out["rsi_long"] = rsi_long
    return out
