"""
core/screener_engine.py — Screening and ranking logic.

Extended vs NSE_1000Cr_Momentum with two pluggable hooks so backtest
strategies can customise behaviour without touching this file:

  config["rank_fn"]:    callable(ind_row, ticker_list) -> pd.Series
                        Returns a rank score Series indexed by ticker.
                        Default: sma_short / sma_long (original behaviour).

  config["skip_filters"]: list of filter names to skip.
                        Allowed values: "rsi", "cmf", "sma", "high52w", "vol"
                        Default: [] (all filters active).

Everything else — MCap, ADV, screen date anchoring — is unchanged.
"""

import pandas as pd


def find_screen_date(ind, anchors):
    available_anchors = [t for t in anchors if t in ind["close"].columns]
    if available_anchors:
        anchor_close = ind["close"][available_anchors[0]]
        valid_dates  = anchor_close.dropna().index
        if len(valid_dates):
            return valid_dates[-1]
    non_null  = ind["close"].notna().sum(axis=1)
    threshold = len(ind["close"].columns) * 0.50
    return non_null[non_null >= threshold].index[-1]


def run_screen(ind, config):
    MIN_MCAP       = config["min_mcap"]
    MIN_ADV        = config["min_adv"]
    MAX_VOLATILITY = config["max_volatility"]
    RSI_THRESHOLD  = config["rsi_threshold"]
    MAX_FROM_HIGH  = config["max_from_high"]
    CMF_THRESHOLD  = config["cmf_threshold"]
    PORTFOLIO_SIZE = config["portfolio_size"]
    SMA_BUFFER     = config.get("sma_buffer", 0.05)
    anchors        = config["anchors"]
    skip_filters   = set(config.get("skip_filters", []))
    rank_fn        = config.get("rank_fn", None)   # callable or None

    screen_date = find_screen_date(ind, anchors)
    idx = ind["close"].index.get_indexer([screen_date], method="ffill")[0]

    close_row  = ind["close"].iloc[idx]
    sma_s_row  = ind["sma_short"].iloc[idx]
    sma_l_row  = ind["sma_long"].iloc[idx]
    rsi_row    = ind["rsi"].iloc[idx]
    vol_row    = ind["ann_vol"].iloc[idx]
    adv_row    = ind["adv"].iloc[idx]
    high52_row = ind["high_52w"].iloc[idx]
    cmf_row    = ind["cmf"].iloc[idx]
    mcap_row   = ind["mcap"].iloc[idx]

    valid = close_row.notna() & sma_l_row.notna() & sma_s_row.notna()
    print(f"   screen_date={screen_date.date()}, valid={int(valid.sum())}/{len(valid)}")

    # ── Filters ──────────────────────────────────────────────────────────────
    m_mcap = mcap_row.ge(MIN_MCAP).fillna(False)
    m_adv  = adv_row.ge(MIN_ADV)
    m_vol  = vol_row.le(MAX_VOLATILITY)  if "vol"   not in skip_filters else pd.Series(True, index=close_row.index)
    m_rsi  = rsi_row.ge(RSI_THRESHOLD)  if "rsi"   not in skip_filters else pd.Series(True, index=close_row.index)
    m_sma  = close_row.ge(sma_s_row.mul(1 - SMA_BUFFER)) if "sma" not in skip_filters else pd.Series(True, index=close_row.index)
    m_high = close_row.ge(high52_row.mul(1 - MAX_FROM_HIGH)) if "high52w" not in skip_filters else pd.Series(True, index=close_row.index)
    m_cmf  = cmf_row.ge(CMF_THRESHOLD)  if "cmf"   not in skip_filters else pd.Series(True, index=close_row.index)

    passed = valid & m_mcap & m_adv & m_vol & m_rsi & m_sma & m_high & m_cmf

    rejections = {
        "no_data"   : int((~valid).sum()),
        "mcap"      : int((valid & ~m_mcap).sum()),
        "adv"       : int((valid & m_mcap & ~m_adv).sum()),
        "volatility": int((valid & m_mcap & m_adv & ~m_vol).sum()),
        "rsi"       : int((valid & m_mcap & m_adv & m_vol & ~m_rsi).sum()),
        "sma"       : int((valid & m_mcap & m_adv & m_vol & m_rsi & ~m_sma).sum()),
        "high52w"   : int((valid & m_mcap & m_adv & m_vol & m_rsi & m_sma & ~m_high).sum()),
        "cmf"       : int((valid & m_mcap & m_adv & m_vol & m_rsi & m_sma & m_high & ~m_cmf).sum()),
    }
    print("── Rejection waterfall ──")
    for k, v in rejections.items():
        print(f"   {k:<12}: {v}")

    # ── Rank score ────────────────────────────────────────────────────────────
    all_tickers = valid[valid].index.tolist()
    if rank_fn is not None:
        rank_row = rank_fn(ind, idx, all_tickers)
    else:
        rank_row = (ind["sma_short"].iloc[idx] / ind["sma_long"].iloc[idx].replace(0, float("nan")))

    universe_df = pd.DataFrame({
        "ticker"        : all_tickers,
        "price"         : close_row[all_tickers].values,
        "rank_score"    : rank_row[all_tickers].values,
        "rsi"           : rsi_row[all_tickers].values,
        "volatility_pct": vol_row[all_tickers].values * 100,
        "adv_m"         : adv_row[all_tickers].values,
        "mcap_m"        : mcap_row[all_tickers].values,
        "pct_from_high" : (close_row[all_tickers].values / high52_row[all_tickers].values - 1) * 100,
        "cmf"           : cmf_row[all_tickers].values,
        "sma21"         : sma_s_row[all_tickers].values,
        "sma200"        : sma_l_row[all_tickers].values,
        "p_mcap"        : m_mcap[all_tickers].values,
        "p_adv"         : m_adv[all_tickers].values,
        "p_vol"         : m_vol[all_tickers].values,
        "p_rsi"         : m_rsi[all_tickers].values,
        "p_sma"         : m_sma[all_tickers].values,
        "p_high"        : m_high[all_tickers].values,
        "p_cmf"         : m_cmf[all_tickers].values,
        "passes_all"    : passed[all_tickers].values,
    }).sort_values("rank_score", ascending=False).reset_index(drop=True)
    universe_df.index += 1

    # ── Near-miss detection: exactly 1 strict filter failing within 10% ─────
    NEAR_MISS_TOL = 0.10
    NM_MIN_MCAP   = MIN_MCAP        * (1 - NEAR_MISS_TOL)
    NM_MIN_ADV    = MIN_ADV         * (1 - NEAR_MISS_TOL)
    NM_MAX_VOL    = MAX_VOLATILITY  * (1 + NEAR_MISS_TOL)
    NM_RSI        = RSI_THRESHOLD   * (1 - NEAR_MISS_TOL)
    NM_SMA_BUFFER = SMA_BUFFER      * (1 + NEAR_MISS_TOL)
    NM_MAX_HIGH   = MAX_FROM_HIGH   * (1 + NEAR_MISS_TOL)
    NM_CMF        = CMF_THRESHOLD   * (1 - NEAR_MISS_TOL)

    nm_mcap = mcap_row.ge(NM_MIN_MCAP).fillna(False)
    nm_adv  = adv_row.ge(NM_MIN_ADV)
    nm_vol  = vol_row.le(NM_MAX_VOL)  if "vol"    not in skip_filters else pd.Series(True, index=close_row.index)
    nm_rsi  = rsi_row.ge(NM_RSI)      if "rsi"    not in skip_filters else pd.Series(True, index=close_row.index)
    nm_sma  = close_row.ge(sma_s_row.mul(1 - NM_SMA_BUFFER)) if "sma" not in skip_filters else pd.Series(True, index=close_row.index)
    nm_high = close_row.ge(high52_row.mul(1 - NM_MAX_HIGH))   if "high52w" not in skip_filters else pd.Series(True, index=close_row.index)
    nm_cmf  = cmf_row.ge(NM_CMF)      if "cmf"    not in skip_filters else pd.Series(True, index=close_row.index)

    filter_pairs = [
        ("mcap", m_mcap, nm_mcap), ("adv",  m_adv,  nm_adv),
        ("vol",  m_vol,  nm_vol),  ("rsi",  m_rsi,  nm_rsi),
        ("sma",  m_sma,  nm_sma),  ("high", m_high, nm_high),
        ("cmf",  m_cmf,  nm_cmf),
    ]

    is_near_miss_map    = {}
    near_miss_filter_map = {}
    for t in all_tickers:
        if passed[t]:
            is_near_miss_map[t]     = False
            near_miss_filter_map[t] = None
            continue
        strict_fails  = [name for name, sm, _  in filter_pairs if not sm[t]]
        relaxed_fails = [name for name, _,  rm in filter_pairs if not rm[t]]
        if len(strict_fails) == 1 and len(relaxed_fails) == 0:
            is_near_miss_map[t]     = True
            near_miss_filter_map[t] = strict_fails[0]
        else:
            is_near_miss_map[t]     = False
            near_miss_filter_map[t] = None

    universe_df["is_near_miss"]     = [is_near_miss_map[t]     for t in all_tickers]
    universe_df["near_miss_filter"] = [near_miss_filter_map[t] for t in all_tickers]

    # ── Walk by rank: include strict pass OR near-miss (top 50 only) ──────────
    HOLD_ZONE_SIZE = config.get("hold_zone_size", 25)

    top_n_rows     = []
    hold_zone_rows = []
    n_promoted     = 0

    for rank_pos, row in universe_df.iterrows():  # 1-based, sorted by rank desc
        if len(hold_zone_rows) >= HOLD_ZONE_SIZE:
            break
        eligible = row["passes_all"] or (row["is_near_miss"] and rank_pos <= 50)
        if not eligible:
            continue
        hold_zone_rows.append(row)
        if len(top_n_rows) < PORTFOLIO_SIZE:
            top_n_rows.append(row)
            if row["is_near_miss"]:
                n_promoted += 1

    top_n_df     = pd.DataFrame(top_n_rows).reset_index(drop=True)
    hold_zone_df = pd.DataFrame(hold_zone_rows).reset_index(drop=True)
    all_passing  = universe_df[universe_df["passes_all"]].copy().reset_index(drop=True)

    if top_n_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), rejections, screen_date

    top_n_df.index     += 1
    hold_zone_df.index += 1
    all_passing.index  += 1

    n_strict = len(top_n_df) - n_promoted
    n_cash   = PORTFOLIO_SIZE - len(top_n_df)
    print(f"\n✅ Screen date: {screen_date.date()} | Strict: {len(all_passing)} | "
          f"Near-miss promoted: {n_promoted} | Hold zone: {len(hold_zone_df)} | Cash slots: {n_cash}")

    return top_n_df, all_passing, hold_zone_df, rejections, screen_date
