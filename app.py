#!/usr/bin/env python3
"""
Streamlit Web Dashboard: Tactical Sector Rotation Strategy
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, Sequence
import pandas as pd
import streamlit as st
import yfinance as yf

# Set page config
st.set_page_config(
    page_title="Sector Rotation Screener",
    page_icon="📈",
    layout="wide"
)

# --- Universe Definitions ---
CORE_SECTORS = [
    "XLK", "XLF", "XLV", "XLE", "XLI",
    "XLY", "XLP", "XLU", "XLRE", "XLB", "XLC",
    "SPY", "QQQ", "IWM", "GLD"
]
SUBSECTORS = [
    "SMH", "XHB", "IYT", "KRE", "XRT", "ITA", "IGV", "XME"
]
DEFENSIVE_SECTORS = ["XLP", "XLU", "XLV"]
BENCHMARK = "SPY"
VIX_TICKER = "^VIX"
CASH_PROXY = "SGOV"

SMA_WINDOW = 200
VIX_SHORT_WINDOW = 5
VIX_LONG_WINDOW = 20

GICS_FAMILY_MAP = {
    "SPY": "Broad Market",
    "QQQ": "Broad Tech",
    "IWM": "Small Cap",
    "GLD": "Commodities",
    "XLK": "Information Technology",
    "IGV": "Information Technology",
    "SMH": "Information Technology",
    "XLF": "Financials",
    "KRE": "Financials",
    "XLV": "Health Care",
    "XLE": "Energy",
    "XME": "Materials",
    "XLI": "Industrials",
    "IYT": "Industrials",
    "ITA": "Industrials",
    "XLY": "Consumer Discretionary",
    "XHB": "Consumer Discretionary",
    "XRT": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLU": "Utilities",
    "XLRE": "Real Estate",
    "XLB": "Materials",
    "XLC": "Communication Services"
}

HIGH_BETA_SUBSECTORS = {"SMH", "KRE", "XRT", "XHB", "IGV", "XME"}

# --- Models & Strategy Classes ---
@dataclass(frozen=True)
class StrategyConfig:
    min_discount_52w: float = 0.0
    max_discount_core_52w: float = 0.25
    max_discount_sub_52w: float = 0.35
    req_alpha_21d: float = 0.0
    min_vol_ratio: float = 1.0
    vix_caution_level: float = 20.0
    vix_spike_ratio: float = 1.15
    vix_crisis_level: float = 32.0
    max_positions: int = 4
    max_single_weight: float = 0.35
    high_beta_slot_cap: float = 0.25
    sweep_asset: str = CASH_PROXY
    defensive_sectors_only: bool = False
    defensive_min_abs_ret_21d: Optional[float] = 0.0

@dataclass(frozen=True)
class RegimeState:
    name: str
    eligible_universe: list
    req_alpha_21d: float
    stop_sma_buffer: float
    use_sma20_emergency_stop: bool
    sweep_asset: str
    req_abs_ret_21d: Optional[float] = None

# --- Cached Data Fetchers ---
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_market_data(tickers: tuple[str, ...], period: str = "2y") -> pd.DataFrame:
    df = yf.download(list(tickers), period=period, auto_adjust=True)['Close']
    return df

# --- Logic Functions ---
def detect_macro_regime(df_window: pd.DataFrame, cfg: StrategyConfig) -> RegimeState:
    spy = df_window[BENCHMARK]
    vix = df_window[VIX_TICKER]

    sma200 = spy.rolling(SMA_WINDOW).mean().iloc[-1]
    current_spy = spy.iloc[-1]
    current_vix = vix.iloc[-1]
    v5 = vix.rolling(VIX_SHORT_WINDOW).mean().iloc[-1]
    v20 = vix.rolling(VIX_LONG_WINDOW).mean().iloc[-1]

    vix_ratio = v5 / v20 if v20 > 0 else 1.0
    bull = current_spy >= sma200
    elevated = current_vix > cfg.vix_caution_level or vix_ratio >= cfg.vix_spike_ratio
    crisis = current_vix > cfg.vix_crisis_level

    if crisis:
        return RegimeState("CASH_PROTECTION", [], 0.05, 0.95, True, cfg.sweep_asset)
    if bull and not elevated:
        return RegimeState("RISK_ON", CORE_SECTORS + SUBSECTORS, 0.0, 0.99, False, cfg.sweep_asset)
    if bull:
        return RegimeState("VOL_CAUTION", CORE_SECTORS, 0.01, 0.98, True, cfg.sweep_asset)
    return RegimeState(
        "DEFENSIVE_ONLY",
        list(DEFENSIVE_SECTORS) if cfg.defensive_sectors_only else list(CORE_SECTORS),
        0.02, 0.97, True, cfg.sweep_asset, cfg.defensive_min_abs_ret_21d
    )

def passes_discount_filter(price: float, high_52w: float, ticker: str, cfg: StrategyConfig) -> bool:
    if pd.isna(price) or pd.isna(high_52w) or high_52w <= 0:
        return False
    discount = max(0.0, 1.0 - price / high_52w)
    max_disc = cfg.max_discount_sub_52w if ticker in SUBSECTORS else cfg.max_discount_core_52w
    return cfg.min_discount_52w <= discount <= max_disc

def passes_return_filters(ret_21d: float, spy_ret_21d: float, regime: RegimeState) -> bool:
    if pd.isna(ret_21d) or pd.isna(spy_ret_21d):
        return False
    if (ret_21d - spy_ret_21d) < regime.req_alpha_21d:
        return False
    if regime.req_abs_ret_21d is not None and ret_21d < regime.req_abs_ret_21d:
        return False
    return True

def select_positions(
    df_screened: pd.DataFrame,
    cfg: StrategyConfig,
    eligible_universe: Optional[Sequence[str]] = None,
    sort_col: str = "Alpha_21d"
) -> list[str]:
    df = df_screened.sort_values(sort_col, ascending=False, kind="mergesort")
    eligible = set(eligible_universe) if eligible_universe is not None else None
    hb_limit = max(1, math.floor(cfg.high_beta_slot_cap * cfg.max_positions + 1e-9)) if cfg.high_beta_slot_cap > 0 else 0
    selected, seen_families, hb_count = [], set(), 0

    for _, row in df.iterrows():
        ticker = row["Ticker"]
        if eligible is not None and ticker not in eligible:
            continue
        fam = row["Family"]
        if fam in seen_families:
            continue
        is_hb = ticker in HIGH_BETA_SUBSECTORS
        if is_hb and hb_count >= hb_limit:
            continue

        selected.append(ticker)
        seen_families.add(fam)
        hb_count += int(is_hb)
        if len(selected) >= cfg.max_positions:
            break

    return selected

# ==========================================
# STREAMLIT UI & DASHBOARD LAYOUT
# ==========================================

st.title("Tactical Sector Rotation Screener")

# Sidebar Controls
st.sidebar.header("Strategy Controls")
max_pos = st.sidebar.slider("Portfolio Slots", min_value=2, max_value=8, value=4)
vix_caution = st.sidebar.slider("VIX Caution Level", min_value=15.0, max_value=30.0, value=20.0, step=0.5)
high_beta_cap = st.sidebar.slider("High-Beta Cap Ratio", min_value=0.0, max_value=1.0, value=0.25, step=0.05)
force_defensive = st.sidebar.checkbox("Defensive Only Mode", value=False)

cfg = StrategyConfig(
    max_positions=max_pos,
    vix_caution_level=vix_caution,
    high_beta_slot_cap=high_beta_cap,
    defensive_sectors_only=force_defensive
)

# Fetch Macro Data
with st.spinner("Downloading macro and universe data..."):
    df_macro = fetch_market_data((BENCHMARK, VIX_TICKER), period="2y").dropna()
    regime = detect_macro_regime(df_macro, cfg)
    
    universe_tickers = tuple(sorted(list(set(regime.eligible_universe + [BENCHMARK]))))
    df_prices = fetch_market_data(universe_tickers, period="1y").dropna()

# Regime Metrics Header
col1, col2, col3, col4 = st.columns(4)
col1.metric("Current Regime", regime.name)
col2.metric("Eligible Universe", f"{len(regime.eligible_universe)} tickers")
col3.metric("Required 21d Alpha", f"{regime.req_alpha_21d:.1%}")
col4.metric("Sweep / Cash Proxy", regime.sweep_asset)

st.divider()

# Screening Logic
ret_21d = df_prices.pct_change(21).iloc[-1]
spy_ret_21d = ret_21d[BENCHMARK]
latest_prices = df_prices.iloc[-1]
highs_52w = df_prices.max()

records = []
for ticker in regime.eligible_universe:
    if ticker not in latest_prices:
        continue
    price = latest_prices[ticker]
    high = highs_52w[ticker]
    r21 = ret_21d[ticker]

    if passes_discount_filter(price, high, ticker, cfg) and passes_return_filters(r21, spy_ret_21d, regime):
        records.append({
            "Ticker": ticker,
            "Family": GICS_FAMILY_MAP.get(ticker, "Unknown"),
            "Price": round(price, 2),
            "52w_High": round(high, 2),
            "Discount_52w": max(0.0, 1.0 - price / high),
            "Return_21d": r21,
            "Alpha_21d": r21 - spy_ret_21d
        })

df_screened = pd.DataFrame(records)

# Portfolio Selection Display
st.subheader("Tactical Allocations")
if not df_screened.empty:
    selected = select_positions(df_screened, cfg, eligible_universe=regime.eligible_universe)
    target_weight = f"{(1.0 / len(selected)):.1%}" if selected else "0.0%"

    df_selected = df_screened[df_screened["Ticker"].isin(selected)].copy()
    df_selected["Weight"] = target_weight
    
    # Format percentages for table
    display_table = df_selected[["Ticker", "Family", "Weight", "Price", "52w_High", "Discount_52w", "Return_21d", "Alpha_21d"]].copy()
    display_table["Discount_52w"] = display_table["Discount_52w"].map("{:.2%}".format)
    display_table["Return_21d"] = display_table["Return_21d"].map("{:.2%}".format)
    display_table["Alpha_21d"] = display_table["Alpha_21d"].map("{:.2%}".format)
    
    st.dataframe(display_table.reset_index(drop=True), use_container_width=True)
else:
    st.warning(f"No sector ETFs cleared filters. 100% of allocation directed to {regime.sweep_asset}.")

# Footer Metadata
st.caption(f"Data window: {df_prices.index[0].strftime('%Y-%m-%d')} to {df_prices.index[-1].strftime('%Y-%m-%d')} | Rebalanced dynamically with 1-hour cache.")
