# main.py
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

import functions as f

# Columns produced by the implied-volatility step (kept as a constant so an
# empty result still has the right schema for downstream code).
IV_COLUMNS = [
    "ContractSymbol", "StrikePrice", "TimeToExpiry", "ImpliedVolatility",
    "OptionType", "Expiration", "Volume", "OpenInterest", "SpreadPct",
]


@st.cache_data(show_spinner="Fetching spot price…", ttl=900)
def get_stock_data(ticker_symbol="SPY", period="5d"):
    """Fetch recent price history and derive the spot price.

    Cached per (ticker, period) for 15 min. A single ``history`` call covers
    both needs, so the old duplicate 1d request is gone. Raises ValueError on
    an unknown / empty ticker so callers can show a friendly message.
    """
    hist = yf.Ticker(ticker_symbol).history(period=period)
    if hist.empty or "Close" not in hist or hist["Close"].dropna().empty:
        raise ValueError(
            f"No price data available for '{ticker_symbol}'. "
            "Check the ticker symbol or try again later."
        )

    spot_prices = hist["Close"].dropna().to_frame()
    spot_price = float(spot_prices["Close"].iloc[-1])
    return spot_prices, spot_price


@st.cache_data(show_spinner="Fetching option chains…", ttl=900)
def get_options_data(ticker_symbol):
    """Download every call AND put option chain for a ticker, in parallel.

    Keyed by the ticker *string* (hashable) so the whole result is cached.
    Returns a single long DataFrame tagged with an ``optionType`` column
    ('C'/'P') plus the tuple of expiration dates. Chains are fetched
    concurrently, which cuts wall-clock time roughly in proportion to the
    worker count.
    """
    expirations = tuple(yf.Ticker(ticker_symbol).options)
    if not expirations:
        return pd.DataFrame(), expirations

    def fetch(date):
        try:
            # Fresh Ticker per thread avoids sharing a non-thread-safe session.
            chain = yf.Ticker(ticker_symbol).option_chain(date)
            calls = chain.calls.copy()
            calls["optionType"] = "C"
            puts = chain.puts.copy()
            puts["optionType"] = "P"
            both = pd.concat([calls, puts], ignore_index=True)
            both["expiration"] = date
            return both
        except Exception:
            return None

    max_workers = min(8, len(expirations))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        frames = [c for c in executor.map(fetch, expirations)
                  if c is not None and not c.empty]

    options_all = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return options_all, expirations


@st.cache_data(show_spinner=False)
def prepare_options(options_data, spot_price, min_time_to_expiry=0.07):
    """Blend to the out-of-the-money side, add TimeToExpiry / midPrice / liquidity.

    A real vol surface is built from OTM options on each side (OTM puts below
    spot, OTM calls above), because those are the liquid, informative quotes --
    ITM options carry wide spreads and stale last prices. For each strike we
    therefore keep the put when strike <= spot and the call when strike > spot,
    leaving exactly one contract per (expiration, strike).

    Strike-independent, so it runs once on the full chain (not per slider move).
    TimeToExpiry uses a single ``now`` mapped over unique dates -- deterministic
    and cheap. Liquidity columns are carried through so the app can filter them
    in memory without recomputing IV.
    """
    if options_data.empty:
        return options_data

    df = options_data.copy()

    now = datetime.now()
    tte_by_date = {
        date: f.calculate_time_to_expiration(date, now=now)
        for date in df["expiration"].unique()
    }
    df["TimeToExpiry"] = df["expiration"].map(tte_by_date)
    df = df[df["TimeToExpiry"] >= min_time_to_expiry].copy()

    # OTM blend: puts at/below spot, calls above spot.
    is_otm = np.where(df["optionType"].values == "P",
                      df["strike"].values <= spot_price,
                      df["strike"].values > spot_price)
    df = df[is_otm].copy()

    bid = df["bid"]
    ask = df["ask"]
    mid = 0.5 * (bid + ask)
    quoted = (bid > 0) & (ask > 0)
    # Use the bid/ask midpoint when both sides are quoted, else last traded price.
    df["midPrice"] = np.where(quoted, mid, df["lastPrice"])
    # Relative spread in %, only meaningful when both sides are quoted.
    df["SpreadPct"] = np.where(quoted & (mid > 0), 100.0 * (ask - bid) / mid, np.nan)
    df["Volume"] = df.get("volume", pd.Series(index=df.index, dtype=float)).fillna(0)
    df["OpenInterest"] = df.get("openInterest", pd.Series(index=df.index, dtype=float)).fillna(0)

    return df.reset_index(drop=True)


@st.cache_data(show_spinner="Computing implied volatilities…")
def calculate_implied_volatility(prepared_options, spot_price, risk_free_rate, dividend_yield):
    """Solve Black-Scholes implied vol for every OTM option in the chain.

    Uses the put inverter for puts and the call inverter for calls. Depends only
    on (chain, spot, r, q) -- NOT on the strike / liquidity sliders -- so it is
    computed once per parameter set and cached; the app then filters the result
    in memory.
    """
    if prepared_options.empty:
        return pd.DataFrame(columns=IV_COLUMNS)

    S = float(spot_price)
    symbols = prepared_options["contractSymbol"].to_numpy()
    strikes = prepared_options["strike"].to_numpy(dtype=float)
    times = prepared_options["TimeToExpiry"].to_numpy(dtype=float)
    prices = prepared_options["midPrice"].to_numpy(dtype=float)
    kinds = prepared_options["optionType"].to_numpy()
    expirations = prepared_options["expiration"].to_numpy()
    volumes = prepared_options["Volume"].to_numpy(dtype=float)
    ois = prepared_options["OpenInterest"].to_numpy(dtype=float)
    spreads = prepared_options["SpreadPct"].to_numpy(dtype=float)

    rows = []
    for sym, strike, T, price, kind, exp, vol, oi, spr in zip(
        symbols, strikes, times, prices, kinds, expirations, volumes, ois, spreads
    ):
        if not (np.isfinite(price) and price > 0):
            continue
        if not (np.isfinite(T) and T > 0):
            continue
        iv = f.Calculate_IV_Call_Put(S, strike, risk_free_rate, T, price, kind, dividend_yield)
        if np.isfinite(iv):
            rows.append((sym, strike, T, iv, kind, exp, vol, oi, spr))

    return pd.DataFrame(rows, columns=IV_COLUMNS)


def filter_iv_data(iv_data, min_strike_price, max_strike_price,
                   min_volume=0, min_open_interest=0, max_spread_pct=None):
    """Filter an IV table by strike window and liquidity (cheap, in-memory).

    Rows whose spread is unknown (last-price-only quotes) are kept regardless of
    ``max_spread_pct`` -- we simply can't assess their spread.
    """
    if iv_data.empty:
        return iv_data

    mask = (
        (iv_data["StrikePrice"] >= min_strike_price)
        & (iv_data["StrikePrice"] <= max_strike_price)
        & (iv_data["Volume"] >= min_volume)
        & (iv_data["OpenInterest"] >= min_open_interest)
    )
    if max_spread_pct is not None:
        mask &= iv_data["SpreadPct"].isna() | (iv_data["SpreadPct"] <= max_spread_pct)

    return iv_data[mask].reset_index(drop=True)
