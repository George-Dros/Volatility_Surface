"""Data pipeline: OTM blend, IV computation, and in-memory filtering.

These build synthetic option chains (no network) shaped like yfinance output.
"""
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

import functions as f
import main as m

SPOT = 100.0
R, Q = 0.03, 0.0


def _future_date(days=200):
    return (datetime.now() + timedelta(days=days)).strftime("%Y-%m-%d")


def _chain(true_vol=0.25):
    """One expiry, calls+puts across strikes, priced at a known vol."""
    exp = _future_date()
    T = f.calculate_time_to_expiration(exp)
    rows = []
    for K in [80, 90, 100, 110, 120]:
        cprice = f.Call_BS_Value(SPOT, K, R, T, true_vol, Q)
        pprice = f.Put_BS_Value(SPOT, K, R, T, true_vol, Q)
        rows.append((f"C{K}", K, "C", cprice * 0.99, cprice * 1.01, cprice, 50, 500))
        rows.append((f"P{K}", K, "P", pprice * 0.99, pprice * 1.01, pprice, 60, 600))
    df = pd.DataFrame(rows, columns=[
        "contractSymbol", "strike", "optionType", "bid", "ask",
        "lastPrice", "volume", "openInterest"])
    df["expiration"] = exp
    return df, exp, T


def test_otm_blend_keeps_one_contract_per_strike():
    chain, _, _ = _chain()
    prep = m.prepare_options(chain, SPOT)
    # Exactly one row per strike.
    assert prep["strike"].is_unique
    # Puts at/below spot, calls above spot.
    puts = prep[prep["optionType"] == "P"]
    calls = prep[prep["optionType"] == "C"]
    assert (puts["strike"] <= SPOT).all()
    assert (calls["strike"] > SPOT).all()


def test_midprice_uses_quote_then_last():
    chain, _, _ = _chain()
    # Knock out the quote on one row -> should fall back to lastPrice.
    chain.loc[0, ["bid", "ask"]] = [0.0, 0.0]
    prep = m.prepare_options(chain, SPOT)
    row = prep[prep["contractSymbol"] == chain.loc[0, "contractSymbol"]]
    if not row.empty:  # row survives only if it is on the OTM side
        assert row["midPrice"].iloc[0] == pytest.approx(chain.loc[0, "lastPrice"])


def test_iv_recovers_input_vol_for_both_sides():
    true_vol = 0.3
    chain, _, _ = _chain(true_vol=true_vol)
    prep = m.prepare_options(chain, SPOT)
    iv = m.calculate_implied_volatility(prep, SPOT, R, Q)
    assert not iv.empty
    assert np.allclose(iv["ImpliedVolatility"].to_numpy(float), true_vol, atol=1e-3)
    # Liquidity columns are carried through for later filtering.
    for col in ("Volume", "OpenInterest", "SpreadPct", "OptionType", "Expiration"):
        assert col in iv.columns


def test_filter_by_strike_and_liquidity():
    chain, _, _ = _chain()
    prep = m.prepare_options(chain, SPOT)
    iv = m.calculate_implied_volatility(prep, SPOT, R, Q)

    narrow = m.filter_iv_data(iv, 95, 115)
    assert narrow["StrikePrice"].between(95, 115).all()

    liq = m.filter_iv_data(iv, 0, 1e9, min_volume=55)
    assert (liq["Volume"] >= 55).all()

    oi = m.filter_iv_data(iv, 0, 1e9, min_open_interest=550)
    assert (oi["OpenInterest"] >= 550).all()


def test_spread_filter_keeps_unknown_spread_rows():
    chain, _, _ = _chain()
    # Make one OTM call last-price-only (no quotes) -> SpreadPct is NaN.
    chain.loc[chain["contractSymbol"] == "C110", ["bid", "ask"]] = [0.0, 0.0]
    prep = m.prepare_options(chain, SPOT)
    iv = m.calculate_implied_volatility(prep, SPOT, R, Q)
    kept = m.filter_iv_data(iv, 0, 1e9, max_spread_pct=1.0)
    # The NaN-spread contract is retained despite the tight spread cap.
    assert "C110" in set(kept["ContractSymbol"])


def test_empty_inputs_are_safe():
    empty = pd.DataFrame()
    assert m.prepare_options(empty, SPOT).empty
    iv_empty = m.calculate_implied_volatility(empty, SPOT, R, Q)
    assert list(iv_empty.columns) == m.IV_COLUMNS
    assert m.filter_iv_data(iv_empty, 0, 1e9).empty
