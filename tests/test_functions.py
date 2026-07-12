"""Black-Scholes pricing, implied-vol inversion, and time-to-expiry."""
from datetime import datetime, timedelta

import numpy as np
import pytest

import functions as f


@pytest.mark.parametrize("vol", [0.1, 0.2, 0.5, 1.0])
def test_call_iv_round_trip(vol):
    S, X, r, T, q = 100, 100, 0.03, 0.5, 0.01
    price = f.Call_BS_Value(S, X, r, T, vol, q)
    assert f.Call_IV(S, X, r, T, price, q) == pytest.approx(vol, abs=1e-4)


@pytest.mark.parametrize("vol", [0.1, 0.2, 0.5, 1.0])
def test_put_iv_round_trip(vol):
    S, X, r, T, q = 100, 100, 0.03, 0.5, 0.01
    price = f.Put_BS_Value(S, X, r, T, vol, q)
    assert f.Put_IV(S, X, r, T, price, q) == pytest.approx(vol, abs=1e-4)


def test_put_call_parity():
    S, X, r, T, q, v = 100, 95, 0.03, 0.75, 0.01, 0.25
    call = f.Call_BS_Value(S, X, r, T, v, q)
    put = f.Put_BS_Value(S, X, r, T, v, q)
    # C - P = S e^{-qT} - X e^{-rT}
    lhs = call - put
    rhs = S * np.exp(-q * T) - X * np.exp(-r * T)
    assert lhs == pytest.approx(rhs, abs=1e-9)


def test_dispatch_matches_dedicated_functions():
    S, X, r, T, q = 100, 105, 0.03, 0.4, 0.0
    cprice = f.Call_BS_Value(S, X, r, T, 0.3, q)
    pprice = f.Put_BS_Value(S, X, r, T, 0.3, q)
    assert f.Calculate_IV_Call_Put(S, X, r, T, cprice, "C", q) == pytest.approx(
        f.Call_IV(S, X, r, T, cprice, q))
    assert f.Calculate_IV_Call_Put(S, X, r, T, pprice, "p", q) == pytest.approx(
        f.Put_IV(S, X, r, T, pprice, q))
    assert np.isnan(f.Calculate_IV_Call_Put(S, X, r, T, cprice, "X", q))


def test_price_out_of_bounds_returns_nan():
    S, X, r, T, q = 100, 100, 0.03, 0.5, 0.01
    _, upper = f.call_price_bounds(S, X, r, T, q)
    assert np.isnan(f.Call_IV(S, X, r, T, upper + 1.0, q))   # above intrinsic-upper
    assert np.isnan(f.Call_IV(S, X, r, T, -1.0, q))          # negative price


def test_expired_and_zero_vol_edges():
    assert f.Call_BS_Value(100, 90, 0.03, 0.0, 0.2, 0.01) == pytest.approx(10.0)
    assert f.Put_BS_Value(100, 110, 0.03, 0.0, 0.2, 0.01) == pytest.approx(10.0)
    # zero vol -> discounted intrinsic on the forward, never negative
    assert f.Call_BS_Value(100, 200, 0.03, 1.0, 0.0, 0.0) == pytest.approx(0.0)


def test_time_to_expiration_is_deterministic():
    """Regression: same expiration date must give ONE T when 'now' is fixed."""
    now = datetime(2026, 7, 12, 16, 0, 0)
    vals = {f.calculate_time_to_expiration("2026-10-12", now=now) for _ in range(1000)}
    assert len(vals) == 1
    expected = (datetime(2026, 10, 12) - now).total_seconds() / (365.0 * 24 * 3600)
    assert vals.pop() == pytest.approx(expected, abs=1e-9)


def test_time_to_expiration_never_negative():
    past = datetime.now() + timedelta(days=10)
    assert f.calculate_time_to_expiration("2000-01-01", now=past) == 0.0
