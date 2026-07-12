"""SVI slice fitting and surface construction."""
import numpy as np
import pytest

import svi


def test_recovers_known_parameters():
    true = svi.SVIParams(a=0.04, b=0.4, rho=-0.5, m=0.02, sigma=0.15)
    k = np.linspace(-0.5, 0.4, 25)
    w = svi.svi_total_variance(k, true)
    fit = svi.fit_svi_slice(k, w)
    assert fit is not None
    for got, want in zip(fit, true):
        assert got == pytest.approx(want, abs=0.02)


def test_total_variance_is_non_negative_arbitrage_free():
    # a>=0, b>=0, |rho|<1, sigma>0  => w(k) >= 0 for all k
    params = svi.SVIParams(a=0.02, b=0.3, rho=-0.7, m=0.0, sigma=0.1)
    k = np.linspace(-2, 2, 400)
    assert np.all(svi.svi_total_variance(k, params) >= 0)


def test_too_few_points_returns_none():
    k = np.array([0.0, 0.1, 0.2])
    w = np.array([0.04, 0.045, 0.05])
    assert svi.fit_svi_slice(k, w, min_points=6) is None


def test_fit_slice_ignores_nonpositive_and_nan():
    true = svi.SVIParams(a=0.04, b=0.4, rho=-0.3, m=0.0, sigma=0.12)
    k = np.linspace(-0.4, 0.4, 20)
    w = svi.svi_total_variance(k, true)
    w[0] = np.nan
    w[1] = -1.0
    assert svi.fit_svi_slice(k, w) is not None


def _synthetic_iv_df():
    import pandas as pd
    rows = []
    for exp, T in [("2026-10-01", 0.22), ("2026-12-01", 0.39), ("2027-03-01", 0.64)]:
        F = 100 * np.exp(0.02 * T)
        params = svi.SVIParams(a=0.03, b=0.35, rho=-0.6, m=0.0, sigma=0.12)
        for K in np.linspace(70, 130, 15):
            k = np.log(K / F)
            iv = float(svi.svi_iv(k, T, params))
            rows.append((exp, T, K, iv, 10, 100))
    return pd.DataFrame(rows, columns=[
        "Expiration", "TimeToExpiry", "StrikePrice", "ImpliedVolatility",
        "Volume", "OpenInterest"])


def test_fit_all_slices_and_build_surface():
    df = _synthetic_iv_df()
    fits = svi.fit_all_slices(df, spot_price=100, risk_free_rate=0.02, dividend_yield=0.0)
    assert len(fits) == 3
    Ts, k_grid, Z = svi.build_svi_surface(fits, n_k=30)
    assert Z.shape == (30, 3)
    assert np.all(np.diff(Ts) > 0)          # maturities sorted ascending
    assert np.all(Z > 0)                     # positive vols


def test_build_surface_needs_two_slices():
    df = _synthetic_iv_df()
    df = df[df["Expiration"] == "2026-10-01"]
    fits = svi.fit_all_slices(df, 100, 0.02, 0.0)
    assert svi.build_svi_surface(fits) is None
