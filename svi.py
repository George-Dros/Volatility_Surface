"""Gatheral 'raw' SVI parameterisation and per-expiry slice fitting.

The raw SVI total-variance smile is

    w(k) = a + b * ( rho * (k - m) + sqrt((k - m)^2 + sigma^2) )

where ``k = ln(K / F)`` is log-moneyness (F = forward), and ``w = IV^2 * T`` is
total implied variance. Fitting one slice per expiry gives a smooth curve that
is free of butterfly (static) arbitrage when the parameters satisfy simple
bound constraints -- a cleaner alternative to raw grid interpolation of noisy
quotes.

Cross-expiry (calendar) arbitrage is NOT enforced here; each expiry is fit
independently.
"""

from collections import namedtuple

import numpy as np
from scipy.optimize import least_squares

SVIParams = namedtuple("SVIParams", ["a", "b", "rho", "m", "sigma"])

# Minimum quotes needed to fit 5 parameters with a little redundancy.
MIN_POINTS = 6


def svi_total_variance(k, params):
    """Total implied variance w(k) for the given SVI parameters."""
    a, b, rho, m, sigma = params
    km = k - m
    return a + b * (rho * km + np.sqrt(km * km + sigma * sigma))


def svi_iv(k, T, params):
    """Implied volatility (not %) implied by an SVI slice at maturity T."""
    w = np.maximum(svi_total_variance(k, params), 0.0)
    return np.sqrt(w / T)


def fit_svi_slice(k, w, weights=None, min_points=MIN_POINTS):
    """Fit one SVI slice to observed (log-moneyness, total-variance) points.

    Returns an ``SVIParams`` on success or ``None`` if there are too few points
    or the optimiser fails. Bounds keep the slice arbitrage-free: a >= 0, b >= 0,
    sigma > 0 and |rho| < 1 together guarantee w(k) >= 0 everywhere.
    """
    k = np.asarray(k, dtype=float)
    w = np.asarray(w, dtype=float)

    good = np.isfinite(k) & np.isfinite(w) & (w > 0)
    k, w = k[good], w[good]
    if weights is not None:
        weights = np.asarray(weights, dtype=float)[good]
    if k.size < min_points:
        return None

    sqrt_w = np.ones_like(w) if weights is None else np.sqrt(np.maximum(weights, 0) + 1e-9)

    def residuals(p):
        return (svi_total_variance(k, p) - w) * sqrt_w

    w_min = max(w.min(), 1e-6)
    k_lo, k_hi = float(k.min()), float(k.max())
    span = max(k_hi - k_lo, 1e-3)

    # (a, b, rho, m, sigma)
    p0 = np.array([w_min, 0.1, -0.3, 0.0, 0.1])
    lower = np.array([0.0, 0.0, -0.999, k_lo - span, 1e-4])
    upper = np.array([10.0 * w.max() + 1e-6, 10.0, 0.999, k_hi + span, 5.0])
    p0 = np.clip(p0, lower + 1e-9, upper - 1e-9)

    try:
        res = least_squares(residuals, p0, bounds=(lower, upper),
                            method="trf", max_nfev=2000)
    except Exception:
        return None

    if not res.success and res.status <= 0:
        return None
    return SVIParams(*res.x)


def slice_rmse(k, w, params):
    """RMSE of a fitted slice against observed total variances."""
    resid = svi_total_variance(np.asarray(k, float), params) - np.asarray(w, float)
    return float(np.sqrt(np.mean(resid ** 2)))


def fit_all_slices(iv_df, spot_price, risk_free_rate, dividend_yield,
                   min_points=MIN_POINTS, use_liquidity_weights=True):
    """Fit an SVI slice for every expiry in an IV table.

    ``iv_df`` must have columns Expiration, TimeToExpiry, StrikePrice,
    ImpliedVolatility (and optionally Volume / OpenInterest for weighting).

    Returns a dict keyed by expiration date string with, per expiry:
    ``params`` (SVIParams), ``T``, ``F`` (forward), ``rmse`` and ``n`` (points).
    """
    fits = {}
    if iv_df.empty:
        return fits

    S = float(spot_price)
    for expiration, grp in iv_df.groupby("Expiration"):
        T = float(grp["TimeToExpiry"].mean())
        if T <= 0 or len(grp) < min_points:
            continue
        F = S * np.exp((risk_free_rate - dividend_yield) * T)
        if F <= 0:
            continue

        k = np.log(grp["StrikePrice"].to_numpy(float) / F)
        iv = grp["ImpliedVolatility"].to_numpy(float)
        w = iv * iv * T

        weights = None
        if use_liquidity_weights and {"Volume", "OpenInterest"}.issubset(grp.columns):
            weights = grp["Volume"].to_numpy(float) + grp["OpenInterest"].to_numpy(float)

        params = fit_svi_slice(k, w, weights=weights, min_points=min_points)
        if params is None:
            continue

        fits[str(expiration)] = {
            "params": params,
            "T": T,
            "F": float(F),
            "rmse": slice_rmse(k, w, params),
            "n": int(len(grp)),
        }
    return fits


def build_svi_surface(fits, k_grid=None, n_k=40, k_range=(-0.6, 0.4)):
    """Assemble a clean (expiry x log-moneyness) IV surface from slice fits.

    Returns ``(Ts, k_grid, Z_pct)`` where ``Ts`` is the sorted maturities,
    ``k_grid`` the shared log-moneyness axis, and ``Z_pct`` an
    ``(n_k, n_expiries)`` matrix of implied volatility in percent. Returns
    ``None`` if fewer than two expiries were fit.
    """
    if len(fits) < 2:
        return None

    if k_grid is None:
        k_grid = np.linspace(k_range[0], k_range[1], n_k)

    items = sorted(fits.values(), key=lambda d: d["T"])
    Ts = np.array([d["T"] for d in items])
    Z = np.column_stack([svi_iv(k_grid, d["T"], d["params"]) * 100.0 for d in items])
    return Ts, k_grid, Z
