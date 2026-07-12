import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.interpolate import griddata

import main as m
import svi

st.set_page_config(page_title="Implied Volatility Surface", layout="wide")

st.title("Implied Volatility Surface Interactive App")
st.sidebar.header("User Inputs")

ticker = st.sidebar.text_input("Ticker", value="SPY").strip().upper()
risk_free_rate = st.sidebar.number_input(
    "Risk-Free Rate", min_value=0.0, max_value=1.0, value=0.01, format="%.4f"
)
dividend_yield = st.sidebar.number_input(
    "Dividend Yield", min_value=0.0, max_value=1.0, value=0.001, format="%.4f"
)

y_axis = st.sidebar.selectbox("Y-Axis (raw surface)", ["Strike Price", "Moneyness"])
method = st.sidebar.radio("Surface method", ["Raw (interpolated)", "SVI fit"])

if not ticker:
    st.info("Enter a ticker symbol in the sidebar to begin.")
    st.stop()


@st.cache_data(show_spinner="Fitting SVI slices…")
def cached_svi_fits(iv_df, spot, r, q):
    """Cache the per-expiry SVI fits (independent of strike/liquidity sliders)."""
    return svi.fit_all_slices(iv_df, spot, r, q)


# --- Data (cached; only refetched when the ticker changes) ---------------
try:
    spot_prices, spot_price = m.get_stock_data(ticker)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

options_data, expiration_dates = m.get_options_data(ticker)
if options_data.empty:
    st.error("No options data returned for this ticker (or Yahoo blocked the request). Try another ticker.")
    st.stop()

# OTM blend + strike-independent prep (cached).
prepared = m.prepare_options(options_data, spot_price)
if prepared.empty:
    st.error("No options left after dropping very short-dated expiries. Try another ticker.")
    st.stop()

# Implied vol for the WHOLE (blended) chain — cached per ticker/r/q.
imp_vol_all = m.calculate_implied_volatility(
    prepared, spot_price, risk_free_rate, dividend_yield
)
if imp_vol_all.empty:
    st.error("IV computation returned no valid points (bad quotes / illiquid options). Try another ticker.")
    st.stop()

# --- Filters (cheap: only re-filter an in-memory table) -------------------
st.sidebar.subheader("Strike range")
strike_pct = st.sidebar.slider(
    "Strike Price Range (as % of Spot Price)",
    min_value=20, max_value=200, value=(70, 130),
)
min_strike_price = spot_price * (strike_pct[0] / 100)
max_strike_price = spot_price * (strike_pct[1] / 100)

st.sidebar.subheader("Liquidity filters")
min_volume = st.sidebar.number_input("Min volume", min_value=0, value=0, step=1)
min_open_interest = st.sidebar.number_input("Min open interest", min_value=0, value=0, step=1)
max_spread_pct = st.sidebar.slider("Max bid-ask spread (%)", min_value=1, max_value=100, value=100)

imp_vol_data = m.filter_iv_data(
    imp_vol_all, min_strike_price, max_strike_price,
    min_volume=min_volume, min_open_interest=min_open_interest,
    max_spread_pct=max_spread_pct,
)
if imp_vol_data.empty:
    st.error("No options matched your strike/liquidity filters. Loosen them in the sidebar.")
    st.stop()

n_calls = int((imp_vol_data["OptionType"] == "C").sum())
n_puts = int((imp_vol_data["OptionType"] == "P").sum())
st.caption(
    f"**{ticker}**  ·  spot ${spot_price:,.2f}  ·  {len(imp_vol_data):,} IV points "
    f"({n_puts:,} puts / {n_calls:,} calls) across {imp_vol_data['TimeToExpiry'].nunique()} expiries"
)

surface_tab, smile_tab, data_tab = st.tabs(["3D Surface", "Smile / Skew", "Data & Export"])

# =========================================================================
# 3D SURFACE
# =========================================================================
with surface_tab:
    if method == "SVI fit":
        fits = cached_svi_fits(imp_vol_all, spot_price, risk_free_rate, dividend_yield)
        built = svi.build_svi_surface(fits)
        if built is None:
            st.warning("Not enough expiries with a good SVI fit. Falling back to the raw surface.")
            method = "Raw (interpolated)"
        else:
            Ts, k_grid, Z = built
            fig = go.Figure(data=[go.Surface(x=Ts, y=k_grid, z=Z, colorscale="Viridis")])
            fig.update_layout(
                title=f"SVI Implied Volatility Surface of {ticker}",
                scene=dict(
                    xaxis_title="Time to Expiration (years)",
                    yaxis_title="Log-moneyness ln(K/F)",
                    zaxis_title="Implied Volatility (%)",
                ),
                height=800,
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                f"SVI fit on {len(fits)} expiries · median slice RMSE (total variance): "
                f"{np.median([d['rmse'] for d in fits.values()]):.2e}"
            )

    if method == "Raw (interpolated)":
        X = imp_vol_data["TimeToExpiry"].values
        Z = imp_vol_data["ImpliedVolatility"].values * 100

        if y_axis == "Moneyness":
            T = imp_vol_data["TimeToExpiry"].values
            F = np.maximum(spot_price * np.exp((risk_free_rate - dividend_yield) * T), 1e-12)
            Y = np.log(imp_vol_data["StrikePrice"].values / F)
            y_label = "Log-moneyness ln(K/F)"
        else:
            Y = imp_vol_data["StrikePrice"].values
            y_label = "Strike Price ($)"

        if len(np.unique(X)) < 2 or len(np.unique(Y)) < 2:
            st.error("Not enough variation in expiry/strike to build a surface. Widen the filters.")
            st.stop()

        xi = np.linspace(X.min(), X.max(), 30)
        yi = np.linspace(Y.min(), Y.max(), 30)
        xi, yi = np.meshgrid(xi, yi)
        zi = griddata((X, Y), Z, (xi, yi), method="linear")
        zi_nearest = griddata((X, Y), Z, (xi, yi), method="nearest")
        zi = np.where(np.isnan(zi), zi_nearest, zi)

        fig = go.Figure(data=[go.Surface(x=xi, y=yi, z=zi, colorscale="Viridis")])
        fig.update_layout(
            title=f"Implied Volatility Surface of {ticker}",
            scene=dict(
                xaxis_title="Time to Expiration (years)",
                yaxis_title=y_label,
                zaxis_title="Implied Volatility (%)",
            ),
            height=800,
        )
        st.plotly_chart(fig, use_container_width=True)

# =========================================================================
# SMILE / SKEW (2D slice for a single expiry)
# =========================================================================
with smile_tab:
    expiries = sorted(imp_vol_data["Expiration"].unique())
    if not expiries:
        st.info("No expiries available with the current filters.")
    else:
        chosen = st.selectbox("Expiration", expiries)
        slice_df = imp_vol_data[imp_vol_data["Expiration"] == chosen].sort_values("StrikePrice")
        T = float(slice_df["TimeToExpiry"].mean())
        F = spot_price * np.exp((risk_free_rate - dividend_yield) * T)

        use_moneyness = y_axis == "Moneyness"
        if use_moneyness:
            x_obs = np.log(slice_df["StrikePrice"].to_numpy(float) / F)
            x_label = "Log-moneyness ln(K/F)"
        else:
            x_obs = slice_df["StrikePrice"].to_numpy(float)
            x_label = "Strike ($)"

        fig2 = go.Figure()
        for kind, name, color in [("P", "OTM puts", "#EF553B"), ("C", "OTM calls", "#636EFA")]:
            sub = slice_df[slice_df["OptionType"] == kind]
            if not sub.empty:
                xs = (np.log(sub["StrikePrice"].to_numpy(float) / F) if use_moneyness
                      else sub["StrikePrice"].to_numpy(float))
                fig2.add_trace(go.Scatter(
                    x=xs, y=sub["ImpliedVolatility"].to_numpy(float) * 100,
                    mode="markers", name=name, marker=dict(color=color, size=7),
                ))

        # Overlay the SVI fit for this expiry, if available.
        fits = cached_svi_fits(imp_vol_all, spot_price, risk_free_rate, dividend_yield)
        fit = fits.get(str(chosen))
        if fit is not None:
            k_line = np.linspace(x_obs.min() if use_moneyness else np.log(slice_df["StrikePrice"].min() / F),
                                 x_obs.max() if use_moneyness else np.log(slice_df["StrikePrice"].max() / F),
                                 100)
            iv_line = svi.svi_iv(k_line, fit["T"], fit["params"]) * 100
            x_line = k_line if use_moneyness else F * np.exp(k_line)
            fig2.add_trace(go.Scatter(
                x=x_line, y=iv_line, mode="lines", name="SVI fit",
                line=dict(color="#00CC96", width=2),
            ))

        fig2.update_layout(
            title=f"{ticker} volatility smile — {chosen} (T={T:.3f}y)",
            xaxis_title=x_label, yaxis_title="Implied Volatility (%)",
            height=550,
        )
        st.plotly_chart(fig2, use_container_width=True)

# =========================================================================
# DATA & EXPORT
# =========================================================================
with data_tab:
    st.dataframe(imp_vol_data, use_container_width=True, height=400)
    csv = imp_vol_data.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download IV data (CSV)", data=csv,
        file_name=f"{ticker}_iv_surface.csv", mime="text/csv",
    )
    html = fig.to_html(include_plotlyjs="cdn")
    st.download_button(
        "⬇️ Download surface (interactive HTML)", data=html,
        file_name=f"{ticker}_iv_surface.html", mime="text/html",
    )
