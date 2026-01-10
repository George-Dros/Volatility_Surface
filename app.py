import streamlit as st
import main as m
import numpy as np
from scipy.interpolate import griddata
import plotly.graph_objects as go

st.title("Implied Volatility Surface Interactive App")
st.sidebar.header("User Inputs")

ticker = st.sidebar.text_input("Ticker", value="SPY")
risk_free_rate = st.sidebar.number_input(
    "Risk-Free Rate", min_value=0.0, max_value=1.0, value=0.01, format="%.4f"
)
dividend_yield = st.sidebar.number_input(
    "Dividend Yield", min_value=0.0, max_value=1.0, value=0.001, format="%.4f"
)

option_type = st.sidebar.selectbox(
    "Select Strike Price or Moneyness", ["Strike Price", "Moneyness"]
)

# Stock data
stock, spot_prices, spot_price = m.get_stock_data(ticker)

# Options data (fetch ONCE)
calls_data, expiration_dates = m.get_options_data(stock)
if calls_data.empty:
    st.error("No options data returned for this ticker (or Yahoo blocked the request). Try another ticker.")
    st.stop()

# Slider
dynamic_min_percentage = 20
dynamic_max_percentage = 200
default_min_percentage = 70
default_max_percentage = 130

strike_price_range_percentage = st.sidebar.slider(
    "Strike Price Range (as % of Spot Price)",
    min_value=dynamic_min_percentage,
    max_value=dynamic_max_percentage,
    value=(default_min_percentage, default_max_percentage),
)

min_strike_price = spot_price * (strike_price_range_percentage[0] / 100)
max_strike_price = spot_price * (strike_price_range_percentage[1] / 100)

# Filter
filtered_calls_data = m.filter_calls_data(calls_data, spot_price, min_strike_price, max_strike_price)
if filtered_calls_data.empty:
    st.error("No options matched your strike/expiry filters. Widen the strike range or lower the min expiry.")
    st.stop()

# IV
imp_vol_data = m.calculate_implied_volatility(filtered_calls_data, spot_price, risk_free_rate, dividend_yield)
if imp_vol_data.empty:
    st.error("IV computation returned no valid points (bad quotes / illiquid options). Try widening range or another ticker.")
    st.stop()

# Prepare plot data
X = imp_vol_data["TimeToExpiry"].values
Z = imp_vol_data["ImpliedVolatility"].values * 100

if option_type == "Moneyness":
    # Forward log-moneyness: ln(K/F), F = S * exp((r-q)T)
    T = imp_vol_data["TimeToExpiry"].values
    F = spot_price * np.exp((risk_free_rate - dividend_yield) * T)
    F = np.maximum(F, 1e-12)  # safety

    imp_vol_data["LogMoneyness"] = np.log(imp_vol_data["StrikePrice"].values / F)
    Y = imp_vol_data["LogMoneyness"].values
    y_label = "Log-moneyness ln(K/F)"
else:
    Y = imp_vol_data["StrikePrice"].values
    y_label = "Strike Price ($)"

# Robustness for interpolation
if len(np.unique(X)) < 2 or len(np.unique(Y)) < 2:
    st.error("Not enough variation in expiry/strike to build a surface. Widen strike range or include more expiries.")
    st.stop()

# Interpolate
xi = np.linspace(X.min(), X.max(), 30)
yi = np.linspace(Y.min(), Y.max(), 30)
xi, yi = np.meshgrid(xi, yi)

zi = griddata((X, Y), Z, (xi, yi), method="linear")
zi2 = griddata((X, Y), Z, (xi, yi), method="nearest")
zi = np.where(np.isnan(zi), zi2, zi)

# Plot
fig = go.Figure(data=[go.Surface(x=xi, y=yi, z=zi, colorscale="Viridis")])
fig.update_layout(
    title=f"Implied Volatility Surface of {ticker}",
    scene=dict(
        xaxis_title="Time to Expiration (years)",
        yaxis_title=y_label,
        zaxis_title="Implied Volatility (%)",
    ),
    width=1000,
    height=800,
)

st.plotly_chart(fig)
