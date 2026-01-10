# main.py
import yfinance as yf
import pandas as pd
import numpy as np
import functions as f
import streamlit as st
from scipy.interpolate import griddata
import plotly.graph_objects as go


def get_stock_data(ticker_symbol="SPY", period="1y"):
    stock = yf.Ticker(ticker_symbol)
    spot_prices = stock.history(period=period)["Close"].to_frame()

    # Attempt to get today's spot price safely
    spot_data = stock.history(period="1d")["Close"]

    if not spot_data.empty:
        spot_price = spot_data.iloc[-1]
    else:
        st.warning(f"No recent data available for ticker {ticker_symbol}. Defaulting to last available price from historical data.")
        spot_price = spot_prices.iloc[-1, 0] if not spot_prices.empty else None

    if spot_price is None:
        raise ValueError(f"No data available for ticker {ticker_symbol}. Please check the ticker symbol or try again later.")
    
    return stock, spot_prices, spot_price


def get_options_data(stock):
    calls_frames = []
    for date in stock.options:
        try:
            chain = stock.option_chain(date).calls.copy()
            chain["expiration"] = date
            calls_frames.append(chain)
        except Exception:
            continue

    calls_all = pd.concat(calls_frames, ignore_index=True) if calls_frames else pd.DataFrame()
    return calls_all, stock.options

def filter_calls_data(calls_data, spot_price, min_strike_price, max_strike_price):
    filtered_calls_data = calls_data[
        (calls_data["strike"] >= min_strike_price) &
        (calls_data["strike"] <= max_strike_price)].copy()

    filtered_calls_data["TimeToExpiry"] = filtered_calls_data["expiration"].map(f.calculate_time_to_expiration)
    filtered_calls_data = filtered_calls_data[filtered_calls_data["TimeToExpiry"] >= 0.07]

    bid = filtered_calls_data["bid"]
    ask = filtered_calls_data["ask"]
    mid = 0.5 * (bid + ask)

    filtered_calls_data["midPrice"] = np.where(
        (bid > 0) & (ask > 0),
        mid,
        filtered_calls_data["lastPrice"])


    return filtered_calls_data.reset_index(drop=True)


def calculate_implied_volatility(filtered_calls_data, spot_price, risk_free_rate, dividend_yield):
    rows = []
    for _, row in filtered_calls_data.iterrows():
        T = row["TimeToExpiry"]
        price = row["midPrice"]

        if not np.isfinite(price) or price <= 0: continue
        if not np.isfinite(T) or T <= 0: continue
        iv = f.Call_IV(spot_price, row["strike"], risk_free_rate, T, price, dividend_yield)
        if np.isfinite(iv):
            rows.append((row["contractSymbol"], row["strike"], T, iv))

    imp_vol_data = pd.DataFrame(rows, columns=["ContractSymbol","StrikePrice","TimeToExpiry","ImpliedVolatility"])


    return imp_vol_data.dropna().reset_index(drop=True)

def get_plot_data(filtered_df):
    X = filtered_df['TimeToExpiry'].values
    Y = filtered_df['StrikePrice'].values
    Z = filtered_df['ImpliedVolatility'].values * 100

    return X, Y, Z

# Optional: a function to create the plot if needed.
def plot_implied_volatility(X, Y, Z):
    xi = np.linspace(X.min(), X.max(), 50)
    yi = np.linspace(Y.min(), Y.max(), 50)
    xi, yi = np.meshgrid(xi, yi)

    zi = griddata((X, Y), Z, (xi, yi), method="linear")
    zi2 = griddata((X, Y), Z, (xi, yi), method="nearest")
    zi = np.where(np.isnan(zi), zi2, zi)

    fig = go.Figure(data=[go.Surface(x=xi, y=yi, z=zi, colorscale="Viridis")])
    fig.update_layout(
        title="Implied Volatility Surface",
        scene=dict(
            xaxis_title="Time to Expiration (years)",
            yaxis_title="Strike Price ($)",
            zaxis_title="Implied Volatility (%)",
        ),
    )
    return fig

