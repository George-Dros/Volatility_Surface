
from scipy.optimize import brentq
from datetime import datetime
import numpy as np
from scipy.stats import norm


def Call_BS_Value(S, X, r, T, v, q):
    # Calculates the value of a call option (Black-Scholes formula for call options with dividends)
    # S is the share price at time T
    # X is the strike price
    # r is the risk-free interest rate
    # T is the time to maturity in years (days/365)
    # v is the volatility
    # q is the dividend yield

    if S <= 0 or X <= 0:
        return np.nan

    # If expired or basically expired: price is intrinsic
    if T <= 0:
        return max(S - X, 0.0)

    # If vol is basically zero: BS formula divides by ~0, so handle limit case
    if v <= 0:
        # In the zero-vol limit, the option is worth discounted intrinsic on the forward
        return max(S*np.exp(-q*T) - X*np.exp(-r*T), 0.0)

    d_1 = (np.log(S / X) + (r - q + v ** 2 * 0.5) * T) / (v * np.sqrt(T))
    d_2 = d_1 - v * np.sqrt(T)
    return S * np.exp(-q * T) * norm.cdf(d_1) - X * np.exp(-r * T) * norm.cdf(d_2)


def call_price_bounds(S, X, r, T, q):
    lower = max(S*np.exp(-q*T) - X*np.exp(-r*T), 0.0)
    upper = S*np.exp(-q*T)
    return lower, upper

def put_price_bounds(S, X, r, T, q):
    lower = max(X*np.exp(-r*T) - S*np.exp(-q*T), 0.0)
    upper = X*np.exp(-r*T)
    return lower, upper



def Call_IV(S, X, r, T, Call_Price, q, a=1e-6, b=5.0, xtol=1e-6):
    # Calculates the implied volatility for a call option with Brent's method
    # The first four parameters are explained in the Call_BS_Value function
    # Call_Price is the price of the call option
    # q is the dividend yield
    # Last three variables are needed for Brent's method
    if T <= 0 or S <= 0 or X <= 0:
        return np.nan

    low, high = call_price_bounds(S, X, r, T, q)
    if not (low <= Call_Price <= high):
        return np.nan

    def fcn(v):
        return Call_Price - Call_BS_Value(S, X, r, T, v, q)

    try:
        result = brentq(fcn, a=a, b=b, xtol=xtol)
        return np.nan if result <= xtol else result
    except ValueError:
        return np.nan



def Put_BS_Value(S, X, r, T, v, q):
    # Calculates the value of a put option (Black-Scholes formula for put options with dividends)
    # The parameters are explained in the Call_BS_Value function

    if S <= 0 or X <= 0:
        return np.nan
    if T <= 0:
        return max(X - S, 0.0)
    if v <= 0:
        return max(X*np.exp(-r*T) - S*np.exp(-q*T), 0.0)

    d_1 = (np.log(S / X) + (r - q + v ** 2 * 0.5) * T) / (v * np.sqrt(T))
    d_2 = d_1 - v * np.sqrt(T)
    return X * np.exp(-r * T) * norm.cdf(-d_2) - S * np.exp(-q * T) * norm.cdf(-d_1)


def Put_IV(S, X, r, T, Put_Price, q, a=1e-6, b=5.0, xtol=1e-6):
    # Calculates the implied volatility for a put option with Brent's method
    # The first four parameters are explained in the Call_BS_Value function
    # Put_Price is the price of the put option
    # q is the dividend yield
    # Last three variables are needed for Brent's method
    if T <= 0 or S <= 0 or X <= 0:
        return np.nan

    low, high = put_price_bounds(S, X, r, T, q)
    if not (low <= Put_Price <= high):
        return np.nan

    def fcn(v):
        return Put_Price - Put_BS_Value(S, X, r, T, v, q)

    try:
        result = brentq(fcn, a=a, b=b, xtol=xtol)
        return np.nan if result <= xtol else result
    except ValueError:
        return np.nan


def Calculate_IV_Call_Put(S, X, r, T, Option_Price, Put_or_Call, q):
    # This is a general function witch summarizes Call_IV and Put_IV (delivers the same results)
    # Can be used for a Lambda function within Pandas
    # The first four parameters are explained in the Call_BS_Value function
    # Put_or_Call:
    # 'C' returns the implied volatility of a call
    # 'P' returns the implied volatility of a put
    # Option_Price is the price of the option.
    # q is the dividend yield

    pc = str(Put_or_Call).upper()
    if pc == 'C':
        return Call_IV(S, X, r, T, Option_Price, q)
    if pc == 'P':
        return Put_IV(S, X, r, T, Option_Price, q)
    else:
        return np.nan


def calculate_time_to_expiration(expiration_date_str: str) -> float:
    """
    Calculate the time to expiration in years from today.

    Parameters:
    expiration_date_str (str): Expiration date in the format 'YYYY-MM-DD'

    Returns:
    float: Time to expiration in years
    """
    # Parse the expiration date string to a datetime object
    expiration_date = datetime.strptime(expiration_date_str, "%Y-%m-%d")

    # Get today's date
    current_date = datetime.now()

    # Calculate the number of days to expiration
    dt = expiration_date - current_date

    # Convert seconds to years (use 365 for simplicity)
    T = dt.total_seconds() / (365.0 * 24 * 3600)

    return max(T, 0.0)

