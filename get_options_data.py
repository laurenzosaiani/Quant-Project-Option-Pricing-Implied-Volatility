# get_option_data.py
"""
Fetch options market data for a given stock using yfinance.

This module provides a function to retrieve the closest-to-ATM option prices,
strikes, time to expiry, current stock price, and dividend yield.
"""

import yfinance as yf
import datetime as dt
from typing import Tuple, List

def get_options_data(
    ticker_symbol: str, 
    num_options: int = 5
) -> Tuple[List[float], List[float], List[float], float, float, float]:
    """
    Fetch options market data for a given ticker.

    Parameters:
    -----------
    ticker_symbol : str
        Stock ticker symbol (e.g., 'AAPL').
    num_options : int
        Number of closest-to-ATM options to retrieve (default is 5).

    Returns:
    --------
    market_call_price : List[float]
        Last traded prices of the selected call options.
    market_put_price : List[float]
        Last traded prices of the selected put options.
    strikes : List[float]
        Strikes of the selected options.
    time_to_expiry : float
        Time to expiry in years (approximate).
    stock_price : float
        Current stock price.
    dividend_yield : float
        Annualized dividend yield (approximate).
    """
    ticker = yf.Ticker(ticker_symbol)
    
    # Get latest stock price
    stock_price = ticker.history(period="1d")['Close'].iloc[0]

    # Estimate annual dividend yield
    dividends = ticker.dividends
    if len(dividends) >= 4:
        annual_dividend = dividends[-4:].sum()
    else:
        annual_dividend = 0.0
    dividend_yield = annual_dividend / stock_price

    # Get option expiry date (use 16th expiry as example)
    expiries = ticker.options
    if len(expiries) < 1:
        raise ValueError("No options data available for this ticker.")
    
    expiry_str = expiries[min(16, len(expiries) - 1)]
    expiry_date = dt.datetime.strptime(expiry_str, "%Y-%m-%d")
    today = dt.datetime.today()
    time_to_expiry = max((expiry_date - today).days / 365, 0.0)

    # Get option chain
    option_chain = ticker.option_chain(expiry_str)
    calls = option_chain.calls.copy()
    puts = option_chain.puts.copy()

    # Sort options by distance from current stock price (closest to ATM)
    calls['distance'] = abs(calls['strike'] - stock_price)
    puts['distance'] = abs(puts['strike'] - stock_price)

    selected_calls = calls.nsmallest(num_options, 'distance')
    selected_puts = puts.nsmallest(num_options, 'distance')

    # Extract prices and strikes
    market_call_price = selected_calls['lastPrice'].tolist()
    market_put_price = selected_puts['lastPrice'].tolist()
    strikes = selected_calls['strike'].tolist()

    return market_call_price, market_put_price, strikes, time_to_expiry, stock_price, dividend_yield
