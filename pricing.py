# pricing.py
"""
Black-Scholes Option Pricing Module.

This module provides functions to calculate European call and put option prices
using the Black-Scholes model.
"""

import numpy as np
from scipy.stats import norm


def black_scholes_option_price(
    S: float, 
    K: float, 
    r: float, 
    q: float, 
    T: float, 
    sigma: float
) -> tuple[float, float]:
    """
    Calculate European call and put option prices using the Black-Scholes formula.

    Parameters:
    -----------
    S : float
        Current stock price.
    K : float
        Option strike price.
    r : float
        Risk-free interest rate (annualized, continuous compounding).
    q : float
        Dividend yield (annualized, continuous compounding).
    T : float
        Time to expiry in years.
    sigma : float
        Volatility of the underlying asset (annualized).

    Returns:
    --------
    call_price : float
        Price of the European call option.
    put_price : float
        Price of the European put option.
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        raise ValueError("Invalid input: S, K, sigma, and T must be positive.")

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    call_price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

    return call_price, put_price
