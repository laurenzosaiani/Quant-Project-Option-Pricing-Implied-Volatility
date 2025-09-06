# implied_volatility.py
"""
Calculate implied volatility for European call options. 

This module provides a function to compute the vega-weighted average implied
volatility from market option prices using the Black-Scholes model.
"""

from typing import List 
import numpy as np
from scipy.stats import norm
from scipy.optimize import newton, brentq
from src.quant_options.pricing import black_scholes_option_price


def get_implied_volatility(
    market_call_prices: List[float],
    S: float,
    strikes: List[float],
    r: float,
    q: float,
    T: float,
    num_options: int,
    sigma_init: float = 0.1
) -> float:
    """
    Compute the vega-weighted average implied volatility for a set of call options.

    Parameters:
    -----------
    market_call_prices : List[float]
        Observed market prices of call options.
    S : float
        Current stock price.
    strikes : List[float]
        Strike prices corresponding to the call options.
    r : float
        Risk-free rate (annualized, continuous compounding).
    q : float
        Dividend yield (annualized, continuous compounding).
    T : float
        Time to expiry in years.
    num_options : int
        Number of options to consider.
    sigma_init : float
        Initial guess for volatility in Newton-Raphson solver.

    Returns:
    --------
    float
        Vega-weighted average implied volatility.
    """
    iv_list = []
    vega_list = []
    
    for i in range(num_options):
        market_price = market_call_prices[i]
        K = strikes[i]

        # Define the difference function
        def diff(sigma):
            call_price, _ = black_scholes_option_price(S, K, r, q, T, sigma)
            return call_price - market_price

        # Try Newton-Raphson first
        try:
            iv = newton(diff, sigma_init, tol=1e-6, maxiter=100)
            if iv <= 0:  # fallback if solver returns negative
                raise RuntimeError
        except (RuntimeError, OverflowError):
            # Fallback to brentq method
            try:
                iv = brentq(diff, 1e-9, 5.0)
            except ValueError:
                iv = 1e-9  # fallback tiny volatility

        iv_list.append(iv)

        # Compute vega for weighting
        d1 = (np.log(S / K) + (r - q + 0.5 * iv ** 2) * T) / (iv * np.sqrt(T))
        vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)
        vega_list.append(vega)

    # Vega-weighted average
    iv_vega_weighted = sum(iv * vega for iv, vega in zip(iv_list, vega_list)) / sum(vega_list)
    return iv_vega_weighted


