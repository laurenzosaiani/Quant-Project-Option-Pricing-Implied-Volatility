# gbm_simulated_paths.py
"""
Stock price simulation using Geometric Brownian Motion (GBM).

This module provides a function to generate a stock price path under the GBM model.
"""

from typing import Tuple
import numpy as np


def gbm_stock_path(
    S0: float, 
    r: float, 
    sigma: float, 
    T: float, 
    steps_per_year: int = 252
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate a stock price path using the Geometric Brownian Motion (GBM) model.

    Parameters:
    -----------
    S0 : float
        Initial stock price.
    r : float
        Risk-free rate (annualized, continuous compounding).
    sigma : float
        Volatility of the underlying asset (annualized).
    T : float
        Time horizon in years.
    steps_per_year : int, optional
        Number of simulation steps per year (default = 252, trading days).

    Returns:
    --------
    prices : np.ndarray
        Simulated stock price path (length = steps + 1).
    times : np.ndarray
        Corresponding time steps in years.
    """
    if S0 <= 0 or sigma <= 0 or T <= 0:
        raise ValueError("Inputs S0, sigma, and T must be positive.")

    N = int(steps_per_year * T)  # total number of steps
    dt = 1 / steps_per_year

    # Pre-generate random shocks
    Z = np.random.normal(0, 1, N)

    # Drift + diffusion
    increments = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z

    # Log prices (cumulative sum of increments)
    log_prices = np.cumsum(increments)
    log_prices = np.insert(log_prices, 0, 0.0)  # start at log(S0)

    # Convert back to price path
    prices = S0 * np.exp(log_prices)
    times = np.linspace(0, T, N + 1)

    return prices, times
