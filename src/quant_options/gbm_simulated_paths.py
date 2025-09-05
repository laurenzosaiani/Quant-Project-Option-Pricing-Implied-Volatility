# gbm_simulated_paths.py
"""
Stock price simulation using Geometric Brownian Motion (GBM).

This module provides a function to generate multiple stock price paths under the GBM model.
"""

import numpy as np


def gbm_stock_path(
    S0: float,
    r: float,
    sigma: float,
    T: float,
    steps_per_year: int = 252,
    n_paths: int = 100000,
) -> np.ndarray:
    """
    Simulate multiple stock price paths using the Geometric Brownian Motion (GBM) model.

    Parameters
    ----------
    S0 : float
        Initial stock price.
    r : float
        Risk-free rate (annualized, continuous compounding).
    sigma : float
        Volatility of the underlying asset (annualized).
    T : float
        Time horizon in years.
    steps_per_year : int, optional
        Number of steps per year (default=252).
    n_paths : int, optional
        Number of Monte Carlo paths to simulate (default=1000).

    Returns
    -------
    prices : np.ndarray
        Simulated stock price paths of shape (n_paths, steps + 1).
        Each row is a path; each column is a time step.
    """
    if S0 <= 0 or sigma <= 0 or T <= 0:
        raise ValueError("S0, sigma, and T must be positive.")

    N = int(steps_per_year * T)  # number of steps
    dt = 1 / steps_per_year

    # Random shocks: shape (n_paths, N)
    Z = np.random.normal(0, 1, size=(n_paths, N))

    # GBM increments
    increments = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z

    # Log prices
    log_paths = np.cumsum(increments, axis=1)
    log_paths = np.hstack([np.zeros((n_paths, 1)), log_paths])  # prepend initial log(S0)

    # Convert back to prices
    prices = S0 * np.exp(log_paths)

    return prices
