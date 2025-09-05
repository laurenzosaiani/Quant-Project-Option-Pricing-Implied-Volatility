# analysis.py
"""
Analysis and visualization tools for option simulations.

This module provides plotting functions to analyze cumulative PnL from Monte Carlo simulations.
"""

from typing import List
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import lognorm

def plot_cumulative_pnl(
    num_of_sims: List[int],
    cumulative_call_pnl: List[float],
    cumulative_put_pnl: List[float]
) -> None:
    """
    Plot the cumulative PnL of call and put options across Monte Carlo simulations.

    Parameters:
    -----------
    num_of_sims : List[int]
        Simulation indices (e.g., [1, 2, ..., N]).
    cumulative_call_pnl : List[float]
        Cumulative profit/loss from call options.
    cumulative_put_pnl : List[float]
        Cumulative profit/loss from put options.

    Returns:
    --------
    None
        Displays a matplotlib plot.
    """
    plt.figure(figsize=(10, 6))

    plt.plot(num_of_sims, cumulative_call_pnl, label="Call Cumulative PnL", linewidth=1.5)
    plt.plot(num_of_sims, cumulative_put_pnl, label="Put Cumulative PnL", linewidth=1.5)

    plt.axhline(0, color="black", linestyle="--", linewidth=1)

    plt.xlabel("Simulation #")
    plt.ylabel("Cumulative PnL")
    plt.title("Monte Carlo Cumulative PnL")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.xlim(0, len(num_of_sims))

    plt.tight_layout()
    plt.show()


def plot_terminal_distribution(ST: np.ndarray, K: float) -> None:
    """
    Plot histogram of simulated terminal stock prices S_T,
    with fitted lognormal PDF and strike marker.

    Parameters
    ----------
    ST : np.ndarray
        Array of simulated terminal stock prices.
    K : float
        Strike price.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(ST, bins=50, density=True, alpha=0.6, label="Simulated $S_T$")

    # Fit lognormal PDF to terminal prices
    shape, loc, scale = lognorm.fit(ST, floc=0)
    x = np.linspace(min(ST), max(ST), 300)
    pdf = lognorm.pdf(x, shape, loc, scale)
    plt.plot(x, pdf, "r--", lw=2, label="Fitted lognormal PDF")

    # Strike line
    plt.axvline(K, color="black", linestyle="--", lw=1.5, label=f"Strike K={K}")

    plt.title("Distribution of Terminal Stock Prices $S_T$")
    plt.xlabel("Price at Expiry")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()


def plot_payoff_vs_distribution(
    ST: np.ndarray, K: float, call_prem: float, put_prem: float, show_histogram: bool = True
) -> None:
    """
    Plot option payoffs (in $) vs simulated terminal stock prices.
    Optionally overlay a histogram of terminal stock prices on a secondary axis.

    Parameters
    ----------
    ST : np.ndarray
        Array of simulated terminal stock prices.
    K : float
        Strike price.
    call_prem : float
        Premium paid for the call option.
    put_prem : float
        Premium paid for the put option.
    show_histogram : bool, default True
        Whether to overlay histogram of terminal stock prices.
    """
    x = np.linspace(min(ST), max(ST), 400)
    call_payoff = np.maximum(x - K, 0.0) - call_prem
    put_payoff = np.maximum(K - x, 0.0) - put_prem

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot payoff curves in dollars
    ax1.plot(x, call_payoff, "g-", lw=2, label="Call Payoff ($)")
    ax1.plot(x, put_payoff, "r-", lw=2, label="Put Payoff ($)")
    ax1.set_xlabel("Terminal Stock Price $S_T$")
    ax1.set_ylabel("Option Payoff ($)")
    ax1.axvline(K, color="black", linestyle="--", lw=1.5, label=f"Strike K={K}")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="upper left")

    # Optional histogram on secondary axis
    if show_histogram:
        ax2 = ax1.twinx()
        ax2.hist(ST, bins=50, density=True, alpha=0.3, color="blue", label="Simulated $S_T$")
        ax2.set_ylabel("Density")
        ax2.legend(loc="upper right")

    plt.title("Option Payoff vs Terminal Stock Price")
    plt.tight_layout()
    plt.show()
