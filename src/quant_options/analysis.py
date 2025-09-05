# analysis.py
"""
Analysis and visualization tools for option simulations.

This module provides plotting functions to analyze cumulative PnL from Monte Carlo simulations.
"""

from typing import List
import matplotlib.pyplot as plt


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
    ST: np.ndarray, K: float, call_prem: float, put_prem: float
) -> None:
    """
    Overlay option payoffs with distribution of simulated terminal stock prices.

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
    """
    x = np.linspace(min(ST), max(ST), 400)
    call_payoff = np.maximum(x - K, 0.0) - call_prem
    put_payoff = np.maximum(K - x, 0.0) - put_prem

    plt.figure(figsize=(10, 6))
    plt.hist(ST, bins=50, density=True, alpha=0.5, label="Simulated $S_T$")
    plt.plot(x, call_payoff, "g-", lw=2, label="Call Payoff (net)")
    plt.plot(x, put_payoff, "r-", lw=2, label="Put Payoff (net)")

    # Strike line
    plt.axvline(K, color="black", linestyle="--", lw=1.5, label=f"Strike K={K}")

    plt.title("Option Payoff vs Distribution of Terminal Stock Price")
    plt.xlabel("Price at Expiry $S_T$")
    plt.ylabel("Payoff / Density")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()
