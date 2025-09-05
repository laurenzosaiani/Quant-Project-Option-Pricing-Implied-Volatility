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
