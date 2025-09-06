"""
Main script for option pricing and Monte Carlo simulation (vectorised version).

This program:
1. Fetches market option data from Yahoo Finance.
2. Estimates implied volatility using Black-Scholes.
3. Computes theoretical option prices.
4. Runs Monte Carlo simulations of stock prices using GBM.
5. Calculates profit/loss (PnL).
6. Plots cumulative PnL, terminal stock price distribution, and payoff overlay.
"""

import numpy as np

from src.quant_options.pricing import black_scholes_option_price
from src.quant_options.pnl import calculate_pnl_present_value
from src.quant_options.analysis import plot_cumulative_pnl, plot_terminal_distribution, plot_payoff_vs_distribution
from src.quant_options.get_option_data import get_options_data
from src.quant_options.implied_volatility import get_implied_volatility
from src.quant_options.gbm_simulated_paths import gbm_stock_path



def main() -> None:
    """Run option pricing, implied volatility estimation, and Monte Carlo simulation."""
    # Parameters
    r = 0.04  # risk-free rate
    num_options = 5
    num_sims = 100_000
    steps_per_year = 252

    # User inputs
    ticker_symbol = input("Enter the stock ticker symbol: ").upper()
    C = float(input("Enter your option's call premium: "))
    P = float(input("Enter your option's put premium: "))
    T = float(input("Enter your option's time to expiry (in years): "))
    K = float(input("Enter your option's strike price: "))

    # Fetch market data
    call_prices_mk, put_prices_mk, strikes, t_mk, S, q = get_options_data(
        ticker_symbol, num_options
    )

    # Estimate implied volatility
    sigma = get_implied_volatility(call_prices_mk, S, strikes, r, q, t_mk, num_options)
    print(f"\nEstimated vega-weighted implied volatility: {sigma:.4f}")

    # Fair prices
    call_price_bs, put_price_bs = black_scholes_option_price(S, K, r, q, T, sigma)
    print(
        f"\nTheoretical fair prices under Black-Scholes:"
        f"\n  Call Price: ${call_price_bs:.2f}"
        f"\n  Put Price : ${put_price_bs:.2f}"
    )

    # Vectorised Monte Carlo simulation (n_paths = num_sims)
    S_paths = gbm_stock_path(S, r, sigma, T, steps_per_year, n_paths=num_sims)
    S_T = S_paths[:, -1]  # terminal prices

    # Vectorised PnL at present value
    discount = np.exp(-r * T)
    call_pnl = (np.maximum(S_T - K, 0) * discount) - C
    put_pnl = (np.maximum(K - S_T, 0) * discount) - P

    cumulative_call_pnl = np.cumsum(call_pnl)
    cumulative_put_pnl = np.cumsum(put_pnl)
    num_of_sims = np.arange(1, num_sims + 1)

    # Expected PnL
    expected_pnl_call = (call_price_bs - C) * discount
    expected_pnl_put = (put_price_bs - P) * discount

    print(
        f"\nExpected returns (present value):"
        f"\n  Call Option: ${expected_pnl_call:.2f}"
        f"\n  Put Option : ${expected_pnl_put:.2f}"
    )

    # Plot cumulative PnL
    plot_cumulative_pnl(num_of_sims.tolist(), cumulative_call_pnl.tolist(), cumulative_put_pnl.tolist())

    # --- New visuals ---
    plot_terminal_distribution(S_T, K)
    plot_payoff_vs_distribution(S_T, K, C, P)


if __name__ == "__main__":
    main()
