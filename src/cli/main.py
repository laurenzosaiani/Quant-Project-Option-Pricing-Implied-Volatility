"""
Main script for option pricing and Monte Carlo simulation (vectorised version) using argparse.

This program:
1. Fetches market option data from Yahoo Finance.
2. Estimates implied volatility using Black-Scholes.
3. Computes theoretical option prices.
4. Runs Monte Carlo simulations of stock prices using GBM.
5. Calculates profit/loss (PnL).
6. Plots cumulative PnL, terminal stock price distribution, and payoff overlay.
"""

import argparse
import numpy as np

from src.quant_options.pricing import black_scholes_option_price
from src.quant_options.pnl import calculate_pnl_present_value
from src.quant_options.analysis import plot_cumulative_pnl, plot_terminal_distribution, plot_payoff_vs_distribution
from src.quant_options.get_options_data import get_options_data
from src.quant_options.implied_volatility import get_implied_volatility
from src.quant_options.gbm_simulated_paths import gbm_stock_path


def parse_arguments():
    parser = argparse.ArgumentParser(description="Option pricing & Monte Carlo simulation.")
    parser.add_argument("--ticker", type=str, help="Stock ticker symbol")
    parser.add_argument("--call-prem", type=float, help="Call option premium")
    parser.add_argument("--put-prem", type=float, help="Put option premium")
    parser.add_argument("--expiry-years", type=float, help="Time to expiry in years")
    parser.add_argument("--strike", type=float, help="Strike price")
    parser.add_argument("--rate", type=float, default=0.04, help="Risk-free interest rate")
    parser.add_argument("--sims", type=int, default=100_000, help="Number of Monte Carlo simulations")
    return parser.parse_args()


def main():
    args = parse_arguments()

    # Fallback to input() if any required argument is missing
    ticker_symbol = args.ticker or input("Enter the stock ticker symbol: ").upper()
    C = args.call_prem or float(input("Enter your option's call premium: "))
    P = args.put_prem or float(input("Enter your option's put premium: "))
    T = args.expiry_years or float(input("Enter your option's time to expiry (in years): "))
    K = args.strike or float(input("Enter your option's strike price: "))
    r = args.rate
    num_sims = args.sims
    steps_per_year = 252

    # Fetch market data
    call_prices_mk, put_prices_mk, strikes, t_mk, S, q = get_options_data(ticker_symbol, num_options=5)

    # Estimate implied volatility
    sigma = get_implied_volatility(call_prices_mk, S, strikes, r, q, t_mk, num_options=5)
    print(f"\nEstimated vega-weighted implied volatility: {sigma:.4f}")

    # Fair prices
    call_price_bs, put_price_bs = black_scholes_option_price(S, K, r, q, T, sigma)
    print(
        f"\nTheoretical fair prices under Black-Scholes:"
        f"\n  Call Price: ${call_price_bs:.2f}"
        f"\n  Put Price : ${put_price_bs:.2f}"
    )

    # Monte Carlo simulation
    S_paths = gbm_stock_path(S, r, sigma, T, steps_per_year, n_paths=num_sims)
    S_T = S_paths[:, -1]  # terminal prices

    # PnL at present value
    discount = np.exp(-r * T)
    call_pnl = (np.maximum(S_T - K, 0) * discount) - C
    put_pnl = (np.maximum(K - S_T, 0) * discount) - P
    cumulative_call_pnl = np.cumsum(call_pnl)
    cumulative_put_pnl = np.cumsum(put_pnl)
    num_of_sims = np.arange(1, num_sims + 1)

    expected_pnl_call = (call_price_bs - C) * discount
    expected_pnl_put = (put_price_bs - P) * discount
    print(
        f"\nExpected returns (present value):"
        f"\n  Call Option: ${expected_pnl_call:.2f}"
        f"\n  Put Option : ${expected_pnl_put:.2f}"
    )

    # Plots
    plot_cumulative_pnl(num_of_sims.tolist(), cumulative_call_pnl.tolist(), cumulative_put_pnl.tolist())
    plot_terminal_distribution(S_T, K)
    plot_payoff_vs_distribution(S_T, K, C, P)


if __name__ == "__main__":
    main()
