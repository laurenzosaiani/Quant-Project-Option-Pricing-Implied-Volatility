# pnl.py
"""
Profit and Loss (PnL) calculations for European options.

This module provides functions to compute option PnL under different assumptions:

Functions:
----------
- calculate_pnl_expiry(S_t, K, call_premium, put_premium)
    Profit/loss at expiry without discounting.

- calculate_pnl_present_value(S_t, K, r, T, call_premium, put_premium)
    Profit/loss in present value terms (discounted to time 0).

- calculate_pnl_opportunity_cost(S_t, K, r, T, call_premium, put_premium)
    Profit/loss adjusted for opportunity cost of capital.
"""

from typing import Tuple
import numpy as np


def calculate_pnl_expiry(
    S_t: list[float], 
    K: float, 
    call_premium: float, 
    put_premium: float
) -> Tuple[float, float]:
    S_T = S_t[-1]  # final stock price
    call_pnl = max(S_T - K, 0) - call_premium
    put_pnl = max(K - S_T, 0) - put_premium
    return call_pnl, put_pnl


def calculate_pnl_present_value(
    S_t: list[float], 
    K: float, 
    r: float, 
    T: float, 
    call_premium: float, 
    put_premium: float
) -> Tuple[float, float]:
    S_T = S_t[-1]
    discount = np.exp(-r * T)

    call_pnl_pv = max(S_T - K, 0) * discount - call_premium
    put_pnl_pv = max(K - S_T, 0) * discount - put_premium
    return call_pnl_pv, put_pnl_pv


def calculate_pnl_opportunity_cost(
    S_t: list[float], 
    K: float, 
    r: float, 
    T: float, 
    call_premium: float, 
    put_premium: float
) -> Tuple[float, float]:
    S_T = S_t[-1]
    growth = np.exp(r * T)

    call_pnl_oc = max(S_T - K, 0) - call_premium * growth
    put_pnl_oc = max(K - S_T, 0) - put_premium * growth
    return call_pnl_oc, put_pnl_oc
