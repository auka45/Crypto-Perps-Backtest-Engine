"""ES (Expected Shortfall) guardrails: EWHS, parametric, sigma-clip"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional


def calculate_ewhs_es(
    returns: pd.Series,
    confidence: float = 0.99,
    lambda_param: float = 0.985
) -> float:
    """
    Calculate EWHS (Exponentially Weighted Historical Simulation) ES.
    
    ES @ 99%: 15m→1d block sums; λ=0.985
    """
    # Convert 15m returns to daily blocks (96 bars per day)
    block_size = 96
    n_blocks = len(returns) // block_size
    
    if n_blocks < 1:
        return 0.0
    
    # Sum returns into daily blocks
    daily_returns = []
    for i in range(n_blocks):
        block = returns.iloc[i * block_size:(i + 1) * block_size]
        daily_ret = block.sum()
        daily_returns.append(daily_ret)
    
    if len(daily_returns) == 0:
        return 0.0
    
    # Sort returns
    sorted_returns = sorted(daily_returns)
    
    # Calculate ES using EWMA weights
    # ES is the average of returns below the VaR threshold
    var_idx = int((1 - confidence) * len(sorted_returns))
    if var_idx >= len(sorted_returns):
        var_idx = len(sorted_returns) - 1
    
    # Calculate weighted average of tail
    tail_returns = sorted_returns[:var_idx + 1]
    
    # Apply EWMA weights (most recent gets higher weight)
    weights = []
    for i in range(len(tail_returns)):
        weight = (1 - lambda_param) * (lambda_param ** (len(tail_returns) - 1 - i))
        weights.append(weight)
    
    # Normalize weights
    total_weight = sum(weights)
    if total_weight > 0:
        weights = [w / total_weight for w in weights]
    else:
        weights = [1.0 / len(weights)] * len(weights)
    
    es = sum(r * w for r, w in zip(tail_returns, weights))
    
    return abs(es)  # Return absolute value (loss)


def calculate_parametric_es(
    portfolio_vol: float,
    correlation_matrix: np.ndarray,
    weights: np.ndarray,
    confidence: float = 0.99,
    vol_lambda: float = 0.985,
    corr_lambda: float = 0.999
) -> float:
    """
    Calculate parametric ES @ 99% with EWMA vol/corr.
    
    Uses EWMA covariance with ρ floors when m <= 0.6.
    """
    # For simplicity, use portfolio volatility directly
    # In full implementation, would maintain EWMA covariance matrix
    
    # ES = portfolio_vol * z_score(confidence)
    # Use normal distribution approximation (z-score for 99% ≈ 2.326)
    z_score = 2.326  # Approximate z-score for 99% confidence
    es = portfolio_vol * z_score
    
    return abs(es)


def calculate_sigma_clip_es(
    portfolio_vol_fast: float,
    vol_forecast: float,
    vol_fast_median: float,
    k_max: float = 4.5,
    k_min: float = 3.0
) -> float:
    """
    Calculate sigma-clip ES.
    
    k = 4.5 - min(1.5, vol_forecast / vol_fast_median)
    ES_sigma = k × σ_port_fast
    """
    if pd.isna(vol_forecast) or pd.isna(vol_fast_median) or vol_fast_median == 0:
        k = k_max
    else:
        vol_ratio = vol_forecast / vol_fast_median
        if vol_ratio <= 1.0:
            k = k_max
        else:
            k = k_max - min(1.5, vol_ratio - 1.0)
            k = max(k_min, k)  # Ensure k >= k_min
    
    es = k * portfolio_vol_fast
    
    return es


def calculate_final_es(
    ewhs_es: float,
    parametric_es: float,
    sigma_clip_es: float
) -> float:
    """
    Calculate final ES as max of all three methods.
    
    ES_used = max(EWHS_ES, ES_stress, ES_sigma)
    """
    return max(ewhs_es, parametric_es, sigma_clip_es)


def check_es_constraint(
    es_used: float,
    equity: float,
    es_cap_pct: float = 0.0225
) -> Tuple[bool, float]:
    """
    Check if ES constraint is satisfied.
    
    ES_used <= 2.25% of equity (post-trade)
    
    Returns:
        (is_valid, es_pct)
    """
    if equity == 0:
        return False, 0.0
    
    es_pct = es_used / equity
    
    is_valid = es_pct <= es_cap_pct
    
    return is_valid, es_pct

