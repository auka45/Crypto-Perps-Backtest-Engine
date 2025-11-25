"""Position sizing: vol-based sizing with module factors"""
import pandas as pd
import numpy as np
from typing import Dict, Optional


def calculate_size_multiplier(
    vol_forecast: float,
    vol_fast_median: float,
    returns_15m: pd.Series,
    sigma_15m: float,
    slope_z: float,
    params: dict
) -> float:
    """
    Calculate size multiplier m.
    
    Base: m = clamp(1 / (1 + vol_forecast / vol_fast_median), 0.3, 1.0)
    Jump filter: if any of last 6 bars has |r| >= 3.0 × σ̂15m, m = min(m, 0.5)
    Momentum safeguard: if slope_z >= 2.0, m = max(m, 0.6)
    """
    sizing_params = params.get('sizing', {})
    size_mult_min = sizing_params.get('size_mult_min', 0.3)
    size_mult_max = sizing_params.get('size_mult_max', 1.0)
    jump_sigma_mult = sizing_params.get('jump_sigma_mult', 3.0)
    jump_lookback_bars = sizing_params.get('jump_lookback_bars', 6)
    momentum_safeguard_slopez = sizing_params.get('momentum_safeguard_slopez', 2.0)
    momentum_min_size_mult = sizing_params.get('momentum_min_size_mult', 0.6)
    
    if pd.isna(vol_forecast) or pd.isna(vol_fast_median) or vol_fast_median == 0:
        return size_mult_min
    
    # Base multiplier
    m = 1.0 / (1.0 + vol_forecast / vol_fast_median)
    m = max(size_mult_min, min(size_mult_max, m))
    
    # Jump filter
    if len(returns_15m) >= jump_lookback_bars and not pd.isna(sigma_15m) and sigma_15m > 0:
        recent_returns = returns_15m.iloc[-jump_lookback_bars:]
        jump_threshold = jump_sigma_mult * sigma_15m
        has_jump = (abs(recent_returns) >= jump_threshold).any()
        if has_jump:
            m = min(m, 0.5)
    
    # Momentum safeguard
    if not pd.isna(slope_z) and slope_z >= momentum_safeguard_slopez:
        m = max(m, momentum_min_size_mult)
    
    return m


def calculate_position_size(
    equity: float,
    entry_price: float,
    stop_price: float,
    size_mult: float,
    module_factor: float,
    r_base: float,
    step_size: float
) -> float:
    """
    Calculate position quantity.
    
    risk_$ = equity × r_base × module_factor × m
    qty_raw = risk_$ / |entry - stop|
    Round down to stepSize
    """
    # Calculate risk amount
    risk_dollar = equity * r_base * module_factor * size_mult
    
    # Calculate stop distance
    stop_distance = abs(entry_price - stop_price)
    
    if stop_distance == 0:
        return 0.0
    
    # Calculate raw quantity
    qty_raw = risk_dollar / stop_distance
    
    # Round down to stepSize
    if step_size > 0:
        qty = np.floor(qty_raw / step_size) * step_size
    else:
        qty = qty_raw
    
    return max(0.0, qty)


def calculate_max_possible_notional(
    equity: float,
    entry_price_estimate: float,
    stop_distance_estimate: float,
    size_mult: float,
    module_factors: Dict[str, float],
    r_base: float
) -> float:
    """
    Calculate max_possible_notional for VACUUM/THIN detection.
    
    Compute ex-ante over modules that could signal on t+1:
    For each module M: derive stop distance and entry estimate known at t,
    compute risk_$, qty_max, notional_max. Keep the maximum across modules.
    """
    max_notional = 0.0
    
    for module, module_factor in module_factors.items():
        # Calculate risk for this module
        risk_dollar = equity * r_base * module_factor * size_mult
        
        # Calculate max quantity
        if stop_distance_estimate > 0:
            qty_max = risk_dollar / stop_distance_estimate
        else:
            qty_max = 0.0
        
        # Calculate notional
        notional_max = qty_max * entry_price_estimate
        
        max_notional = max(max_notional, notional_max)
    
    return max_notional


def get_module_factor(module: str, params: dict) -> float:
    """Get module risk factor"""
    sizing_params = params.get('sizing', {})
    module_factors = sizing_params.get('module_factors', {})
    return module_factors.get(module, 1.0)

