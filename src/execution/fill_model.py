"""Stop-run fill model (deterministic)"""
import pandas as pd
import numpy as np
from typing import Literal, Optional, Tuple


def calculate_slippage(
    order_notional: float,
    adv_60m: float,
    base_slip_intercept: float = 2.0,
    base_slip_slope: float = 20.0,
    regime_adder: float = 0.0,
    governed_universe: bool = False,
    fallback_participation_est: float = 0.001,
    stress_slip: bool = False
) -> Tuple[float, float]:
    """
    Calculate slippage in bps.
    
    Base slippage: 2 + 20 × participation_pct
    Total: base + regime_adder
    
    FIX 5: If ADV_60m <= 0:
    - If governed_universe: raise ValueError (block entries)
    - Otherwise: use conservative fallback: max(6, 2 + 20×participation_est) or max(8, ...) if stress_slip
    """
    if adv_60m <= 0:
        if governed_universe:
            raise ValueError("ADV_60m must be positive when --governed-universe is enabled")
        else:
            # FIX 5: Conservative fallback for non-governed universe
            participation_pct = fallback_participation_est
            base_slip_bps = base_slip_intercept + base_slip_slope * participation_pct
            # Stress test: use 8 bps minimum instead of 6 bps
            min_slip_bps = 8.0 if stress_slip else 6.0
            total_slip_bps = max(min_slip_bps, base_slip_bps) + regime_adder
            return total_slip_bps, participation_pct
    
    participation_pct = order_notional / adv_60m
    
    base_slip_bps = base_slip_intercept + base_slip_slope * participation_pct
    total_slip_bps = base_slip_bps + regime_adder
    
    return total_slip_bps, participation_pct


def fill_stop_run(
    trigger_price: float,
    side: Literal['LONG', 'SHORT'],
    bar_high: float,
    bar_low: float,
    bar_mid: float,
    slippage_bps: float
) -> Tuple[float, bool]:
    """
    Stop-run fill model.
    
    slip_px = mid_bar × slip_bps_total / 10,000
    Buy fills: min(High, trigger + slip_px)
    Sell fills: max(Low, trigger − slip_px)
    Bound within [Low, High]
    
    Returns:
        (fill_price, gap_through)
        gap_through: True if fill hit bar bound and requested trigger±slip exceeded that bound
    """
    slip_px = bar_mid * slippage_bps / 10000.0
    gap_through = False
    
    if side == 'LONG':
        # Buy fill
        requested_fill = trigger_price + slip_px
        fill_price = min(bar_high, requested_fill)
        fill_price = max(bar_low, fill_price)  # Bound to [Low, High]
        # Gap-through: fill hit High and requested exceeded High
        if fill_price == bar_high and requested_fill > bar_high:
            gap_through = True
    else:  # SHORT
        # Sell fill
        requested_fill = trigger_price - slip_px
        fill_price = max(bar_low, requested_fill)
        fill_price = min(bar_high, fill_price)  # Bound to [Low, High]
        # Gap-through: fill hit Low and requested was below Low
        if fill_price == bar_low and requested_fill < bar_low:
            gap_through = True
    
    return fill_price, gap_through


def calculate_adv_60m(notional_series: pd.Series, current_idx: int) -> float:
    """
    Calculate 60-minute ADV (sum of last 4 bars notional).
    
    ADV_60m = sum(notional of last 4 bars)
    """
    if current_idx < 3:
        # Not enough bars
        window = notional_series.iloc[:current_idx + 1]
    else:
        window = notional_series.iloc[current_idx - 3:current_idx + 1]
    
    return window.sum()


def check_partial_fill_allowed(
    signal_age_bars: int,
    es_headroom_pct: float,
    extra_slip_bps: float,
    max_signal_age: int = 2,
    min_es_headroom: float = 0.005,
    max_extra_slip: float = 8.0
) -> bool:
    """
    Check if partial fill taker sweep is allowed next bar.
    
    Allowed if:
    - signal_age <= 2 bars
    - ES headroom >= 0.5%
    - extra slip <= 8 bps
    """
    return (
        signal_age_bars <= max_signal_age and
        es_headroom_pct >= min_es_headroom and
        extra_slip_bps <= max_extra_slip
    )

