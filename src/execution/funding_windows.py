"""Funding window enforcement"""
import pandas as pd
from typing import Dict, Optional
from datetime import timedelta


def get_funding_times(current_ts: pd.Timestamp) -> Dict[str, pd.Timestamp]:
    """
    Get funding times for current UTC day.
    Binance funding occurs at 00:00, 08:00, 16:00 UTC.
    
    Returns:
        Dict with 'next_funding' timestamp
    """
    # Binance funding schedule: 00:00, 08:00, 16:00 UTC
    funding_hours = [0, 8, 16]
    
    current_hour = current_ts.hour
    current_date = current_ts.normalize()
    
    # Find next funding time
    next_funding = None
    for hour in funding_hours:
        funding_ts = current_date + timedelta(hours=hour)
        if funding_ts > current_ts:
            next_funding = funding_ts
            break
    
    # If no funding today, use first funding tomorrow
    if next_funding is None:
        next_funding = current_date + timedelta(days=1, hours=0)
    
    return {
        'next_funding': next_funding,
        'prev_funding': next_funding - timedelta(hours=8)  # Previous funding
    }


def check_funding_window(
    current_ts: pd.Timestamp,
    funding_throttle_minutes: Optional[float] = 15.0,
    squeeze_disable_minutes: Optional[float] = 7.5
) -> Dict[str, bool]:
    """
    Check if we're in a funding window.
    
    Returns:
        {
            'block_entries': bool,  # True if ±15min from funding
            'disable_squeeze': bool  # True if ±7.5min from funding
        }
    """
    funding_times = get_funding_times(current_ts)
    next_funding = funding_times['next_funding']
    prev_funding = funding_times['prev_funding']
    
    # Calculate time to/from funding
    time_to_next = (next_funding - current_ts).total_seconds() / 60.0
    time_from_prev = (current_ts - prev_funding).total_seconds() / 60.0
    
    # Use defaults if None
    funding_throttle = funding_throttle_minutes if funding_throttle_minutes is not None else 15.0
    squeeze_disable = squeeze_disable_minutes if squeeze_disable_minutes is not None else 7.5
    
    # Block entries ±15 minutes
    block_entries = (
        time_to_next <= funding_throttle or
        time_from_prev <= funding_throttle
    )
    
    # Disable SQUEEZE ±7.5 minutes (engine-agnostic: parameter name kept for compatibility)
    disable_squeeze = (
        time_to_next <= squeeze_disable or
        time_from_prev <= squeeze_disable
    )
    
    return {
        'block_entries': block_entries,
        'disable_squeeze': disable_squeeze,
        'time_to_next_funding': time_to_next,
        'time_from_prev_funding': time_from_prev
    }

