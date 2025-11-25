"""AVWAP (Anchored Volume-Weighted Average Price) with re-anchoring logic"""
import pandas as pd
import numpy as np
from typing import Optional, Tuple


class AVWAP:
    """Anchored VWAP calculator with incremental updates and re-anchoring"""
    
    def __init__(self, anchor_ts: pd.Timestamp, initial_price: float, initial_volume: float):
        self.anchor_ts = anchor_ts
        self.cumulative_price_volume = initial_price * initial_volume
        self.cumulative_volume = initial_volume
        self.last_recompute_ts = anchor_ts
        self.reanchor_count = 0
    
    def update(self, price: float, volume: float) -> float:
        """Incremental update of AVWAP"""
        self.cumulative_price_volume += price * volume
        self.cumulative_volume += volume
        
        if self.cumulative_volume > 0:
            return self.cumulative_price_volume / self.cumulative_volume
        return np.nan
    
    def get_value(self) -> float:
        """Get current AVWAP value"""
        if self.cumulative_volume > 0:
            return self.cumulative_price_volume / self.cumulative_volume
        return np.nan
    
    def reanchor(self, new_anchor_ts: pd.Timestamp, initial_price: float, initial_volume: float):
        """Re-anchor AVWAP at new timestamp"""
        self.anchor_ts = new_anchor_ts
        self.cumulative_price_volume = initial_price * initial_volume
        self.cumulative_volume = initial_volume
        self.last_recompute_ts = new_anchor_ts
        self.reanchor_count += 1


def compute_avwap(
    df: pd.DataFrame,
    anchor_ts: Optional[pd.Timestamp] = None,
    ema50: Optional[pd.Series] = None,
    vol_forecast: Optional[pd.Series] = None,
    vol_fast_median: Optional[pd.Series] = None,
    avwap_drift_base_bps: float = 10.0
) -> Tuple[pd.Series, pd.Series]:
    """
    Compute AVWAP with re-anchoring logic.
    
    Re-anchoring triggers:
    1. |AVWAP - EMA50| > 5.0%
    2. Daily recompute drift > adaptive threshold (10 bps * max(1, vol_forecast/vol_fast_median))
    
    Returns:
        - avwap_series: AVWAP values
        - reanchor_flags: Boolean series indicating re-anchoring events
    """
    if anchor_ts is None:
        # Default anchor to first bar
        anchor_ts = df['ts'].iloc[0]
    
    avwap_series = pd.Series(index=df.index, dtype=float)
    reanchor_flags = pd.Series(index=df.index, dtype=bool)
    
    avwap_calc = AVWAP(
        anchor_ts=anchor_ts,
        initial_price=df['close'].iloc[0],
        initial_volume=df['volume'].iloc[0]
    )
    
    # Track last daily recompute
    last_daily_recompute = df['ts'].iloc[0].normalize()
    
    for i in range(len(df)):
        current_ts = df['ts'].iloc[i]
        current_price = df['close'].iloc[i]
        current_volume = df['volume'].iloc[i]
        
        # Check if we need daily full recompute (at start of new UTC day)
        current_day = current_ts.normalize()
        if current_day > last_daily_recompute:
            # Daily full recompute check
            if i > 0:
                prev_avwap = avwap_calc.get_value()
                # Recompute from anchor
                avwap_calc = AVWAP(
                    anchor_ts=avwap_calc.anchor_ts,
                    initial_price=df.loc[df['ts'] >= avwap_calc.anchor_ts, 'close'].iloc[0],
                    initial_volume=df.loc[df['ts'] >= avwap_calc.anchor_ts, 'volume'].iloc[0]
                )
                
                # Rebuild cumulative values from anchor to current
                anchor_idx = df[df['ts'] >= avwap_calc.anchor_ts].index[0]
                for j in range(anchor_idx, i):
                    avwap_calc.update(df['close'].iloc[j], df['volume'].iloc[j])
                
                new_avwap = avwap_calc.get_value()
                
                # Check drift threshold
                if not pd.isna(prev_avwap) and not pd.isna(new_avwap) and prev_avwap > 0:
                    drift_bps = abs(new_avwap - prev_avwap) / prev_avwap * 10000
                    
                    # Adaptive threshold
                    if vol_forecast is not None and vol_fast_median is not None:
                        vol_scale = max(1.0, vol_forecast.iloc[i] / vol_fast_median.iloc[i]) if not pd.isna(vol_forecast.iloc[i]) and not pd.isna(vol_fast_median.iloc[i]) else 1.0
                    else:
                        vol_scale = 1.0
                    
                    adaptive_threshold = avwap_drift_base_bps * vol_scale
                    
                    if drift_bps > adaptive_threshold:
                        # Re-anchor
                        avwap_calc.reanchor(current_ts, current_price, current_volume)
                        reanchor_flags.iloc[i] = True
            
            last_daily_recompute = current_day
        
        # Check EMA50 distance trigger
        if ema50 is not None and not pd.isna(ema50.iloc[i]):
            current_avwap = avwap_calc.get_value()
            if not pd.isna(current_avwap) and current_avwap > 0:
                ema50_dist_pct = abs(current_avwap - ema50.iloc[i]) / current_avwap
                if ema50_dist_pct > 0.05:  # 5.0%
                    # Re-anchor
                    avwap_calc.reanchor(current_ts, current_price, current_volume)
                    reanchor_flags.iloc[i] = True
        
        # Update AVWAP
        avwap_value = avwap_calc.update(current_price, current_volume)
        avwap_series.iloc[i] = avwap_value
    
    return avwap_series, reanchor_flags

