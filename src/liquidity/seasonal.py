"""Seasonal profile computation for THIN regime"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from datetime import datetime, timezone


class SeasonalProfile:
    """Compute and maintain seasonal liquidity profiles"""
    
    def __init__(self, params: dict):
        self.params = params
        self.liquidity_params = params.get('liquidity_regimes', {})
        
        self.recalc_hour_utc = self.liquidity_params.get('seasonal_profile_recalc_hour_utc', 0)
        self.window_days = self.liquidity_params.get('seasonal_profile_window_days', 30)
        self.padding_days = self.liquidity_params.get('seasonal_profile_padding_days', 7)
        
        # Cache: {date: {hour: {spread_z: float, depth_pct: float}}}
        self._profile_cache: Dict[str, Dict[int, Dict[str, float]]] = {}
        self._last_recalc_date: Optional[pd.Timestamp] = None
    
    def compute_seasonal_profile(
        self,
        df: pd.DataFrame,
        current_ts: pd.Timestamp
    ) -> Dict[int, Dict[str, float]]:
        """
        Compute seasonal profile.
        
        Computed daily at 00:00 UTC on data ending D−7 using [D−37, D−7] window; frozen for D.
        """
        # Check if we need to recompute
        current_date = current_ts.normalize()
        current_hour = current_ts.hour
        
        # Recompute at specified hour UTC
        if (self._last_recalc_date is None or
            current_date > self._last_recalc_date or
            (current_date == self._last_recalc_date and current_hour >= self.recalc_hour_utc)):
            
            # Calculate window: [D−37, D−7] ending D−7
            end_date = current_date - pd.Timedelta(days=self.padding_days)
            start_date = end_date - pd.Timedelta(days=self.window_days)
            
            # Filter data to window
            mask = (df['ts'] >= start_date) & (df['ts'] < end_date)
            window_df = df[mask].copy()
            
            if len(window_df) == 0:
                # No data, return empty profile
                return {}
            
            # Group by hour of day (UTC)
            window_df['hour'] = window_df['ts'].dt.hour
            
            profile = {}
            
            for hour in range(24):
                hour_data = window_df[window_df['hour'] == hour]
                
                if len(hour_data) == 0:
                    continue
                
                # Calculate spread z-score
                if 'spread_bps' in hour_data.columns:
                    spread_mean = hour_data['spread_bps'].mean()
                    spread_std = hour_data['spread_bps'].std()
                    
                    if spread_std > 0:
                        # Z-score for current hour's spread vs all hours
                        all_spread_mean = window_df['spread_bps'].mean()
                        all_spread_std = window_df['spread_bps'].std()
                        
                        if all_spread_std > 0:
                            spread_z = (spread_mean - all_spread_mean) / all_spread_std
                        else:
                            spread_z = 0.0
                    else:
                        spread_z = 0.0
                else:
                    spread_z = None
                
                # Calculate depth percentile
                if 'Depth5_bid_usd' in hour_data.columns and 'Depth5_ask_usd' in hour_data.columns:
                    hour_data['depth5_total'] = hour_data['Depth5_bid_usd'] + hour_data['Depth5_ask_usd']
                    depth_pct = (hour_data['depth5_total'] < window_df['Depth5_bid_usd'].add(window_df.get('Depth5_ask_usd', 0), fill_value=0)).sum() / len(window_df) * 100
                else:
                    depth_pct = None
                
                profile[hour] = {
                    'spread_z': spread_z,
                    'depth_pct': depth_pct
                }
            
            # Cache profile
            cache_key = current_date.strftime('%Y-%m-%d')
            self._profile_cache[cache_key] = profile
            self._last_recalc_date = current_date
            
            return profile
        
        # Return cached profile for current date
        cache_key = current_date.strftime('%Y-%m-%d')
        return self._profile_cache.get(cache_key, {})
    
    def get_seasonal_values(
        self,
        df: pd.DataFrame,
        current_ts: pd.Timestamp
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Get seasonal spread z-score and depth percentile for current hour.
        
        Returns:
            (spread_z, depth_pct)
        """
        profile = self.compute_seasonal_profile(df, current_ts)
        
        current_hour = current_ts.hour
        
        if current_hour in profile:
            hour_data = profile[current_hour]
            return hour_data.get('spread_z'), hour_data.get('depth_pct')
        
        return None, None

