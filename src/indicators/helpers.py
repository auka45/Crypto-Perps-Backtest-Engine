"""Helper indicators: rv_pct, slope_z, vol forecast, vol_fast_median"""
import pandas as pd
import numpy as np
from typing import Tuple


def rv_pct(close: pd.Series, window_days: int = 60) -> pd.Series:
    """
    Realized volatility percentile: percentile of 15m sigma over trailing window_days.
    Returns percentile (0-100) of current 15m sigma vs historical distribution.
    """
    # Calculate 15m log returns
    returns = np.log(close / close.shift(1))
    
    # Rolling standard deviation (15m sigma)
    rolling_std = returns.rolling(window=1).std()  # Single bar std (or use small window)
    
    # For each bar, calculate percentile vs trailing window_days
    # window_days = 60 days = 60 * 24 * 4 = 5760 bars (15m)
    window_bars = window_days * 24 * 4
    
    percentiles = []
    for i in range(len(rolling_std)):
        if i < window_bars:
            percentiles.append(np.nan)
        else:
            # Get historical window
            hist_window = rolling_std.iloc[i - window_bars:i]
            current_val = rolling_std.iloc[i]
            
            if pd.isna(current_val) or len(hist_window.dropna()) == 0:
                percentiles.append(np.nan)
            else:
                # Calculate percentile
                pct = (hist_window.dropna() < current_val).sum() / len(hist_window.dropna()) * 100
                percentiles.append(pct)
    
    return pd.Series(percentiles, index=close.index)


def slope_z(close: pd.Series, macd: pd.Series, lookback_bars: int = 20, history_days: int = 60) -> pd.Series:
    """
    Z-score of Δ(MACD/price) over lookback_bars vs 60-day history.
    """
    # Calculate MACD/price ratio
    macd_price_ratio = macd / close
    
    # Delta over lookback_bars
    delta = macd_price_ratio.diff(lookback_bars)
    
    # History window for z-score calculation
    history_bars = history_days * 24 * 4
    
    z_scores = []
    for i in range(len(delta)):
        if i < history_bars:
            z_scores.append(np.nan)
        else:
            # Get historical window
            hist_window = delta.iloc[i - history_bars:i]
            current_val = delta.iloc[i]
            
            if pd.isna(current_val) or len(hist_window.dropna()) < 10:
                z_scores.append(np.nan)
            else:
                # Calculate z-score
                mean_hist = hist_window.dropna().mean()
                std_hist = hist_window.dropna().std()
                
                if std_hist == 0:
                    z_scores.append(0.0)
                else:
                    z = (current_val - mean_hist) / std_hist
                    z_scores.append(z)
    
    return pd.Series(z_scores, index=close.index)


def vol_forecast(close: pd.Series, mad_days: int = 30, ewma_lambda: float = 0.995) -> pd.Series:
    """
    Volatility forecast: 15m log returns → 30-day rolling MAD → σ̂1d = MAD × 1.4826 × √96 → EWMA(λ=0.995)
    Returns daily volatility forecast.
    """
    # 15m log returns
    returns = np.log(close / close.shift(1))
    
    # Rolling MAD over mad_days
    mad_window_bars = mad_days * 24 * 4
    rolling_mad = returns.rolling(window=mad_window_bars).apply(
        lambda x: np.median(np.abs(x - np.median(x))), raw=True
    )
    
    # Convert to daily volatility: MAD × 1.4826 × √96
    # 1.4826 is the scaling factor for MAD to approximate std for normal distribution
    # √96 converts 15m to daily (96 = 24*4 bars per day)
    sigma_1d = rolling_mad * 1.4826 * np.sqrt(96)
    
    # EWMA smoothing
    sigma_1d_ewma = sigma_1d.ewm(alpha=1 - ewma_lambda, adjust=False).mean()
    
    return sigma_1d_ewma


def vol_fast_median(vol_forecast: pd.Series, median_days: int = 30) -> pd.Series:
    """
    Median of vol_forecast over trailing median_days.
    """
    median_window_bars = median_days * 24 * 4
    return vol_forecast.rolling(window=median_window_bars).median()


def compute_helper_indicators(df: pd.DataFrame, vol_forecast_series: pd.Series = None) -> pd.DataFrame:
    """Compute helper indicators and add to dataframe"""
    result = df.copy()
    
    # rv_pct (60-day percentile)
    result['rv_pct'] = rv_pct(result['close'], window_days=60)
    
    # slope_z (20-bar z-score vs 60-day history)
    if 'macd' in result.columns:
        result['slope_z'] = slope_z(result['close'], result['macd'], lookback_bars=20, history_days=60)
    else:
        result['slope_z'] = np.nan
    
    # vol_forecast (if not provided, compute it)
    if vol_forecast_series is None:
        result['vol_forecast'] = vol_forecast(result['close'], mad_days=30, ewma_lambda=0.995)
    else:
        result['vol_forecast'] = vol_forecast_series
    
    # vol_fast_median
    result['vol_fast_median'] = vol_fast_median(result['vol_forecast'], median_days=30)
    
    return result

