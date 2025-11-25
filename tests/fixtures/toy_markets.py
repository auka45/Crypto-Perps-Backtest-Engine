"""Generate synthetic OHLCV market data for testing"""
import pandas as pd
import numpy as np
from datetime import timedelta
from typing import Dict


def generate_toy_market(
    market_type: str,
    start_ts: pd.Timestamp,
    num_bars: int = 100,
    base_price: float = 50000.0,
    bar_interval_minutes: int = 15
) -> pd.DataFrame:
    """
    Generate synthetic OHLCV data for testing.
    
    Market types:
    - UP: Monotonic upward drift with noise
    - DOWN: Monotonic downward drift with noise
    - CHOP: Zero drift mean-reverting noise (choppy)
    - GAP_SHOCK: One large gap bar, then normal drift
    
    Args:
        market_type: Type of market ('UP', 'DOWN', 'CHOP', 'GAP_SHOCK')
        start_ts: Starting timestamp
        num_bars: Number of 15m bars to generate
        base_price: Starting price
        bar_interval_minutes: Minutes between bars (default 15)
    
    Returns:
        DataFrame with columns: ts, open, high, low, close, volume, notional
    """
    timestamps = [start_ts + timedelta(minutes=bar_interval_minutes*i) for i in range(num_bars)]
    
    if market_type == 'UP':
        # Upward drift: +0.1% per bar on average, with noise
        drift = 0.001  # 0.1% per bar
        noise_std = 0.002  # 0.2% noise
        returns = np.random.normal(drift, noise_std, num_bars)
        prices = [base_price]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
    
    elif market_type == 'DOWN':
        # Downward drift: -0.1% per bar on average, with noise
        drift = -0.001  # -0.1% per bar
        noise_std = 0.002  # 0.2% noise
        returns = np.random.normal(drift, noise_std, num_bars)
        prices = [base_price]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
    
    elif market_type == 'CHOP':
        # Zero drift mean-reverting noise
        noise_std = 0.003  # 0.3% noise
        returns = np.random.normal(0.0, noise_std, num_bars)
        # Add mean reversion: if price deviates too far, pull back
        prices = [base_price]
        for ret in returns[1:]:
            deviation = (prices[-1] - base_price) / base_price
            mean_reversion = -0.1 * deviation  # Pull back 10% of deviation
            prices.append(prices[-1] * (1 + ret + mean_reversion))
    
    elif market_type == 'GAP_SHOCK':
        # One large gap bar at bar 20, then normal drift
        drift = 0.0005  # 0.05% per bar
        noise_std = 0.002
        returns = np.random.normal(drift, noise_std, num_bars)
        
        # Add large gap at bar 20 (index 20)
        gap_size = 0.05  # 5% gap
        if num_bars > 20:
            returns[20] = gap_size  # Large positive return
        
        prices = [base_price]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
    
    else:
        raise ValueError(f"Unknown market_type: {market_type}")
    
    # Generate OHLC from prices
    df_data = []
    for i, (ts, price) in enumerate(zip(timestamps, prices)):
        # Add small intra-bar movement
        spread_pct = 0.0005  # 0.05% spread
        high = price * (1 + spread_pct / 2)
        low = price * (1 - spread_pct / 2)
        
        # Open is previous close (or base_price for first bar)
        if i == 0:
            open_price = price
        else:
            open_price = prices[i-1]
        
        # Close is current price
        close = price
        
        # Volume: constant for simplicity
        volume = 1000.0
        notional = price * volume
        
        df_data.append({
            'ts': ts,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume,
            'notional': notional
        })
    
    df = pd.DataFrame(df_data)
    return df


def create_toy_data_loader(markets: Dict[str, pd.DataFrame], tmp_path) -> 'DataLoader':
    """
    Create a DataLoader with toy market data.
    
    Args:
        markets: Dict mapping symbol names to DataFrames
        tmp_path: Temporary directory path
    
    Returns:
        DataLoader instance
    """
    from engine_core.src.data.loader import DataLoader
    
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Write each market to CSV - use the format DataLoader expects: {symbol}_15m.csv
    for symbol, df in markets.items():
        csv_path = data_dir / f"{symbol}_15m.csv"
        df.to_csv(csv_path, index=False)
    
    loader = DataLoader(str(data_dir))
    for symbol in markets.keys():
        loader.load_symbol(symbol, require_liquidity=False)
    return loader

