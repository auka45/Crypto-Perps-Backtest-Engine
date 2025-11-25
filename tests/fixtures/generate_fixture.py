"""Generate small deterministic test fixture dataset"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta, timezone


def generate_fixture(output_dir: str = "tests/fixtures/data"):
    """Generate 2-3 days of deterministic 15m bars for BTCUSDT and ETHUSDT"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Fixed seed for determinism
    np.random.seed(42)
    
    # Start time: 2024-01-01 00:00 UTC
    start_ts = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    n_bars = 3 * 24 * 4  # 3 days * 24 hours * 4 bars per hour
    
    symbols = ['BTCUSDT', 'ETHUSDT']
    
    for symbol in symbols:
        # Base price (deterministic)
        if symbol == 'BTCUSDT':
            base_price = 42000.0
        else:
            base_price = 2500.0
        
        # Generate deterministic price series
        prices = [base_price]
        for i in range(1, n_bars):
            # Small random walk with drift
            change = np.random.normal(0, 0.002)  # 0.2% volatility
            prices.append(prices[-1] * (1 + change))
        
        # Generate OHLC bars
        bars = []
        for i in range(n_bars):
            ts = start_ts + timedelta(minutes=15 * i)
            close = prices[i]
            
            # Generate OHLC with some intraday movement
            high_frac = np.random.uniform(0.001, 0.005)
            low_frac = np.random.uniform(0.001, 0.005)
            
            high = close * (1 + high_frac)
            low = close * (1 - low_frac)
            open_price = prices[i-1] if i > 0 else close
            
            # Volume (deterministic pattern)
            volume = np.random.uniform(0.5, 2.0) * (1 + 0.1 * np.sin(i / 10))
            notional = volume * close
            
            bars.append({
                'ts': ts,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume,
                'notional': notional
            })
        
        df = pd.DataFrame(bars)
        df.to_csv(output_path / f"{symbol}_15m.csv", index=False)
        
        # Generate contract metadata
        import json
        if symbol == 'BTCUSDT':
            metadata = {
                'tickSize': 0.01,
                'stepSize': 0.001,
                'minQty': 0.001,
                'minNotional': 5.0
            }
        else:
            metadata = {
                'tickSize': 0.01,
                'stepSize': 0.01,
                'minQty': 0.01,
                'minNotional': 5.0
            }
        
        with open(output_path / f"{symbol}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Generate simple funding data (8-hourly)
        funding_bars = []
        for i in range(0, n_bars, 32):  # Every 8 hours
            ts = start_ts + timedelta(minutes=15 * i)
            funding_bars.append({
                'funding_ts': ts,
                'funding_rate': np.random.uniform(-0.0001, 0.0001)
            })
        
        funding_df = pd.DataFrame(funding_bars)
        funding_df.to_csv(output_path / f"{symbol}_funding.csv", index=False)
        
        print(f"Generated fixture for {symbol}: {len(df)} bars")


if __name__ == '__main__':
    generate_fixture()

