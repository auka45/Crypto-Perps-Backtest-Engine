"""Fixture dataset loader for deterministic backtesting"""
from pathlib import Path
from typing import Dict, Optional, List
import pandas as pd
from .schema import DataSchema


class DataLoader:
    """Loads and validates backtest data from CSV/Parquet files"""

    def __init__(self, data_path: str, start_ts: Optional[pd.Timestamp] = None, end_ts: Optional[pd.Timestamp] = None):
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data path does not exist: {data_path}")

        # Store date filter for lazy loading
        self._date_filter = (start_ts, end_ts)

        self._15m_bars: Dict[str, pd.DataFrame] = {}
        self._higher_tf: Dict[str, Dict[str, pd.DataFrame]] = {}  # {symbol: {tf: df}}
        self._liquidity: Dict[str, pd.DataFrame] = {}
        self._funding: Dict[str, pd.DataFrame] = {}
        self._contract_metadata: Dict[str, Dict] = {}
        self._oi: Dict[str, pd.DataFrame] = {}
    
    def load_symbol(self, symbol: str, require_liquidity: bool = False) -> List[str]:
        """
        Load all data for a symbol.
        Returns list of validation errors (empty if valid).
        """
        errors = []
        
        # Load 15m bars
        bar_path = self.data_path / f"{symbol}_15m.csv"
        if not bar_path.exists():
            bar_path = self.data_path / f"{symbol}_15m.parquet"
        
        if bar_path.exists():
            if bar_path.suffix == '.csv':
                # OPTIMIZATION: Use chunked reading with date filtering for large CSV files
                if self._date_filter[0] is not None and self._date_filter[1] is not None:
                    # Read in chunks and filter to avoid loading entire file
                    chunks = []
                    for chunk in pd.read_csv(bar_path, chunksize=50000, parse_dates=['ts']):
                        # Filter chunk to date range
                        mask = (chunk['ts'] >= self._date_filter[0]) & (chunk['ts'] <= self._date_filter[1])
                        if mask.any():
                            chunks.append(chunk[mask])
                        # Early exit if we've passed the end date
                        if chunk['ts'].max() > self._date_filter[1]:
                            break
                    df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
                else:
                    df = pd.read_csv(bar_path, parse_dates=['ts'])
            else:
                # For parquet, we can filter more efficiently
                df = pd.read_parquet(bar_path)
                if self._date_filter[0] is not None and self._date_filter[1] is not None:
                    if 'ts' in df.columns:
                        mask = (df['ts'] >= self._date_filter[0]) & (df['ts'] <= self._date_filter[1])
                        df = df[mask].copy()

            # Convert ts to datetime if string
            if 'ts' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['ts']):
                df['ts'] = pd.to_datetime(df['ts'], utc=True)

            # Ensure we have data and ts column
            if len(df) == 0:
                errors.append(f"{symbol}: No data found in date range")
                return errors

            if 'ts' not in df.columns:
                errors.append(f"{symbol}: Missing 'ts' column in data")
                return errors

            df = df.sort_values('ts').reset_index(drop=True)
            errors.extend(DataSchema.validate_15m_bars(df, symbol))
            self._15m_bars[symbol] = df
        else:
            errors.append(f"{symbol}: 15m bar file not found")
        
        # Load liquidity data (optional unless require_liquidity=True)
        liquidity_path = self.data_path / f"{symbol}_liquidity.csv"
        if not liquidity_path.exists():
            liquidity_path = self.data_path / f"{symbol}_liquidity.parquet"
        
        if liquidity_path.exists():
            if liquidity_path.suffix == '.csv':
                df = pd.read_csv(liquidity_path)
            else:
                df = pd.read_parquet(liquidity_path)
            
            if 'ts' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['ts']):
                df['ts'] = pd.to_datetime(df['ts'], utc=True)
            
            df = df.sort_values('ts').reset_index(drop=True)
            errors.extend(DataSchema.validate_liquidity_data(df, symbol, require_liquidity))
            self._liquidity[symbol] = df
        elif require_liquidity:
            errors.append(f"{symbol}: Liquidity data required but not found")
        
        # Load funding data
        funding_path = self.data_path / f"{symbol}_funding.csv"
        if not funding_path.exists():
            funding_path = self.data_path / f"{symbol}_funding.parquet"
        
        if funding_path.exists():
            if funding_path.suffix == '.csv':
                df = pd.read_csv(funding_path)
            else:
                df = pd.read_parquet(funding_path)
            
            if 'funding_ts' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['funding_ts']):
                df['funding_ts'] = pd.to_datetime(df['funding_ts'], utc=True, format='ISO8601')
            
            df = df.sort_values('funding_ts').reset_index(drop=True)
            errors.extend(DataSchema.validate_funding_data(df, symbol))
            self._funding[symbol] = df
        else:
            # Create empty funding dataframe if not provided
            self._funding[symbol] = pd.DataFrame(columns=['funding_ts', 'funding_rate'])
        
        # Load higher timeframes
        for tf in ['1h', '4h', 'daily']:
            tf_path = self.data_path / f"{symbol}_{tf}.csv"
            if not tf_path.exists():
                tf_path = self.data_path / f"{symbol}_{tf}.parquet"
            
            if tf_path.exists():
                if tf_path.suffix == '.csv':
                    df = pd.read_csv(tf_path)
                else:
                    df = pd.read_parquet(tf_path)
                
                if 'ts' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['ts']):
                    df['ts'] = pd.to_datetime(df['ts'], utc=True)
                
                df = df.sort_values('ts').reset_index(drop=True)
                errors.extend(DataSchema.validate_higher_tf(df, tf, symbol))
                
                if symbol not in self._higher_tf:
                    self._higher_tf[symbol] = {}
                self._higher_tf[symbol][tf] = df
        
        # Load contract metadata
        metadata_path = self.data_path / f"{symbol}_metadata.json"
        if metadata_path.exists():
            import json
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            errors.extend(DataSchema.validate_contract_metadata(metadata, symbol))
            self._contract_metadata[symbol] = metadata
        else:
            # Default metadata (should be provided, but allow defaults for testing)
            self._contract_metadata[symbol] = {
                'tickSize': 0.01,
                'stepSize': 0.001,
                'minQty': 0.001,
                'minNotional': 5.0
            }
        
        # Load open interest
        oi_path = self.data_path / f"{symbol}_oi.csv"
        if not oi_path.exists():
            oi_path = self.data_path / f"{symbol}_oi.parquet"
        
        if oi_path.exists():
            if oi_path.suffix == '.csv':
                df = pd.read_csv(oi_path)
            else:
                df = pd.read_parquet(oi_path)
            
            if 'ts' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['ts']):
                df['ts'] = pd.to_datetime(df['ts'], utc=True)
            
            df = df.sort_values('ts').reset_index(drop=True)
            self._oi[symbol] = df
        
        return errors
    
    def get_15m_bars(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get 15-minute bars for symbol"""
        return self._15m_bars.get(symbol)
    
    def get_higher_tf(self, symbol: str, tf: str) -> Optional[pd.DataFrame]:
        """Get higher timeframe data (1h, 4h, daily)"""
        return self._higher_tf.get(symbol, {}).get(tf)
    
    def get_liquidity(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get liquidity/microstructure data"""
        return self._liquidity.get(symbol)
    
    def get_funding(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get funding rate data"""
        return self._funding.get(symbol)
    
    def get_contract_metadata(self, symbol: str) -> Dict:
        """Get contract metadata (tickSize, stepSize, etc.)"""
        return self._contract_metadata.get(symbol, {})
    
    def get_oi(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get open interest data"""
        return self._oi.get(symbol)
    
    def get_symbols(self) -> List[str]:
        """Get list of loaded symbols"""
        return list(self._15m_bars.keys())
    
    def get_time_range(self) -> tuple:
        """Get (start_ts, end_ts) across all symbols"""
        if not self._15m_bars:
            return None, None
        
        all_starts = []
        all_ends = []
        for df in self._15m_bars.values():
            if len(df) > 0:
                all_starts.append(df['ts'].min())
                all_ends.append(df['ts'].max())
        
        if not all_starts:
            return None, None
        
        return min(all_starts), max(all_ends)

