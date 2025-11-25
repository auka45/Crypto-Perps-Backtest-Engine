"""Data schema validation for backtest inputs"""
from typing import Dict, List, Optional
import pandas as pd


class DataSchema:
    """Validates data schema per BACKTEST_SPEC.md §1"""
    
    # Required fields for 15-minute bars
    REQUIRED_15M_FIELDS = ['ts', 'open', 'high', 'low', 'close', 'volume', 'notional']
    
    # Optional but required for liquidity regimes
    LIQUIDITY_FIELDS = ['bid', 'ask', 'spread_bps', 'Depth5_bid_usd', 'Depth5_ask_usd']
    
    # Required for funding
    FUNDING_FIELDS = ['funding_ts', 'funding_rate']
    
    # Contract metadata (symbol-level)
    CONTRACT_FIELDS = ['tickSize', 'stepSize', 'minQty', 'minNotional']
    
    # Open interest for universe governance
    OI_FIELD = 'open_interest_usd'
    
    @staticmethod
    def validate_15m_bars(df: pd.DataFrame, symbol: str) -> List[str]:
        """Validate 15-minute bar data. Returns list of missing fields."""
        missing = []
        for field in DataSchema.REQUIRED_15M_FIELDS:
            if field not in df.columns:
                missing.append(f"{symbol}: Missing required field '{field}'")
        
        # Check data types
        if 'ts' in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df['ts']):
                missing.append(f"{symbol}: 'ts' must be datetime type")
        
        numeric_fields = ['open', 'high', 'low', 'close', 'volume', 'notional']
        for field in numeric_fields:
            if field in df.columns:
                if not pd.api.types.is_numeric_dtype(df[field]):
                    missing.append(f"{symbol}: '{field}' must be numeric")
        
        return missing
    
    @staticmethod
    def validate_liquidity_data(df: pd.DataFrame, symbol: str, require: bool = False) -> List[str]:
        """Validate liquidity/microstructure data. Returns list of missing fields."""
        missing = []
        
        # Check if we have bid/ask or can derive from mid+spread
        has_bid_ask = 'bid' in df.columns and 'ask' in df.columns
        has_mid_spread = 'spread_bps' in df.columns
        
        if require and not (has_bid_ask or has_mid_spread):
            missing.append(f"{symbol}: Missing liquidity data (need bid/ask or spread_bps)")
        
        if 'Depth5_bid_usd' in df.columns or 'Depth5_ask_usd' in df.columns:
            if 'Depth5_bid_usd' not in df.columns:
                missing.append(f"{symbol}: Missing 'Depth5_bid_usd'")
            if 'Depth5_ask_usd' not in df.columns:
                missing.append(f"{symbol}: Missing 'Depth5_ask_usd'")
        
        return missing
    
    @staticmethod
    def validate_funding_data(df: pd.DataFrame, symbol: str) -> List[str]:
        """Validate funding rate data. Returns list of missing fields."""
        missing = []
        for field in DataSchema.FUNDING_FIELDS:
            if field not in df.columns:
                missing.append(f"{symbol}: Missing funding field '{field}'")
        return missing
    
    @staticmethod
    def validate_contract_metadata(metadata: Dict[str, any], symbol: str) -> List[str]:
        """Validate contract metadata. Returns list of missing fields."""
        missing = []
        for field in DataSchema.CONTRACT_FIELDS:
            if field not in metadata:
                missing.append(f"{symbol}: Missing contract field '{field}'")
        return missing
    
    @staticmethod
    def validate_higher_tf(df: pd.DataFrame, tf: str, symbol: str) -> List[str]:
        """Validate higher timeframe data (1h, 4h, daily). Returns list of missing fields."""
        missing = []
        for field in ['ts', 'open', 'high', 'low', 'close']:
            if field not in df.columns:
                missing.append(f"{symbol} {tf}: Missing required field '{field}'")
        return missing

