"""ORACLE module: test-only oracle strategies for validation"""
import pandas as pd
import numpy as np
from typing import Optional
from dataclasses import dataclass


@dataclass
class OracleSignal:
    """ORACLE module signal (for validation/testing only)"""
    symbol: str
    side: str  # 'LONG' or 'SHORT'
    entry_price: float
    stop_price: float
    signal_bar_idx: int
    signal_ts: pd.Timestamp
    module: str = 'ORACLE'
    exit_on_last_bar: bool = False  # For Buy & Hold: exit at last bar


class OracleModule:
    """ORACLE module for generating test signals (bypasses all filters)"""
    
    def __init__(self, params: dict):
        self.params = params
        self.oracle_params = params.get('general', {}).get('oracle_mode', None)
        self.random_seed = params.get('general', {}).get('oracle_random_seed', 42)
        if self.random_seed:
            np.random.seed(self.random_seed)
        # Track which symbols have already generated their first signal
        self._first_signal_generated: set = set()
    
    def __init__(self, params: dict):
        self.params = params
        self.oracle_params = params.get('general', {}).get('oracle_mode', None)
        self.random_seed = params.get('general', {}).get('oracle_random_seed', 42)
        if self.random_seed:
            np.random.seed(self.random_seed)
        # Track which symbols have already generated their first signal
        self._first_signal_generated: set = set()
    
    def generate_always_long(
        self,
        symbol: str,
        df: pd.DataFrame,
        idx: int,
        current_ts: pd.Timestamp
    ) -> Optional[OracleSignal]:
        """
        Generate always-long signal (for UP market test).
        
        Returns signal on first bar of filtered time range, exits on last bar if exit_on_last_bar=True.
        Uses a flag to ensure we only generate once per symbol per run.
        """
        # Generate signal only on first call for this symbol
        if symbol not in self._first_signal_generated:
            self._first_signal_generated.add(symbol)
            # Entry on first bar of filtered range
            entry_price = df.iloc[idx]['close']
            # Stop at 5% below entry
            stop_price = entry_price * 0.95
            return OracleSignal(
                symbol=symbol,
                side='LONG',
                entry_price=entry_price,
                stop_price=stop_price,
                signal_bar_idx=idx,
                signal_ts=current_ts,
                exit_on_last_bar=True
            )
        return None
    
    def generate_always_short(
        self,
        symbol: str,
        df: pd.DataFrame,
        idx: int,
        current_ts: pd.Timestamp
    ) -> Optional[OracleSignal]:
        """
        Generate always-short signal (for DOWN market test).
        
        Returns signal on first bar of filtered time range, exits on last bar if exit_on_last_bar=True.
        Uses a flag to ensure we only generate once per symbol per run.
        """
        # Generate signal only on first call for this symbol
        if symbol not in self._first_signal_generated:
            self._first_signal_generated.add(symbol)
            # Entry on first bar of filtered range
            entry_price = df.iloc[idx]['close']
            # Stop at 5% above entry
            stop_price = entry_price * 1.05
            return OracleSignal(
                symbol=symbol,
                side='SHORT',
                entry_price=entry_price,
                stop_price=stop_price,
                signal_bar_idx=idx,
                signal_ts=current_ts,
                exit_on_last_bar=True
            )
        return None
    
    def generate_flat(self) -> None:
        """Generate no signals (flat strategy)"""
        return None
    
    def generate_random(
        self,
        symbol: str,
        df: pd.DataFrame,
        idx: int,
        current_ts: pd.Timestamp
    ) -> Optional[OracleSignal]:
        """
        Generate random signals (for baseline testing).
        
        Randomly enters long/short with 10% probability per bar.
        Exits after random holding period (1-10 bars).
        """
        # 10% chance of entry per bar
        if np.random.random() < 0.1:
            side = 'LONG' if np.random.random() < 0.5 else 'SHORT'
            entry_price = df.iloc[idx]['close']
            if side == 'LONG':
                stop_price = entry_price * 0.95
            else:
                stop_price = entry_price * 1.05
            return OracleSignal(
                symbol=symbol,
                side=side,
                entry_price=entry_price,
                stop_price=stop_price,
                signal_bar_idx=idx,
                signal_ts=current_ts
            )
        return None

