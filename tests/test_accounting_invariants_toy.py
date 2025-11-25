"""Test accounting invariants on toy fills"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
from datetime import datetime, timedelta, timezone

from engine_core.config.params_loader import ParamsLoader
from engine_core.src.data.loader import DataLoader
from engine_core.src.engine import BacktestEngine


class TestAccountingInvariantsToy:
    """Test accounting invariants with minimal toy scenarios"""
    
    @pytest.fixture
    def tmp_data_dir(self, tmp_path):
        """Create temporary data directory with minimal test data"""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        
        # Create minimal OHLCV data for one symbol
        symbol = "BTCUSDT"
        symbol_dir = data_dir / symbol
        symbol_dir.mkdir()
        
        # Create 100 bars of synthetic data (simple upward trend)
        start_ts = pd.Timestamp('2021-01-01 00:00:00', tz='UTC')
        timestamps = [start_ts + timedelta(minutes=15*i) for i in range(100)]
        
        base_price = 50000.0
        prices = [base_price + i * 10.0 + np.random.normal(0, 50) for i in range(100)]
        
        df = pd.DataFrame({
            'ts': timestamps,
            'open': prices,
            'high': [p * 1.001 for p in prices],
            'low': [p * 0.999 for p in prices],
            'close': prices,
            'volume': [1000.0] * 100,
            'notional': [p * 1000.0 for p in prices]
        })
        
        # Save as CSV
        csv_path = symbol_dir / "15m.csv"
        df.to_csv(csv_path, index=False)
        
        return str(data_dir)
    
    @pytest.fixture
    def params_with_invariants(self):
        """Create params with debug_invariants enabled"""
        params = ParamsLoader(overrides={'general': {'debug_invariants': True}}, strict=False)
        return params
    
    def test_equity_identity_on_entry_exit(self, tmp_data_dir, params_with_invariants):
        """Test equity identity invariant through entry and exit"""
        data_loader = DataLoader(tmp_data_dir)
        engine = BacktestEngine(data_loader, params_with_invariants, require_liquidity_data=False)
        
        # Run a very short backtest (just a few bars)
        start_ts = pd.Timestamp('2021-01-01 00:00:00', tz='UTC')
        end_ts = pd.Timestamp('2021-01-01 01:00:00', tz='UTC')  # 4 bars
        
        # This should complete without invariant violations
        # If invariants fail, an AssertionError will be raised
        engine.run(start_ts=start_ts, end_ts=end_ts, output_dir=str(Path(tmp_data_dir).parent / "reports"))
        
        # Verify final equity equals cash (no positions should remain)
        assert len(engine.portfolio.positions) == 0 or engine.portfolio.equity == engine.portfolio.cash + sum(
            (engine.symbol_data[symbol].iloc[-1]['close'] - pos.entry_price) * pos.qty 
            if pos.side == 'LONG' else 
            (pos.entry_price - engine.symbol_data[symbol].iloc[-1]['close']) * pos.qty
            for symbol, pos in engine.portfolio.positions.items()
        )
    
    def test_cost_toggle_invariants(self, tmp_data_dir):
        """Test that cost toggle invariants are checked"""
        # Test with costs OFF
        params_costs_off = ParamsLoader(overrides={
            'cost_model': {'enabled': False},
            'general': {'debug_invariants': True}
        }, strict=False)
        
        data_loader = DataLoader(tmp_data_dir)
        engine = BacktestEngine(data_loader, params_costs_off, require_liquidity_data=False)
        
        start_ts = pd.Timestamp('2021-01-01 00:00:00', tz='UTC')
        end_ts = pd.Timestamp('2021-01-01 01:00:00', tz='UTC')
        
        # Should complete without cost-related invariant violations
        engine.run(start_ts=start_ts, end_ts=end_ts, output_dir=str(Path(tmp_data_dir).parent / "reports"))
        
        # Verify all costs are zero
        assert abs(engine.portfolio.fees_paid) < 0.01, f"Fees should be 0, got {engine.portfolio.fees_paid}"
        assert abs(engine.portfolio.slippage_paid) < 0.01, f"Slippage should be 0, got {engine.portfolio.slippage_paid}"
        assert abs(engine.portfolio.funding_paid) < 0.01, f"Funding should be 0, got {engine.portfolio.funding_paid}"
    
    def test_cost_signs_invariants(self, tmp_data_dir, params_with_invariants):
        """Test that cost signs are correct (fees, slippage <= 0)"""
        data_loader = DataLoader(tmp_data_dir)
        engine = BacktestEngine(data_loader, params_with_invariants, require_liquidity_data=False)
        
        start_ts = pd.Timestamp('2021-01-01 00:00:00', tz='UTC')
        end_ts = pd.Timestamp('2021-01-01 01:00:00', tz='UTC')
        
        # Should complete without cost sign violations
        engine.run(start_ts=start_ts, end_ts=end_ts, output_dir=str(Path(tmp_data_dir).parent / "reports"))
        
        # Verify all ledger entries have correct cost signs
        for entry in engine.ledger:
            assert entry.get('fee_usd', 0.0) <= 0.01, f"Fee should be <= 0, got {entry.get('fee_usd')}"
            assert entry.get('slippage_cost_usd', 0.0) <= 0.01, f"Slippage should be <= 0, got {entry.get('slippage_cost_usd')}"

