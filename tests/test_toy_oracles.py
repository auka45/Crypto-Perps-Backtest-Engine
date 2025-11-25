"""Test oracle strategies on toy markets using ORACLE module"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta, timezone

from engine_core.config.params_loader import ParamsLoader
from engine_core.src.data.loader import DataLoader
from engine_core.src.engine import BacktestEngine
from engine_core.tests.fixtures.toy_markets import generate_toy_market, create_toy_data_loader


class TestToyOracles:
    """Test oracle strategies on synthetic markets using ORACLE module"""
    
    @pytest.fixture
    def params_costs_off(self):
        """Params with costs disabled and oracle mode"""
        # Base params - oracle_mode will be set per test
        return {
            'cost_model': {'enabled': False},
            'general': {'oracle_mode': None},  # Will be set per test
            'es_guardrails': {'es_cap_of_equity': 1.0},
            'risk': {'max_positions': {'default': 10}}
        }
    
    @pytest.fixture
    def params_costs_on(self):
        """Params with costs enabled and oracle mode"""
        # Base params - oracle_mode will be set per test
        return {
            'cost_model': {'enabled': True},
            'general': {'oracle_mode': None},  # Will be set per test
            'es_guardrails': {'es_cap_of_equity': 1.0},
            'risk': {'max_positions': {'default': 10}}
        }
    
    def test_always_long_on_up_market(self, tmp_path, params_costs_off):
        """Test Always-Long on UP market: assert pnl > 0 (with costs OFF)"""
        # Generate UP market
        start_ts = pd.Timestamp('2021-01-01 00:00:00', tz='UTC')
        up_market = generate_toy_market('UP', start_ts, num_bars=200, base_price=50000.0)
        
        # Create data loader
        markets = {'BTCUSDT': up_market}
        data_loader = create_toy_data_loader(markets, tmp_path)
        
        # Set oracle mode to always_long with debug enabled
        overrides = params_costs_off.copy()
        overrides['general']['oracle_mode'] = 'always_long'
        overrides['general']['debug_oracle_flow'] = True
        params = ParamsLoader(overrides=overrides, strict=False)
        
        # Run backtest
        engine = BacktestEngine(data_loader, params, require_liquidity_data=False)
        print(f"[TEST BEFORE RUN] Loaded symbols: {data_loader.get_symbols()}")
        end_ts = start_ts + timedelta(minutes=15*200)
        engine.run(start_ts=start_ts, end_ts=end_ts, output_dir=str(tmp_path / "reports"))
        
        # Check PnL is positive (long on up market)
        pnl = engine.portfolio.total_pnl
        print(f"\\n[TEST] Total PnL: {pnl}, Trades count: {len(engine.trades)}")
        print(f"[TEST] Portfolio positions: {list(engine.portfolio.positions.keys())}")
        assert pnl > 0, f"Always-Long on UP market should have positive PnL, got {pnl}"
        
        # Verify we have at least one trade
        assert len(engine.trades) > 0, "Should have at least one trade"
    
    def test_always_short_on_down_market(self, tmp_path, params_costs_off):
        """Test Always-Short on DOWN market: assert pnl > 0 (with costs OFF)"""
        # Generate DOWN market
        start_ts = pd.Timestamp('2021-01-01 00:00:00', tz='UTC')
        down_market = generate_toy_market('DOWN', start_ts, num_bars=200, base_price=50000.0)
        
        # Create data loader
        markets = {'BTCUSDT': down_market}
        data_loader = create_toy_data_loader(markets, tmp_path)
        
        # Set oracle mode to always_short
        overrides = params_costs_off.copy()
        overrides['general']['oracle_mode'] = 'always_short'
        params = ParamsLoader(overrides=overrides, strict=False)
        
        # Run backtest
        engine = BacktestEngine(data_loader, params, require_liquidity_data=False)
        end_ts = start_ts + timedelta(minutes=15*200)
        engine.run(start_ts=start_ts, end_ts=end_ts, output_dir=str(tmp_path / "reports"))
        
        # Check PnL is positive (short on down market)
        pnl = engine.portfolio.total_pnl
        assert pnl > 0, f"Always-Short on DOWN market should have positive PnL, got {pnl}"
        
        # Verify we have at least one trade
        assert len(engine.trades) > 0, "Should have at least one trade"
    
    def test_always_long_on_down_market(self, tmp_path, params_costs_off):
        """Test Always-Long on DOWN market: assert pnl < 0 (sanity check)"""
        # Generate DOWN market
        start_ts = pd.Timestamp('2021-01-01 00:00:00', tz='UTC')
        down_market = generate_toy_market('DOWN', start_ts, num_bars=200, base_price=50000.0)
        
        # Create data loader
        markets = {'BTCUSDT': down_market}
        data_loader = create_toy_data_loader(markets, tmp_path)
        
        # Set oracle mode to always_long
        overrides = params_costs_off.copy()
        overrides['general']['oracle_mode'] = 'always_long'
        params = ParamsLoader(overrides=overrides, strict=False)
        
        # Run backtest
        engine = BacktestEngine(data_loader, params, require_liquidity_data=False)
        end_ts = start_ts + timedelta(minutes=15*200)
        engine.run(start_ts=start_ts, end_ts=end_ts, output_dir=str(tmp_path / "reports"))
        
        # Check PnL is negative (long on down market)
        pnl = engine.portfolio.total_pnl
        assert pnl < 0, f"Always-Long on DOWN market should have negative PnL, got {pnl}"
    
    def test_always_short_on_up_market(self, tmp_path, params_costs_off):
        """Test Always-Short on UP market: assert pnl < 0 (sanity check)"""
        # Generate UP market
        start_ts = pd.Timestamp('2021-01-01 00:00:00', tz='UTC')
        up_market = generate_toy_market('UP', start_ts, num_bars=200, base_price=50000.0)
        
        # Create data loader
        markets = {'BTCUSDT': up_market}
        data_loader = create_toy_data_loader(markets, tmp_path)
        
        # Set oracle mode to always_short
        overrides = params_costs_off.copy()
        overrides['general']['oracle_mode'] = 'always_short'
        params = ParamsLoader(overrides=overrides, strict=False)
        
        # Run backtest
        engine = BacktestEngine(data_loader, params, require_liquidity_data=False)
        end_ts = start_ts + timedelta(minutes=15*200)
        engine.run(start_ts=start_ts, end_ts=end_ts, output_dir=str(tmp_path / "reports"))
        
        # Check PnL is negative (short on up market)
        pnl = engine.portfolio.total_pnl
        assert pnl < 0, f"Always-Short on UP market should have negative PnL, got {pnl}"
    
    def test_flat_oracle(self, tmp_path, params_costs_off):
        """Test Flat oracle: assert 0 trades and pnl == 0"""
        # Generate any market (doesn't matter for flat)
        start_ts = pd.Timestamp('2021-01-01 00:00:00', tz='UTC')
        up_market = generate_toy_market('UP', start_ts, num_bars=200, base_price=50000.0)
        
        # Create data loader
        markets = {'BTCUSDT': up_market}
        data_loader = create_toy_data_loader(markets, tmp_path)
        
        # Set oracle mode to flat
        overrides = params_costs_off.copy()
        overrides['general']['oracle_mode'] = 'flat'
        params = ParamsLoader(overrides=overrides, strict=False)
        
        # Run backtest
        engine = BacktestEngine(data_loader, params, require_liquidity_data=False)
        end_ts = start_ts + timedelta(minutes=15*200)
        engine.run(start_ts=start_ts, end_ts=end_ts, output_dir=str(tmp_path / "reports"))
        
        # Check 0 trades and pnl == 0
        assert len(engine.trades) == 0, f"Flat oracle should have 0 trades, got {len(engine.trades)}"
        pnl = engine.portfolio.total_pnl
        assert abs(pnl) < 0.01, f"Flat oracle should have pnl == 0, got {pnl}"
    
    def test_chop_with_costs_off(self, tmp_path, params_costs_off):
        """Test CHOP market with costs OFF: assert |pnl| < 1% of initial capital"""
        # Generate CHOP market
        start_ts = pd.Timestamp('2021-01-01 00:00:00', tz='UTC')
        chop_market = generate_toy_market('CHOP', start_ts, num_bars=200, base_price=50000.0)
        
        # Create data loader
        markets = {'BTCUSDT': chop_market}
        data_loader = create_toy_data_loader(markets, tmp_path)
        
        # Set oracle mode to always_long (to generate a trade)
        overrides = params_costs_off.copy()
        overrides['general']['oracle_mode'] = 'always_long'
        params = ParamsLoader(overrides=overrides, strict=False)
        
        # Run backtest
        engine = BacktestEngine(data_loader, params, require_liquidity_data=False)
        end_ts = start_ts + timedelta(minutes=15*200)
        engine.run(start_ts=start_ts, end_ts=end_ts, output_dir=str(tmp_path / "reports"))
        
        # CHOP market should have small PnL (within 1% of initial capital)
        initial_capital = engine.portfolio.initial_capital
        pnl = engine.portfolio.total_pnl
        pnl_pct = abs(pnl) / initial_capital
        
        assert pnl_pct < 0.01, f"CHOP market with costs OFF should have |pnl| < 1% of initial capital, got {pnl_pct:.2%}"
    
    def test_costs_reduce_pnl(self, tmp_path):
        """Test Costs ON vs OFF: assert costs reduce pnl magnitude (monotonic)"""
        # Generate UP market
        start_ts = pd.Timestamp('2021-01-01 00:00:00', tz='UTC')
        up_market = generate_toy_market('UP', start_ts, num_bars=200, base_price=50000.0)
        
        markets = {'BTCUSDT': up_market}
        
        # Run with costs OFF
        params_costs_off = ParamsLoader(overrides={
            'cost_model': {'enabled': False},
            'general': {'oracle_mode': 'always_long'},
            'es_guardrails': {'es_cap_of_equity': 1.0},
            'risk': {'max_positions': {'default': 10}}
        }, strict=False)
        
        data_loader_off = create_toy_data_loader(markets, tmp_path / "data_off")
        engine_off = BacktestEngine(data_loader_off, params_costs_off, require_liquidity_data=False)
        end_ts = start_ts + timedelta(minutes=15*200)
        engine_off.run(start_ts=start_ts, end_ts=end_ts, output_dir=str(tmp_path / "reports_off"))
        
        # Run with costs ON
        params_costs_on = ParamsLoader(overrides={
            'cost_model': {'enabled': True},
            'general': {'oracle_mode': 'always_long'},
            'es_guardrails': {'es_cap_of_equity': 1.0},
            'risk': {'max_positions': {'default': 10}}
        }, strict=False)
        
        data_loader_on = create_toy_data_loader(markets, tmp_path / "data_on")
        engine_on = BacktestEngine(data_loader_on, params_costs_on, require_liquidity_data=False)
        engine_on.run(start_ts=start_ts, end_ts=end_ts, output_dir=str(tmp_path / "reports_on"))
        
        # Both should have trades
        assert len(engine_off.trades) > 0, "Costs OFF should have trades"
        assert len(engine_on.trades) > 0, "Costs ON should have trades"
        
        # Costs ON should have lower (or equal) PnL than costs OFF
        pnl_off = engine_off.portfolio.total_pnl
        pnl_on = engine_on.portfolio.total_pnl
        
        assert pnl_on <= pnl_off, f"Costs ON should reduce PnL vs costs OFF. OFF: {pnl_off}, ON: {pnl_on}"
        
        # Verify costs were actually applied
        total_costs_on = engine_on.portfolio.fees_paid + engine_on.portfolio.slippage_paid + engine_on.portfolio.funding_paid
        assert total_costs_on > 0, "Costs ON should have non-zero costs"
        
        total_costs_off = engine_off.portfolio.fees_paid + engine_off.portfolio.slippage_paid + engine_off.portfolio.funding_paid
        assert abs(total_costs_off) < 0.01, "Costs OFF should have zero costs"
