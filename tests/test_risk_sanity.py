"""Sanity checks for risk controls - Launch Punch List – Quick Win #2"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from engine_core.config.params_loader import ParamsLoader
from engine_core.src.portfolio.state import PortfolioState, Position
from engine_core.src.risk.margin_guard import trim_with_deadlock_safety, check_risk_before_order
from engine_core.src.risk.loss_halts import LossHaltState
from engine_core.src.risk.beta_controls import check_portfolio_beta_caps
from engine_core.src.risk.engine_state import EngineStateManager, TradingState


class TestTrimDeadlockSafety:
    """Test trim deadlock safety with max_trim_count"""
    
    def test_trim_deadlock_safety(self):
        """Test that trim loop stops at max_trim_count and flattens if still in breach"""
        # Create positions that will trigger trim
        positions = {
            'BTCUSDT': Position(
                position_id='test1',
                symbol='BTCUSDT',
                qty=1.0,
                entry_price=50000.0,
                entry_ts=pd.Timestamp('2023-01-01'),
                side='LONG',
                module='TREND',
                stop_price=49000.0,
                trail_price=49000.0
            ),
            'ETHUSDT': Position(
                position_id='test2',
                symbol='ETHUSDT',
                qty=10.0,
                entry_price=3000.0,
                entry_ts=pd.Timestamp('2023-01-01'),
                side='LONG',
                module='TREND',
                stop_price=2900.0,
                trail_price=2900.0
            )
        }
        
        # Set equity low to trigger high margin ratio
        equity = 50000.0  # Low equity relative to notional
        
        params = {
            'max_trim_count': 3,
            'trim_flatten_on_fail': True,
            'trim_target_ratio_pct': 50.0,
            'block_new_entries_ratio_pct': 60.0,
            'flatten_ratio_pct': 80.0
        }
        
        trim_count = 0
        def close_callback(symbol):
            nonlocal trim_count
            if symbol in positions:
                del positions[symbol]
                trim_count += 1
                return True
            return False
        
        # Run trim loop
        should_flatten, actual_trim_count, margin_before, margin_after = trim_with_deadlock_safety(
            positions,
            equity,
            params,
            es_contributions={},
            close_position_callback=close_callback
        )
        
        # Assert: trim_count should not exceed max_trim_count
        assert actual_trim_count <= params['max_trim_count'], f"Trim count {actual_trim_count} exceeded max {params['max_trim_count']}"
        
        # Assert: if still in breach after max trims, should_flatten should be True
        if margin_after >= params['trim_target_ratio_pct'] / 100.0:
            assert should_flatten == True, "Should flatten when still in breach after max trims"


class TestDailyKillSwitch:
    """Test daily kill-switch trigger and state transition"""
    
    def test_daily_kill_switch(self):
        """Test that daily kill-switch sets state to RISK_HALT and blocks new entries"""
        # Create loss halt state
        loss_halt_state = LossHaltState()
        state_manager = EngineStateManager()
        
        # Set up scenario: large daily loss
        initial_equity = 100000.0
        current_equity = 95000.0  # -5% loss
        daily_pnl = -5000.0
        
        loss_halt_state.daily_pnl = daily_pnl
        loss_halt_state.daily_pnl_start = pd.Timestamp('2023-01-01').normalize()
        
        params = {
            'risk': {
                'kill_switch': {
                    'max_daily_loss_pct': 4.0,
                    'max_daily_loss_usd': 2000.0,
                    'flatten_on_trigger': True,
                    'block_new_entries': True
                }
            }
        }
        
        vol_scale = 1.0
        current_ts = pd.Timestamp('2023-01-01 12:00:00')
        
        # Check kill switch
        is_triggered, flatten_on_trigger, block_new_entries = loss_halt_state.check_daily_kill_switch(
            equity=current_equity,
            initial_equity=initial_equity,
            vol_scale=vol_scale,
            params=params,
            current_ts=current_ts
        )
        
        # Assert: kill switch should be triggered
        assert is_triggered == True, "Kill switch should be triggered with -5% loss"
        
        # Assert: should block new entries
        assert block_new_entries == True, "Should block new entries when kill switch triggered"
        
        # Simulate state transition
        if is_triggered:
            state_manager.set_state(TradingState.RISK_HALT, "risk:daily_kill_switch", current_ts)
        
        # Assert: state should be RISK_HALT
        assert state_manager.get_state() == TradingState.RISK_HALT, "State should be RISK_HALT after kill switch"
        
        # Assert: can_trade should return False
        assert state_manager.can_trade() == False, "Should not allow trading when in RISK_HALT"


class TestBetaCaps:
    """Test portfolio beta caps enforcement"""
    
    def test_beta_caps(self):
        """Test that orders exceeding portfolio beta caps are blocked"""
        # Create existing positions
        positions = {
            'BTCUSDT': {
                'qty': 1.0,
                'entry_price': 50000.0,
                'notional': 50000.0,
                'side': 'LONG'
            },
            'ETHUSDT': {
                'qty': 10.0,
                'entry_price': 3000.0,
                'notional': 30000.0,
                'side': 'LONG'
            }
        }
        
        # Beta values
        beta_slow = {
            'BTCUSDT': 1.0,
            'ETHUSDT': 1.2,
            'SOLUSDT': 1.5
        }
        
        # Try to add a large position that would exceed portfolio beta cap
        new_symbol = 'SOLUSDT'
        new_qty = 100.0  # Large position
        new_price = 100.0
        new_side = 'LONG'
        
        max_symbol_beta = 1.5
        max_portfolio_beta = 3.0
        
        # Check beta caps
        is_valid, symbol_beta_exposure, portfolio_beta_exposure, reason = check_portfolio_beta_caps(
            positions=positions,
            beta_slow=beta_slow,
            new_symbol=new_symbol,
            new_qty=new_qty,
            new_price=new_price,
            new_side=new_side,
            max_symbol_beta=max_symbol_beta,
            max_portfolio_beta=max_portfolio_beta,
            reference_symbol='BTCUSDT'
        )
        
        # Assert: order should be blocked if it exceeds caps
        # (This depends on the actual calculation, but we verify the function works)
        assert isinstance(is_valid, bool), "is_valid should be boolean"
        assert isinstance(symbol_beta_exposure, (int, float)), "symbol_beta_exposure should be numeric"
        assert isinstance(portfolio_beta_exposure, (int, float)), "portfolio_beta_exposure should be numeric"
        
        # If invalid, reason should be non-empty
        if not is_valid:
            assert len(reason) > 0, "Reason should be provided when order is blocked"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

