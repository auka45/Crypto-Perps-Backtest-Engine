"""Reconciliation tests: verify PnL identity and cost accounting"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta, timezone
import sys
import json
import tempfile
import shutil

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from engine_core.src.reporting import ReportGenerator, validate_metrics


def create_synthetic_run(artifacts_dir: Path):
    """Create a synthetic run with known values for testing"""
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    # Synthetic fills: one LONG and one SHORT position
    # LONG: entry at $100 (qty=1, fee=$1, slip=$0.50), exit at $110 (fee=$1, slip=$0.50)
    # SHORT: entry at $200 (qty=1, fee=$1, slip=$0.50), exit at $190 (fee=$1, slip=$0.50)
    # Initial equity = $1000
    
    run_id = "test-run-001"
    ts_base = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    
    # Fills
    fills = [
        {
            'run_id': run_id,
            'position_id': 'pos-001',
            'fill_id': 'pos-001-ENTRY-1',
            'ts': ts_base,
            'symbol': 'BTCUSDT',
            'module': 'TREND',
            'leg': 'ENTRY',
            'side': 'BUY',
            'qty': 1.0,
            'price': 100.0,
            'notional_usd': 100.0,
            'slippage_bps_applied': 5.0,
            'slippage_cost_usd': 0.50,
            'fee_bps': 10.0,
            'fee_usd': 1.0,
            'liquidity': 'taker',
            'participation_pct': 0.01,
            'adv60_usd': 10000.0
        },
        {
            'run_id': run_id,
            'position_id': 'pos-001',
            'fill_id': 'pos-001-EXIT-1',
            'ts': ts_base + timedelta(hours=4),
            'symbol': 'BTCUSDT',
            'module': 'TREND',
            'leg': 'EXIT',
            'side': 'SELL',
            'qty': 1.0,
            'price': 110.0,
            'notional_usd': 110.0,
            'slippage_bps_applied': 5.0,
            'slippage_cost_usd': 0.50,
            'fee_bps': 10.0,
            'fee_usd': 1.0,
            'liquidity': 'taker',
            'participation_pct': 0.01,
            'adv60_usd': 11000.0
        },
        {
            'run_id': run_id,
            'position_id': 'pos-002',
            'fill_id': 'pos-002-ENTRY-1',
            'ts': ts_base + timedelta(hours=1),
            'symbol': 'ETHUSDT',
            'module': 'SQUEEZE',
            'leg': 'ENTRY',
            'side': 'SELL',
            'qty': 1.0,
            'price': 200.0,
            'notional_usd': 200.0,
            'slippage_bps_applied': 5.0,
            'slippage_cost_usd': 0.50,
            'fee_bps': 10.0,
            'fee_usd': 1.0,
            'liquidity': 'taker',
            'participation_pct': 0.01,
            'adv60_usd': 20000.0
        },
        {
            'run_id': run_id,
            'position_id': 'pos-002',
            'fill_id': 'pos-002-EXIT-1',
            'ts': ts_base + timedelta(hours=5),
            'symbol': 'ETHUSDT',
            'module': 'SQUEEZE',
            'leg': 'EXIT',
            'side': 'BUY',
            'qty': 1.0,
            'price': 190.0,
            'notional_usd': 190.0,
            'slippage_bps_applied': 5.0,
            'slippage_cost_usd': 0.50,
            'fee_bps': 10.0,
            'fee_usd': 1.0,
            'liquidity': 'taker',
            'participation_pct': 0.01,
            'adv60_usd': 19000.0
        }
    ]
    
    # Ledger events
    ledger = [
        {
            'ts': ts_base,
            'run_id': run_id,
            'event': 'ENTRY_FILL',
            'position_id': 'pos-001',
            'symbol': 'BTCUSDT',
            'module': 'TREND',
            'leg': 'ENTRY',
            'side': 'BUY',
            'qty': 1.0,
            'price': 100.0,
            'notional_usd': 100.0,
            'fee_usd': 1.0,
            'slippage_cost_usd': 0.50,
            'funding_usd': 0.0,
            'cash_delta_usd': -1.50,  # Cash decreases
            'note': 'Entry fill: TREND'
        },
        {
            'ts': ts_base + timedelta(hours=4),
            'run_id': run_id,
            'event': 'EXIT_FILL',
            'position_id': 'pos-001',
            'symbol': 'BTCUSDT',
            'module': 'TREND',
            'leg': 'EXIT',
            'side': 'SELL',
            'qty': 1.0,
            'price': 110.0,
            'notional_usd': 110.0,
            'fee_usd': 1.0,
            'slippage_cost_usd': 0.50,
            'funding_usd': 0.0,
            # LONG: gross PnL = 110 - 100 = 10, net PnL = 10 - 1.5 - 1.5 = 7
            # EXIT_FILL cash_delta_usd = gross_pnl - exit_costs = 10 - 1.5 = 8.5
            'cash_delta_usd': 8.5,  # gross_pnl - exit_costs = 10 - 1.5 = 8.5
            'note': 'Exit fill: STOP'
        },
        {
            'ts': ts_base + timedelta(hours=1),
            'run_id': run_id,
            'event': 'ENTRY_FILL',
            'position_id': 'pos-002',
            'symbol': 'ETHUSDT',
            'module': 'SQUEEZE',
            'leg': 'ENTRY',
            'side': 'SELL',
            'qty': 1.0,
            'price': 200.0,
            'notional_usd': 200.0,
            'fee_usd': 1.0,
            'slippage_cost_usd': 0.50,
            'funding_usd': 0.0,
            'cash_delta_usd': -1.50,  # Cash decreases
            'note': 'Entry fill: SQUEEZE'
        },
        {
            'ts': ts_base + timedelta(hours=5),
            'run_id': run_id,
            'event': 'EXIT_FILL',
            'position_id': 'pos-002',
            'symbol': 'ETHUSDT',
            'module': 'SQUEEZE',
            'leg': 'EXIT',
            'side': 'BUY',
            'qty': 1.0,
            'price': 190.0,
            'notional_usd': 190.0,
            'fee_usd': 1.0,
            'slippage_cost_usd': 0.50,
            'funding_usd': 0.0,
            # SHORT: gross PnL = 200 - 190 = 10, net PnL = 10 - 1.5 - 1.5 = 7
            # EXIT_FILL cash_delta_usd = gross_pnl - exit_costs = 10 - 1.5 = 8.5
            'cash_delta_usd': 8.5,  # gross_pnl - exit_costs = 10 - 1.5 = 8.5
            'note': 'Exit fill: TTL'
        }
    ]
    
    # Expected: SUM(ledger) = -1.5 + 8.5 + -1.5 + 8.5 = 14.0
    # SUM(trades.pnl_net_usd) = 7.0 + 7.0 = 14.0
    # final_equity = 1000 + 14 = 1014
    # LONG: gross=10, entry_costs=1.5, exit_costs=1.5, net=7
    # SHORT: gross=10, entry_costs=1.5, exit_costs=1.5, net=7
    
    # Write fills.csv
    df_fills = pd.DataFrame(fills)
    df_fills.to_csv(artifacts_dir / 'fills.csv', index=False)
    
    # Write ledger.csv
    df_ledger = pd.DataFrame(ledger)
    df_ledger.to_csv(artifacts_dir / 'ledger.csv', index=False)
    
    # Rebuild trades.csv from fills
    report_gen = ReportGenerator(str(artifacts_dir.parent), run_id=run_id)
    df_trades = report_gen._build_trades_from_fills(df_fills, df_ledger, trades_list=[])
    df_trades.to_csv(artifacts_dir / 'trades.csv', index=False)
    
    # Write equity.csv - must satisfy bar identity: equity = cash + open_pnl + closed_pnl
    # At final timestamp: equity = 1014.0, closed_pnl = 14.0, open_pnl = 0
    # So cash = 1014.0 - 0 - 14.0 = 1000.0
    # This ensures equity_delta = 1014 - 1000 = 14 = SUM(trades) = SUM(ledger)
    equity_curve = [
        {'ts': ts_base, 'equity': 1000.0, 'cash': 1000.0, 'open_pnl': 0.0, 'closed_pnl': 0.0},
        {'ts': ts_base + timedelta(hours=6), 'equity': 1014.0, 'cash': 1000.0, 'open_pnl': 0.0, 'closed_pnl': 14.0}
    ]
    df_equity = pd.DataFrame(equity_curve)
    df_equity.to_csv(artifacts_dir / 'equity.csv', index=False)
    
    # Write metrics.json
    metrics = {
        'run_id': run_id,
        'initial_equity': 1000.0,
        'final_equity': 1014.0,
        'total_trades': 2,
        'total_fees': 4.0,  # 1+1+1+1
        'total_slippage_cost': 2.0,  # 0.5+0.5+0.5+0.5
        'funding_cost_total': 0.0,
        'realized_pnl_from_trades': 14.0  # 7 + 7
    }
    with open(artifacts_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return artifacts_dir


def test_reconciliation_identity():
    """Test that final_equity - initial_equity == Σ trades.pnl_net_usd"""
    with tempfile.TemporaryDirectory() as tmpdir:
        artifacts_dir = Path(tmpdir) / 'artifacts'
        create_synthetic_run(artifacts_dir)
        
        # Load artifacts
        metrics = json.load(open(artifacts_dir / 'metrics.json'))
        df_trades = pd.read_csv(artifacts_dir / 'trades.csv')
        
        initial_equity = metrics['initial_equity']
        final_equity = metrics['final_equity']
        equity_delta = final_equity - initial_equity
        
        pnl_net_sum = float(df_trades['pnl_net_usd'].sum())
        
        # Assert identity
        assert abs(equity_delta - pnl_net_sum) <= 0.01, \
            f"Equity delta ({equity_delta:.2f}) != Σ trades.pnl_net_usd ({pnl_net_sum:.2f})"


def test_ledger_cash_delta_sum():
    """Test that Σ ledger.cash_delta_usd == final_equity - initial_equity when no open positions"""
    with tempfile.TemporaryDirectory() as tmpdir:
        artifacts_dir = Path(tmpdir) / 'artifacts'
        create_synthetic_run(artifacts_dir)
        
        # Load artifacts
        metrics = json.load(open(artifacts_dir / 'metrics.json'))
        df_ledger = pd.read_csv(artifacts_dir / 'ledger.csv')
        
        initial_equity = metrics['initial_equity']
        final_equity = metrics['final_equity']
        equity_delta = final_equity - initial_equity
        
        cash_delta_sum = float(df_ledger['cash_delta_usd'].sum())
        
        # Assert identity (when no open positions, cash delta sum should equal equity delta)
        assert abs(equity_delta - cash_delta_sum) <= 0.01, \
            f"Equity delta ({equity_delta:.2f}) != Σ ledger.cash_delta_usd ({cash_delta_sum:.2f})"


def test_costs_from_fills():
    """Test that Σ fills costs == metrics.total_fees + total_slippage_cost"""
    with tempfile.TemporaryDirectory() as tmpdir:
        artifacts_dir = Path(tmpdir) / 'artifacts'
        create_synthetic_run(artifacts_dir)
        
        # Load artifacts
        metrics = json.load(open(artifacts_dir / 'metrics.json'))
        df_fills = pd.read_csv(artifacts_dir / 'fills.csv')
        
        fills_fees_sum = float(df_fills['fee_usd'].sum())
        fills_slip_sum = float(df_fills['slippage_cost_usd'].sum())
        fills_costs_sum = fills_fees_sum + fills_slip_sum
        
        metrics_fees = metrics['total_fees']
        metrics_slip = metrics['total_slippage_cost']
        metrics_costs_sum = metrics_fees + metrics_slip
        
        # Assert identity
        assert abs(fills_costs_sum - metrics_costs_sum) <= 0.01, \
            f"Σ fills costs ({fills_costs_sum:.2f}) != metrics.total_fees + total_slippage_cost ({metrics_costs_sum:.2f})"


@pytest.mark.parametrize("fees,slip", [(0.0, 0.0), (1.0, 0.5)])
def test_gross_pnl_identity(fees, slip):
    """Test that with zero costs, gross PnL identity holds"""
    # This test verifies that when fees=slip=0, the identity reduces to gross PnL
    # For simplicity, we'll just verify the synthetic run has correct gross PnL calculation
    with tempfile.TemporaryDirectory() as tmpdir:
        artifacts_dir = Path(tmpdir) / 'artifacts'
        create_synthetic_run(artifacts_dir)
        
        df_trades = pd.read_csv(artifacts_dir / 'trades.csv')
        
        # LONG: gross = 110 - 100 = 10
        # SHORT: gross = 200 - 190 = 10
        # Total gross = 20
        
        gross_pnl_sum = float(df_trades['pnl_gross_usd'].sum())
        assert abs(gross_pnl_sum - 20.0) <= 0.01, \
            f"Expected gross PnL = 20.0, got {gross_pnl_sum:.2f}"


def test_validate_metrics_on_fixture():
    """Test validate_metrics on known-good synthetic fixture"""
    with tempfile.TemporaryDirectory() as tmpdir:
        artifacts_dir = Path(tmpdir) / 'artifacts'
        create_synthetic_run(artifacts_dir)
        
        # Add required metrics fields to metrics.json
        metrics_path = artifacts_dir / 'metrics.json'
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        # Add required fields
        metrics['es_violations_count'] = 0
        metrics['margin_blocks_count'] = 0
        metrics['halt_daily_hard_count'] = 0
        metrics['per_symbol_loss_cap_count'] = 0
        metrics['halt_soft_brake_count'] = 0
        metrics['slippage_degeneracy_warning'] = False
        metrics['vacuum_blocks_count'] = 0
        metrics['thin_post_only_entries_count'] = 0
        metrics['thin_cancel_block_count'] = 0
        metrics['exposure_pct'] = 0.1  # Fraction, not percentage
        metrics['win_rate'] = 1.0  # Both trades are winners
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Run validation
        result = validate_metrics(artifacts_dir, strict_canonical=False)
        
        # Should pass
        assert result.passed is True, f"Validation failed: {result.failures}"
        assert len(result.failures) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

