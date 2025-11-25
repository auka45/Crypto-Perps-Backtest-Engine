"""
Metrics Sanity Test
Verifies cost toggles and metric calculations from a real experiment artifacts folder.
"""
import unittest
import json
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from engine_core.src.engine import BacktestEngine
from engine_core.config.params_loader import ParamsLoader
from engine_core.src.data.loader import DataLoader

class TestMetricsSanity(unittest.TestCase):
    def setUp(self):
        # Path to Exp 1 (Raw Edge, Costs OFF) artifacts
        # Adjust path relative to test file execution context
        self.project_root = Path(__file__).parent.parent
        self.artifacts_path = self.project_root / "experiments_1y_sanity" / "exp_1_Raw_Edge_Costs_OFF" / "artifacts"
        
        if not self.artifacts_path.exists():
            self.skipTest(f"Artifacts not found at {self.artifacts_path}. Run experiment 1 first.")
            
        # Load metrics
        with open(self.artifacts_path / "metrics.json", 'r') as f:
            self.metrics = json.load(f)
            
        # Load trades
        trades_file = self.artifacts_path / "trades.csv"
        if trades_file.exists():
            self.trades_df = pd.read_csv(trades_file)
        else:
            self.trades_df = None

    def test_costs_are_zero(self):
        """Verify that all cost metrics are exactly 0.0"""
        # Global metrics
        self.assertEqual(self.metrics.get('total_fees', 0.0), 0.0, "Total fees should be 0.0")
        self.assertEqual(self.metrics.get('total_slippage_cost', 0.0), 0.0, "Total slippage should be 0.0")
        self.assertEqual(self.metrics.get('funding_cost_total', 0.0), 0.0, "Funding cost should be 0.0")
        
        # Per-trade verification
        if self.trades_df is not None and not self.trades_df.empty:
            # Check if columns exist before accessing
            if 'entry_costs_usd' in self.trades_df.columns:
                self.assertTrue((self.trades_df['entry_costs_usd'] == 0.0).all(), "Entry costs must be 0 for all trades")
            if 'exit_costs_usd' in self.trades_df.columns:
                self.assertTrue((self.trades_df['exit_costs_usd'] == 0.0).all(), "Exit costs must be 0 for all trades")
            if 'fees' in self.trades_df.columns:
                self.assertTrue((self.trades_df['fees'] == 0.0).all(), "Fees column must be 0 for all trades")
            if 'fee_usd' in self.trades_df.columns:
                self.assertTrue((self.trades_df['fee_usd'] == 0.0).all(), "Fee USD column must be 0 for all trades")
            
            # If slip_bps is logged, realized cost should still be 0 if cost model disabled,
            # BUT engine might log 'slip_bps' as theoretical value even if cost is 0.
            # We check the dollar cost columns.
            # slippage_cost_usd usually isn't a direct column in trades.csv unless added recently,
            # but entry_costs_usd includes it.
            pass

    def test_pnl_gross_equals_net(self):
        """Verify Gross PnL == Net PnL when costs are off"""
        if self.trades_df is not None and not self.trades_df.empty:
            # Allow small float tolerance
            diff = abs(self.trades_df['pnl_gross_usd'] - self.trades_df['pnl_net_usd'])
            self.assertTrue((diff < 1e-9).all(), "Gross PnL must equal Net PnL when costs are OFF")

    def test_total_return_calculation(self):
        """Verify total_return metric against equity curve logic"""
        # Load equity curve if available
        equity_file = self.artifacts_path / "equity.csv"
        if not equity_file.exists():
            self.skipTest("equity.csv not found")
            
        equity_df = pd.read_csv(equity_file)
        initial_equity = equity_df.iloc[0]['equity']
        final_equity = equity_df.iloc[-1]['equity']
        
        expected_return = (final_equity / initial_equity) - 1.0
        metric_return = self.metrics.get('total_return', 0.0)
        
        # Check consistency with reasonable precision
        # Note: metrics.json might use 'total_return_pct' or 'total_return' (decimal).
        # Let's check both or whatever is present.
        
        # If 'total_return' in JSON is decimal
        self.assertAlmostEqual(metric_return, expected_return, places=4, 
                               msg=f"Metric return {metric_return} != Calculated {expected_return}")
        
        # Also check if summary csv has 0.0
        summary_path = self.project_root / "experiments_1y_sanity" / "artifacts" / "experiments_summary.csv"
        if summary_path.exists():
            summary_df = pd.read_csv(summary_path)
            # Filter for exp 1
            exp_row = summary_df[summary_df['exp_id'] == 1]
            if not exp_row.empty:
                summary_return = exp_row.iloc[0]['total_return']
                # This is the bug suspect: summary shows 0.0
                print(f"\nSummary CSV total_return: {summary_return}")
                print(f"Metric total_return: {metric_return}")
                
                # We expect this to FAIL if the bug exists
                self.assertNotEqual(summary_return, 0.0, "Summary CSV total_return should not be 0.0 if metric is non-zero")

    def test_cost_toggle_fix(self):
        """
        Verify cost toggle fix using a small synthetic run
        """
        # Setup synthetic data
        dates = pd.date_range(start='2024-01-01', periods=10, freq='15min', tz='UTC')
        df = pd.DataFrame({
            'ts': dates, 'open': 100.0, 'high': 105.0, 'low': 95.0, 'close': 100.0, 'volume': 1000.0,
            'quote_asset_volume': 100000.0, 'trades': 100, 'taker_buy_base_asset_volume': 500.0, 
            'taker_buy_quote_asset_volume': 50000.0
        })
        df['atr'] = 1.0
        df['notional'] = df['close'] * df['volume']
        
        # Use overrides to disable cost model
        # Note: taker_fee_bps override must match dict structure of base params
        overrides = {
            "cost_model": {"enabled": False},
            "general": {"taker_fee_bps": {"default": 10.0}} 
        }
        
        # We need to use a temporary directory for data
        import tempfile
        import shutil
        
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir)
            df.to_csv(data_path / "BTCUSDT_15m.csv", index=False)
            
            # Initialize Engine with overrides
            params = ParamsLoader(overrides=overrides)
            data_loader = DataLoader(str(data_path))
            data_loader.load_symbol('BTCUSDT', require_liquidity=False)
            
            engine = BacktestEngine(data_loader, params)
            
            # Mock a trade execution directly to test cost logic
            # We need to access private methods or mock events
            from src.execution.sequencing import OrderEvent
            from collections import namedtuple
            
            Signal = namedtuple('Signal', ['side', 'stop_price'])
            
            event = OrderEvent(
                event_type='TREND_ENTRY',
                symbol='BTCUSDT',
                module='TREND',
                priority=1,
                signal_ts=dates[5],
                order_id='test_fix_1'
            )
            
            # Setup pending signal and mock guards
            class MockSignal:
                def __init__(self):
                    self.side = 'LONG'
                    self.stop_price = 90.0
                    self.entry_price = 100.0
                    self.module = 'TREND'
                    self.signal_bar_idx = 5
            
            engine.symbol_pending_signals['BTCUSDT'] = [MockSignal()]
            engine._passes_es_guard = lambda *args: (True, 0, 0)
            engine._check_beta_caps_with_new_position = lambda *args: True
            engine.liquidity_detector.get_participation_cap = lambda *args: 1.0
            engine.liquidity_detector.get_slippage_adder = lambda *args: 0.0
            
            # Manually setup data structures needed by execute_entry
            engine.symbol_data = {'BTCUSDT': df}
            engine.symbol_ts_to_idx = {'BTCUSDT': {ts: i for i, ts in enumerate(dates)}}
            
            # Execute entry
            fill_bar = df.iloc[5]
            fill_ts = dates[5]
            engine.execute_entry(event, fill_bar, fill_ts)
            
            # Check fills
            self.assertEqual(len(engine.fills), 1)
            fill = engine.fills[0]
            
            # Assert costs are ZERO
            self.assertEqual(fill['fee_bps'], 0.0, "Fee BPS should be 0.0 when cost_model.enabled=False")
            self.assertEqual(fill['fee_usd'], 0.0, "Fee USD should be 0.0 when cost_model.enabled=False")
            self.assertEqual(fill['slippage_bps_applied'], 0.0, "Slippage BPS should be 0.0 when cost_model.enabled=False")
            self.assertEqual(fill['slippage_cost_usd'], 0.0, "Slippage USD should be 0.0 when cost_model.enabled=False")

if __name__ == '__main__':
    unittest.main()
