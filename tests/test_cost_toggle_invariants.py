"""Test cost toggle invariants on real artifacts"""
import pytest
import json
from pathlib import Path


class TestCostToggleInvariants:
    """Test cost toggle invariants using real experiment artifacts"""
    
    def test_exp1_costs_are_zero(self):
        """Test that Exp 1 (Raw Edge, Costs OFF) has zero costs"""
        # Path to Exp 1 artifacts
        exp1_path = Path("experiments_1y_fixed/exp_1_Raw_Edge_Costs_OFF/artifacts")
        
        if not exp1_path.exists():
            pytest.skip(f"Exp 1 artifacts not found at {exp1_path}")
        
        metrics_path = exp1_path / "metrics.json"
        if not metrics_path.exists():
            pytest.skip(f"metrics.json not found at {metrics_path}")
        
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        # Check that all costs are zero
        tolerance = 0.01
        
        total_fees = metrics.get('total_fees', 0.0)
        assert abs(total_fees) < tolerance, f"total_fees should be 0, got {total_fees}"
        
        total_slippage_cost = metrics.get('total_slippage_cost', 0.0)
        assert abs(total_slippage_cost) < tolerance, f"total_slippage_cost should be 0, got {total_slippage_cost}"
        
        funding_cost_total = metrics.get('funding_cost_total', 0.0)
        assert abs(funding_cost_total) < tolerance, f"funding_cost_total should be 0, got {funding_cost_total}"
        
        # Check that pnl_net equals pnl_gross (since costs are zero)
        pnl_net = metrics.get('pnl_net_usd', 0.0)
        pnl_gross = metrics.get('pnl_gross_usd', 0.0)
        
        pnl_diff = abs(pnl_net - pnl_gross)
        assert pnl_diff < tolerance, f"pnl_net should equal pnl_gross when costs are off, diff={pnl_diff}"
    
    def test_exp1_total_return_calculation(self):
        """Test that total_return is calculated correctly"""
        exp1_path = Path("experiments_1y_fixed/exp_1_Raw_Edge_Costs_OFF/artifacts")
        
        if not exp1_path.exists():
            pytest.skip(f"Exp 1 artifacts not found at {exp1_path}")
        
        metrics_path = exp1_path / "metrics.json"
        if not metrics_path.exists():
            pytest.skip(f"metrics.json not found at {metrics_path}")
        
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        # Check total_return calculation
        initial_equity = metrics.get('initial_equity', 100000.0)
        final_equity = metrics.get('final_equity', initial_equity)
        total_return = metrics.get('total_return', 0.0)
        
        # total_return should be (final_equity / initial_equity) - 1
        expected_total_return = (final_equity / initial_equity) - 1.0
        
        tolerance = 0.0001  # 1 bps
        diff = abs(total_return - expected_total_return)
        assert diff < tolerance, (
            f"total_return calculation mismatch: "
            f"expected {expected_total_return:.6f}, got {total_return:.6f}, diff={diff:.6f}"
        )

