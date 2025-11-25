"""Beta controls: EWMA-OLS beta, shrinkage, caps"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from engine_core.src.risk.logging import log_risk_event


def calculate_ewma_ols_beta(
    asset_returns: pd.Series,
    market_returns: pd.Series,
    lambda_param: float = 0.999
) -> float:
    """
    Calculate EWMA-OLS beta.
    
    Uses exponentially weighted least squares regression.
    """
    if len(asset_returns) < 2 or len(market_returns) < 2:
        return 1.0
    
    # Align series
    aligned = pd.DataFrame({
        'asset': asset_returns,
        'market': market_returns
    }).dropna()
    
    if len(aligned) < 2:
        return 1.0
    
    # Calculate EWMA weights
    n = len(aligned)
    weights = []
    for i in range(n):
        weight = (1 - lambda_param) * (lambda_param ** (n - 1 - i))
        weights.append(weight)
    
    # Normalize weights
    total_weight = sum(weights)
    if total_weight > 0:
        weights = np.array([w / total_weight for w in weights])
    else:
        weights = np.ones(n) / n
    
    # Weighted OLS
    x = aligned['market'].values
    y = aligned['asset'].values
    
    # Weighted means
    x_mean = np.average(x, weights=weights)
    y_mean = np.average(y, weights=weights)
    
    # Weighted covariance and variance
    x_centered = x - x_mean
    y_centered = y - y_mean
    
    cov_xy = np.average(x_centered * y_centered, weights=weights)
    var_x = np.average(x_centered ** 2, weights=weights)
    
    if var_x == 0:
        return 1.0
    
    beta = cov_xy / var_x
    
    return beta


def apply_beta_shrinkage(
    beta_est: float,
    beta_prior: float,
    shrinkage_weight: float = 0.1
) -> float:
    """
    Apply shrinkage to beta estimate.
    
    β̂ = 0.90·β_est + 0.10·β_prior
    """
    return (1 - shrinkage_weight) * beta_est + shrinkage_weight * beta_prior


def check_beta_caps(
    positions: Dict[str, Dict],
    beta_slow: Dict[str, float],
    cap_net_beta_abs: float = 1.0,
    cap_gross_beta: float = 2.2
) -> Tuple[bool, float, float]:
    """
    Check beta caps.
    
    Net: |Σ w_i β_slow,i| ≤ 1.0
    Gross: Σ |w_i| |β_slow,i| ≤ 2.2
    
    Returns:
        (is_valid, net_beta, gross_beta)
    """
    net_beta = 0.0
    gross_beta = 0.0
    
    total_equity = sum(pos.get('notional', 0) for pos in positions.values())
    
    if total_equity == 0:
        return True, 0.0, 0.0
    
    for symbol, pos in positions.items():
        qty = pos.get('qty', 0)
        entry_price = pos.get('entry_price', 0)
        notional = abs(qty * entry_price)
        
        if notional == 0:
            continue
        
        # Weight
        w = notional / total_equity if total_equity > 0 else 0
        
        # Beta
        beta = beta_slow.get(symbol, 1.0)
        
        # Side (long = +1, short = -1)
        side_mult = 1.0 if qty > 0 else -1.0
        
        # Net beta
        net_beta += w * beta * side_mult
        
        # Gross beta
        gross_beta += w * abs(beta)
    
    # Check caps
    is_valid = abs(net_beta) <= cap_net_beta_abs and gross_beta <= cap_gross_beta
    
    return is_valid, net_beta, gross_beta


def apply_correlation_stress(
    correlation_matrix: np.ndarray,
    rho_floor_btc_alt: float = 0.9,
    rho_floor_cross_alt: float = 0.75,
    symbols: List[str] = None
) -> np.ndarray:
    """
    Apply correlation stress overlay when m <= 0.6.
    
    Floors: BTC-alt ρ=0.90; cross-alt ρ=0.75
    """
    if symbols is None:
        return correlation_matrix
    
    stressed = correlation_matrix.copy()
    
    # Identify BTC and alts
    btc_idx = None
    alt_indices = []
    
    for i, sym in enumerate(symbols):
        if 'BTC' in sym:
            btc_idx = i
        else:
            alt_indices.append(i)
    
    # Apply floors
    if btc_idx is not None:
        for alt_idx in alt_indices:
            # BTC-alt floor
            if stressed[btc_idx, alt_idx] < rho_floor_btc_alt:
                stressed[btc_idx, alt_idx] = rho_floor_btc_alt
                stressed[alt_idx, btc_idx] = rho_floor_btc_alt
    
    # Cross-alt floor
    for i in range(len(alt_indices)):
        for j in range(i + 1, len(alt_indices)):
            idx_i = alt_indices[i]
            idx_j = alt_indices[j]
            if stressed[idx_i, idx_j] < rho_floor_cross_alt:
                stressed[idx_i, idx_j] = rho_floor_cross_alt
                stressed[idx_j, idx_i] = rho_floor_cross_alt
    
    return stressed


# Launch Punch List – Blocker #4: enforce BTC-beta caps (symbol + portfolio)
def check_portfolio_beta_caps(
    positions: Dict[str, Dict],
    beta_slow: Dict[str, float],
    new_symbol: str,
    new_qty: float,
    new_price: float,
    new_side: str,  # 'LONG' or 'SHORT'
    max_symbol_beta: float = 1.5,
    max_portfolio_beta: float = 3.0,
    reference_symbol: str = "BTCUSDT"
) -> Tuple[bool, float, float, str]:
    """
    Check portfolio-level beta caps including a new position.
    
    Computes per-symbol beta exposure and portfolio total, and blocks if caps would be exceeded.
    
    Args:
        positions: Current positions dict
        beta_slow: Beta values per symbol
        new_symbol: Symbol for new position
        new_qty: Quantity for new position
        new_price: Price for new position
        new_side: Side for new position ('LONG' or 'SHORT')
        max_symbol_beta: Maximum beta exposure per symbol
        max_portfolio_beta: Maximum total portfolio beta exposure
        reference_symbol: Reference symbol for beta calculation (default: BTCUSDT)
    
    Returns:
        (is_valid, symbol_beta_exposure, portfolio_beta_exposure, reason)
        - is_valid: True if order is allowed, False if blocked
        - symbol_beta_exposure: Beta exposure for the symbol (beta * notional)
        - portfolio_beta_exposure: Total portfolio beta exposure
        - reason: Reason for block (empty if allowed)
    """
    # Create projected positions dict including new position
    projected_positions = {}
    for sym, pos in positions.items():
        if hasattr(pos, 'qty'):
            projected_positions[sym] = {
                'qty': pos.qty,
                'entry_price': pos.entry_price,
                'notional': abs(pos.qty * pos.entry_price),
                'side': pos.side if hasattr(pos, 'side') else 'LONG'
            }
        else:
            projected_positions[sym] = {
                'qty': pos.get('qty', 0),
                'entry_price': pos.get('entry_price', 0),
                'notional': abs(pos.get('qty', 0) * pos.get('entry_price', 0)),
                'side': pos.get('side', 'LONG')
            }
    
    # Add new position
    new_notional = abs(new_qty * new_price)
    if new_symbol in projected_positions:
        # If symbol already has position, update it (for simplicity, replace)
        projected_positions[new_symbol] = {
            'qty': new_qty,
            'entry_price': new_price,
            'notional': new_notional,
            'side': new_side
        }
    else:
        projected_positions[new_symbol] = {
            'qty': new_qty,
            'entry_price': new_price,
            'notional': new_notional,
            'side': new_side
        }
    
    # Calculate total equity (sum of notionals as proxy)
    total_equity = sum(pos['notional'] for pos in projected_positions.values())
    if total_equity == 0:
        return True, 0.0, 0.0, ""
    
    # Calculate per-symbol beta exposure
    symbol_beta_exposure = 0.0
    if new_symbol in beta_slow:
        beta = beta_slow[new_symbol]
        side_mult = 1.0 if new_side == 'LONG' else -1.0
        # Beta exposure = beta * notional * side_mult
        symbol_beta_exposure = beta * new_notional * side_mult
    
    # Calculate portfolio total beta exposure
    portfolio_beta_exposure = 0.0
    for sym, pos in projected_positions.items():
        if pos['notional'] == 0:
            continue
        beta = beta_slow.get(sym, 1.0)
        side_mult = 1.0 if pos['side'] == 'LONG' else -1.0
        # Weighted beta exposure
        weight = pos['notional'] / total_equity
        portfolio_beta_exposure += weight * beta * side_mult * total_equity  # Scale by total equity
    
    # Normalize portfolio beta exposure to absolute value
    portfolio_beta_exposure_abs = abs(portfolio_beta_exposure)
    symbol_beta_exposure_abs = abs(symbol_beta_exposure)
    
    # Check symbol beta cap
    if symbol_beta_exposure_abs > max_symbol_beta * total_equity:
        reason = f"SYMBOL_BETA: {new_symbol} beta_exposure={symbol_beta_exposure_abs:.2f} > max={max_symbol_beta * total_equity:.2f}"
        # Launch Punch List – Quick Win #1: structured risk logging
        log_risk_event(
            'beta_block',
            {
                'symbol': new_symbol,
                'reason': reason,
                'symbol_beta_exposure': symbol_beta_exposure_abs,
                'max_symbol_beta': max_symbol_beta * total_equity,
                'portfolio_beta_exposure': portfolio_beta_exposure_abs,
                'total_equity': total_equity
            }
        )
        return False, symbol_beta_exposure_abs, portfolio_beta_exposure_abs, reason
    
    # Check portfolio beta cap
    if portfolio_beta_exposure_abs > max_portfolio_beta * total_equity:
        reason = f"PORTFOLIO_BETA: portfolio beta_exposure={portfolio_beta_exposure_abs:.2f} > max={max_portfolio_beta * total_equity:.2f}"
        # Launch Punch List – Quick Win #1: structured risk logging
        log_risk_event(
            'beta_block',
            {
                'symbol': new_symbol,
                'reason': reason,
                'symbol_beta_exposure': symbol_beta_exposure_abs,
                'portfolio_beta_exposure': portfolio_beta_exposure_abs,
                'max_portfolio_beta': max_portfolio_beta * total_equity,
                'total_equity': total_equity
            }
        )
        return False, symbol_beta_exposure_abs, portfolio_beta_exposure_abs, reason
    
    # All checks passed
    return True, symbol_beta_exposure_abs, portfolio_beta_exposure_abs, ""

