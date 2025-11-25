"""Exchange constraint enforcement"""
import numpy as np
from typing import Dict, Optional, Tuple


def enforce_tick_size(price: float, tick_size: float) -> float:
    """Round price to nearest tick"""
    if tick_size <= 0:
        return price
    return round(price / tick_size) * tick_size


def enforce_step_size(qty: float, step_size: float) -> float:
    """Round quantity down to step size"""
    if step_size <= 0:
        return qty
    return np.floor(qty / step_size) * step_size


def validate_order_constraints(
    qty: float,
    price: float,
    contract_metadata: Dict,
    side: str = 'LONG'
) -> Tuple[bool, Optional[str], float, float]:
    """
    Validate and enforce exchange constraints.
    
    Returns:
        (is_valid, error_message, adjusted_qty, adjusted_price)
    """
    tick_size = contract_metadata.get('tickSize', 0.01)
    step_size = contract_metadata.get('stepSize', 0.001)
    min_qty = contract_metadata.get('minQty', 0.001)
    min_notional = contract_metadata.get('minNotional', 5.0)
    
    # Enforce step size (round down)
    adjusted_qty = enforce_step_size(qty, step_size)
    
    # Enforce tick size
    adjusted_price = enforce_tick_size(price, tick_size)
    
    # Check minimum quantity
    if adjusted_qty < min_qty:
        return False, f"Quantity {adjusted_qty} below minQty {min_qty}", 0.0, adjusted_price
    
    # Check minimum notional
    notional = adjusted_qty * adjusted_price
    if notional < min_notional:
        return False, f"Notional {notional} below minNotional {min_notional}", 0.0, adjusted_price
    
    # Reject if qty == 0 after rounding
    if adjusted_qty == 0:
        return False, "Quantity is zero after step size rounding", 0.0, adjusted_price
    
    return True, None, adjusted_qty, adjusted_price

