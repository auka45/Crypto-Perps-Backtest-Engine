"""Order management: stale orders, TTL, OCO"""
from typing import Dict, List, Optional
from dataclasses import dataclass
import pandas as pd


@dataclass
class PendingOrder:
    """Pending order"""
    order_id: str
    symbol: str
    side: str  # 'LONG' or 'SHORT'
    order_type: str  # 'ENTRY', 'STOP', 'TRAIL', 'OCO'
    trigger_price: float
    qty: float
    module: str
    signal_bar_idx: int
    signal_ts: pd.Timestamp
    created_at_ts: pd.Timestamp
    ttl_bars: Optional[int] = None
    oco_id: Optional[str] = None
    filled: bool = False
    filled_price: Optional[float] = None
    filled_ts: Optional[pd.Timestamp] = None


class OrderManager:
    """Manage pending orders: stale cleanup, TTL, OCO"""
    
    def __init__(self, params: dict):
        self.params = params
        stale_order_bars_val = params.get('general', {}).get('stale_order_bars', {})
        if isinstance(stale_order_bars_val, dict):
            self.stale_order_bars = stale_order_bars_val.get('default', 3)
        else:
            self.stale_order_bars = stale_order_bars_val
        self.pending_orders: Dict[str, PendingOrder] = {}
    
    def add_order(self, order: PendingOrder):
        """Add pending order"""
        self.pending_orders[order.order_id] = order
    
    def cancel_order(self, order_id: str):
        """Cancel order"""
        if order_id in self.pending_orders:
            del self.pending_orders[order_id]
    
    def fill_order(self, order_id: str, fill_price: float, fill_ts: pd.Timestamp):
        """Mark order as filled"""
        if order_id in self.pending_orders:
            order = self.pending_orders[order_id]
            order.filled = True
            order.filled_price = fill_price
            order.filled_ts = fill_ts
    
    def check_stale_orders(self, current_bar_idx: int, current_ts: pd.Timestamp) -> List[str]:
        """
        Check for stale orders (unfilled entries aged > 3 bars).
        
        Do NOT cancel working OCO exits.
        """
        stale_order_ids = []
        
        for order_id, order in list(self.pending_orders.items()):
            if order.filled:
                continue
            
            # Only check entry orders (not stops/trails)
            if order.order_type != 'ENTRY':
                continue
            
            # Check age
            age_bars = current_bar_idx - order.signal_bar_idx
            
            if age_bars > self.stale_order_bars:
                stale_order_ids.append(order_id)
        
        return stale_order_ids
    
    def check_ttl_orders(self, current_bar_idx: int, current_ts: pd.Timestamp) -> List[str]:
        """
        Check for expired TTL orders (SQUEEZE expires after 48 bars).
        """
        expired_order_ids = []
        
        for order_id, order in list(self.pending_orders.items()):
            if order.filled:
                continue
            
            if order.ttl_bars is None:
                continue
            
            # Check if TTL expired
            age_bars = current_bar_idx - order.signal_bar_idx
            
            if age_bars >= order.ttl_bars:
                expired_order_ids.append(order_id)
        
        return expired_order_ids
    
    def cancel_oco_orders(self, oco_id: str, reason: str = 'OCO_CANCEL'):
        """Cancel all orders with matching OCO ID"""
        cancelled_ids = []
        
        for order_id, order in list(self.pending_orders.items()):
            if order.oco_id == oco_id:
                cancelled_ids.append(order_id)
                del self.pending_orders[order_id]
        
        return cancelled_ids
    
    def get_orders_by_symbol(self, symbol: str) -> List[PendingOrder]:
        """Get all pending orders for symbol"""
        return [order for order in self.pending_orders.values() if order.symbol == symbol and not order.filled]
    
    def get_orders_by_module(self, module: str) -> List[PendingOrder]:
        """Get all pending orders for module"""
        return [order for order in self.pending_orders.values() if order.module == module and not order.filled]

