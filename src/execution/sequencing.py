"""Event sequencing per symbol per bar"""
from typing import List, Dict, Literal, Optional
from dataclasses import dataclass
import pandas as pd


@dataclass
class OrderEvent:
    """Order event for sequencing"""
    event_type: Literal['STOP', 'SQUEEZE_ENTRY', 'TREND_ENTRY', 'RANGE_ENTRY', 'SQUEEZE_NEW', 'NEUTRAL_ENTRY', 'TRAIL', 'TTL', 'STALE_CANCEL']
    symbol: str
    module: str
    priority: int  # Lower = higher priority
    signal_ts: Optional[pd.Timestamp] = None
    order_id: Optional[str] = None
    side: Optional[str] = None  # 'LONG' or 'SHORT' (optional for some events)


class EventSequencer:
    """Sequence events per symbol per bar according to STRATEGY_LOGIC.md §8.2"""
    
    def __init__(self):
        pass
    
    def sequence_events(
        self,
        events: List[OrderEvent]
    ) -> List[OrderEvent]:
        """
        Sequence events in correct order.
        
        Order:
        1. Stops first for non-SQUEEZE (adverse_first)
        2. SQUEEZE module (entry_first)
        3. New entries in fixed order: TREND → RANGE → SQUEEZE → NEUTRAL_Probe
        4. Trails: tighten only
        5. TTL/Expiry: SQUEEZE expires after 48 bars
        6. Stale: unfilled entries aged > 3 bars → cancel
        """
        # Define priority mapping
        priority_map = {
            'STOP': 1,
            'SQUEEZE_ENTRY': 2,
            'TREND_ENTRY': 3,
            'RANGE_ENTRY': 4,
            'SQUEEZE_NEW': 5,
            'NEUTRAL_ENTRY': 6,
            'TRAIL': 7,
            'TTL': 8,
            'STALE_CANCEL': 9
        }
        
        # Sort by priority, then by signal timestamp (oldest first for tie-breaker)
        sorted_events = sorted(
            events,
            key=lambda e: (
                priority_map.get(e.event_type, 999),
                e.signal_ts if e.signal_ts is not None else pd.Timestamp.max
            )
        )
        
        return sorted_events
    
    def group_by_symbol(self, events: List[OrderEvent]) -> Dict[str, List[OrderEvent]]:
        """Group events by symbol"""
        grouped = {}
        for event in events:
            if event.symbol not in grouped:
                grouped[event.symbol] = []
            grouped[event.symbol].append(event)
        return grouped

