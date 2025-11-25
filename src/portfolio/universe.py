"""Universe governance: OI/ADV filters, drop rules"""
import pandas as pd
import numpy as np
from typing import Dict, List, Set, Optional
from datetime import timedelta


class UniverseManager:
    """Manage trading universe"""
    
    def __init__(self, params: dict):
        self.params = params
        self.universe_params = params.get('universe', {})
        
        self.initial_symbols = self.universe_params.get('initial', [])
        self.refresh_days = self.universe_params.get('refresh_days', {}).get('default', 7)
        
        self.include_thresholds = self.universe_params.get('include_thresholds', {})
        self.drop_rules = self.universe_params.get('drop_rules', {})
        self.readd_days = self.universe_params.get('readd_days', 14)
        self.memecoin_cap = self.universe_params.get('memecoin_cap', 1)
        
        # State
        self.active_symbols: Set[str] = set(self.initial_symbols)
        self.disabled_symbols: Dict[str, pd.Timestamp] = {}  # {symbol: disabled_until}
        self.symbol_quality_scores: Dict[str, float] = {}
        self.last_refresh: Optional[pd.Timestamp] = None
    
    def check_include_thresholds(
        self,
        symbol: str,
        oi_usd: float,
        adv60_usd: float
    ) -> bool:
        """Check if symbol meets include thresholds"""
        oi_min = self.include_thresholds.get('open_interest_min_usd', 10000000)
        adv_min = self.include_thresholds.get('adv60_min_usd', 50000000)
        
        # BTC/ETH always included
        if 'BTC' in symbol or 'ETH' in symbol:
            return True
        
        return oi_usd >= oi_min and adv60_usd >= adv_min
    
    def check_drop_conditions(
        self,
        symbol: str,
        oi_usd: float,
        adv60_usd: float,
        median_spread_bps: float,
        consecutive_fail_days: int
    ) -> bool:
        """Check if symbol should be dropped"""
        # BTC/ETH never dropped
        if 'BTC' in symbol or 'ETH' in symbol:
            return False
        
        oi_min = self.drop_rules.get('oi_min_usd', 10000000)
        adv_min = self.drop_rules.get('adv60_min_usd', 50000000)
        spread_max = self.drop_rules.get('median_spread_bps_7d_max', 15)
        consecutive_days = self.drop_rules.get('consecutive_days', 3)
        
        # Check if any condition fails
        fails_oi = oi_usd < oi_min
        fails_adv = adv60_usd < adv_min
        fails_spread = median_spread_bps > spread_max
        
        fails_any = fails_oi or fails_adv or fails_spread
        
        if fails_any and consecutive_fail_days >= consecutive_days:
            return True
        
        return False
    
    def calculate_quality_score(
        self,
        oi_usd: float,
        adv60_usd: float,
        spread_bps: float
    ) -> float:
        """
        Calculate quality score.
        
        Score = 0.5·z(OI) + 0.3·z(ADV) − 0.2·z(spread)
        """
        # For simplicity, use normalized values (would need historical distribution for true z-scores)
        # This is a placeholder - full implementation would maintain rolling distributions
        oi_z = np.log1p(oi_usd / 10000000)  # Approximate z-score
        adv_z = np.log1p(adv60_usd / 50000000)
        spread_z = -np.log1p(spread_bps / 10.0)  # Negative (lower spread = better)
        
        score = 0.5 * oi_z + 0.3 * adv_z - 0.2 * spread_z
        
        return score
    
    def refresh_universe(
        self,
        current_ts: pd.Timestamp,
        symbol_data: Dict[str, Dict]  # {symbol: {oi, adv60, spread, ...}}
    ):
        """Refresh universe (weekly)"""
        # Check if it's time to refresh
        if self.last_refresh is None:
            self.last_refresh = current_ts
        
        days_since_refresh = (current_ts - self.last_refresh).days
        
        if days_since_refresh < self.refresh_days:
            return
        
        # Refresh logic
        for symbol, data in symbol_data.items():
            oi = data.get('oi_usd', 0)
            adv60 = data.get('adv60_usd', 0)
            spread = data.get('median_spread_bps', 0)
            
            # Check include thresholds
            if self.check_include_thresholds(symbol, oi, adv60):
                if symbol not in self.active_symbols:
                    # Check if re-add period has passed
                    if symbol in self.disabled_symbols:
                        if current_ts >= self.disabled_symbols[symbol]:
                            self.active_symbols.add(symbol)
                            del self.disabled_symbols[symbol]
                    else:
                        self.active_symbols.add(symbol)
            
            # Check drop conditions
            if symbol in self.active_symbols:
                consecutive_fail = data.get('consecutive_fail_days', 0)
                if self.check_drop_conditions(symbol, oi, adv60, spread, consecutive_fail):
                    self.active_symbols.remove(symbol)
                    self.disabled_symbols[symbol] = current_ts + timedelta(days=self.readd_days)
        
        self.last_refresh = current_ts
    
    def get_active_symbols(self) -> List[str]:
        """Get list of active symbols"""
        return sorted(list(self.active_symbols))

