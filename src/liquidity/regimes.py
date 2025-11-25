"""Liquidity regimes: VACUUM, THIN, NORMAL"""
import pandas as pd
import numpy as np
from typing import Literal, Optional, Tuple
from dataclasses import dataclass


LiquidityRegime = Literal['VACUUM', 'THIN', 'NORMAL']


@dataclass
class LiquidityState:
    """Liquidity regime state"""
    regime: LiquidityRegime
    spread_bps: float
    depth5_usd: float
    dwell_bars: int = 0
    entered_at_ts: Optional[pd.Timestamp] = None


class LiquidityRegimeDetector:
    """Detect and manage liquidity regimes"""
    
    def __init__(self, params: dict):
        self.params = params
        self.liquidity_params = params.get('liquidity_regimes', {})
        self.slippage_params = params.get('slippage_costs', {})
        
        # VACUUM thresholds
        self.vacuum_spread_enter = self.liquidity_params.get('vacuum_spread_enter_bps', 50.0)
        self.vacuum_spread_exit = self.liquidity_params.get('vacuum_spread_exit_bps', 30.0)
        self.vacuum_depth_enter_frac = self.liquidity_params.get('vacuum_depth_enter_frac', 0.1)
        self.vacuum_depth_exit_frac = self.liquidity_params.get('vacuum_depth_exit_frac', 0.2)
        self.vacuum_dwell_exit = self.liquidity_params.get('vacuum_dwell_exit_bars', 3)
        
        # THIN thresholds
        self.thin_seasonal_z_enter = self.liquidity_params.get('thin_seasonal_z_enter', 3.0)
        self.thin_seasonal_depth_pct_enter = self.liquidity_params.get('thin_seasonal_depth_pct_enter', 10.0)
        
        # Participation caps
        self.participation_cap_normal = self.slippage_params.get('participation_cap_normal', 0.01)
        self.participation_cap_thin = self.slippage_params.get('participation_cap_thin', 0.001)
    
    def detect_vacuum(
        self,
        spread_bps: float,
        depth5_usd: float,
        max_possible_notional: float,
        current_state: Optional[LiquidityState] = None
    ) -> Tuple[bool, bool]:
        """
        Detect VACUUM regime (reactive).
        
        Enter if: 1-min avg spread >= 50 bps OR mean Depth5 < 0.1 × max_possible_notional
        Exit when ALL: spread < 30 bps, depth > 0.2 × max_possible_notional, dwell >= 3 bars
        
        Returns:
            (should_enter, should_exit)
        """
        if current_state is None or current_state.regime != 'VACUUM':
            # Check enter conditions
            enter_spread = spread_bps >= self.vacuum_spread_enter
            enter_depth = depth5_usd < self.vacuum_depth_enter_frac * max_possible_notional
            
            should_enter = enter_spread or enter_depth
            
            return should_enter, False
        else:
            # Check exit conditions (all must be true)
            exit_spread = spread_bps < self.vacuum_spread_exit
            exit_depth = depth5_usd > self.vacuum_depth_exit_frac * max_possible_notional
            exit_dwell = current_state.dwell_bars >= self.vacuum_dwell_exit
            
            should_exit = exit_spread and exit_depth and exit_dwell
            
            return False, should_exit
    
    def detect_thin(
        self,
        spread_bps: float,
        depth5_usd: float,
        seasonal_spread_z: Optional[float] = None,
        seasonal_depth_pct: Optional[float] = None
    ) -> bool:
        """
        Detect THIN regime (seasonal).
        
        Enter if: seasonal spread z >= 3 OR seasonal depth <= 10th percentile
        """
        enter_spread = False
        enter_depth = False
        
        if seasonal_spread_z is not None:
            enter_spread = seasonal_spread_z >= self.thin_seasonal_z_enter
        
        if seasonal_depth_pct is not None:
            enter_depth = seasonal_depth_pct <= self.thin_seasonal_depth_pct_enter
        
        return enter_spread or enter_depth
    
    def update_regime(
        self,
        spread_bps: float,
        depth5_usd: float,
        max_possible_notional: float,
        current_state: Optional[LiquidityState],
        current_ts: pd.Timestamp,
        seasonal_spread_z: Optional[float] = None,
        seasonal_depth_pct: Optional[float] = None
    ) -> LiquidityState:
        """
        Update liquidity regime state.
        
        Priority: VACUUM > THIN > NORMAL
        """
        # Check VACUUM
        should_enter_vacuum, should_exit_vacuum = self.detect_vacuum(
            spread_bps, depth5_usd, max_possible_notional, current_state
        )
        
        if current_state is None or current_state.regime != 'VACUUM':
            if should_enter_vacuum:
                return LiquidityState(
                    regime='VACUUM',
                    spread_bps=spread_bps,
                    depth5_usd=depth5_usd,
                    dwell_bars=1,
                    entered_at_ts=current_ts
                )
        else:
            if should_exit_vacuum:
                # Exit VACUUM, check THIN
                current_state = None
            else:
                # Stay in VACUUM
                current_state.dwell_bars += 1
                current_state.spread_bps = spread_bps
                current_state.depth5_usd = depth5_usd
                return current_state
        
        # Check THIN (if not in VACUUM)
        is_thin = self.detect_thin(spread_bps, depth5_usd, seasonal_spread_z, seasonal_depth_pct)
        
        if is_thin:
            if current_state is None or current_state.regime != 'THIN':
                return LiquidityState(
                    regime='THIN',
                    spread_bps=spread_bps,
                    depth5_usd=depth5_usd,
                    dwell_bars=1,
                    entered_at_ts=current_ts
                )
            else:
                current_state.dwell_bars += 1
                current_state.spread_bps = spread_bps
                current_state.depth5_usd = depth5_usd
                return current_state
        
        # NORMAL
        if current_state is None or current_state.regime != 'NORMAL':
            return LiquidityState(
                regime='NORMAL',
                spread_bps=spread_bps,
                depth5_usd=depth5_usd,
                dwell_bars=1,
                entered_at_ts=current_ts
            )
        else:
            current_state.dwell_bars += 1
            current_state.spread_bps = spread_bps
            current_state.depth5_usd = depth5_usd
            return current_state
    
    def get_participation_cap(self, regime: LiquidityRegime) -> float:
        """Get participation cap for regime"""
        if regime == 'VACUUM':
            return 0.0  # Block entries
        elif regime == 'THIN':
            return self.participation_cap_thin
        else:  # NORMAL
            return self.participation_cap_normal
    
    def get_slippage_adder(self, regime: LiquidityRegime) -> float:
        """Get slippage adder for regime"""
        if regime == 'THIN':
            return self.slippage_params.get('thin_adder_bps', 15.0)
        else:
            return 0.0

