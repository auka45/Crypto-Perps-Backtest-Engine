"""
Strategy-specific event collection methods (ARCHIVE - NOT USED BY CORE ENGINE).

These methods were part of the original strategy implementation but are NOT used
by the Model-1 core engine. They are kept only as a reference for strategy
integrators who want to understand how strategy-specific logic was previously
implemented.

**DO NOT IMPORT FROM THIS MODULE IN CORE ENGINE CODE.**
This code is for reference only and is not part of the engine's public API.

All methods in this module return empty lists when oracle_mode=True, which is
the default for Model-1. They are never called in the core engine flow.
"""

from typing import List
import pandas as pd
from engine_core.src.execution.sequencing import OrderEvent


# NOTE: The full implementations of these methods are preserved in git history.
# They have been removed from engine.py to maintain Model-1 compliance.
# To view the original implementations, check git history before the freeze commit.

def collect_squeeze_tp1_events_stub():
    """
    Stub for collect_squeeze_tp1_events() - Strategy-specific (SQUEEZE TP1 exits).
    
    Original location: engine.py:1849-1884
    Status: Moved to archive - not used in Model-1
    """
    pass


def collect_squeeze_vol_exit_events_stub():
    """
    Stub for collect_squeeze_vol_exit_events() - Strategy-specific (SQUEEZE vol expansion exits).
    
    Original location: engine.py:1886-1958
    Status: Moved to archive - not used in Model-1
    """
    pass


def collect_squeeze_entry_events_stub():
    """
    Stub for collect_squeeze_entry_events() - Strategy-specific (SQUEEZE entries).
    
    Original location: engine.py:1960-2020
    Status: Moved to archive - not used in Model-1
    """
    pass


def collect_range_time_stops_stub():
    """
    Stub for collect_range_time_stops() - Strategy-specific (RANGE time stops).
    
    Original location: engine.py:2312-2360
    Status: Moved to archive - not used in Model-1
    """
    pass

