"""
Strategy-specific execution methods (ARCHIVE - NOT USED BY CORE ENGINE).

These methods were part of the original strategy implementation but are NOT used
by the Model-1 core engine. They are kept only as a reference for strategy
integrators who want to understand how strategy-specific logic was previously
implemented.

**DO NOT IMPORT FROM THIS MODULE IN CORE ENGINE CODE.**
This code is for reference only and is not part of the engine's public API.

All methods in this module handle strategy-specific execution logic for
TREND, RANGE, SQUEEZE, and NEUTRAL_PROBE modules, which are not supported
in Model-1.
"""

from typing import List
import pandas as pd


# NOTE: The full implementations of these methods are preserved in git history.
# They have been removed from engine.py to maintain Model-1 compliance.
# To view the original implementations, check git history before the freeze commit.

def execute_squeeze_tp1_stub():
    """
    Stub for execute_squeeze_tp1() - Strategy-specific (SQUEEZE TP1 exit execution).
    
    Original location: engine.py:2763-2911
    Status: Moved to archive - not used in Model-1
    """
    pass


def execute_squeeze_vol_exit_stub():
    """
    Stub for execute_squeeze_vol_exit() - Strategy-specific (SQUEEZE vol expansion exit execution).
    
    Original location: engine.py:2913-3049
    Status: Moved to archive - not used in Model-1
    """
    pass


def execute_squeeze_entry_stub():
    """
    Stub for execute_squeeze_entry() - Strategy-specific (SQUEEZE entry execution).
    
    Original location: engine.py:3051-3246
    Status: Moved to archive - not used in Model-1
    """
    pass

