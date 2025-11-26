# Dead Code Analysis & Classification

**Date**: 2025-01-XX  
**Coverage Run**: `coverage run -m pytest tests/`  
**Overall Coverage**: 51% (7091 statements, 3479 missing)

## Strategy-Specific Code Classification

### Classification A: Truly Unused (Delete)

None identified - all strategy-specific code is referenced in engine flow (even if guarded by oracle_mode).

### Classification B: Dead Branches (Remove unreachable code)

1. **`src/engine.py:1849-1884`** - `collect_squeeze_tp1_events()`
   - Classification: **C** (Extension hook - kept for reference but not used in Model-1)
   - Action: Move to `src/archive/strategy_event_collectors.py`
   - Reason: Strategy-specific, guarded by `oracle_mode` check, returns empty in Model-1

2. **`src/engine.py:1886-1958`** - `collect_squeeze_vol_exit_events()`
   - Classification: **C** (Extension hook)
   - Action: Move to `src/archive/strategy_event_collectors.py`
   - Reason: Strategy-specific, guarded by `oracle_mode` check

3. **`src/engine.py:1960-2020`** - `collect_squeeze_entry_events()`
   - Classification: **C** (Extension hook)
   - Action: Move to `src/archive/strategy_event_collectors.py`
   - Reason: Strategy-specific, guarded by `oracle_mode` check

4. **`src/engine.py:2312-2360`** - `collect_range_time_stops()`
   - Classification: **C** (Extension hook)
   - Action: Move to `src/archive/strategy_event_collectors.py`
   - Reason: Strategy-specific, guarded by `oracle_mode` check

5. **`src/engine.py:2763-2911`** - `execute_squeeze_tp1()`
   - Classification: **C** (Extension hook)
   - Action: Move to `src/archive/strategy_executors.py`
   - Reason: Strategy-specific executor, never called in Model-1

6. **`src/engine.py:2913-3049`** - `execute_squeeze_vol_exit()`
   - Classification: **C** (Extension hook)
   - Action: Move to `src/archive/strategy_executors.py`
   - Reason: Strategy-specific executor, never called in Model-1

7. **`src/engine.py:3051-3246`** - `execute_squeeze_entry()`
   - Classification: **C** (Extension hook)
   - Action: Move to `src/archive/strategy_executors.py`
   - Reason: Strategy-specific executor, never called in Model-1

### Classification C: Extension Hooks / Experimental (Move to archive/)

**Strategy-specific event collection methods** (all in `src/engine.py`):
- `collect_squeeze_tp1_events()` - lines 1849-1884
- `collect_squeeze_vol_exit_events()` - lines 1886-1958
- `collect_squeeze_entry_events()` - lines 1960-2020
- `collect_range_time_stops()` - lines 2312-2360

**Strategy-specific execution methods** (all in `src/engine.py`):
- `execute_squeeze_tp1()` - lines 2763-2911
- `execute_squeeze_vol_exit()` - lines 2913-3049
- `execute_squeeze_entry()` - lines 3051-3246

**Strategy-specific logic in active methods**:
- `collect_new_entry_events()` - lines 2022-2278: Contains TREND/RANGE/SQUEEZE/NEUTRAL_PROBE logic
  - Action: Simplify to only handle ORACLE signals
  - Remove: module_order loop for TREND/RANGE/SQUEEZE/NEUTRAL_PROBE
  - Keep: ORACLE signal handling

- `collect_trail_events()` - lines 2280-2310: Contains TREND/SQUEEZE-specific logic
  - Action: Simplify to generic trailing (remove module checks)
  - Or: Remove if only used for strategy-specific modules

- `collect_ttl_events()` - lines 2358-2382: Contains SQUEEZE-specific TTL logic
  - Action: Simplify to generic TTL or remove if only for SQUEEZE

**Event types in `src/execution/sequencing.py`**:
- `SQUEEZE_ENTRY`, `TREND_ENTRY`, `RANGE_ENTRY`, `SQUEEZE_NEW`, `NEUTRAL_ENTRY`
  - Action: Remove from `OrderEvent.event_type` Literal, keep only `ORACLE_ENTRY` and generic types

### Classification D: Public API, Hard to Test (Add tests or document)

1. **`src/portfolio/universe.py`** - 33% coverage
   - Many methods untested
   - Action: Review if truly needed, add tests if public API

2. **`src/liquidity/seasonal.py`** - 26% coverage
   - Seasonal profile logic untested
   - Action: Review usage, add tests if used

3. **`src/risk/beta_controls.py`** - 46% coverage
   - Beta calculation logic partially tested
   - Action: Add tests for uncovered paths

## Files Requiring Modification

### Core Engine Files
1. **`src/engine.py`** (major cleanup)
   - Remove/quarantine: 7 strategy-specific methods
   - Simplify: `collect_new_entry_events()` to ORACLE-only
   - Simplify: `collect_trail_events()` to generic
   - Simplify: `collect_ttl_events()` to generic
   - Update: Event handling to remove strategy-specific event types

2. **`src/execution/sequencing.py`**
   - Remove: Strategy-specific event types from `OrderEvent.event_type` Literal
   - Update: Comments to remove strategy references
   - Keep: Generic sequencing logic

3. **`src/portfolio/state.py`**
   - Review: `initial_R` and `tp1_price` fields (SQUEEZE-specific?)
   - Action: Keep if used by oracle, document if strategy-specific

4. **`src/execution/funding_windows.py`**
   - Review: `squeeze_disable_minutes` parameter name
   - Action: Consider renaming to generic `funding_throttle_minutes` or document as legacy

### Archive Files to Create
1. **`src/archive/__init__.py`** - Empty, marks as archive namespace
2. **`src/archive/strategy_event_collectors.py`** - Moved strategy-specific collectors
3. **`src/archive/strategy_executors.py`** - Moved strategy-specific executors

## Coverage Summary by Module

| Module | Coverage | Missing Lines | Status |
|--------|----------|---------------|--------|
| `src/engine.py` | 42% | 1190 | **Needs cleanup** - strategy-specific code |
| `src/reporting.py` | 67% | 369 | Acceptable - reporting edge cases |
| `src/portfolio/universe.py` | 33% | 46 | **Needs review** - low coverage |
| `src/liquidity/seasonal.py` | 26% | 43 | **Needs review** - low coverage |
| `src/risk/beta_controls.py` | 46% | 61 | **Needs tests** - public API |
| `src/risk/loss_halts.py` | 60% | 49 | Acceptable |
| `src/execution/order_manager.py` | 57% | 31 | Acceptable |
| `src/data/loader.py` | 48% | 72 | Acceptable - data loading edge cases |
| `src/indicators/*` | 69-96% | Low | **Good** |
| `src/execution/funding_windows.py` | 100% | 0 | **Excellent** |
| `src/modules/oracle.py` | 85% | 8 | **Good** |

## Recommended Actions

1. **Immediate (Phase 3)**:
   - Create `src/archive/` directory
   - Move 7 strategy-specific methods to archive
   - Simplify `collect_new_entry_events()` to ORACLE-only
   - Update `sequencing.py` to remove strategy event types
   - Re-run tests after each change

2. **Follow-up (Phase 4)**:
   - Review low-coverage modules (`universe.py`, `seasonal.py`)
   - Add tests for `beta_controls.py` uncovered paths
   - Update documentation to reflect cleaned codebase

3. **Documentation**:
   - Document `archive/` as reference-only for strategy integrators
   - Update `ENGINE_OVERVIEW.md` to clarify Model-1 boundary
   - Create `INTEGRATING_YOUR_STRATEGY.md` guide

