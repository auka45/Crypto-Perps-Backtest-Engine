# REPO MAP

**Generated on:** 2025-11-23  
**Purpose:** Complete enumeration of execution/cost/metrics call-sites for validation

## Data Pipeline
- `src/data/loader.py`: Handles loading of OHLCV, liquidity, funding, and contract metadata. Supports CSV/Parquet.
- `src/data/schema.py`: Validates dataframes against expected schemas.

## Signal Generation
- `src/regimes/classifier.py`: Market regime classification (BULL/BEAR/NEUTRAL).
- `src/regimes/master_side.py`: Determines master bias.
- `src/modules/trend.py`: Trend-following signal generation logic.
- `src/modules/range.py`: Mean-reversion signal generation logic.
- `src/modules/squeeze.py`: Volatility breakout signal generation logic.
- `src/indicators/`: Technical indicator calculations.

## Execution & Order Management

### Core Event Loop (`src/engine.py`)
- `BacktestEngine.process_bar_t()`: Signal generation phase (bar t)
- `BacktestEngine.process_bar_t_plus_1()`: Order execution phase (bar t+1)
- `BacktestEngine._execute_events()`: Handles fills from event queue
- `BacktestEngine._collect_new_entry_events()`: Collects entry signals into OrderEvents
- `BacktestEngine.collect_squeeze_entry_events()`: Collects SQUEEZE entry signals
- `BacktestEngine._apply_direction_gates()`: Filters entry events by long_only/short_only

### Entry Execution (`src/engine.py`)
- `BacktestEngine.execute_entry()`: **PRIMARY ENTRY PATH**
  - Line ~3410: Calculates fill price via `fill_stop_run()`
  - Line ~3703: Calculates fees (maker/taker, checks `cost_model.enabled`)
  - Line ~3707-3709: **COST TOGGLE CHECK**: Sets fees/slippage to 0 if `cost_model_enabled=False`
  - Line ~3738-3745: Calculates slippage_bps_applied, zeroes if cost disabled
  - Line ~3747: Calculates slippage_cost_usd
  - Line ~3750-3768: Records fill via `_record_fill()`
  - Line ~3771-3787: Records ledger event via `_record_ledger_event()`
  - Line ~3730: Adds position via `portfolio.add_position()` (includes fees)
- `BacktestEngine.execute_squeeze_entry()`: SQUEEZE entry path (similar cost logic)

### Exit Execution (`src/engine.py`)
- `BacktestEngine.execute_stop()`: Stop loss exit (market order, taker fees)
- `BacktestEngine.execute_exit()`: Profit target exit (limit order)
- `BacktestEngine.execute_trail()`: Trailing stop exit (market order, taker fees)
- `BacktestEngine.execute_ttl()`: Time-based exit for SQUEEZE (market order, taker fees)
- `BacktestEngine.execute_squeeze_tp1()`: SQUEEZE TP1 exit
- `BacktestEngine.execute_squeeze_vol_exit()`: SQUEEZE volatility expansion exit
- All exit methods:
  - Calculate fees (taker fees for market orders)
  - Check `cost_model_enabled` and zero costs if disabled
  - Record fill and ledger events
  - Close position via `portfolio.close_position()` (calculates realized PnL)

### Margin Flatten (`src/engine.py`)
- Line ~1227-1297: Margin flatten logic
  - Calculates fees and slippage for forced exits
  - Checks `cost_model_enabled` (line ~1239-1240, ~1272-1273)
  - Records fills and closes positions

### Order Management
- `src/execution/order_manager.py`: Manages pending orders (stale checks, TTL)
- `src/execution/sequencing.py`: Sorts events by priority (`OrderEvent` dataclass)
- `src/execution/fill_model.py`: 
  - `calculate_slippage()`: Calculates slippage bps from participation rate
  - `fill_stop_run()`: Deterministic fill price calculation (stop-run model)
  - `calculate_adv_60m()`: Calculates 60-minute average dollar volume
- `src/execution/constraints.py`: Order validation (min qty, tick size)
- `src/execution/funding_windows.py`: Checks for funding rate windows to block/close trades

## Risk Management
- `src/risk/sizing.py`: Position sizing logic, volatility scaling, module factors
- `src/risk/es_guardrails.py`: Expected Shortfall (ES) calculation and checks
- `src/risk/beta_controls.py`: Portfolio beta capping
- `src/risk/margin_guard.py`: Margin ratio checks and trimming logic
- `src/risk/loss_halts.py`: Daily loss limits and kill-switch logic

## Cost Model Call-Sites

### Fee Calculation
1. **Entry Fees** (`src/engine.py:execute_entry()` ~line 3703-3711):
   - Maker fee: `params.get_default('general', 'maker_fee_bps')` (default 2 bps)
   - Taker fee: `params.get_default('general', 'taker_fee_bps')` (default 4 bps)
   - Entry orders: Taker fees unless `post_only=True` (THIN regime)
   - **COST TOGGLE**: Line 3707-3709: `if not self.cost_model_enabled: fee_bps = 0.0`
   - Applied: `fees = notional * (fee_bps / 10000.0)`

2. **Exit Fees** (`src/engine.py:execute_stop/exit/trail/ttl()`):
   - All exit orders: Taker fees (market orders)
   - **COST TOGGLE**: Checked in each exit method
   - Applied same formula: `fees = notional * (fee_bps / 10000.0)`

3. **Margin Flatten Fees** (`src/engine.py` ~line 1235-1242):
   - Taker fees for forced exits
   - **COST TOGGLE**: Line 1239-1240

### Slippage Calculation
1. **Entry Slippage** (`src/engine.py:execute_entry()` ~line 3738-3747):
   - Calculated via `fill_stop_run()` in `src/execution/fill_model.py`
   - Uses `calculate_slippage()` to get slippage_bps
   - **COST TOGGLE**: Line 3744-3745: `if not self.cost_model_enabled: slippage_bps_applied = 0.0`
   - Applied: `slippage_cost_usd = notional * (slippage_bps_applied / 10000.0)`

2. **Exit Slippage**: Similar logic in exit methods

3. **Slippage Model** (`src/execution/fill_model.py`):
   - `calculate_slippage()`: Base 2 bps + 20 bps × participation_pct + regime_adder
   - `fill_stop_run()`: Applies slippage to fill price (bound by bar high/low)

### Funding Costs
1. **Funding Application** (`src/engine.py:apply_funding_costs()` ~line 4344-4387):
   - Called at exact funding times: 00:00, 08:00, 16:00 UTC (line 4355-4370)
   - Calculates via `portfolio.calculate_funding_cost()` (line 4382)
   - **COST TOGGLE**: Line 4384-4385: `if not self.cost_model_enabled: cost = 0.0`
   - Applied: `portfolio.cash -= cost` (line 278 in `portfolio/state.py`)

2. **Funding Cost Calculation** (`src/portfolio/state.py:calculate_funding_cost()` ~line 257-280):
   - Adverse funding only: Long pays when rate > 0, Short pays when rate < 0
   - Formula: `cost = notional * funding_rate` (for adverse cases)

## Portfolio State Management (`src/portfolio/state.py`)

### Position Management
- `PortfolioState.add_position()`: Adds new position, updates cash (subtracts fees)
- `PortfolioState.close_position()`: Closes position, calculates realized PnL, updates cash
- `PortfolioState.update_position()`: Updates position PnL (unrealized)

### Equity Tracking
- `PortfolioState.equity`: Updated in `BacktestEngine._update_equity_curve()`
- Formula: `equity = cash + Σ(unrealized_pnl)` (for futures)
- Updated after each fill and funding event

### Cost Accumulation
- `PortfolioState.fees_paid`: Accumulated in `add_position()` and `close_position()`
- `PortfolioState.slippage_paid`: Accumulated in fill recording
- `PortfolioState.funding_paid`: Accumulated in `calculate_funding_cost()`

## Reporting & Metrics (`src/reporting.py`)

### Artifact Generation
- `ReportGenerator._write_fills_artifact()`: Writes `fills.csv` (SSOT for execution details)
- `ReportGenerator._write_trades_artifact()`: Writes `trades.csv` (rebuilds from fills, calculates round-trip metrics)
- `ReportGenerator._write_ledger_artifact()`: Writes `ledger.csv` (SSOT for cash movements)
- `ReportGenerator._write_equity_artifact()`: Writes `equity.csv` (equity curve)

### Metrics Calculation (`src/reporting.py:_calculate_metrics()` ~line 524-1247)
- **Performance Metrics**: total_return, cagr, sharpe, sortino, calmar, mar
- **Trading Metrics**: total_trades, win_rate, profit_factor, avg_win, avg_loss
- **Cost Metrics**: 
  - `total_fees`: Summed from `fills.csv` (line 1069-1079)
  - `funding_cost_total`: Summed from `ledger.csv` (line 1086-1098)
  - `slippage_bps_realized`: Calculated from fills
- **Risk Metrics**: max_drawdown, es_violations_count, margin_trim_count

### Metric Sources (SSOT)
- **Fills**: `fills.csv` (entry/exit fills with fees, slippage, participation)
- **Trades**: `trades.csv` (round-trip trades with PnL, costs)
- **Ledger**: `ledger.csv` (all cash movements: fees, funding, PnL)
- **Equity**: `equity.csv` (equity curve over time)

## Invariant Checks (`src/engine.py:_check_invariants()` ~line 4418-4570)

Called when `general.debug_invariants=true`:
1. **Equity Identity**: `equity = cash + unrealized_pnl` (line 4456-4469)
2. **Position Conservation**: Implicit (positions only added/removed via methods)
3. **Realized PnL Conservation**: Verified via ledger reconciliation
4. **Cost Signs**: Fees/slippage <= 0, funding sign matches position (line 4477-4512)
5. **Cost Toggle**: If `cost_model.enabled=false`, all costs == 0.0 (line 4514-4555)
6. **No Ghost Trades**: Fills must have corresponding ledger entries

## Utilities
- `config/params_loader.py`: Loads `STRATEGY_PARAMS.json` with overrides
- `scripts/run_experiments.py`: Orchestrates backtest runs with parameter overrides
- `scripts/diagnose_edge.py`: Analyzes trade performance
- `scripts/evaluate_win_checklist.py`: Grades backtest against success criteria

