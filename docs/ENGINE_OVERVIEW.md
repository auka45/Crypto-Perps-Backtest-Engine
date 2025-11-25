# Engine Core Overview

## What is this engine?

This is a **production-grade backtesting engine** for cryptocurrency perpetual futures trading on Binance USDⓈ-M Futures. The engine provides a realistic simulation environment that models the key aspects of futures trading: order execution with slippage, trading fees (maker/taker), funding costs, risk controls, and comprehensive accounting.

The engine is **strategy-agnostic**—it does not contain any trading strategy logic. It focuses solely on the mechanics of backtesting: loading market data, processing signals, executing orders, applying costs, managing risk, and generating performance metrics. Strategy logic (entry/exit signals, regime classification, position sizing rules) must be provided externally.

**What it models:**
- Multi-symbol backtesting (15-minute base timeframe)
- Realistic fill modeling (slippage based on participation rate, stop-run fills)
- Trading costs (maker/taker fees, slippage, funding costs)
- Risk controls (Expected Shortfall guardrails, margin checks, loss halts, beta caps)
- Portfolio state management (cash, positions, unrealized/realized PnL)
- Comprehensive metrics generation (returns, Sharpe ratio, drawdown, trade statistics)

**What it does NOT include:**
- Trading strategy modules (TREND, RANGE, SQUEEZE, NEUTRAL_PROBE)
- Regime classification logic
- Master side computation
- Strategy-specific signal generation
- Hard-coded entry/exit rules

The engine only supports `oracle_mode` for validation and testing. For production strategies, users must provide their own signal generators and integrate them with the engine.

---

## Architecture Overview

### `engine_core/src/engine.py` — Main Orchestrator

The `BacktestEngine` class orchestrates the entire backtest run. It manages the event loop, coordinates between components, and handles signal generation (via Oracle module in validation mode). The engine processes bars in two phases:

- **Bar t (signal generation)**: Generates signals, updates indicators, checks risk constraints
- **Bar t+1 (execution)**: Executes orders, applies fills, updates portfolio state, applies funding costs

Key methods:
- `run()`: Main entry point that executes the backtest
- `process_bar_t()`: Signal generation phase
- `process_bar_t_plus_1()`: Order execution phase
- `generate_signals()`: Generates Oracle signals (validation only)

### `portfolio/` — Portfolio State Management

- **`state.py`**: Manages cash, positions, equity, and PnL calculations. Tracks realized/unrealized PnL, applies funding costs, and maintains position state.
- **`universe.py`**: Manages symbol selection and universe refresh logic.

### `risk/` — Risk Controls

- **`es_guardrails.py`**: Expected Shortfall (ES) calculation and guardrails to limit portfolio risk.
- **`margin_guard.py`**: Margin ratio checks and position trimming when margin constraints are violated.
- **`loss_halts.py`**: Daily loss limits and kill-switch logic to halt trading after large losses.
- **`beta_controls.py`**: Portfolio beta capping to limit directional exposure.
- **`sizing.py`**: Position sizing calculations based on volatility and risk parameters.
- **`engine_state.py`**: Engine state management (NORMAL, HALTED, NEUTRAL_ONLY).

### `execution/` — Order Execution & Fill Modeling

- **`fill_model.py`**: Calculates fill prices using stop-run model, computes slippage based on participation rate and liquidity regime.
- **`order_manager.py`**: Manages pending orders, handles stale order cancellation, TTL expiration.
- **`sequencing.py`**: Event sequencing logic (stops first, then entries, then trails).
- **`constraints.py`**: Order validation (min quantity, tick size, max position size).
- **`funding_windows.py`**: Checks for funding rate windows to block/close trades near funding times.

### `data/` — Data Loading & Validation

- **`loader.py`**: Loads OHLCV data from CSV/Parquet files, supports date filtering, loads liquidity/funding/contract metadata.
- **`schema.py`**: Validates dataframes against expected schemas.

### `indicators/` — Technical Indicators

- **`technical.py`**: Generic technical indicators (RSI, ADX, Bollinger Bands, ATR, MACD, etc.).
- **`avwap.py`**: Anchored VWAP calculation.
- **`helpers.py`**: Helper indicator functions.

### `liquidity/` — Liquidity Diagnostics

- **`regimes.py`**: Liquidity regime detection (VACUUM, THIN, NORMAL) based on volume thresholds.
- **`seasonal.py`**: Seasonal liquidity profiles for time-of-day adjustments.

### `modules/oracle.py` — Validation Module

The Oracle module generates deterministic signals for validation and testing. It supports:
- `always_long`: Enters long on first bar, exits on last bar
- `always_short`: Enters short on first bar, exits on last bar
- `flat`: No signals
- `random`: Random entries/exits (seeded for reproducibility)

**Note**: The Oracle module is the only signal generator included in the engine. Production strategies must provide their own signal generation logic.

### `reporting.py` — Metrics & Artifacts

Generates comprehensive performance metrics and artifacts:
- **Metrics**: Total return, CAGR, Sharpe ratio, Sortino ratio, Calmar ratio, max drawdown, win rate, profit factor, etc.
- **Artifacts**: `fills.csv` (all fills with costs), `trades.csv` (round-trip trades), `ledger.csv` (cash movements), `equity.csv` (equity curve).

---

## Data Expectations

### File Structure

The engine expects data files in the following structure:

```
data/
├── BTCUSDT_15m.csv
├── ETHUSDT_15m.csv
├── SOLUSDT_15m.csv
└── ...
```

### CSV Format

Each CSV file should contain 15-minute OHLCV bars with the following columns:

- `ts`: Timestamp (UTC, timezone-aware pandas Timestamp)
- `open`: Open price
- `high`: High price
- `low`: Low price
- `close`: Close price
- `volume`: Volume (base currency)

**Example:**
```csv
ts,open,high,low,close,volume
2021-06-01 00:00:00+00:00,36750.50,36800.00,36700.00,36750.00,1250.5
2021-06-01 00:15:00+00:00,36750.00,36850.00,36720.00,36800.00,1320.2
```

### Timestamp Assumptions

- Timestamps must be UTC and timezone-aware
- Bars must be aligned to 15-minute grid (00:00, 00:15, 00:30, 00:45, etc.)
- Timestamps must be strictly monotonic (no duplicates, no gaps)
- Data should be validated using `scripts/validate_data_integrity.py` before backtesting

### Optional Data

The engine can also load:
- **Liquidity data**: `{symbol}_liquidity.csv` (for liquidity regime detection)
- **Funding data**: `{symbol}_funding.csv` (for funding cost calculation)
- **Contract metadata**: `{symbol}_contract.json` (for contract specifications)

---

## How to Use It

### Installation

```bash
# Clone repository
git clone <repo-url>
cd engine_core

# Install dependencies
pip install -r requirements.txt

# Install engine in editable mode
pip install -e .
```

### Quick Start: Oracle Example

The simplest way to test the engine is using the Oracle module:

```bash
python scripts/run_example_oracle.py
```

This runs a minimal example using the Oracle module with `always_long` mode on synthetic UP market data.

### Running Baselines

Run baseline strategies (Buy & Hold, Flat, Random) on real data:

```bash
python scripts/run_baselines.py \
  --data-path ../data/ \
  --start-date 2021-06-01 \
  --end-date 2021-09-01
```

### Validating Data

Before running backtests, validate your data:

```bash
python scripts/validate_data_integrity.py --data-path ../data/
```

### Integrating Your Own Strategy

To integrate a custom strategy:

1. **Prepare your data** in the expected format (see Data Expectations above).

2. **Create a signal generator** that produces signals compatible with the engine. Signals should include:
   - Symbol
   - Side (LONG/SHORT)
   - Entry price
   - Stop price
   - Signal timestamp

3. **Integrate with the engine**:
   - Modify `generate_signals()` in `engine.py` to call your signal generator, OR
   - Create a custom module similar to `modules/oracle.py` and integrate it into the engine

4. **Configure parameters** via `config/base_params.json` or use `ParamsLoader` with overrides.

5. **Run the backtest**:
   ```python
   from engine_core.src.engine import BacktestEngine
   from engine_core.src.data.loader import DataLoader
   from engine_core.config.params_loader import ParamsLoader
   
   data_loader = DataLoader('data/', start_ts=..., end_ts=...)
   params = ParamsLoader(overrides={...})
   engine = BacktestEngine(data_loader, params)
   engine.run()
   metrics = engine.get_metrics()
   ```

6. **Access results**:
   - Metrics: `engine.get_metrics()`
   - Artifacts: `engine.report_generator.write_artifacts(output_dir)`

---

## What is NOT in this Engine

The following are **explicitly excluded** from the engine core:

- ❌ **TREND module**: Trend-following strategy logic
- ❌ **RANGE module**: Mean-reversion strategy logic
- ❌ **SQUEEZE module**: Volatility breakout strategy logic
- ❌ **NEUTRAL_PROBE module**: Neutral regime strategy logic
- ❌ **Regime classifier**: Market regime classification (TREND/RANGE/UNCERTAIN)
- ❌ **Master side logic**: Master bias computation (BULL/BEAR/NEUTRAL)
- ❌ **Strategy-specific filters**: Entry/exit filters based on regime or master side
- ❌ **Hard-coded strategy rules**: Any trading rules beyond risk controls

**Note**: The engine contains **dead code** (event handlers for strategy modules) that is never executed in oracle mode. This code is left in place for potential future strategy integration but is effectively inactive.

The engine is designed to be a **generic backtesting framework**. Strategy logic should live in a separate package or be provided by the user.

---

## Configuration

Engine parameters are defined in `config/base_params.json`. Key sections:

- `general`: Capital, fees, venue settings, oracle mode
- `universe`: Symbol selection and refresh rules
- `risk`: ES guardrails, margin, loss halts, beta controls
- `slippage_costs`: Fill model parameters
- `liquidity_regimes`: VACUUM/THIN detection thresholds

Use `ParamsLoader` to override parameters:

```python
from engine_core.config.params_loader import ParamsLoader

params = ParamsLoader(
    overrides={
        'general': {'oracle_mode': 'always_long'},
        'risk': {'es_cap_of_equity': 1.0}
    },
    strict=False
)
```

---

## Validation & Testing

The engine includes a comprehensive validation harness:

1. **Data Integrity** (`scripts/validate_data_integrity.py`): Validates OHLCV data for timestamps, gaps, NaNs, sanity checks
2. **Accounting Invariants** (`tests/test_invariants.py`): Verifies cash/position/PnL bookkeeping is mathematically consistent
3. **Toy Oracles** (`tests/test_toy_oracles.py`): Validates PnL directionality on synthetic markets
4. **Baseline Benchmarks** (`scripts/run_baselines.py`): Ensures engine matches trivial real-market expectations
5. **Cross-Backtester Parity** (`scripts/check_parity.py`): Cross-validates PnL calculation via independent replay

Run all tests:

```bash
pytest tests/ -v
```

See `docs/AUDIT_REPORT.md` for validation evidence.

---

## License

MIT License (or as specified)

---

## Contributing

This is an extracted engine core. For strategy development, see `../strategy_v2026/`.

