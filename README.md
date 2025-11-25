# Crypto Perps Backtesting Engine Core

**Status:** Validated and Extracted  
**Version:** v0 (engine_validated_v0)

A production-grade backtesting engine for Binance USDⓈ-M Futures with realistic execution modeling, risk controls, and comprehensive validation gates.

## What This Is

This is the **core backtesting engine** extracted from a larger strategy project. It provides:

- ✅ Multi-symbol backtesting (15m base timeframe)
- ✅ Realistic fill modeling (slippage, fees, funding)
- ✅ Risk controls (ES guardrails, margin checks, loss halts, beta caps)
- ✅ Liquidity diagnostics utilities (generic thresholds; optional for strategies)
- ✅ Comprehensive validation harness (oracle module, baselines, parity checks)
- ✅ Full accounting and metrics generation

## What This Is NOT

- ❌ Strategy logic (TREND, RANGE, SQUEEZE modules removed)
- ❌ Regime classification (strategy-specific logic removed)
- ❌ Master side logic (strategy-specific logic removed)

The engine is **strategy-agnostic** and only supports `oracle_mode` for validation/testing. For production strategies, provide signal generators externally.

## Installation

```bash
# Clone repository
git clone <repo-url>
cd engine_core

# Install dependencies
pip install -r requirements.txt

# Run validation tests
pytest tests/ -v
```

## Quick Start

### Example: Oracle Long Strategy

```bash
python scripts/run_example_oracle.py
```

This runs a minimal example using the Oracle module with `always_long` mode on toy UP market data.

### Run Baselines

```bash
python scripts/run_baselines.py --data-path ../data/ --start-date 2021-06-01 --end-date 2021-09-01
```

### Validate Data Integrity

```bash
python scripts/validate_data_integrity.py --data-path ../data/
```

## Validation Gates

The engine includes a comprehensive validation harness:

1. **Phase 1: Data Integrity** - Timestamp monotonicity, gaps, NaNs, OHLC sanity
2. **Phase 2: Accounting Invariants** - Equity identity, position conservation, PnL conservation
3. **Phase 3: Toy Oracles** - Deterministic signal validation on synthetic markets
4. **Phase 4: Baseline Benchmarks** - Buy & Hold, Flat, Random strategies
5. **Phase 5: Cross-Backtester Parity** - Independent PnL replay validation

See `docs/AUDIT_REPORT.md` for validation evidence.

## Architecture

```
engine_core/
├── src/              # Core engine components
│   ├── engine.py    # Main BacktestEngine orchestrator
│   ├── portfolio/    # Portfolio state management
│   ├── risk/         # Risk controls (ES, margin, loss halts, beta)
│   ├── execution/    # Fill model, order management, sequencing
│   ├── data/         # Data loading and schema
│   ├── indicators/   # Technical indicators
│   ├── liquidity/    # Liquidity diagnostics utilities
│   ├── reporting.py  # Metrics generation
│   └── modules/      # Oracle module (validation only)
├── config/           # Parameter system
│   ├── params_loader.py
│   ├── base_params.json
│   └── example_overrides/
├── scripts/          # Validation and utility scripts
├── tests/            # Validation tests
└── docs/             # Documentation
```

For detailed architecture and usage information, see `docs/ENGINE_OVERVIEW.md`.

## Adding a Strategy

The engine is strategy-agnostic. To add a strategy:

1. Create a strategy module that generates signals (similar to `src/modules/oracle.py`)
2. Integrate signal generation into your strategy runner
3. Pass signals to the engine via a callback or signal queue

**Note:** The engine currently only supports `oracle_mode` for validation. Full strategy integration requires engine modifications to accept external signal generators.

## Configuration

Engine parameters are defined in `config/base_params.json`. Key sections:

- `general`: Capital, fees, venue settings
- `universe`: Symbol selection and refresh rules
- `risk`: ES guardrails, margin, loss halts, beta controls
- `slippage_costs`: Fill model parameters
- `liquidity_regimes`: VACUUM/THIN detection thresholds

Use `ParamsLoader` to override parameters:

```python
from engine_core.config.params_loader import ParamsLoader

params = ParamsLoader(overrides={
    'general': {'oracle_mode': 'always_long'},
    'es_guardrails': {'es_cap_of_equity': 1.0}
}, strict=False)
```

## Running Validation

```bash
# Run all validation tests
pytest tests/ -v

# Run specific validation phase
pytest tests/test_toy_oracles.py -v          # Phase 3
pytest tests/test_baselines_smoke.py -v      # Phase 4
pytest tests/test_accounting_invariants_toy.py -v  # Phase 2

# Run validation scripts
python scripts/validate_data_integrity.py --data-path ../data/
python scripts/run_baselines.py --data-path ../data/ --start-date 2021-06-01 --end-date 2021-09-01
```

## Data Format

The engine expects 15m OHLCV data in CSV format:

```
data/
├── BTCUSDT_15m.csv
├── ETHUSDT_15m.csv
└── ...
```

CSV format: `timestamp,open,high,low,close,volume`

## License

MIT License (or as specified)

## Contributing

This is a standalone, strategy-agnostic backtesting engine. For strategy development, create your own strategy modules externally and integrate them with the engine.

## Validation Status

✅ **Data Integrity:** PASS  
✅ **Accounting Invariants:** PASS  
✅ **Toy Oracles:** PASS (7/7 tests)  
✅ **Baseline Benchmarks:** PASS  
✅ **Cross-Backtester Parity:** PASS

See `docs/AUDIT_REPORT.md` for detailed evidence.

