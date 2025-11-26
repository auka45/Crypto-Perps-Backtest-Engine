# Engine Core File Tree

Clean file structure of `engine_core/` for GitHub (excluding temporary files, artifacts, and `__pycache__`).

```
engine_core/
├── README.md
├── pyproject.toml
├── requirements.txt
├── VERIFICATION_REPORT.md
│
├── config/
│   ├── __init__.py
│   ├── base_params.json
│   ├── params_loader.py
│   └── example_overrides/
│       └── oracle_long.json
│
├── src/
│   ├── __init__.py
│   ├── engine.py                    # Main BacktestEngine orchestrator
│   ├── reporting.py                 # Metrics generation and artifacts
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py                # Data loading from CSV/Parquet
│   │   └── schema.py                 # Data schema validation
│   │
│   ├── portfolio/
│   │   ├── __init__.py
│   │   ├── state.py                  # Portfolio state (cash, positions, PnL)
│   │   └── universe.py               # Universe management
│   │
│   ├── risk/
│   │   ├── __init__.py
│   │   ├── es_guardrails.py          # Expected Shortfall guardrails
│   │   ├── margin_guard.py           # Margin ratio checks and trimming
│   │   ├── loss_halts.py             # Daily loss limits and kill-switch
│   │   ├── beta_controls.py          # Portfolio beta capping
│   │   ├── sizing.py                 # Position sizing calculations
│   │   ├── engine_state.py           # Engine state management
│   │   └── logging.py                # Risk event logging
│   │
│   ├── execution/
│   │   ├── __init__.py
│   │   ├── fill_model.py             # Fill price and slippage calculation
│   │   ├── order_manager.py          # Order management (pending, TTL, stale)
│   │   ├── sequencing.py             # Event sequencing (stops, entries, trails)
│   │   ├── constraints.py            # Order constraint validation
│   │   └── funding_windows.py        # Funding window checks
│   │
│   ├── indicators/
│   │   ├── __init__.py
│   │   ├── technical.py              # Technical indicators (RSI, ADX, BB, etc.)
│   │   ├── avwap.py                   # Anchored VWAP calculation
│   │   └── helpers.py                 # Indicator helper functions
│   │
│   ├── liquidity/
│   │   ├── __init__.py
│   │   ├── regimes.py                 # Liquidity regime detection (VACUUM/THIN/NORMAL)
│   │   └── seasonal.py                # Seasonal liquidity profiles
│   │
│   ├── modules/
│   │   ├── __init__.py
│   │   └── oracle.py                  # Oracle module (validation only)
│   │
│   ├── archive/
│   │   ├── __init__.py                 # Archive namespace (reference only)
│   │   ├── strategy_event_collectors.py # Strategy-specific event collectors (archived)
│   │   └── strategy_executors.py       # Strategy-specific executors (archived)
│   │
│   └── forensics/
│       ├── __init__.py
│       ├── deep_dive_run.py           # Forensic analysis logic
│       └── report_generator.py        # Forensic report generation
│
├── scripts/
│   ├── __init__.py
│   ├── run_example_oracle.py         # Minimal Oracle example
│   ├── run_baselines.py              # Baseline strategies (Buy & Hold, Flat, Random)
│   ├── validate_data_integrity.py    # Data integrity validation
│   ├── export_signals.py             # Signal export for parity check
│   ├── parity_replay.py               # Parity replay PnL calculation
│   └── check_parity.py                # Parity check script
│
├── tests/
│   ├── __init__.py
│   ├── fixtures/
│   │   ├── __init__.py
│   │   ├── toy_markets.py            # Synthetic market data generators
│   │   └── generate_fixture.py        # Fixture generation utilities
│   ├── test_toy_oracles.py           # Oracle validation tests (Phase 3)
│   ├── test_baselines_smoke.py       # Baseline smoke tests (Phase 4)
│   ├── test_accounting_invariants_toy.py  # Accounting invariant tests (Phase 2)
│   ├── test_cost_toggle_invariants.py     # Cost toggle invariant tests (Phase 2)
│   ├── test_data_integrity_smoke.py       # Data integrity smoke tests (Phase 1)
│   ├── test_params_loader.py              # Params loader tests
│   ├── test_indicators.py                  # Indicator tests
│   ├── test_risk_sanity.py                 # Risk sanity tests
│   ├── test_metrics_sanity.py              # Metrics sanity tests
│   ├── test_invariants.py                  # Invariant tests
│   └── test_reconciliation.py             # Reconciliation tests
│
└── docs/
    ├── ENGINE_OVERVIEW.md            # Comprehensive engine overview
    ├── AUDIT_REPORT.md                # Validation evidence
    ├── VALIDATION_PLAN.md             # Validation methodology
    ├── ENGINE_EXTRACTION_MAP.md      # Extraction classification
    ├── REPO_MAP.md                    # Repository map (call-site enumeration)
    ├── RISK_CONTROLS.md               # Risk controls documentation
    ├── FILE_TREE.md                   # This file
    └── specs/
        └── BACKTEST_SPEC.md           # Backtest specification
```

## Notes

- **Temporary files excluded**: `__pycache__/`, `artifacts/`, `runs/`, `*.egg-info/`, `test_debug/`, `test_run/`
- **Data directory**: Not included (external dependency)
- **Artifacts**: Generated during validation runs (not committed to repo)
- **Runs**: Temporary backtest outputs (not committed to repo)

