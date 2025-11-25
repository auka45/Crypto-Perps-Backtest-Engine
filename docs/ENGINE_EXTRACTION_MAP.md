# Engine Extraction Classification Map

**Generated:** 2025-11-23  
**Purpose:** Classify all repository files into extraction buckets for engine core separation

## Classification Buckets

- **CORE_ENGINE**: Core backtesting engine, portfolio, risk, execution, data loaders, params loader, metrics, technical indicators, liquidity regimes
- **VALIDATION_HARNESS**: Oracle module, toy markets, validation scripts/tests/docs
- **STRATEGY_SPECIFIC**: TREND/RANGE/SQUEEZE/NEUTRAL_PROBE modules, regime classifier, master_side logic, strategy params, experiment runners
- **MISC/LEGACY**: Old experiments, debug scripts, legacy docs, project explanations

---

## CORE_ENGINE

| File Path | Justification |
|-----------|---------------|
| `src/engine.py` | Main backtest engine orchestrator |
| `src/portfolio/state.py` | Portfolio state management (cash, positions, equity) |
| `src/portfolio/universe.py` | Universe management (symbol selection, refresh) |
| `src/portfolio/__init__.py` | Package init |
| `src/risk/es_guardrails.py` | Expected Shortfall (ES) risk guardrails |
| `src/risk/margin_guard.py` | Margin ratio checks and position trimming |
| `src/risk/loss_halts.py` | Daily loss halt logic |
| `src/risk/sizing.py` | Position sizing calculations |
| `src/risk/beta_controls.py` | Beta exposure controls |
| `src/risk/engine_state.py` | Engine state management |
| `src/risk/logging.py` | Risk event logging |
| `src/risk/__init__.py` | Package init |
| `src/execution/fill_model.py` | Fill price and slippage calculation |
| `src/execution/order_manager.py` | Order management (pending orders, OCO) |
| `src/execution/sequencing.py` | Event sequencing (stops, entries, trails) |
| `src/execution/constraints.py` | Order constraint validation |
| `src/execution/funding_windows.py` | Funding window checks |
| `src/execution/__init__.py` | Package init |
| `src/data/loader.py` | Data loading from CSV/Parquet |
| `src/data/schema.py` | Data schema validation |
| `src/data/__init__.py` | Package init |
| `src/indicators/technical.py` | Technical indicators (RSI, ADX, BB, ATR, etc.) |
| `src/indicators/avwap.py` | Anchored VWAP calculation |
| `src/indicators/helpers.py` | Indicator helper functions |
| `src/indicators/__init__.py` | Package init |
| `src/liquidity/regimes.py` | Liquidity regime detection (VACUUM, THIN, NORMAL) |
| `src/liquidity/seasonal.py` | Seasonal liquidity profiles |
| `src/liquidity/__init__.py` | Package init |
| `src/reporting.py` | Metrics generation and reporting |
| `src/forensics/deep_dive_run.py` | Forensic analysis (core analysis logic) |
| `src/forensics/report_generator.py` | Forensic report generation |
| `src/forensics/__init__.py` | Package init |
| `config/params_loader.py` | Parameter loading and override system |
| `config/__init__.py` | Package init |
| `src/__init__.py` | Package init |

---

## VALIDATION_HARNESS

| File Path | Justification |
|-----------|---------------|
| `src/modules/oracle.py` | Oracle module for validation/testing (bypasses filters) |
| `tests/fixtures/toy_markets.py` | Toy market data generators (UP, DOWN, CHOP, GAP_SHOCK) |
| `tests/fixtures/__init__.py` | Package init |
| `tests/fixtures/generate_fixture.py` | Fixture generation utilities |
| `tests/test_toy_oracles.py` | Oracle validation tests (Phase 3) |
| `tests/test_baselines_smoke.py` | Baseline smoke tests (Phase 4) |
| `tests/test_accounting_invariants_toy.py` | Accounting invariant tests (Phase 2) |
| `tests/test_cost_toggle_invariants.py` | Cost toggle invariant tests (Phase 2) |
| `tests/test_data_integrity_smoke.py` | Data integrity smoke tests (Phase 1) |
| `scripts/validate_data_integrity.py` | Data integrity validation script (Phase 1) |
| `scripts/run_baselines.py` | Baseline benchmark runner (Phase 4) |
| `scripts/export_signals.py` | Signal export for parity check (Phase 5) |
| `scripts/parity_replay.py` | Parity replay PnL calculation (Phase 5) |
| `scripts/check_parity.py` | Parity check script (Phase 5) |
| `docs/VALIDATION_PLAN.md` | Validation plan document |
| `docs/AUDIT_REPORT.md` | Audit report with evidence |
| `docs/REPO_MAP.md` | Repository map (call-site enumeration) |

---

## STRATEGY_SPECIFIC

| File Path | Justification |
|-----------|---------------|
| `src/modules/trend.py` | TREND module (strategy-specific entry/exit logic) |
| `src/modules/range.py` | RANGE module (strategy-specific counter-trend logic) |
| `src/modules/squeeze.py` | SQUEEZE module (strategy-specific Donchian breakout) |
| `src/modules/neutral_probe.py` | NEUTRAL_PROBE module (strategy-specific neutral regime logic) |
| `src/modules/__init__.py` | Package init (contains strategy modules) |
| `src/regimes/classifier.py` | Regime classifier (TREND/RANGE/UNCERTAIN) - strategy-specific logic |
| `src/regimes/master_side.py` | Master side classification (BULL/BEAR/NEUTRAL) - strategy-specific |
| `src/regimes/master_side_helper.py` | Master side helper functions - strategy-specific |
| `src/regimes/__init__.py` | Package init |
| `scripts/run_experiments.py` | Strategy experiment runner |
| `STRATEGY_PARAMS.json` | Strategy-specific parameter configuration |
| `config/variants/STRATEGY_PARAMS_BTCETH.json` | Strategy variant config |
| `config/variants/STRATEGY_PARAMS_trend_range_only.json` | Strategy variant config |
| `STRATEGY.md` | Strategy design document |
| `docs/STRATEGY_DESIGN_v1.md` | Strategy design v1 |
| `docs/STRATEGY_v1_eval_2023Q1.md` | Strategy evaluation |
| `docs/TREND_alignment_v1.md` | TREND module alignment doc |
| `docs/SQUEEZE_alignment_v1.md` | SQUEEZE module alignment doc |
| `docs/SQUEEZE_v1_exits_alignment.md` | SQUEEZE exits alignment |
| `docs/SQUEEZE_v1_range_calibration.md` | SQUEEZE range calibration |
| `docs/SQUEEZE_v1_range_eval.md` | SQUEEZE range evaluation |
| `tests/test_trend_toy.py` | TREND module toy tests |
| `tests/test_trend_unit.py` | TREND module unit tests |
| `tests/test_direction_gates.py` | Direction gating tests (strategy-specific) |
| `tests/test_gates.py` | Gate tests (strategy-specific) |
| `scripts/diagnose_edge.py` | Strategy edge diagnosis |
| `scripts/diagnose_signals.py` | Strategy signal diagnosis |
| `scripts/debug_trend_gating_real.py` | TREND gating debug script |
| `scripts/debug_master_side.py` | Master side debug script |
| `scripts/test_master_side.py` | Master side test script |

---

## MISC/LEGACY

| File Path | Justification |
|-----------|---------------|
| `main.py` | Legacy main entry point (may be strategy-specific) |
| `test_oracle_params.py` | Temporary test file |
| `experiments/` | Old experiment results (all subdirectories) |
| `experiments_1y/` | Old experiment results |
| `experiments_1y_fixed/` | Old experiment results |
| `experiments_1y_sanity/` | Old experiment results |
| `experiments_3m_fixed/` | Old experiment results |
| `experiments_3m_fixed_v2/` | Old experiment results |
| `experiments_debug/` | Debug experiment results |
| `experiments_debug_oracle/` | Oracle debug experiments |
| `experiments_debug_oracle_dummy/` | Oracle dummy experiments |
| `runs/` | Legacy run outputs (all subdirectories) |
| `artifacts/` | Generated artifacts (validation outputs OK, but old runs are legacy) |
| `data/` | Data files (keep for validation, but mark as external dependency) |
| `scripts/debug_oracle.py` | Debug script (temporary) |
| `scripts/deep_dive_run.py` | Deep dive analysis (may be strategy-specific) |
| `scripts/compare_backtest_runs.py` | Run comparison (may be strategy-specific) |
| `scripts/comprehensive_backtest_analysis.py` | Comprehensive analysis (may be strategy-specific) |
| `scripts/generate_professional_report.py` | Report generation (may be strategy-specific) |
| `scripts/generate_summary.py` | Summary generation (may be strategy-specific) |
| `scripts/analyze_reports.py` | Report analysis (may be strategy-specific) |
| `scripts/verify_all.py` | Verification script (may be strategy-specific) |
| `scripts/verify_analytics.py` | Analytics verification (may be strategy-specific) |
| `scripts/verify_fixes.py` | Fix verification (may be strategy-specific) |
| `scripts/run_gates.py` | Gate runner (may be strategy-specific) |
| `scripts/run_exp_with_invariants.py` | Experiment runner with invariants (validation use OK) |
| `scripts/sanity_checks.py` | Sanity checks (may be strategy-specific) |
| `scripts/reconcile_debug.py` | Reconciliation debug (may be strategy-specific) |
| `scripts/evaluate_win_checklist.py` | WIN checklist evaluation (strategy-specific) |
| `scripts/download_data.py` | Data download utility (external tool) |
| `scripts/enumerate_repo.py` | Repo enumeration utility |
| `scripts/check_symbol_dates.py` | Symbol date checker (utility) |
| `scripts/cleanup_artifacts.py` | Artifact cleanup utility |
| `scripts/run_suite.ps1` | PowerShell run suite (Windows-specific) |
| `scripts/__init__.py` | Package init |
| `tests/test_cost_toggle.py` | Cost toggle test (may be validation, needs review) |
| `tests/test_experiment_runner_smoke.py` | Experiment runner smoke test (may be validation) |
| `tests/test_metrics_sanity.py` | Metrics sanity test (may be validation) |
| `tests/test_params_loader.py` | Params loader test (core test) |
| `tests/test_risk_sanity.py` | Risk sanity test (may be validation) |
| `tests/test_indicators.py` | Indicator tests (core test) |
| `tests/test_reconciliation.py` | Reconciliation test (may be validation) |
| `tests/test_invariants.py` | Invariant tests (may be validation) |
| `tests/test_go_nogo.py` | Go/No-Go test (may be strategy-specific) |
| `tests/test_regression.py` | Regression test (may be strategy-specific) |
| `tests/test_trade_duration.py` | Trade duration test (may be strategy-specific) |
| `tests/__init__.py` | Package init |
| `docs/README.md` | Docs README |
| `docs/RISK_CONTROLS.md` | Risk controls documentation (core) |
| `docs/guides/HOW_TO_BACKTEST.md` | How-to guide (may be strategy-specific) |
| `docs/guides/OPERATOR_CHECKLIST.md` | Operator checklist (may be strategy-specific) |
| `docs/guides/PROJECT_GUIDE.md` | Project guide (legacy) |
| `docs/guides/QUICK_START.md` | Quick start guide (legacy) |
| `docs/guides/SETUP_GUIDE.md` | Setup guide (legacy) |
| `docs/specs/BACKTEST_SPEC.md` | Backtest spec (core) |
| `docs/specs/STRATEGY_LOGIC.md` | Strategy logic spec (strategy-specific) |
| `docs/specs/WIN_SPEC.json` | WIN spec (strategy-specific) |
| `docs/status/FINAL_INVARIANT_REPORT.md` | Status doc (legacy) |
| `docs/status/FINAL_VERIFICATION.md` | Status doc (legacy) |
| `docs/status/FIXES_APPLIED.md` | Status doc (legacy) |
| `docs/status/IMPLEMENTATION_COMPLETE.md` | Status doc (legacy) |
| `docs/status/INTEGRATION_CLEANUP_REPORT.md` | Status doc (legacy) |
| `docs/status/STRATEGY_ALIGNMENT_STATUS.md` | Status doc (strategy-specific) |
| `docs/status/SUCCESS_CRITERIA_VERIFICATION.md` | Status doc (legacy) |
| `docs/status/TEST_VALIDATION_REPORT.md` | Status doc (legacy) |
| `README.md` | Root README (project-level) |
| `CODEBASE_MAP.md` | Codebase map (legacy) |
| `DOCUMENTATION_ORGANIZATION.md` | Doc organization (legacy) |
| `FINAL_VERIFICATION_REPORT.md` | Final verification (legacy) |
| `SMOKE_TEST_ANALYSIS.md` | Smoke test analysis (legacy) |
| `PROJECT_EXPLANATION_SSO.md` | Project explanation (legacy) |
| `PROJECT_EXPLANATION_SSO_TR.md` | Project explanation (legacy) |
| `project_tree.txt` | Project tree (legacy) |
| `requirements.txt` | Python requirements (core) |
| `.gitignore` | Git ignore (core) |
| `signals_export.csv` | Temporary export file |
| `replay_metrics.json` | Temporary replay file |
| `debug_oracle_output/` | Debug output directory (temporary) |

---

## Files Needing Human Decision

| File Path | Classification | Notes |
|-----------|----------------|-------|
| `src/regimes/classifier.py` | STRATEGY_SPECIFIC or CORE_ENGINE? | Regime classification may be strategy-specific, but could be core if made generic |
| `src/regimes/master_side.py` | STRATEGY_SPECIFIC | Master side is strategy-specific, but engine uses it - needs abstraction |
| `src/regimes/master_side_helper.py` | STRATEGY_SPECIFIC | Helper for master side - strategy-specific |
| `tests/test_metrics_sanity.py` | VALIDATION_HARNESS or MISC? | Could be validation test |
| `tests/test_risk_sanity.py` | VALIDATION_HARNESS or MISC? | Could be validation test |
| `tests/test_reconciliation.py` | VALIDATION_HARNESS or MISC? | Could be validation test |
| `tests/test_invariants.py` | VALIDATION_HARNESS or MISC? | Could be validation test |
| `scripts/run_exp_with_invariants.py` | VALIDATION_HARNESS or MISC? | Used for validation but runs experiments |
| `docs/specs/BACKTEST_SPEC.md` | CORE_ENGINE or MISC? | Core spec but may have strategy details |

---

## Extraction Notes

1. **Engine Strategy Dependencies**: `src/engine.py` imports TREND, RANGE, SQUEEZE, NEUTRAL_PROBE modules. These must be removed and engine made strategy-agnostic.

2. **Regime/Master Side**: `src/regimes/classifier.py` and `src/regimes/master_side.py` are used by engine but contain strategy-specific logic. Consider making these pluggable or removing strategy-specific parts.

3. **Data Directory**: Keep `data/` for validation, but mark as external dependency in engine_core.

4. **Artifacts**: Keep validation artifacts (baselines, parity reports, data integrity reports) but remove old experiment artifacts.

5. **Tests**: Some tests are borderline - when in doubt, keep in validation harness for safety.

