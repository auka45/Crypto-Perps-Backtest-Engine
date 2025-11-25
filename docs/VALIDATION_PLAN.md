# VALIDATION_PLAN.md

**Project:** Regime-Switching Hybrid Trading Strategy Backtester  
**Goal:** Establish whether negative PnL comes from (A) data issues, (B) engine/accounting bugs, (C) cost/execution bugs, or (D) strategy edge failure.  
**Principle:** **Do not change strategy logic until engine trust is proven.**  
**Trust Gates:** Each level must pass before moving on.

## ⚠️ CRITICAL PRINCIPLE

**Files existing ≠ validated. Only passing tests on real runs = validated.**

This plan requires **actual evidence** of passing tests, not just file existence. Each phase must produce:
- Executable commands that can be run locally
- Measurable acceptance criteria with exact tolerances
- Artifacts that prove the criteria were met
- Explicit "PASS" or "FAIL" verdicts based on test results

---

## Phase 0 — Repo Map & Validation Plan Lock

### Task 0.1: Repo scan
- Identify and list all execution/cost/metrics call-sites
- Output: `docs/REPO_MAP.md` with complete enumeration

### Task 0.2: Lock this validation doc
- This file is the contract for what "engine trusted" means
- Output: `docs/VALIDATION_PLAN.md` (this file)

**Acceptance Criteria:**
- ✅ `docs/REPO_MAP.md` exists and enumerates:
  - All fee calculation call-sites (entry, exit, margin flatten)
  - All slippage calculation call-sites
  - All funding cost application call-sites
  - All metrics aggregation call-sites
  - All invariant check call-sites
- ✅ `docs/VALIDATION_PLAN.md` includes strict acceptance criteria for all phases

**PASS Evidence:**
- File `docs/REPO_MAP.md` contains section "Cost Model Call-Sites" with line numbers
- File `docs/VALIDATION_PLAN.md` contains this section with measurable criteria

---

## Phase 1 — Level 0 Data Integrity (REAL DATA)

**Objective:** Confirm historical inputs are sane and aligned.  
**Deliverables:**  
- `scripts/validate_data_integrity.py`  
- `tests/test_data_integrity_smoke.py`  
- `artifacts/data_integrity_report.md`  
- `artifacts/data_integrity_flags.csv`

### Task 1.1: Data integrity script
Checks per symbol & interval:
1. **Timestamp monotonicity & duplicates**: strictly increasing, duplicates = 0
2. **Interval gap detection**: gaps > 1.5× expected bar duration flagged
3. **NaN / inf / missing OHLCV**: NaN rate computed per column
4. **OHLC sanity**: `low <= min(open, close) <= high`, `high >= max(open, close) >= low`
5. **Volume sanity**: volume >= 0, no inf
6. **Return outliers**: abs(15m return) > 25% flagged (log only, no deletion)
7. **Price scale consistency**: median price within reasonable bounds
8. **Timezone & candle alignment**: UTC normalized, aligned to grid (00:00, 00:15, 00:30...)

### Task 1.2: Smoke tests
- Inject toy issue (duplicate timestamp, NaN, OHLC violation)
- Assert script catches injected issue
- Verify script runs without crashing

### Task 1.3: Run validation on real data
- Execute: `python scripts/validate_data_integrity.py --data-path data/`
- Verify outputs are generated
- Check that flags are explainable (no silent corrections)

**Acceptance Criteria:**
- ✅ Script runs on `data/` without crashing
- ✅ `artifacts/data_integrity_report.md` exists and contains per-symbol summary
- ✅ `artifacts/data_integrity_flags.csv` exists and contains flagged rows (if any)
- ✅ Smoke test passes: `pytest -q tests/test_data_integrity_smoke.py` → **all tests pass**
- ✅ Real data run completes: duplicates = 0, NaN rate < 0.01% per OHLC column
- ✅ Gap count logged (may be >0, but must be surfaced in report)
- ✅ Outlier bars logged (not auto-fixed)

**PASS Evidence:**
```bash
# Command:
python scripts/validate_data_integrity.py --data-path data/

# Expected output:
# "Data integrity check complete. Report: artifacts/data_integrity_report.md"
# "Flags: artifacts/data_integrity_flags.csv"

# Verify:
# 1. artifacts/data_integrity_report.md exists
# 2. Report shows: duplicates=0, NaN_rate < 0.01% for all OHLC columns
# 3. artifacts/data_integrity_flags.csv exists (may be empty if no flags)

# Smoke test:
pytest -q tests/test_data_integrity_smoke.py
# Expected: "X passed" (all tests pass)
```

**Go/No-Go:** If duplicates/NaNs exceed threshold → fix data pipeline before Phase 2.

---

## Phase 2 — Level 1 Accounting Invariants (DEBUG MODE)

**Objective:** Prove cash/position/PnL bookkeeping is mathematically consistent.  
**Deliverables:**  
- `general.debug_invariants` param in `STRATEGY_PARAMS.json`  
- Invariant checks in `src/engine.py` (active only if debug)  
- `tests/test_accounting_invariants_toy.py`  
- `tests/test_cost_toggle_invariants.py`

### Task 2.1: Add debug toggle
- Add `general.debug_invariants: false` default to `STRATEGY_PARAMS.json`

### Task 2.2: Implement invariants (checked every bar when debug)
1. **Equity identity**: `equity = cash + unrealized_pnl` (NOT position_notional for futures)
2. **Position conservation**: `pos_t = pos_{t-1} + fills_t`
3. **Realized PnL conservation**: realized PnL changes only on fills/closures
4. **Cost signs**: fees <= 0, slippage <= 0, funding sign matches position sign
5. **Cost toggle invariants**: if `cost_model.enabled=false`, all costs == 0.0
6. **No ghost trades**: if no order → no fill → no pnl impact

### Task 2.3: Tests on toy fills
- Construct minimal sequences of fills and prices
- Verify invariants with tolerance

### Task 2.4: Run real experiment with invariants enabled
- Run Exp 1 (or small subset) with `general.debug_invariants=true`
- Verify no invariant violations occur
- If violations occur, fix engine bugs before proceeding

**Acceptance Criteria:**
- ✅ `pytest -q tests/test_accounting_invariants_toy.py` → **all tests pass**
- ✅ `pytest -q tests/test_cost_toggle_invariants.py` → **all tests pass**
- ✅ Real Exp1 run with `debug_invariants=true` completes without `AssertionError`
- ✅ All 6 invariants are checked (equity identity, position conservation, realized PnL, cost signs, cost toggle, no ghost trades)
- ✅ Tolerance: `abs(equity_error) < max(1e-6 * equity, $0.01)`

**PASS Evidence:**
```bash
# Unit tests:
pytest -q tests/test_accounting_invariants_toy.py
# Expected: "X passed" (all tests pass)

pytest -q tests/test_cost_toggle_invariants.py
# Expected: "X passed" (all tests pass)

# Real run with invariants:
python scripts/run_experiments.py \
  --data-path data/ \
  --start-date 2021-06-01 \
  --end-date 2021-07-01 \
  --experiments 1 \
  --output-dir experiments_invariant_test \
  --override '{"general": {"debug_invariants": true}}'

# Expected: No AssertionError, run completes successfully
# Verify: No "INVARIANT_VIOLATION" entries in forensic_log
```

**Go/No-Go:** Any invariant failure → engine not trusted; fix before Phase 3.

---

## Phase 3 — Level 2 Toy Oracle Markets (TRUE ORACLES)

**Objective:** Validate PnL directionality on synthetic markets using oracle strategies that bypass regime/risk gating.  
**Deliverables:**  
- `src/modules/oracle.py` (ORACLE module)  
- `tests/fixtures/toy_markets.py`  
- `tests/test_toy_oracles.py`

### Task 3.1: Create ORACLE module
- Define `OracleSignal` dataclass (similar to TrendSignal)
- Implement `OracleModule` class with:
  - `generate_always_long(symbol, df, idx, current_ts)` → OracleSignal
  - `generate_always_short(symbol, df, idx, current_ts)` → OracleSignal
  - `generate_flat()` → None (no signals)
- Signals must have: symbol, side, entry_price, stop_price, signal_bar_idx, signal_ts, module='ORACLE'

### Task 3.2: Integrate ORACLE module into engine
- Add `self.oracle_module = OracleModule(self.params_dict)` in `BacktestEngine.__init__`
- Modify `generate_signals()` to check for `general.oracle_mode` parameter
- If `oracle_mode == 'always_long'`: call `oracle_module.generate_always_long()` and append to `symbol_pending_signals`
- If `oracle_mode == 'always_short'`: call `oracle_module.generate_always_short()`
- If `oracle_mode == 'flat'`: do nothing
- Oracle signals bypass all regime/master_side/RSI/volume filters

### Task 3.3: Update toy market fixtures
- Verify `tests/fixtures/toy_markets.py` generates UP, DOWN, CHOP, GAP_SHOCK correctly
- Ensure markets have sufficient bars for oracle tests

### Task 3.4: Rewrite oracle tests with true oracle module
- Always-Long on UP market: assert pnl > 0 (with costs OFF)
- Always-Short on DOWN market: assert pnl > 0 (with costs OFF)
- Always-Long on DOWN market: assert pnl < 0 (sanity check)
- Always-Short on UP market: assert pnl < 0 (sanity check)
- Flat: assert 0 trades and pnl == 0
- CHOP with costs OFF: assert |pnl| < 1% of initial capital
- Costs ON vs OFF: assert costs reduce pnl magnitude (monotonic)

**Acceptance Criteria:**
- ✅ `pytest -q tests/test_toy_oracles.py` → **all tests pass**
- ✅ Oracle signals traverse full engine path: signal → order → fill → costs → metrics
- ✅ PnL directionality matches expectations:
  - Always-Long on UP: `pnl > 0` (with costs OFF)
  - Always-Short on DOWN: `pnl > 0` (with costs OFF)
  - Always-Long on DOWN: `pnl < 0` (sanity)
  - Always-Short on UP: `pnl < 0` (sanity)
  - Flat: `trades == 0` and `pnl == 0`
  - CHOP with costs OFF: `abs(pnl) < 0.01 * initial_capital`
  - Costs ON: `abs(pnl_ON) < abs(pnl_OFF)` (monotonic)

**PASS Evidence:**
```bash
# Run oracle tests:
pytest -q tests/test_toy_oracles.py -v

# Expected output:
# test_always_long_on_up_market PASSED
# test_always_short_on_down_market PASSED
# test_always_long_on_down_market PASSED (pnl < 0)
# test_always_short_on_up_market PASSED (pnl < 0)
# test_flat_oracle PASSED (0 trades, pnl == 0)
# test_chop_with_costs_off PASSED (|pnl| < 1% initial capital)
# test_costs_reduce_pnl PASSED (|pnl_ON| < |pnl_OFF|)

# All tests pass
```

**Go/No-Go:** If oracles fail → execution/cost/metrics path broken.

---

## Phase 4 — Level 3 Baseline Benchmarks (TRUE BASELINES)

**Objective:** Ensure engine matches trivial real-market expectations through the *same* execution path.  
**Deliverables:**  
- `scripts/run_baselines.py` (using ORACLE module)  
- `tests/test_baselines_smoke.py`  
- `artifacts/baselines_summary.csv`

### Task 4.1: Implement true baseline strategies using ORACLE module
- **Buy & Hold**: Use `oracle_mode='always_long'` with entry on first bar, exit on last bar
- **Flat**: Use `oracle_mode='flat'` (no signals)
- **Random**: Create `oracle_mode='random'` with seeded random entries/exits

### Task 4.2: Update baseline scripts
- Replace simplified implementations with true oracle-based baselines
- Ensure baselines write `artifacts/baselines_summary.csv`
- Write per-baseline `metrics.json` and `trades.csv` to subdirectories

### Task 4.3: Update smoke tests
- Buy & Hold: Calculate naive return = `(last_close / first_close) - 1`
- Assert Buy & Hold return within 1-2 bps of naive return
- Flat: Assert 0 trades, pnl == 0
- Random: Assert non-crashing, reasonable trade count

### Task 4.4: Run baselines on real data
- Execute: `python scripts/run_baselines.py --data-path data/ --start-date 2021-06-01 --end-date 2021-09-01`
- Verify `artifacts/baselines_summary.csv` is generated
- Verify Buy & Hold return matches naive return within tolerance

**Acceptance Criteria:**
- ✅ `pytest -q tests/test_baselines_smoke.py` → **all tests pass**
- ✅ Baselines run on real data without errors
- ✅ Buy & Hold return within 1-2 bps of naive return: `abs(buyhold_return - naive_return) < 0.0002`
- ✅ Flat produces 0 trades and 0 PnL: `trades == 0` and `abs(pnl) < $0.01`
- ✅ Random: non-crashing, `trades > 0`

**PASS Evidence:**
```bash
# Run baselines:
python scripts/run_baselines.py \
  --data-path data/ \
  --start-date 2021-06-01 \
  --end-date 2021-09-01

# Expected output:
# "Baselines complete. Summary: artifacts/baselines_summary.csv"

# Verify:
# 1. artifacts/baselines_summary.csv exists
# 2. Buy & Hold return within 1-2 bps of naive return
# 3. Flat: trades=0, pnl=0

# Smoke test:
pytest -q tests/test_baselines_smoke.py
# Expected: "X passed" (all tests pass)
```

**Go/No-Go:** If baselines fail → engine not trusted.

---

## Phase 5 — Level 4 Cross-Backtester Parity (MUST PASS)

**Objective:** Cross-validate PnL calculation via independent replay.  
**Deliverables:**  
- `scripts/export_signals.py`  
- `scripts/parity_replay.py`  
- `scripts/check_parity.py`  
- `artifacts/parity_report.md`

### Task 5.1: Verify/fix parity scripts
- `scripts/export_signals.py`: Ensure it exports from `fills.csv` correctly
- `scripts/parity_replay.py`: Verify PnL calculation matches engine logic
- `scripts/check_parity.py`: Ensure tolerances are strict (1-2 bps, not %-level)

### Task 5.2: Run parity check on real experiment
- Export signals from `experiments_1y_fixed/exp_1_Raw_Edge_Costs_OFF/artifacts`
- Replay PnL using `parity_replay.py`
- Compare with `metrics.json` using `check_parity.py`
- Verify parity passes (trade count identical, PnL diff < 1-2 bps)

### Task 5.3: Fix any parity failures
- If parity fails, identify root cause (fill price calculation, cost application, etc.)
- Fix bugs in engine or replay logic
- Re-run until parity passes

**Acceptance Criteria:**
- ✅ Parity check passes on Exp 1 (or chosen experiment)
- ✅ Trade count: identical (0 diff)
- ✅ Per-trade PnL diff: < 1 bps notional (`abs(diff) < 0.0001 * notional`)
- ✅ Total PnL diff: < 5 bps equity (`abs(diff) < 0.0005 * equity`)
- ✅ Total costs diff: < $0.01 absolute
- ✅ MaxDD/CAGR within epsilon (reported in parity_report.md)

**PASS Evidence:**
```bash
# Export signals:
python scripts/export_signals.py \
  --artifacts-dir experiments_1y_fixed/exp_1_Raw_Edge_Costs_OFF/artifacts \
  --output signals_export.csv

# Replay PnL:
python scripts/parity_replay.py \
  --signals signals_export.csv \
  --output replay_metrics.json

# Check parity:
python scripts/check_parity.py \
  --engine-metrics experiments_1y_fixed/exp_1_Raw_Edge_Costs_OFF/artifacts/metrics.json \
  --replay-metrics replay_metrics.json \
  --output artifacts/parity_report.md

# Expected output in parity_report.md:
# "Trade count: IDENTICAL (0 diff)"
# "Per-trade PnL diff: PASS (< 1 bps notional)"
# "Total PnL diff: PASS (< 5 bps equity)"
# "Total costs diff: PASS (< $0.01)"
# "VERDICT: PASS"
```

**Go/No-Go:** Parity fail means bug; stop and fix.

---

## Phase 6 — Consolidated Audit Report

**Objective:** Produce a clear trust verdict with actual evidence.  
**Deliverable:** `docs/AUDIT_REPORT.md`

### Task 6.1: Rewrite audit report
- Remove placeholder "✅ PASS" claims
- Add actual command outputs showing passes
- Include explicit Yes/No verdicts for:
  - Data integrity: Yes/No (with evidence)
  - Accounting correctness: Yes/No (with evidence)
  - Execution/cost model correctness: Yes/No (with evidence)
  - Parity correctness: Yes/No (with evidence)
- Only set "trusted" if all phases pass to acceptance

### Task 6.2: Create run checklist
- Document exact commands to run for each phase
- Specify what artifacts to inspect
- Define what "PASS" looks like for each phase

**Acceptance Criteria:**
- ✅ Audit report includes actual evidence (command outputs, file paths, metrics)
- ✅ Trust verdict is based on real test results, not assumptions
- ✅ Run checklist is complete and executable
- ✅ 3-axis trust verdict:
  1. Data trusted? (Yes/No with evidence)
  2. Accounting trusted? (Yes/No with evidence)
  3. Execution/cost model trusted? (Yes/No with evidence)

**PASS Evidence:**
```bash
# Audit report must contain:
# 1. Phase pass/fail table with evidence links
# 2. Command outputs showing actual test results
# 3. File paths to artifacts proving each phase passed
# 4. Explicit Yes/No verdicts for each trust axis
# 5. "Run These Commands" section with exact commands
```

**Go/No-Go:** If any axis = No → do not proceed to strategy modifications.

---

## Phase 7 — Strategy-Level Work (Only if Phases 1–6 pass)

**Objective:** Now treat losses as strategy edge issues.  
**Deliverables:**  
- Updated experiments  
- `artifacts/edge_diagnosis.md`  
- A/B regression tests for any change

### Tasks
1. Re-run stress & recovery regimes (multi-year)
2. Decompose with `diagnose_edge.py`
3. Propose minimal changes tied to diagnosis
4. **Ship regression tests** proving each change addresses the bleed

**Acceptance Criteria:**
- ✅ Every change has a before/after experiment + regression test
- ✅ Regression tests prove each change addresses the identified issue

---

## Execution Order (Strict)

1. Phase 0: Update plan doc
2. Phase 1: Data integrity (run on real data)
3. Phase 2: Invariants (fix implementation, test on real run)
4. Phase 3: Oracles (create ORACLE module, true tests)
5. Phase 4: Baselines (use ORACLE module, true baselines)
6. Phase 5: Parity (run on real experiment, must pass)
7. Phase 6: Audit report (with evidence)
8. Phase 7: Strategy work (conditional)

**Rule:** At each Go/No-Go gate, stop, summarize, and fix before moving on. **Do not proceed until acceptance criteria are met with real evidence.**
