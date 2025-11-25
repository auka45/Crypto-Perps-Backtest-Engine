# Risk Controls Documentation

**Launch Punch List – Quick Win #3: config surfacing for new risk controls**

This document explains the risk controls implemented in the backtest engine, including all caps, thresholds, and halt mechanisms.

---

## Overview

The backtest engine implements multiple layers of risk controls to prevent excessive losses, deadlocks, and unauthorized risk-taking. All risk parameters are configurable via `base_params.json` (or strategy-specific overrides).

---

## 1. Trim Deadlock Safety

**Purpose:** Prevent infinite trim loops when margin constraints cannot be resolved.

**Parameters (`margin` section):**
- `max_trim_count` (default: 3): Maximum number of trim iterations before declaring deadlock
- `trim_flatten_on_fail` (default: true): Whether to flatten all positions if deadlock occurs

**Behavior:**
- When margin ratio exceeds `trim_target_ratio_pct` (50%), the engine attempts to trim positions
- Trimming closes entire positions one at a time, starting with largest notional
- After `max_trim_count` trims, if margin ratio still breaches:
  - If `trim_flatten_on_fail=true`: All positions are flattened and state is set to `RISK_HALT`
  - If `trim_flatten_on_fail=false`: Warning is logged but no automatic flatten

**Events:**
- `trim_fail`: Logged when trim deadlock occurs

---

## 2. Centralized ES + Margin Guardrails

**Purpose:** Single point of risk checking before any order is submitted.

**Parameters (`risk.limits` section):**
- `min_margin_ratio_pct` (default: 9.0): Minimum margin headroom required (as %)
- `max_es_pct` (default: 2.25): Maximum Expected Shortfall usage (as % of equity)
- `max_notional_usd` (default: null): Optional maximum notional cap

**Behavior:**
- Before any entry order, computes projected margin ratio and ES usage
- Blocks order if:
  - Projected margin ratio >= `block_new_entries_ratio_pct` (60%)
  - Projected ES usage > `max_es_pct` (2.25%)
  - Margin headroom < `min_margin_ratio_pct` (9.0%)

**Events:**
- `es_margin_block`: Logged when order blocked due to margin constraint
- `es_block`: Logged when order blocked due to ES constraint
- `margin_headroom_block`: Logged when order blocked due to insufficient margin headroom

---

## 3. Daily Loss Kill-Switch

**Purpose:** Hard stop on daily losses to prevent catastrophic days.

**Parameters (`risk.kill_switch` section):**
- `max_daily_loss_pct` (default: 4.0): Maximum daily loss as % of starting equity (vol-adjusted)
- `max_daily_loss_usd` (default: 2000.0): Maximum daily loss in USD (vol-adjusted)
- `flatten_on_trigger` (default: true): Whether to flatten all positions when triggered
- `block_new_entries` (default: true): Whether to block new entries when triggered
- `reset_time` (default: "00:00:00"): UTC time when daily PnL resets

**Behavior:**
- Tracks daily PnL (realized + unrealized) from UTC day start
- When either threshold is breached:
  - If `flatten_on_trigger=true`: All open positions are immediately closed
  - If `block_new_entries=true`: All new entry orders are blocked
  - Trading state is set to `RISK_HALT`
- State persists across restarts via `risk_state.json` file

**Events:**
- `daily_kill`: Logged when kill-switch is triggered (first time only)

**Recovery:**
- Kill-switch resets at UTC day boundary (00:00:00)
- Manual recovery: Delete `risk_state.json` file or wait for new UTC day

---

## 4. Portfolio BTC-Beta Caps

**Purpose:** Limit portfolio exposure to BTC price movements.

**Parameters (`risk.beta` section):**
- `max_symbol_beta` (default: 1.5): Maximum beta exposure per symbol (scaled by notional)
- `max_portfolio_beta` (default: 3.0): Maximum total portfolio beta exposure (scaled by total equity)
- `reference_symbol` (default: "BTCUSDT"): Reference symbol for beta calculation

**Behavior:**
- Before any entry order, computes:
  - Per-symbol beta exposure: `beta_symbol * notional_symbol * side_mult`
  - Portfolio total beta exposure: sum over all positions
- Blocks order if:
  - Symbol beta exposure > `max_symbol_beta * total_equity`
  - Portfolio beta exposure > `max_portfolio_beta * total_equity`

**Events:**
- `beta_block`: Logged when order blocked due to beta cap

---

## 5. Global Trading State Machine

**Purpose:** Centralized state management for trading halts.

**States:**
- `RUNNING`: Normal operation, all trading allowed
- `RISK_HALT`: Risk-based halt (kill-switch, trim deadlock, ES overrun)
- `TECH_HALT`: Technical/infrastructure halt (data gaps, API errors)
- `NEUTRAL_ONLY`: Only SQUEEZE/NEUTRAL_Probe modules allowed (if enabled)

**Parameters (`engine` section):**
- `allow_neutral_only_mode` (default: false): Whether to allow NEUTRAL_ONLY state
- `state_persistence_path` (default: "runs/{run_name}/engine_state.json"): Path to state file

**Behavior:**
- State is checked before every entry order
- Only `RUNNING` (and optionally `NEUTRAL_ONLY` with module filter) allows new entries
- State transitions are logged and persisted to file
- State persists across restarts

**Events:**
- `state_change`: Logged on every state transition

**Recovery:**
- Manual recovery: Delete `engine_state.json` file or modify state in file
- Automatic recovery: Some states may auto-recover (e.g., RISK_HALT → RUNNING at UTC day boundary)

---

## Event Triggers Summary

| Event | Trigger Condition | Action |
|-------|-------------------|--------|
| `trim_fail` | Trim loop reaches `max_trim_count` without resolving | Flatten (if enabled) + RISK_HALT |
| `es_margin_block` | Projected margin ratio >= block threshold | Block order |
| `es_block` | Projected ES usage > max ES | Block order |
| `margin_headroom_block` | Margin headroom < minimum | Block order |
| `daily_kill` | Daily loss breaches pct or USD threshold | Flatten (if enabled) + block entries + RISK_HALT |
| `beta_block` | Beta exposure exceeds symbol or portfolio cap | Block order |
| `state_change` | Trading state transitions | Log transition |

---

## How to Re-enable Trading After a Halt

### After Daily Kill-Switch

1. **Automatic:** Wait for new UTC day (00:00:00 UTC) - kill-switch resets automatically
2. **Manual:** Delete `runs/<run_name>/risk_state.json` file

### After Trim Deadlock

1. **Manual:** Delete `runs/<run_name>/engine_state.json` file
2. **Manual:** Set state to `RUNNING` in `engine_state.json`:
   ```json
   {
     "current_state": "RUNNING",
     "last_reason": "manual_reset",
     ...
   }
   ```

### After Any RISK_HALT

1. **Manual:** Delete or modify `runs/<run_name>/engine_state.json`:
   ```json
   {
     "current_state": "RUNNING",
     "last_reason": "manual_reset",
     ...
   }
   ```

---

## Parameter Reference

All risk parameters are in `base_params.json` (engine core) or strategy-specific overrides:

```json
{
  "margin": {
    "max_trim_count": 3,
    "trim_flatten_on_fail": true,
    ...
  },
  "risk": {
    "limits": {
      "min_margin_ratio_pct": 9.0,
      "max_es_pct": 2.25,
      "max_notional_usd": null
    },
    "kill_switch": {
      "max_daily_loss_pct": 4.0,
      "max_daily_loss_usd": 2000.0,
      "flatten_on_trigger": true,
      "block_new_entries": true,
      "reset_time": "00:00:00"
    },
    "beta": {
      "max_symbol_beta": 1.5,
      "max_portfolio_beta": 3.0,
      "reference_symbol": "BTCUSDT"
    }
  },
  "engine": {
    "allow_neutral_only_mode": false,
    "state_persistence_path": "runs/{run_name}/engine_state.json"
  }
}
```

---

## Testing

Run sanity checks to verify risk controls:

```bash
# Pytest
pytest tests/test_risk_sanity.py -v

# Standalone script
python scripts/sanity_checks.py
```

---

## Notes

- All risk controls are **config-driven** - no hard-coded thresholds
- State persistence allows recovery across restarts
- Structured logging makes debugging risk decisions easy
- All controls are **non-negotiable** for launch safety

