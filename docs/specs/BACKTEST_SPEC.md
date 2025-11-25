# BACKTEST_SPEC.md — Data, Fills, Sequencing, Metrics, Tests

Spec version: PROD-2025-11-11-FINAL

## 1) Data Schema (Minimum Required)
**15‑minute bars (primary):**
- `ts` (UTC end‑time), `open`, `high`, `low`, `close`, `volume` (base), `notional` (quote volume).

**Higher TFs (derived from 15‑m or provided):**
- 1h, 4h, daily. Use **closed** bars only; daily indicators use prior day’s close across the current day.

**Microstructure / Liquidity:**
- `bid`, `ask` (or mid & spread to derive), `spread_bps` (if provided), `Depth5_bid_usd`, `Depth5_ask_usd` snapshots at 0.5 s cadence aggregated into 1‑min averages centered on the 15‑m bar boundary [t−30s,t+30s]. If not present, backtest must **error** when VACUUM/THIN logic is enabled.

**Funding & Contract:**
- `funding_ts`, `funding_rate` per symbol; apply **adverse** funding only.
- Contract filters: `tickSize`, `stepSize`, `minQty`, `minNotional` (symbol metadata).

**Open Interest:**
- `open_interest_usd` for universe governance.

**Seasonal profile (THIN):**
- Computed daily at 00:00 UTC on data ending D−7 using [D−37, D−7] window; frozen for D.

## 2) Costs & Slippage
- Fees: maker 2 bps; taker 4 bps (per notional).
- Base slippage (bps): `2 + 20 × participation_pct` where `participation_pct = order_notional / ADV_60m` and `ADV_60m = sum(notional of last 4 bars)`.
- Regime adder: THIN +15 bps; VACUUM blocks entries.
- Partial fill taker sweep allowed **next bar** if `signal_age ≤ 2` bars & ES headroom ≥ 0.5% & extra slip ≤ 8 bps.

## 3) Sequencing & Fill Model (Deterministic)
- Signals at bar t close; all orders simulated on bar t+1 using that bar’s OHLC.
- Stop‑run model:
  - `slip_bps_total = base_slip + regime_adder`
  - `slip_px = mid_bar × slip_bps_total / 10,000`
  - Buy fills: `min(High, trigger + slip_px)`; Sell fills: `max(Low, trigger − slip_px)`
  - Bound within [Low, High]. Apply to entries and stop‑loss exits.
- Per‑symbol event order on t+1:
  1) Stops first for non‑SQUEEZE (`adverse_first`)
  2) SQUEEZE (`entry_first`)
  3) New entries (strategy-specific modules in fixed order, if provided), applying β caps then ES cap
  4) Trails tighten (never widen)
  5) TTL/expiries processed
  6) Stale orders (>3 bars) cancelled

## 4) Risk Model
- Vol forecast: 15‑m returns → 30‑day MAD × 1.4826 × √96 → EWMA λ=0.995.
- `vol_fast_median`: median of the last 30 days of σ̂1d.
- Size multiplier `m = clamp(1/(1+vol_forecast/vol_fast_median), 0.3, 1.0)`; jump and momentum guards as in logic.
- ES guardrails:
  - EWHS ES @ 99%, λ=0.985 on 15‑m to 1‑d blocks.
  - Parametric ES @ 99% with EWMA vol λ=0.985, corr λ=0.999, ρ floors (BTC‑alt 0.90, cross‑alt 0.75) when `m ≤ 0.6`.
  - Sigma‑clip: `k = 4.5 − min(1.5, vol_forecast/vol_fast_median)`; `ES_sigma = k × σ_port_fast`.
  - Final: `ES_used = max(EWHS_ES, ES_stress, ES_sigma) ≤ 2.25%` post‑trade.
- β caps: net |Σ w_i β_slow,i| ≤ 1.0; gross Σ |w_i| |β_slow,i| ≤ 2.2 (β_slow EWMA‑OLS λ=0.999; shrinkage 10% prior).

## 5) Margin & Halts
- Cross‑margin ratio gates: block ≥60%; trim toward 50%; flatten ≥80%; deadlock ≥3 trims → flatten + `HALT_MANUAL`.
- Loss halts (scaled by vol): daily hard stop −2.5% × vol_scale; soft brake −1.5% × vol_scale for 6h; per‑symbol cap −0.8% × vol_scale; drawdown ladder −10% size×0.5, −20% size×0.1 + `HALT_MANUAL`.

**Counter Semantics (deduplicated):**
- `halt_daily_hard_count`: Count of **unique UTC days** when daily hard stop was triggered (across entire portfolio). Deduplicated by `(utc_date,)`.
- `halt_soft_brake_count`: Count of **unique UTC days** when soft brake was activated (across entire portfolio). Deduplicated by `(utc_date,)`.
- `per_symbol_loss_cap_count`: Count of **distinct (symbol, UTC day)** combinations when per-symbol loss cap was hit. Deduplicated by `(symbol, utc_date)`.
- All counters use deduplication sets to ensure no double-counting within the same UTC day (or symbol×UTC-day for per-symbol counters).

## 6) Universe Governance
- Weekly refresh; include OI ≥ $10M and ADV60 ≥ $50M (BTC/ETH always in). Drop after 3 consecutive failing days; re‑add after 14 days above; memecoin cap ≤1 slot.

## 7) Reports (Outputs)
- `/reports/trades.csv` — one row per fill: ts, symbol, side, module, qty, price, fees, slip_bps, stop_dist, ES_used_before/after, reason (ENTRY/EXIT/TRIM/STOP/TP/STALE_CANCEL/EXPIRE).
- `/reports/equity_curve.csv` — ts, equity, drawdown, daily_pnl, rolling_vol.
- `/reports/positions.csv` — ts, symbol, qty, entry_px, stop_px, trail_px, module, age_bars.
- `/reports/metrics.json` — summary metrics (see below).  
- `/reports/log_forensic.jsonl` — exact fields listed in strategy §10 (forensic log).

## 8) Metrics (Minimum)
- Total return, CAGR, MaxDD, MAR, Sharpe (daily), Sortino (daily), Calmar, PF, Win rate, Avg win/loss, Avg R, Exposure %, Turnover, Slippage bps realized, Fee bps, ES violations count (should be 0), Margin blocks count, VACUUM/THIN dwell %, Funding cost bps, Hit ratio per module, PnL per module, Avg trade duration.

## 9) Reproducibility
- Deterministic engine; no RNG. If a random seed is required for any library, fix to `42` and record in metrics.json.
- All parameters loaded from `base_params.json` (engine core) or strategy-specific overrides. Config snapshot saved to `/reports/params_used.json` on each run.
- Timezone: UTC; DST‑sensitive tests use UTC timestamps and must pass unchanged.

## 10) GO / NO‑GO Tests (must pass)
Map each of these to a unit/integration test:
1. Trim deadlock: after 3 trims → flatten + `HALT_MANUAL`; no auto‑resume.
2. Trim precedence: equal ES & notional → oldest first; tie → lexicographic.
3. Sigma‑clip: quiet → k≈4.5; jump shock → k decreases.
4. VACUUM dual exit: spread=28 bps + depth=0.05×max_notional → stay VACUUM; depth=0.25× → exit after 3 bars.
5. Latency log: order delays recorded; stale data > 2s → alert.
6. Rate‑limit: 429 → pause 60s → resume without data loss.
7. Margin guard: ≥60% blocks entries; trims to ≤50%.
8. Funding throttle: T−14m blocked; SQUEEZE disabled T−7.5m; T+10m still blocked.
9. AVWAP re‑anchor: drift 0.12% (adaptive) triggers; |AVWAP−EMA50|>5% triggers.
10. Stale order cancel: age>3 bars cancels `STALE_CANCEL`; OCO exits intact.
11. DST safety: 2023‑03‑12 and 2023‑11‑05 unchanged (UTC‑based windowing).
12. Vol‑adjust halts: vol_forecast=2×vol_fast_median → daily stop −5.0%.
13. Stop‑run fills: buy/sell bounds as specified (min/max of OHLC with slip).
14. Entry sequencing: Strategy-specific modules in fixed order (if provided), with ES headroom recomputed between.
15. max_possible_notional: if vol_forecast spikes 2× → max_notional drops → earlier VACUUM exit.

## 11) Implementation Checklist
- Enforce `tickSize`, `stepSize`, `minQty`, `minNotional` prior to simulated send.
- Use symbol metadata for filters and funding schedule.
- Compute `ADV_60m` as sum of last 4 bars notional (no look‑ahead).
- Persist `HALT_MANUAL` state in `strategy_state.json` in backtests where applicable.
