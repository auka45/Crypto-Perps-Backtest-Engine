# Integrating Your Strategy with the Engine

This guide explains how to use the `engine_core` backtesting engine with your own strategy logic.

## What the Engine Does

The engine provides:

1. **Data Loading**: Loads OHLCV and funding data from CSV/Parquet files
2. **Execution Modeling**: Realistic fill prices, slippage, and fees
3. **Risk Controls**: Expected Shortfall (ES) guardrails, margin checks, loss halts, beta caps
4. **Portfolio Management**: Position tracking, PnL calculation, cash management
5. **Reporting**: Comprehensive metrics, equity curves, trade logs, forensic analysis

## What the Engine Does NOT Do

The engine is **strategy-agnostic** (Model-1). It does NOT:

- Generate trading signals (no alpha, no regimes, no strategy logic)
- Implement TREND, RANGE, SQUEEZE, or NEUTRAL_PROBE modules
- Make trading decisions
- Define entry/exit rules beyond basic stops and TTL

All strategy logic must be implemented **outside** the engine.

## Minimal Interface

### 1. Instantiate BacktestEngine

```python
from engine_core.config.params_loader import ParamsLoader
from engine_core.src.engine import BacktestEngine
from engine_core.src.data.loader import DataLoader

# Load parameters
params = ParamsLoader('config/base_params.json')

# Create data loader
data_loader = DataLoader(data_path='../data/', symbols=['BTCUSDT', 'ETHUSDT'])

# Create engine
engine = BacktestEngine(
    data_loader=data_loader,
    params=params,
    require_liquidity_data=False,  # Set True if you have liquidity data
    stress_fees=False,
    stress_slip=False
)
```

### 2. Data Format Expected by DataLoader

The engine expects CSV files with the following structure:

**File naming**: `{SYMBOL}_{TIMEFRAME}.csv` (e.g., `BTCUSDT_15m.csv`)

**Required columns**:
- `ts`: Timestamp (datetime)
- `open`, `high`, `low`, `close`: OHLC prices (float)
- `volume`: Volume (float)
- `notional`: Notional volume in USD (float)

**Optional columns** (for advanced features):
- `funding_rate`: Funding rate (float)
- `atr`: Average True Range (float)
- `spread_bps`: Spread in basis points (float)
- `depth5_usd`: Depth at 5 bps (float)

See `docs/ENGINE_OVERVIEW.md` for complete data schema.

### 3. Plugging in Your Signal Generation

The engine uses the **Oracle pattern** for signal integration. You provide signals via the `add_signal()` method:

```python
# Your strategy logic (outside engine)
def generate_my_signals(df, current_bar_idx):
    """Your custom signal generation logic"""
    signals = []
    
    # Example: Simple moving average crossover
    if current_bar_idx > 20:
        sma_fast = df['close'].iloc[current_bar_idx-5:current_bar_idx+1].mean()
        sma_slow = df['close'].iloc[current_bar_idx-20:current_bar_idx+1].mean()
        
        if sma_fast > sma_slow:
            signals.append({
                'module': 'ORACLE',  # Must be 'ORACLE' for Model-1
                'side': 'LONG',
                'signal_ts': df.iloc[current_bar_idx]['ts'],
                'signal_bar_idx': current_bar_idx
            })
    
    return signals

# During backtest loop (pseudocode)
for bar_idx, bar in enumerate(data):
    # Your strategy generates signals
    signals = generate_my_signals(data, bar_idx)
    
    # Add signals to engine
    for signal in signals:
        engine.add_signal(
            symbol='BTCUSDT',
            signal=signal  # Must have module='ORACLE', side, signal_ts, signal_bar_idx
        )
    
    # Engine processes bar
    engine.process_bar(bar['ts'])
```

### 4. Complete Example

```python
"""
Minimal example: Simple long-only strategy
"""
import pandas as pd
from engine_core.config.params_loader import ParamsLoader
from engine_core.src.engine import BacktestEngine
from engine_core.src.data.loader import DataLoader

# Setup
params = ParamsLoader('config/base_params.json')
data_loader = DataLoader(data_path='../data/', symbols=['BTCUSDT'])
engine = BacktestEngine(data_loader=data_loader, params=params)

# Load data
data_loader.load_symbol('BTCUSDT', require_liquidity=False)
df = data_loader.get_symbol_data('BTCUSDT')

# Simple strategy: Buy on first bar, hold
for idx, row in df.iterrows():
    if idx == 0:
        # Generate signal
        engine.add_signal(
            symbol='BTCUSDT',
            signal={
                'module': 'ORACLE',
                'side': 'LONG',
                'signal_ts': row['ts'],
                'signal_bar_idx': idx
            }
        )
    
    # Process bar
    engine.process_bar(row['ts'])

# Generate report
report = engine.generate_report()
print(f"Total PnL: ${report['metrics']['total_pnl']:.2f}")
print(f"Total Trades: {report['metrics']['total_trades']}")
```

## Key Points

1. **Signals must have `module='ORACLE'`**: The engine only processes ORACLE signals in Model-1
2. **Signals are added before processing the bar**: Add signals, then call `process_bar()`
3. **Engine handles execution automatically**: You don't need to manage fills, stops, or exits (except stops via signal)
4. **All strategy logic is external**: The engine is a pure execution and risk management system

## Advanced: Custom Oracle Module

For more complex strategies, you can create a custom oracle module similar to `src/modules/oracle.py`:

```python
class MyStrategyOracle:
    """Your custom strategy oracle"""
    
    def generate_signals(self, df, current_idx):
        """Generate signals based on your strategy logic"""
        signals = []
        # Your logic here
        return signals
```

Then integrate it into your backtest loop.

## See Also

- `docs/ENGINE_OVERVIEW.md`: Complete engine architecture and capabilities
- `scripts/run_example_oracle.py`: Working example with toy data
- `src/modules/oracle.py`: Reference oracle implementation

