"""Unit tests for indicators"""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from engine_core.src.indicators.technical import ema, sma, rsi, atr, macd, bollinger_bands, adx, compute_all_indicators


def test_ema():
    """Test EMA calculation"""
    series = pd.Series([100, 101, 102, 103, 104])
    ema_result = ema(series, period=3)
    
    assert len(ema_result) == len(series)
    assert not pd.isna(ema_result.iloc[-1])


def test_sma():
    """Test SMA calculation"""
    series = pd.Series([100, 101, 102, 103, 104])
    sma_result = sma(series, period=3)
    
    assert len(sma_result) == len(series)
    # Last value should be average of last 3
    assert abs(sma_result.iloc[-1] - 103.0) < 0.1


def test_rsi():
    """Test RSI calculation"""
    series = pd.Series([100, 102, 101, 103, 105, 104, 106])
    rsi_result = rsi(series, period=3)
    
    assert len(rsi_result) == len(series)
    # RSI should be between 0 and 100
    assert 0 <= rsi_result.iloc[-1] <= 100


def test_atr():
    """Test ATR calculation"""
    high = pd.Series([102, 103, 104, 105, 106])
    low = pd.Series([100, 101, 102, 103, 104])
    close = pd.Series([101, 102, 103, 104, 105])
    
    atr_result = atr(high, low, close, period=3)
    
    assert len(atr_result) == len(close)
    assert atr_result.iloc[-1] > 0


def test_macd():
    """Test MACD calculation"""
    series = pd.Series(range(100, 200))
    macd_result = macd(series, fast=12, slow=26, signal=9)
    
    assert 'macd' in macd_result.columns
    assert 'signal' in macd_result.columns
    assert 'histogram' in macd_result.columns


def test_bollinger_bands():
    """Test Bollinger Bands calculation"""
    series = pd.Series(range(100, 200))
    bb_result = bollinger_bands(series, period=20, num_std=2.0)
    
    assert 'upper' in bb_result.columns
    assert 'middle' in bb_result.columns
    assert 'lower' in bb_result.columns
    assert 'width' in bb_result.columns
    
    # Upper should be > middle > lower
    assert bb_result['upper'].iloc[-1] > bb_result['middle'].iloc[-1]
    assert bb_result['middle'].iloc[-1] > bb_result['lower'].iloc[-1]


def test_compute_all_indicators():
    """Test compute_all_indicators"""
    df = pd.DataFrame({
        'open': range(100, 200),
        'high': range(101, 201),
        'low': range(99, 199),
        'close': range(100, 200),
        'volume': [1.0] * 100
    })
    
    result = compute_all_indicators(df)
    
    assert 'adx' in result.columns
    assert 'ema50' in result.columns
    assert 'ema200' in result.columns
    assert 'rsi' in result.columns
    assert 'atr' in result.columns

