"""
Forex-specific strategies with long AND short signals.

Signals: 1 = go long, -1 = go short, 0 = do nothing
"""

import pandas as pd
import numpy as np
from backtest.indicators import compute_sma, compute_ema, compute_rsi, compute_macd, compute_bollinger_bands, compute_atr


def sma_crossover_bidirectional(data, fast=10, slow=30):
    """
    SMA crossover — long AND short.
    Long when fast > slow, short when fast < slow.
    """
    close = data["Close"]
    fast_sma = compute_sma(close, fast)
    slow_sma = compute_sma(close, slow)

    signals = pd.Series(0, index=data.index)
    bullish = fast_sma > slow_sma

    cross_up = bullish & (~bullish).shift(1, fill_value=False)
    cross_down = (~bullish) & bullish.shift(1, fill_value=True)

    signals[cross_up] = 1
    signals[cross_down] = -1
    return signals


def rsi_reversal(data, rsi_period=7, oversold=25, overbought=75):
    """
    RSI mean reversion — long when oversold, short when overbought.
    """
    close = data["Close"]
    rsi = compute_rsi(close, rsi_period)

    signals = pd.Series(0, index=data.index)
    signals[(rsi < oversold) & (rsi.shift(1) >= oversold)] = 1
    signals[(rsi > overbought) & (rsi.shift(1) <= overbought)] = -1
    return signals


def bollinger_reversal(data, period=20, num_std=2.0):
    """
    Bollinger Band mean reversion — long at lower band, short at upper.
    """
    close = data["Close"]
    middle, upper, lower = compute_bollinger_bands(close, period, num_std)

    signals = pd.Series(0, index=data.index)
    signals[close <= lower] = 1
    signals[close >= upper] = -1
    return signals


def macd_crossover(data, fast=12, slow=26, signal_period=9):
    """
    MACD line/signal crossover — bi-directional.
    """
    close = data["Close"]
    macd_line, signal_line, _ = compute_macd(close, fast, slow, signal_period)

    signals = pd.Series(0, index=data.index)
    above = macd_line > signal_line
    cross_up = above & (~above).shift(1, fill_value=False)
    cross_down = (~above) & above.shift(1, fill_value=True)

    signals[cross_up] = 1
    signals[cross_down] = -1
    return signals


def keltner_breakout(data, ema_period=20, atr_period=14, atr_mult=1.5):
    """
    Keltner Channel breakout — long above upper, short below lower.
    Trend-following: catches breakouts from consolidation.
    """
    close = data["Close"]
    high = data["High"]
    low = data["Low"]

    ema = compute_ema(close, ema_period)
    atr = compute_atr(high, low, close, atr_period)

    upper = ema + atr_mult * atr
    lower = ema - atr_mult * atr

    signals = pd.Series(0, index=data.index)
    signals[close > upper] = 1
    signals[close < lower] = -1
    return signals


def triple_ema(data, fast=5, medium=13, slow=34):
    """
    Triple EMA alignment — strong trend confirmation.
    Long when fast > medium > slow. Short when fast < medium < slow.
    Only signals on fresh alignments.
    """
    close = data["Close"]
    ema_f = compute_ema(close, fast)
    ema_m = compute_ema(close, medium)
    ema_s = compute_ema(close, slow)

    bull = (ema_f > ema_m) & (ema_m > ema_s)
    bear = (ema_f < ema_m) & (ema_m < ema_s)

    signals = pd.Series(0, index=data.index)
    signals[bull & (~bull).shift(1, fill_value=False)] = 1
    signals[bear & (~bear).shift(1, fill_value=False)] = -1
    return signals


def zscore_mean_reversion(data, lookback=20, z_entry=2.0):
    """
    Z-score mean reversion — long when z < -threshold, short when z > threshold.
    Price tends to revert to its rolling mean.
    """
    close = data["Close"]
    mean = close.rolling(lookback).mean()
    std = close.rolling(lookback).std()
    zscore = (close - mean) / std

    signals = pd.Series(0, index=data.index)
    signals[zscore < -z_entry] = 1
    signals[zscore > z_entry] = -1
    return signals


def momentum_breakout(data, lookback=20, vol_mult=1.3):
    """
    Breakout above/below N-day range with volume confirmation.
    """
    close = data["Close"]
    high = data["High"]
    low = data["Low"]
    volume = data["Volume"]

    highest = high.rolling(lookback).max().shift(1)
    lowest = low.rolling(lookback).min().shift(1)
    avg_vol = volume.rolling(20).mean()

    vol_ok = (volume > avg_vol * vol_mult) | (volume == 0)  # some forex has 0 volume

    signals = pd.Series(0, index=data.index)
    signals[(close > highest) & vol_ok] = 1
    signals[(close < lowest) & vol_ok] = -1
    return signals


def consecutive_candles(data, n_candles=3):
    """
    Mean reversion after N consecutive up/down candles.
    After 3+ down candles, go long (expect bounce).
    After 3+ up candles, go short (expect pullback).
    """
    close = data["Close"]
    up = (close > close.shift(1)).astype(int)
    down = (close < close.shift(1)).astype(int)

    consec_up = up.rolling(n_candles).sum()
    consec_down = down.rolling(n_candles).sum()

    signals = pd.Series(0, index=data.index)
    signals[consec_down == n_candles] = 1   # oversold bounce
    signals[consec_up == n_candles] = -1    # overbought pullback
    return signals


def inside_bar_breakout(data):
    """
    Inside bar pattern — consolidation followed by breakout.
    Inside bar: today's high < yesterday's high AND today's low > yesterday's low.
    Trade the breakout direction next day.
    """
    high = data["High"]
    low = data["Low"]
    close = data["Close"]

    inside = (high < high.shift(1)) & (low > low.shift(1))

    # On the bar after an inside bar, go with the breakout direction
    signals = pd.Series(0, index=data.index)

    for i in range(2, len(data)):
        if inside.iloc[i-1]:
            if close.iloc[i] > high.iloc[i-1]:
                signals.iloc[i] = 1
            elif close.iloc[i] < low.iloc[i-1]:
                signals.iloc[i] = -1

    return signals


FOREX_STRATEGIES = {
    "SMA Crossover": {
        "fn": sma_crossover_bidirectional,
        "params_grid": [
            {"fast": 5, "slow": 20},
            {"fast": 5, "slow": 30},
            {"fast": 10, "slow": 30},
            {"fast": 10, "slow": 50},
            {"fast": 15, "slow": 30},
            {"fast": 15, "slow": 50},
            {"fast": 20, "slow": 50},
            {"fast": 20, "slow": 75},
        ],
    },
    "RSI Reversal": {
        "fn": rsi_reversal,
        "params_grid": [
            {"rsi_period": 5, "oversold": 20, "overbought": 80},
            {"rsi_period": 5, "oversold": 25, "overbought": 75},
            {"rsi_period": 7, "oversold": 25, "overbought": 75},
            {"rsi_period": 7, "oversold": 30, "overbought": 70},
            {"rsi_period": 10, "oversold": 25, "overbought": 75},
            {"rsi_period": 10, "oversold": 30, "overbought": 70},
            {"rsi_period": 14, "oversold": 25, "overbought": 75},
            {"rsi_period": 14, "oversold": 30, "overbought": 70},
        ],
    },
    "Bollinger Reversal": {
        "fn": bollinger_reversal,
        "params_grid": [
            {"period": 15, "num_std": 2.0},
            {"period": 20, "num_std": 2.0},
            {"period": 20, "num_std": 2.5},
            {"period": 25, "num_std": 2.0},
            {"period": 25, "num_std": 2.5},
            {"period": 30, "num_std": 2.0},
        ],
    },
    "MACD Crossover": {
        "fn": macd_crossover,
        "params_grid": [
            {"fast": 8, "slow": 21, "signal_period": 7},
            {"fast": 12, "slow": 26, "signal_period": 9},
            {"fast": 12, "slow": 30, "signal_period": 9},
            {"fast": 16, "slow": 30, "signal_period": 9},
            {"fast": 8, "slow": 17, "signal_period": 9},
            {"fast": 5, "slow": 15, "signal_period": 5},
        ],
    },
    "Keltner Breakout": {
        "fn": keltner_breakout,
        "params_grid": [
            {"ema_period": 15, "atr_period": 10, "atr_mult": 1.0},
            {"ema_period": 15, "atr_period": 10, "atr_mult": 1.5},
            {"ema_period": 20, "atr_period": 14, "atr_mult": 1.0},
            {"ema_period": 20, "atr_period": 14, "atr_mult": 1.5},
            {"ema_period": 20, "atr_period": 14, "atr_mult": 2.0},
            {"ema_period": 30, "atr_period": 14, "atr_mult": 1.5},
        ],
    },
    "Triple EMA": {
        "fn": triple_ema,
        "params_grid": [
            {"fast": 3, "medium": 8, "slow": 21},
            {"fast": 5, "medium": 13, "slow": 34},
            {"fast": 5, "medium": 13, "slow": 50},
            {"fast": 8, "medium": 21, "slow": 55},
            {"fast": 5, "medium": 10, "slow": 20},
            {"fast": 3, "medium": 10, "slow": 30},
        ],
    },
    "Z-Score Reversion": {
        "fn": zscore_mean_reversion,
        "params_grid": [
            {"lookback": 10, "z_entry": 1.5},
            {"lookback": 15, "z_entry": 1.5},
            {"lookback": 15, "z_entry": 2.0},
            {"lookback": 20, "z_entry": 1.5},
            {"lookback": 20, "z_entry": 2.0},
            {"lookback": 20, "z_entry": 2.5},
            {"lookback": 30, "z_entry": 2.0},
        ],
    },
    "Momentum Breakout": {
        "fn": momentum_breakout,
        "params_grid": [
            {"lookback": 10, "vol_mult": 1.0},
            {"lookback": 15, "vol_mult": 1.0},
            {"lookback": 20, "vol_mult": 1.0},
            {"lookback": 20, "vol_mult": 1.3},
            {"lookback": 30, "vol_mult": 1.0},
            {"lookback": 10, "vol_mult": 1.3},
        ],
    },
    "Consecutive Candles": {
        "fn": consecutive_candles,
        "params_grid": [
            {"n_candles": 3},
            {"n_candles": 4},
            {"n_candles": 5},
        ],
    },
    "Inside Bar Breakout": {
        "fn": inside_bar_breakout,
        "params_grid": [{}],
    },
}
