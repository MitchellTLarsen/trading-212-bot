"""
Swing trading strategies — hold for days to weeks max.

Each function returns entry signals (1 = buy, 0 = nothing).
Exits are handled by the SwingEngine (time, stop loss, take profit).
"""

import pandas as pd
import numpy as np
from backtest.indicators import compute_sma, compute_ema, compute_rsi, compute_bollinger_bands, compute_atr


def consecutive_down_days(data, n_days=3, trend_sma=100):
    """
    Buy after N consecutive down days in an uptrend.

    Statistical edge: stocks that drop several days in a row
    tend to bounce. Filtered by long-term uptrend to avoid
    catching falling knives in bear markets.
    """
    close = data["Close"]
    trend = compute_sma(close, trend_sma)

    down = (close < close.shift(1)).astype(int)
    consecutive = down.rolling(n_days).sum()

    signals = pd.Series(0, index=data.index)
    signals[(consecutive == n_days) & (close > trend)] = 1
    return signals


def rsi_extreme_bounce(data, rsi_period=5, rsi_threshold=15, trend_sma=100):
    """
    Buy when short-term RSI hits extreme oversold.

    RSI(5) below 15 is a very rare event — strong statistical tendency
    to bounce. Trend filter keeps us on the right side.
    """
    close = data["Close"]
    rsi = compute_rsi(close, rsi_period)
    trend = compute_sma(close, trend_sma)

    signals = pd.Series(0, index=data.index)
    # Buy when RSI drops to extreme and we're in uptrend
    signals[(rsi < rsi_threshold) & (close > trend)] = 1
    return signals


def bollinger_snap(data, bb_period=20, bb_std=2.0, trend_sma=100):
    """
    Buy when price closes below lower Bollinger Band in uptrend.

    Price snapping back from below the lower band is a reliable
    mean-reversion pattern for swing trades.
    """
    close = data["Close"]
    middle, upper, lower = compute_bollinger_bands(close, bb_period, bb_std)
    trend = compute_sma(close, trend_sma)

    signals = pd.Series(0, index=data.index)
    # Price closes below lower band, overall uptrend
    signals[(close < lower) & (close > trend)] = 1
    return signals


def gap_down_reversal(data, gap_pct=2, trend_sma=50):
    """
    Buy on gap-down open that recovers intraday.

    If a stock gaps down X% at open but closes higher than open,
    it signals strong buying pressure — likely to continue up.
    """
    close = data["Close"]
    open_ = data["Open"]
    prev_close = close.shift(1)
    trend = compute_sma(close, trend_sma)

    # Gap down at open
    gap = (open_ / prev_close - 1) * 100
    gapped_down = gap < -gap_pct

    # But recovered intraday (close > open)
    recovered = close > open_

    signals = pd.Series(0, index=data.index)
    signals[gapped_down & recovered & (close > trend)] = 1
    return signals


def weekly_momentum_burst(data, lookback=5, min_return=3, vol_mult=1.5):
    """
    Buy when a stock shows a momentum burst: strong 1-week return
    with above-average volume (institutional buying).

    The idea: big money moves cause trends that persist for 1-2 more weeks.
    """
    close = data["Close"]
    volume = data["Volume"]

    week_return = (close / close.shift(lookback) - 1) * 100
    avg_vol = volume.rolling(20).mean()

    signals = pd.Series(0, index=data.index)
    signals[(week_return > min_return) & (volume > avg_vol * vol_mult)] = 1
    return signals


def mean_reversion_zscore(data, lookback=20, z_threshold=-2.0, trend_sma=100):
    """
    Buy when price z-score drops below threshold.

    Z-score measures how many standard deviations price is from its
    recent mean. Below -2 is statistically rare and tends to revert.
    """
    close = data["Close"]
    trend = compute_sma(close, trend_sma)

    rolling_mean = close.rolling(lookback).mean()
    rolling_std = close.rolling(lookback).std()
    zscore = (close - rolling_mean) / rolling_std

    signals = pd.Series(0, index=data.index)
    signals[(zscore < z_threshold) & (close > trend)] = 1
    return signals


def atr_breakout(data, atr_period=14, atr_mult=1.5, lookback=20):
    """
    Buy when price breaks above recent range by more than 1.5x ATR.

    Large ATR moves signal new trend initiation. Combined with
    volume confirmation for higher conviction.
    """
    close = data["Close"]
    high = data["High"]
    low = data["Low"]
    volume = data["Volume"]

    atr = compute_atr(high, low, close, atr_period)
    recent_high = high.rolling(lookback).max().shift(1)
    avg_vol = volume.rolling(20).mean()

    breakout = close > (recent_high + atr_mult * atr)
    vol_confirm = volume > avg_vol * 1.2

    signals = pd.Series(0, index=data.index)
    signals[breakout & vol_confirm] = 1
    return signals


def dip_and_rip(data, dip_pct=5, rsi_period=7, rsi_threshold=30, trend_sma=50):
    """
    Buy after a sharp multi-day drop with RSI confirmation.

    Combines price drop percentage with RSI oversold for
    double confirmation of oversold conditions.
    """
    close = data["Close"]
    rsi = compute_rsi(close, rsi_period)
    trend = compute_sma(close, trend_sma)

    recent_high = close.rolling(10).max()
    drop_pct = (close / recent_high - 1) * 100

    signals = pd.Series(0, index=data.index)
    signals[(drop_pct < -dip_pct) & (rsi < rsi_threshold) & (close > trend)] = 1
    return signals


# Registry with parameter grids
SWING_STRATEGIES = {
    "Consecutive Down Days": {
        "fn": consecutive_down_days,
        "params_grid": [
            {"n_days": 3, "trend_sma": 50},
            {"n_days": 3, "trend_sma": 100},
            {"n_days": 3, "trend_sma": 200},
            {"n_days": 4, "trend_sma": 50},
            {"n_days": 4, "trend_sma": 100},
            {"n_days": 4, "trend_sma": 200},
            {"n_days": 5, "trend_sma": 50},
            {"n_days": 5, "trend_sma": 100},
        ],
    },
    "RSI Extreme Bounce": {
        "fn": rsi_extreme_bounce,
        "params_grid": [
            {"rsi_period": 3, "rsi_threshold": 10, "trend_sma": 50},
            {"rsi_period": 3, "rsi_threshold": 15, "trend_sma": 50},
            {"rsi_period": 5, "rsi_threshold": 15, "trend_sma": 50},
            {"rsi_period": 5, "rsi_threshold": 15, "trend_sma": 100},
            {"rsi_period": 5, "rsi_threshold": 20, "trend_sma": 50},
            {"rsi_period": 5, "rsi_threshold": 20, "trend_sma": 100},
            {"rsi_period": 7, "rsi_threshold": 20, "trend_sma": 50},
            {"rsi_period": 7, "rsi_threshold": 25, "trend_sma": 100},
        ],
    },
    "Bollinger Snap": {
        "fn": bollinger_snap,
        "params_grid": [
            {"bb_period": 15, "bb_std": 2.0, "trend_sma": 50},
            {"bb_period": 20, "bb_std": 2.0, "trend_sma": 50},
            {"bb_period": 20, "bb_std": 2.0, "trend_sma": 100},
            {"bb_period": 20, "bb_std": 2.5, "trend_sma": 50},
            {"bb_period": 20, "bb_std": 2.5, "trend_sma": 100},
            {"bb_period": 25, "bb_std": 2.0, "trend_sma": 100},
        ],
    },
    "Gap Down Reversal": {
        "fn": gap_down_reversal,
        "params_grid": [
            {"gap_pct": 1.5, "trend_sma": 50},
            {"gap_pct": 2, "trend_sma": 50},
            {"gap_pct": 2, "trend_sma": 100},
            {"gap_pct": 3, "trend_sma": 50},
            {"gap_pct": 3, "trend_sma": 100},
            {"gap_pct": 1.5, "trend_sma": 100},
        ],
    },
    "Weekly Momentum Burst": {
        "fn": weekly_momentum_burst,
        "params_grid": [
            {"lookback": 5, "min_return": 2, "vol_mult": 1.3},
            {"lookback": 5, "min_return": 3, "vol_mult": 1.3},
            {"lookback": 5, "min_return": 3, "vol_mult": 1.5},
            {"lookback": 5, "min_return": 4, "vol_mult": 1.3},
            {"lookback": 5, "min_return": 5, "vol_mult": 1.2},
            {"lookback": 3, "min_return": 2, "vol_mult": 1.3},
        ],
    },
    "Z-Score Mean Reversion": {
        "fn": mean_reversion_zscore,
        "params_grid": [
            {"lookback": 15, "z_threshold": -1.5, "trend_sma": 50},
            {"lookback": 15, "z_threshold": -2.0, "trend_sma": 50},
            {"lookback": 20, "z_threshold": -1.5, "trend_sma": 50},
            {"lookback": 20, "z_threshold": -2.0, "trend_sma": 50},
            {"lookback": 20, "z_threshold": -2.0, "trend_sma": 100},
            {"lookback": 20, "z_threshold": -2.5, "trend_sma": 50},
            {"lookback": 30, "z_threshold": -2.0, "trend_sma": 100},
        ],
    },
    "ATR Breakout": {
        "fn": atr_breakout,
        "params_grid": [
            {"atr_period": 10, "atr_mult": 1.0, "lookback": 15},
            {"atr_period": 14, "atr_mult": 1.0, "lookback": 20},
            {"atr_period": 14, "atr_mult": 1.5, "lookback": 20},
            {"atr_period": 14, "atr_mult": 1.5, "lookback": 10},
            {"atr_period": 10, "atr_mult": 1.5, "lookback": 15},
            {"atr_period": 20, "atr_mult": 1.0, "lookback": 20},
        ],
    },
    "Dip and Rip": {
        "fn": dip_and_rip,
        "params_grid": [
            {"dip_pct": 3, "rsi_period": 5, "rsi_threshold": 30, "trend_sma": 50},
            {"dip_pct": 3, "rsi_period": 7, "rsi_threshold": 30, "trend_sma": 50},
            {"dip_pct": 5, "rsi_period": 5, "rsi_threshold": 30, "trend_sma": 50},
            {"dip_pct": 5, "rsi_period": 7, "rsi_threshold": 35, "trend_sma": 50},
            {"dip_pct": 5, "rsi_period": 7, "rsi_threshold": 30, "trend_sma": 100},
            {"dip_pct": 7, "rsi_period": 5, "rsi_threshold": 25, "trend_sma": 50},
            {"dip_pct": 4, "rsi_period": 5, "rsi_threshold": 30, "trend_sma": 50},
        ],
    },
}
