import pandas as pd
import numpy as np
from strategies.base import Strategy
from backtest.indicators import compute_sma, compute_ema, compute_rsi, compute_atr


class CrashAvoider(Strategy):
    """
    Stay long by default, only exit when a major downtrend is confirmed.
    Re-enter as soon as the trend recovers.

    The idea: buy & hold works great except during crashes. If we can
    sit out the worst drops and re-enter on recovery, we beat B&H.

    Exit when price drops below long SMA AND short-term momentum is negative.
    Re-enter when price reclaims the SMA from below.
    """

    name = "Crash Avoider"

    def generate_signals(self, data, trend_sma=50, confirm_days=3,
                         rsi_exit=40, rsi_reentry=50):
        close = data["Close"]
        trend = compute_sma(close, trend_sma)
        rsi = compute_rsi(close, 14)

        signals = pd.Series(0, index=data.index)

        below_trend = close < trend
        # Price has been below trend for N consecutive days
        consecutive_below = below_trend.rolling(window=confirm_days).sum() == confirm_days

        # Exit: confirmed below trend + weak momentum
        exit_signal = consecutive_below & (rsi < rsi_exit)

        # Re-enter: price crosses back above trend + momentum recovering
        above_trend = close > trend
        cross_above = above_trend & (~above_trend).shift(1, fill_value=False)
        reentry = cross_above & (rsi > rsi_reentry)

        # Start with a buy on the first bar (we want to be long by default)
        signals.iloc[trend_sma + 1] = 1

        signals[reentry] = 1
        signals[exit_signal] = -1

        return signals

    def default_params(self):
        return {"trend_sma": 50, "confirm_days": 3, "rsi_exit": 40, "rsi_reentry": 50}

    def param_grid(self):
        return {
            "trend_sma": [30, 40, 50, 60, 75, 100],
            "confirm_days": [2, 3, 5, 7],
            "rsi_exit": [30, 35, 40, 45],
            "rsi_reentry": [45, 50, 55, 60],
        }
