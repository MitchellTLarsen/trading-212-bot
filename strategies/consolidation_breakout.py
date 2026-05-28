import pandas as pd
import numpy as np
from strategies.base import Strategy
from backtest.indicators import compute_bollinger_bands, compute_sma, compute_atr


class ConsolidationBreakout(Strategy):
    """
    Bollinger Band Squeeze -> Breakout strategy.

    Detects when volatility contracts (Bollinger Band width shrinks to
    recent lows = "squeeze"), then enters when price breaks out upward.

    The idea: low volatility precedes big moves. By waiting for the squeeze
    and then trading the breakout direction, we catch the start of trends.

    Exit via trailing stop (engine-managed) or signal.
    """

    name = "Consolidation Breakout"

    def generate_signals(self, data, bb_period=20, bb_std=2.0,
                         squeeze_lookback=50, squeeze_percentile=25,
                         trend_sma=50):
        close = data["Close"]
        middle, upper, lower = compute_bollinger_bands(close, bb_period, bb_std)
        trend = compute_sma(close, trend_sma)

        # Band width as % of price
        bandwidth = (upper - lower) / middle * 100
        bw_threshold = bandwidth.rolling(squeeze_lookback).apply(
            lambda x: np.percentile(x, squeeze_percentile), raw=True
        )

        signals = pd.Series(0, index=data.index)

        # Squeeze: bandwidth below its Nth percentile
        in_squeeze = bandwidth < bw_threshold

        # Was recently in squeeze (within last 5 bars)
        was_squeezed = in_squeeze.rolling(5).max().fillna(0).astype(bool)

        # Breakout: price breaks above upper band after squeeze, in uptrend
        breakout = (close > upper) & was_squeezed & (close > trend)
        signals[breakout] = 1

        # Sell on breakdown below middle band
        below_middle = close < middle
        cross_below = below_middle & (~below_middle).shift(1, fill_value=False)
        signals[cross_below] = -1

        return signals

    def default_params(self):
        return {"bb_period": 20, "bb_std": 2.0, "squeeze_lookback": 50,
                "squeeze_percentile": 25, "trend_sma": 50}

    def param_grid(self):
        return {
            "bb_period": [15, 20, 25],
            "bb_std": [1.5, 2.0, 2.5],
            "squeeze_lookback": [30, 50, 75],
            "squeeze_percentile": [15, 20, 25, 30],
            "trend_sma": [30, 50, 75],
        }
