import pandas as pd
import numpy as np
from strategies.base import Strategy
from backtest.indicators import compute_sma, compute_ema, compute_atr


class MomentumBreakout(Strategy):
    """
    Momentum Breakout — designed for volatile stocks like TSLA.

    Buy when price breaks above the N-day high with above-average volume.
    Uses trailing stop for exits (managed by the engine).

    Entry is only taken when the broader trend (long SMA) is up,
    preventing breakout buys in bear markets.
    """

    name = "Momentum Breakout"

    def generate_signals(self, data, lookback=20, vol_mult=1.3, trend_sma=50):
        close = data["Close"]
        high = data["High"]
        volume = data["Volume"]

        trend = compute_sma(close, trend_sma)
        avg_vol = volume.rolling(window=lookback).mean()
        highest = high.rolling(window=lookback).max().shift(1)  # previous N-day high

        signals = pd.Series(0, index=data.index)

        # Buy: breakout above N-day high, above-average volume, in uptrend
        breakout = close > highest
        vol_surge = volume > avg_vol * vol_mult
        uptrend = close > trend

        signals[breakout & vol_surge & uptrend] = 1

        return signals

    def default_params(self):
        return {"lookback": 20, "vol_mult": 1.3, "trend_sma": 50}

    def param_grid(self):
        return {
            "lookback": [10, 15, 20, 30, 40],
            "vol_mult": [1.0, 1.2, 1.5, 2.0],
            "trend_sma": [20, 50, 100],
        }
