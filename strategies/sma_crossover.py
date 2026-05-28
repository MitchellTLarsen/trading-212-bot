import pandas as pd
import numpy as np
from strategies.base import Strategy
from backtest.indicators import compute_sma


class SMACrossover(Strategy):
    """
    Dual Simple Moving Average Crossover.

    Buy when fast SMA crosses above slow SMA (golden cross).
    Sell when fast SMA crosses below slow SMA (death cross).

    Classic trend-following strategy that captures medium-to-long term moves.
    """

    name = "SMA Crossover"

    def generate_signals(self, data, fast_period=20, slow_period=50):
        close = data["Close"]
        fast_sma = compute_sma(close, fast_period)
        slow_sma = compute_sma(close, slow_period)

        signals = pd.Series(0, index=data.index)

        # Fast above slow = bullish, fast below slow = bearish
        bullish = fast_sma > slow_sma
        bearish = fast_sma < slow_sma

        # Signal on crossover only
        cross_up = bullish & (~bullish).shift(1, fill_value=False)
        cross_down = bearish & (~bearish).shift(1, fill_value=False)

        signals[cross_up] = 1
        signals[cross_down] = -1

        return signals

    def default_params(self):
        return {"fast_period": 20, "slow_period": 50}

    def param_grid(self):
        return {
            "fast_period": [5, 10, 15, 20, 30],
            "slow_period": [30, 50, 75, 100, 150, 200],
        }
