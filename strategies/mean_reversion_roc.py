import pandas as pd
import numpy as np
from strategies.base import Strategy
from backtest.indicators import compute_sma, compute_rsi


class MeanReversionROC(Strategy):
    """
    Mean Reversion with Rate of Change confirmation.

    Buy after a sharp multi-day decline (oversold bounce play).
    Specifically looks for:
    - Price dropped X% over N days (oversold)
    - RSI confirms oversold
    - Price is still above long-term trend (not a collapse)

    Exits on recovery to moving average or via trailing stop.
    Designed for volatile stocks that mean-revert after sharp drops.
    """

    name = "Mean Reversion ROC"

    def generate_signals(self, data, roc_period=5, roc_threshold=-8,
                         rsi_period=10, rsi_threshold=35, exit_sma=20,
                         trend_sma=100):
        close = data["Close"]
        roc = (close / close.shift(roc_period) - 1) * 100
        rsi = compute_rsi(close, rsi_period)
        exit_ma = compute_sma(close, exit_sma)
        trend = compute_sma(close, trend_sma)

        signals = pd.Series(0, index=data.index)

        # Buy: sharp drop + RSI oversold + still in long-term uptrend
        oversold = (roc < roc_threshold) & (rsi < rsi_threshold) & (close > trend)
        signals[oversold] = 1

        # Sell: recovered to exit MA
        above_exit = close >= exit_ma
        cross_above = above_exit & (~above_exit).shift(1, fill_value=False)
        signals[cross_above] = -1

        return signals

    def default_params(self):
        return {"roc_period": 5, "roc_threshold": -8, "rsi_period": 10,
                "rsi_threshold": 35, "exit_sma": 20, "trend_sma": 100}

    def param_grid(self):
        return {
            "roc_period": [3, 5, 7, 10],
            "roc_threshold": [-5, -8, -10, -12, -15],
            "rsi_period": [7, 10, 14],
            "rsi_threshold": [25, 30, 35, 40],
            "exit_sma": [10, 15, 20],
            "trend_sma": [50, 100, 150],
        }
