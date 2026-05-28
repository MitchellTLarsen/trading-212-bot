import pandas as pd
import numpy as np
from strategies.base import Strategy
from backtest.indicators import compute_rsi, compute_sma


class RSIMeanReversion(Strategy):
    """
    RSI Mean Reversion with Trend Filter.

    Buy when RSI drops below oversold level AND price is above a long SMA
    (only buy dips in an uptrend). Sell when RSI rises above overbought level.

    The trend filter prevents buying into falling knives in bear markets.
    """

    name = "RSI Mean Reversion"

    def generate_signals(self, data, rsi_period=14, oversold=30, overbought=70,
                         trend_sma=200):
        close = data["Close"]
        rsi = compute_rsi(close, rsi_period)
        trend = compute_sma(close, trend_sma)

        signals = pd.Series(0, index=data.index)

        in_uptrend = close > trend

        # Buy: RSI crosses below oversold in uptrend
        rsi_oversold = (rsi < oversold) & (rsi.shift(1) >= oversold)
        signals[rsi_oversold & in_uptrend] = 1

        # Sell: RSI crosses above overbought
        rsi_overbought = (rsi > overbought) & (rsi.shift(1) <= overbought)
        signals[rsi_overbought] = -1

        return signals

    def default_params(self):
        return {"rsi_period": 14, "oversold": 30, "overbought": 70, "trend_sma": 200}

    def param_grid(self):
        return {
            "rsi_period": [7, 10, 14, 21],
            "oversold": [20, 25, 30, 35],
            "overbought": [65, 70, 75, 80],
            "trend_sma": [100, 150, 200],
        }
