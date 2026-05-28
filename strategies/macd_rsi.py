import pandas as pd
import numpy as np
from strategies.base import Strategy
from backtest.indicators import compute_macd, compute_rsi


class MACDRSICombo(Strategy):
    """
    MACD + RSI Confirmation Strategy.

    Buy when MACD line crosses above signal line AND RSI is below
    the overbought threshold (avoids buying at tops).

    Sell when MACD line crosses below signal line AND RSI is above
    the oversold threshold (avoids panic selling at bottoms).

    Multi-factor confirmation reduces false signals compared to
    using either indicator alone.
    """

    name = "MACD + RSI"

    def generate_signals(self, data, macd_fast=12, macd_slow=26, macd_signal=9,
                         rsi_period=14, rsi_upper=70, rsi_lower=30):
        close = data["Close"]
        macd_line, signal_line, histogram = compute_macd(close, macd_fast, macd_slow, macd_signal)
        rsi = compute_rsi(close, rsi_period)

        signals = pd.Series(0, index=data.index)

        # MACD bullish crossover
        macd_above = macd_line > signal_line
        macd_cross_up = macd_above & (~macd_above).shift(1, fill_value=False)

        # MACD bearish crossover
        macd_below = macd_line < signal_line
        macd_cross_down = macd_below & (~macd_below).shift(1, fill_value=False)

        # Buy: MACD crosses up AND RSI not overbought
        signals[macd_cross_up & (rsi < rsi_upper)] = 1

        # Sell: MACD crosses down AND RSI not oversold
        signals[macd_cross_down & (rsi > rsi_lower)] = -1

        return signals

    def default_params(self):
        return {
            "macd_fast": 12, "macd_slow": 26, "macd_signal": 9,
            "rsi_period": 14, "rsi_upper": 70, "rsi_lower": 30,
        }

    def param_grid(self):
        return {
            "macd_fast": [8, 12, 16],
            "macd_slow": [21, 26, 30],
            "macd_signal": [7, 9, 11],
            "rsi_period": [10, 14],
            "rsi_upper": [65, 70, 75],
            "rsi_lower": [25, 30, 35],
        }
