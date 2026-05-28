import pandas as pd
import numpy as np
from strategies.base import Strategy
from backtest.indicators import compute_bollinger_bands, compute_rsi


class BollingerBandReversion(Strategy):
    """
    Bollinger Band Mean Reversion.

    Buy when price touches the lower band and RSI confirms oversold.
    Sell when price reaches the middle band (take profit) or upper band.

    Works well in ranging/mean-reverting markets. The RSI filter
    avoids catching falling knives during genuine breakdowns.
    """

    name = "Bollinger Band Reversion"

    def generate_signals(self, data, bb_period=20, bb_std=2.0, rsi_period=14,
                         rsi_threshold=40):
        close = data["Close"]
        middle, upper, lower = compute_bollinger_bands(close, bb_period, bb_std)
        rsi = compute_rsi(close, rsi_period)

        signals = pd.Series(0, index=data.index)

        # Buy: price touches lower band and RSI is below threshold
        at_lower = close <= lower
        rsi_ok = rsi < rsi_threshold
        signals[at_lower & rsi_ok] = 1

        # Sell: price reaches middle band or above
        at_middle_or_above = close >= middle
        # Only signal sell on the cross, not continuously
        cross_middle = at_middle_or_above & (~at_middle_or_above).shift(1, fill_value=False)
        signals[cross_middle] = -1

        return signals

    def default_params(self):
        return {"bb_period": 20, "bb_std": 2.0, "rsi_period": 14, "rsi_threshold": 40}

    def param_grid(self):
        return {
            "bb_period": [15, 20, 25, 30],
            "bb_std": [1.5, 2.0, 2.5],
            "rsi_period": [7, 10, 14],
            "rsi_threshold": [30, 35, 40, 45],
        }
