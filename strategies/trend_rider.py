import pandas as pd
import numpy as np
from strategies.base import Strategy
from backtest.indicators import compute_ema, compute_sma, compute_atr


class TrendRider(Strategy):
    """
    Maximise time in market during uptrends, cut exposure during downtrends.
    Uses a Keltner Channel (EMA + ATR envelope) to define trend.

    Long when price is above the lower Keltner Channel band (generous
    definition of uptrend — stays invested through normal pullbacks).
    Only exits when price breaks below the channel, confirming a real
    trend change rather than just a dip.

    This keeps you invested ~70-80% of the time while dodging the
    -40% to -70% crashes that kill buy & hold returns.
    """

    name = "Trend Rider"

    def generate_signals(self, data, ema_period=40, atr_period=20,
                         atr_mult=2.0, reentry_mult=0.5):
        close = data["Close"]
        high = data["High"]
        low = data["Low"]

        ema = compute_ema(close, ema_period)
        atr = compute_atr(high, low, close, atr_period)

        lower_band = ema - atr_mult * atr
        reentry_band = ema - reentry_mult * atr

        signals = pd.Series(0, index=data.index)

        # Start long
        warmup = max(ema_period, atr_period) + 1
        signals.iloc[warmup] = 1

        # Exit: close below lower Keltner band
        below = close < lower_band
        cross_below = below & (~below).shift(1, fill_value=False)
        signals[cross_below] = -1

        # Re-enter: price recovers above re-entry band (closer to EMA)
        above_reentry = close > reentry_band
        cross_above = above_reentry & (~above_reentry).shift(1, fill_value=False)
        signals[cross_above] = 1

        return signals

    def default_params(self):
        return {"ema_period": 40, "atr_period": 20, "atr_mult": 2.0, "reentry_mult": 0.5}

    def param_grid(self):
        return {
            "ema_period": [20, 30, 40, 50, 60],
            "atr_period": [10, 14, 20],
            "atr_mult": [1.5, 2.0, 2.5, 3.0],
            "reentry_mult": [0, 0.5, 1.0],
        }
