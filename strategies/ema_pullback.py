import pandas as pd
import numpy as np
from strategies.base import Strategy
from backtest.indicators import compute_ema, compute_atr


class EMAPullback(Strategy):
    """
    EMA Trend + Pullback Entry.

    Identifies the trend using fast/slow EMA alignment, then enters
    on pullbacks to the fast EMA. Uses ATR for dynamic exit sizing.

    Buy when:
    - Fast EMA > Slow EMA (uptrend confirmed)
    - Price pulls back to within 1 ATR of fast EMA
    - Price was previously above fast EMA (genuine pullback, not breakdown)

    Sell when:
    - Fast EMA crosses below slow EMA (trend reversal)
    - OR price drops more than a multiple of ATR below entry (stop loss)
    """

    name = "EMA Pullback"

    def generate_signals(self, data, fast_ema=21, slow_ema=55, atr_period=14,
                         pullback_atr_mult=1.0, stop_atr_mult=2.5):
        close = data["Close"]
        high = data["High"]
        low = data["Low"]

        fast = compute_ema(close, fast_ema)
        slow = compute_ema(close, slow_ema)
        atr = compute_atr(high, low, close, atr_period)

        signals = pd.Series(0, index=data.index)

        uptrend = fast > slow
        above_fast = close > fast
        near_fast = (close >= fast - pullback_atr_mult * atr) & (close <= fast)

        # Pullback: was above, now touching EMA from above
        was_above = above_fast.shift(1, fill_value=False) | above_fast.shift(2, fill_value=False)
        pullback_buy = uptrend & near_fast & was_above

        signals[pullback_buy] = 1

        # Sell: trend reversal
        trend_reversal = (~uptrend) & uptrend.shift(1, fill_value=True)
        signals[trend_reversal] = -1

        return signals

    def default_params(self):
        return {
            "fast_ema": 21, "slow_ema": 55, "atr_period": 14,
            "pullback_atr_mult": 1.0, "stop_atr_mult": 2.5,
        }

    def param_grid(self):
        return {
            "fast_ema": [10, 15, 21, 30],
            "slow_ema": [40, 55, 75, 100],
            "atr_period": [10, 14, 20],
            "pullback_atr_mult": [0.5, 1.0, 1.5],
            "stop_atr_mult": [2.0, 2.5, 3.0],
        }
