import pandas as pd
import numpy as np
from strategies.base import Strategy
from backtest.indicators import compute_sma, compute_ema, compute_rsi


class DipBuyer(Strategy):
    """
    Aggressive dip buyer — stay long through uptrends, buy every
    significant dip, and only fully exit during confirmed bear markets.

    Buy when price pulls back to a rising EMA.
    Hold as long as the long-term trend (slow EMA) is rising.
    Only exit when the slow EMA itself starts declining (bear market).

    Designed to maximize time in market while buying dips for better
    average entry prices.
    """

    name = "Dip Buyer"

    def generate_signals(self, data, fast_ema=10, slow_ema=50,
                         dip_pct=3, trend_lookback=10):
        close = data["Close"]
        fast = compute_ema(close, fast_ema)
        slow = compute_ema(close, slow_ema)

        # Slow EMA slope (is the long-term trend rising?)
        slow_slope = slow - slow.shift(trend_lookback)
        trend_rising = slow_slope > 0

        # Dip detection: price fell X% from recent high
        recent_high = close.rolling(20).max()
        dip = (close / recent_high - 1) * 100 <= -dip_pct

        signals = pd.Series(0, index=data.index)

        # Enter: first bar after warmup
        signals.iloc[slow_ema + 1] = 1

        # Buy dips when trend is rising
        buy_dip = dip & trend_rising
        signals[buy_dip] = 1

        # Also re-enter when price crosses back above fast EMA in uptrend
        above_fast = close > fast
        cross_above = above_fast & (~above_fast).shift(1, fill_value=False)
        signals[cross_above & trend_rising] = 1

        # Exit: trend is no longer rising AND price below slow EMA
        bear_market = (~trend_rising) & (close < slow)
        # Confirm with consecutive days
        confirmed_bear = bear_market.rolling(3).sum() == 3
        signals[confirmed_bear] = -1

        return signals

    def default_params(self):
        return {"fast_ema": 10, "slow_ema": 50, "dip_pct": 3, "trend_lookback": 10}

    def param_grid(self):
        return {
            "fast_ema": [5, 10, 15, 21],
            "slow_ema": [30, 40, 50, 60, 75],
            "dip_pct": [2, 3, 5, 7],
            "trend_lookback": [5, 10, 15, 20],
        }
