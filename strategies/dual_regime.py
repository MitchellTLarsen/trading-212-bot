import pandas as pd
import numpy as np
from strategies.base import Strategy
from backtest.indicators import compute_sma, compute_ema, compute_rsi


class DualRegime(Strategy):
    """
    Regime detection strategy — classifies market as bull/bear using
    multiple confirming indicators, then acts accordingly.

    Bull regime (stay long):
      - Price above 200-day SMA
      - 50-day SMA above 200-day SMA
    Bear regime (go to cash):
      - Both conditions fail

    The beauty is simplicity. The 50/200 SMA cross (golden/death cross)
    is one of the most reliable long-term trend indicators. By requiring
    BOTH price above 200 SMA AND 50 above 200, we filter out false
    signals while staying invested through normal corrections.
    """

    name = "Dual Regime"

    def generate_signals(self, data, fast_sma=50, slow_sma=200,
                         reentry_buffer=1.02):
        close = data["Close"]
        fast = compute_sma(close, fast_sma)
        slow = compute_sma(close, slow_sma)

        signals = pd.Series(0, index=data.index)

        # Bull regime: price above slow SMA and fast above slow
        bull = (close > slow) & (fast > slow)

        # Bear regime: both fail
        bear = (close < slow) & (fast < slow)

        # Enter on transition to bull
        enter_bull = bull & (~bull).shift(1, fill_value=False)
        signals[enter_bull] = 1

        # Exit on confirmed bear (not just a dip)
        # Require bear for 2 consecutive days
        confirmed_bear = bear.rolling(2).sum() == 2
        exit_bear = confirmed_bear & (~confirmed_bear).shift(1, fill_value=False)
        signals[exit_bear] = -1

        # Also enter at start if in bull regime
        warmup = slow_sma + 1
        if warmup < len(data) and bull.iloc[warmup]:
            signals.iloc[warmup] = 1

        return signals

    def default_params(self):
        return {"fast_sma": 50, "slow_sma": 200, "reentry_buffer": 1.02}

    def param_grid(self):
        return {
            "fast_sma": [20, 30, 40, 50, 60, 75],
            "slow_sma": [100, 120, 150, 175, 200],
            "reentry_buffer": [1.0, 1.01, 1.02, 1.03],
        }
