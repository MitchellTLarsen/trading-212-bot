from abc import ABC, abstractmethod
import pandas as pd


class Strategy(ABC):
    """Base class for all trading strategies."""

    name = "Base"

    @abstractmethod
    def generate_signals(self, data, **params):
        """
        Generate trading signals from OHLCV data.

        Returns:
            pd.Series of signals: 1 (buy), -1 (sell), 0 (hold)
        """
        pass

    @abstractmethod
    def default_params(self):
        """Return default parameter dict."""
        pass

    @abstractmethod
    def param_grid(self):
        """Return dict of parameter names -> list of values to search."""
        pass
