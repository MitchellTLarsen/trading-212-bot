"""
Portfolio-level backtester for multi-asset allocation strategies.

Instead of single-stock buy/sell signals, strategies return target weights
for each asset. The engine rebalances on specified dates and tracks the
combined portfolio.
"""

import pandas as pd
import numpy as np


class PortfolioEngine:
    """
    Backtests portfolio allocation strategies.

    Strategies return a dict of {ticker: weight} on each rebalance date.
    Weights should sum to <= 1.0 (remainder stays in cash).
    """

    def __init__(self, initial_capital=10000, rebalance_freq="M",
                 slippage_pct=0.05, commission_pct=0.0):
        self.initial_capital = initial_capital
        self.rebalance_freq = rebalance_freq  # "M" = monthly, "W" = weekly
        self.slippage_pct = slippage_pct / 100
        self.commission_pct = commission_pct / 100

    def run(self, datasets, weight_fn):
        """
        Run portfolio backtest.

        Args:
            datasets: dict of {ticker: DataFrame} with OHLCV data
            weight_fn: callable(date, datasets, lookback_data) -> {ticker: weight}

        Returns:
            PortfolioResult
        """
        # Align all data to common date index
        all_dates = None
        for ticker, df in datasets.items():
            if all_dates is None:
                all_dates = set(df.index)
            else:
                all_dates = all_dates.intersection(set(df.index))

        all_dates = sorted(all_dates)
        if len(all_dates) < 60:
            raise ValueError("Not enough overlapping data")

        # Build price matrix
        close_prices = pd.DataFrame(
            {t: df.loc[df.index.isin(all_dates), "Close"] for t, df in datasets.items()},
            index=all_dates
        )

        # Determine rebalance dates
        if self.rebalance_freq == "M":
            rebalance_mask = pd.Series(all_dates).apply(
                lambda d: d.month).diff().fillna(1).astype(bool)
            rebalance_dates = set(pd.Series(all_dates)[rebalance_mask].values)
        elif self.rebalance_freq == "W":
            rebalance_mask = pd.Series(all_dates).apply(
                lambda d: d.isocalendar()[1]).diff().fillna(1).astype(bool)
            rebalance_dates = set(pd.Series(all_dates)[rebalance_mask].values)
        else:
            rebalance_dates = set(all_dates)

        # Track portfolio
        cash = self.initial_capital
        holdings = {}  # {ticker: num_shares}
        equity_curve = []
        trades = []
        weight_history = []

        for i, date in enumerate(all_dates):
            prices = {t: close_prices.loc[date, t] for t in datasets}

            # Rebalance?
            if date in rebalance_dates and i >= 60:  # warmup period
                target_weights = weight_fn(date, datasets, close_prices.loc[:date])

                if target_weights:
                    # Calculate current portfolio value
                    port_value = cash + sum(
                        holdings.get(t, 0) * prices[t] for t in datasets
                    )

                    weight_history.append({
                        "date": date,
                        "weights": target_weights.copy(),
                    })

                    # Calculate target positions
                    new_holdings = {}
                    total_allocated = 0
                    for ticker, weight in target_weights.items():
                        if weight <= 0 or ticker not in prices:
                            continue
                        target_value = port_value * weight
                        price = prices[ticker]
                        shares = int(target_value / (price * (1 + self.slippage_pct)))
                        new_holdings[ticker] = shares
                        total_allocated += shares * price

                    # Execute trades
                    # Sell first (free up cash)
                    for ticker in list(holdings.keys()):
                        old_shares = holdings.get(ticker, 0)
                        new_shares = new_holdings.get(ticker, 0)
                        if new_shares < old_shares:
                            sell_shares = old_shares - new_shares
                            sell_price = prices[ticker] * (1 - self.slippage_pct)
                            revenue = sell_shares * sell_price * (1 - self.commission_pct)
                            cash += revenue
                            trades.append({
                                "date": date, "ticker": ticker, "action": "sell",
                                "shares": sell_shares, "price": sell_price,
                            })

                    # Then buy
                    for ticker, new_shares in new_holdings.items():
                        old_shares = holdings.get(ticker, 0)
                        if new_shares > old_shares:
                            buy_shares = new_shares - old_shares
                            buy_price = prices[ticker] * (1 + self.slippage_pct)
                            cost = buy_shares * buy_price * (1 + self.commission_pct)
                            if cost <= cash:
                                cash -= cost
                                trades.append({
                                    "date": date, "ticker": ticker, "action": "buy",
                                    "shares": buy_shares, "price": buy_price,
                                })
                            else:
                                # Can't afford full position, buy what we can
                                affordable = int(cash / (buy_price * (1 + self.commission_pct)))
                                if affordable > 0:
                                    cost = affordable * buy_price * (1 + self.commission_pct)
                                    cash -= cost
                                    new_holdings[ticker] = old_shares + affordable

                    holdings = {t: s for t, s in new_holdings.items() if s > 0}

            # Track equity
            port_value = cash + sum(
                holdings.get(t, 0) * prices.get(t, 0) for t in holdings
            )
            equity_curve.append({"date": date, "equity": port_value, "cash": cash})

        equity_df = pd.DataFrame(equity_curve).set_index("date")
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()

        # Build benchmark (equal-weight buy & hold)
        bh_weights = {t: 1.0 / len(datasets) for t in datasets}
        bh_returns = close_prices.pct_change().dropna()
        bh_port = (bh_returns * pd.Series(bh_weights)).sum(axis=1)
        bh_equity = self.initial_capital * (1 + bh_port).cumprod()

        return PortfolioResult(
            equity=equity_df,
            trades=trades_df,
            initial_capital=self.initial_capital,
            benchmark_equity=bh_equity,
            weight_history=weight_history,
            close_prices=close_prices,
        )


class PortfolioResult:
    """Portfolio backtest results with metrics."""

    def __init__(self, equity, trades, initial_capital, benchmark_equity,
                 weight_history, close_prices):
        self.equity = equity
        self.trades = trades
        self.initial_capital = initial_capital
        self.benchmark = benchmark_equity
        self.weight_history = weight_history
        self.close_prices = close_prices
        self._metrics = None

    @property
    def metrics(self):
        if self._metrics is None:
            self._metrics = self._compute_metrics()
        return self._metrics

    def _compute_metrics(self):
        eq = self.equity["equity"]
        returns = eq.pct_change().dropna()

        final_equity = eq.iloc[-1]
        total_return = (final_equity / self.initial_capital - 1) * 100

        days = (eq.index[-1] - eq.index[0]).days
        years = days / 365.25
        annual_return = ((final_equity / self.initial_capital) ** (1 / years) - 1) * 100 if years > 0 else 0

        sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0

        peak = eq.cummax()
        drawdown = (eq - peak) / peak
        max_drawdown = drawdown.min() * 100

        # Benchmark metrics
        bh_final = self.benchmark.iloc[-1]
        bh_return = (bh_final / self.initial_capital - 1) * 100

        bh_returns = self.benchmark.pct_change().dropna()
        bh_sharpe = (bh_returns.mean() / bh_returns.std()) * np.sqrt(252) if bh_returns.std() > 0 else 0
        bh_peak = self.benchmark.cummax()
        bh_dd = ((self.benchmark - bh_peak) / bh_peak).min() * 100

        # Calmar ratio (annual return / max drawdown)
        calmar = abs(annual_return / max_drawdown) if max_drawdown != 0 else 0

        # Number of rebalances
        n_rebalances = len(self.weight_history)

        return {
            "total_return_pct": round(total_return, 2),
            "annual_return_pct": round(annual_return, 2),
            "sharpe_ratio": round(sharpe, 3),
            "max_drawdown_pct": round(max_drawdown, 2),
            "calmar_ratio": round(calmar, 3),
            "final_equity": round(final_equity, 2),
            "rebalances": n_rebalances,
            "bh_return_pct": round(bh_return, 2),
            "bh_sharpe": round(bh_sharpe, 3),
            "bh_max_drawdown_pct": round(bh_dd, 2),
        }
