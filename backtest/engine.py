import pandas as pd
import numpy as np
from backtest.indicators import compute_atr


class BacktestEngine:
    """
    Event-driven backtester for long-only equity strategies.

    Signals:
        1  = buy (enter long)
       -1  = sell (exit position)
        0  = hold (do nothing)

    Supports trailing stops, fixed stop losses, and take profit targets.
    Executes at next bar's open to avoid look-ahead bias.
    """

    def __init__(self, initial_capital=10000, commission_pct=0.0, slippage_pct=0.05,
                 trailing_stop_pct=0, fixed_stop_pct=0, take_profit_pct=0):
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct / 100
        self.slippage_pct = slippage_pct / 100
        self.trailing_stop_pct = trailing_stop_pct / 100   # e.g. 10 = 10% trailing stop
        self.fixed_stop_pct = fixed_stop_pct / 100         # e.g. 5 = 5% fixed stop loss
        self.take_profit_pct = take_profit_pct / 100       # e.g. 15 = 15% take profit

    def run(self, data, signals):
        df = data.copy()
        df["signal"] = signals

        cash = self.initial_capital
        shares = 0
        equity_curve = []
        trades = []
        entry_price = 0
        entry_date = None
        highest_since_entry = 0

        for i in range(len(df)):
            date = df.index[i]
            close = df["Close"].iloc[i]
            high = df["High"].iloc[i]
            low = df["Low"].iloc[i]
            signal = df["signal"].iloc[i]

            if i + 1 < len(df):
                exec_price = df["Open"].iloc[i + 1]
            else:
                exec_price = close

            # Check stop/target exits during the bar (using high/low)
            forced_exit = False
            forced_exit_price = None

            if shares > 0:
                highest_since_entry = max(highest_since_entry, high)

                # Trailing stop: exit if price drops X% from peak
                if self.trailing_stop_pct > 0:
                    trail_stop_level = highest_since_entry * (1 - self.trailing_stop_pct)
                    if low <= trail_stop_level:
                        forced_exit = True
                        forced_exit_price = trail_stop_level

                # Fixed stop loss
                if self.fixed_stop_pct > 0 and not forced_exit:
                    stop_level = entry_price * (1 - self.fixed_stop_pct)
                    if low <= stop_level:
                        forced_exit = True
                        forced_exit_price = stop_level

                # Take profit
                if self.take_profit_pct > 0 and not forced_exit:
                    tp_level = entry_price * (1 + self.take_profit_pct)
                    if high >= tp_level:
                        forced_exit = True
                        forced_exit_price = tp_level

            # Execute forced exit
            if forced_exit and shares > 0:
                sell_price = forced_exit_price * (1 - self.slippage_pct)
                revenue = shares * sell_price * (1 - self.commission_pct)
                pnl = revenue - (shares * entry_price)
                pnl_pct = (sell_price / entry_price - 1) * 100

                trades.append({
                    "entry_date": entry_date,
                    "exit_date": date,
                    "entry_price": entry_price,
                    "exit_price": sell_price,
                    "shares": shares,
                    "pnl": pnl,
                    "pnl_pct": pnl_pct,
                    "exit_reason": "trailing_stop" if self.trailing_stop_pct > 0 and low <= highest_since_entry * (1 - self.trailing_stop_pct) else "stop/tp",
                })

                cash += revenue
                shares = 0
                entry_price = 0
                entry_date = None
                highest_since_entry = 0

            # Signal-based entry/exit
            elif signal == 1 and shares == 0:
                buy_price = exec_price * (1 + self.slippage_pct)
                cost = buy_price * (1 + self.commission_pct)
                shares = int(cash / cost)
                if shares > 0:
                    cash -= shares * cost
                    entry_price = buy_price
                    entry_date = date
                    highest_since_entry = buy_price

            elif signal == -1 and shares > 0:
                sell_price = exec_price * (1 - self.slippage_pct)
                revenue = shares * sell_price * (1 - self.commission_pct)
                pnl = revenue - (shares * entry_price)
                pnl_pct = (sell_price / entry_price - 1) * 100

                trades.append({
                    "entry_date": entry_date,
                    "exit_date": date,
                    "entry_price": entry_price,
                    "exit_price": sell_price,
                    "shares": shares,
                    "pnl": pnl,
                    "pnl_pct": pnl_pct,
                    "exit_reason": "signal",
                })

                cash += revenue
                shares = 0
                entry_price = 0
                entry_date = None
                highest_since_entry = 0

            portfolio_value = cash + shares * close
            equity_curve.append({
                "date": date,
                "equity": portfolio_value,
                "cash": cash,
                "shares": shares,
            })

        equity_df = pd.DataFrame(equity_curve).set_index("date")
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()

        return BacktestResult(
            equity=equity_df,
            trades=trades_df,
            initial_capital=self.initial_capital,
            data=data,
        )


class BacktestResult:
    """Holds backtest results and computes performance metrics."""

    def __init__(self, equity, trades, initial_capital, data):
        self.equity = equity
        self.trades = trades
        self.initial_capital = initial_capital
        self.data = data
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
        total_return_pct = (final_equity / self.initial_capital - 1) * 100

        days = (eq.index[-1] - eq.index[0]).days
        years = days / 365.25
        if years > 0 and final_equity > 0:
            annual_return = ((final_equity / self.initial_capital) ** (1 / years) - 1) * 100
        else:
            annual_return = 0

        if len(returns) > 1 and returns.std() > 0:
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
        else:
            sharpe = 0

        peak = eq.cummax()
        drawdown = (eq - peak) / peak
        max_drawdown = drawdown.min() * 100

        if len(self.trades) > 0:
            wins = self.trades[self.trades["pnl"] > 0]
            losses = self.trades[self.trades["pnl"] <= 0]
            win_rate = len(wins) / len(self.trades) * 100
            avg_win = wins["pnl_pct"].mean() if len(wins) > 0 else 0
            avg_loss = losses["pnl_pct"].mean() if len(losses) > 0 else 0
            profit_factor = (wins["pnl"].sum() / abs(losses["pnl"].sum())
                            if len(losses) > 0 and losses["pnl"].sum() != 0 else float("inf"))
            total_trades = len(self.trades)
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
            total_trades = 0

        bh_return = (self.data["Close"].iloc[-1] / self.data["Close"].iloc[0] - 1) * 100

        return {
            "total_return_pct": round(total_return_pct, 2),
            "annual_return_pct": round(annual_return, 2),
            "sharpe_ratio": round(sharpe, 3),
            "max_drawdown_pct": round(max_drawdown, 2),
            "total_trades": total_trades,
            "win_rate_pct": round(win_rate, 1),
            "avg_win_pct": round(avg_win, 2),
            "avg_loss_pct": round(avg_loss, 2),
            "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else "inf",
            "final_equity": round(final_equity, 2),
            "buy_hold_return_pct": round(bh_return, 2),
        }

    def summary(self):
        m = self.metrics
        lines = [
            f"  Total Return:    {m['total_return_pct']:>8.2f}%",
            f"  Annual Return:   {m['annual_return_pct']:>8.2f}%",
            f"  Sharpe Ratio:    {m['sharpe_ratio']:>8.3f}",
            f"  Max Drawdown:    {m['max_drawdown_pct']:>8.2f}%",
            f"  Total Trades:    {m['total_trades']:>8d}",
            f"  Win Rate:        {m['win_rate_pct']:>8.1f}%",
            f"  Avg Win:         {m['avg_win_pct']:>8.2f}%",
            f"  Avg Loss:        {m['avg_loss_pct']:>8.2f}%",
            f"  Profit Factor:   {str(m['profit_factor']):>8s}",
            f"  Final Equity:    ${m['final_equity']:>10.2f}",
            f"  Buy & Hold:      {m['buy_hold_return_pct']:>8.2f}%",
        ]
        return "\n".join(lines)
