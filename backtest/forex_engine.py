"""
Forex backtester with long AND short support.

Signals:
    1  = go long
   -1  = go short
    0  = close position / no action

Supports max holding periods, stop loss, and take profit for both directions.
"""

import pandas as pd
import numpy as np


class ForexEngine:
    def __init__(self, initial_capital=10000, max_hold_days=10,
                 stop_loss_pct=1.0, take_profit_pct=2.0,
                 spread_pips=1.5, leverage=1):
        self.initial_capital = initial_capital
        self.max_hold_days = max_hold_days
        self.stop_loss_pct = stop_loss_pct / 100
        self.take_profit_pct = take_profit_pct / 100
        self.spread_cost = spread_pips * 0.0001  # approximate spread cost
        self.leverage = leverage

    def run(self, data, signals):
        df = data.copy()
        df["signal"] = signals

        cash = self.initial_capital
        position = 0       # +1 = long, -1 = short, 0 = flat
        position_size = 0   # notional value
        entry_price = 0
        entry_date = None
        bars_held = 0

        equity_curve = []
        trades = []

        for i in range(len(df)):
            date = df.index[i]
            close = df["Close"].iloc[i]
            high = df["High"].iloc[i]
            low = df["Low"].iloc[i]
            signal = df["signal"].iloc[i]

            exit_triggered = False
            exit_price = None
            exit_reason = None

            # Check exits on open positions
            if position != 0:
                bars_held += 1

                if position == 1:  # long
                    # Stop loss
                    stop = entry_price * (1 - self.stop_loss_pct)
                    if low <= stop:
                        exit_triggered = True
                        exit_price = stop
                        exit_reason = "stop_loss"

                    # Take profit
                    if not exit_triggered:
                        tp = entry_price * (1 + self.take_profit_pct)
                        if high >= tp:
                            exit_triggered = True
                            exit_price = tp
                            exit_reason = "take_profit"

                elif position == -1:  # short
                    # Stop loss (price goes up)
                    stop = entry_price * (1 + self.stop_loss_pct)
                    if high >= stop:
                        exit_triggered = True
                        exit_price = stop
                        exit_reason = "stop_loss"

                    # Take profit (price goes down)
                    if not exit_triggered:
                        tp = entry_price * (1 - self.take_profit_pct)
                        if low <= tp:
                            exit_triggered = True
                            exit_price = tp
                            exit_reason = "take_profit"

                # Time exit
                if not exit_triggered and bars_held >= self.max_hold_days:
                    exit_triggered = True
                    exit_price = close
                    exit_reason = "time_exit"

                # Signal reversal closes position
                if not exit_triggered and signal != 0 and signal != position:
                    exit_triggered = True
                    exit_price = close
                    exit_reason = "reversal"

            # Execute exit
            if exit_triggered and position != 0:
                if position == 1:
                    pnl_pct = (exit_price / entry_price - 1) * 100 * self.leverage
                else:
                    pnl_pct = (entry_price / exit_price - 1) * 100 * self.leverage

                pnl = position_size * (pnl_pct / 100) - (position_size * self.spread_cost)

                trades.append({
                    "entry_date": entry_date,
                    "exit_date": date,
                    "direction": "long" if position == 1 else "short",
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl": pnl,
                    "pnl_pct": round(pnl_pct, 4),
                    "bars_held": bars_held,
                    "exit_reason": exit_reason,
                })

                cash += pnl
                position = 0
                position_size = 0
                entry_price = 0
                bars_held = 0

            # Enter new position
            if position == 0 and signal != 0:
                exec_price = close
                entry_price = exec_price
                entry_date = date
                position = signal
                position_size = cash * 0.95  # use 95% of capital
                bars_held = 0
                cash -= position_size * self.spread_cost  # pay spread on entry

            # Track equity
            unrealized = 0
            if position != 0:
                if position == 1:
                    unrealized = position_size * ((close / entry_price - 1) * self.leverage)
                else:
                    unrealized = position_size * ((entry_price / close - 1) * self.leverage)

            equity_curve.append({
                "date": date,
                "equity": cash + position_size + unrealized if position != 0 else cash,
            })

        eq_df = pd.DataFrame(equity_curve).set_index("date")
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()

        return ForexResult(eq_df, trades_df, self.initial_capital, data)


class ForexResult:
    def __init__(self, equity, trades, initial_capital, data):
        self.equity = equity
        self.trades = trades
        self.initial_capital = initial_capital
        self.data = data
        self._metrics = None

    @property
    def metrics(self):
        if self._metrics is None:
            self._metrics = self._compute()
        return self._metrics

    def _compute(self):
        eq = self.equity["equity"]
        returns = eq.pct_change().dropna()

        final = eq.iloc[-1]
        total_ret = (final / self.initial_capital - 1) * 100

        days = (eq.index[-1] - eq.index[0]).days
        years = days / 365.25
        annual = ((final / self.initial_capital) ** (1 / years) - 1) * 100 if years > 0 and final > 0 else 0

        sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if len(returns) > 1 and returns.std() > 0 else 0

        peak = eq.cummax()
        dd = ((eq - peak) / peak).min() * 100

        bh = (self.data["Close"].iloc[-1] / self.data["Close"].iloc[0] - 1) * 100

        if len(self.trades) > 0:
            wins = self.trades[self.trades["pnl"] > 0]
            losses = self.trades[self.trades["pnl"] <= 0]
            win_rate = len(wins) / len(self.trades) * 100
            avg_win = wins["pnl_pct"].mean() if len(wins) > 0 else 0
            avg_loss = losses["pnl_pct"].mean() if len(losses) > 0 else 0
            avg_hold = self.trades["bars_held"].mean()
            pf = wins["pnl"].sum() / abs(losses["pnl"].sum()) if len(losses) > 0 and losses["pnl"].sum() != 0 else float("inf")
            n_trades = len(self.trades)

            long_trades = self.trades[self.trades["direction"] == "long"]
            short_trades = self.trades[self.trades["direction"] == "short"]
            long_wr = (len(long_trades[long_trades["pnl"] > 0]) / len(long_trades) * 100) if len(long_trades) > 0 else 0
            short_wr = (len(short_trades[short_trades["pnl"] > 0]) / len(short_trades) * 100) if len(short_trades) > 0 else 0
        else:
            win_rate = avg_win = avg_loss = avg_hold = pf = n_trades = long_wr = short_wr = 0

        return {
            "total_return_pct": round(total_ret, 2),
            "annual_return_pct": round(annual, 2),
            "sharpe_ratio": round(sharpe, 3),
            "max_drawdown_pct": round(dd, 2),
            "total_trades": n_trades,
            "win_rate_pct": round(win_rate, 1),
            "avg_win_pct": round(avg_win, 4),
            "avg_loss_pct": round(avg_loss, 4),
            "avg_hold_days": round(avg_hold, 1) if avg_hold else 0,
            "profit_factor": round(pf, 2) if pf != float("inf") else "inf",
            "long_win_rate": round(long_wr, 1),
            "short_win_rate": round(short_wr, 1),
            "final_equity": round(final, 2),
            "buy_hold_pct": round(bh, 2),
        }
