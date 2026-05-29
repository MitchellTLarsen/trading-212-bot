"""
Swing trading backtester with fixed holding periods.

Unlike the signal-based engine, this one:
- Enters on buy signals
- Exits after a fixed number of days OR on a stop/target hit
- Can take multiple sequential trades (re-enters on next signal after exit)
"""

import pandas as pd
import numpy as np


class SwingEngine:
    """
    Backtester for swing trades with max holding periods.
    """

    def __init__(self, initial_capital=10000, max_hold_days=10,
                 stop_loss_pct=5, take_profit_pct=0,
                 slippage_pct=0.05):
        self.initial_capital = initial_capital
        self.max_hold_days = max_hold_days
        self.stop_loss_pct = stop_loss_pct / 100
        self.take_profit_pct = take_profit_pct / 100 if take_profit_pct > 0 else None
        self.slippage_pct = slippage_pct / 100

    def run(self, data, entry_signals):
        """
        data: OHLCV DataFrame
        entry_signals: Series with 1 = enter long, 0 = no signal

        Exits are automatic: max hold days, stop loss, or take profit.
        """
        df = data.copy()
        df["signal"] = entry_signals

        cash = self.initial_capital
        shares = 0
        equity_curve = []
        trades = []
        entry_price = 0
        entry_date = None
        bars_held = 0

        for i in range(len(df)):
            date = df.index[i]
            close = df["Close"].iloc[i]
            high = df["High"].iloc[i]
            low = df["Low"].iloc[i]
            signal = df["signal"].iloc[i]

            exit_triggered = False
            exit_price = None
            exit_reason = None

            if shares > 0:
                bars_held += 1

                # Check stop loss (intrabar)
                if self.stop_loss_pct > 0:
                    stop_level = entry_price * (1 - self.stop_loss_pct)
                    if low <= stop_level:
                        exit_triggered = True
                        exit_price = stop_level
                        exit_reason = "stop_loss"

                # Check take profit (intrabar)
                if self.take_profit_pct and not exit_triggered:
                    tp_level = entry_price * (1 + self.take_profit_pct)
                    if high >= tp_level:
                        exit_triggered = True
                        exit_price = tp_level
                        exit_reason = "take_profit"

                # Check max hold period
                if not exit_triggered and bars_held >= self.max_hold_days:
                    exit_triggered = True
                    exit_price = close
                    exit_reason = "time_exit"

            # Execute exit
            if exit_triggered and shares > 0:
                sell_price = exit_price * (1 - self.slippage_pct)
                revenue = shares * sell_price
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
                    "bars_held": bars_held,
                    "exit_reason": exit_reason,
                })

                cash += revenue
                shares = 0
                entry_price = 0
                entry_date = None
                bars_held = 0

            # Check entry
            if signal == 1 and shares == 0:
                if i + 1 < len(df):
                    exec_price = df["Open"].iloc[i + 1] * (1 + self.slippage_pct)
                else:
                    exec_price = close * (1 + self.slippage_pct)

                shares = int(cash / exec_price)
                if shares > 0:
                    cash -= shares * exec_price
                    entry_price = exec_price
                    entry_date = date
                    bars_held = 0

            port_value = cash + shares * close
            equity_curve.append({"date": date, "equity": port_value})

        eq_df = pd.DataFrame(equity_curve).set_index("date")
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()

        return SwingResult(eq_df, trades_df, self.initial_capital, data)


class SwingResult:
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
        annual = ((final / self.initial_capital) ** (1 / years) - 1) * 100 if years > 0 else 0

        sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0

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

            # Time in market
            total_bars_held = self.trades["bars_held"].sum()
            time_in_market = total_bars_held / len(self.data) * 100
        else:
            win_rate = avg_win = avg_loss = avg_hold = pf = n_trades = time_in_market = 0

        return {
            "total_return_pct": round(total_ret, 2),
            "annual_return_pct": round(annual, 2),
            "sharpe_ratio": round(sharpe, 3),
            "max_drawdown_pct": round(dd, 2),
            "total_trades": n_trades,
            "win_rate_pct": round(win_rate, 1),
            "avg_win_pct": round(avg_win, 2),
            "avg_loss_pct": round(avg_loss, 2),
            "avg_hold_days": round(avg_hold, 1),
            "profit_factor": round(pf, 2) if pf != float("inf") else "inf",
            "time_in_market_pct": round(time_in_market, 1),
            "final_equity": round(final, 2),
            "buy_hold_pct": round(bh, 2),
        }
