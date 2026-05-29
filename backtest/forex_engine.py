"""
Forex backtester with long AND short support.

Signals:
    1  = go long
   -1  = go short
    0  = close position / no action

Risk-per-trade position sizing: instead of betting a fixed % of capital,
sizes each position so the stop loss = X% of account. This naturally
caps drawdowns.
"""

import pandas as pd
import numpy as np


class ForexEngine:
    def __init__(self, initial_capital=10000, max_hold_days=10,
                 stop_loss_pct=1.0, take_profit_pct=2.0,
                 spread_pips=1.5, leverage=1, risk_per_trade_pct=2.0):
        self.initial_capital = initial_capital
        self.max_hold_days = max_hold_days
        self.stop_loss_pct = stop_loss_pct / 100
        self.take_profit_pct = take_profit_pct / 100
        self.spread_cost = spread_pips * 0.0001
        self.leverage = leverage
        self.risk_per_trade_pct = risk_per_trade_pct / 100  # max account loss per trade

    def run(self, data, signals):
        df = data.copy()
        df["signal"] = signals

        cash = self.initial_capital
        position = 0
        position_size = 0
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

            if position != 0:
                bars_held += 1

                if position == 1:
                    stop = entry_price * (1 - self.stop_loss_pct)
                    if low <= stop:
                        exit_triggered = True
                        exit_price = stop
                        exit_reason = "stop_loss"

                    if not exit_triggered and self.take_profit_pct > 0:
                        tp = entry_price * (1 + self.take_profit_pct)
                        if high >= tp:
                            exit_triggered = True
                            exit_price = tp
                            exit_reason = "take_profit"

                elif position == -1:
                    stop = entry_price * (1 + self.stop_loss_pct)
                    if high >= stop:
                        exit_triggered = True
                        exit_price = stop
                        exit_reason = "stop_loss"

                    if not exit_triggered and self.take_profit_pct > 0:
                        tp = entry_price * (1 - self.take_profit_pct)
                        if low <= tp:
                            exit_triggered = True
                            exit_price = tp
                            exit_reason = "take_profit"

                if not exit_triggered and bars_held >= self.max_hold_days:
                    exit_triggered = True
                    exit_price = close
                    exit_reason = "time_exit"

                if not exit_triggered and signal != 0 and signal != position:
                    exit_triggered = True
                    exit_price = close
                    exit_reason = "reversal"

            # Execute exit
            if exit_triggered and position != 0:
                if position == 1:
                    pnl_raw = (exit_price / entry_price - 1)
                else:
                    pnl_raw = (entry_price / exit_price - 1)

                pnl = position_size * pnl_raw * self.leverage
                pnl -= position_size * self.spread_cost  # spread cost
                pnl_pct = (pnl / cash) * 100 if cash > 0 else 0

                trades.append({
                    "entry_date": entry_date,
                    "exit_date": date,
                    "direction": "long" if position == 1 else "short",
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "position_size": position_size,
                    "pnl": pnl,
                    "pnl_pct": round(pnl_pct, 4),
                    "bars_held": bars_held,
                    "exit_reason": exit_reason,
                })

                cash += pnl
                if cash < 0:
                    cash = 0
                position = 0
                position_size = 0
                entry_price = 0
                bars_held = 0

            # Enter new position
            if position == 0 and signal != 0 and cash > 0:
                entry_price = close
                entry_date = date
                position = signal
                bars_held = 0

                # Risk-based position sizing:
                # If stop loss = 1% and we risk 2% of account,
                # position size = (account * risk%) / stop_loss%
                # This means if stopped out, we lose exactly risk_per_trade of account
                if self.stop_loss_pct > 0:
                    position_size = min(
                        (cash * self.risk_per_trade_pct) / self.stop_loss_pct,
                        cash * self.leverage  # can't exceed leveraged capital
                    )
                else:
                    position_size = cash * 0.1  # small default if no stop

                cash -= position_size * self.spread_cost

            # Track equity
            unrealized = 0
            if position != 0 and position_size > 0:
                if position == 1:
                    unrealized = position_size * ((close / entry_price - 1) * self.leverage)
                else:
                    unrealized = position_size * ((entry_price / close - 1) * self.leverage)

            equity = cash + (position_size + unrealized if position != 0 else 0)
            equity_curve.append({"date": date, "equity": max(equity, 0)})

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

            # Max consecutive losses
            pnl_signs = (self.trades["pnl"] > 0).astype(int)
            max_consec_loss = 0
            current_streak = 0
            for s in pnl_signs:
                if s == 0:
                    current_streak += 1
                    max_consec_loss = max(max_consec_loss, current_streak)
                else:
                    current_streak = 0
        else:
            win_rate = avg_win = avg_loss = avg_hold = pf = n_trades = 0
            long_wr = short_wr = max_consec_loss = 0

        return {
            "total_return_pct": round(total_ret, 2),
            "annual_return_pct": round(annual, 2),
            "sharpe_ratio": round(sharpe, 3),
            "max_drawdown_pct": round(dd, 2),
            "total_trades": n_trades,
            "win_rate_pct": round(win_rate, 1),
            "avg_win_pct": round(avg_win, 2),
            "avg_loss_pct": round(avg_loss, 2),
            "avg_hold_days": round(avg_hold, 1) if avg_hold else 0,
            "profit_factor": round(pf, 2) if pf != float("inf") else "inf",
            "long_win_rate": round(long_wr, 1),
            "short_win_rate": round(short_wr, 1),
            "max_consec_losses": max_consec_loss,
            "final_equity": round(final, 2),
        }
