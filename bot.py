"""
Trading 212 Practice Account Bot

Runs the best backtested strategy (SMA 15/30 Crossover with trailing stop
and take profit) against your Trading 212 practice account.

Usage:
    python bot.py              # Check signals and show recommendations
    python bot.py --execute    # Actually place orders (practice account)
    python bot.py --loop 60    # Run every 60 minutes continuously
"""

import sys
import time
import json
import os
from datetime import datetime

import yfinance as yf
import pandas as pd
import numpy as np

from client import Trading212Client
from backtest.indicators import compute_sma


# --- Strategy Config (from backtesting) ---
TICKER_YF = "TSLA"              # Yahoo Finance ticker for data
TICKER_T212 = "TSLA_US_EQ"     # Trading 212 ticker (check your account)
FAST_SMA = 15
SLOW_SMA = 30
TRAILING_STOP_PCT = 10.0        # Sell if price drops 10% from peak
TAKE_PROFIT_PCT = 20.0          # Sell if price rises 20% from entry
POSITION_SIZE_PCT = 95          # Use 95% of cash for each buy
STATE_FILE = "bot_state.json"


class TradingBot:
    def __init__(self, execute=False):
        self.client = Trading212Client()
        self.execute = execute
        self.state = self._load_state()

    def _load_state(self):
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE) as f:
                return json.load(f)
        return {"entry_price": 0, "highest_since_entry": 0, "in_position": False}

    def _save_state(self):
        with open(STATE_FILE, "w") as f:
            json.dump(self.state, f, indent=2)

    def show_account_summary(self):
        cash = self.client.get_account_cash()
        portfolio = self.client.get_portfolio()

        print("=== Account Summary ===")
        print(f"  Free cash:  {cash.get('free', 'N/A')}")
        print(f"  Total:      {cash.get('total', 'N/A')}")
        print(f"  Invested:   {cash.get('invested', 'N/A')}")
        print(f"  P&L:        {cash.get('ppl', 'N/A')}")
        print()

        if portfolio:
            print("=== Open Positions ===")
            for pos in portfolio:
                ticker = pos.get("ticker", "?")
                qty = pos.get("quantity", 0)
                avg_price = pos.get("averagePrice", 0)
                ppl = pos.get("ppl", 0)
                current = pos.get("currentPrice", 0)
                print(f"  {ticker}: {qty} shares @ {avg_price:.2f} "
                      f"(current: {current:.2f}, P&L: {ppl:.2f})")
        else:
            print("  No open positions.")
        print()

        return cash, portfolio

    def get_signal(self):
        """Fetch recent data and compute current SMA crossover signal."""
        print(f"=== Signal Analysis ({TICKER_YF}) ===")

        df = yf.download(TICKER_YF, period="6mo", interval="1d", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        close = df["Close"]
        fast_sma = compute_sma(close, FAST_SMA)
        slow_sma = compute_sma(close, SLOW_SMA)

        current_price = close.iloc[-1]
        fast_val = fast_sma.iloc[-1]
        slow_val = slow_sma.iloc[-1]
        prev_fast = fast_sma.iloc[-2]
        prev_slow = slow_sma.iloc[-2]

        print(f"  Price:    ${current_price:.2f}")
        print(f"  SMA({FAST_SMA}):  ${fast_val:.2f}")
        print(f"  SMA({SLOW_SMA}):  ${slow_val:.2f}")

        # Determine signal
        currently_bullish = fast_val > slow_val
        was_bullish = prev_fast > prev_slow

        if currently_bullish and not was_bullish:
            signal = "BUY"
        elif not currently_bullish and was_bullish:
            signal = "SELL"
        elif currently_bullish:
            signal = "HOLD_LONG"
        else:
            signal = "HOLD_CASH"

        print(f"  Signal:   {signal}")
        print()

        return signal, current_price

    def check_risk_management(self, current_price):
        """Check trailing stop and take profit levels."""
        if not self.state["in_position"]:
            return None

        entry = self.state["entry_price"]
        highest = max(self.state["highest_since_entry"], current_price)
        self.state["highest_since_entry"] = highest

        trail_stop = highest * (1 - TRAILING_STOP_PCT / 100)
        take_profit = entry * (1 + TAKE_PROFIT_PCT / 100)
        current_pnl = (current_price / entry - 1) * 100

        print(f"=== Risk Management ===")
        print(f"  Entry:         ${entry:.2f}")
        print(f"  Current:       ${current_price:.2f} ({current_pnl:+.1f}%)")
        print(f"  Peak:          ${highest:.2f}")
        print(f"  Trail stop:    ${trail_stop:.2f}")
        print(f"  Take profit:   ${take_profit:.2f}")

        if current_price <= trail_stop:
            print(f"  >>> TRAILING STOP HIT - SELL")
            return "TRAILING_STOP"
        elif current_price >= take_profit:
            print(f"  >>> TAKE PROFIT HIT - SELL")
            return "TAKE_PROFIT"
        else:
            dist_to_stop = (current_price / trail_stop - 1) * 100
            dist_to_tp = (take_profit / current_price - 1) * 100
            print(f"  Distance to stop:  {dist_to_stop:.1f}%")
            print(f"  Distance to TP:    {dist_to_tp:.1f}%")

        print()
        return None

    def run(self):
        """Run one iteration of the strategy."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n{'='*50}")
        print(f"  TSLA Trading Bot — {now}")
        print(f"  Strategy: SMA({FAST_SMA}/{SLOW_SMA}) + Trail {TRAILING_STOP_PCT}% + TP {TAKE_PROFIT_PCT}%")
        print(f"  Mode: {'LIVE EXECUTION' if self.execute else 'SIGNALS ONLY'}")
        print(f"{'='*50}\n")

        cash, portfolio = self.show_account_summary()

        # Check if we have a TSLA position
        has_position = False
        position_qty = 0
        if portfolio:
            for pos in portfolio:
                if TICKER_T212 in pos.get("ticker", ""):
                    has_position = True
                    position_qty = pos.get("quantity", 0)
                    if not self.state["in_position"]:
                        # Sync state with account
                        self.state["in_position"] = True
                        self.state["entry_price"] = pos.get("averagePrice", 0)
                        self.state["highest_since_entry"] = pos.get("currentPrice", 0)

        if not has_position and self.state["in_position"]:
            self.state["in_position"] = False
            self.state["entry_price"] = 0
            self.state["highest_since_entry"] = 0

        # Get signal
        signal, current_price = self.get_signal()

        # Check risk management
        risk_action = self.check_risk_management(current_price)

        # Decision
        action = None

        if risk_action in ("TRAILING_STOP", "TAKE_PROFIT"):
            action = "SELL"
            reason = risk_action
        elif signal == "BUY" and not has_position:
            action = "BUY"
            reason = "SMA crossover buy signal"
        elif signal == "SELL" and has_position:
            action = "SELL"
            reason = "SMA crossover sell signal"
        else:
            reason = "No action needed"

        print(f"=== Decision ===")
        print(f"  Action: {action or 'HOLD'}")
        print(f"  Reason: {reason}")
        print()

        if action == "BUY" and self.execute:
            free_cash = cash.get("free", 0)
            buy_amount = free_cash * (POSITION_SIZE_PCT / 100)
            qty = int(buy_amount / current_price)
            if qty > 0:
                print(f"  Placing market buy: {qty} shares of {TICKER_T212}...")
                try:
                    result = self.client.place_market_order(TICKER_T212, qty)
                    print(f"  Order placed: {result}")
                    self.state["in_position"] = True
                    self.state["entry_price"] = current_price
                    self.state["highest_since_entry"] = current_price
                except Exception as e:
                    print(f"  Order failed: {e}")
            else:
                print(f"  Not enough cash for even 1 share (${free_cash:.2f})")

        elif action == "SELL" and self.execute and position_qty > 0:
            print(f"  Placing market sell: {position_qty} shares of {TICKER_T212}...")
            try:
                result = self.client.place_market_order(TICKER_T212, -position_qty)
                print(f"  Order placed: {result}")
                self.state["in_position"] = False
                self.state["entry_price"] = 0
                self.state["highest_since_entry"] = 0
            except Exception as e:
                print(f"  Order failed: {e}")

        elif action and not self.execute:
            print(f"  (Dry run — use --execute to place real orders)")

        self._save_state()


def main():
    execute = "--execute" in sys.argv

    bot = TradingBot(execute=execute)

    if "--loop" in sys.argv:
        idx = sys.argv.index("--loop")
        interval_min = int(sys.argv[idx + 1]) if idx + 1 < len(sys.argv) else 60
        print(f"Running every {interval_min} minutes. Ctrl+C to stop.\n")
        while True:
            try:
                bot.run()
                print(f"\nSleeping {interval_min} minutes...\n")
                time.sleep(interval_min * 60)
            except KeyboardInterrupt:
                print("\nStopped.")
                break
    else:
        bot.run()


if __name__ == "__main__":
    main()
