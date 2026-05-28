"""
Trading 212 Practice Account Bot

A starting point for building trading strategies.
Modify the `run_strategy` method to implement your own logic.
"""

from client import Trading212Client


class TradingBot:
    def __init__(self):
        self.client = Trading212Client()

    def show_account_summary(self):
        cash = self.client.get_account_cash()
        portfolio = self.client.get_portfolio()

        print("=== Account Summary ===")
        print(f"Free cash:  {cash.get('free', 'N/A')}")
        print(f"Total:      {cash.get('total', 'N/A')}")
        print(f"Invested:   {cash.get('invested', 'N/A')}")
        print(f"P&L:        {cash.get('ppl', 'N/A')}")
        print()

        if portfolio:
            print("=== Open Positions ===")
            for pos in portfolio:
                ticker = pos.get("ticker", "?")
                qty = pos.get("quantity", 0)
                avg_price = pos.get("averagePrice", 0)
                ppl = pos.get("ppl", 0)
                print(f"  {ticker}: {qty} shares @ {avg_price:.2f} (P&L: {ppl:.2f})")
        else:
            print("No open positions.")
        print()

    def run_strategy(self):
        """
        Implement your trading strategy here.

        This is intentionally left minimal — decide what you want to trade
        and when, then use self.client to place orders.

        Example ideas:
        - Simple moving average crossover
        - Mean reversion on specific tickers
        - Momentum / breakout detection
        - Scheduled rebalancing
        """
        print("No strategy implemented yet. Edit bot.py to add yours.")


def main():
    bot = TradingBot()

    # Show what's in the account
    bot.show_account_summary()

    # Run your strategy
    bot.run_strategy()


if __name__ == "__main__":
    main()
