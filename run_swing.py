"""
Swing Trading Backtester

Tests short-term strategies (hold max 1-2 weeks) across multiple stocks.
Goal: beat buy & hold with rapid-fire swing trades.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from backtest.data import fetch_data
from backtest.swing_engine import SwingEngine
from strategies.swing_strategies import SWING_STRATEGIES

warnings.filterwarnings("ignore")

TICKERS = ["AAPL", "MSFT", "NVDA", "AVGO", "META", "GOOGL", "AMZN",
           "LLY", "JPM", "COST", "V", "WMT", "TSLA"]
INITIAL_CAPITAL = 10000
DATA_PERIOD = "5y"
RESULTS_DIR = "results"

# Holding period + risk configs to test
SWING_CONFIGS = [
    {"name": "5d SL3% TP6%",   "max_hold_days": 5,  "stop_loss_pct": 3,  "take_profit_pct": 6},
    {"name": "5d SL5% TP10%",  "max_hold_days": 5,  "stop_loss_pct": 5,  "take_profit_pct": 10},
    {"name": "5d SL3% noTP",   "max_hold_days": 5,  "stop_loss_pct": 3,  "take_profit_pct": 0},
    {"name": "10d SL3% TP8%",  "max_hold_days": 10, "stop_loss_pct": 3,  "take_profit_pct": 8},
    {"name": "10d SL5% TP10%", "max_hold_days": 10, "stop_loss_pct": 5,  "take_profit_pct": 10},
    {"name": "10d SL5% noTP",  "max_hold_days": 10, "stop_loss_pct": 5,  "take_profit_pct": 0},
    {"name": "7d SL4% TP8%",   "max_hold_days": 7,  "stop_loss_pct": 4,  "take_profit_pct": 8},
    {"name": "3d SL2% TP4%",   "max_hold_days": 3,  "stop_loss_pct": 2,  "take_profit_pct": 4},
]


def run():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 70)
    print("  SWING TRADING BACKTESTER — BEAT B&H IN WEEKS")
    print("=" * 70)
    print(f"\n  Stocks: {len(TICKERS)}")
    print(f"  Strategies: {len(SWING_STRATEGIES)}")
    print(f"  Swing configs: {len(SWING_CONFIGS)}")

    # Fetch data
    print(f"\nFetching data...")
    datasets = {}
    for ticker in TICKERS:
        try:
            df = fetch_data(ticker, period=DATA_PERIOD)
            datasets[ticker] = df
            print(f"  {ticker}: {len(df)} bars")
        except Exception as e:
            print(f"  {ticker}: failed ({e})")

    all_results = []

    for strat_name, strat_info in SWING_STRATEGIES.items():
        print(f"\n{'='*70}")
        print(f"  {strat_name}")
        print(f"{'='*70}")

        fn = strat_info["fn"]

        for ticker, data in datasets.items():
            split_idx = int(len(data) * 0.7)
            train_data = data.iloc[:split_idx]
            test_data = data.iloc[split_idx:]
            oos_bh = (test_data["Close"].iloc[-1] / test_data["Close"].iloc[0] - 1) * 100

            best_score = -999
            best_entry = None

            for params in strat_info["params_grid"]:
                for sc in SWING_CONFIGS:
                    engine = SwingEngine(
                        initial_capital=INITIAL_CAPITAL,
                        max_hold_days=sc["max_hold_days"],
                        stop_loss_pct=sc["stop_loss_pct"],
                        take_profit_pct=sc["take_profit_pct"],
                    )

                    # Train
                    try:
                        train_signals = fn(train_data, **params)
                        train_result = engine.run(train_data, train_signals)
                        tm = train_result.metrics

                        score = tm["total_return_pct"]
                        if score > best_score and tm["total_trades"] >= 3:
                            # Evaluate on test
                            test_signals = fn(test_data, **params)
                            test_result = engine.run(test_data, test_signals)
                            om = test_result.metrics

                            # Full period for equity curve
                            full_signals = fn(data, **params)
                            full_result = engine.run(data, full_signals)

                            best_score = score
                            best_entry = {
                                "strategy": strat_name,
                                "ticker": ticker,
                                "params": params,
                                "swing_config": sc["name"],
                                "sc": sc,
                                "train_metrics": tm,
                                "test_metrics": om,
                                "full_result": full_result,
                                "beats_bh": om["total_return_pct"] > oos_bh,
                                "oos_bh": oos_bh,
                            }
                    except Exception:
                        continue

            if best_entry:
                om = best_entry["test_metrics"]
                tag = "BEAT" if best_entry["beats_bh"] else "    "
                print(f"  {tag} {ticker:<6} [{best_entry['swing_config']:<16}] "
                      f"Ret={om['total_return_pct']:>7.1f}% vs B&H={best_entry['oos_bh']:>5.1f}%  "
                      f"WR={om['win_rate_pct']:>5.1f}%  Trades={om['total_trades']:>3}  "
                      f"AvgHold={om['avg_hold_days']:.0f}d  MaxDD={om['max_drawdown_pct']:>6.1f}%")
                all_results.append(best_entry)

    # Final report
    print(f"\n\n{'='*70}")
    print(f"  SWING TRADING RESULTS")
    print(f"{'='*70}\n")

    winners = [r for r in all_results if r["beats_bh"] and r["test_metrics"]["total_return_pct"] > 0]
    winners.sort(key=lambda x: x["test_metrics"]["total_return_pct"], reverse=True)

    all_results.sort(key=lambda x: x["test_metrics"]["total_return_pct"], reverse=True)

    print(f"  {len(winners)} combos beat buy & hold out-of-sample\n")

    print(f"  {'Strategy':<24} {'Ticker':<7} {'Config':<17} {'OOS%':>7} {'B&H%':>7} {'WR%':>6} {'Trds':>5} {'Hold':>5} {'MaxDD':>7} {'PF':>6}")
    print(f"  {'-'*24} {'-'*7} {'-'*17} {'-'*7} {'-'*7} {'-'*6} {'-'*5} {'-'*5} {'-'*7} {'-'*6}")

    for r in all_results[:30]:
        om = r["test_metrics"]
        mark = " **" if r["beats_bh"] else ""
        pf = str(om["profit_factor"])
        print(f"  {r['strategy']:<24} {r['ticker']:<7} {r['swing_config']:<17} "
              f"{om['total_return_pct']:>6.1f}% {r['oos_bh']:>6.1f}% "
              f"{om['win_rate_pct']:>5.1f}% {om['total_trades']:>5} "
              f"{om['avg_hold_days']:>4.0f}d {om['max_drawdown_pct']:>7.1f} {pf:>6}{mark}")

    if winners:
        best = winners[0]
        om = best["test_metrics"]
        print(f"\n  BEST SWING TRADE:")
        print(f"    {best['strategy']} on {best['ticker']}")
        print(f"    Config: {best['swing_config']}")
        print(f"    Params: {_fmt(best['params'])}")
        print(f"    OOS Return:    {om['total_return_pct']:.1f}% (B&H: {best['oos_bh']:.1f}%)")
        print(f"    Win Rate:      {om['win_rate_pct']:.1f}%")
        print(f"    Avg Hold:      {om['avg_hold_days']:.0f} days")
        print(f"    Total Trades:  {om['total_trades']}")
        print(f"    Max Drawdown:  {om['max_drawdown_pct']:.1f}%")
        print(f"    Profit Factor: {om['profit_factor']}")
        print(f"    Time in Market:{om['time_in_market_pct']:.1f}%")

    generate_charts(all_results[:8], datasets)
    save_csv(all_results)
    print(f"\n  Results saved to {RESULTS_DIR}/")


def _fmt(params):
    return ", ".join(f"{k}={v}" for k, v in params.items())


def generate_charts(top_results, datasets):
    if not top_results:
        return

    fig, ax = plt.subplots(figsize=(14, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, len(top_results)))

    for i, r in enumerate(top_results):
        eq = r["full_result"].equity["equity"]
        om = r["test_metrics"]
        label = f"{r['ticker']} {r['strategy'][:15]} ({om['total_return_pct']:.0f}%)"
        ax.plot(eq.index, eq, label=label, color=colors[i], linewidth=1.5)

        # B&H for comparison
        data = datasets[r["ticker"]]
        bh = INITIAL_CAPITAL * data["Close"] / data["Close"].iloc[0]
        ax.plot(bh.index, bh, color=colors[i], linewidth=0.5, linestyle=":", alpha=0.3)

    ax.set_title(f"Swing Trading — Top Strategies (${INITIAL_CAPITAL:,} start)", fontsize=14)
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend(loc="upper left", fontsize=7)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/swing_equity.png", dpi=150)
    plt.close()


def save_csv(results):
    rows = []
    for r in results:
        row = {
            "Strategy": r["strategy"],
            "Ticker": r["ticker"],
            "Swing_Config": r["swing_config"],
            "Params": _fmt(r["params"]),
            "Beats_BH": r["beats_bh"],
            "OOS_BH": r["oos_bh"],
        }
        for k, v in r["test_metrics"].items():
            row[f"OOS_{k}"] = v
        rows.append(row)
    pd.DataFrame(rows).to_csv(f"{RESULTS_DIR}/swing_results.csv", index=False)


if __name__ == "__main__":
    run()
