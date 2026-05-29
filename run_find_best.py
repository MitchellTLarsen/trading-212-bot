"""
Find the best stock + strategy combo for minimal drawdowns while beating B&H.

Tests across a range of large-cap stocks with steady trends,
using our top strategies. Optimizes for a composite score that
heavily penalizes drawdowns.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from backtest.data import fetch_data
from backtest.engine import BacktestEngine
from backtest.optimizer import walk_forward_optimize
from strategies import BH_BEATER_STRATEGIES

warnings.filterwarnings("ignore")

# Stable, trending large-caps available on Trading 212
TICKERS = [
    "MSFT",    # Microsoft — steady grower
    "AAPL",    # Apple — consistent
    "COST",    # Costco — very steady uptrend
    "V",       # Visa — payment network
    "MA",      # Mastercard — payment network
    "UNH",     # UnitedHealth — healthcare
    "LLY",     # Eli Lilly — pharma, strong trend
    "AVGO",    # Broadcom — semis, strong trend
    "JPM",     # JPMorgan — financials
    "WMT",     # Walmart — consumer staples
    "HD",      # Home Depot — steady retail
    "NVDA",    # Nvidia — high growth (more volatile)
    "TSLA",    # Tesla — for comparison
]

INITIAL_CAPITAL = 10000
DATA_PERIOD = "5y"
RESULTS_DIR = "results"

RISK_CONFIGS = [
    {"name": "No stops",   "trailing_stop_pct": 0,  "fixed_stop_pct": 0, "take_profit_pct": 0},
    {"name": "Trail 10%",  "trailing_stop_pct": 10, "fixed_stop_pct": 0, "take_profit_pct": 0},
    {"name": "Trail 15%",  "trailing_stop_pct": 15, "fixed_stop_pct": 0, "take_profit_pct": 0},
    {"name": "Trail 20%",  "trailing_stop_pct": 20, "fixed_stop_pct": 0, "take_profit_pct": 0},
]


def composite_score(metrics):
    """
    Score that rewards return while heavily penalizing drawdown.
    Higher is better.
    """
    ret = metrics["total_return_pct"]
    dd = abs(metrics["max_drawdown_pct"])
    sharpe = metrics["sharpe_ratio"]

    if dd == 0:
        return 0

    # Return-to-drawdown ratio, boosted by Sharpe
    return (ret / dd) * (1 + max(sharpe, 0))


def run():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 70)
    print("  FIND BEST STOCK — MINIMAL DRAWDOWN, BEAT B&H")
    print("=" * 70)
    print(f"\n  Stocks: {len(TICKERS)}")
    print(f"  Strategies: {len(BH_BEATER_STRATEGIES)}")
    print(f"  Risk configs: {len(RISK_CONFIGS)}")
    total = len(TICKERS) * len(BH_BEATER_STRATEGIES) * len(RISK_CONFIGS)
    print(f"  Total combos: {total}")

    all_results = []

    for ticker in TICKERS:
        print(f"\n{'='*70}")
        print(f"  {ticker}")
        print(f"{'='*70}")

        try:
            data = fetch_data(ticker, period=DATA_PERIOD)
        except Exception as e:
            print(f"  Failed to fetch: {e}")
            continue

        split_idx = int(len(data) * 0.7)
        oos_data = data.iloc[split_idx:]
        oos_bh = (oos_data["Close"].iloc[-1] / oos_data["Close"].iloc[0] - 1) * 100
        full_bh = (data["Close"].iloc[-1] / data["Close"].iloc[0] - 1) * 100

        print(f"  {len(data)} bars | OOS B&H: {oos_bh:.1f}% | Full B&H: {full_bh:.1f}%")

        for strategy in BH_BEATER_STRATEGIES:
            for rc in RISK_CONFIGS:
                engine = BacktestEngine(
                    initial_capital=INITIAL_CAPITAL,
                    trailing_stop_pct=rc["trailing_stop_pct"],
                    fixed_stop_pct=rc["fixed_stop_pct"],
                    take_profit_pct=rc["take_profit_pct"],
                )

                result = walk_forward_optimize(
                    strategy, data, strategy.param_grid(),
                    engine=engine, metric="total_return_pct"
                )

                if result is None:
                    continue

                om = result["test_metrics"]
                beats = om["total_return_pct"] > oos_bh
                score = composite_score(om)

                entry = {
                    "ticker": ticker,
                    "strategy_name": strategy.name,
                    "risk_config": rc["name"],
                    "params": result["best_params"],
                    "test_metrics": om,
                    "train_metrics": result["train_metrics"],
                    "full_result": result["full_result"],
                    "beats_bh": beats,
                    "oos_bh": oos_bh,
                    "score": score,
                    "data": data,
                }
                all_results.append(entry)

        # Print best for this ticker
        ticker_results = [r for r in all_results if r["ticker"] == ticker]
        if ticker_results:
            best = max(ticker_results, key=lambda x: x["score"])
            om = best["test_metrics"]
            tag = "BEATS B&H" if best["beats_bh"] else ""
            print(f"  Best: {best['strategy_name']} + {best['risk_config']}")
            print(f"    Ret={om['total_return_pct']:.1f}%  MaxDD={om['max_drawdown_pct']:.1f}%  "
                  f"Sharpe={om['sharpe_ratio']:.3f}  WR={om['win_rate_pct']:.1f}%  {tag}")

    # Final report
    print(f"\n\n{'='*70}")
    print(f"  FINAL RANKINGS — MINIMAL DRAWDOWN + BEAT B&H")
    print(f"{'='*70}\n")

    # Filter to only those that beat B&H with positive return
    winners = [r for r in all_results if r["beats_bh"] and r["test_metrics"]["total_return_pct"] > 0]
    winners.sort(key=lambda x: abs(x["test_metrics"]["max_drawdown_pct"]))

    if winners:
        print(f"  {len(winners)} combos beat buy & hold. Ranked by smallest drawdown:\n")
        print(f"  {'Ticker':<7} {'Strategy':<22} {'Risk':<15} {'OOS Ret%':>9} {'B&H%':>7} {'MaxDD%':>8} {'Sharpe':>7} {'WR%':>6} {'Trades':>7}")
        print(f"  {'-'*7} {'-'*22} {'-'*15} {'-'*9} {'-'*7} {'-'*8} {'-'*7} {'-'*6} {'-'*7}")

        for r in winners[:25]:
            om = r["test_metrics"]
            print(f"  {r['ticker']:<7} {r['strategy_name']:<22} {r['risk_config']:<15} "
                  f"{om['total_return_pct']:>8.1f}% {r['oos_bh']:>6.1f}% "
                  f"{om['max_drawdown_pct']:>8.1f} {om['sharpe_ratio']:>7.3f} "
                  f"{om['win_rate_pct']:>5.1f}% {om['total_trades']:>7}")

        # Top pick by composite score (return/drawdown)
        best_score = max(winners, key=lambda x: x["score"])
        om = best_score["test_metrics"]
        print(f"\n  TOP PICK (best return/drawdown ratio):")
        print(f"    {best_score['ticker']} — {best_score['strategy_name']} + {best_score['risk_config']}")
        print(f"    Params: {_fmt(best_score['params'])}")
        print(f"    OOS Return:  {om['total_return_pct']:.1f}% (B&H: {best_score['oos_bh']:.1f}%)")
        print(f"    Max Drawdown: {om['max_drawdown_pct']:.1f}%")
        print(f"    Sharpe:      {om['sharpe_ratio']:.3f}")
        print(f"    Win Rate:    {om['win_rate_pct']:.1f}%")

        # Lowest drawdown winner
        lowest_dd = winners[0]
        om = lowest_dd["test_metrics"]
        print(f"\n  SAFEST (lowest drawdown that still beats B&H):")
        print(f"    {lowest_dd['ticker']} — {lowest_dd['strategy_name']} + {lowest_dd['risk_config']}")
        print(f"    Params: {_fmt(lowest_dd['params'])}")
        print(f"    OOS Return:  {om['total_return_pct']:.1f}% (B&H: {lowest_dd['oos_bh']:.1f}%)")
        print(f"    Max Drawdown: {om['max_drawdown_pct']:.1f}%")
        print(f"    Sharpe:      {om['sharpe_ratio']:.3f}")
        print(f"    Win Rate:    {om['win_rate_pct']:.1f}%")

        generate_charts(winners, all_results)

    else:
        print("  No combos beat buy & hold. Showing top by score:\n")
        all_results.sort(key=lambda x: x["score"], reverse=True)
        for r in all_results[:15]:
            om = r["test_metrics"]
            print(f"  {r['ticker']:<7} {r['strategy_name']:<22} {r['risk_config']:<15} "
                  f"Ret={om['total_return_pct']:>6.1f}%  MaxDD={om['max_drawdown_pct']:>6.1f}%")

    save_csv(all_results)
    print(f"\n  All results saved to {RESULTS_DIR}/")


def _fmt(params):
    return ", ".join(f"{k}={v}" for k, v in params.items())


def generate_charts(winners, all_results):
    # Top 5 winners equity curves
    top5 = sorted(winners, key=lambda x: x["score"], reverse=True)[:5]

    fig, ax = plt.subplots(figsize=(14, 7))
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0"]

    for i, r in enumerate(top5):
        eq = r["full_result"].equity["equity"]
        om = r["test_metrics"]
        data = r["data"]

        # Also plot B&H for this stock
        bh = INITIAL_CAPITAL * data["Close"] / data["Close"].iloc[0]
        ax.plot(bh.index, bh, color=colors[i], linewidth=1, linestyle=":", alpha=0.3)

        label = f"{r['ticker']} {r['strategy_name']} ({om['total_return_pct']:.0f}%, DD:{om['max_drawdown_pct']:.0f}%)"
        ax.plot(eq.index, eq, label=label, color=colors[i], linewidth=2)

    ax.set_title(f"Top 5 — Minimal Drawdown, Beat B&H (${INITIAL_CAPITAL:,} start)", fontsize=14)
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/best_stocks.png", dpi=150)
    plt.close()

    # Drawdown comparison
    fig, ax = plt.subplots(figsize=(14, 5))
    for i, r in enumerate(top5):
        eq = r["full_result"].equity["equity"]
        peak = eq.cummax()
        dd = (eq - peak) / peak * 100
        label = f"{r['ticker']} {r['strategy_name']}"
        ax.fill_between(dd.index, dd, alpha=0.25, color=colors[i])
        ax.plot(dd.index, dd, color=colors[i], linewidth=1, label=label)
    ax.set_title("Drawdown Comparison — Top 5", fontsize=14)
    ax.set_ylabel("Drawdown %")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/best_drawdowns.png", dpi=150)
    plt.close()

    # Scatter: return vs drawdown for ALL results
    fig, ax = plt.subplots(figsize=(10, 7))
    for r in all_results:
        om = r["test_metrics"]
        if om["total_trades"] == 0:
            continue
        color = "#4CAF50" if r["beats_bh"] else "#BDBDBD"
        alpha = 0.8 if r["beats_bh"] else 0.2
        ax.scatter(abs(om["max_drawdown_pct"]), om["total_return_pct"],
                   c=color, alpha=alpha, s=30, edgecolors="white", linewidths=0.3)
        if r in top5:
            ax.annotate(r["ticker"], (abs(om["max_drawdown_pct"]), om["total_return_pct"]),
                        fontsize=8, fontweight="bold")

    ax.set_xlabel("Max Drawdown % (lower = safer)")
    ax.set_ylabel("OOS Return %")
    ax.set_title("Return vs Drawdown — All Combos (green = beats B&H)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/return_vs_drawdown.png", dpi=150)
    plt.close()


def save_csv(results):
    rows = []
    for r in results:
        row = {
            "Ticker": r["ticker"],
            "Strategy": r["strategy_name"],
            "Risk_Config": r["risk_config"],
            "Params": _fmt(r["params"]),
            "Beats_BH": r["beats_bh"],
            "OOS_BH_pct": r["oos_bh"],
            "Score": round(r["score"], 3),
        }
        for k, v in r["test_metrics"].items():
            row[f"OOS_{k}"] = v
        rows.append(row)
    pd.DataFrame(rows).to_csv(f"{RESULTS_DIR}/all_stocks_results.csv", index=False)


if __name__ == "__main__":
    run()
