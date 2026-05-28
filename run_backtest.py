"""
TSLA Strategy Optimizer — Iterates until we find strategies with:
  - Positive win rate (>50%)
  - Positive ROI out-of-sample
  - Decent Sharpe ratio

Tests all strategies with multiple risk management configurations
(trailing stops, fixed stops, take profits).
"""

import os
import sys
import warnings
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from backtest.data import fetch_data
from backtest.engine import BacktestEngine
from backtest.optimizer import walk_forward_optimize
from strategies import ALL_STRATEGIES

warnings.filterwarnings("ignore")

TICKER = "TSLA"
INITIAL_CAPITAL = 10000
DATA_PERIOD = "5y"
RESULTS_DIR = "results"

# Risk management configurations to try with each strategy
RISK_CONFIGS = [
    {"name": "No stops",        "trailing_stop_pct": 0,  "fixed_stop_pct": 0,  "take_profit_pct": 0},
    {"name": "Trail 8%",        "trailing_stop_pct": 8,  "fixed_stop_pct": 0,  "take_profit_pct": 0},
    {"name": "Trail 12%",       "trailing_stop_pct": 12, "fixed_stop_pct": 0,  "take_profit_pct": 0},
    {"name": "Trail 15%",       "trailing_stop_pct": 15, "fixed_stop_pct": 0,  "take_profit_pct": 0},
    {"name": "Trail 20%",       "trailing_stop_pct": 20, "fixed_stop_pct": 0,  "take_profit_pct": 0},
    {"name": "Stop 5% TP 10%",  "trailing_stop_pct": 0,  "fixed_stop_pct": 5,  "take_profit_pct": 10},
    {"name": "Stop 5% TP 15%",  "trailing_stop_pct": 0,  "fixed_stop_pct": 5,  "take_profit_pct": 15},
    {"name": "Stop 7% TP 15%",  "trailing_stop_pct": 0,  "fixed_stop_pct": 7,  "take_profit_pct": 15},
    {"name": "Stop 7% TP 20%",  "trailing_stop_pct": 0,  "fixed_stop_pct": 7,  "take_profit_pct": 20},
    {"name": "Stop 10% TP 25%", "trailing_stop_pct": 0,  "fixed_stop_pct": 10, "take_profit_pct": 25},
    {"name": "Trail 10% TP 20%","trailing_stop_pct": 10, "fixed_stop_pct": 0,  "take_profit_pct": 20},
    {"name": "Trail 12% TP 25%","trailing_stop_pct": 12, "fixed_stop_pct": 0,  "take_profit_pct": 25},
]


def run():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 70)
    print("  TSLA STRATEGY OPTIMIZER — ITERATIVE SEARCH")
    print("=" * 70)
    print(f"\n  Capital:       ${INITIAL_CAPITAL:,}")
    print(f"  Ticker:        {TICKER}")
    print(f"  Period:        {DATA_PERIOD}")
    print(f"  Strategies:    {len(ALL_STRATEGIES)}")
    print(f"  Risk configs:  {len(RISK_CONFIGS)}")
    print(f"  Total combos:  {len(ALL_STRATEGIES) * len(RISK_CONFIGS)}")

    print(f"\nFetching {TICKER} data...")
    data = fetch_data(TICKER, period=DATA_PERIOD)
    print(f"  {len(data)} bars ({data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')})")

    bh_return = (data["Close"].iloc[-1] / data["Close"].iloc[0] - 1) * 100
    print(f"  Buy & Hold: {bh_return:.1f}%\n")

    all_results = []
    passing = []

    for strategy in ALL_STRATEGIES:
        print(f"\n{'='*70}")
        print(f"  {strategy.name}")
        print(f"{'='*70}")

        for rc in RISK_CONFIGS:
            engine = BacktestEngine(
                initial_capital=INITIAL_CAPITAL,
                trailing_stop_pct=rc["trailing_stop_pct"],
                fixed_stop_pct=rc["fixed_stop_pct"],
                take_profit_pct=rc["take_profit_pct"],
            )

            result = walk_forward_optimize(
                strategy, data, strategy.param_grid(),
                engine=engine, metric="sharpe_ratio"
            )

            if result is None:
                continue

            om = result["test_metrics"]
            oos_pass = om["win_rate_pct"] > 50 and om["total_return_pct"] > 0 and om["total_trades"] >= 3

            tag = "PASS" if oos_pass else "    "
            print(f"  {tag} [{rc['name']:<18s}] OOS: Ret={om['total_return_pct']:>7.1f}%  "
                  f"Sharpe={om['sharpe_ratio']:>6.3f}  WR={om['win_rate_pct']:>5.1f}%  "
                  f"Trades={om['total_trades']:>3}  MaxDD={om['max_drawdown_pct']:>7.1f}%")

            entry = {
                "strategy_name": strategy.name,
                "risk_config": rc["name"],
                "params": result["best_params"],
                "test_metrics": om,
                "train_metrics": result["train_metrics"],
                "full_result": result["full_result"],
                "oos_pass": oos_pass,
                "rc": rc,
            }
            all_results.append(entry)
            if oos_pass:
                passing.append(entry)

    # Report
    print(f"\n\n{'='*70}")
    print(f"  FINAL RESULTS — TSLA")
    print(f"{'='*70}")
    print(f"\n  Buy & Hold: {bh_return:.1f}%")
    print(f"  Tested: {len(all_results)} strategy+risk combos")
    print(f"  Passed: {len(passing)} (Win Rate > 50%, Return > 0%, >= 3 trades OOS)\n")

    if passing:
        # Sort by Sharpe
        passing.sort(key=lambda x: x["test_metrics"]["sharpe_ratio"], reverse=True)

        print(f"  {'Strategy':<25} {'Risk Mgmt':<20} {'Ret%':>7} {'Sharpe':>7} {'WR%':>6} {'Trades':>7} {'MaxDD%':>8} {'PF':>6}")
        print(f"  {'-'*25} {'-'*20} {'-'*7} {'-'*7} {'-'*6} {'-'*7} {'-'*8} {'-'*6}")

        for r in passing:
            om = r["test_metrics"]
            pf = str(om["profit_factor"])
            print(f"  {r['strategy_name']:<25} {r['risk_config']:<20} "
                  f"{om['total_return_pct']:>6.1f}% {om['sharpe_ratio']:>7.3f} "
                  f"{om['win_rate_pct']:>5.1f}% {om['total_trades']:>7} "
                  f"{om['max_drawdown_pct']:>8.1f} {pf:>6}")

        # Top pick
        best = passing[0]
        print(f"\n  TOP PICK: {best['strategy_name']} + {best['risk_config']}")
        print(f"  Params: {_fmt(best['params'])}")
        print(f"  OOS Return: {best['test_metrics']['total_return_pct']:.1f}%")
        print(f"  OOS Sharpe: {best['test_metrics']['sharpe_ratio']:.3f}")
        print(f"  OOS Win Rate: {best['test_metrics']['win_rate_pct']:.1f}%")
        print(f"  OOS Max Drawdown: {best['test_metrics']['max_drawdown_pct']:.1f}%")

        generate_charts(passing[:5], all_results, data, bh_return)
    else:
        print("  No strategies passed all criteria.")
        # Show top 5 by Sharpe anyway
        all_results.sort(key=lambda x: x["test_metrics"]["sharpe_ratio"], reverse=True)
        print(f"\n  Top 5 by Sharpe (didn't fully pass):\n")
        for r in all_results[:5]:
            om = r["test_metrics"]
            print(f"  {r['strategy_name']:<25} {r['risk_config']:<20} "
                  f"Ret={om['total_return_pct']:>6.1f}%  Sharpe={om['sharpe_ratio']:>6.3f}  "
                  f"WR={om['win_rate_pct']:>5.1f}%  Trades={om['total_trades']}")

        generate_charts(all_results[:5], all_results, data, bh_return)

    save_csv(all_results)
    print(f"\n  Results saved to {RESULTS_DIR}/")

    return passing, all_results


def _fmt(params):
    return ", ".join(f"{k}={v}" for k, v in params.items())


def generate_charts(top_results, all_results, data, bh_return):
    # Equity curves for top strategies
    fig, ax = plt.subplots(figsize=(14, 7))

    bh_equity = INITIAL_CAPITAL * data["Close"] / data["Close"].iloc[0]
    ax.plot(data.index, bh_equity, label=f"Buy & Hold ({bh_return:.0f}%)",
            color="gray", linewidth=2.5, linestyle="--", alpha=0.7)

    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0"]
    for i, r in enumerate(top_results[:5]):
        eq = r["full_result"].equity["equity"]
        om = r["test_metrics"]
        label = f"{r['strategy_name']} + {r['risk_config']} ({om['total_return_pct']:.0f}%)"
        ax.plot(eq.index, eq, label=label, color=colors[i], linewidth=1.8)

    ax.set_title(f"TSLA — Top Strategy Equity Curves (${INITIAL_CAPITAL:,} start)", fontsize=14)
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/tsla_top_equity.png", dpi=150)
    plt.close()

    # Drawdown chart
    fig, ax = plt.subplots(figsize=(14, 4))
    for i, r in enumerate(top_results[:5]):
        eq = r["full_result"].equity["equity"]
        peak = eq.cummax()
        dd = (eq - peak) / peak * 100
        label = f"{r['strategy_name']} + {r['risk_config']}"
        ax.fill_between(dd.index, dd, alpha=0.25, color=colors[i], label=label)
        ax.plot(dd.index, dd, color=colors[i], alpha=0.6, linewidth=0.5)
    ax.set_title("TSLA — Drawdown Comparison", fontsize=14)
    ax.set_ylabel("Drawdown %")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/tsla_drawdowns.png", dpi=150)
    plt.close()

    # Win rate vs return scatter for all results
    fig, ax = plt.subplots(figsize=(10, 7))
    for r in all_results:
        om = r["test_metrics"]
        if om["total_trades"] == 0:
            continue
        color = "#4CAF50" if r["oos_pass"] else "#9E9E9E"
        alpha = 0.9 if r["oos_pass"] else 0.3
        ax.scatter(om["win_rate_pct"], om["total_return_pct"], c=color,
                   alpha=alpha, s=40, edgecolors="white", linewidths=0.5)
        if r["oos_pass"]:
            ax.annotate(r["strategy_name"][:10], (om["win_rate_pct"], om["total_return_pct"]),
                        fontsize=7, alpha=0.8)

    ax.axhline(y=0, color="red", linewidth=0.5, linestyle="--")
    ax.axvline(x=50, color="red", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Win Rate %")
    ax.set_ylabel("OOS Return %")
    ax.set_title("TSLA — All Strategy+Risk Combos (green = passed)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/tsla_scatter.png", dpi=150)
    plt.close()


def save_csv(results):
    rows = []
    for r in results:
        row = {
            "Strategy": r["strategy_name"],
            "Risk_Config": r["risk_config"],
            "Params": _fmt(r["params"]),
            "Pass": r["oos_pass"],
        }
        for k, v in r["test_metrics"].items():
            row[f"OOS_{k}"] = v
        rows.append(row)
    pd.DataFrame(rows).to_csv(f"{RESULTS_DIR}/tsla_all_results.csv", index=False)


if __name__ == "__main__":
    run()
