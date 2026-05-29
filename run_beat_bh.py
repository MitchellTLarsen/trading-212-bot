"""
Beat Buy & Hold — TSLA

Runs strategies designed to maximize total return while dodging crashes.
Optimizes for total return (not Sharpe), then checks if OOS return > B&H.
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
from strategies import BH_BEATER_STRATEGIES

warnings.filterwarnings("ignore")

TICKER = "TSLA"
INITIAL_CAPITAL = 10000
DATA_PERIOD = "5y"
RESULTS_DIR = "results"

RISK_CONFIGS = [
    {"name": "No stops",         "trailing_stop_pct": 0,  "fixed_stop_pct": 0,  "take_profit_pct": 0},
    {"name": "Trail 10%",        "trailing_stop_pct": 10, "fixed_stop_pct": 0,  "take_profit_pct": 0},
    {"name": "Trail 15%",        "trailing_stop_pct": 15, "fixed_stop_pct": 0,  "take_profit_pct": 0},
    {"name": "Trail 20%",        "trailing_stop_pct": 20, "fixed_stop_pct": 0,  "take_profit_pct": 0},
    {"name": "Trail 25%",        "trailing_stop_pct": 25, "fixed_stop_pct": 0,  "take_profit_pct": 0},
    {"name": "Trail 30%",        "trailing_stop_pct": 30, "fixed_stop_pct": 0,  "take_profit_pct": 0},
]


def run():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 70)
    print("  BEAT BUY & HOLD — TSLA")
    print("=" * 70)

    print(f"\nFetching {TICKER} data...")
    data = fetch_data(TICKER, period=DATA_PERIOD)
    print(f"  {len(data)} bars ({data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')})")

    # Calculate B&H for full period and OOS period
    split_idx = int(len(data) * 0.7)
    oos_data = data.iloc[split_idx:]
    full_bh = (data["Close"].iloc[-1] / data["Close"].iloc[0] - 1) * 100
    oos_bh = (oos_data["Close"].iloc[-1] / oos_data["Close"].iloc[0] - 1) * 100

    print(f"\n  Full period B&H:  {full_bh:.1f}%")
    print(f"  OOS period B&H:   {oos_bh:.1f}%  (this is the target to beat)")
    print(f"  OOS period: {oos_data.index[0].strftime('%Y-%m-%d')} to {oos_data.index[-1].strftime('%Y-%m-%d')}")
    print(f"\n  Strategies: {len(BH_BEATER_STRATEGIES)}")
    print(f"  Risk configs: {len(RISK_CONFIGS)}")
    print(f"  Optimizing for: total_return_pct")

    all_results = []
    beating_bh = []

    for strategy in BH_BEATER_STRATEGIES:
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
                engine=engine, metric="total_return_pct"  # optimize for return!
            )

            if result is None:
                continue

            om = result["test_metrics"]
            beats = om["total_return_pct"] > oos_bh
            tag = "BEAT" if beats else "    "

            print(f"  {tag} [{rc['name']:<18s}] OOS: Ret={om['total_return_pct']:>7.1f}% vs B&H={oos_bh:.1f}%  "
                  f"Sharpe={om['sharpe_ratio']:>6.3f}  WR={om['win_rate_pct']:>5.1f}%  "
                  f"Trades={om['total_trades']:>3}  MaxDD={om['max_drawdown_pct']:>7.1f}%")

            entry = {
                "strategy_name": strategy.name,
                "risk_config": rc["name"],
                "params": result["best_params"],
                "test_metrics": om,
                "train_metrics": result["train_metrics"],
                "full_result": result["full_result"],
                "beats_bh": beats,
                "rc": rc,
                "oos_bh": oos_bh,
            }
            all_results.append(entry)
            if beats:
                beating_bh.append(entry)

    # Final report
    print(f"\n\n{'='*70}")
    print(f"  RESULTS — BEAT BUY & HOLD")
    print(f"{'='*70}")
    print(f"\n  OOS Buy & Hold: {oos_bh:.1f}%")
    print(f"  Tested: {len(all_results)} combos")
    print(f"  Beating B&H: {len(beating_bh)}\n")

    # Sort all by OOS return
    all_results.sort(key=lambda x: x["test_metrics"]["total_return_pct"], reverse=True)

    print(f"  Top 15 by OOS Return:\n")
    print(f"  {'Strategy':<22} {'Risk Mgmt':<18} {'OOS Ret%':>9} {'B&H%':>7} {'Diff':>7} {'Sharpe':>7} {'WR%':>6} {'MaxDD%':>8} {'Trades':>7}")
    print(f"  {'-'*22} {'-'*18} {'-'*9} {'-'*7} {'-'*7} {'-'*7} {'-'*6} {'-'*8} {'-'*7}")

    for r in all_results[:15]:
        om = r["test_metrics"]
        diff = om["total_return_pct"] - oos_bh
        mark = " **" if r["beats_bh"] else ""
        print(f"  {r['strategy_name']:<22} {r['risk_config']:<18} "
              f"{om['total_return_pct']:>8.1f}% {oos_bh:>6.1f}% {diff:>+6.1f}% "
              f"{om['sharpe_ratio']:>7.3f} {om['win_rate_pct']:>5.1f}% "
              f"{om['max_drawdown_pct']:>8.1f} {om['total_trades']:>7}{mark}")

    if beating_bh:
        best = beating_bh[0] if beating_bh[0]["test_metrics"]["total_return_pct"] == all_results[0]["test_metrics"]["total_return_pct"] else max(beating_bh, key=lambda x: x["test_metrics"]["total_return_pct"])
        om = best["test_metrics"]
        print(f"\n  BEST: {best['strategy_name']} + {best['risk_config']}")
        print(f"  Params: {_fmt(best['params'])}")
        print(f"  OOS Return: {om['total_return_pct']:.1f}% (B&H: {oos_bh:.1f}%, diff: {om['total_return_pct']-oos_bh:+.1f}%)")
        print(f"  Sharpe: {om['sharpe_ratio']:.3f}")
        print(f"  Win Rate: {om['win_rate_pct']:.1f}%")
        print(f"  Max Drawdown: {om['max_drawdown_pct']:.1f}%")

    generate_charts(all_results[:5], data, oos_bh)
    save_csv(all_results, oos_bh)
    print(f"\n  Results saved to {RESULTS_DIR}/")

    return all_results


def _fmt(params):
    return ", ".join(f"{k}={v}" for k, v in params.items())


def generate_charts(top_results, data, oos_bh):
    fig, ax = plt.subplots(figsize=(14, 7))

    bh_equity = INITIAL_CAPITAL * data["Close"] / data["Close"].iloc[0]
    ax.plot(data.index, bh_equity, label=f"Buy & Hold",
            color="gray", linewidth=2.5, linestyle="--", alpha=0.7)

    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0"]
    for i, r in enumerate(top_results[:5]):
        eq = r["full_result"].equity["equity"]
        om = r["test_metrics"]
        label = f"{r['strategy_name']} + {r['risk_config']} ({om['total_return_pct']:.0f}%)"
        ax.plot(eq.index, eq, label=label, color=colors[i], linewidth=1.8)

    # Draw OOS split line
    split_idx = int(len(data) * 0.7)
    split_date = data.index[split_idx]
    ax.axvline(x=split_date, color="red", linewidth=1, linestyle=":", alpha=0.7)
    ax.text(split_date, ax.get_ylim()[1] * 0.95, " OOS -->", color="red", fontsize=9)

    ax.set_title(f"TSLA — Beat Buy & Hold (${INITIAL_CAPITAL:,} start)", fontsize=14)
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/tsla_beat_bh.png", dpi=150)
    plt.close()


def save_csv(results, oos_bh):
    rows = []
    for r in results:
        row = {
            "Strategy": r["strategy_name"],
            "Risk_Config": r["risk_config"],
            "Params": _fmt(r["params"]),
            "Beats_BH": r["beats_bh"],
            "OOS_BH_pct": oos_bh,
        }
        for k, v in r["test_metrics"].items():
            row[f"OOS_{k}"] = v
        rows.append(row)
    pd.DataFrame(rows).to_csv(f"{RESULTS_DIR}/tsla_beat_bh.csv", index=False)


if __name__ == "__main__":
    run()
