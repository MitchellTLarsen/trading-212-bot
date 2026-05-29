"""
Order Flow Strategy Backtester

Tests Smart Money Concepts + order flow strategies on forex.
Tight risk management: 1% risk per trade, low leverage.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from backtest.data import fetch_data
from backtest.forex_engine import ForexEngine
from strategies.orderflow_strategies import FOREX_ORDERFLOW_STRATEGIES

warnings.filterwarnings("ignore")

PAIRS = [
    "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X",
    "EURGBP=X", "NZDUSD=X", "USDCHF=X", "EURJPY=X", "GBPJPY=X",
]

INITIAL_CAPITAL = 10000
RESULTS_DIR = "results"

# Tight drawdown configs — low risk per trade
CONFIGS = [
    {"name": "5d R1% SL0.5% TP1% 5x",   "max_hold_days": 5,  "stop_loss_pct": 0.5, "take_profit_pct": 1.0, "leverage": 5,  "risk_per_trade_pct": 1.0},
    {"name": "7d R1% SL0.7% TP1.4% 5x",  "max_hold_days": 7,  "stop_loss_pct": 0.7, "take_profit_pct": 1.4, "leverage": 5,  "risk_per_trade_pct": 1.0},
    {"name": "10d R1% SL1% TP2% 5x",     "max_hold_days": 10, "stop_loss_pct": 1.0, "take_profit_pct": 2.0, "leverage": 5,  "risk_per_trade_pct": 1.0},
    {"name": "5d R1% SL0.5% TP1% 10x",   "max_hold_days": 5,  "stop_loss_pct": 0.5, "take_profit_pct": 1.0, "leverage": 10, "risk_per_trade_pct": 1.0},
    {"name": "7d R1% SL0.7% TP1.4% 10x", "max_hold_days": 7,  "stop_loss_pct": 0.7, "take_profit_pct": 1.4, "leverage": 10, "risk_per_trade_pct": 1.0},
    {"name": "10d R1% SL1% TP2% 10x",    "max_hold_days": 10, "stop_loss_pct": 1.0, "take_profit_pct": 2.0, "leverage": 10, "risk_per_trade_pct": 1.0},
    {"name": "10d R2% SL1% TP2% 5x",     "max_hold_days": 10, "stop_loss_pct": 1.0, "take_profit_pct": 2.0, "leverage": 5,  "risk_per_trade_pct": 2.0},
    {"name": "10d R2% SL1.5% TP3% 10x",  "max_hold_days": 10, "stop_loss_pct": 1.5, "take_profit_pct": 3.0, "leverage": 10, "risk_per_trade_pct": 2.0},
    {"name": "7d R0.5% SL0.5% TP1% 10x", "max_hold_days": 7,  "stop_loss_pct": 0.5, "take_profit_pct": 1.0, "leverage": 10, "risk_per_trade_pct": 0.5},
    {"name": "10d R0.5% SL1% TP2% 10x",  "max_hold_days": 10, "stop_loss_pct": 1.0, "take_profit_pct": 2.0, "leverage": 10, "risk_per_trade_pct": 0.5},
]


def run():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 70)
    print("  ORDER FLOW STRATEGIES — FOREX")
    print("=" * 70)

    print(f"\nFetching data...")
    datasets = {}
    for pair in PAIRS:
        try:
            df = fetch_data(pair, period="5y")
            datasets[pair] = df
            print(f"  {pair}: {len(df)} bars")
        except Exception as e:
            print(f"  {pair}: failed")

    all_results = []

    for strat_name, strat_info in FOREX_ORDERFLOW_STRATEGIES.items():
        print(f"\n{'='*70}")
        print(f"  {strat_name}")
        print(f"{'='*70}")

        fn = strat_info["fn"]

        for pair, data in datasets.items():
            split_idx = int(len(data) * 0.7)
            train = data.iloc[:split_idx]
            test = data.iloc[split_idx:]

            best_score = -999
            best_entry = None

            for params in strat_info["params_grid"]:
                for fc in CONFIGS:
                    engine = ForexEngine(
                        initial_capital=INITIAL_CAPITAL,
                        max_hold_days=fc["max_hold_days"],
                        stop_loss_pct=fc["stop_loss_pct"],
                        take_profit_pct=fc["take_profit_pct"],
                        leverage=fc["leverage"],
                        risk_per_trade_pct=fc["risk_per_trade_pct"],
                    )

                    try:
                        train_sigs = fn(train, **params)
                        train_result = engine.run(train, train_sigs)
                        tm = train_result.metrics

                        dd = abs(tm["max_drawdown_pct"])
                        if dd > 0 and tm["total_trades"] >= 3:
                            score = tm["total_return_pct"] / dd * (1 + max(tm["sharpe_ratio"], 0))
                        else:
                            continue

                        if score > best_score:
                            test_sigs = fn(test, **params)
                            test_result = engine.run(test, test_sigs)
                            om = test_result.metrics

                            full_sigs = fn(data, **params)
                            full_result = engine.run(data, full_sigs)

                            best_score = score
                            best_entry = {
                                "strategy": strat_name,
                                "pair": pair,
                                "params": params,
                                "config": fc["name"],
                                "train_metrics": tm,
                                "test_metrics": om,
                                "full_result": full_result,
                                "profitable": om["total_return_pct"] > 0 and om["total_trades"] >= 3,
                            }
                    except Exception:
                        continue

            if best_entry:
                om = best_entry["test_metrics"]
                tag = "WIN " if best_entry["profitable"] else "    "
                print(f"  {tag}{pair:12s} Ret={om['total_return_pct']:>7.1f}%  "
                      f"MaxDD={om['max_drawdown_pct']:>6.1f}%  WR={om['win_rate_pct']:>5.1f}%  "
                      f"Trades={om['total_trades']:>3}  Sharpe={om['sharpe_ratio']:>6.3f}  "
                      f"Hold={om['avg_hold_days']:.0f}d  [{best_entry['config']}]")
                all_results.append(best_entry)

    # Report
    print(f"\n\n{'='*70}")
    print(f"  ORDER FLOW RESULTS")
    print(f"{'='*70}\n")

    profitable = [r for r in all_results if r["profitable"]]
    profitable.sort(key=lambda x: abs(x["test_metrics"]["max_drawdown_pct"]))

    print(f"  Profitable: {len(profitable)} / {len(all_results)}\n")

    if profitable:
        print(f"  Ranked by LOWEST DRAWDOWN:\n")
        print(f"  {'Strategy':<24} {'Pair':<12} {'OOS%':>7} {'MaxDD':>7} {'WR%':>6} {'Trds':>5} {'Shrp':>6} {'Hold':>5} {'PF':>6}")
        print(f"  {'-'*24} {'-'*12} {'-'*7} {'-'*7} {'-'*6} {'-'*5} {'-'*6} {'-'*5} {'-'*6}")

        for r in profitable:
            om = r["test_metrics"]
            pf = str(om["profit_factor"])
            print(f"  {r['strategy']:<24} {r['pair']:<12} "
                  f"{om['total_return_pct']:>6.1f}% {om['max_drawdown_pct']:>6.1f}% "
                  f"{om['win_rate_pct']:>5.1f}% {om['total_trades']:>5} "
                  f"{om['sharpe_ratio']:>6.3f} {om['avg_hold_days']:>4.0f}d {pf:>6}")

    # Also show all results
    print(f"\n  All results by return:\n")
    all_results.sort(key=lambda x: x["test_metrics"]["total_return_pct"], reverse=True)
    for r in all_results:
        om = r["test_metrics"]
        tag = "**" if r["profitable"] else ""
        print(f"  {r['strategy']:<24} {r['pair']:<12} "
              f"Ret={om['total_return_pct']:>6.1f}%  DD={om['max_drawdown_pct']:>6.1f}%  "
              f"WR={om['win_rate_pct']:>5.1f}% {tag}")

    generate_charts(profitable[:6] if profitable else all_results[:6])
    save_csv(all_results)
    print(f"\n  Results saved to {RESULTS_DIR}/")


def generate_charts(top):
    if not top:
        return
    fig, ax = plt.subplots(figsize=(14, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, len(top)))
    for i, r in enumerate(top):
        eq = r["full_result"].equity["equity"]
        om = r["test_metrics"]
        label = f"{r['pair'][:6]} {r['strategy'][:15]} ({om['total_return_pct']:.0f}% DD:{om['max_drawdown_pct']:.0f}%)"
        ax.plot(eq.index, eq, label=label, color=colors[i], linewidth=1.5)
    ax.axhline(y=INITIAL_CAPITAL, color="gray", linewidth=1, linestyle="--", alpha=0.5)
    ax.set_title("Order Flow Strategies — Forex", fontsize=14)
    ax.set_ylabel("Equity ($)")
    ax.legend(loc="upper left", fontsize=7)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/orderflow_equity.png", dpi=150)
    plt.close()


def save_csv(results):
    rows = []
    for r in results:
        row = {"Strategy": r["strategy"], "Pair": r["pair"], "Config": r["config"]}
        for k, v in r["test_metrics"].items():
            row[f"OOS_{k}"] = v
        rows.append(row)
    pd.DataFrame(rows).to_csv(f"{RESULTS_DIR}/orderflow_results.csv", index=False)


if __name__ == "__main__":
    run()
