"""
Forex Strategy Backtester

Tests 10 strategies across 10 major pairs with multiple risk configs.
Supports long AND short. Walk-forward optimized.
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
from backtest.forex_engine import ForexEngine
from strategies.forex_strategies import FOREX_STRATEGIES

warnings.filterwarnings("ignore")

PAIRS = [
    "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X",
    "EURGBP=X", "NZDUSD=X", "USDCHF=X", "EURJPY=X", "GBPJPY=X",
]

INITIAL_CAPITAL = 10000
DATA_PERIOD = "5y"
RESULTS_DIR = "results"

FOREX_CONFIGS = [
    {"name": "5d SL0.5% TP1%",   "max_hold_days": 5,  "stop_loss_pct": 0.5, "take_profit_pct": 1.0, "leverage": 1},
    {"name": "5d SL1% TP2%",     "max_hold_days": 5,  "stop_loss_pct": 1.0, "take_profit_pct": 2.0, "leverage": 1},
    {"name": "10d SL1% TP2%",    "max_hold_days": 10, "stop_loss_pct": 1.0, "take_profit_pct": 2.0, "leverage": 1},
    {"name": "10d SL1.5% TP3%",  "max_hold_days": 10, "stop_loss_pct": 1.5, "take_profit_pct": 3.0, "leverage": 1},
    {"name": "10d SL1% noTP",    "max_hold_days": 10, "stop_loss_pct": 1.0, "take_profit_pct": 0,   "leverage": 1},
    {"name": "7d SL0.7% TP1.5%", "max_hold_days": 7,  "stop_loss_pct": 0.7, "take_profit_pct": 1.5, "leverage": 1},
    {"name": "5d SL1% TP2% 10x", "max_hold_days": 5,  "stop_loss_pct": 1.0, "take_profit_pct": 2.0, "leverage": 10},
    {"name": "10d SL1% TP2% 10x","max_hold_days": 10, "stop_loss_pct": 1.0, "take_profit_pct": 2.0, "leverage": 10},
    {"name": "7d SL0.7% 1.5% 10x","max_hold_days": 7, "stop_loss_pct": 0.7, "take_profit_pct": 1.5, "leverage": 10},
    {"name": "10d SL1.5% 3% 10x","max_hold_days": 10, "stop_loss_pct": 1.5, "take_profit_pct": 3.0, "leverage": 10},
]


def run():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 70)
    print("  FOREX STRATEGY BACKTESTER")
    print("=" * 70)
    print(f"\n  Pairs: {len(PAIRS)}")
    print(f"  Strategies: {len(FOREX_STRATEGIES)}")
    print(f"  Configs: {len(FOREX_CONFIGS)}")

    # Fetch data
    print(f"\nFetching forex data...")
    datasets = {}
    for pair in PAIRS:
        try:
            df = fetch_data(pair, period=DATA_PERIOD)
            datasets[pair] = df
            print(f"  {pair}: {len(df)} bars")
        except Exception as e:
            print(f"  {pair}: failed ({e})")

    all_results = []

    for strat_name, strat_info in FOREX_STRATEGIES.items():
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
                for fc in FOREX_CONFIGS:
                    engine = ForexEngine(
                        initial_capital=INITIAL_CAPITAL,
                        max_hold_days=fc["max_hold_days"],
                        stop_loss_pct=fc["stop_loss_pct"],
                        take_profit_pct=fc["take_profit_pct"],
                        leverage=fc["leverage"],
                    )

                    try:
                        train_sigs = fn(train, **params)
                        train_result = engine.run(train, train_sigs)
                        tm = train_result.metrics

                        score = tm["total_return_pct"]
                        if score > best_score and tm["total_trades"] >= 5:
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
                                "fc": fc,
                                "train_metrics": tm,
                                "test_metrics": om,
                                "full_result": full_result,
                            }
                    except Exception:
                        continue

            if best_entry:
                om = best_entry["test_metrics"]
                profitable = om["total_return_pct"] > 0 and om["total_trades"] >= 3
                tag = "WIN " if profitable else "    "
                print(f"  {tag}{pair:12s} [{best_entry['config']:<22}] "
                      f"Ret={om['total_return_pct']:>7.1f}%  WR={om['win_rate_pct']:>5.1f}%  "
                      f"Trades={om['total_trades']:>3}  Sharpe={om['sharpe_ratio']:>6.3f}  "
                      f"MaxDD={om['max_drawdown_pct']:>6.1f}%  Hold={om['avg_hold_days']:.0f}d")
                best_entry["profitable"] = profitable
                all_results.append(best_entry)

    # Final report
    print(f"\n\n{'='*70}")
    print(f"  FOREX RESULTS")
    print(f"{'='*70}\n")

    winners = [r for r in all_results if r.get("profitable")]
    winners.sort(key=lambda x: x["test_metrics"]["total_return_pct"], reverse=True)
    all_results.sort(key=lambda x: x["test_metrics"]["total_return_pct"], reverse=True)

    print(f"  Profitable out-of-sample: {len(winners)} / {len(all_results)}\n")

    print(f"  {'Strategy':<22} {'Pair':<12} {'Config':<23} {'OOS%':>7} {'WR%':>6} {'Trds':>5} {'Shrp':>6} {'MaxDD':>7} {'Hold':>5} {'LWR':>5} {'SWR':>5}")
    print(f"  {'-'*22} {'-'*12} {'-'*23} {'-'*7} {'-'*6} {'-'*5} {'-'*6} {'-'*7} {'-'*5} {'-'*5} {'-'*5}")

    for r in all_results[:40]:
        om = r["test_metrics"]
        mark = " **" if r.get("profitable") else ""
        print(f"  {r['strategy']:<22} {r['pair']:<12} {r['config']:<23} "
              f"{om['total_return_pct']:>6.1f}% {om['win_rate_pct']:>5.1f}% "
              f"{om['total_trades']:>5} {om['sharpe_ratio']:>6.3f} "
              f"{om['max_drawdown_pct']:>7.1f} {om['avg_hold_days']:>4.0f}d "
              f"{om['long_win_rate']:>4.0f}% {om['short_win_rate']:>4.0f}%{mark}")

    if winners:
        best = winners[0]
        om = best["test_metrics"]
        print(f"\n  BEST FOREX STRATEGY:")
        print(f"    {best['strategy']} on {best['pair']}")
        print(f"    Config:    {best['config']}")
        print(f"    Params:    {_fmt(best['params'])}")
        print(f"    OOS Return:    {om['total_return_pct']:.1f}%")
        print(f"    Win Rate:      {om['win_rate_pct']:.1f}% (Long: {om['long_win_rate']:.0f}%, Short: {om['short_win_rate']:.0f}%)")
        print(f"    Sharpe:        {om['sharpe_ratio']:.3f}")
        print(f"    Max Drawdown:  {om['max_drawdown_pct']:.1f}%")
        print(f"    Trades:        {om['total_trades']}")
        print(f"    Avg Hold:      {om['avg_hold_days']:.0f} days")
        print(f"    Profit Factor: {om['profit_factor']}")

    generate_charts(all_results[:8])
    save_csv(all_results)
    print(f"\n  Results saved to {RESULTS_DIR}/")


def _fmt(params):
    return ", ".join(f"{k}={v}" for k, v in params.items())


def generate_charts(top):
    if not top:
        return

    fig, ax = plt.subplots(figsize=(14, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, len(top)))

    for i, r in enumerate(top):
        eq = r["full_result"].equity["equity"]
        om = r["test_metrics"]
        label = f"{r['pair'][:6]} {r['strategy'][:12]} ({om['total_return_pct']:.0f}%)"
        ax.plot(eq.index, eq, label=label, color=colors[i], linewidth=1.5)

    ax.axhline(y=INITIAL_CAPITAL, color="gray", linewidth=1, linestyle="--", alpha=0.5)
    ax.set_title(f"Forex — Top Strategies (${INITIAL_CAPITAL:,} start)", fontsize=14)
    ax.set_ylabel("Equity ($)")
    ax.legend(loc="upper left", fontsize=7)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/forex_equity.png", dpi=150)
    plt.close()


def save_csv(results):
    rows = []
    for r in results:
        row = {
            "Strategy": r["strategy"], "Pair": r["pair"],
            "Config": r["config"], "Params": _fmt(r["params"]),
            "Profitable": r.get("profitable", False),
        }
        for k, v in r["test_metrics"].items():
            row[f"OOS_{k}"] = v
        rows.append(row)
    pd.DataFrame(rows).to_csv(f"{RESULTS_DIR}/forex_results.csv", index=False)


if __name__ == "__main__":
    run()
