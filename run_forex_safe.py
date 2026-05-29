"""
Forex — Minimal Drawdown Edition

Risk-per-trade sizing: each trade risks only 1-2% of account.
If stopped out, you lose AT MOST that percentage.
This caps drawdowns while still allowing good returns via compounding.
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
from strategies.forex_strategies import FOREX_STRATEGIES

warnings.filterwarnings("ignore")

PAIRS = [
    "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X",
    "EURGBP=X", "NZDUSD=X", "USDCHF=X", "EURJPY=X", "GBPJPY=X",
]

INITIAL_CAPITAL = 10000
RESULTS_DIR = "results"

# Conservative configs — risk 1-2% per trade, various leverage
SAFE_CONFIGS = [
    {"name": "5d R1% SL0.5% TP1% 5x",  "max_hold_days": 5,  "stop_loss_pct": 0.5, "take_profit_pct": 1.0, "leverage": 5,  "risk_per_trade_pct": 1.0},
    {"name": "5d R1% SL0.5% TP1% 10x", "max_hold_days": 5,  "stop_loss_pct": 0.5, "take_profit_pct": 1.0, "leverage": 10, "risk_per_trade_pct": 1.0},
    {"name": "5d R2% SL0.5% TP1% 10x", "max_hold_days": 5,  "stop_loss_pct": 0.5, "take_profit_pct": 1.0, "leverage": 10, "risk_per_trade_pct": 2.0},
    {"name": "7d R1% SL0.7% TP1.5% 10x","max_hold_days": 7,  "stop_loss_pct": 0.7, "take_profit_pct": 1.5, "leverage": 10, "risk_per_trade_pct": 1.0},
    {"name": "7d R2% SL0.7% TP1.5% 10x","max_hold_days": 7,  "stop_loss_pct": 0.7, "take_profit_pct": 1.5, "leverage": 10, "risk_per_trade_pct": 2.0},
    {"name": "10d R1% SL1% TP2% 10x",  "max_hold_days": 10, "stop_loss_pct": 1.0, "take_profit_pct": 2.0, "leverage": 10, "risk_per_trade_pct": 1.0},
    {"name": "10d R2% SL1% TP2% 10x",  "max_hold_days": 10, "stop_loss_pct": 1.0, "take_profit_pct": 2.0, "leverage": 10, "risk_per_trade_pct": 2.0},
    {"name": "10d R1% SL1.5% TP3% 10x","max_hold_days": 10, "stop_loss_pct": 1.5, "take_profit_pct": 3.0, "leverage": 10, "risk_per_trade_pct": 1.0},
    {"name": "10d R2% SL1.5% TP3% 10x","max_hold_days": 10, "stop_loss_pct": 1.5, "take_profit_pct": 3.0, "leverage": 10, "risk_per_trade_pct": 2.0},
    {"name": "10d R1% SL1% noTP 10x",  "max_hold_days": 10, "stop_loss_pct": 1.0, "take_profit_pct": 0,   "leverage": 10, "risk_per_trade_pct": 1.0},
    {"name": "5d R1% SL1% TP2% 20x",   "max_hold_days": 5,  "stop_loss_pct": 1.0, "take_profit_pct": 2.0, "leverage": 20, "risk_per_trade_pct": 1.0},
    {"name": "10d R0.5% SL1% TP2% 10x","max_hold_days": 10, "stop_loss_pct": 1.0, "take_profit_pct": 2.0, "leverage": 10, "risk_per_trade_pct": 0.5},
]


def run():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 70)
    print("  FOREX — MINIMAL DRAWDOWN")
    print("=" * 70)
    print(f"  Risk-per-trade position sizing (max loss per trade = 1-2% of account)")

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
                for fc in SAFE_CONFIGS:
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

                        # Score: reward return, heavily penalize drawdown
                        dd = abs(tm["max_drawdown_pct"])
                        if dd > 0 and tm["total_trades"] >= 5:
                            score = tm["total_return_pct"] / dd * (1 + max(tm["sharpe_ratio"], 0))
                        else:
                            score = -999

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
                print(f"  {tag}{pair:12s} Ret={om['total_return_pct']:>7.1f}%  "
                      f"MaxDD={om['max_drawdown_pct']:>6.1f}%  WR={om['win_rate_pct']:>5.1f}%  "
                      f"Trades={om['total_trades']:>3}  Sharpe={om['sharpe_ratio']:>6.3f}  "
                      f"Hold={om['avg_hold_days']:.0f}d  MaxLoss={om['max_consec_losses']}")
                best_entry["profitable"] = profitable
                all_results.append(best_entry)

    # Report
    print(f"\n\n{'='*70}")
    print(f"  RESULTS — MINIMAL DRAWDOWN FOREX")
    print(f"{'='*70}\n")

    profitable = [r for r in all_results if r.get("profitable")]
    profitable.sort(key=lambda x: x["test_metrics"]["total_return_pct"], reverse=True)
    all_results.sort(key=lambda x: x["test_metrics"]["total_return_pct"], reverse=True)

    print(f"  Profitable: {len(profitable)} / {len(all_results)}")

    # Show profitable sorted by lowest drawdown
    if profitable:
        by_dd = sorted(profitable, key=lambda x: abs(x["test_metrics"]["max_drawdown_pct"]))

        print(f"\n  Ranked by LOWEST DRAWDOWN (profitable only):\n")
        print(f"  {'Strategy':<22} {'Pair':<12} {'OOS%':>7} {'MaxDD':>7} {'WR%':>6} {'Trds':>5} {'Shrp':>6} {'Hold':>5} {'AvgW':>6} {'AvgL':>7} {'PF':>6}")
        print(f"  {'-'*22} {'-'*12} {'-'*7} {'-'*7} {'-'*6} {'-'*5} {'-'*6} {'-'*5} {'-'*6} {'-'*7} {'-'*6}")

        for r in by_dd[:30]:
            om = r["test_metrics"]
            pf = str(om["profit_factor"])
            print(f"  {r['strategy']:<22} {r['pair']:<12} "
                  f"{om['total_return_pct']:>6.1f}% {om['max_drawdown_pct']:>6.1f}% "
                  f"{om['win_rate_pct']:>5.1f}% {om['total_trades']:>5} "
                  f"{om['sharpe_ratio']:>6.3f} {om['avg_hold_days']:>4.0f}d "
                  f"{om['avg_win_pct']:>5.1f}% {om['avg_loss_pct']:>6.1f}% {pf:>6}")

        # Best by return/drawdown ratio
        best_ratio = max(profitable, key=lambda x: x["test_metrics"]["total_return_pct"] / max(abs(x["test_metrics"]["max_drawdown_pct"]), 0.1))
        om = best_ratio["test_metrics"]
        ratio = om["total_return_pct"] / abs(om["max_drawdown_pct"])
        print(f"\n  BEST RETURN/DRAWDOWN RATIO:")
        print(f"    {best_ratio['strategy']} on {best_ratio['pair']}")
        print(f"    Config:    {best_ratio['config']}")
        print(f"    Params:    {_fmt(best_ratio['params'])}")
        print(f"    Return:    {om['total_return_pct']:.1f}%")
        print(f"    Max DD:    {om['max_drawdown_pct']:.1f}%")
        print(f"    Ratio:     {ratio:.2f}x (earned {ratio:.1f}% for every 1% of drawdown)")
        print(f"    Win Rate:  {om['win_rate_pct']:.1f}%")
        print(f"    Sharpe:    {om['sharpe_ratio']:.3f}")
        print(f"    Trades:    {om['total_trades']}")
        print(f"    PF:        {om['profit_factor']}")

        # Lowest drawdown winner
        safest = by_dd[0]
        om = safest["test_metrics"]
        print(f"\n  SAFEST (lowest drawdown):")
        print(f"    {safest['strategy']} on {safest['pair']}")
        print(f"    Config:    {safest['config']}")
        print(f"    Params:    {_fmt(safest['params'])}")
        print(f"    Return:    {om['total_return_pct']:.1f}%")
        print(f"    Max DD:    {om['max_drawdown_pct']:.1f}%")
        print(f"    Win Rate:  {om['win_rate_pct']:.1f}%")
        print(f"    Sharpe:    {om['sharpe_ratio']:.3f}")
        print(f"    Trades:    {om['total_trades']}")

        # Highest return
        best_ret = profitable[0]
        om = best_ret["test_metrics"]
        print(f"\n  HIGHEST RETURN:")
        print(f"    {best_ret['strategy']} on {best_ret['pair']}")
        print(f"    Config:    {best_ret['config']}")
        print(f"    Params:    {_fmt(best_ret['params'])}")
        print(f"    Return:    {om['total_return_pct']:.1f}%")
        print(f"    Max DD:    {om['max_drawdown_pct']:.1f}%")
        print(f"    Win Rate:  {om['win_rate_pct']:.1f}%")
        print(f"    Sharpe:    {om['sharpe_ratio']:.3f}")
        print(f"    Trades:    {om['total_trades']}")

    generate_charts(profitable[:8] if profitable else all_results[:8])
    save_csv(all_results)
    print(f"\n  Results saved to {RESULTS_DIR}/")


def _fmt(params):
    return ", ".join(f"{k}={v}" for k, v in params.items())


def generate_charts(top):
    if not top:
        return

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 1]})

    colors = plt.cm.tab10(np.linspace(0, 1, min(len(top), 8)))
    for i, r in enumerate(top[:8]):
        eq = r["full_result"].equity["equity"]
        om = r["test_metrics"]
        label = f"{r['pair'][:6]} {r['strategy'][:12]} (Ret:{om['total_return_pct']:.0f}% DD:{om['max_drawdown_pct']:.0f}%)"
        axes[0].plot(eq.index, eq, label=label, color=colors[i], linewidth=1.5)

    axes[0].axhline(y=INITIAL_CAPITAL, color="gray", linewidth=1, linestyle="--", alpha=0.5)
    axes[0].set_title("Forex — Minimal Drawdown Strategies", fontsize=14)
    axes[0].set_ylabel("Equity ($)")
    axes[0].legend(loc="upper left", fontsize=7)
    axes[0].grid(True, alpha=0.3)

    for i, r in enumerate(top[:8]):
        eq = r["full_result"].equity["equity"]
        peak = eq.cummax()
        dd = (eq - peak) / peak * 100
        axes[1].plot(dd.index, dd, color=colors[i], linewidth=1, alpha=0.7)

    axes[1].set_ylabel("Drawdown %")
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/forex_safe.png", dpi=150)
    plt.close()


def save_csv(results):
    rows = []
    for r in results:
        row = {
            "Strategy": r["strategy"], "Pair": r["pair"],
            "Config": r["config"], "Params": _fmt(r["params"]),
        }
        for k, v in r["test_metrics"].items():
            row[f"OOS_{k}"] = v
        rows.append(row)
    pd.DataFrame(rows).to_csv(f"{RESULTS_DIR}/forex_safe_results.csv", index=False)


if __name__ == "__main__":
    run()
