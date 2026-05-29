"""
Portfolio Strategy Backtester

Tests advanced multi-stock strategies (Dual Momentum, Ensemble Voting,
Regime Rotation, Composite) across a basket of large-cap stocks.

Walk-forward: optimize params on first 70%, evaluate on last 30%.
"""

import os
import sys
import warnings
import functools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from backtest.data import fetch_data
from backtest.portfolio_engine import PortfolioEngine
from portfolio_strategies import PORTFOLIO_STRATEGIES

warnings.filterwarnings("ignore")

# Diverse large-cap basket available on Trading 212
UNIVERSE = [
    "MSFT", "AAPL", "GOOGL", "AMZN", "META",   # Big tech
    "NVDA", "AVGO",                               # Semiconductors
    "UNH", "LLY", "JNJ",                          # Healthcare
    "JPM", "V", "MA",                              # Financials
    "COST", "WMT", "HD",                           # Consumer
]

INITIAL_CAPITAL = 10000
DATA_PERIOD = "5y"
RESULTS_DIR = "results"


def run():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 70)
    print("  PORTFOLIO STRATEGY BACKTESTER")
    print("=" * 70)
    print(f"\n  Universe: {len(UNIVERSE)} stocks")
    print(f"  Capital: ${INITIAL_CAPITAL:,}")
    print(f"  Period: {DATA_PERIOD}")
    print(f"  Strategies: {len(PORTFOLIO_STRATEGIES)}")

    # Fetch all data
    print(f"\nFetching data...")
    datasets = {}
    for ticker in UNIVERSE:
        try:
            df = fetch_data(ticker, period=DATA_PERIOD)
            if len(df) > 200:
                datasets[ticker] = df
                print(f"  {ticker}: {len(df)} bars")
        except Exception as e:
            print(f"  {ticker}: failed ({e})")

    print(f"\n  {len(datasets)} stocks loaded")

    # Split for walk-forward
    # Find common date range
    all_dates = None
    for df in datasets.values():
        dates = set(df.index)
        all_dates = dates if all_dates is None else all_dates.intersection(dates)
    all_dates = sorted(all_dates)

    split_idx = int(len(all_dates) * 0.7)
    split_date = all_dates[split_idx]

    train_datasets = {t: df[df.index <= split_date] for t, df in datasets.items()}
    test_datasets = {t: df[df.index > split_date] for t, df in datasets.items()}

    # Equal-weight B&H benchmark
    engine = PortfolioEngine(initial_capital=INITIAL_CAPITAL, rebalance_freq="M")

    # Calculate B&H returns
    bh_close = pd.DataFrame({t: df.loc[df.index.isin(all_dates), "Close"]
                             for t, df in datasets.items()})
    test_close = bh_close.loc[bh_close.index > split_date]
    bh_returns = test_close.pct_change().dropna().mean(axis=1)
    bh_total = ((1 + bh_returns).cumprod().iloc[-1] - 1) * 100

    print(f"\n  Train period: {all_dates[0].strftime('%Y-%m-%d')} to {split_date.strftime('%Y-%m-%d')}")
    print(f"  Test period:  {split_date.strftime('%Y-%m-%d')} to {all_dates[-1].strftime('%Y-%m-%d')}")
    print(f"  Equal-weight B&H (test): {bh_total:.1f}%")

    all_results = []

    for strat_name, strat_info in PORTFOLIO_STRATEGIES.items():
        print(f"\n{'='*70}")
        print(f"  {strat_name}")
        print(f"{'='*70}")

        fn = strat_info["fn"]
        best_score = -999
        best_params = None
        best_train_result = None

        # Grid search on training data
        for params in strat_info["params_grid"]:
            weight_fn = functools.partial(fn, **params)
            try:
                result = engine.run(train_datasets, weight_fn)
                m = result.metrics
                score = m["total_return_pct"]

                if score > best_score:
                    best_score = score
                    best_params = params
                    best_train_result = result

                print(f"  Train: {_fmt(params):60s} Ret={m['total_return_pct']:>7.1f}%  "
                      f"Sharpe={m['sharpe_ratio']:>6.3f}  MaxDD={m['max_drawdown_pct']:>7.1f}%")
            except Exception as e:
                print(f"  Train: {_fmt(params):60s} ERROR: {e}")

        if best_params is None:
            print(f"  No valid results")
            continue

        # Evaluate best params on test data
        weight_fn = functools.partial(fn, **best_params)
        try:
            test_result = engine.run(test_datasets, weight_fn)
            tm = test_result.metrics

            # Also run on full data for equity curve
            full_result = engine.run(datasets, weight_fn)
            fm = full_result.metrics
        except Exception as e:
            print(f"  Test failed: {e}")
            continue

        tm_train = best_train_result.metrics
        beats = tm["total_return_pct"] > bh_total

        print(f"\n  Best params: {_fmt(best_params)}")
        print(f"  IN-SAMPLE:   Ret={tm_train['total_return_pct']:>7.1f}%  Sharpe={tm_train['sharpe_ratio']:>6.3f}  MaxDD={tm_train['max_drawdown_pct']:>7.1f}%")
        print(f"  OUT-SAMPLE:  Ret={tm['total_return_pct']:>7.1f}%  Sharpe={tm['sharpe_ratio']:>6.3f}  MaxDD={tm['max_drawdown_pct']:>7.1f}%")
        print(f"  FULL PERIOD: Ret={fm['total_return_pct']:>7.1f}%  Sharpe={fm['sharpe_ratio']:>6.3f}  MaxDD={fm['max_drawdown_pct']:>7.1f}%")
        print(f"  vs B&H:      {'BEATS' if beats else 'loses'} ({tm['total_return_pct']:.1f}% vs {bh_total:.1f}%)")

        all_results.append({
            "name": strat_name,
            "params": best_params,
            "train_metrics": tm_train,
            "test_metrics": tm,
            "full_metrics": fm,
            "full_result": full_result,
            "test_result": test_result,
            "beats_bh": beats,
            "bh_total": bh_total,
        })

    # Final report
    print(f"\n\n{'='*70}")
    print(f"  FINAL RESULTS — PORTFOLIO STRATEGIES")
    print(f"{'='*70}")
    print(f"\n  Universe: {', '.join(sorted(datasets.keys()))}")
    print(f"  Equal-weight B&H (OOS): {bh_total:.1f}%\n")

    all_results.sort(key=lambda x: x["test_metrics"]["total_return_pct"], reverse=True)

    print(f"  {'Strategy':<22} {'OOS Ret%':>9} {'B&H%':>7} {'Sharpe':>7} {'MaxDD%':>8} {'Calmar':>7} {'Rebal':>6}")
    print(f"  {'-'*22} {'-'*9} {'-'*7} {'-'*7} {'-'*8} {'-'*7} {'-'*6}")

    for r in all_results:
        tm = r["test_metrics"]
        mark = " **" if r["beats_bh"] else ""
        print(f"  {r['name']:<22} {tm['total_return_pct']:>8.1f}% {bh_total:>6.1f}% "
              f"{tm['sharpe_ratio']:>7.3f} {tm['max_drawdown_pct']:>8.1f} "
              f"{tm['calmar_ratio']:>7.3f} {tm['rebalances']:>6}{mark}")

    # Detail on best
    if all_results:
        best = all_results[0]
        tm = best["test_metrics"]
        fm = best["full_metrics"]
        print(f"\n  BEST: {best['name']}")
        print(f"  Params: {_fmt(best['params'])}")
        print(f"  OOS Return:     {tm['total_return_pct']:.1f}%")
        print(f"  OOS Sharpe:     {tm['sharpe_ratio']:.3f}")
        print(f"  OOS Max DD:     {tm['max_drawdown_pct']:.1f}%")
        print(f"  OOS Calmar:     {tm['calmar_ratio']:.3f}")
        print(f"  Full Return:    {fm['total_return_pct']:.1f}%")
        print(f"  Full Sharpe:    {fm['sharpe_ratio']:.3f}")
        print(f"  Full Max DD:    {fm['max_drawdown_pct']:.1f}%")

    generate_charts(all_results, datasets, all_dates, split_date)
    save_csv(all_results, bh_total)
    print(f"\n  Results saved to {RESULTS_DIR}/")


def _fmt(params):
    return ", ".join(f"{k}={v}" for k, v in params.items())


def generate_charts(results, datasets, all_dates, split_date):
    if not results:
        return

    # Equity curves
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 1]})
    ax = axes[0]

    # B&H benchmark
    bh_close = pd.DataFrame({t: df.loc[df.index.isin(all_dates), "Close"]
                             for t, df in datasets.items()})
    bh_norm = bh_close.div(bh_close.iloc[0]).mean(axis=1) * INITIAL_CAPITAL
    ax.plot(bh_norm.index, bh_norm, label="Equal-Weight B&H", color="gray",
            linewidth=2.5, linestyle="--", alpha=0.7)

    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]
    for i, r in enumerate(results):
        eq = r["full_result"].equity["equity"]
        fm = r["full_metrics"]
        label = f"{r['name']} ({fm['total_return_pct']:.0f}%, DD:{fm['max_drawdown_pct']:.0f}%)"
        ax.plot(eq.index, eq, label=label, color=colors[i % len(colors)], linewidth=2)

    ax.axvline(x=split_date, color="red", linewidth=1, linestyle=":", alpha=0.7)
    ax.text(split_date, ax.get_ylim()[1] * 0.95, " OOS", color="red", fontsize=9)

    ax.set_title(f"Portfolio Strategies — {len(datasets)} Stocks (${INITIAL_CAPITAL:,} start)", fontsize=14)
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    # Drawdown subplot
    ax2 = axes[1]
    for i, r in enumerate(results):
        eq = r["full_result"].equity["equity"]
        peak = eq.cummax()
        dd = (eq - peak) / peak * 100
        ax2.fill_between(dd.index, dd, alpha=0.2, color=colors[i % len(colors)])
        ax2.plot(dd.index, dd, color=colors[i % len(colors)], linewidth=1,
                 label=r["name"])
    ax2.axvline(x=split_date, color="red", linewidth=1, linestyle=":", alpha=0.7)
    ax2.set_ylabel("Drawdown %")
    ax2.legend(loc="lower left", fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/portfolio_strategies.png", dpi=150)
    plt.close()

    # Allocation over time for best strategy
    best = results[0]
    if best["full_result"].weight_history:
        fig, ax = plt.subplots(figsize=(14, 6))
        wh = best["full_result"].weight_history
        dates = [w["date"] for w in wh]
        tickers = sorted(datasets.keys())

        bottom = np.zeros(len(dates))
        cmap = plt.cm.tab20
        for j, ticker in enumerate(tickers):
            weights = [w["weights"].get(ticker, 0) for w in wh]
            ax.bar(dates, weights, bottom=bottom, width=20,
                   label=ticker, color=cmap(j / len(tickers)), alpha=0.8)
            bottom += np.array(weights)

        cash = [1 - sum(w["weights"].values()) for w in wh]
        ax.bar(dates, cash, bottom=bottom, width=20,
               label="Cash", color="#E0E0E0", alpha=0.8)

        ax.set_title(f"Allocation Over Time — {best['name']}", fontsize=14)
        ax.set_ylabel("Weight")
        ax.legend(loc="upper right", fontsize=7, ncol=4)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{RESULTS_DIR}/portfolio_allocation.png", dpi=150)
        plt.close()


def save_csv(results, bh_total):
    rows = []
    for r in results:
        row = {
            "Strategy": r["name"],
            "Params": _fmt(r["params"]),
            "Beats_BH": r["beats_bh"],
            "BH_pct": bh_total,
        }
        for k, v in r["test_metrics"].items():
            row[f"OOS_{k}"] = v
        for k, v in r["full_metrics"].items():
            row[f"Full_{k}"] = v
        rows.append(row)
    pd.DataFrame(rows).to_csv(f"{RESULTS_DIR}/portfolio_results.csv", index=False)


if __name__ == "__main__":
    run()
