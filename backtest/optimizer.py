import itertools
import numpy as np
from backtest.engine import BacktestEngine


def grid_search(strategy, data, param_grid, engine=None, metric="sharpe_ratio",
                verbose=False):
    """
    Exhaustive grid search over parameter combinations.

    Returns list of (params, metrics) sorted by the target metric (descending).
    """
    if engine is None:
        engine = BacktestEngine()

    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(itertools.product(*param_values))

    results = []

    for combo in combinations:
        params = dict(zip(param_names, combo))

        # Skip invalid combos (e.g. fast >= slow period)
        if "fast_period" in params and "slow_period" in params:
            if params["fast_period"] >= params["slow_period"]:
                continue
        if "fast_ema" in params and "slow_ema" in params:
            if params["fast_ema"] >= params["slow_ema"]:
                continue
        if "macd_fast" in params and "macd_slow" in params:
            if params["macd_fast"] >= params["macd_slow"]:
                continue

        try:
            signals = strategy.generate_signals(data, **params)
            result = engine.run(data, signals)
            m = result.metrics

            score = m.get(metric, 0)
            if isinstance(score, str):  # e.g. "inf"
                score = 0

            results.append((params, m, score))

            if verbose:
                print(f"  {params} -> {metric}={score:.3f}")
        except Exception:
            continue

    results.sort(key=lambda x: x[2], reverse=True)
    return results


def walk_forward_optimize(strategy, data, param_grid, train_pct=0.7, engine=None,
                          metric="sharpe_ratio"):
    """
    Walk-forward optimization to prevent overfitting.

    1. Split data into train (first 70%) and test (last 30%)
    2. Grid search best params on training data
    3. Evaluate those params on unseen test data
    4. Return both in-sample and out-of-sample results
    """
    if engine is None:
        engine = BacktestEngine()

    split_idx = int(len(data) * train_pct)
    train_data = data.iloc[:split_idx]
    test_data = data.iloc[split_idx:]

    # Find best params on training data
    train_results = grid_search(strategy, train_data, param_grid, engine, metric)

    if not train_results:
        return None

    best_params, train_metrics, train_score = train_results[0]

    # Evaluate on test data
    test_signals = strategy.generate_signals(test_data, **best_params)
    test_result = engine.run(test_data, test_signals)
    test_metrics = test_result.metrics

    # Also run on full dataset for equity curve
    full_signals = strategy.generate_signals(data, **best_params)
    full_result = engine.run(data, full_signals)

    return {
        "best_params": best_params,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "full_result": full_result,
        "train_score": train_score,
        "test_score": test_metrics.get(metric, 0),
        "combos_tested": len(train_results),
    }
