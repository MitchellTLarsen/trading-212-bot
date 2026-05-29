"""
Advanced portfolio-level trading strategies.

Each strategy is a function that takes (date, datasets, price_history)
and returns target weights {ticker: weight} that sum to <= 1.0.

All strategies use inverse-volatility position sizing and maintain
a minimum allocation floor to avoid sitting in cash during bull markets.
"""

import numpy as np
import pandas as pd
from backtest.indicators import compute_sma, compute_ema, compute_rsi


def _inv_vol_weights(selected, vols, max_alloc=0.95):
    """Inverse-volatility weighting for selected tickers."""
    if not selected:
        return {}
    inv_vols = {t: 1.0 / max(vols.get(t, 0.3), 0.01) for t in selected}
    total = sum(inv_vols.values())
    return {t: (v / total) * max_alloc for t, v in inv_vols.items()}


def _get_vol(prices, window=20):
    """Annualized volatility from price series."""
    ret = prices.pct_change().dropna()
    if len(ret) < window:
        return 0.3
    return ret.iloc[-window:].std() * np.sqrt(252)


# ==========================================================================
#  1. DUAL MOMENTUM (Antonacci-style)
# ==========================================================================

def dual_momentum(date, datasets, price_history,
                  abs_lookback=252, rel_lookback=126,
                  top_n=3, vol_adjust=True, min_alloc=0.5):
    """
    Dual Momentum with minimum allocation floor.

    When absolute momentum filters out too many stocks, falls back to
    top relative momentum stocks with reduced allocation.
    """
    scores = {}
    abs_positive = {}
    vols = {}

    for ticker in datasets:
        if ticker not in price_history.columns:
            continue
        prices = price_history[ticker].dropna()
        if len(prices) < abs_lookback:
            continue

        current = prices.iloc[-1]
        vols[ticker] = _get_vol(prices)

        # Absolute momentum
        abs_start = prices.iloc[-abs_lookback]
        abs_mom = (current / abs_start) - 1
        abs_positive[ticker] = abs_mom > 0

        # Relative momentum
        rel_start = prices.iloc[-rel_lookback] if len(prices) >= rel_lookback else prices.iloc[0]
        scores[ticker] = (current / rel_start) - 1

    if not scores:
        return {}

    # Primary: stocks with positive absolute momentum
    candidates = {t: s for t, s in scores.items() if abs_positive.get(t, False)}

    if len(candidates) >= top_n:
        ranked = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
        selected = [t for t, _ in ranked[:top_n]]
        return _inv_vol_weights(selected, vols, 0.95)

    # Fallback: not enough with positive abs momentum
    # Use top relative momentum with reduced allocation
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    selected = [t for t, _ in ranked[:top_n] if scores[t] > -0.05]  # not deeply negative

    if selected:
        alloc = max(min_alloc, 0.95 * (len(candidates) / top_n))
        return _inv_vol_weights(selected, vols, alloc)

    return {}


# ==========================================================================
#  2. ENSEMBLE VOTING
# ==========================================================================

def ensemble_voting(date, datasets, price_history,
                    min_votes=3, top_n=4, min_alloc=0.6):
    """
    5-indicator ensemble with minimum allocation floor.

    Always invests in at least the top stocks by vote count, even if
    below threshold. Allocation scales with signal conviction.
    """
    all_scores = {}
    vols = {}

    for ticker in datasets:
        if ticker not in price_history.columns:
            continue
        prices = price_history[ticker].dropna()
        if len(prices) < 252:
            continue

        current = prices.iloc[-1]
        vols[ticker] = _get_vol(prices)
        votes = 0
        strength = 0

        # Vote 1: Price > 200 SMA
        sma200 = prices.rolling(200).mean().iloc[-1]
        if current > sma200:
            votes += 1
            strength += (current / sma200 - 1)

        # Vote 2: Price > 50 SMA
        sma50 = prices.rolling(50).mean().iloc[-1]
        if current > sma50:
            votes += 1
            strength += (current / sma50 - 1)

        # Vote 3: 12-month momentum positive
        mom_12m = (current / prices.iloc[-252]) - 1
        if mom_12m > 0:
            votes += 1
            strength += mom_12m

        # Vote 4: RSI 40-70 (healthy momentum)
        delta = prices.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1/14, min_periods=14).mean().iloc[-1]
        avg_loss = loss.ewm(alpha=1/14, min_periods=14).mean().iloc[-1]
        rsi = 100 - (100 / (1 + avg_gain / avg_loss)) if avg_loss > 0 else 50
        if 40 < rsi < 70:
            votes += 1
            strength += 0.05

        # Vote 5: 50-SMA slope positive
        sma50_prev = prices.rolling(50).mean().iloc[-20] if len(prices) >= 220 else sma50
        if sma50 > sma50_prev:
            votes += 1
            strength += 0.05

        all_scores[ticker] = {"votes": votes, "strength": strength}

    if not all_scores:
        return {}

    # Primary: stocks meeting threshold
    strong = {t: s["strength"] for t, s in all_scores.items() if s["votes"] >= min_votes}

    if len(strong) >= top_n:
        ranked = sorted(strong.items(), key=lambda x: x[1], reverse=True)
        selected = [t for t, _ in ranked[:top_n]]
        return _inv_vol_weights(selected, vols, 0.95)

    # Fallback: relax threshold, use whatever has most votes
    by_votes = sorted(all_scores.items(), key=lambda x: (x[1]["votes"], x[1]["strength"]), reverse=True)
    selected = [t for t, _ in by_votes[:top_n]]

    # Scale allocation by average conviction
    avg_votes = np.mean([all_scores[t]["votes"] for t in selected])
    conviction = avg_votes / 5.0
    alloc = min_alloc + (0.95 - min_alloc) * conviction

    return _inv_vol_weights(selected, vols, alloc)


# ==========================================================================
#  3. REGIME-ADAPTIVE ROTATION
# ==========================================================================

def regime_rotation(date, datasets, price_history,
                    adx_threshold=25, mom_lookback=63, top_n=3,
                    min_alloc=0.6):
    """
    Regime-adaptive rotation with fallback allocation.

    Trending market: ride momentum winners.
    Ranging market: buy relative strength dips in uptrending stocks.
    Always maintains minimum allocation.
    """
    returns_nm = {}
    vols = {}
    long_term_trend = {}

    for ticker in datasets:
        if ticker not in price_history.columns:
            continue
        prices = price_history[ticker].dropna()
        if len(prices) < 252:
            continue

        current = prices.iloc[-1]
        past = prices.iloc[-mom_lookback] if len(prices) >= mom_lookback else prices.iloc[0]
        returns_nm[ticker] = (current / past) - 1
        vols[ticker] = _get_vol(prices)

        sma200 = prices.rolling(200).mean().iloc[-1]
        long_term_trend[ticker] = current > sma200

    if len(returns_nm) < 3:
        return {}

    avg_return = np.mean(list(returns_nm.values()))
    trending = avg_return > 0.02

    if trending:
        # Momentum: top performers with positive long-term trend
        candidates = {t: r for t, r in returns_nm.items() if long_term_trend.get(t, False)}
        if not candidates:
            candidates = returns_nm

        ranked = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
        selected = [t for t, _ in ranked[:top_n] if candidates[t] > -0.05]

        if not selected:
            selected = [t for t, _ in ranked[:top_n]]

        return _inv_vol_weights(selected, vols, 0.95)

    else:
        # Ranging: oversold stocks in long-term uptrend (or least-damaged)
        candidates = {}
        for ticker in datasets:
            if ticker not in price_history.columns:
                continue
            prices = price_history[ticker].dropna()
            if len(prices) < 252:
                continue

            sma50 = prices.rolling(50).mean().iloc[-1]
            deviation = (prices.iloc[-1] / sma50) - 1

            if long_term_trend.get(ticker, False):
                candidates[ticker] = deviation
            elif deviation > -0.15:
                candidates[ticker] = deviation + 0.5  # penalty for no uptrend

        if not candidates:
            # Nothing looks good, equal-weight the least bad
            ranked = sorted(returns_nm.items(), key=lambda x: x[1], reverse=True)
            selected = [t for t, _ in ranked[:top_n]]
            return _inv_vol_weights(selected, vols, min_alloc)

        ranked = sorted(candidates.items(), key=lambda x: x[1])
        selected = [t for t, _ in ranked[:top_n]]
        return _inv_vol_weights(selected, vols, 0.90)


# ==========================================================================
#  4. COMPOSITE: Best of all approaches
# ==========================================================================

def composite_strategy(date, datasets, price_history,
                       top_n=4, min_votes=2, min_alloc=0.6):
    """
    Composite: momentum + ensemble + regime scoring.

    Each stock gets a composite score from:
    - Relative momentum (6-month)
    - Ensemble vote count
    - Risk-adjusted quality (Sharpe of recent returns)

    Always invests in top stocks. Allocation scales with conviction.
    """
    scores = {}
    vols = {}

    for ticker in datasets:
        if ticker not in price_history.columns:
            continue
        prices = price_history[ticker].dropna()
        if len(prices) < 252:
            continue

        current = prices.iloc[-1]
        ret = prices.pct_change().dropna()
        vols[ticker] = _get_vol(prices)

        # Component 1: 6-month momentum (0-1 normalized later)
        mom_6m = (current / prices.iloc[-126]) - 1 if len(prices) >= 126 else 0

        # Component 2: Ensemble votes (0-1)
        sma200 = prices.rolling(200).mean().iloc[-1]
        sma50 = prices.rolling(50).mean().iloc[-1]
        mom_12m = (current / prices.iloc[-252]) - 1

        votes = 0
        if current > sma200: votes += 1
        if current > sma50: votes += 1
        if mom_12m > 0: votes += 1
        if sma50 > prices.rolling(50).mean().iloc[-20]: votes += 1

        # RSI check
        delta = prices.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1/14, min_periods=14).mean().iloc[-1]
        avg_loss = loss.ewm(alpha=1/14, min_periods=14).mean().iloc[-1]
        rsi = 100 - (100 / (1 + avg_gain / avg_loss)) if avg_loss > 0 else 50
        if 40 < rsi < 70: votes += 1

        vote_score = votes / 5.0

        # Component 3: Risk-adjusted quality (60-day Sharpe)
        recent = ret.iloc[-60:]
        quality = (recent.mean() / recent.std()) if len(recent) >= 20 and recent.std() > 0 else 0
        quality_norm = max(min(quality / 0.15, 1.0), -1.0)  # normalize

        # Composite: weighted combination
        vol = max(vols[ticker], 0.01)
        composite = (0.4 * mom_6m + 0.3 * vote_score + 0.3 * quality_norm) / vol

        scores[ticker] = {
            "composite": composite,
            "votes": votes,
            "mom": mom_6m,
        }

    if not scores:
        return {}

    # Rank by composite score
    ranked = sorted(scores.items(), key=lambda x: x[1]["composite"], reverse=True)
    selected = [t for t, _ in ranked[:top_n]]

    # Conviction-based allocation
    avg_votes = np.mean([scores[t]["votes"] for t in selected])
    positive_mom = sum(1 for t in selected if scores[t]["mom"] > 0)
    conviction = (avg_votes / 5.0 + positive_mom / len(selected)) / 2

    alloc = min_alloc + (0.95 - min_alloc) * conviction
    return _inv_vol_weights(selected, vols, alloc)


# ==========================================================================
#  5. MOMENTUM + QUALITY FACTOR
# ==========================================================================

def momentum_quality(date, datasets, price_history,
                     mom_lookback=126, quality_lookback=60,
                     top_n=4, min_alloc=0.6):
    """
    Two-factor model: Momentum + Quality.

    Momentum: 6-month price return (captures trends)
    Quality: risk-adjusted return (Sharpe ratio over 60 days)

    Ranks stocks by combined score, always invests in top N.
    This avoids the "going to cash" problem while still rotating
    into the best risk-adjusted opportunities.
    """
    scores = {}
    vols = {}

    for ticker in datasets:
        if ticker not in price_history.columns:
            continue
        prices = price_history[ticker].dropna()
        if len(prices) < 252:
            continue

        current = prices.iloc[-1]
        ret = prices.pct_change().dropna()
        vols[ticker] = _get_vol(prices)

        # Momentum factor
        past = prices.iloc[-mom_lookback] if len(prices) >= mom_lookback else prices.iloc[0]
        momentum = (current / past) - 1

        # Quality factor (risk-adjusted return)
        recent = ret.iloc[-quality_lookback:]
        if len(recent) >= 20 and recent.std() > 0:
            quality = recent.mean() / recent.std()
        else:
            quality = 0

        # Combined score (equal weight, vol-adjusted)
        vol = max(vols[ticker], 0.01)
        scores[ticker] = (momentum + quality * 0.1) / vol

    if not scores:
        return {}

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    selected = [t for t, _ in ranked[:top_n]]

    # Always at least min_alloc invested
    return _inv_vol_weights(selected, vols, max(min_alloc, 0.90))


# Strategy registry
PORTFOLIO_STRATEGIES = {
    "Dual Momentum": {
        "fn": dual_momentum,
        "params_grid": [
            {"abs_lookback": 252, "rel_lookback": 126, "top_n": 3, "vol_adjust": True, "min_alloc": 0.5},
            {"abs_lookback": 252, "rel_lookback": 126, "top_n": 3, "vol_adjust": True, "min_alloc": 0.7},
            {"abs_lookback": 252, "rel_lookback": 63,  "top_n": 3, "vol_adjust": True, "min_alloc": 0.6},
            {"abs_lookback": 200, "rel_lookback": 126, "top_n": 3, "vol_adjust": True, "min_alloc": 0.6},
            {"abs_lookback": 200, "rel_lookback": 63,  "top_n": 4, "vol_adjust": True, "min_alloc": 0.5},
            {"abs_lookback": 252, "rel_lookback": 126, "top_n": 4, "vol_adjust": True, "min_alloc": 0.6},
            {"abs_lookback": 252, "rel_lookback": 63,  "top_n": 4, "vol_adjust": True, "min_alloc": 0.7},
            {"abs_lookback": 252, "rel_lookback": 126, "top_n": 2, "vol_adjust": True, "min_alloc": 0.6},
            {"abs_lookback": 200, "rel_lookback": 126, "top_n": 5, "vol_adjust": True, "min_alloc": 0.7},
            {"abs_lookback": 126, "rel_lookback": 63,  "top_n": 3, "vol_adjust": True, "min_alloc": 0.6},
        ],
    },
    "Ensemble Voting": {
        "fn": ensemble_voting,
        "params_grid": [
            {"min_votes": 3, "top_n": 3, "min_alloc": 0.6},
            {"min_votes": 3, "top_n": 4, "min_alloc": 0.6},
            {"min_votes": 3, "top_n": 5, "min_alloc": 0.6},
            {"min_votes": 2, "top_n": 3, "min_alloc": 0.6},
            {"min_votes": 2, "top_n": 4, "min_alloc": 0.7},
            {"min_votes": 2, "top_n": 5, "min_alloc": 0.7},
            {"min_votes": 4, "top_n": 3, "min_alloc": 0.5},
            {"min_votes": 4, "top_n": 4, "min_alloc": 0.5},
        ],
    },
    "Regime Rotation": {
        "fn": regime_rotation,
        "params_grid": [
            {"adx_threshold": 25, "mom_lookback": 63,  "top_n": 3, "min_alloc": 0.6},
            {"adx_threshold": 25, "mom_lookback": 63,  "top_n": 4, "min_alloc": 0.6},
            {"adx_threshold": 25, "mom_lookback": 42,  "top_n": 3, "min_alloc": 0.6},
            {"adx_threshold": 25, "mom_lookback": 42,  "top_n": 4, "min_alloc": 0.7},
            {"adx_threshold": 25, "mom_lookback": 126, "top_n": 3, "min_alloc": 0.6},
            {"adx_threshold": 25, "mom_lookback": 63,  "top_n": 5, "min_alloc": 0.7},
            {"adx_threshold": 25, "mom_lookback": 42,  "top_n": 5, "min_alloc": 0.7},
            {"adx_threshold": 25, "mom_lookback": 63,  "top_n": 3, "min_alloc": 0.8},
        ],
    },
    "Composite": {
        "fn": composite_strategy,
        "params_grid": [
            {"top_n": 3, "min_votes": 2, "min_alloc": 0.6},
            {"top_n": 4, "min_votes": 2, "min_alloc": 0.6},
            {"top_n": 4, "min_votes": 2, "min_alloc": 0.7},
            {"top_n": 5, "min_votes": 2, "min_alloc": 0.7},
            {"top_n": 3, "min_votes": 3, "min_alloc": 0.6},
            {"top_n": 4, "min_votes": 3, "min_alloc": 0.6},
            {"top_n": 5, "min_votes": 3, "min_alloc": 0.7},
            {"top_n": 3, "min_votes": 2, "min_alloc": 0.8},
        ],
    },
    "Momentum + Quality": {
        "fn": momentum_quality,
        "params_grid": [
            {"mom_lookback": 126, "quality_lookback": 60, "top_n": 3, "min_alloc": 0.9},
            {"mom_lookback": 126, "quality_lookback": 60, "top_n": 4, "min_alloc": 0.9},
            {"mom_lookback": 126, "quality_lookback": 60, "top_n": 5, "min_alloc": 0.9},
            {"mom_lookback": 63,  "quality_lookback": 40, "top_n": 3, "min_alloc": 0.9},
            {"mom_lookback": 63,  "quality_lookback": 40, "top_n": 4, "min_alloc": 0.9},
            {"mom_lookback": 252, "quality_lookback": 60, "top_n": 3, "min_alloc": 0.9},
            {"mom_lookback": 252, "quality_lookback": 60, "top_n": 4, "min_alloc": 0.9},
            {"mom_lookback": 126, "quality_lookback": 90, "top_n": 4, "min_alloc": 0.9},
        ],
    },
}
