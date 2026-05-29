"""
Order flow strategies — combine volume/price flow analysis with technicals.

Forex strategies use price-derived flow (no volume needed).
Stock strategies use full volume-based flow.
"""

import pandas as pd
import numpy as np
from backtest.indicators import compute_sma, compute_ema, compute_rsi, compute_atr
from backtest.orderflow import (
    compute_candle_pressure, detect_order_blocks, detect_fair_value_gaps,
    detect_liquidity_sweeps, compute_displacement,
    compute_obv, compute_mfi, compute_cmf, compute_volume_delta,
)


# ========================================================================
#  FOREX ORDER FLOW (price-derived, no volume needed)
# ========================================================================

def smart_money_reversal(data, ob_mult=1.5, rsi_period=7, rsi_oversold=30,
                         rsi_overbought=70, lookback=20):
    """
    Smart Money Concepts: order blocks + RSI confirmation.

    Buy when a bullish order block forms AND RSI is oversold.
    Short when a bearish order block forms AND RSI is overbought.

    Order blocks signal institutional accumulation/distribution.
    RSI confirms the reversal timing.
    """
    close = data["Close"]
    open_ = data["Open"]
    high = data["High"]
    low = data["Low"]

    ob = detect_order_blocks(open_, high, low, close,
                             min_body_mult=ob_mult, lookback=lookback)
    rsi = compute_rsi(close, rsi_period)

    signals = pd.Series(0, index=data.index)
    signals[(ob == 1) & (rsi < rsi_oversold)] = 1
    signals[(ob == -1) & (rsi > rsi_overbought)] = -1
    return signals


def liquidity_sweep_reversal(data, sweep_lookback=20, rsi_period=7,
                             confirm_candles=1):
    """
    Liquidity sweep + reversal.

    Institutions sweep stops below/above key levels then reverse.
    Enter after the sweep when price confirms direction change.

    Bullish: price sweeps below recent low then closes back above.
    Bearish: price sweeps above recent high then closes back below.
    """
    close = data["Close"]
    high = data["High"]
    low = data["Low"]

    sweep = detect_liquidity_sweeps(high, low, close, lookback=sweep_lookback)
    rsi = compute_rsi(close, rsi_period)

    signals = pd.Series(0, index=data.index)

    # Bullish sweep + RSI not overbought
    signals[(sweep == 1) & (rsi < 60)] = 1
    # Bearish sweep + RSI not oversold
    signals[(sweep == -1) & (rsi > 40)] = -1

    return signals


def fvg_fill_strategy(data, min_gap_pct=0.1, trend_sma=50):
    """
    Fair Value Gap fill — trade in the direction price needs to go
    to fill imbalances.

    Bullish FVG (gap up) means price may pull back to fill it.
    Enter short near a recent bullish FVG (expect pullback).
    Enter long near a recent bearish FVG (expect bounce).

    With trend filter to avoid counter-trend trades.
    """
    close = data["Close"]
    high = data["High"]
    low = data["Low"]

    fvg_bull, fvg_bear = detect_fair_value_gaps(high, low, close, min_gap_pct)
    trend = compute_sma(close, trend_sma)

    signals = pd.Series(0, index=data.index)

    # Bearish FVG in uptrend = buy opportunity (dip to fill gap)
    signals[(fvg_bear > 0) & (close > trend)] = 1
    # Bullish FVG in downtrend = short opportunity (rally to fill gap)
    signals[(fvg_bull > 0) & (close < trend)] = -1

    return signals


def displacement_momentum(data, disp_period=3, disp_threshold=1.5,
                          pressure_period=5, rsi_period=10):
    """
    Displacement + candle pressure momentum.

    Enter when there's a sudden aggressive move (displacement)
    confirmed by sustained candle pressure in the same direction.

    Displacement = institutional money entering the market aggressively.
    Pressure = the move has follow-through, not just a spike.
    """
    close = data["Close"]
    open_ = data["Open"]
    high = data["High"]
    low = data["Low"]

    disp = compute_displacement(close, disp_period, disp_threshold)
    pressure = compute_candle_pressure(open_, high, low, close, pressure_period)
    rsi = compute_rsi(close, rsi_period)

    signals = pd.Series(0, index=data.index)

    # Bullish: displacement up + positive pressure + RSI not extreme
    signals[(disp == 1) & (pressure > 1) & (rsi < 75)] = 1
    # Bearish: displacement down + negative pressure + RSI not extreme
    signals[(disp == -1) & (pressure < -1) & (rsi > 25)] = -1

    return signals


def combined_flow_reversal(data, rsi_period=7, rsi_oversold=30, rsi_overbought=70,
                           pressure_period=5, sweep_lookback=15, ob_mult=1.5):
    """
    Multi-factor order flow strategy — combines ALL flow signals.

    Requires at least 2 of: order block, liquidity sweep, candle pressure
    PLUS RSI confirmation. Highest conviction trades only.
    """
    close = data["Close"]
    open_ = data["Open"]
    high = data["High"]
    low = data["Low"]

    rsi = compute_rsi(close, rsi_period)
    ob = detect_order_blocks(open_, high, low, close, ob_mult)
    sweep = detect_liquidity_sweeps(high, low, close, sweep_lookback)
    pressure = compute_candle_pressure(open_, high, low, close, pressure_period)

    signals = pd.Series(0, index=data.index)

    # Bullish: count how many flow signals agree
    bull_votes = ((ob == 1).astype(int) +
                  (sweep == 1).astype(int) +
                  (pressure > 1.5).astype(int))

    bear_votes = ((ob == -1).astype(int) +
                  (sweep == -1).astype(int) +
                  (pressure < -1.5).astype(int))

    # Need 2+ flow signals + RSI confirmation
    signals[(bull_votes >= 2) & (rsi < rsi_oversold + 10)] = 1
    signals[(bear_votes >= 2) & (rsi > rsi_overbought - 10)] = -1

    return signals


# ========================================================================
#  STOCK ORDER FLOW (volume-based)
# ========================================================================

def obv_divergence(data, obv_sma=20, price_sma=20):
    """
    OBV divergence — when OBV trends up but price trends down (or vice versa),
    it signals that smart money is accumulating/distributing before a move.

    Buy: price making lower lows but OBV making higher lows (accumulation).
    Sell: price making higher highs but OBV making lower highs (distribution).
    """
    close = data["Close"]
    volume = data["Volume"]

    obv = compute_obv(close, volume)
    obv_ma = compute_sma(obv, obv_sma)
    price_ma = compute_sma(close, price_sma)

    # OBV trend
    obv_rising = obv_ma > obv_ma.shift(5)
    obv_falling = obv_ma < obv_ma.shift(5)

    # Price trend
    price_falling = price_ma < price_ma.shift(5)
    price_rising = price_ma > price_ma.shift(5)

    signals = pd.Series(0, index=data.index)

    # Bullish divergence: price down, OBV up
    bull_div = price_falling & obv_rising
    signals[bull_div & (~bull_div).shift(1, fill_value=False)] = 1

    # Bearish divergence: price up, OBV down (for short/exit)
    bear_div = price_rising & obv_falling
    signals[bear_div & (~bear_div).shift(1, fill_value=False)] = -1

    return signals


def mfi_extreme(data, mfi_period=14, oversold=20, overbought=80, trend_sma=50):
    """
    Money Flow Index extremes with trend filter.

    MFI is RSI but volume-weighted — oversold/overbought signals are
    higher conviction because they account for actual money flow.
    """
    close = data["Close"]
    high = data["High"]
    low = data["Low"]
    volume = data["Volume"]

    mfi = compute_mfi(high, low, close, volume, mfi_period)
    trend = compute_sma(close, trend_sma)

    signals = pd.Series(0, index=data.index)

    # Buy: MFI oversold in uptrend
    signals[(mfi < oversold) & (close > trend)] = 1
    # Sell: MFI overbought
    signals[(mfi > overbought)] = -1

    return signals


def volume_delta_momentum(data, delta_period=5, threshold=0.6, trend_sma=50):
    """
    Volume delta — approximates buy vs sell volume from candle structure.

    When buy volume consistently exceeds sell volume (positive delta),
    institutions are accumulating. Trade in that direction.
    """
    close = data["Close"]
    open_ = data["Open"]
    high = data["High"]
    low = data["Low"]
    volume = data["Volume"]

    delta, buy_vol, sell_vol = compute_volume_delta(open_, close, high, low, volume)
    delta_ma = delta.rolling(delta_period).mean()
    avg_vol = volume.rolling(20).mean()
    trend = compute_sma(close, trend_sma)

    signals = pd.Series(0, index=data.index)

    # Normalized delta
    norm_delta = delta_ma / avg_vol.replace(0, np.nan)
    norm_delta = norm_delta.fillna(0)

    # Buy: strong positive delta in uptrend
    signals[(norm_delta > threshold) & (close > trend)] = 1
    # Sell: strong negative delta
    signals[(norm_delta < -threshold)] = -1

    return signals


def cmf_trend(data, cmf_period=20, threshold=0.1, ema_period=21):
    """
    Chaikin Money Flow trend-following.

    CMF > 0 = buying pressure, < 0 = selling pressure.
    Trade in direction of sustained money flow with EMA confirmation.
    """
    close = data["Close"]
    high = data["High"]
    low = data["Low"]
    volume = data["Volume"]

    cmf = compute_cmf(high, low, close, volume, cmf_period)
    ema = compute_ema(close, ema_period)

    signals = pd.Series(0, index=data.index)

    # Buy: CMF crosses above threshold + price above EMA
    bull = (cmf > threshold) & (cmf.shift(1) <= threshold) & (close > ema)
    signals[bull] = 1

    # Sell: CMF crosses below negative threshold
    bear = (cmf < -threshold) & (cmf.shift(1) >= -threshold)
    signals[bear] = -1

    return signals


# ========================================================================
#  Strategy registry
# ========================================================================

FOREX_ORDERFLOW_STRATEGIES = {
    "Smart Money Reversal": {
        "fn": smart_money_reversal,
        "params_grid": [
            {"ob_mult": 1.5, "rsi_period": 7, "rsi_oversold": 25, "rsi_overbought": 75, "lookback": 20},
            {"ob_mult": 1.5, "rsi_period": 7, "rsi_oversold": 30, "rsi_overbought": 70, "lookback": 20},
            {"ob_mult": 1.5, "rsi_period": 5, "rsi_oversold": 25, "rsi_overbought": 75, "lookback": 15},
            {"ob_mult": 2.0, "rsi_period": 7, "rsi_oversold": 30, "rsi_overbought": 70, "lookback": 20},
            {"ob_mult": 1.3, "rsi_period": 10, "rsi_oversold": 30, "rsi_overbought": 70, "lookback": 25},
            {"ob_mult": 1.5, "rsi_period": 7, "rsi_oversold": 35, "rsi_overbought": 65, "lookback": 15},
        ],
    },
    "Liquidity Sweep": {
        "fn": liquidity_sweep_reversal,
        "params_grid": [
            {"sweep_lookback": 15, "rsi_period": 7, "confirm_candles": 1},
            {"sweep_lookback": 20, "rsi_period": 7, "confirm_candles": 1},
            {"sweep_lookback": 10, "rsi_period": 5, "confirm_candles": 1},
            {"sweep_lookback": 20, "rsi_period": 10, "confirm_candles": 1},
            {"sweep_lookback": 30, "rsi_period": 7, "confirm_candles": 1},
            {"sweep_lookback": 15, "rsi_period": 5, "confirm_candles": 1},
        ],
    },
    "FVG Fill": {
        "fn": fvg_fill_strategy,
        "params_grid": [
            {"min_gap_pct": 0.05, "trend_sma": 50},
            {"min_gap_pct": 0.1, "trend_sma": 50},
            {"min_gap_pct": 0.1, "trend_sma": 100},
            {"min_gap_pct": 0.15, "trend_sma": 50},
            {"min_gap_pct": 0.05, "trend_sma": 30},
            {"min_gap_pct": 0.2, "trend_sma": 50},
        ],
    },
    "Displacement Momentum": {
        "fn": displacement_momentum,
        "params_grid": [
            {"disp_period": 3, "disp_threshold": 1.5, "pressure_period": 5, "rsi_period": 10},
            {"disp_period": 3, "disp_threshold": 2.0, "pressure_period": 5, "rsi_period": 10},
            {"disp_period": 5, "disp_threshold": 1.5, "pressure_period": 5, "rsi_period": 10},
            {"disp_period": 3, "disp_threshold": 1.5, "pressure_period": 3, "rsi_period": 7},
            {"disp_period": 3, "disp_threshold": 1.5, "pressure_period": 7, "rsi_period": 14},
            {"disp_period": 2, "disp_threshold": 1.5, "pressure_period": 3, "rsi_period": 7},
        ],
    },
    "Combined Flow": {
        "fn": combined_flow_reversal,
        "params_grid": [
            {"rsi_period": 7, "rsi_oversold": 30, "rsi_overbought": 70, "pressure_period": 5, "sweep_lookback": 15, "ob_mult": 1.5},
            {"rsi_period": 7, "rsi_oversold": 25, "rsi_overbought": 75, "pressure_period": 5, "sweep_lookback": 20, "ob_mult": 1.5},
            {"rsi_period": 5, "rsi_oversold": 30, "rsi_overbought": 70, "pressure_period": 3, "sweep_lookback": 15, "ob_mult": 1.3},
            {"rsi_period": 10, "rsi_oversold": 35, "rsi_overbought": 65, "pressure_period": 7, "sweep_lookback": 20, "ob_mult": 1.5},
            {"rsi_period": 7, "rsi_oversold": 30, "rsi_overbought": 70, "pressure_period": 5, "sweep_lookback": 10, "ob_mult": 2.0},
        ],
    },
}

STOCK_ORDERFLOW_STRATEGIES = {
    "OBV Divergence": {
        "fn": obv_divergence,
        "params_grid": [
            {"obv_sma": 10, "price_sma": 10},
            {"obv_sma": 15, "price_sma": 15},
            {"obv_sma": 20, "price_sma": 20},
            {"obv_sma": 20, "price_sma": 10},
            {"obv_sma": 10, "price_sma": 20},
        ],
    },
    "MFI Extreme": {
        "fn": mfi_extreme,
        "params_grid": [
            {"mfi_period": 10, "oversold": 20, "overbought": 80, "trend_sma": 50},
            {"mfi_period": 14, "oversold": 20, "overbought": 80, "trend_sma": 50},
            {"mfi_period": 14, "oversold": 15, "overbought": 85, "trend_sma": 50},
            {"mfi_period": 10, "oversold": 25, "overbought": 75, "trend_sma": 100},
            {"mfi_period": 7, "oversold": 20, "overbought": 80, "trend_sma": 50},
        ],
    },
    "Volume Delta": {
        "fn": volume_delta_momentum,
        "params_grid": [
            {"delta_period": 3, "threshold": 0.4, "trend_sma": 50},
            {"delta_period": 5, "threshold": 0.5, "trend_sma": 50},
            {"delta_period": 5, "threshold": 0.6, "trend_sma": 50},
            {"delta_period": 5, "threshold": 0.4, "trend_sma": 100},
            {"delta_period": 7, "threshold": 0.5, "trend_sma": 50},
        ],
    },
    "CMF Trend": {
        "fn": cmf_trend,
        "params_grid": [
            {"cmf_period": 15, "threshold": 0.05, "ema_period": 21},
            {"cmf_period": 20, "threshold": 0.1, "ema_period": 21},
            {"cmf_period": 20, "threshold": 0.05, "ema_period": 21},
            {"cmf_period": 20, "threshold": 0.1, "ema_period": 50},
            {"cmf_period": 10, "threshold": 0.05, "ema_period": 15},
        ],
    },
}
