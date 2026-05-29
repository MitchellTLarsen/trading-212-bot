"""
Order flow indicators — both volume-based (stocks) and price-derived (forex).

Volume-based: OBV, MFI, Chaikin Money Flow, VWAP
Price-derived: Order blocks, fair value gaps, liquidity sweeps, candle pressure
"""

import pandas as pd
import numpy as np


# ========================================================================
#  VOLUME-BASED (requires volume data — stocks, some futures)
# ========================================================================

def compute_obv(close, volume):
    """On-Balance Volume — running total, +vol on up days, -vol on down days."""
    direction = np.sign(close.diff())
    obv = (direction * volume).cumsum()
    return obv


def compute_mfi(high, low, close, volume, period=14):
    """Money Flow Index — RSI but volume-weighted. 0-100 scale."""
    typical_price = (high + low + close) / 3
    raw_money_flow = typical_price * volume

    delta = typical_price.diff()
    pos_flow = raw_money_flow.where(delta > 0, 0.0)
    neg_flow = raw_money_flow.where(delta < 0, 0.0)

    pos_sum = pos_flow.rolling(period).sum()
    neg_sum = neg_flow.rolling(period).sum()

    money_ratio = pos_sum / neg_sum.replace(0, np.nan)
    mfi = 100 - (100 / (1 + money_ratio))
    return mfi.fillna(50)


def compute_cmf(high, low, close, volume, period=20):
    """Chaikin Money Flow — measures buying/selling pressure. -1 to +1."""
    clv = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    clv = clv.fillna(0)
    cmf = (clv * volume).rolling(period).sum() / volume.rolling(period).sum()
    return cmf.fillna(0)


def compute_vwap(high, low, close, volume):
    """VWAP — Volume Weighted Average Price (daily reset)."""
    typical = (high + low + close) / 3
    cum_tp_vol = (typical * volume).cumsum()
    cum_vol = volume.cumsum()
    return cum_tp_vol / cum_vol.replace(0, np.nan)


def compute_ad_line(high, low, close, volume):
    """Accumulation/Distribution Line."""
    clv = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    clv = clv.fillna(0)
    ad = (clv * volume).cumsum()
    return ad


def compute_volume_delta(open_, close, high, low, volume):
    """
    Approximate buy/sell volume split using candle structure.
    Buy volume ~ proportional to (close - low) / (high - low)
    Sell volume ~ proportional to (high - close) / (high - low)
    """
    rng = (high - low).replace(0, np.nan)
    buy_ratio = (close - low) / rng
    sell_ratio = (high - close) / rng

    buy_vol = volume * buy_ratio.fillna(0.5)
    sell_vol = volume * sell_ratio.fillna(0.5)
    delta = buy_vol - sell_vol
    return delta, buy_vol, sell_vol


# ========================================================================
#  PRICE-DERIVED (no volume needed — works on forex)
# ========================================================================

def compute_candle_pressure(open_, high, low, close, period=5):
    """
    Candle buying/selling pressure without volume.

    Body ratio: body_size / total_range. High = strong conviction.
    Direction pressure: rolling sum of directional candles.
    """
    body = abs(close - open_)
    rng = (high - low).replace(0, np.nan)
    body_ratio = body / rng
    body_ratio = body_ratio.fillna(0)

    direction = np.sign(close - open_)  # +1 bullish, -1 bearish
    pressure = (direction * body_ratio).rolling(period).sum()

    return pressure


def detect_order_blocks(open_, high, low, close, min_body_mult=1.5, lookback=20):
    """
    Order blocks — large institutional candles that create supply/demand zones.

    Bullish OB: large bearish candle followed by strong bullish move
    (institutions absorbed selling, then reversed)
    Bearish OB: large bullish candle followed by strong bearish move

    Returns: Series with 1 = bullish OB, -1 = bearish OB, 0 = none
    """
    body = abs(close - open_)
    avg_body = body.rolling(lookback).mean()

    ob_signal = pd.Series(0, index=close.index)

    for i in range(lookback + 2, len(close)):
        # Bullish order block: big bearish candle, then price reverses up
        if (close.iloc[i-2] < open_.iloc[i-2] and  # bearish candle
            body.iloc[i-2] > avg_body.iloc[i-2] * min_body_mult and  # large body
            close.iloc[i] > high.iloc[i-2]):  # price now above that candle's high
            ob_signal.iloc[i] = 1

        # Bearish order block: big bullish candle, then price reverses down
        if (close.iloc[i-2] > open_.iloc[i-2] and  # bullish candle
            body.iloc[i-2] > avg_body.iloc[i-2] * min_body_mult and  # large body
            close.iloc[i] < low.iloc[i-2]):  # price now below that candle's low
            ob_signal.iloc[i] = -1

    return ob_signal


def detect_fair_value_gaps(high, low, close, min_gap_pct=0.1):
    """
    Fair Value Gaps (FVG) — imbalances where price moved so fast
    it left a gap between candle 1's low and candle 3's high (bullish)
    or candle 1's high and candle 3's low (bearish).

    Price tends to revisit these gaps to "fill" them.

    Returns: DataFrame with 'fvg_bullish' and 'fvg_bearish' columns.
    """
    fvg_bull = pd.Series(0.0, index=close.index)
    fvg_bear = pd.Series(0.0, index=close.index)

    for i in range(2, len(close)):
        # Bullish FVG: candle 3's low > candle 1's high (gap up)
        gap_up = low.iloc[i] - high.iloc[i-2]
        if gap_up > 0:
            gap_pct = gap_up / close.iloc[i] * 100
            if gap_pct > min_gap_pct:
                fvg_bull.iloc[i] = gap_pct

        # Bearish FVG: candle 1's low > candle 3's high (gap down)
        gap_down = low.iloc[i-2] - high.iloc[i]
        if gap_down > 0:
            gap_pct = gap_down / close.iloc[i] * 100
            if gap_pct > min_gap_pct:
                fvg_bear.iloc[i] = gap_pct

    return fvg_bull, fvg_bear


def detect_liquidity_sweeps(high, low, close, lookback=20, min_sweep_pct=0.1):
    """
    Liquidity sweeps — price breaks above/below recent highs/lows
    then quickly reverses. Indicates stop hunts by institutions.

    Bullish sweep: breaks below recent low then closes above it (bear trap)
    Bearish sweep: breaks above recent high then closes below it (bull trap)
    """
    recent_high = high.rolling(lookback).max().shift(1)
    recent_low = low.rolling(lookback).min().shift(1)

    sweep = pd.Series(0, index=close.index)

    # Bullish sweep: low went below recent low but close recovered above it
    bull_sweep = (low < recent_low) & (close > recent_low)
    sweep[bull_sweep] = 1

    # Bearish sweep: high went above recent high but close fell back below
    bear_sweep = (high > recent_high) & (close < recent_high)
    sweep[bear_sweep] = -1

    return sweep


def compute_displacement(close, period=3, threshold=1.5):
    """
    Displacement — sudden, aggressive price movement.
    Measured as N-bar return relative to recent average move.

    Large displacement = institutional activity.
    """
    returns = close.pct_change(period).abs()
    avg_move = returns.rolling(20).mean()

    bullish_disp = (close.pct_change(period) > avg_move * threshold)
    bearish_disp = (close.pct_change(period) < -avg_move * threshold)

    disp = pd.Series(0, index=close.index)
    disp[bullish_disp] = 1
    disp[bearish_disp] = -1

    return disp
