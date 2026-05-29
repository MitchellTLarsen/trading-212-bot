from .sma_crossover import SMACrossover
from .rsi_mean_reversion import RSIMeanReversion
from .bollinger_bands import BollingerBandReversion
from .macd_rsi import MACDRSICombo
from .ema_pullback import EMAPullback
from .momentum_breakout import MomentumBreakout
from .mean_reversion_roc import MeanReversionROC
from .consolidation_breakout import ConsolidationBreakout
from .crash_avoider import CrashAvoider
from .dip_buyer import DipBuyer
from .trend_rider import TrendRider
from .dual_regime import DualRegime

ALL_STRATEGIES = [
    SMACrossover(),
    RSIMeanReversion(),
    BollingerBandReversion(),
    MACDRSICombo(),
    EMAPullback(),
    MomentumBreakout(),
    MeanReversionROC(),
    ConsolidationBreakout(),
    CrashAvoider(),
    DipBuyer(),
    TrendRider(),
    DualRegime(),
]

# Strategies designed to beat buy & hold
BH_BEATER_STRATEGIES = [
    CrashAvoider(),
    DipBuyer(),
    TrendRider(),
    DualRegime(),
    SMACrossover(),   # also test original winner with new optimization target
]
