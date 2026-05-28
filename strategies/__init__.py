from .sma_crossover import SMACrossover
from .rsi_mean_reversion import RSIMeanReversion
from .bollinger_bands import BollingerBandReversion
from .macd_rsi import MACDRSICombo
from .ema_pullback import EMAPullback
from .momentum_breakout import MomentumBreakout
from .mean_reversion_roc import MeanReversionROC
from .consolidation_breakout import ConsolidationBreakout

ALL_STRATEGIES = [
    SMACrossover(),
    RSIMeanReversion(),
    BollingerBandReversion(),
    MACDRSICombo(),
    EMAPullback(),
    MomentumBreakout(),
    MeanReversionROC(),
    ConsolidationBreakout(),
]
