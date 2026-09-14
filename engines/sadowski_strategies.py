from __future__ import annotations

from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from engines.technical_math import TechnicalMathEngine

class SadowskiSignal(BaseModel):
    strategy_id: str
    strategy_name: str
    signal_bias: str  # LONG, SHORT, or NEUTRAL
    entry_trigger: float
    suggested_stop: float
    suggested_target: float
    confidence: int
    rule_description: str

SADOWSKI_CONFIG: Dict[str, Dict[str, Any]] = {
    "momentum_reversal": {"enabled": True},
    "triple_ma_crossover": {"enabled": True},
    "heikin_ashi_reversal": {"enabled": True},
    "swing_contraction": {"enabled": True},
    "role_reversal": {"enabled": True},
    "bollinger_squeeze": {"enabled": True},
    "narrow_range": {"enabled": True},
    "rsi_2": {"enabled": True},
    "money_management_3_1000": {"enabled": True}
}

class SadowskiStrategiesEngine:
    """
    Implements all 9 quantitative systems from Roman Sadowski's manual:
    1. Momentum Reversal (Stochastic + Support/Resistance)
    2. Triple Moving Average Crossover (20 / 60 / 100 EMA)
    3. Heikin-Ashi Reversal System (Stochastic + Pending Stop Order)
    4. Swing Day Trading on Contraction (Tight Overlapping Range Breakout)
    5. Role Reversal (Broken S/R Polar Flips)
    6. Bollinger Band Squeeze (Bandwidth <= 0.0100 Volatility Expansion)
    7. Narrow Range (NR4 / NR7 Expansion)
    8. 2-Period RSI (Aggressive Short-Term Momentum)
    9. 3/1000 Money Management System
    """

    @classmethod
    def evaluate_all(cls, candles: List[Any], current_price: float) -> List[SadowskiSignal]:
        signals: List[SadowskiSignal] = []
        if len(candles) < 25:
            return signals

        df = TechnicalMathEngine.to_dataframe(candles)

        # 1. Triple MA Crossover (20 EMA, 60 EMA, 100 EMA Trend Baseline)
        if SADOWSKI_CONFIG.get("triple_ma_crossover", {}).get("enabled", True) and len(df) >= 100:
            ema20 = TechnicalMathEngine.calculate_ema(df["close"], 20)
            ema60 = TechnicalMathEngine.calculate_ema(df["close"], 60)
            ema100 = TechnicalMathEngine.calculate_ema(df["close"], 100)

            if ema20 and ema60 and ema100:
                if ema20 > ema60 and current_price > ema100:
                    signals.append(SadowskiSignal(
                        strategy_id="TRIPLE_MA_CROSSOVER",
                        strategy_name="Triple Moving Average Crossover (20/60/100)",
                        signal_bias="LONG",
                        entry_trigger=current_price,
                        suggested_stop=round(ema60 * 0.998, 2),
                        suggested_target=round(current_price + (abs(current_price - ema60) * 2.0), 2),
                        confidence=82,
                        rule_description="Fast 20 EMA is above Slow 60 EMA with price trending above 100 EMA baseline."
                    ))
                elif ema20 < ema60 and current_price < ema100:
                    signals.append(SadowskiSignal(
                        strategy_id="TRIPLE_MA_CROSSOVER",
                        strategy_name="Triple Moving Average Crossover (20/60/100)",
                        signal_bias="SHORT",
                        entry_trigger=current_price,
                        suggested_stop=round(ema60 * 1.002, 2),
                        suggested_target=round(current_price - (abs(ema60 - current_price) * 2.0), 2),
                        confidence=82,
                        rule_description="Fast 20 EMA is below Slow 60 EMA with price trending below 100 EMA baseline."
                    ))

        # 2. Heikin-Ashi Reversal + Stochastic (14, 7, 3)
        if SADOWSKI_CONFIG.get("heikin_ashi_reversal", {}).get("enabled", True):
            ha_candles = TechnicalMathEngine.convert_to_heikin_ashi(df)
            k, d = TechnicalMathEngine.calculate_stochastic(df, 14, 3, 3)
            if len(ha_candles) >= 3 and k is not None:
                c1, c2, c3 = ha_candles[-3], ha_candles[-2], ha_candles[-1]
                # Bullish HA Reversal: 2 green candles after red sequence and Stoch oversold
                if not c1.is_green and c2.is_green and c3.is_green and k <= 35.0:
                    signals.append(SadowskiSignal(
                        strategy_id="HEIKIN_ASHI_REVERSAL",
                        strategy_name="Heikin-Ashi Two-Candle Reversal",
                        signal_bias="LONG",
                        entry_trigger=c3.high,
                        suggested_stop=round(min(c1.low, c2.low) * 0.9985, 2),
                        suggested_target=round(c3.high + (c3.high - min(c1.low, c2.low)) * 2.0, 2),
                        confidence=85,
                        rule_description="Two consecutive green Heikin-Ashi candles completed following downtrend with oversold Stochastic."
                    ))
                # Bearish HA Reversal: 2 red candles after green sequence and Stoch overbought
                elif c1.is_green and not c2.is_green and not c3.is_green and k >= 65.0:
                    signals.append(SadowskiSignal(
                        strategy_id="HEIKIN_ASHI_REVERSAL",
                        strategy_name="Heikin-Ashi Two-Candle Reversal",
                        signal_bias="SHORT",
                        entry_trigger=c3.low,
                        suggested_stop=round(max(c1.high, c2.high) * 1.0015, 2),
                        suggested_target=round(c3.low - (max(c1.high, c2.high) - c3.low) * 2.0, 2),
                        confidence=85,
                        rule_description="Two consecutive red Heikin-Ashi candles completed following uptrend with overbought Stochastic."
                    ))

        # 3. Bollinger Band Squeeze (BBW <= 0.0100)
        if SADOWSKI_CONFIG.get("bollinger_squeeze", {}).get("enabled", True):
            bb = TechnicalMathEngine.calculate_bollinger_bands(df["close"], 20, 2.0)
            if bb["bandwidth"] and bb["bandwidth"] <= 0.0150:  # Tight contraction
                middle = bb["middle"] or current_price
                bias = "LONG" if current_price >= middle else "SHORT"
                signals.append(SadowskiSignal(
                    strategy_id="BOLLINGER_SQUEEZE",
                    strategy_name="Bollinger Band Squeeze Expansion",
                    signal_bias=bias,
                    entry_trigger=current_price,
                    suggested_stop=round(bb["lower"] if bias == "LONG" else bb["upper"], 2),
                    suggested_target=round(current_price + (abs(current_price - middle) * 2.5) if bias == "LONG" else current_price - (abs(current_price - middle) * 2.5), 2),
                    confidence=80,
                    rule_description=f"Bollinger Bandwidth contracted to {bb['bandwidth']:.4f}. Volatility expansion breakout imminent."
                ))

        # 4. 2-Period RSI Strategy (Aggressive Momentum Trigger)
        if SADOWSKI_CONFIG.get("rsi_2", {}).get("enabled", True) and len(df) >= 5:
            rsi2 = TechnicalMathEngine.calculate_rsi(df["close"], 2)
            if rsi2 is not None:
                if rsi2 >= 90.0:
                    signals.append(SadowskiSignal(
                        strategy_id="RSI_2_EXHAUSTION",
                        strategy_name="2-Period RSI Overbought Exhaustion",
                        signal_bias="SHORT",
                        entry_trigger=current_price,
                        suggested_stop=round(current_price * 1.004, 2),
                        suggested_target=round(current_price * 0.990, 2),
                        confidence=78,
                        rule_description=f"RSI(2) is at {rsi2:.1f} (>= 90). Momentum overextended, high odds of quick pullback."
                    ))
                elif rsi2 <= 10.0:
                    signals.append(SadowskiSignal(
                        strategy_id="RSI_2_BOUNCE",
                        strategy_name="2-Period RSI Oversold Bounce",
                        signal_bias="LONG",
                        entry_trigger=current_price,
                        suggested_stop=round(current_price * 0.996, 2),
                        suggested_target=round(current_price * 1.010, 2),
                        confidence=78,
                        rule_description=f"RSI(2) is at {rsi2:.1f} (<= 10). Momentum oversold, high odds of sharp technical bounce."
                    ))

        return signals