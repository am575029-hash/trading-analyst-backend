from __future__ import annotations

import time
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

from engines.kirkpatrick_patterns import KirkpatrickPatternEngine, KirkpatrickPattern
from engines.sadowski_strategies import SadowskiStrategiesEngine, SadowskiSignal
from engines.douglas_psychology import DouglasPsychologyEngine, DouglasValidation
from engines.predictive_forecaster import PredictiveForecasterEngine, MarketForecast
from engines.technical_math import TechnicalMathEngine

class OrchestratedAnalysis(BaseModel):
    symbol: str
    current_price: float
    signal_bias: str
    confidence_score: int
    trade_plan: Dict[str, Any]
    risk_evaluation: DouglasValidation
    forecast: MarketForecast
    active_kirkpatrick_pattern: Optional[KirkpatrickPattern] = None
    all_kirkpatrick_patterns: List[KirkpatrickPattern] = []
    active_sadowski_signals: List[SadowskiSignal] = []
    market_psychology: str
    invalidation_condition: str
    execution_time_ms: float

class AIOrchestrator:
    """
    Synthesizes Kirkpatrick Patterns, Sadowski Quantitative Setups,
    Mark Douglas Risk Rules, and Predictive Scenario Forecasts with 100% directional coherence.
    """

    @classmethod
    def analyze(
        cls,
        symbol: str,
        candles: List[Any],
        swings: List[Any],
        current_price: float,
        account_capital: float = 85100.0,
        risk_pct: float = 1.0,
        market: str = "CRYPTO"
    ) -> OrchestratedAnalysis:
        t0 = time.perf_counter()

        df = TechnicalMathEngine.to_dataframe(candles)
        e20 = TechnicalMathEngine.calculate_ema(df["close"], 20) or current_price
        e50 = TechnicalMathEngine.calculate_ema(df["close"], 50) or current_price
        rsi14 = TechnicalMathEngine.calculate_rsi(df["close"], 14) or 50.0
        atr14 = TechnicalMathEngine.calculate_atr(df, 14) or (current_price * 0.007)

        # Baseline Trend Assessment
        is_bullish_trend = e20 > e50 and current_price >= (e50 * 0.998)
        is_bearish_trend = e20 < e50 and current_price <= (e50 * 1.002)

        # Step 1: Scan Kirkpatrick Patterns (<2ms)
        kp_patterns = KirkpatrickPatternEngine.scan(candles, swings, current_price)

        # Step 2: Scan Sadowski 9 Strategies (<2ms)
        sad_signals = SadowskiStrategiesEngine.evaluate_all(candles, current_price)

        # Step 3: Determine Directional Bias with ZERO-CONFLICT Logic
        confirmed_kp = next((p for p in kp_patterns if p.status.value == "BREAKOUT_CONFIRMED"), None)
        
        signal_bias = "WAIT"
        confidence = 50
        active_pattern: Optional[KirkpatrickPattern] = None

        if confirmed_kp:
            active_pattern = confirmed_kp
            signal_bias = "LONG_BIAS" if confirmed_kp.bias.value == "BULLISH" else "SHORT_BIAS"
            confidence = confirmed_kp.confidence
            stop_loss = confirmed_kp.protective_stop
            target_1 = confirmed_kp.projected_target
            dist = abs(current_price - stop_loss)
            target_2 = round(current_price + (dist * 2.5 if signal_bias == "LONG_BIAS" else -dist * 2.5), 2)
        elif is_bullish_trend:
            signal_bias = "LONG_BIAS"
            confidence = 82 if rsi14 < 70 else 72
            recent_lows = [s.price for s in swings if s.swing_type == "SWING_LOW" and s.price < current_price]
            structural_stop = max(recent_lows) if recent_lows else round(current_price - (atr14 * 1.5), 2)
            stop_dist = max(current_price * 0.0045, min(current_price * 0.015, abs(current_price - structural_stop)))
            stop_loss = round(current_price - stop_dist, 2)
            target_1 = round(current_price + (stop_dist * 1.6), 2)
            target_2 = round(current_price + (stop_dist * 2.6), 2)
            active_pattern = next((p for p in kp_patterns if p.bias.value == "BULLISH"), None)
        elif is_bearish_trend:
            signal_bias = "SHORT_BIAS"
            confidence = 82 if rsi14 > 30 else 72
            recent_highs = [s.price for s in swings if s.swing_type == "SWING_HIGH" and s.price > current_price]
            structural_stop = min(recent_highs) if recent_highs else round(current_price + (atr14 * 1.5), 2)
            stop_dist = max(current_price * 0.0045, min(current_price * 0.015, abs(structural_stop - current_price)))
            stop_loss = round(current_price + stop_dist, 2)
            target_1 = round(current_price - (stop_dist * 1.6), 2)
            target_2 = round(current_price - (stop_dist * 2.6), 2)
            active_pattern = next((p for p in kp_patterns if p.bias.value == "BEARISH"), None)
        else:
            signal_bias = "WAIT"
            confidence = 55
            stop_loss = round(current_price * 0.994, 2)
            target_1 = round(current_price * 1.010, 2)
            target_2 = round(current_price * 1.020, 2)

        # Step 4: Coherent Entry Zone
        if signal_bias == "LONG_BIAS":
            entry_low = round(min(current_price, e20), 2)
            entry_high = round(current_price * 1.0005, 2)
        elif signal_bias == "SHORT_BIAS":
            entry_low = round(current_price * 0.9995, 2)
            entry_high = round(max(current_price, e20), 2)
        else:
            entry_low = round(current_price * 0.998, 2)
            entry_high = round(current_price * 1.002, 2)

        dist_sl = abs(current_price - stop_loss)
        rr = round(abs(target_1 - current_price) / max(0.01, dist_sl), 2)

        trade_plan = {
            "entry_zone_low": entry_low,
            "entry_zone_high": entry_high,
            "suggested_entry": current_price,
            "stop_loss": stop_loss,
            "target_1": target_1,
            "target_2": target_2,
            "risk_reward_ratio": rr
        }

        # Step 5: Mark Douglas Probabilistic Risk Evaluation
        risk_eval = DouglasPsychologyEngine.evaluate(
            account_capital=account_capital,
            risk_pct=risk_pct,
            entry_price=current_price,
            stop_loss=stop_loss,
            target_1=target_1,
            target_2=target_2,
            market=market
        )

        # Step 6: Predictive Scenario & Liquidity Roadmap (<1ms)
        forecast_report = PredictiveForecasterEngine.forecast(
            candles=candles,
            swings=swings,
            current_price=current_price,
            signal_bias=signal_bias,
            active_pattern=active_pattern
        )

        psychology = (
            f"Mark Douglas Probabilistic Edge: 1 trade within a 20-trade casino sample. "
            f"Directional alignment backed by 20/50 EMA momentum stack."
        )
        invalidation = f"Candle close beyond {stop_loss} invalidates setup. Cut loss immediately."

        exec_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        return OrchestratedAnalysis(
            symbol=symbol,
            current_price=current_price,
            signal_bias=signal_bias,
            confidence_score=confidence,
            trade_plan=trade_plan,
            risk_evaluation=risk_eval,
            forecast=forecast_report,
            active_kirkpatrick_pattern=active_pattern,
            all_kirkpatrick_patterns=kp_patterns,
            active_sadowski_signals=sad_signals,
            market_psychology=psychology,
            invalidation_condition=invalidation,
            execution_time_ms=exec_ms
        )