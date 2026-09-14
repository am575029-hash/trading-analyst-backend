from __future__ import annotations

import math
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import pandas as pd
import numpy as np

from engines.technical_math import TechnicalMathEngine

class ScenarioPath(BaseModel):
    name: str                          
    probability_pct: int               
    trigger_level: float               
    trigger_rule: str                  
    projected_target: float            
    invalidation_level: float          
    tactical_action: str               

class LiquidityPool(BaseModel):
    pool_type: str                     
    price_level: float
    description: str
    whale_trap_risk: str

class MarketForecast(BaseModel):
    volatility_state: str              
    expected_bar_span: str             
    projected_range_points: float      
    primary_scenario: ScenarioPath
    alternative_scenario: ScenarioPath
    liquidity_pools: List[LiquidityPool]
    early_warning_radar: str
    confluence_score: int              

class PredictiveForecasterEngine:
    """
    Predictive Scenario & Liquidity Forecaster optimized for scalpers.
    Calculates tight micro-targets (next 3-5 candles) and dual directional probabilities.
    """

    @classmethod
    def forecast(
        cls,
        candles: List[Any],
        swings: List[Any],
        current_price: float,
        signal_bias: str,
        active_pattern: Optional[Any] = None
    ) -> MarketForecast:
        df = TechnicalMathEngine.to_dataframe(candles)
        e20 = TechnicalMathEngine.calculate_ema(df["close"], 20) or current_price
        atr14 = TechnicalMathEngine.calculate_atr(df, 14) or (current_price * 0.003)
        bb = TechnicalMathEngine.calculate_bollinger_bands(df["close"], 20, 2.0)
        bandwidth = bb.get("bandwidth", 0.02)

        high_swings = [s for s in swings if s.swing_type == "SWING_HIGH" and s.price > current_price]
        low_swings = [s for s in swings if s.swing_type == "SWING_LOW" and s.price < current_price]

        liquidity_pools: List[LiquidityPool] = []

        # 1. Strict Scalper Upper BSL Target (Tight 0.4% - 1.0% range above current price)
        if high_swings:
            nearest_upper = min(high_swings, key=lambda x: x.price)
            # If swing is too far (> 0.8% away), use a tight scalping ATR step (0.6x ATR)
            if (nearest_upper.price - current_price) / current_price > 0.008:
                scalp_upper = round(current_price + (atr14 * 0.6), 2)
                liquidity_pools.append(LiquidityPool(
                    pool_type="BUY_SIDE_LIQUIDITY (BSL - SCALP TARGET)",
                    price_level=scalp_upper,
                    description=f"Immediate micro-resistance at {scalp_upper:.1f}.",
                    whale_trap_risk="Quick scalp profit target. Expect wick reaction."
                ))
            else:
                liquidity_pools.append(LiquidityPool(
                    pool_type="BUY_SIDE_LIQUIDITY (BSL - RESISTANCE)",
                    price_level=round(nearest_upper.price, 2),
                    description=f"Overhead pool at {nearest_upper.price:.1f}.",
                    whale_trap_risk="Magnetic scalping wave target."
                ))
        else:
            scalp_upper = round(current_price + (atr14 * 0.6), 2)
            liquidity_pools.append(LiquidityPool(
                pool_type="BUY_SIDE_LIQUIDITY (BSL - EXTENSION)",
                price_level=scalp_upper,
                description=f"Immediate micro-target at {scalp_upper:.1f}.",
                whale_trap_risk="Watch for high-volume absorption."
            ))

        # 2. Strict Scalper Lower SSL Support (Tight 0.4% - 1.0% range below current price)
        if low_swings:
            nearest_lower = max(low_swings, key=lambda x: x.price)
            if (current_price - nearest_lower.price) / current_price > 0.008:
                scalp_lower = round(current_price - (atr14 * 0.6), 2)
                liquidity_pools.append(LiquidityPool(
                    pool_type="SELL_SIDE_LIQUIDITY (SSL - SCALP SUPPORT)",
                    price_level=scalp_lower,
                    description=f"Immediate micro-support floor at {scalp_lower:.1f}.",
                    whale_trap_risk="Defensive trailing floor for scalps."
                ))
            else:
                liquidity_pools.append(LiquidityPool(
                    pool_type="SELL_SIDE_LIQUIDITY (SSL - SUPPORT)",
                    price_level=round(nearest_lower.price, 2),
                    description=f"Support cushion at {nearest_lower.price:.1f}.",
                    whale_trap_risk="Pullback support floor."
                ))
        else:
            scalp_lower = round(current_price - (atr14 * 0.6), 2)
            liquidity_pools.append(LiquidityPool(
                pool_type="SELL_SIDE_LIQUIDITY (SSL - FLOOR)",
                price_level=scalp_lower,
                description=f"Structural support cushion at {scalp_lower:.1f}.",
                whale_trap_risk="Breakdown triggers short-term stop cascade."
            ))

        # 3. Volatility State & Scalper Displacement (Next 3 to 5 bars)
        is_compressed = bandwidth <= 0.0120
        volatility_state = (
            "EXPANSION_IMMINENT" if is_compressed
            else ("TREND_CONTINUATION" if abs(current_price - e20) <= (atr14 * 1.5) else "MOMENTUM_OVEREXTENDED")
        )

        # Scalper displacement: tight 0.8x ATR step for immediate candle targets
        expected_displacement = round(atr14 * 0.8, 2)
        is_bullish = (signal_bias == "LONG_BIAS" or current_price >= e20)

        upper_pool = next((p for p in liquidity_pools if "BUY_SIDE_LIQUIDITY" in p.pool_type and p.price_level > current_price), None)
        lower_pool = next((p for p in liquidity_pools if "SELL_SIDE_LIQUIDITY" in p.pool_type and p.price_level < current_price), None)

        upper_target = upper_pool.price_level if upper_pool else round(current_price + expected_displacement, 2)
        lower_support = lower_pool.price_level if lower_pool else round(current_price - expected_displacement, 2)

        if is_bullish:
            path_a_prob = 76 if is_compressed else 72
            path_a_trigger = round(current_price + (atr14 * 0.15), 2)
            path_a_target = upper_target
            path_a_invalidation = round(min(e20, current_price - (atr14 * 0.5)), 2)

            scenario_a = ScenarioPath(
                name="Path A: Scalp Trend Continuation",
                probability_pct=path_a_prob,
                trigger_level=path_a_trigger,
                trigger_rule=f"Candle close above {path_a_trigger:.1f} confirms immediate push to target.",
                projected_target=path_a_target,
                invalidation_level=path_a_invalidation,
                tactical_action=f"Target exact micro-resistance at {path_a_target:.1f}. Trail stop to breakeven quickly."
            )

            scenario_b = ScenarioPath(
                name="Path B: Pullback & Support Test",
                probability_pct=(100 - path_a_prob),
                trigger_level=path_a_invalidation,
                trigger_rule=f"Dip below 20 EMA ({path_a_invalidation:.1f}).",
                projected_target=lower_support,
                invalidation_level=current_price,
                tactical_action=f"Exit long scalp. Wait for retest of lower support floor at {lower_support:.1f}."
            )

            radar_alert = (
                f"Bullish scalping stack active. Upper BSL target at {upper_target:.1f} is the immediate magnetic target (Next 3-5 bars), "
                f"while Lower SSL support at {lower_support:.1f} serves as the defensive floor."
            )
            confluence = 84

        else:
            path_a_prob = 76 if is_compressed else 72
            path_a_trigger = round(current_price - (atr14 * 0.15), 2)
            path_a_target = lower_support
            path_a_invalidation = round(max(e20, current_price + (atr14 * 0.5)), 2)

            scenario_a = ScenarioPath(
                name="Path A: Scalp Breakdown Flush",
                probability_pct=path_a_prob,
                trigger_level=path_a_trigger,
                trigger_rule=f"Candle close below {path_a_trigger:.1f} confirms downward scalp.",
                projected_target=path_a_target,
                invalidation_level=path_a_invalidation,
                tactical_action=f"Target lower support at {path_a_target:.1f}. Protect profits promptly."
            )

            scenario_b = ScenarioPath(
                name="Path B: Short Squeeze Rebound",
                probability_pct=(100 - path_a_prob),
                trigger_level=path_a_invalidation,
                trigger_rule=f"Reclaim of 20 EMA ({path_a_invalidation:.1f}).",
                projected_target=upper_target,
                invalidation_level=current_price,
                tactical_action=f"Cut short scalp immediately on reclaim of {path_a_invalidation:.1f}."
            )

            radar_alert = (
                f"Bearish scalping momentum active. Lower SSL support at {lower_support:.1f} is the immediate downside target, "
                f"while Upper BSL target at {upper_target:.1f} acts as the short defense ceiling."
            )
            confluence = 80

        return MarketForecast(
            volatility_state=volatility_state,
            expected_bar_span="Next 3 to 5 candles (15m)",
            projected_range_points=expected_displacement,
            primary_scenario=scenario_a,
            alternative_scenario=scenario_b,
            liquidity_pools=liquidity_pools,
            early_warning_radar=radar_alert,
            confluence_score=confluence
        )