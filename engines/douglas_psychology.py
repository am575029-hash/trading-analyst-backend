from __future__ import annotations

import math
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

class DouglasValidation(BaseModel):
    is_valid: bool
    capital_at_risk: float
    risk_percentage: float
    recommended_position_units: float
    estimated_position_value: float
    breakeven_trigger_price: float
    scale_out_plan: List[Dict[str, Any]]
    psychological_reminders: List[str]
    rule_violations: List[str]

class DouglasPsychologyEngine:
    """
    Enforces Mark Douglas's core principles from "Trading in the Zone":
    1. Predefine risk prior to trade entry (Fundamental Truth 1-5).
    2. Enforce fractional position sizing (1.0% - 3.0% maximum equity risk).
    3. Mandatory Breakeven Trigger Rule at +1.0R to lock in a 'Risk-Free Opportunity'.
    4. 3-Part Scale-Out Management ("I pay myself as the market makes money available").
    """

    FIVE_FUNDAMENTAL_TRUTHS = [
        "1. Anything can happen.",
        "2. You don't need to know what is going to happen next in order to make money.",
        "3. There is a random distribution between wins and losses for any given set of variables that define an edge.",
        "4. An edge is nothing more than an indication of a higher probability of one thing happening over another.",
        "5. Every moment in the market is unique."
    ]

    @classmethod
    def evaluate(
        cls,
        account_capital: float,
        risk_pct: float,
        entry_price: float,
        stop_loss: float,
        target_1: float,
        target_2: float,
        market: str = "CRYPTO"
    ) -> DouglasValidation:
        violations: List[str] = []

        # Zero or invalid stop loss check
        if entry_price == stop_loss or entry_price <= 0 or stop_loss <= 0:
            return DouglasValidation(
                is_valid=False,
                capital_at_risk=0.0,
                risk_percentage=0.0,
                recommended_position_units=0.0,
                estimated_position_value=0.0,
                breakeven_trigger_price=entry_price,
                scale_out_plan=[],
                psychological_reminders=["Mark Douglas Principle: Never enter a trade without predefining your stop loss level."],
                rule_violations=["Entry price cannot equal Stop Loss."]
            )

        # Enforce Mark Douglas 1% - 3% risk cap
        effective_risk_pct = min(3.0, max(0.2, risk_pct))
        if risk_pct > 3.0:
            violations.append(f"Risk percentage ({risk_pct}%) exceeded 3.0% maximum. Clamped to 3.0% per Douglas risk limits.")

        capital_at_risk = round(account_capital * (effective_risk_pct / 100.0), 2)
        stop_distance = abs(entry_price - stop_loss)
        is_long = entry_price > stop_loss

        # Position Sizing = Capital at Risk / Stop Distance
        raw_units = capital_at_risk / stop_distance
        precision = 4 if market.upper() == "CRYPTO" else 0
        units = round(raw_units, precision) if precision > 0 else max(1, math.floor(raw_units))
        position_val = round(units * entry_price, 2)

        # Breakeven trigger at +1.0R (Risk-Free Opportunity)
        be_price = round(entry_price + (stop_distance if is_long else -stop_distance), 2)

        # Scale out plan divided into thirds (1/3 at 1.5R, 1/3 at 2.5R, 1/3 trailing)
        t1 = target_1 or round(entry_price + (stop_distance * 1.5 if is_long else -stop_distance * 1.5), 2)
        t2 = target_2 or round(entry_price + (stop_distance * 2.5 if is_long else -stop_distance * 2.5), 2)

        dist1 = abs(t1 - entry_price)
        dist2 = abs(t2 - entry_price)
        rr1 = round(dist1 / stop_distance, 2)
        rr2 = round(dist2 / stop_distance, 2)

        scale_out = [
            {
                "step": "Scale 1 (33%)",
                "target_price": t1,
                "risk_reward_ratio": rr1,
                "action": "Bank 33% profit; immediately trail protective stop to Breakeven (1.0R)."
            },
            {
                "step": "Scale 2 (33%)",
                "target_price": t2,
                "risk_reward_ratio": rr2,
                "action": "Bank next 33% at structural target; trail final third with trailing swing stops."
            },
            {
                "step": "Scale 3 (34%)",
                "target_price": round(entry_price + (stop_distance * 3.5 if is_long else -stop_distance * 3.5), 2),
                "risk_reward_ratio": round(3.5, 2),
                "action": "Let winner run into macro liquidity; protect all accumulated profits."
            }
        ]

        reminders = [
            f"Casino Mindset: Evaluate this trade as 1 event of a 20-trade sample size.",
            f"Risk-Free Opportunity: Stop moves to Entry once price touches {be_price} (+1.0R)."
        ]

        return DouglasValidation(
            is_valid=len(violations) == 0,
            capital_at_risk=capital_at_risk,
            risk_percentage=effective_risk_pct,
            recommended_position_units=units,
            estimated_position_value=position_val,
            breakeven_trigger_price=be_price,
            scale_out_plan=scale_out,
            psychological_reminders=reminders,
            rule_violations=violations
        )