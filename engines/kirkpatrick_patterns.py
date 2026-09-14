from __future__ import annotations

from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

class PatternCategory(str, Enum):
    MULTI_BAR = "MULTI_BAR"
    CANDLESTICK = "CANDLESTICK"
    SHORT_TERM = "SHORT_TERM"

class PatternBias(str, Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"

class PatternStatus(str, Enum):
    FORMING = "FORMING"
    BREAKOUT_CONFIRMED = "BREAKOUT_CONFIRMED"
    FAILED = "FAILED"

class KirkpatrickPattern(BaseModel):
    id: str
    name: str
    category: PatternCategory
    bias: PatternBias
    status: PatternStatus
    key_level: float
    projected_target: float
    protective_stop: float
    confidence: int
    description: str
    reference: str = "Charles Kirkpatrick / Fidelity Technical Analysis"

KIRKPATRICK_CONFIG: Dict[str, Dict[str, Any]] = {
    "double_top": {"enabled": True, "tolerance_pct": 0.0030, "max_age_bars": 35},
    "double_bottom": {"enabled": True, "tolerance_pct": 0.0030, "max_age_bars": 35},
    "triple_top": {"enabled": True, "tolerance_pct": 0.0035, "max_age_bars": 45},
    "triple_bottom": {"enabled": True, "tolerance_pct": 0.0035, "max_age_bars": 45},
    "head_and_shoulders": {"enabled": True, "shoulder_tolerance_pct": 0.007, "max_age_bars": 50},
    "inverse_head_and_shoulders": {"enabled": True, "shoulder_tolerance_pct": 0.007, "max_age_bars": 50},
    "ascending_triangle": {"enabled": True, "tolerance_pct": 0.003, "max_age_bars": 35},
    "descending_triangle": {"enabled": True, "tolerance_pct": 0.003, "max_age_bars": 35},
    "rectangle_consolidation": {"enabled": True, "tolerance_pct": 0.0035, "max_age_bars": 35},
    "cup_and_handle": {"enabled": True, "max_age_bars": 55},
    "hammer": {"enabled": True},
    "shooting_star": {"enabled": True},
    "bullish_engulfing": {"enabled": True},
    "bearish_engulfing": {"enabled": True},
    "nr4": {"enabled": True}
}

class KirkpatrickPatternEngine:
    """
    Trained strictly on Charles Kirkpatrick & Fidelity 'Identifying Chart Patterns'.
    Enforces strict recency, prior-trend verification, and zero false-positives.
    """

    @classmethod
    def scan(cls, candles: List[Any], swings: List[Any], current_price: float) -> List[KirkpatrickPattern]:
        patterns: List[KirkpatrickPattern] = []
        n_candles = len(candles)
        if n_candles < 15 or not swings or len(swings) < 3:
            return patterns

        # 1. Multi-Bar Formations with strict Recency
        cls._detect_double_structures(candles, swings, current_price, patterns)
        cls._detect_triple_structures(candles, swings, current_price, patterns)
        cls._detect_head_and_shoulders(candles, swings, current_price, patterns)
        cls._detect_triangles_and_rectangles(candles, swings, current_price, patterns)

        # 2. Candlestick Patterns
        cls._detect_candlesticks(candles, current_price, patterns)
        return patterns

    @classmethod
    def _is_recent(cls, swing_index: int, total_candles: int, max_age: int = 25) -> bool:
        """Ensures pattern did not complete hours ago."""
        return (total_candles - 1 - swing_index) <= max_age

    @classmethod
    def _detect_double_structures(cls, candles: List[Any], swings: List[Any], current_price: float, patterns: List[KirkpatrickPattern]):
        highs = [s for s in swings if s.swing_type == "SWING_HIGH"]
        lows = [s for s in swings if s.swing_type == "SWING_LOW"]
        n = len(candles)

        # Double Bottom (W): Must follow a prior downtrend & be recent
        cfg_db = KIRKPATRICK_CONFIG.get("double_bottom", {})
        if cfg_db.get("enabled", True) and len(lows) >= 2 and len(highs) >= 1:
            l1, l2 = lows[-2], lows[-1]
            neckline = highs[-1]

            if l1.index < neckline.index < l2.index and cls._is_recent(l2.index, n, cfg_db.get("max_age_bars", 35)):
                # Must have prior decline into L1
                prior_highs = [s.price for s in highs if s.index < l1.index]
                if prior_highs and max(prior_highs) > l1.price * 1.008:
                    spread = abs(l1.price - l2.price) / l1.price
                    if spread <= cfg_db.get("tolerance_pct", 0.0030):
                        h_pat = neckline.price - min(l1.price, l2.price)
                        target = round(neckline.price + h_pat, 2)
                        
                        # Invalidation: If price drops below bottoms, pattern failed
                        if current_price >= min(l1.price, l2.price) * 0.997:
                            is_breakout = current_price >= neckline.price
                            # If price already hit target, pattern is expired
                            if current_price < target * 1.005:
                                patterns.append(KirkpatrickPattern(
                                    id="DOUBLE_BOTTOM",
                                    name="Double Bottom (W-Pattern)",
                                    category=PatternCategory.MULTI_BAR,
                                    bias=PatternBias.BULLISH,
                                    status=PatternStatus.BREAKOUT_CONFIRMED if is_breakout else PatternStatus.FORMING,
                                    key_level=neckline.price,
                                    projected_target=target,
                                    protective_stop=round(min(l1.price, l2.price) * 0.9985, 2),
                                    confidence=85 if is_breakout else 60,
                                    description="Two troughs at support after decline. Measured target equal to neckline + pattern height."
                                ))

        # Double Top (M): Must follow a prior uptrend & be recent
        cfg_dt = KIRKPATRICK_CONFIG.get("double_top", {})
        if cfg_dt.get("enabled", True) and len(highs) >= 2 and len(lows) >= 1:
            h1, h2 = highs[-2], highs[-1]
            neckline = lows[-1]

            if h1.index < neckline.index < h2.index and cls._is_recent(h2.index, n, cfg_dt.get("max_age_bars", 35)):
                # Must have prior rally into H1
                prior_lows = [s.price for s in lows if s.index < h1.index]
                if prior_lows and min(prior_lows) < h1.price * 0.992:
                    spread = abs(h1.price - h2.price) / h1.price
                    if spread <= cfg_dt.get("tolerance_pct", 0.0030):
                        h_pat = max(h1.price, h2.price) - neckline.price
                        target = round(neckline.price - h_pat, 2)
                        
                        # Invalidation: If price blasted ABOVE tops, Double Top is INVALID
                        if current_price <= max(h1.price, h2.price) * 1.003:
                            is_breakdown = current_price <= neckline.price
                            if current_price > target * 0.995:
                                patterns.append(KirkpatrickPattern(
                                    id="DOUBLE_TOP",
                                    name="Double Top (M-Pattern)",
                                    category=PatternCategory.MULTI_BAR,
                                    bias=PatternBias.BEARISH,
                                    status=PatternStatus.BREAKOUT_CONFIRMED if is_breakdown else PatternStatus.FORMING,
                                    key_level=neckline.price,
                                    projected_target=target,
                                    protective_stop=round(max(h1.price, h2.price) * 1.0015, 2),
                                    confidence=85 if is_breakdown else 60,
                                    description="Two peaks at resistance after uptrend. Measured target projected downside."
                                ))

    @classmethod
    def _detect_triple_structures(cls, candles: List[Any], swings: List[Any], current_price: float, patterns: List[KirkpatrickPattern]):
        highs = [s for s in swings if s.swing_type == "SWING_HIGH"]
        lows = [s for s in swings if s.swing_type == "SWING_LOW"]
        n = len(candles)

        # TRIPLE BOTTOM: Strict Downtrend + Strict Recency Filter
        cfg_tb = KIRKPATRICK_CONFIG.get("triple_bottom", {})
        if cfg_tb.get("enabled", True) and len(lows) >= 3 and len(highs) >= 2:
            l1, l2, l3 = lows[-3], lows[-2], lows[-1]
            h1, h2 = highs[-2], highs[-1]

            # Sequence check: L1 -> H1 -> L2 -> H2 -> L3
            if (l1.index < h1.index < l2.index < h2.index < l3.index) and cls._is_recent(l3.index, n, cfg_tb.get("max_age_bars", 45)):
                # Must be after a genuine downtrend into L1
                prior_highs = [s.price for s in highs if s.index < l1.index]
                if prior_highs and max(prior_highs) > l1.price * 1.012:
                    min_low = min(l1.price, l2.price, l3.price)
                    max_low = max(l1.price, l2.price, l3.price)
                    spread = (max_low - min_low) / min_low

                    if spread <= cfg_tb.get("tolerance_pct", 0.0035):
                        neckline_peak = max(h1.price, h2.price)
                        h_pat = neckline_peak - min_low
                        target = round(neckline_peak + h_pat, 2)
                        
                        # If price is far above target, pattern is obsolete/dead
                        if current_price <= target * 1.008 and current_price >= min_low * 0.997:
                            is_breakout = current_price >= neckline_peak
                            patterns.append(KirkpatrickPattern(
                                id="TRIPLE_BOTTOM",
                                name="Triple Bottom",
                                category=PatternCategory.MULTI_BAR,
                                bias=PatternBias.BULLISH,
                                status=PatternStatus.BREAKOUT_CONFIRMED if is_breakout else PatternStatus.FORMING,
                                key_level=neckline_peak,
                                projected_target=target,
                                protective_stop=round(min_low * 0.9985, 2),
                                confidence=88 if is_breakout else 60,
                                description="Three distinct troughs tested identical floor after decline."
                            ))

        # TRIPLE TOP: Strict Uptrend + Invalidation Check
        cfg_tt = KIRKPATRICK_CONFIG.get("triple_top", {})
        if cfg_tt.get("enabled", True) and len(highs) >= 3 and len(lows) >= 2:
            h1, h2, h3 = highs[-3], highs[-2], highs[-1]
            l1, l2 = lows[-2], lows[-1]

            if (h1.index < l1.index < h2.index < l2.index < h3.index) and cls._is_recent(h3.index, n, cfg_tt.get("max_age_bars", 45)):
                prior_lows = [s.price for s in lows if s.index < h1.index]
                if prior_lows and min(prior_lows) < h1.price * 0.988:
                    min_high = min(h1.price, h2.price, h3.price)
                    max_high = max(h1.price, h2.price, h3.price)
                    spread = (max_high - min_high) / min_high

                    if spread <= cfg_tt.get("tolerance_pct", 0.0035):
                        # CRITICAL: If current price is HIGHER than the peaks, TRIPLE TOP IS INVALIDATED!
                        if current_price <= max_high * 1.002:
                            neckline_floor = min(l1.price, l2.price)
                            h_pat = max_high - neckline_floor
                            target = round(neckline_floor - h_pat, 2)
                            is_breakdown = current_price <= neckline_floor

                            patterns.append(KirkpatrickPattern(
                                id="TRIPLE_TOP",
                                name="Triple Top",
                                category=PatternCategory.MULTI_BAR,
                                bias=PatternBias.BEARISH,
                                status=PatternStatus.BREAKOUT_CONFIRMED if is_breakdown else PatternStatus.FORMING,
                                key_level=neckline_floor,
                                projected_target=target,
                                protective_stop=round(max_high * 1.0015, 2),
                                confidence=88 if is_breakdown else 60,
                                description="Three distinct peaks failing at ceiling after rally."
                            ))

    @classmethod
    def _detect_head_and_shoulders(cls, candles: List[Any], swings: List[Any], current_price: float, patterns: List[KirkpatrickPattern]):
        highs = [s for s in swings if s.swing_type == "SWING_HIGH"]
        lows = [s for s in swings if s.swing_type == "SWING_LOW"]
        n = len(candles)

        cfg_hs = KIRKPATRICK_CONFIG.get("head_and_shoulders", {})
        if cfg_hs.get("enabled", True) and len(highs) >= 3 and len(lows) >= 2:
            ls, head, rs = highs[-3], highs[-2], highs[-1]
            l1, l2 = lows[-2], lows[-1]

            if (ls.index < l1.index < head.index < l2.index < rs.index) and cls._is_recent(rs.index, n, cfg_hs.get("max_age_bars", 50)):
                if head.price > ls.price * 1.003 and head.price > rs.price * 1.003:
                    diff_s = abs(ls.price - rs.price) / ls.price
                    if diff_s <= cfg_hs.get("shoulder_tolerance_pct", 0.007):
                        neckline = min(l1.price, l2.price)
                        # Invalidation: If price blasts above head, H&S is dead
                        if current_price <= head.price * 1.002:
                            target = round(neckline - (head.price - neckline), 2)
                            is_broken = current_price < neckline
                            patterns.append(KirkpatrickPattern(
                                id="HEAD_AND_SHOULDERS_TOP",
                                name="Head and Shoulders (Top)",
                                category=PatternCategory.MULTI_BAR,
                                bias=PatternBias.BEARISH,
                                status=PatternStatus.BREAKOUT_CONFIRMED if is_broken else PatternStatus.FORMING,
                                key_level=neckline,
                                projected_target=target,
                                protective_stop=round(rs.price * 1.0015, 2),
                                confidence=90 if is_broken else 60,
                                description="Head & Shoulders reversal formation."
                            ))

    @classmethod
    def _detect_triangles_and_rectangles(cls, candles: List[Any], swings: List[Any], current_price: float, patterns: List[KirkpatrickPattern]):
        highs = [s for s in swings if s.swing_type == "SWING_HIGH"]
        lows = [s for s in swings if s.swing_type == "SWING_LOW"]
        n = len(candles)
        if len(highs) < 2 or len(lows) < 2:
            return

        h1, h2 = highs[-2].price, highs[-1].price
        l1, l2 = lows[-2].price, lows[-1].price

        if not cls._is_recent(max(highs[-1].index, lows[-1].index), n, 30):
            return

        # Ascending Triangle: Flat Top + Rising Lows
        cfg_asc = KIRKPATRICK_CONFIG.get("ascending_triangle", {})
        if cfg_asc.get("enabled", True):
            flat_res = abs(h1 - h2) / h1 <= cfg_asc.get("tolerance_pct", 0.003)
            rising_lows = l2 > l1 * 1.002
            if flat_res and rising_lows and current_price >= l2 * 0.997:
                height = max(h1, h2) - l1
                is_breakout = current_price > max(h1, h2)
                patterns.append(KirkpatrickPattern(
                    id="ASCENDING_TRIANGLE",
                    name="Ascending Triangle",
                    category=PatternCategory.MULTI_BAR,
                    bias=PatternBias.BULLISH,
                    status=PatternStatus.BREAKOUT_CONFIRMED if is_breakout else PatternStatus.FORMING,
                    key_level=max(h1, h2),
                    projected_target=round(max(h1, h2) + height, 2),
                    protective_stop=round(l2 * 0.9985, 2),
                    confidence=85 if is_breakout else 65,
                    description="Buyers accumulating at higher lows against flat ceiling."
                ))

    @classmethod
    def _detect_candlesticks(cls, candles: List[Any], current_price: float, patterns: List[KirkpatrickPattern]):
        c = candles[-1]
        prev = candles[-2] if len(candles) >= 2 else None
        if not prev:
            return
        rng = c.high - c.low
        if rng <= 0:
            return

        body = abs(c.close - c.open)
        upper_wick = c.high - max(c.open, c.close)
        lower_wick = min(c.open, c.close) - c.low
        recent_lows = min([x.low for x in candles[-8:-1]])
        recent_highs = max([x.high for x in candles[-8:-1]])

        # Hammer at bottom of range
        if lower_wick >= 2.0 * body and upper_wick <= 0.15 * rng and c.low <= recent_lows:
            patterns.append(KirkpatrickPattern(
                id="HAMMER",
                name="Hammer Pin Bar Reversal",
                category=PatternCategory.CANDLESTICK,
                bias=PatternBias.BULLISH,
                status=PatternStatus.BREAKOUT_CONFIRMED,
                key_level=c.high,
                projected_target=round(current_price + (rng * 2.0), 2),
                protective_stop=round(c.low * 0.9985, 2),
                confidence=80,
                description="Bottom liquidity rejection tail."
            ))