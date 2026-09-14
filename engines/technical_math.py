from __future__ import annotations

import math
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from pydantic import BaseModel

class HACandle(BaseModel):
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    is_green: bool

class TechnicalMathEngine:
    """
    Vectorized mathematical indicators computed purely using Pandas and NumPy.
    Sub-millisecond execution with zero external network dependencies.
    """

    @staticmethod
    def to_dataframe(candles: List[Any]) -> pd.DataFrame:
        if not candles:
            return pd.DataFrame()
        records = [
            {
                "timestamp": getattr(c, "timestamp", 0),
                "open": float(getattr(c, "open", 0.0)),
                "high": float(getattr(c, "high", 0.0)),
                "low": float(getattr(c, "low", 0.0)),
                "close": float(getattr(c, "close", 0.0)),
                "volume": float(getattr(c, "volume", 0.0))
            }
            for c in candles
        ]
        df = pd.DataFrame.from_records(records)
        df.sort_values(by="timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    @classmethod
    def calculate_ema(cls, series: pd.Series, period: int) -> Optional[float]:
        if len(series) < period:
            return None
        val = series.ewm(span=period, adjust=False).mean().iloc[-1]
        return round(float(val), 4) if not pd.isna(val) else None

    @classmethod
    def calculate_rsi(cls, series: pd.Series, period: int = 14) -> Optional[float]:
        if len(series) <= period:
            return None
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

        g = avg_gain.iloc[-1]
        l = avg_loss.iloc[-1]
        if pd.isna(g) or pd.isna(l):
            return None
        if l == 0:
            return 100.0 if g > 0 else 50.0
        rs = g / l
        return round(float(100.0 - (100.0 / (1.0 + rs))), 2)

    @classmethod
    def calculate_stochastic(
        cls, df: pd.DataFrame, k_period: int = 14, d_period: int = 3, slowing: int = 3
    ) -> Tuple[Optional[float], Optional[float]]:
        """Stochastic Oscillator %K and %D with internal smoothing."""
        if len(df) < k_period + slowing:
            return None, None
        low_min = df["low"].rolling(window=k_period).min()
        high_max = df["high"].rolling(window=k_period).max()
        fast_k = 100.0 * ((df["close"] - low_min) / (high_max - low_min).replace(0, np.nan))
        k = fast_k.rolling(window=slowing).mean()
        d = k.rolling(window=d_period).mean()

        k_val = k.iloc[-1]
        d_val = d.iloc[-1]
        return (
            round(float(k_val), 2) if not pd.isna(k_val) else None,
            round(float(d_val), 2) if not pd.isna(d_val) else None
        )

    @classmethod
    def calculate_bollinger_bands(
        cls, series: pd.Series, period: int = 20, num_std: float = 2.0
    ) -> Dict[str, Optional[float]]:
        if len(series) < period:
            return {"upper": None, "middle": None, "lower": None, "bandwidth": None}
        sma = series.rolling(window=period).mean()
        std = series.rolling(window=period).std(ddof=0)
        upper = sma + (num_std * std)
        lower = sma - (num_std * std)
        u_val = upper.iloc[-1]
        m_val = sma.iloc[-1]
        l_val = lower.iloc[-1]
        bw = ((u_val - l_val) / m_val) if m_val and m_val != 0 else 0.0
        return {
            "upper": round(float(u_val), 4) if not pd.isna(u_val) else None,
            "middle": round(float(m_val), 4) if not pd.isna(m_val) else None,
            "lower": round(float(l_val), 4) if not pd.isna(l_val) else None,
            "bandwidth": round(float(bw), 4)
        }

    @classmethod
    def calculate_atr(cls, df: pd.DataFrame, period: int = 14) -> Optional[float]:
        if len(df) <= period:
            return None
        h = df["high"]
        l = df["low"]
        pc = df["close"].shift(1)
        tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean().iloc[-1]
        return round(float(atr), 4) if not pd.isna(atr) else None

    @classmethod
    def calculate_vwap(cls, df: pd.DataFrame) -> Optional[float]:
        if df.empty or "volume" not in df.columns:
            return None
        tv = df["volume"].sum()
        if tv <= 0:
            return None
        typical = (df["high"] + df["low"] + df["close"]) / 3.0
        return round(float((typical * df["volume"]).sum() / tv), 4)

    @classmethod
    def convert_to_heikin_ashi(cls, df: pd.DataFrame) -> List[HACandle]:
        """
        Roman Sadowski Heikin-Ashi formulas:
        HA_Close = (O + H + L + C) / 4
        HA_Open = (prev_HA_Open + prev_HA_Close) / 2
        HA_High = max(High, HA_Open, HA_Close)
        HA_Low = min(Low, HA_Open, HA_Close)
        """
        if df.empty:
            return []
        ha_list: List[HACandle] = []
        n = len(df)

        prev_open = float(df["open"].iloc[0])
        prev_close = float(df["close"].iloc[0])

        for i in range(n):
            row = df.iloc[i]
            ts = int(row["timestamp"])
            o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])

            ha_close = (o + h + l + c) / 4.0
            ha_open = (prev_open + prev_close) / 2.0 if i > 0 else (o + c) / 2.0
            ha_high = max(h, ha_open, ha_close)
            ha_low = min(l, ha_open, ha_close)

            is_green = ha_close >= ha_open
            ha_list.append(
                HACandle(
                    timestamp=ts,
                    open=round(ha_open, 4),
                    high=round(ha_high, 4),
                    low=round(ha_low, 4),
                    close=round(ha_close, 4),
                    is_green=is_green
                )
            )
            prev_open = ha_open
            prev_close = ha_close

        return ha_list