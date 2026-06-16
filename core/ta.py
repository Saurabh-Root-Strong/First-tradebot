"""core.ta — small technical-analysis helpers shared across modules (EMA, ATR).

(The large indicators.py holds the heavyweight studies; these are the primitives
that kept getting copy-pasted into the lightweight analysis modules.)
"""
from __future__ import annotations

import pandas as pd


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Wilder ATR from an OHLC frame (columns: high, low, close)."""
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()
