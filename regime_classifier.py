"""
regime_classifier.py — the three market MOODS, on price structure, causally.

The trader's frame (user, 2026-06-30): the market is in exactly one of three moods
and the right action differs by mood:

  • BIG TREND   (up/down) — strong, efficient directional travel. Continuation pays;
                            a trend-aligned lean is the one place a directional bet
                            has any business being taken.
  • SMALL TREND (up/down) — mild drift, low conviction. Lean lightly or not at all.
  • CONSOLIDATION         — "both-side stop-loss hunting". Price thrashes, nets nowhere.
                            This is where naked-option arrows go to die (whipsaw, you
                            pay the premium inflation and the reversion both ways).

This module turns that frame into a transparent, LOOKAHEAD-FREE label. No new alpha is
claimed — it is a calibrated CONTEXT/GATE so the desk stops emitting directional arrows
into the chop where the standing data says they only bleed (see backtest_engine cost
floor; intraday_scout HONESTY note). The whole bet is SUBTRACTIVE: trade less in the
mood that hunts stops.

METHOD (every input uses only bars up to and including the bar being labelled)
  • Kaufman Efficiency Ratio (ER) over N bars = |net travel| / Σ|bar-to-bar travel|.
    ER→1 = clean one-way trend; ER→0 = lots of motion, no progress = the SL-hunt chop.
    ER is the cleanest trend-vs-range discriminator and is causal by construction.
  • Drift in ATR units = (close - close[-N]) / ATR — scales BIG vs SMALL and signs it.
  • Mapping:
        ER >= er_hi  and |drift| >= drift_big   → BIG_TREND   (sign = drift sign)
        ER >= er_lo                              → SMALL_TREND (sign = drift sign)
        else                                     → CONSOLIDATION (no sign)

Thresholds are deliberately tunable — calibrate on history, do not trust defaults
blind. backtest_regime.py sweeps them and proves (or kills) the "stops cluster in
consolidation" thesis before this gates anything live.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

# ── Defaults (calibrate via backtest_regime.py; intraday ER runs low/noisy) ──────
ER_WINDOW   = 14      # bars over which efficiency + drift are measured
ER_HI       = 0.45    # ER at/above this = efficient enough to call a BIG trend
ER_LO       = 0.30    # ER at/above this = a real (small) trend; below = chop
DRIFT_BIG   = 2.0     # |net travel| in ATRs to qualify BIG (with ER_HI)

BIG_UP, BIG_DOWN     = "BIG_TREND_UP", "BIG_TREND_DOWN"
SMALL_UP, SMALL_DOWN = "SMALL_TREND_UP", "SMALL_TREND_DOWN"
CHOP                 = "CONSOLIDATION"

TREND_UP   = {BIG_UP, SMALL_UP}
TREND_DOWN = {BIG_DOWN, SMALL_DOWN}
BIG        = {BIG_UP, BIG_DOWN}


@dataclass
class Mood:
    mood: str               # one of the labels above
    er: float               # efficiency ratio in [0,1] (nan until warm)
    drift_atr: float        # signed net travel in ATR units
    sign: int               # +1 up, -1 down, 0 chop

    @property
    def is_chop(self) -> bool:
        return self.mood == CHOP

    @property
    def direction(self) -> str:
        return "UP" if self.sign > 0 else "DOWN" if self.sign < 0 else "FLAT"

    @property
    def short(self) -> str:
        return {BIG_UP: "BIG↑", BIG_DOWN: "BIG↓", SMALL_UP: "small↑",
                SMALL_DOWN: "small↓", CHOP: "CHOP"}[self.mood]


# ── Causal primitives ───────────────────────────────────────────────────────────
def efficiency_ratio(close: pd.Series, n: int = ER_WINDOW) -> pd.Series:
    """Kaufman ER over the trailing n bars. Causal (only past+current bars)."""
    net = (close - close.shift(n)).abs()
    path = close.diff().abs().rolling(n).sum()
    return (net / path.replace(0, np.nan)).clip(0, 1)


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()   # Wilder ATR


def _label(er: float, drift_atr: float,
           er_hi: float, er_lo: float, drift_big: float) -> tuple[str, int]:
    if not np.isfinite(er) or not np.isfinite(drift_atr):
        return CHOP, 0
    sign = 1 if drift_atr > 0 else -1 if drift_atr < 0 else 0
    if er >= er_hi and abs(drift_atr) >= drift_big and sign:
        return (BIG_UP if sign > 0 else BIG_DOWN), sign
    if er >= er_lo and sign:
        return (SMALL_UP if sign > 0 else SMALL_DOWN), sign
    return CHOP, 0


# ── Vectorised: label every bar of an OHLCV frame (for backtest) ─────────────────
def classify_series(df: pd.DataFrame, n: int = ER_WINDOW, er_hi: float = ER_HI,
                    er_lo: float = ER_LO, drift_big: float = DRIFT_BIG) -> pd.DataFrame:
    """Return df indexed-aligned columns: er, drift_atr, mood, sign. Causal."""
    close = df["close"].astype("float64")
    er = efficiency_ratio(close, n)
    atr = _atr(df)
    drift = (close - close.shift(n)) / atr.replace(0, np.nan)
    moods, signs = [], []
    for e, d in zip(er.to_numpy(), drift.to_numpy()):
        m, s = _label(e, d, er_hi, er_lo, drift_big)
        moods.append(m); signs.append(s)
    return pd.DataFrame({"er": er.values, "drift_atr": drift.values,
                         "mood": moods, "sign": signs}, index=df.index)


# ── Point classify: the mood AS OF a bar index (live + replay) ───────────────────
def classify_at(df: pd.DataFrame, i: Optional[int] = None, n: int = ER_WINDOW,
                er_hi: float = ER_HI, er_lo: float = ER_LO,
                drift_big: float = DRIFT_BIG) -> Mood:
    """Mood at row i (default last row) using only rows 0..i. df = OHLCV, time-asc."""
    if df is None or len(df) == 0:
        return Mood(CHOP, float("nan"), float("nan"), 0)
    i = len(df) - 1 if i is None else i
    sub = df.iloc[: i + 1]
    if len(sub) <= n:
        return Mood(CHOP, float("nan"), float("nan"), 0)
    close = sub["close"].astype("float64")
    er = float(efficiency_ratio(close, n).iloc[-1])
    atr = float(_atr(sub).iloc[-1])
    drift = (float(close.iloc[-1]) - float(close.iloc[-1 - n])) / atr if atr else float("nan")
    mood, sign = _label(er, drift, er_hi, er_lo, drift_big)
    return Mood(mood, round(er, 3), round(drift, 2) if np.isfinite(drift) else drift, sign)


def classify_from_bars(ser: dict, n: int = ER_WINDOW, er_hi: float = ER_HI,
                       er_lo: float = ER_LO, drift_big: float = DRIFT_BIG) -> Mood:
    """Mood from a build_series-style dict of OHLC lists (intraday_scout `ser`).

    Lookahead-free, matching price_structure.analyze: drops the last (forming) bar so
    the mood is read off CLOSED bars only.
    """
    try:
        df = pd.DataFrame({k: pd.to_numeric(pd.Series(ser.get(k) or []), errors="coerce")
                           for k in ("open", "high", "low", "close")}).dropna()
    except Exception:
        return Mood(CHOP, float("nan"), float("nan"), 0)
    if len(df) < n + 2:
        return Mood(CHOP, float("nan"), float("nan"), 0)
    df = df.iloc[:-1].reset_index(drop=True)          # drop forming bar
    return classify_at(df, None, n, er_hi, er_lo, drift_big)


def veto(mood: Mood, direction: str) -> tuple[bool, str]:
    """Veto the arrow when the mood is CONSOLIDATION (the both-side SL-hunt chop the
    day-trend read lets through as NEUTRAL). Trend moods never veto here — counter-trend
    is price_structure.trend_veto's job. Pure read; caller decides whether to act."""
    if direction not in ("CE", "PE") or mood is None:
        return False, ""
    if mood.is_chop:
        er_txt = f"{mood.er:.2f}" if np.isfinite(mood.er) else "n/a"
        return True, (f"VETO {direction}: CONSOLIDATION (efficiency {er_txt}) — chop / "
                      f"both-side SL hunting, no trend to ride")
    return False, ""
