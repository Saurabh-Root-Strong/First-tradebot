"""
feature_engine.py — causal intraday FEATURE + 8-REGIME + action engine (NSE index F&O).

The institutional decision-support foundation. Given OHLCV (+ optional option-chain /
futures snapshot) up to `as_of`, it computes the feature vector the desk actually reads,
classifies the session into one of 8 regimes, scores confidence, and maps regime → the
HONEST action (not a fake directional signal — this desk proved intraday taker-directional
is dead at cost; the real actions are risk-map / stand-aside / sell-theta / post-3pm carry).

Components (matches the desk spec):
  1. Feature Engine   — VWAP, ER, ADX, Choppiness, ATR + percentile, realised-vol +
                        vol-of-vol, relative-volume, range-position  (+ option/fut hooks:
                        ATM-IV, 25Δ skew, PCR, OI walls, basis, DTE/theta-clock).
  2. Regime Engine    — 8 regimes: OPENING / STRONG_TREND / WEAK_TREND / RANGE /
                        BREAKOUT / FALSE_BREAKOUT / HIGH_VOL / TRANSITION.
  3. Scoring          — rule-based confidence (feature agreement − conflict).
  4. Strategy Selector— regime → honest action (+ the post-3pm→BTST carry hook that turns
                        a late intraday setup into the validated overnight trade).

CAUSAL: every feature uses only bars/quotes ≤ as_of; the forming bar is dropped for
structure. No lookahead. Degrades gracefully when option/futures data is absent (price-
only regime still works — the price backbone is fully computable from 2yr OHLCV).
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import pandas as pd

import regime_classifier as rc

SESSION_OPEN = dt.time(9, 15)
SESSION_CLOSE = dt.time(15, 30)
POST3 = dt.time(15, 0)           # after this, a new setup carries overnight (BTST carve)

# 3 regimes — REDUCED from 8 by a forward-behaviour distinguishability test (regime_reduce,
# 2026-07-01, 482 NIFTY sessions). STRONG_TREND / WEAK_TREND / RANGE / TRANSITION / BREAKOUT
# were statistically IDENTICAL forward (Δmove within 2·SE, Δefficiency <0.02, direction ≈ coin
# — "STRONG_TREND" continued only 47%). Only HIGH_VOL separated (3× the forward move). Merging
# the 5 fakes loses ZERO predictive information and keeps the classifier production-trivial:
# a clock + one cross-day vol percentile.
OPENING, NORMAL, HIGH_VOL = "OPENING", "NORMAL", "HIGH_VOL"


# ── Causal price primitives ─────────────────────────────────────────────────────
def _ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def _atr(df, n=14):
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def _adx(df, n=14):
    h, l, c = df["high"], df["low"], df["close"]
    up = h.diff(); dn = -l.diff()
    plus = np.where((up > dn) & (up > 0), up, 0.0)
    minus = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / n, adjust=False).mean().replace(0, np.nan)
    pdi = 100 * pd.Series(plus, index=df.index).ewm(alpha=1 / n, adjust=False).mean() / atr
    mdi = 100 * pd.Series(minus, index=df.index).ewm(alpha=1 / n, adjust=False).mean() / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False).mean()


def _choppiness(df, n=14):
    """Choppiness Index: ~100 = choppy/range, ~0 = trending. log-scaled ATR-sum vs range."""
    atr1 = pd.concat([(df["high"] - df["low"]),
                      (df["high"] - df["close"].shift()).abs(),
                      (df["low"] - df["close"].shift()).abs()], axis=1).max(axis=1)
    sum_tr = atr1.rolling(n).sum()
    rng = df["high"].rolling(n).max() - df["low"].rolling(n).min()
    return 100 * np.log10((sum_tr / rng.replace(0, np.nan))) / np.log10(n)


def _vwap(df):
    tp = (df["high"] + df["low"] + df["close"]) / 3
    cum_v = df["volume"].cumsum().replace(0, np.nan)
    return (tp * df["volume"]).cumsum() / cum_v


@dataclass
class Features:
    as_of: Optional[str] = None
    n_bars: int = 0
    # price/structure
    vwap: float = float("nan"); vwap_dist_atr: float = float("nan")
    er: float = float("nan"); adx: float = float("nan"); chop: float = float("nan")
    atr: float = float("nan"); atr_pctile: float = float("nan")
    rvol_pctile: float = float("nan"); vol_of_vol: float = float("nan")
    day_vol_pctile: float = float("nan")            # today's vol vs trailing DAYS (caller-set)
    range_pos: float = float("nan"); day_range_atr: float = float("nan")
    rel_volume: float = float("nan")
    drift_atr: float = float("nan")
    # options / futures (nan when absent)
    atm_iv: float = float("nan"); iv_rv: float = float("nan"); skew: float = float("nan")
    pcr: float = float("nan"); basis: float = float("nan")
    call_wall: float = float("nan"); put_wall: float = float("nan")
    # time / expiry
    phase: str = ""; mins_to_close: int = 0; dte: Optional[int] = None; post3: bool = False
    # EVENT FLAGS — detection only, NOT regimes. They sharpen the risk-map + confidence
    # (where the stop/band sits, when NOT to initiate). They do NOT predict direction.
    gap_pct: float = float("nan"); gap_type: str = ""       # FLAT/GAP_UP/GAP_DOWN/SHARP_*
    or_high: float = float("nan"); or_low: float = float("nan")
    broke: str = ""             # "" / HIGH / LOW  — price beyond the opening range
    fake_break: str = ""        # "" / FAILED_HIGH / FAILED_LOW — broke then reclaimed
    transition: bool = False    # regime destabilising (vol expanding fast) — don't initiate


def compute(df: pd.DataFrame, as_of: Optional[dt.datetime] = None,
            chain: Optional[pd.DataFrame] = None, basis: Optional[float] = None,
            dte: Optional[int] = None, day_vol_pctile: Optional[float] = None,
            prev_close: Optional[float] = None) -> Features:
    """Causal feature vector from intraday OHLCV (df: ts,open,high,low,close,volume).
    Structure reads CLOSED bars only (drops the forming bar)."""
    d = df.copy()
    d["ts"] = pd.to_datetime(d["ts"])
    if as_of is not None:
        d = d[d["ts"] <= pd.Timestamp(as_of)]
    if len(d) < 20:
        return Features(as_of=as_of.isoformat() if as_of else None, n_bars=len(d))
    closed = d.iloc[:-1]                                   # drop forming bar for structure
    c = closed["close"]
    f = Features(as_of=d["ts"].iloc[-1].isoformat(), n_bars=len(d))

    atr = _atr(closed); f.atr = float(atr.iloc[-1])
    f.er = float(rc.efficiency_ratio(c, 10).iloc[-1])
    f.adx = float(_adx(closed).iloc[-1])
    f.chop = float(_choppiness(closed).iloc[-1])
    vw = _vwap(closed); f.vwap = float(vw.iloc[-1])
    f.vwap_dist_atr = float((c.iloc[-1] - vw.iloc[-1]) / f.atr) if f.atr else float("nan")
    # realised vol + vol-of-vol + ATR percentile (within-session distribution)
    ret = c.pct_change()
    rv = ret.rolling(10).std()
    f.vol_of_vol = float(rv.rolling(10).std().iloc[-1])
    f.atr_pctile = float((atr <= atr.iloc[-1]).mean())
    f.rvol_pctile = float((rv <= rv.iloc[-1]).mean()) if rv.notna().any() else float("nan")
    # range position (where price sits in the day's range) + day range in ATR
    hi, lo = closed["high"].max(), closed["low"].min()
    f.range_pos = float((c.iloc[-1] - lo) / (hi - lo)) if hi > lo else 0.5
    f.day_range_atr = float((hi - lo) / f.atr) if f.atr else float("nan")
    f.drift_atr = float((c.iloc[-1] - c.iloc[-11]) / f.atr) if (f.atr and len(c) > 11) else float("nan")
    # relative volume vs same-count trailing baseline
    v = closed["volume"]
    f.rel_volume = float(v.iloc[-5:].mean() / v.iloc[:-5].mean()) if v.iloc[:-5].mean() > 0 else float("nan")

    # options (optional) — ATM IV, 25Δ skew, PCR, walls
    if chain is not None and not chain.empty and {"strike", "side", "ltp"}.issubset(chain.columns):
        try:
            piv = chain.pivot_table(index="strike", columns="side", values="ltp", aggfunc="last").dropna()
            if {"CE", "PE"}.issubset(piv.columns):
                spot = float((piv.index + piv["CE"] - piv["PE"]).median())
                k = piv.index[np.abs(piv.index.to_numpy() - spot).argmin()]
                if "iv" in chain.columns:
                    ivp = chain.pivot_table(index="strike", columns="side", values="iv", aggfunc="last")
                    f.atm_iv = float(np.nanmean([ivp.loc[k].get("CE", np.nan), ivp.loc[k].get("PE", np.nan)]))
                    if rv.notna().any():
                        f.iv_rv = f.atm_iv / (float(rv.iloc[-1]) * np.sqrt(252 * 75)) if rv.iloc[-1] > 0 else float("nan")
                if "oi" in chain.columns:
                    ow = chain.pivot_table(index="strike", columns="side", values="oi", aggfunc="last")
                    if "CE" in ow: f.call_wall = float(ow["CE"].idxmax())
                    if "PE" in ow: f.put_wall = float(ow["PE"].idxmax())
                    tot = ow.sum()
                    f.pcr = float(tot.get("PE", np.nan) / tot.get("CE", np.nan)) if tot.get("CE", 0) else float("nan")
        except Exception:
            pass
    if basis is not None:
        f.basis = float(basis)
    if day_vol_pctile is not None:
        f.day_vol_pctile = float(day_vol_pctile)
    f.dte = dte

    # time / phase
    t = d["ts"].iloc[-1].time()
    f.mins_to_close = max(0, int((dt.datetime.combine(dt.date.today(), SESSION_CLOSE) -
                                  dt.datetime.combine(dt.date.today(), t)).total_seconds() / 60))
    f.post3 = t >= POST3
    f.phase = ("OPENING" if t < dt.time(9, 45) else "PRE_CLOSE" if t >= POST3
               else "LUNCH" if dt.time(12, 0) <= t < dt.time(13, 30) else "MID")

    # ── EVENT FLAGS (causal detection; risk-map / confidence, not direction) ─────
    today_open = float(closed["open"].iloc[0])
    cur = float(c.iloc[-1])
    # opening gap vs prior close
    if prev_close:
        f.gap_pct = (today_open - prev_close) / prev_close * 100
        a = abs(f.gap_pct)
        if a < 0.15:
            f.gap_type = "FLAT"
        else:
            sharp = a >= 0.75
            f.gap_type = ("SHARP_GAP_UP" if f.gap_pct > 0 else "SHARP_GAP_DOWN") if sharp else \
                         ("GAP_UP" if f.gap_pct > 0 else "GAP_DOWN")
    # opening range (first ~30 min) + break / fake-break
    oq = closed.head(6)
    f.or_high = float(oq["high"].max()); f.or_low = float(oq["low"].min())
    if len(closed) > 6:
        post_or = closed.iloc[6:]
        if cur > f.or_high:
            f.broke = "HIGH"
        elif cur < f.or_low:
            f.broke = "LOW"
        # broke beyond OR earlier but reclaimed = failed/fake break (range held)
        if post_or["high"].max() > f.or_high and cur <= f.or_high:
            f.fake_break = "FAILED_HIGH"
        elif post_or["low"].min() < f.or_low and cur >= f.or_low:
            f.fake_break = "FAILED_LOW"
    # transition pressure = recent realised vol expanding fast vs the session (regime shifting)
    if len(ret.dropna()) >= 10:
        r5 = float(ret.iloc[-5:].std()); rall = float(ret.std())
        f.transition = bool(rall > 0 and r5 > 1.6 * rall)
    return f


# ── Regime classification (8) + confidence ──────────────────────────────────────
def classify(f: Features) -> dict:
    """Rule-based 3-regime label + confidence + honest action. Causal (pure fn of f).
    Production classifier = a clock + one cross-day vol percentile. Nothing more separates
    NIFTY intraday sessions (proven by the forward-behaviour merge test)."""
    if f.n_bars < 20 or f.phase == "OPENING":
        return _out(OPENING, 50, "Wait / mark the opening range — regime not yet formed", f)

    # HIGH_VOL = today's realised vol in the top ~15% vs the trailing 20 days (the ONLY
    # statistically distinct regime — 3× the forward move, direction slightly persists).
    if f.day_vol_pctile == f.day_vol_pctile and f.day_vol_pctile >= 0.85:
        return _out(HIGH_VOL, 65,
                    "WIDE-RANGE / trend day (top-15% vol) — SIZE DOWN; momentum-aware "
                    "(direction slightly persists ~71%); do NOT sell naked vol (fat tail)", f)

    # NORMAL = ~82% of sessions. STRONG/WEAK/RANGE/TRANSITION collapse here — forward direction
    # is a coin-flip (~50-60%), so the honest action is the risk-map, never a directional bet.
    # `structure` (range vs drift) is a within-NORMAL DESCRIPTOR for band width, NOT a regime.
    ranging = (f.chop == f.chop and f.chop >= 61) or (f.er < 0.30 and f.adx < 20)
    if ranging:
        return _out(NORMAL, 62,
                    "NORMAL/range — trade the BAND (~70% reliable), NO directional bet; "
                    "SELL THETA if IV-rich + contained", f, structure="range")
    return _out(NORMAL, 55,
                "NORMAL/drift — two-sided, direction ≈ coin; risk-map / band only, no chase", f,
                structure="drift")


def _out(regime: str, base_conf: int, action: str, f: Features, structure: str = "") -> dict:
    conf = int(max(30, min(85, base_conf)))
    # EVENT FLAGS — detection layer on top of the regime (risk-map + confidence, not direction)
    flags, notes = [], []
    if f.gap_type and f.gap_type != "FLAT":
        flags.append(f"{f.gap_type}{f.gap_pct:+.2f}%")
        notes.append("gap day — wider expected range; don't fade the open, gaps often fade later")
    if f.fake_break:
        flags.append(f.fake_break)
        notes.append("range HELD (failed break) — mean-revert reference valid; failed side = S/R")
    elif f.broke:
        flags.append(f"BROKE_{f.broke}")
        notes.append("opening range broke — move the stop/band reference; do NOT chase (most fail)")
    if f.transition:
        flags.append("SHIFTING")
        notes.append("regime destabilising (vol expanding) — LOWER confidence, don't initiate")
        conf -= 10
    conf = int(max(30, min(85, conf)))
    # post-3pm carry hook: a late STRONG CLOSE (any regime) becomes the validated BTST-long.
    carry = ""
    if f.post3 and f.range_pos == f.range_pos and f.range_pos >= 0.66 and not f.transition:
        carry = ("POST-3PM CARRY → strong close forming; becomes the validated BTST-long "
                 "(close-strength → overnight, exit ~09:30). See btst_signal.py.")
    out = {"regime": regime, "confidence": conf, "action": action, "flags": flags,
           "flag_notes": notes, "carry": carry,
           "features": {k: (round(v, 3) if isinstance(v, float) and v == v else v)
                        for k, v in asdict(f).items()}}
    if structure:
        out["structure"] = structure
    return out


# ── Demo / self-check on real data ──────────────────────────────────────────────
def _demo(sym="NIFTY", date=None):
    fy = {"NIFTY": "NIFTY50", "BANKNIFTY": "NIFTYBANK"}[sym]
    df = pd.read_parquet(f"data/historical/5min/NSE_{fy}_INDEX_5min.parquet")
    df["ts"] = pd.to_datetime(df["ts"]); df["d"] = df["ts"].dt.date
    day = date or df["d"].iloc[-1]
    g = df[df["d"] == day].sort_values("ts")
    print(f"  feature_engine demo — {sym} {day}  ({len(g)} bars)")
    print(f"  {'time':6}{'regime':16}{'conf':5}{'ER':>5}{'ADX':>5}{'Chop':>6}{'rpos':>6}  action")
    for tt in ("10:00", "11:30", "13:00", "14:30", "15:20"):
        hh, mm = map(int, tt.split(":"))
        as_of = dt.datetime.combine(day, dt.time(hh, mm))
        f = compute(g, as_of=as_of)
        r = classify(f)
        print(f"  {tt:6}{r['regime']:16}{r['confidence']:>4} {f.er:>5.2f}{f.adx:>5.0f}"
              f"{f.chop:>6.0f}{f.range_pos:>6.2f}  {r['action'][:52]}")
        if r["carry"]:
            print(f"         ↳ {r['carry'][:80]}")


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _demo()
