"""
oi_moneyness.py — ITM / ATM / OTM bucketing of the live option chain.

WHY THIS MODULE EXISTS, AND WHY IT IS NOT A ±400-POINT RULE
-----------------------------------------------------------
The obvious version — "within ±400 points of spot is ATM, above is OTM, below is
ITM" — has two defects that were measured on live chain data before this was written:

1. ITM/OTM IS SIDE-DEPENDENT. "Above spot = OTM" holds for CALLS only. A PUT above
   spot is IN the money. Applying the call rule to the whole chain labels every put
   bucket backwards, which inverts the put-writing / floor read — the most-used
   signal on the board. Handled here by classifying PER LEG (see _moneyness_by_strike).

2. A FIXED POINT BAND IS A DIFFERENT INSTRUMENT ON EACH INDEX. 400 points is 0.70%
   of BANK NIFTY and 2.71% of MIDCAP. Divided by strike step it also decides how many
   strikes you get. Measured 2026-09-03 on the ±11 window (23 strikes):

       index        400pts     strikes in ±400    ATM zone     wing each side
       NIFTY 50      1.68%          8               16 of 23         3
       BANK NIFTY    0.70%          4                8 of 23         7
       FIN NIFTY     1.54%          8               16 of 23         3
       MIDCAP        2.71%         16               23 of 23         0   <-- no wings

   MIDCAP's ±400 zone needs 33 strikes and only 23 are captured, so every strike is
   ATM and its ITM/OTM panels are structurally empty. Four "same" charts measuring
   four different things.

DEFAULT: DELTA BUCKETS. |delta| >= 0.65 ITM, 0.35-0.65 ATM, < 0.35 OTM. delta and iv
are persisted on every captured leg (verified 100% populated), and delta already
normalises for spot level, strike step, volatility AND time to expiry — so the same
thresholds mean the same thing on all four indices and on any DTE. A fixed point band
silently changes meaning as expiry approaches: ±400 NIFTY points is several sigma at
0-DTE and under 1 sigma twenty days out.

FALLBACK: STRIKE-COUNT buckets, used only when delta is missing/degenerate. Expressed
in STRIKES (not points) so it stays comparable across the four steps (50/100/50/25).

TWO SERIES, NOT ONE — the reclassification trap
------------------------------------------------
Bucket membership is spot-relative, so as spot drifts a strike moves ATM->OTM with NO
OI CHANGE AT ALL. A naive bucket-level dOI series therefore shows "flow" that never
happened: on a 1% rally the window/classification slides up, put-heavy low strikes
leave the ATM bucket and call-heavy high strikes enter, and PCR moves for a purely
mechanical reason. So this module exposes both, and they answer different questions:

  bucket_snapshot()  — LEVELS at one instant, classified live. Honest as a photograph.
                       Correct for "where is the open interest sitting right now".
  basket_delta()     — CHANGE against a basket PINNED at the open (fixed strike set,
                       fixed labels). Correct for "what did positioning actually do",
                       because the constituents cannot change underneath the number.

Pure module: no I/O, no Dash, no broker. Same contract as oi_analytics.py.
"""
from __future__ import annotations

from typing import Iterable, Optional

# ── Delta thresholds. |delta| because PE delta is negative (-1..0) and CE is (0..1).
# 0.65/0.35 are the conventional ITM/ATM/OTM cuts and are deliberately NOT tuned here:
# this module classifies, it does not predict, so there is nothing to fit. Any tuning
# belongs to whatever consumes the buckets, and must be measured there.
ITM_DELTA = 0.65
ATM_DELTA = 0.35

# Strike-count fallback: ATM = ATM_STRIKES either side of the ATM strike, then the
# wings. In STRIKES so BANK (step 100) and MIDCAP (step 25) get comparable structure.
ATM_STRIKES = 3
WINDOW_STRIKES = 11          # ±11 -> 23 strikes, the analysis window

BUCKETS = ("ITM", "ATM", "OTM")


def window_strikes(strikes: Iterable[float], spot: float,
                   n: int = WINDOW_STRIKES) -> list:
    """The 2n+1 strikes nearest spot, sorted ascending.

    An ANALYSIS slice over the wider captured chain — capture stays at ±25 strikes.
    Narrowing the read is free and reversible; narrowing the CAPTURE would throw the
    data away permanently, and the wall audit showed ±15 put the put-floor wall outside
    the map on ~90% of samples. Never narrow capture to match this.

    Returns fewer than 2n+1 when the chain holds fewer (thin index, early snapshot).
    """
    ks = sorted({float(k) for k in strikes if k and k > 0})
    if not ks or not spot:
        return []
    return sorted(sorted(ks, key=lambda k: abs(k - spot))[:2 * n + 1])


def _moneyness_by_strike(strike: float, spot: float, side: str,
                         step: float, atm_strikes: int = ATM_STRIKES) -> str:
    """Fallback classification, PER LEG. The whole point of the side argument:

        CE:  strike < spot  -> ITM      PE:  strike < spot  -> OTM
             strike > spot  -> OTM           strike > spot  -> ITM

    Getting this mirror wrong is what inverts the put/floor read."""
    if not step or step <= 0:
        return "ATM"
    d = (strike - spot) / step                      # signed distance in STRIKES
    if abs(d) <= atm_strikes:
        return "ATM"
    above = d > 0
    if side == "CE":
        return "OTM" if above else "ITM"
    return "ITM" if above else "OTM"


def moneyness(strike: float, spot: float, side: str, delta=None,
              step: float = 0.0, atm_strikes: int = ATM_STRIKES) -> str:
    """ITM / ATM / OTM for one leg. Delta when usable, strike distance otherwise.

    delta is treated as unusable when missing or exactly 0.0 — the broker sends 0.0
    both for "no greeks on this leg" and for a genuinely worthless far-OTM option, and
    those must not be told apart by guessing. A worthless far-OTM leg lands in OTM via
    the strike fallback anyway, so the fallback is the safe read for both.
    """
    try:
        d = abs(float(delta)) if delta is not None else 0.0
    except (TypeError, ValueError):
        d = 0.0
    if d > 0.0:
        if d >= ITM_DELTA:
            return "ITM"
        if d >= ATM_DELTA:
            return "ATM"
        return "OTM"
    return _moneyness_by_strike(strike, spot, side, step, atm_strikes)


def _blank() -> dict:
    return {b: {"oi": 0.0, "oi_chg": 0.0, "volume": 0.0, "n": 0,
                "lo": None, "hi": None} for b in BUCKETS}


def bucket_snapshot(rows: Iterable[dict], spot: float, step: float = 0.0,
                    n: int = WINDOW_STRIKES, use_delta: bool = True) -> dict:
    """LEVELS by bucket at one instant, classified against the CURRENT spot.

    rows: dicts with strike, side ('CE'/'PE'), oi, oich, volume, delta.
    Returns {'CE': {bucket: {...}}, 'PE': {...}, 'window': [strikes], 'spot': spot,
             'basis': 'delta'|'strike', 'atm': strike}.

    This is a photograph, not a flow measurement — see the module docstring. For change,
    use basket_delta().
    """
    rows = [r for r in rows if r and r.get("strike")]
    keep = set(window_strikes((r["strike"] for r in rows), spot, n))
    out = {"CE": _blank(), "PE": _blank(), "window": sorted(keep), "spot": spot,
           "basis": "delta" if use_delta else "strike",
           "atm": min(keep, key=lambda k: abs(k - spot)) if keep else None}
    for r in rows:
        k = float(r["strike"])
        if k not in keep:
            continue
        side = r.get("side")
        if side not in ("CE", "PE"):
            continue
        b = moneyness(k, spot, side, r.get("delta") if use_delta else None, step)
        cell = out[side][b]
        cell["oi"] += float(r.get("oi") or 0.0)
        cell["oi_chg"] += float(r.get("oich") or 0.0)
        cell["volume"] += float(r.get("volume") or 0.0)
        cell["n"] += 1
        cell["lo"] = k if cell["lo"] is None else min(cell["lo"], k)
        cell["hi"] = k if cell["hi"] is None else max(cell["hi"], k)
    return out


def pin_basket(rows: Iterable[dict], spot: float, step: float = 0.0,
               n: int = WINDOW_STRIKES, use_delta: bool = True) -> dict:
    """Freeze the strike set AND its bucket labels from the opening snapshot.

    Anchor on the FIRST IN-SESSION snapshot, never the 09:00-09:15 pre-open auction —
    that publishes indicative prices that never traded, and core.session already drops
    that window for the same reason. Feeding an auction print in here pins the whole
    day's basket to a price nobody transacted at.
    """
    snap = bucket_snapshot(rows, spot, step, n, use_delta)
    labels = {}
    for r in rows:
        k = float(r.get("strike") or 0)
        side = r.get("side")
        if k in set(snap["window"]) and side in ("CE", "PE"):
            labels[(k, side)] = moneyness(k, spot, side,
                                          r.get("delta") if use_delta else None, step)
    return {"labels": labels, "spot0": spot, "atm0": snap["atm"],
            "window": snap["window"],
            "oi0": {(float(r["strike"]), r["side"]): float(r.get("oi") or 0.0)
                    for r in rows if float(r.get("strike") or 0) in set(snap["window"])
                    and r.get("side") in ("CE", "PE")}}


def basket_delta(rows: Iterable[dict], basket: dict) -> dict:
    """OI CHANGE since the open, over the PINNED basket — the honest flow number.

    Constituents and labels are fixed, so every rupee of change here is real OI change
    at a fixed strike. Nothing can enter or leave a bucket because spot moved.

    Legs that vanish from the chain (strike churn, expiry roll) are simply absent; they
    are NOT treated as OI going to zero, which would read as a giant fake unwind.
    """
    out = {"CE": _blank(), "PE": _blank(), "missing": 0,
           "spot0": basket.get("spot0"), "atm0": basket.get("atm0")}
    seen = set()
    for r in rows:
        try:
            k = float(r.get("strike") or 0)
        except (TypeError, ValueError):
            continue
        side = r.get("side")
        key = (k, side)
        b = basket["labels"].get(key)
        if b is None:
            continue
        seen.add(key)
        cell = out[side][b]
        cell["oi"] += float(r.get("oi") or 0.0)
        cell["oi_chg"] += float(r.get("oi") or 0.0) - basket["oi0"].get(key, 0.0)
        cell["volume"] += float(r.get("volume") or 0.0)
        cell["n"] += 1
        cell["lo"] = k if cell["lo"] is None else min(cell["lo"], k)
        cell["hi"] = k if cell["hi"] is None else max(cell["hi"], k)
    out["missing"] = len(set(basket["labels"]) - seen)
    return out


def bucket_pcr(snap: dict, bucket: str = "ATM") -> Optional[float]:
    """PUT/CALL OI ratio WITHIN one moneyness bucket.

    A whole-chain PCR mixes an OTM-put floor with an OTM-call ceiling and calls the sum
    sentiment. Per-bucket separates them: OTM-put PCR is the floor, OTM-call is the
    ceiling. Returns None rather than 0.0 when the call side is empty — an empty bucket
    is missing data, not a PCR of zero (MIDCAP's ITM buckets are routinely empty).
    """
    c = (snap.get("CE") or {}).get(bucket, {}).get("oi") or 0.0
    p = (snap.get("PE") or {}).get(bucket, {}).get("oi") or 0.0
    return (p / c) if c > 0 else None


def escaped(snap: dict, spot: float) -> bool:
    """True when spot has left the analysis window — every remaining strike is on one
    side and the walls/max-pain inside it stop meaning anything.

    Measured over 57 sessions x 4 indices: the day's travel exceeds ±11 strikes from
    the open ATM on 2.6% of index-days (NIFTY/MIDCAP 1.8%, BANK/FIN 3.5%), so this is
    rare but real, and BANK is worst because ±11 is only ±1.92% there — the NARROWEST
    band of the four on the MOST volatile index.
    """
    w = snap.get("window") or []
    return bool(w) and not (min(w) <= spot <= max(w))
