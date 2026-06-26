"""
opening_playbook.py — First-20-minutes F&O opening playbook.

The first 15–20 minutes of the session set the day's structure: the opening
range, the gap reaction, where dealers start writing, how ATM premium behaves,
and what the futures basis says about institutional intent. This module reads
all of it — plus last night's EOD context — and issues ONE concrete morning
call per index, the way a desk would:

    BUY CE <strike>      directional conviction, sane IV → long the move
    BUY PE <strike>
    WRITE PE <strike>    directional but IV pumped → sell the opposite wall
    WRITE CE <strike>    (premium seller's edge instead of paying it)
    BUY/SELL FUT         strong trend but option premium not worth paying
    NO TRADE             conviction below threshold — standing aside is a call

Factor stack (each bucketed to [-1, +1], weighted):
    OR    0.20  opening-range break / position within the range
    GAP   0.10  gap vs prev close and whether the gap holds
    OI    0.25  first-20-min net OI flow (put writing vs call writing)
    PREM  0.15  ATM call vs put premium drift (who is paying up)
    FUT   0.10  basis sign + drift, term structure, futures volume (rollover)
    EOD   0.20  Layer-9 Daily-Context composite from last night's data

Everything reads the lock-free Parquet mirrors in data/intraday/live/
(refreshed ~10s by the dashboard) and the EOD bridge — no live-store coupling,
so it supports `as_of` reconstruction and offline replay of any captured day.

The first call issued at/after PLAYBOOK_READY (09:35) becomes the session
baseline; later calls that change direction are flagged FLIPPED so the
"everything changed within 30 minutes" day is surfaced, not hidden.

All functions degrade to {"has_data": False} while history is thin.
"""

from __future__ import annotations

import datetime
import threading
import time as _time
from pathlib import Path
from typing import Optional

try:
    import pandas as pd
    _PANDAS = True
except Exception:                       # pragma: no cover
    _PANDAS = False

from core.constants import IST   # single source of truth  # noqa: E402
_LIVE_DIR = Path(__file__).parent / "data" / "intraday" / "live"

INDEX_SYMBOLS = [
    "NSE:NIFTY50-INDEX", "NSE:NIFTYBANK-INDEX",
    "NSE:FINNIFTY-INDEX", "NSE:MIDCPNIFTY-INDEX",
]
STRIKE_STEP = {
    "NSE:NIFTY50-INDEX": 50, "NSE:NIFTYBANK-INDEX": 100,
    "NSE:FINNIFTY-INDEX": 50, "NSE:MIDCPNIFTY-INDEX": 25,
}

MKT_OPEN       = datetime.time(9, 15)
OR_END         = datetime.time(9, 35)   # opening-range window close
PLAYBOOK_READY = datetime.time(9, 35)   # earliest moment a call is issued

# Factor weights (sum 1.0)
_W = {"or": 0.20, "gap": 0.10, "oi": 0.25, "prem": 0.15, "fut": 0.10, "eod": 0.20}

# Bucketing scales (per full ±1 unit)
_OI_SCALE   = 400_000     # net put-minus-call OI build over the window
_PREM_SCALE = 10.0        # ATM premium drift (pts)
_GAP_SCALE  = 0.45        # gap as % of prev close for a full unit
_BASIS_FULL = 0.25        # futures basis (% of spot) for a full unit

_MIN_CONVICTION = 0.25    # below this → NO TRADE
# Per-index ATM-IV bands (vol points): below low → cheap, above high → rich.
# Older sessions stored IV on a broken fractional scale; anything < 2.0 is
# treated as unknown and routes to the normal BUY branch.
_IV_BANDS = {
    "NSE:NIFTY50-INDEX":    (10.0, 16.0),
    "NSE:NIFTYBANK-INDEX":  (12.0, 19.0),
    "NSE:FINNIFTY-INDEX":   (12.0, 19.0),
    "NSE:MIDCPNIFTY-INDEX": (14.0, 21.0),
}
_IV_PUMP = 0.75           # IV rise since open that flips us to premium selling


# ── Parquet readers (lock-free, mirror regime_forecast.py) ────────────────────
def _today() -> str:
    return datetime.datetime.now(tz=IST).date().isoformat()


_read_cache: dict = {}        # (tbl, date) → (monotonic_ts, df)
_READ_TTL = 10.0              # mirrors refresh ~10s; playbook_all reads each
_read_lock = threading.Lock()  # table once per cycle instead of once per index


def _read(tbl: str, date: Optional[str] = None) -> "pd.DataFrame | None":
    if not _PANDAS:
        return None
    key = (tbl, date or _today())
    now = _time.monotonic()
    with _read_lock:
        hit = _read_cache.get(key)
        if hit and now - hit[0] < _READ_TTL:
            return hit[1]
    p = _LIVE_DIR / f"{key[1]}_{tbl}.parquet"
    if not p.exists():
        return None
    try:
        df = pd.read_parquet(p)
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
    except Exception:
        return None
    with _read_lock:
        _read_cache[key] = (now, df)
        if len(_read_cache) > 40:      # old replay dates — don't grow unbounded
            _read_cache.pop(next(iter(_read_cache)))
    return df


def _clip(x: float, lo=-1.0, hi=1.0) -> float:
    return max(lo, min(hi, x))


def _sym_window(df, sym: str, as_of: Optional[datetime.datetime]):
    if df is None or df.empty:
        return None
    out = df[df["symbol"] == sym].sort_values("ts")
    if as_of is not None:
        out = out[out["ts"] <= as_of.astimezone(datetime.timezone.utc)]
    return out if len(out) else None


# ── Factor reads ───────────────────────────────────────────────────────────────
_or_rest_cache: dict = {}     # (date, sym) → (or_high, or_low) from Fyers REST


def _rest_or(sym: str) -> "tuple[float, float] | None":
    """
    True 09:15–09:35 opening range via the Fyers history API — recovers the OR
    even when local capture started late. Cached for the day after first success
    (the OR is immutable once 09:35 has passed).
    """
    key = (_today(), sym)
    if key in _or_rest_cache:
        return _or_rest_cache[key]
    try:
        from signals import fetch_ohlcv
        df = fetch_ohlcv(sym, "1", 1)
        if df is None or df.empty:
            return None
        today = datetime.datetime.now(tz=IST).date()
        m = df[(df["ts"].dt.date == today) & (df["ts"].dt.time >= MKT_OPEN)
               & (df["ts"].dt.time < OR_END)]
        if len(m) < 15:               # want nearly the full 20-min window
            return None
        res = (float(m["high"].max()), float(m["low"].min()))
        _or_rest_cache[key] = res
        return res
    except Exception:
        return None


def _opening_range(sym: str, as_of, date: Optional[str]) -> dict:
    """OR high/low from 1-min candles (09:15–09:35) + price position vs the OR."""
    df = _sym_window(_read("candles", date), sym, as_of)
    if df is None or "resolution" not in df.columns:
        return {}
    df = df[df["resolution"] == "1min"]
    if df.empty:
        return {}
    local = df["ts"].dt.tz_convert("Asia/Kolkata")
    orr = df[local.dt.time < OR_END]
    approx = ""
    or_high = or_low = None
    if len(orr) < 10:
        # Capture started late (bot launched after open). The exchange still has
        # the true OR — recover it over REST; only if that fails, approximate from
        # the first 20 minutes of local capture and say so.
        rest = _rest_or(sym) if date is None else None
        if rest:
            or_high, or_low = rest
            approx = ""
        else:
            orr = df.head(20)
            if len(orr) < 10:
                return {}
            approx = " (approx — late capture)"
    if or_high is None:
        or_high, or_low = float(orr["high"].max()), float(orr["low"].min())
    last = float(df.iloc[-1]["close"])
    rng = max(or_high - or_low, 1e-9)
    if last > or_high:
        score = _clip(0.5 + (last - or_high) / rng, 0, 1)
        msg = f"OR breakout ↑ ({last:,.0f} > {or_high:,.0f}){approx}"
    elif last < or_low:
        score = -_clip(0.5 + (or_low - last) / rng, 0, 1)
        msg = f"OR breakdown ↓ ({last:,.0f} < {or_low:,.0f}){approx}"
    else:
        score = _clip(((last - or_low) / rng - 0.5) * 0.8)   # inside: mild lean only
        msg = f"inside OR ({or_low:,.0f}–{or_high:,.0f}){approx}"
    return {"or_high": or_high, "or_low": or_low, "last": last,
            "score": round(score, 3), "msg": msg}


def _tick_state(sym: str, as_of, date: Optional[str]) -> dict:
    """Latest tick: NSE broadcasts prev-close-relative change, so prev_close is
    exact (ltp - ch) — no dependency on the EOD sync being fresh."""
    df = _sym_window(_read("ticks", date), sym, as_of)
    if df is None:
        return {}
    t = df.iloc[-1]
    ltp, ch = float(t["ltp"] or 0), float(t["ch"] or 0)
    return {"ltp": ltp, "ch": ch, "chp": float(t["chp"] or 0),
            "day_open": float(t["day_open"] or 0),
            "prev_close": ltp - ch if ltp and ch is not None else None}


def _gap_read(last: float, tick: dict, prev_close_fallback) -> dict:
    prev_close = tick.get("prev_close") or prev_close_fallback
    if not prev_close or not last:
        return {}
    day_open = tick.get("day_open") or last
    gap_pts = day_open - prev_close
    gap_pct = gap_pts / prev_close * 100
    held = tick.get("chp") or ((last - prev_close) / prev_close * 100)
    raw = _clip(gap_pct / _GAP_SCALE)
    # A gap that is being given back is a fade signal, not a continuation one.
    if raw * held < 0 or abs(held) < abs(gap_pct) * 0.4:
        raw *= -0.5
        msg = f"gap {gap_pts:+,.0f} pts FADING (now {held:+.2f}% vs prev close)"
    else:
        msg = f"gap {gap_pts:+,.0f} pts holding ({held:+.2f}% vs prev close)"
    return {"gap_pts": round(gap_pts, 1), "score": round(_clip(raw * (1 if abs(gap_pct) > 0.08 else 0)), 3),
            "msg": msg}


def _oi_premium_read(sym: str, as_of, date: Optional[str]) -> dict:
    """First-window OI flow + ATM premium drift + IV regime from oi_snapshots."""
    df = _sym_window(_read("oi_snapshots", date), sym, as_of)
    if df is None or len(df) < 3:
        return {}
    first, now = df.iloc[0], df.iloc[-1]
    d_put  = (now["total_put_oi"]  or 0) - (first["total_put_oi"]  or 0)
    d_call = (now["total_call_oi"] or 0) - (first["total_call_oi"] or 0)
    oi_score = _clip((d_put - d_call) / _OI_SCALE)
    oi_msg = (f"{'put-writing' if oi_score > 0 else 'call-writing'} since open "
              f"(ΔP {d_put/1e5:+.1f}L / ΔC {d_call/1e5:+.1f}L)")

    d_cp = (now["atm_call_prem"] or 0) - (first["atm_call_prem"] or 0)
    d_pp = (now["atm_put_prem"]  or 0) - (first["atm_put_prem"]  or 0)
    prem_score = _clip((d_cp - d_pp) / _PREM_SCALE)
    prem_msg = f"ATM prem drift C {d_cp:+.1f} / P {d_pp:+.1f}"

    iv_now   = float(now["atm_iv"] or 0)
    iv_open  = float(first["atm_iv"] or 0)
    iv_chg   = iv_now - iv_open if iv_now and iv_open else 0.0
    both_crushed = d_cp < 0 and d_pp < 0

    return {
        "oi_score": round(oi_score, 3), "oi_msg": oi_msg,
        "prem_score": round(prem_score, 3), "prem_msg": prem_msg,
        "iv_now": round(iv_now, 2), "iv_chg": round(iv_chg, 2),
        "both_crushed": both_crushed,
        "pcr": float(now["pcr"] or 0),
        "call_wall": int(now["call_wall"] or 0), "put_wall": int(now["put_wall"] or 0),
        "max_pain": int(now["max_pain"] or 0), "atm": int(now["atm"] or 0),
        "spot": float(now["spot"] or 0),
    }


def _chain_read(sym: str, as_of, date: Optional[str]) -> dict:
    """
    Strike-level, gap-aware OI read from chain_snapshots (4-quadrant matrix).

    oich/ltpch are vs PREVIOUS CLOSE, so on a gap-up day this correctly splits
    "huge call OI" into trapped writers covering at run-over strikes (bullish
    squeeze fuel) vs fresh writing above spot (real ceiling) — the distinction
    total-OI deltas cannot see. Volume share scales each leg's weight.
    """
    df = _sym_window(_read("chain_snapshots", date), sym, as_of)
    if df is None or df.empty:
        return {}
    cur = df[df["ts"] == df["ts"].max()]
    mx = float(cur["oich"].abs().max() or 0)
    if mx <= 0:
        return {}
    tot_vol = float(cur["volume"].sum() or 0)
    net = tot = 0.0
    notes: list = []
    for _, r in cur.iterrows():
        oich, ltpch = float(r["oich"] or 0), float(r["ltpch"] or 0)
        if abs(oich) < max(0.12 * mx, 1):     # noise floor, adaptive per chain
            continue
        if r["side"] == "CE":
            if   oich > 0 and ltpch <= 0: sgn, lab = -1.0, "fresh writing (ceiling)"
            elif oich > 0:                sgn, lab = +0.6, "long buildup"
            elif ltpch >= 0:              sgn, lab = +1.0, "short-covering (squeeze fuel)"
            else:                         sgn, lab = -0.3, "long unwinding"
        else:
            if   oich > 0 and ltpch <= 0: sgn, lab = +1.0, "fresh writing (floor)"
            elif oich > 0:                sgn, lab = -0.6, "long buildup"
            elif ltpch >= 0:              sgn, lab = -1.0, "short-covering (floor crumbling)"
            else:                         sgn, lab = +0.3, "long unwinding"
        w = abs(oich) * (1 + (float(r["volume"] or 0) / tot_vol if tot_vol else 0))
        net += w * sgn
        tot += w
        notes.append((w, f"{int(r['strike'])}{r['side']} {lab} ({oich/1e5:+.1f}L)"))
    if tot <= 0:
        return {}
    notes.sort(key=lambda x: -x[0])
    return {"score": round(_clip(net / tot), 3), "msgs": [m for _, m in notes[:3]]}


def _futures_read(sym: str, as_of, date: Optional[str]) -> dict:
    """Basis sign + drift, term structure, and futures volume — the rollover read."""
    df = _sym_window(_read("futures_quotes", date), sym, as_of)
    if df is None or len(df) < 2:
        return {}
    first, now = df.iloc[0], df.iloc[-1]
    basis = float(now["near_basis"] or 0)
    basis_drift = basis - float(first["near_basis"] or 0)
    term = str(now["term_structure"] or "")
    spot = float(now["near_ltp"] or 0)
    basis_pct = basis / spot * 100 if spot else 0.0
    score = _clip(basis_pct / _BASIS_FULL) * 0.7 + _clip(basis_drift / 10.0) * 0.3
    if "BACKWARD" in term.upper():
        score -= 0.2
    d_vol = (now["near_vol"] or 0) - (first["near_vol"] or 0)
    return {"score": round(_clip(score), 3), "basis": round(basis, 1),
            "term": term, "near_ltp": spot, "d_vol": int(d_vol),
            "msg": f"basis {basis:+.0f} ({'widening' if basis_drift > 0 else 'narrowing'}) · {term or '—'}"}


def _eod_read(sym: str) -> dict:
    """Last night's Daily-Cash-Market composite + key levels (reference memory)."""
    try:
        from daily_context_bridge import get_bridge
        b = get_bridge()
        score, signals = b.score_index(sym)          # [-4, +4]
        levels = b.get_key_levels(sym) or {}
        top = "; ".join(str(s) for s in (signals or [])[:2])
        return {"score": round(_clip(score / 4.0), 3), "levels": levels,
                "msg": f"EOD context {score:+.1f}/4 — {top[:90]}"}
    except Exception:
        return {}


# ── Instrument selection ──────────────────────────────────────────────────────
def _round_strike(px: float, step: int) -> int:
    return int(round(px / step) * step)


def _pick_instrument(sym: str, direction: str, conviction: float,
                     oi: dict, fut: dict) -> dict:
    """
    Translate (direction, conviction, IV regime, walls) into the actual trade:
      rich IV  → write the opposite wall (premium seller's edge)
      cheap IV + strong trend → futures (don't pay theta for nothing)
      normal   → buy the ATM option
    """
    step = STRIKE_STEP.get(sym, 50)
    spot = oi.get("spot") or fut.get("near_ltp") or 0
    atm = oi.get("atm") or _round_strike(spot, step)
    iv_now, iv_chg = oi.get("iv_now", 0), oi.get("iv_chg", 0)
    bull = direction == "BULLISH"

    iv_lo, iv_hi = _IV_BANDS.get(sym, (10.0, 16.0))
    iv_known = iv_now >= 2.0
    iv_rich  = iv_known and (iv_now >= iv_hi or iv_chg >= _IV_PUMP)
    iv_cheap = (iv_known and iv_now <= iv_lo) or oi.get("both_crushed", False)

    if iv_rich:
        # Sell the wall behind the move: bullish → short put at put wall.
        wall = oi.get("put_wall") if bull else oi.get("call_wall")
        strike = wall or _round_strike(spot - 2 * step if bull else spot + 2 * step, step)
        # Never write an ITM/at-spot strike (walls can sit at spot on fast moves):
        # clamp to at least 2 steps OTM on the safe side.
        if bull:
            strike = min(strike, _round_strike(spot, step) - 2 * step)
        else:
            strike = max(strike, _round_strike(spot, step) + 2 * step)
        return {"action": f"WRITE {'PE' if bull else 'CE'}", "strike": int(strike),
                "why": (f"IV rich ({iv_now:.1f}{f', +{iv_chg:.1f} since open' if iv_chg > 0 else ''})"
                        f" — sell premium at the {'put' if bull else 'call'} wall instead of paying it"),
                "margin_note": "short option — margin trade, size by max loss not premium"}
    if iv_cheap and conviction >= 0.45:
        return {"action": f"{'BUY' if bull else 'SELL'} FUT", "strike": None,
                "why": (f"strong trend but option premium {'crushed' if oi.get('both_crushed') else 'cheap'}"
                        f" (IV {iv_now:.1f}) — futures carry the move without theta"),
                "margin_note": "futures — margin trade, SL is mandatory"}
    return {"action": f"BUY {'CE' if bull else 'PE'}", "strike": int(atm),
            "why": f"directional with sane IV ({iv_now:.1f}) — ATM {'call' if bull else 'put'} carries best delta",
            "margin_note": ""}


# ── Session baseline / flip memory ─────────────────────────────────────────────
_baseline: dict = {}            # (date, sym) → {"direction","action","ts"}
_baseline_lock = threading.Lock()


def _flip_check(sym: str, direction: str, action: str, asof_dt) -> dict:
    key = (asof_dt.date().isoformat(), sym)
    with _baseline_lock:
        base = _baseline.get(key)
        if base is None and direction != "NEUTRAL":
            _baseline[key] = {"direction": direction, "action": action,
                              "ts": asof_dt.strftime("%H:%M")}
            return {"flipped": False}
        if base and direction != "NEUTRAL" and direction != base["direction"]:
            return {"flipped": True, "from": f"{base['action']} @{base['ts']}",
                    "msg": f"⚠ FLIPPED — opened as {base['action']} ({base['ts']}), now {action}"}
    return {"flipped": False}


# ── Public API ─────────────────────────────────────────────────────────────────
def playbook_index(sym: str, as_of: Optional[datetime.datetime] = None,
                   date: Optional[str] = None) -> dict:
    """
    Morning playbook for one index. `as_of` reconstructs the call at a past
    moment; `date` (ISO) replays a previous captured session offline.
    """
    now = as_of or datetime.datetime.now(tz=IST)
    out = {"sym": sym, "has_data": False, "asof": now.isoformat()}
    if date is None and now.time() < PLAYBOOK_READY:
        out["note"] = f"warming up — call issues at {PLAYBOOK_READY:%H:%M} (needs the full opening range)"
        return out

    orr   = _opening_range(sym, as_of, date)
    oi    = _oi_premium_read(sym, as_of, date)
    chain = _chain_read(sym, as_of, date)
    fut   = _futures_read(sym, as_of, date)
    # LEAKAGE GUARD: _eod_read hits the LIVE Daily-Cash-Market bridge (current EOD).
    # That is legitimate for a live call (yesterday's close, known at open) but on a
    # PAST-day replay (date set) the bridge returns a FUTURE day's EOD -> lookahead.
    # Skip it in replay; the composite's "eod" part is then 0 and gap falls back to
    # the exact tick-derived prev_close (ltp-ch), which is already leak-free.
    eod   = _eod_read(sym) if date is None else {}
    if not orr or not oi:
        out["note"] = "insufficient capture (need ~20 min of candles + OI snapshots)"
        return out

    # Strike-level read supersedes the naive total-OI delta when available —
    # totals can't tell fresh writing from trapped writers unwinding after a gap.
    if chain:
        oi["oi_score"] = chain["score"]
        oi["oi_msg"]   = " · ".join(chain["msgs"][:2])

    tick = _tick_state(sym, as_of, date)
    gap = _gap_read(orr.get("last", 0), tick,
                    (eod.get("levels") or {}).get("prev_close"))

    parts = {
        "or":   orr.get("score", 0),
        "gap":  gap.get("score", 0),
        "oi":   oi.get("oi_score", 0),
        "prem": oi.get("prem_score", 0),
        "fut":  fut.get("score", 0),
        "eod":  eod.get("score", 0),
    }
    composite = _clip(sum(_W[k] * v for k, v in parts.items()))
    live = [v for v in parts.values() if abs(v) > 0.2]
    agree = sum(1 for v in live if v * composite > 0)
    conviction = abs(composite) * (1 + 0.12 * max(0, agree - 2))
    conviction = min(conviction, 1.0)

    direction = ("BULLISH" if composite > _MIN_CONVICTION else
                 "BEARISH" if composite < -_MIN_CONVICTION else "NEUTRAL")

    factors = []
    for k, label, msg in (("or", "OR", orr.get("msg")), ("gap", "GAP", gap.get("msg")),
                          ("oi", "OI", oi.get("oi_msg")), ("prem", "PREM", oi.get("prem_msg")),
                          ("fut", "FUT", fut.get("msg")), ("eod", "EOD", eod.get("msg"))):
        if msg:
            factors.append(("bull" if parts[k] > 0.05 else "bear" if parts[k] < -0.05 else "flat",
                            f"{label}: {msg}"))

    if direction == "NEUTRAL":
        rec = {"action": "NO TRADE", "strike": None,
               "why": "factors mixed — standing aside until the tape picks a side",
               "margin_note": ""}
    else:
        rec = _pick_instrument(sym, direction, conviction, oi, fut)

    # Invalidation: the OR boundary against the call (a desk always knows where it's wrong).
    inval = orr.get("or_low") if direction == "BULLISH" else \
            orr.get("or_high") if direction == "BEARISH" else None

    # Flip memory only updates on live "now" calls — a time-machine `as_of`
    # reconstruction must never overwrite the session baseline.
    flip = _flip_check(sym, direction, rec["action"], now) \
        if (date is None and as_of is None) else {"flipped": False}

    out.update({
        "has_data": True, "direction": direction, "composite": round(composite, 3),
        "conviction": int(conviction * 100), "action": rec["action"],
        "strike": rec["strike"], "why": rec["why"], "margin_note": rec["margin_note"],
        "factors": factors, "parts": parts,
        "or_high": orr.get("or_high"), "or_low": orr.get("or_low"),
        "spot": orr.get("last"), "invalidation": inval,
        "iv_now": oi.get("iv_now"), "iv_chg": oi.get("iv_chg"), "pcr": oi.get("pcr"),
        "call_wall": oi.get("call_wall"), "put_wall": oi.get("put_wall"),
        "max_pain": oi.get("max_pain"), "gap_pts": gap.get("gap_pts"),
        "flip": flip,
    })
    return out


def playbook_all(as_of: Optional[datetime.datetime] = None,
                 date: Optional[str] = None) -> dict:
    """Playbook for all indices + a one-line market coherence read."""
    per = {s: playbook_index(s, as_of, date) for s in INDEX_SYMBOLS}
    live = [p for p in per.values() if p.get("has_data")]
    dirs = [p["direction"] for p in live if p["direction"] != "NEUTRAL"]
    coh_dir, coh_n = "", 0
    for d in ("BULLISH", "BEARISH"):
        if dirs.count(d) > coh_n:
            coh_dir, coh_n = d, dirs.count(d)
    return {"per_index": per, "has_data": bool(live),
            "coherence": f"{coh_n}/4 {coh_dir.lower()}" if coh_n else "mixed",
            "asof": (as_of or datetime.datetime.now(tz=IST)).isoformat()}


if __name__ == "__main__":
    # Offline replay:  python opening_playbook.py [YYYY-MM-DD] [HH:MM]
    import sys
    d = sys.argv[1] if len(sys.argv) > 1 else None
    as_of = None
    if len(sys.argv) > 2 and d:
        hh, mm = sys.argv[2].split(":")
        as_of = datetime.datetime.fromisoformat(d).replace(
            hour=int(hh), minute=int(mm), tzinfo=IST)
    res = playbook_all(as_of=as_of, date=d)
    print(f"asof={res['asof']}  coherence={res['coherence']}")
    for s, p in res["per_index"].items():
        if not p.get("has_data"):
            print(f"  {s:26s} — {p.get('note', 'no data')}")
            continue
        strike = f" {p['strike']}" if p["strike"] else ""
        print(f"  {s:26s} {p['direction']:8s} {p['conviction']:3d}%  ->  {p['action']}{strike}")
        print(f"  {'':26s} why: {p['why']}")
        for tone, m in p["factors"]:
            print(f"  {'':28s} [{tone:4s}] {m}")
        if p.get("invalidation"):
            print(f"  {'':26s} wrong below/above: {p['invalidation']:,.0f}")
