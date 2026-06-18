"""
session_conductor.py — the continuous intraday "desk", Stage 1 (fusion core).

The Opening Playbook is a SESSION-OPENING thesis: ~50% of its weight is anchored to
the gap/OR/EOD structure (opening_playbook._W), so it is right at 9:40 and stale by
11:00. The Regime Radar leads (OI velocity) but is only a regime label. Neither is a
single, evolving, actionable position directive — that is the gap this fills.

The conductor fuses, every cycle, four clean mirror-readers into ONE stance per index:

  opening thesis   opening_playbook.playbook_index   — a DECAYING prior (w→0 by ~11:35)
  regime (leading) regime_forecast.forecast_index    — bias + short-window leading edge
  momentum         timeframe_delta.analyze_index      — σ-normalized multi-horizon move
  positioning      timeframe_delta OI fusion          — the index's only "flow" signal

  fused(t) = w_open(t)·thesis + (1-w_open)·(0.42·regime + 0.38·momentum + 0.20·OI)

Then an instrument decision tree turns DIRECTION into the actual trade, from live IV /
theta / option-flow clarity (the user's exact ask):

  rich IV                         → WRITE the opposite wall   (sell expensive premium)
  IV crush / unclear OI / late    → BUY/SELL FUTURES          (carry move without theta)
  cheap IV                        → BUY option                (cheap convexity)
  normal                          → BUY ATM option            (best delta)
  no edge                         → NO TRADE / CLOSE

And a transition layer (vs the current position) with anti-whipsaw HYSTERESIS: a flip
needs a HIGHER bar than an entry — invalidation breached, or high conviction, or a
confirmed regime build — because the cost study proved hair-trigger flipping bleeds.

Stage 1 is decision-support only: it computes and explains the evolving call. It does
NOT auto-execute. Wiring to the paper ledger is Stage 3.

  .venv\\Scripts\\python.exe session_conductor.py
  .venv\\Scripts\\python.exe session_conductor.py --symbol NIFTY
"""
from __future__ import annotations

import argparse
import datetime
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

import opening_playbook as OP
import regime_forecast as RF
import timeframe_delta as TD

from core.constants import IST, INDEX_SYMBOLS, LABELS, STRIKE_STEP, LABEL_TO_SYM  # noqa: E402

# Live fusion weights (renormalised against the decaying opening prior).
_W_LIVE = {"regime": 0.42, "momentum": 0.38, "oi": 0.20}
_DIR_DEADBAND = 0.15      # |fused| below this → FLAT
_ENTRY_CONF   = 45        # min conviction to act on a new stance
_FLIP_CONF    = 65        # conviction that authorises an immediate flip
_THETA_CUTOVER = datetime.time(14, 30)   # late session → theta-averse


# ── Decaying opening prior ──────────────────────────────────────────────────────
def _w_open(now: datetime.datetime) -> float:
    """Opening-thesis weight: 0.45 at 09:35, linear to 0 by 11:35."""
    mins = (now - now.replace(hour=9, minute=35, second=0, microsecond=0)).total_seconds() / 60
    return float(np.clip(0.45 * (1 - mins / 120.0), 0.0, 0.45))


# ── Component votes ─────────────────────────────────────────────────────────────
def _momentum_vote(td: dict) -> tuple[float, str]:
    """Vol-normalised momentum from σ-moves: weight 5/15/60-min, map ~2σ→full unit."""
    z = {r["T"]: r.get("z", 0.0) for r in td.get("rows", [])}
    blend = 0.4 * z.get(5, 0) + 0.4 * z.get(15, 0) + 0.2 * z.get(60, 0)
    detail = "  ".join(f"{T}m {z.get(T,0):+.1f}σ" for T in (5, 15, 60) if T in z)
    return float(np.clip(blend / 2.0, -1, 1)), detail


def _regime_vote(rf: dict) -> tuple[float, str]:
    """Established bias + leading short-window edge."""
    if not rf.get("has_data"):
        return 0.0, "no data"
    v = float(np.clip(0.6 * rf.get("bias", 0) + 0.4 * rf.get("short", 0), -1, 1))
    tail = f" → {rf['next_dir']} {rf['eta']}" if rf.get("stage") in ("BUILDING", "IMMINENT") and rf.get("next_dir") else ""
    return v, f"{rf.get('regime','?')} {rf.get('stage','')}{tail}"


# ── Instrument decision tree ────────────────────────────────────────────────────
def _iv_regime(sym: str, iv_now: float, iv_chg: float) -> str:
    if not iv_now or iv_now < 2.0:
        return "unknown"
    lo, hi = OP._IV_BANDS.get(sym, (10.0, 16.0))
    if iv_now >= hi or iv_chg >= OP._IV_PUMP:
        return "rich"
    if iv_now <= lo:
        return "cheap"
    return "normal"


def _round_strike(px: float, step: int) -> int:
    return int(round(px / step) * step)


def select_instrument(sym, direction, conviction, iv_reg, iv_crush,
                      theta_high, oi_unclear, call_wall, put_wall, spot) -> dict:
    """DIRECTION → concrete trade, from live IV / theta / option-flow clarity."""
    if direction == "FLAT":
        return {"action": "NO TRADE", "strike": None,
                "why": "no directional edge — stand aside / hold flat"}
    step = STRIKE_STEP.get(sym, 50)
    atm  = _round_strike(spot, step)
    bull = direction == "LONG"

    # 1. Rich IV → sell premium at the wall behind the move.
    if iv_reg == "rich":
        wall = put_wall if bull else call_wall
        strike = wall or _round_strike(spot - 2 * step if bull else spot + 2 * step, step)
        strike = (min(strike, atm - 2 * step) if bull else max(strike, atm + 2 * step))
        return {"action": f"WRITE {'PE' if bull else 'CE'}", "strike": int(strike),
                "why": f"IV rich — sell the {'put' if bull else 'call'} wall, don't pay up"}

    # 2. Theta bleed / IV crush / unclear option flow / late session → futures.
    if (iv_crush or oi_unclear or theta_high) and conviction >= _ENTRY_CONF:
        cause = ("IV crushing" if iv_crush else "option flow unclear" if oi_unclear
                 else "late-session theta")
        return {"action": f"{'BUY' if bull else 'SELL'} FUT", "strike": None,
                "why": f"{cause} — futures carry the move without theta drag"}

    # 3. Cheap, stable IV → buy the option for cheap convexity.
    if iv_reg == "cheap":
        return {"action": f"BUY {'CE' if bull else 'PE'}", "strike": int(atm),
                "why": f"IV cheap — long {'call' if bull else 'put'} for cheap convexity"}

    # 4. Normal IV → ATM option, best delta.
    return {"action": f"BUY {'CE' if bull else 'PE'}", "strike": int(atm),
            "why": f"normal IV — ATM {'call' if bull else 'put'} best delta"}


# ── Current position (inferred from the paper ledger) ───────────────────────────
def _current_state(sym: str) -> str:
    try:
        import intraday_trades
        for t in intraday_trades.get_ledger().open_trades():
            if t.get("index_sym") == sym:
                return "LONG_DELTA" if t.get("direction") == "CE" else "SHORT_DELTA"
    except Exception:
        pass
    return "FLAT"


_STATE_OF = {"LONG": "LONG_DELTA", "SHORT": "SHORT_DELTA", "FLAT": "FLAT"}


# ── Core ────────────────────────────────────────────────────────────────────────
def conduct(sym: str, current_state: str | None = None,
            as_of: datetime.datetime | None = None, date: str | None = None,
            pb: dict | None = None, rf: dict | None = None, td: dict | None = None) -> dict:
    # pb/rf/td may be injected by MarketSnapshot so the whole page reflects ONE
    # market state and these primitives are computed once, not per panel. When not
    # injected, compute them here exactly as before (backward compatible).
    now = as_of or datetime.datetime.now(tz=IST)
    pb = pb if pb is not None else OP.playbook_index(sym, as_of=as_of, date=date)
    rf = rf if rf is not None else RF.forecast_index(sym, as_of=as_of, date=date)
    td = td if td is not None else TD.analyze_index(sym, date=date, as_of=as_of)

    if not td.get("has_data"):
        return {"sym": sym, "label": LABELS.get(sym, sym), "has_data": False,
                "note": td.get("note", "no live data")}

    wo = _w_open(now)
    open_dir = float(pb.get("composite", 0.0)) if pb.get("has_data") else 0.0
    reg_v, reg_detail = _regime_vote(rf)
    mom_v, mom_detail = _momentum_vote(td)
    oi_v = float(td.get("oi_bias", 0) or 0)

    live = _W_LIVE["regime"] * reg_v + _W_LIVE["momentum"] * mom_v + _W_LIVE["oi"] * oi_v
    fused = float(np.clip(wo * open_dir + (1 - wo) * live, -1, 1))

    direction = "LONG" if fused > _DIR_DEADBAND else "SHORT" if fused < -_DIR_DEADBAND else "FLAT"
    drivers = [("open", open_dir, f"thesis {pb.get('action','—')}" if pb.get("has_data") else "warming"),
               ("regime", reg_v, reg_detail),
               ("momentum", mom_v, mom_detail),
               ("OI", oi_v, td.get("oi_msg", ""))]
    agree = sum(1 for _, v, _ in drivers if v != 0 and (v > 0) == (fused > 0))
    conviction = int(np.clip(abs(fused) / 0.5 * 70 + max(0, agree - 2) * 8, 0, 95))

    # Live microstructure for the instrument tree.
    iv_now, iv_chg = pb.get("iv_now", 0) or 0, pb.get("iv_chg", 0) or 0
    iv_reg   = _iv_regime(sym, iv_now, iv_chg)
    iv_crush = iv_chg <= -OP._IV_PUMP
    oi_unclear = any(k in (td.get("oi_msg", "") or "")
                     for k in ("no clear", "range compression", "both walls"))
    theta_high = now.time() >= _THETA_CUTOVER
    spot = td.get("spot") or pb.get("spot") or 0

    rec = select_instrument(sym, direction, conviction, iv_reg, iv_crush, theta_high,
                            oi_unclear, pb.get("call_wall"), pb.get("put_wall"), spot)

    # Transition vs current position, with anti-whipsaw hysteresis.
    cur = current_state or _current_state(sym)
    tgt = _STATE_OF[direction]
    inval = pb.get("invalidation")
    breached = bool(inval and spot and (
        (cur == "LONG_DELTA" and spot < inval) or (cur == "SHORT_DELTA" and spot > inval)))
    shock = (td.get("syn", {}).get("verdict", "").startswith("FRESH"))
    # A conviction-driven flip also needs the move to be CONFIRMED (|z|≥2σ). The
    # 0.7-2σ drift band carries zero forward edge (swing_classifier_backtest.py), so
    # flipping a committed stance on it is pure whipsaw cost. Hard invalidations
    # (breach / fresh-impulse shock) still flip regardless.
    mom_confirmed = bool(td.get("syn", {}).get("confirmed", False))
    if cur == tgt:
        transition, act_now = "HOLD", False
    elif tgt == "FLAT":
        transition, act_now = "CLOSE", (conviction >= _ENTRY_CONF or breached)
    elif cur == "FLAT":
        transition, act_now = f"OPEN {direction}", conviction >= _ENTRY_CONF
    else:  # opposite → flip; needs the higher bar AND a confirmed move
        flip_ok = breached or shock or (conviction >= _FLIP_CONF and mom_confirmed)
        transition = f"FLIP {cur}→{tgt}" if flip_ok else f"CLOSE {cur} (wait to flip)"
        act_now = flip_ok

    return {
        "sym": sym, "label": LABELS.get(sym, sym), "has_data": True, "now": now.strftime("%H:%M:%S"),
        "fused": round(fused, 3), "direction": direction, "conviction": conviction,
        "w_open": round(wo, 2), "target_state": tgt, "current_state": cur,
        "transition": transition, "act_now": act_now, "agree": agree,
        "mom_confirmed": mom_confirmed, "zmax": td.get("syn", {}).get("zmax", 0.0),
        "instrument": rec, "drivers": drivers, "invalidation": inval, "spot": spot,
        "iv_regime": iv_reg, "regime_eta": (f"{rf.get('next_dir')} {rf.get('eta')}"
                    if rf.get("stage") in ("BUILDING", "IMMINENT") and rf.get("next_dir") else ""),
        "opening_action": pb.get("action") if pb.get("has_data") else None,
        "opening_dir": pb.get("direction") if pb.get("has_data") else None,
    }


def conduct_all(as_of=None, date=None) -> dict:
    return {s: conduct(s, as_of=as_of, date=date) for s in INDEX_SYMBOLS}


# ── CLI ─────────────────────────────────────────────────────────────────────────
_DIR_CLR = {"LONG": "▲", "SHORT": "▼", "FLAT": "•"}


def _print(r: dict) -> None:
    if not r.get("has_data"):
        print(f"\n  {r['label']:<13} — {r.get('note')}")
        return
    inst = r["instrument"]
    strike = f" {inst['strike']:,}" if inst.get("strike") else ""
    move_tag = (f"CONFIRMED {r.get('zmax',0):.1f}σ" if r.get("mom_confirmed")
                else f"drift {r.get('zmax',0):.1f}σ (<2σ — no fwd edge)")
    print(f"\n  {r['label']:<13} {_DIR_CLR[r['direction']]} {r['direction']:<5} "
          f"conv {r['conviction']:>2}%   w_open {r['w_open']:.2f}   {move_tag}   {r['now']}")
    print(f"     ACTION  {inst['action']}{strike}   ({inst['why']})")
    # evolution: opening thesis → now
    if r.get("opening_action"):
        print(f"     evolved  opened {r['opening_dir']} ({r['opening_action']}) "
              f"→ now {r['direction']} [{r['transition']}{'' if r['act_now'] else ' — wait'}]")
    for name, v, detail in r["drivers"]:
        a = "▲" if v > 0.05 else "▼" if v < -0.05 else "·"
        print(f"       {a} {name:<9} {v:+.2f}  {detail}")
    tail = []
    if r.get("invalidation"):
        tail.append(f"wrong {'below' if r['direction']=='LONG' else 'above'} {r['invalidation']:,.0f}")
    if r.get("regime_eta"):
        tail.append(f"regime → {r['regime_eta']}")
    if tail:
        print(f"     {'  ·  '.join(tail)}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Session Conductor — continuous fused stance per index")
    ap.add_argument("--symbol", help="Single index (e.g. NIFTY). Default: all 4")
    args = ap.parse_args()
    print("=" * 76)
    print("  SESSION CONDUCTOR  —  fused live stance + instrument call (decision-support)")
    print("=" * 76)
    if args.symbol:
        sym = LABEL_TO_SYM.get(args.symbol.upper().replace(" ", ""), args.symbol)
        _print(conduct(sym))
    else:
        for s in INDEX_SYMBOLS:
            _print(conduct(s))
    print()


if __name__ == "__main__":
    main()
