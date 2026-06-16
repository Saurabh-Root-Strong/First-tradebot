"""
smart_money.py — institutional footprint detector from option + futures flow.

Most "OI analysis" is wrong because it reads OI alone. The footprint lives in
OI × VOLUME × PREMIUM together, with two principles:

  1. Quality over size — |ΔOI| / Volume is the discriminator. Near 1 = trades that
     OPENED positions (committed money); near 0 = churn (noise). A big volume number
     with flat OI is retail scalping, not a footprint.

  2. Writers are the smart money — institutions SELL options, retail buys. So the
     read is what the writers do, via OI + premium direction:
         PE OI↑ prem↓  put WRITING        → floor defended      bullish
         CE OI↑ prem↓  call WRITING       → ceiling capped      bearish
         CE OI↓ prem↑  call SHORT-COVER   → squeeze             bullish
         PE OI↓ prem↑  put SHORT-COVER    → floor abandoned     bearish
         (option BUYING / long-unwinding = lighter, retail-tilted weight)

Each strike's signal is weighted by interval volume AND its OI/volume efficiency, so
high-churn/low-conviction strikes are discounted. Futures add a delta-1 aggression
read (volume + price + basis). The key output is whether the footprint LEADS price
(divergence — institutions positioned before the move) or merely confirms it.

Reads the lock-free mirrors (chain_snapshots / futures_quotes / oi_snapshots), so it
replays on any captured day (as_of/date) just like the conductor.

  .venv\\Scripts\\python.exe smart_money.py
  .venv\\Scripts\\python.exe smart_money.py --symbol NIFTY --window 20
"""
from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

IST = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
LIVE_DIR = Path("data") / "intraday" / "live"
INDEX_SYMBOLS = ["NSE:NIFTY50-INDEX", "NSE:NIFTYBANK-INDEX",
                 "NSE:FINNIFTY-INDEX", "NSE:MIDCPNIFTY-INDEX"]
LABELS = {"NSE:NIFTY50-INDEX": "NIFTY 50", "NSE:NIFTYBANK-INDEX": "BANK NIFTY",
          "NSE:FINNIFTY-INDEX": "FIN NIFTY", "NSE:MIDCPNIFTY-INDEX": "MIDCAP NIFTY"}
LABEL_TO_SYM = {"NIFTY": INDEX_SYMBOLS[0], "BANKNIFTY": INDEX_SYMBOLS[1],
                "FINNIFTY": INDEX_SYMBOLS[2], "MIDCPNIFTY": INDEX_SYMBOLS[3],
                "MIDCAPNIFTY": INDEX_SYMBOLS[3]}

WINDOW_MIN = 20          # lookback for interval flow
_NOISE_OI  = 1.0         # ignore strikes whose interval |ΔOI| is below this (in lots/1)

# classification → (directional sign, base weight). Writing/covering = institutional.
_BASE = {"write": 1.0, "cover": 0.9, "buy": 0.5, "unwind": 0.4}


def _today() -> str:
    return datetime.datetime.now(tz=IST).date().isoformat()


def _read(tbl: str, date, as_of):
    p = LIVE_DIR / f"{date or _today()}_{tbl}.parquet"
    if not p.exists() or p.stat().st_size < 100:
        return None
    df = pd.read_parquet(p)
    df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert(IST)
    if as_of is not None:
        df = df[df["ts"] <= pd.Timestamp(as_of)]
    return df if len(df) else None


def _classify(side: str, d_oi: float, d_prem: float) -> tuple[str, float]:
    """Return (kind, signed_direction) where +1 bullish, -1 bearish."""
    if side == "CE":
        if d_oi > 0:  return ("write", -1.0) if d_prem <= 0 else ("buy", +1.0)
        else:         return ("cover", +1.0) if d_prem >= 0 else ("unwind", -1.0)
    else:  # PE
        if d_oi > 0:  return ("write", +1.0) if d_prem <= 0 else ("buy", -1.0)
        else:         return ("cover", -1.0) if d_prem >= 0 else ("unwind", +1.0)


def footprint_index(sym: str, window_min: int = WINDOW_MIN,
                    date=None, as_of=None) -> dict:
    chain = _read("chain_snapshots", date, as_of)
    if chain is None:
        return {"sym": sym, "has_data": False, "note": "no chain snapshots yet"}
    chain = chain[chain["symbol"] == sym]
    if chain is None or chain.empty:
        return {"sym": sym, "has_data": False, "note": "no chain for this index yet"}

    # Day-level footprint from the LATEST snapshot: oich = net OI built TODAY (vs prev
    # close), ltpch = premium move today, volume = today's activity. Robust (no stale-
    # volume-diff problem) and inherently churn-discounting — net OI is what committed.
    now_ts = chain["ts"].max()
    cur = chain[chain["ts"] == now_ts]
    tot_vol = max(float(cur["volume"].sum()), 1.0)

    rows = []
    for c in cur.itertuples():
        oich = float(c.oich or 0)
        if abs(oich) < _NOISE_OI:
            continue
        ltpch = float(c.ltpch or 0)
        vol = max(float(c.volume or 0), 0.0)
        kind, sign = _classify(c.side, oich, ltpch)
        vol_share = vol / tot_vol
        # net OI build is the conviction-weighted size; volume share emphasises the
        # strikes where that positioning came on heavy activity.
        weight = abs(oich) * (1.0 + vol_share) * _BASE[kind]
        eff = min(abs(oich) / max(vol, 1.0), 1.0)
        rows.append({"strike": int(c.strike), "side": c.side, "kind": kind,
                     "sign": sign, "d_oi": oich, "d_vol": vol, "eff": eff,
                     "weight": weight, "signed": weight * sign})
    if not rows:
        return {"sym": sym, "has_data": False, "note": "no net OI build yet"}

    fr = pd.DataFrame(rows)
    net = fr["signed"].sum()
    tot = fr["weight"].sum()
    lean = float(np.clip(net / tot, -1, 1)) if tot > 0 else 0.0

    # spot move today (vs prev close) — to test whether the footprint leads or confirms
    oi = _read("oi_snapshots", date, as_of)
    spot_chg = 0.0
    if oi is not None:
        o = oi[oi["symbol"] == sym].sort_values("ts")
        if len(o) >= 2:
            sp_now = float(o.iloc[-1]["spot"])
            older = o[o["ts"] <= now_ts - pd.Timedelta(minutes=window_min)]
            sp_then = float(older.iloc[-1]["spot"]) if len(older) else float(o.iloc[0]["spot"])
            spot_chg = (sp_now - sp_then) / sp_then * 100 if sp_then else 0.0

    # futures aggression (volume + price + basis) — no futures OI captured yet
    fut = _read("futures_quotes", date, as_of)
    fut_note, fut_sign = "", 0.0
    if fut is not None:
        f = fut[fut["symbol"] == sym].sort_values("ts")
        if len(f) >= 2:
            d_fvol = float(f.iloc[-1]["near_vol"]) - float(f.iloc[0]["near_vol"])
            d_basis = float(f.iloc[-1]["near_basis"]) - float(f.iloc[0]["near_basis"])
            agg = (1 if spot_chg > 0 else -1 if spot_chg < 0 else 0)
            fut_sign = agg * (1 if d_basis >= 0 else 0.5)
            fut_note = (f"futures vol +{d_fvol/1e5:.1f}L, basis {d_basis:+.0f} "
                        f"→ {'aggressive longs' if fut_sign>0 else 'aggressive shorts' if fut_sign<0 else 'neutral'}")

    # Magnitude gate — a clean direction off 0.01L of OI is noise, not a footprint.
    # Conviction must scale with the absolute net OI built (5L = full confidence).
    total_build = float(fr["d_oi"].abs().sum())
    mag = min(total_build / 5e5, 1.0)
    thin = total_build < 1e5                         # < ~1L contracts net build
    conviction = int(np.clip((abs(lean) * 70 + min(fr["eff"].mean(), 1) * 25) * mag, 0, 95))
    if thin:
        direction = "THIN"
        leads = False
    else:
        direction = "BULLISH" if lean > 0.2 else "BEARISH" if lean < -0.2 else "NEUTRAL"
        # lead vs confirm: footprint strong but price hasn't (yet) moved the same way
        leads = abs(lean) > 0.25 and (abs(spot_chg) < 0.05 or (lean > 0) != (spot_chg > 0))

    top = fr.reindex(fr["weight"].sort_values(ascending=False).index).head(4)
    foot = [f"{int(r.strike)}{r.side} {r.kind} (ΔOI {r.d_oi/1e5:+.2f}L · vol {r.d_vol/1e5:.1f}L · eff {r.eff:.2f})"
            for r in top.itertuples()]

    return {"sym": sym, "label": LABELS.get(sym, sym), "has_data": True,
            "now": str(now_ts), "window": window_min,
            "lean": round(lean, 3), "direction": direction, "conviction": conviction,
            "spot_chg": round(spot_chg, 3), "leads": leads, "fut_note": fut_note,
            "footprints": foot, "n_strikes": len(fr),
            "build_L": round(total_build / 1e5, 1)}


def _print(r: dict) -> None:
    if not r.get("has_data"):
        print(f"\n  {LABELS.get(r['sym'], r['sym']):<13} — {r.get('note')}")
        return
    arrow = {"BULLISH": "▲", "BEARISH": "▼", "NEUTRAL": "•", "THIN": "·"}[r["direction"]]
    tag = ("  ⚡LEADS price" if r["leads"]
           else "" if r["direction"] == "THIN" else "  (confirms price)")
    print(f"\n  {r['label']:<13} {arrow} {r['direction']:<7} conv {r['conviction']:>2}%   "
          f"lean {r['lean']:+.2f}   build {r['build_L']}L   spot {r['spot_chg']:+.2f}% / {r['window']}m{tag}")
    for f in r["footprints"]:
        print(f"       • {f}")
    if r["fut_note"]:
        print(f"       ⏩ {r['fut_note']}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Smart-money footprint from option/futures flow")
    ap.add_argument("--symbol", help="single index (NIFTY/BANKNIFTY/...). default: all")
    ap.add_argument("--window", type=int, default=WINDOW_MIN, help=f"lookback min (default {WINDOW_MIN})")
    args = ap.parse_args()
    print("=" * 80)
    print(f"  SMART-MONEY FOOTPRINT  —  option+futures flow, last {args.window}m  "
          f"(OI × volume × premium)")
    print("=" * 80)
    syms = [LABEL_TO_SYM.get(args.symbol.upper().replace(" ", ""), args.symbol)] if args.symbol else INDEX_SYMBOLS
    for s in syms:
        _print(footprint_index(s, args.window))
    print()


if __name__ == "__main__":
    main()
