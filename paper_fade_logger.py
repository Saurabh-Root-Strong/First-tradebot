"""
paper_fade_logger.py — PAPER harness for the band-fade edge + the options-context it needs.

The validated slice (backtest_band_fade.py): fade a candle REJECTION at the sigma-band edge,
enter at the edge, target the anchor, 1-band stop, ~45min hold. Hardened OOS it clears the
cost floor only on LIQUID BANK/NIFTY (~+0.9bps) — thin, and with un-modelled adverse
selection on top. It CANNOT be trusted from backtest alone, and the OI/COI/premium
confirmation the user wants CANNOT be backtested (option data exists only in the ~34-day
live capture). Both gaps close the same way: a forward PAPER log that, at every signal,
records the candle, the exact paper fill/stop, AND the full option context — then grades
outcomes as the sample grows.

This module is that log. Wired to NOTHING that trades. Two uses, one code path (parity):
  • REPLAY a captured day  → backfills the ledger NOW from days that have OI/premium
  • LIVE tick (cron)       → same scan/settle at wall-clock now

Signal (identical to _fade_day so paper == backtest): anchor = the bar HB back; band =
K*sigma*(HB^HURST) around the anchor close; the CURRENT closed bar pierces an edge with a
REJECTION wick (close in the far half) → fade to the anchor, 1-band stop. Regime gate: no
fade INTO a strong trend (Kaufman ER). Lookahead-free: the forming bar is dropped; all
reads are ts<=as_of.

    .venv\\Scripts\\python.exe paper_fade_logger.py --replay 2026-07-15
    .venv\\Scripts\\python.exe paper_fade_logger.py --replay-all
    .venv\\Scripts\\python.exe paper_fade_logger.py --live          # one now-scan (cron)
    .venv\\Scripts\\python.exe paper_fade_logger.py --show
"""
from __future__ import annotations

import argparse
import datetime
import sys

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from core.constants import INDEX_SYMBOLS, LABELS, IST, NIFTY, DATA_DIR
from core.mirror_io import read_mirror
from core.market_calendar import days_to_expiry
import footprint_chart as fc
import intraday_scout as scout
from backtest_band_fade import COST_IDX_BPS, SLIP_IDX_BPS, FY, ER_TREND

K = 0.55
L_VOL = 24            # trailing 5m bars for sigma
HURST = 0.40
HB = 9               # hold / anchor-lookback in 5m bars (~45min = the validated hold)
TF = 5
LEDGER = DATA_DIR / "validation" / "paper_fade_ledger.parquet"
_SCAN_START = datetime.time(9, 45)
_SCAN_END = datetime.time(15, 15)

_COLS = ["date", "sym", "signal_ts", "side", "entry", "stop", "target", "spot_anchor",
         "atm", "clr", "sigma_pct", "band_pct", "er", "dte",
         "fade_side", "fade_oi", "fade_oich", "fade_vol", "fade_prem",
         "call_wall", "put_wall", "wall_dist_pct", "pcr",
         "status", "exit_ts", "exit_px", "net_bps", "outcome"]


def _er(seg: np.ndarray) -> float:
    """Kaufman efficiency ratio over a close segment (causal)."""
    if len(seg) < 3:
        return 0.0
    vol = np.abs(np.diff(seg)).sum()
    return abs(seg[-1] - seg[0]) / vol if vol > 0 else 0.0


def _closed_ohlc(sym: str, date, as_of):
    """Closed 5m OHLC arrays up to as_of (forming last bar dropped). None if thin."""
    ser = fc.build_series(sym, TF, date, as_of)
    if not ser.get("has_data"):
        return None
    o, h, l, c = (np.asarray(ser.get(k) or [], float) for k in ("open", "high", "low", "close"))
    if len(c) < L_VOL + HB + 3:
        return None
    return o[:-1], h[:-1], l[:-1], c[:-1]          # drop forming bar


_MID_N = HB          # rolling-mean window for the anchor (~45min)


def _signal(sym: str, date, as_of):
    """Band-fade rejection signal on the LAST closed bar, or None.

    Anchor = the CURRENT rolling mean (mid), NOT a stale HB-bars-ago price — the earlier
    stale-anchor version fired on continuation into a drift (fades momentum → stop-heavy;
    crosscheck vs backtest _fade_day proved it a parity bug). A current-mean anchor makes
    the excursion FRESH: price must stretch 1 band from where it has RECENTLY been, then
    reject → fade back to the mean. Forward-consistent, no lookahead."""
    ohlc = _closed_ohlc(sym, date, as_of)
    if ohlc is None:
        return None
    o, h, l, c = ohlc
    i = len(c) - 1                                  # last closed bar
    if i - L_VOL < 0:
        return None
    spot0 = float(c[i - _MID_N + 1:i + 1].mean())   # current rolling-mean anchor
    ret = np.diff(c[:i + 1]) / np.where(c[:i] > 0, c[:i], np.nan)
    sig = np.nanstd(ret[-L_VOL:]) * 100.0
    if not (sig > 0):
        return None
    band = K * sig * (HB ** HURST) / 100.0
    lo, up = spot0 * (1 - band), spot0 * (1 + band)
    side = entry = None
    if l[i] <= lo:
        side, entry = "L", lo
    elif h[i] >= up:
        side, entry = "S", up
    if side is None:
        return None
    rng = h[i] - l[i]
    clr = ((c[i] - l[i]) / rng) if rng > 0 else 0.5
    if not ((side == "S" and clr < 0.5) or (side == "L" and clr > 0.5)):
        return None                                 # need the rejection wick
    er = _er(c[i - 10:i + 1]) if i >= 10 else 0.0   # regime gate: no knife-catch
    if er >= ER_TREND:
        tsign = c[i] - c[i - 10]
        if (side == "S" and tsign > 0) or (side == "L" and tsign < 0):
            return None
    stop = entry * (1 - band) if side == "L" else entry * (1 + band)
    return {"side": side, "entry": round(entry, 2), "stop": round(stop, 2),
            "target": round(spot0, 2), "spot_anchor": round(spot0, 2),
            "atm": scout._atm(spot0, sym), "clr": round(clr, 3),
            "sigma_pct": round(sig, 4), "band_pct": round(band * 100, 3), "er": round(er, 3)}


def _opt_ctx(sym: str, date, as_of, atm, side) -> dict:
    """Option context at the signal: fade-strike OI/COI/vol/premium, walls, PCR, DTE.
    fade a SHORT (upper) = you'd buy the PE; a LONG (lower) = the CE."""
    fade_side = "PE" if side == "S" else "CE"
    out = {"fade_side": fade_side, "dte": days_to_expiry(date, weekly=(sym == NIFTY))}
    try:
        ch = read_mirror("chain_snapshots", date, as_of, sym)
    except Exception:
        ch = None
    if ch is None or not len(ch) or "oi" not in ch.columns:
        return out
    last = ch.sort_values("ts").groupby(["strike", "side"]).last().reset_index()
    ce, pe = last[last["side"] == "CE"], last[last["side"] == "PE"]
    row = last[(last["strike"] == atm) & (last["side"] == fade_side)]
    if len(row):
        r = row.iloc[-1]
        out.update(fade_oi=int(r.get("oi") or 0), fade_oich=int(r.get("oich") or 0),
                   fade_vol=int(r.get("volume") or 0),
                   fade_prem=float(r.get("ltp") or 0.0) or None)
    cw = int(ce.loc[ce["oi"].idxmax(), "strike"]) if len(ce) and ce["oi"].max() > 0 else None
    pw = int(pe.loc[pe["oi"].idxmax(), "strike"]) if len(pe) and pe["oi"].max() > 0 else None
    out["call_wall"], out["put_wall"] = cw, pw
    # distance to the wall the fade leans on (short leans on the call wall above; long on put)
    wall = cw if side == "S" else pw
    if wall and atm:
        out["wall_dist_pct"] = round((wall / atm - 1.0) * 100.0, 3)
    tot_ce, tot_pe = float(ce["oi"].sum()), float(pe["oi"].sum())
    out["pcr"] = round(tot_pe / tot_ce, 3) if tot_ce > 0 else None
    return out


def _load() -> pd.DataFrame:
    if LEDGER.exists():
        return pd.read_parquet(LEDGER)
    return pd.DataFrame(columns=_COLS)


def _save(df: pd.DataFrame) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(LEDGER, index=False)


def scan(date, as_of, led: pd.DataFrame) -> pd.DataFrame:
    """Detect a new signal per index (skip if that sym already has an OPEN paper trade)."""
    for sym in INDEX_SYMBOLS:
        open_here = led[(led.date == date) & (led.sym == sym) & (led.status == "OPEN")]
        if len(open_here):
            continue
        sig = _signal(sym, date, as_of)
        if sig is None:
            continue
        ctx = _opt_ctx(sym, date, as_of, sig["atm"], sig["side"])
        row = {"date": date, "sym": sym, "signal_ts": as_of.strftime("%H:%M"),
               "status": "OPEN", "exit_ts": None, "exit_px": None,
               "net_bps": None, "outcome": None, **sig, **ctx}
        led = pd.concat([led, pd.DataFrame([{c: row.get(c) for c in _COLS}])],
                        ignore_index=True)
    return led


def settle(date, as_of, led: pd.DataFrame) -> pd.DataFrame:
    """Close OPEN trades whose target/stop hit, or HB bars elapsed (time-exit). Managed exit
    with per-index cost + adverse stop-slippage — parity with the HARD backtest."""
    for idx in led.index[(led.date == date) & (led.status == "OPEN")]:
        r = led.loc[idx]
        sym = r["sym"]; short = FY.get(_fy_key(sym), sym)
        ohlc = _closed_ohlc(sym, date, as_of)
        if ohlc is None:
            continue
        o, h, l, c = ohlc
        ser_ts = fc.build_series(sym, TF, date, as_of).get("ts") or []
        # locate the signal bar index by time
        sig_t = r["signal_ts"]
        times = [pd.Timestamp(t).strftime("%H:%M") for t in ser_ts[:-1]]  # closed bars
        if sig_t not in times:
            continue
        s_i = times.index(sig_t)
        end_i = min(s_i + HB, len(c) - 1)
        cost = COST_IDX_BPS.get(short, 5.0); slip = SLIP_IDX_BPS.get(short, 3.0) / 1e4
        side, entry, stop, target = r["side"], r["entry"], r["stop"], r["target"]
        exit_px = None; outcome = None; hit_stop = False; exit_i = None
        for j in range(s_i + 1, end_i + 1):
            if side == "S":
                if h[j] >= stop:
                    exit_px, outcome, hit_stop, exit_i = stop, "stop", True, j; break
                if l[j] <= target:
                    exit_px, outcome, exit_i = target, "target", j; break
            else:
                if l[j] <= stop:
                    exit_px, outcome, hit_stop, exit_i = stop, "stop", True, j; break
                if h[j] >= target:
                    exit_px, outcome, exit_i = target, "target", j; break
        if exit_px is None:
            if end_i > s_i and (as_of.time() >= _SCAN_END or end_i == len(c) - 1):
                exit_px, outcome, exit_i = c[end_i], "time", end_i     # time-exit only once matured
            else:
                continue                                                # still open, not matured
        if hit_stop and slip:
            exit_px = exit_px * (1 + slip) if side == "S" else exit_px * (1 - slip)
        net = ((entry / exit_px - 1) if side == "S" else (exit_px / entry - 1)) * 1e4 - cost
        led.loc[idx, ["status", "exit_ts", "exit_px", "net_bps", "outcome"]] = \
            ["CLOSED", times[exit_i] if exit_i is not None else None,
             round(float(exit_px), 2), round(float(net), 2), outcome]
    return led


def _fy_key(sym):
    """Map an INDEX_SYMBOLS entry to the FY short-name key used by the cost dicts."""
    return sym if sym in FY else next((k for k in FY if FY[k] in str(sym)), sym)


def replay_day(date: str) -> None:
    led = _load()
    led = led[led.date != date]                     # idempotent re-replay of this day
    d0 = datetime.date.fromisoformat(date)
    t = datetime.time(_SCAN_START.hour, _SCAN_START.minute)
    step = datetime.timedelta(minutes=TF)
    cur = datetime.datetime.combine(d0, t, tzinfo=IST)
    end = datetime.datetime.combine(d0, _SCAN_END, tzinfo=IST)
    n0 = len(led)
    while cur <= end:
        led = settle(date, cur, led)
        led = scan(date, cur, led)
        cur += step
    led = settle(date, end, led)                    # final maturation pass
    _save(led)
    day = led[led.date == date]
    print(f"  {date}: {len(day)} signals ({int((day.status=='CLOSED').sum())} closed), "
          f"ledger now {len(led)} rows (+{len(led)-n0})")


def _captured_days():
    out = set()
    for p in (DATA_DIR / "intraday" / "live").glob("*_oi_snapshots.parquet"):
        if p.stat().st_size < 1024:
            continue
        stem = p.name.split("_")[0]
        try:
            datetime.date.fromisoformat(stem); out.add(stem)
        except ValueError:
            continue
    return sorted(out)


def show() -> None:
    led = _load()
    if not len(led):
        print("  ledger empty — run --replay-all first."); return
    cl = led[led.status == "CLOSED"].copy()
    print(f"\n  PAPER-FADE LEDGER — {len(led)} signals, {len(cl)} closed, "
          f"{led.date.nunique()} days ({led.date.min()}..{led.date.max()})")
    if not len(cl):
        return
    cl["net_bps"] = pd.to_numeric(cl["net_bps"], errors="coerce")
    print(f"  overall: mean {cl.net_bps.mean():+.1f}bps  win {100*(cl.net_bps>0).mean():.0f}%  "
          f"(NOTE: paper fills — real adverse selection not captured)")
    print("  by index:")
    for sym, g in cl.groupby("sym"):
        print(f"    {LABELS.get(sym, sym):<11} n={len(g):>3}  mean {g.net_bps.mean():+5.1f}bps  "
              f"win {100*(g.net_bps>0).mean():>3.0f}%  outcomes "
              f"{dict(g.outcome.value_counts())}")
    # the fusion seed: does OI/COI context separate winners from losers?
    if "fade_oich" in cl.columns and cl["fade_oich"].notna().any():
        cl["coi_pos"] = pd.to_numeric(cl["fade_oich"], errors="coerce") > 0
        for lab, sub in cl.groupby("coi_pos"):
            if len(sub) >= 3:
                print(f"  fade-strike COI {'building(+)' if lab else 'falling(-)'}: "
                      f"mean {sub.net_bps.mean():+.1f}bps n={len(sub)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay")
    ap.add_argument("--replay-all", action="store_true")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()
    if a.replay:
        replay_day(a.replay)
    elif a.replay_all:
        days = _captured_days()
        print(f"replaying {len(days)} captured days...")
        for d in days:
            try:
                replay_day(d)
            except Exception as e:
                print(f"  {d}: ERROR {e}")
        show()
    elif a.live:
        now = datetime.datetime.now(IST)
        led = _load()
        led = settle(now.strftime("%Y-%m-%d"), now, led)
        led = scan(now.strftime("%Y-%m-%d"), now, led)
        _save(led)
        print(f"  live scan @ {now:%H:%M} — ledger {len(led)} rows")
    else:
        show()


if __name__ == "__main__":
    main()
