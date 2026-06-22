"""
morning_read.py  --  First-20-minute market-condition read.

Fuses two information edges into one structured, honest read per index:
  EOD layer  (last night)  -> daily_context_bridge: DCM forecast range/target,
                              OI walls (R/S), max pain, EOD PCR, FII/FPI flows,
                              VIX, breadth, HMM regime.
  LIVE layer (this morning) -> data/intraday/live/<today>_*.parquet mirrors:
                              ticks, candles, OI snapshots, futures basis,
                              multi-TF technical scores, 9-layer composite.

Deliberately NOT a directional arrow. Repo backtests show the index engine has
~0 directional edge and that EOD OI walls / max pain do NOT bound next-day price
better than a plain ATR band. So this produces:
  * a regime read (trend vs balanced, who holds control intraday),
  * an expected range (DCM ~1sigma band) with how much is already spent,
  * OI walls / max pain as POSITIONING CONTEXT only,
  * a scenario tree with explicit invalidation levels.

Usage:  .venv\\Scripts\\python.exe morning_read.py
"""

import datetime
import glob
import time
from pathlib import Path

import pandas as pd

from daily_context_bridge import get_bridge
from core.constants import IST   # single source of truth

HERE = Path(__file__).parent
LIVE = HERE / "data" / "intraday" / "live"
TODAY = datetime.datetime.now(tz=IST).date().isoformat()

SYMS = [
    ("NIFTY",   "NSE:NIFTY50-INDEX"),
    ("BANK",    "NSE:NIFTYBANK-INDEX"),
    ("FIN",     "NSE:FINNIFTY-INDEX"),
    ("MIDCAP",  "NSE:MIDCPNIFTY-INDEX"),
]

OPEN_T  = datetime.time(9, 15)
OR_END  = datetime.time(9, 30)   # 15-min opening range


def _ist(series: pd.Series) -> pd.Series:
    """Coerce a parquet ts column to tz-aware IST."""
    s = pd.to_datetime(series, utc=True)
    return s.dt.tz_convert(IST)


def _load(kind: str) -> pd.DataFrame:
    f = LIVE / f"{TODAY}_{kind}.parquet"
    if not f.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(f)
    except Exception:
        return pd.DataFrame()


def _last_per_symbol(df: pd.DataFrame) -> dict:
    if df.empty or "symbol" not in df.columns:
        return {}
    df = df.sort_values("ts")
    return {sym: g.iloc[-1] for sym, g in df.groupby("symbol")}


def _fmt(x, nd=0):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "—"
    try:
        return f"{float(x):,.{nd}f}"
    except Exception:
        return str(x)


def main() -> None:
    now = datetime.datetime.now(tz=IST)
    mins_in = (now - now.replace(hour=9, minute=15, second=0, microsecond=0)).total_seconds() / 60

    b = get_bridge()
    for _ in range(30):
        if b.is_available():
            break
        time.sleep(0.5)

    ticks   = _load("ticks")
    candles = _load("candles")
    oi_snap = _load("oi_snapshots")
    futq    = _load("futures_quotes")
    sigs    = _load("signals")
    setups  = _load("trade_setups")

    last_tick = _last_per_symbol(ticks)
    last_oi   = _last_per_symbol(oi_snap)
    last_futq = _last_per_symbol(futq)
    last_sig  = _last_per_symbol(sigs)
    last_setup = _last_per_symbol(setups)

    # opening-range high/low per symbol (first 15 min)
    or_hl = {}
    if not ticks.empty:
        t = ticks.copy()
        t["ist"] = _ist(t["ts"])
        win = t[t["ist"].dt.time <= OR_END]
        for sym, g in win.groupby("symbol"):
            or_hl[sym] = (g["day_high"].max(), g["day_low"].min())
    or_complete = now.time() >= OR_END

    print("=" * 78)
    print(f"  MORNING READ  ·  {now:%Y-%m-%d %H:%M:%S IST}  ·  {mins_in:.0f} min into session")
    print(f"  Opening range (09:15-09:30): {'COMPLETE' if or_complete else 'STILL FORMING'}")
    print("=" * 78)

    for name, fsym in SYMS:
        d = b.get_panel_data(fsym)
        common = d.get("__common__", {})
        tk = last_tick.get(fsym)
        oi = last_oi.get(fsym)
        fq = last_futq.get(fsym)
        sg = last_sig.get(fsym)
        st = last_setup.get(fsym)

        prev_close = d.get("eod_close")            # last night's official close
        dcm_lo, dcm_hi = d.get("range_low"), d.get("range_high")
        dcm_tgt = d.get("target_close")
        em = d.get("expected_move_pts")
        R, S = d.get("top_call_strike"), d.get("top_put_strike")
        mp_eod = d.get("max_pain_price")

        ltp = float(tk["ltp"]) if tk is not None else None
        d_open = float(tk["day_open"]) if tk is not None else None
        d_hi = float(tk["day_high"]) if tk is not None else None
        d_lo = float(tk["day_low"]) if tk is not None else None
        chp = float(tk["chp"]) if tk is not None else None

        gap = (d_open - prev_close) if (d_open and prev_close) else None
        gap_pct = (gap / prev_close * 100) if (gap is not None and prev_close) else None

        vwap = float(sg["vwap_15min"]) if (sg is not None and pd.notna(sg.get("vwap_15min"))) else None
        realized = (d_hi - d_lo) if (d_hi and d_lo) else None
        em_used = (realized / em * 100) if (realized and em) else None

        print(f"\n──────── {name}  ({fsym}) ────────")
        # --- price location ---
        loc = []
        if ltp is not None:
            loc.append(f"LTP {_fmt(ltp,1)} ({chp:+.2f}%)")
        if gap is not None:
            loc.append(f"gap {gap:+.0f} ({gap_pct:+.2f}%) from {_fmt(prev_close,0)}")
        print("  PRICE   " + "  ·  ".join(loc))
        if d_hi and d_lo:
            print(f"          day range {_fmt(d_lo,0)}–{_fmt(d_hi,0)}  (span {realized:.0f} pts)")
        if vwap is not None and ltp is not None:
            side = "ABOVE" if ltp >= vwap else "BELOW"
            print(f"          VWAP {_fmt(vwap,1)} → price {side} VWAP "
                  f"({'buyers' if side=='ABOVE' else 'sellers'} control intraday)")
        if sym_or := or_hl.get(fsym):
            orh, orl = sym_or
            tag = ""
            if ltp is not None:
                if ltp > orh: tag = "→ broken ABOVE OR (trend-up attempt)"
                elif ltp < orl: tag = "→ broken BELOW OR (trend-down attempt)"
                else: tag = "→ INSIDE OR (balance/chop)"
            print(f"          opening range {_fmt(orl,0)}–{_fmt(orh,0)} {tag}")

        # --- EOD expected band & context ---
        print(f"  EOD MAP DCM today: target {_fmt(dcm_tgt,0)} · ~1σ band {_fmt(dcm_lo,0)}–{_fmt(dcm_hi,0)} "
              f"(EM ±{_fmt(em,0)})  [{d.get('direction')} {d.get('confidence')}]")
        if em_used is not None:
            print(f"          realized span = {em_used:.0f}% of expected day move "
                  f"({'most of the move may be spent' if em_used>70 else 'room left in the band'})")
        ctx = []
        if R: ctx.append(f"call wall/R {_fmt(R,0)}")
        if S: ctx.append(f"put wall/S {_fmt(S,0)}")
        if mp_eod: ctx.append(f"max-pain {_fmt(mp_eod,0)}")
        print("          OI context (positioning, NOT a hard boundary): " + " · ".join(ctx))

        # --- intraday OI shift vs EOD ---
        if oi is not None:
            i_pcr = float(oi["pcr"]); i_cw = oi["call_wall"]; i_pw = oi["put_wall"]; i_mp = oi["max_pain"]
            skew = float(oi["put_skew"]) if pd.notna(oi.get("put_skew")) else None
            iv = float(oi["atm_iv"]) if pd.notna(oi.get("atm_iv")) else None
            eod_pcr = d.get("eod_pcr")
            pcr_dir = ""
            if eod_pcr:
                pcr_dir = ("put-writers adding (floor firming)" if i_pcr > eod_pcr * 1.03
                           else "call-writers adding (cap firming)" if i_pcr < eod_pcr * 0.97
                           else "stable")
            print(f"  LIVE OI PCR {i_pcr:.2f} (EOD {_fmt(eod_pcr,2)} → {pcr_dir}) · "
                  f"walls {_fmt(i_pw,0)}/{_fmt(i_cw,0)} · max-pain {_fmt(i_mp,0)}"
                  + (f" · ATM IV {iv:.1f}" if iv else "")
                  + (f" · skew {skew:+.1f}" if skew is not None else ""))

        # --- futures basis ---
        if fq is not None:
            print(f"  FUT     near basis {_fmt(fq['near_basis'],1)} · {fq['term_structure']} "
                  f"· vol {_fmt(fq['near_vol'],0)}")

        # --- multi-TF technical + composite ---
        if sg is not None:
            print(f"  TECH    {sg['overall']} (wt {float(sg['weighted_score']):+.2f}) · "
                  f"5m {sg['signal_5min']}/15m {sg['signal_15min']}/1h {sg['signal_60min']}/D {sg['signal_daily']} "
                  f"· RSI5 {_fmt(sg['rsi_5min'],0)}/RSI15 {_fmt(sg['rsi_15min'],0)}")
        if st is not None:
            print(f"  9-LAYER {st['signal']} comp {float(st['composite_score']):+.2f} "
                  f"(agree {_fmt(st['agreement'],1)}) · phase {st['phase']}")

        # --- macro tape (shared) ---
    print("\n" + "─" * 78)
    print("  MACRO TAPE (last night, shared):")
    nd = b.get_panel_data("NSE:NIFTY50-INDEX")
    cm = nd.get("__common__", {})
    print(f"    VIX {_fmt(cm.get('india_vix'),1)} · breadth {_fmt(cm.get('breadth_pct'),0)}% "
          f"· FII fut 5D {_fmt(nd.get('fii_5d_cr'),0)}Cr · FPI cash 5D {_fmt(nd.get('fpi_equity_5d_cr'),0)}Cr "
          f"· HMM {nd.get('hmm_state')}")
    print("=" * 78)


if __name__ == "__main__":
    main()
