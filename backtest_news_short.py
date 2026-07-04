"""
backtest_news_short.py — do severe NEGATIVE stock events (fraud / SEBI action /
insolvency / auditor exit / downgrade) predict a shortable forward drift?

The user's thesis: "big allegation → good opportunity for a short / swing". This
measures it BEFORE the desk acts on it (house rule: no signal is wired to an
action until the ledger says it pays).

Data
  events : data/intraday/live/<date>_news_events.parquet (Layer-11 capture,
           2026-05-30 onward) — scope STOCK, score <= SCORE_TH.
  prices : Daily_Cash_Market market_data.duckdb daily_data (cash bhavcopy,
           all series incl. BE/BZ trade-for-trade — where frauds migrate).

Method (causal)
  • one event per (ticker, event_type, day) — repeat filings collapse;
  • announcements land all day incl. post-market → ENTRY = next trading day's
    OPEN (the first price a reader of the alert could realistically short/exit);
  • forward returns entry-open → close at +1d, +3d, +5d. Negative = short wins.
  • split by F&O membership (only F&O names are actually swing-shortable via
    futures; non-F&O = avoid/exit only — SLB is 1-month-min and illiquid).

  .venv\\Scripts\\python.exe backtest_news_short.py [--score -7]
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import datetime as _dt

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from core.constants import LIVE_DIR

DCM_DB = Path(r"D:\Python Projects\Daily_Cash_Market\data\market_data.duckdb")
SEVERE = {"Fraud allegation", "SEBI action", "Auditor resignation",
          "Credit downgrade", "Regulatory ban", "Key-exec resignation"}
SEVERE_POS = {"Buyback", "Large order win", "Acquisition", "Promoter buying",
              "Capacity expansion", "Strong earnings"}
HORIZONS = (1, 3, 5)


def load_events(score_th: int, side: str = "neg") -> pd.DataFrame:
    rows = []
    for p in sorted(glob.glob(str(LIVE_DIR / "*_news_events.parquet"))):
        try:
            df = pd.read_parquet(p)
        except Exception:
            continue
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    ev = pd.concat(rows, ignore_index=True)
    ev["ts"] = pd.to_datetime(ev["ts"], utc=True).dt.tz_convert("Asia/Kolkata")
    # RE-SCORE stored headlines with the CURRENT rule-book — classifier fixes
    # (e.g. the SAST routine-filing guard that killed the fake "Acquisition +7"
    # flood) apply retroactively, so the study always measures today's classifier,
    # not the one that happened to be deployed at capture time.
    import news_events as ne
    rescored = ev["headline"].map(ne.score_text)
    ev["event_type"] = rescored.map(lambda t: t[0])
    ev["score"] = rescored.map(lambda t: t[1])
    keep = (ev["score"] <= score_th) if side == "neg" else (ev["score"] >= score_th)
    ev = ev[(ev["scope"] == "STOCK") & keep & ev["ticker"].astype(bool)]
    ev["day"] = ev["ts"].dt.tz_localize(None).dt.normalize()   # naive midnight, matches bhavcopy dates
    # one event per (ticker, event_type, day); keep the earliest sighting
    ev = (ev.sort_values("ts")
            .drop_duplicates(subset=["ticker", "event_type", "day"], keep="first"))
    return ev.reset_index(drop=True)


def load_prices(symbols: list[str]) -> pd.DataFrame:
    import duckdb
    con = duckdb.connect(str(DCM_DB), read_only=True)
    ph = ",".join("?" * len(symbols))
    px = con.execute(
        f"SELECT trade_date, symbol, series, prev_close, open_price, close_price "
        f"FROM daily_data WHERE symbol IN ({ph}) "
        f"AND series IN ('EQ','BE','BZ') ORDER BY trade_date", symbols).df()
    fno = {r[0] for r in con.execute(
        "SELECT DISTINCT symbol FROM fno_bhavcopy "
        "WHERE instrument LIKE 'FUT%' AND symbol NOT LIKE '%NIFTY%' "
        "AND trade_date >= current_date - INTERVAL 45 DAY").fetchall()}
    con.close()
    # prefer EQ over BE/BZ when both exist on a day
    px["pri"] = px["series"].map({"EQ": 0, "BE": 1, "BZ": 2})
    px = (px.sort_values(["symbol", "trade_date", "pri"])
            .drop_duplicates(subset=["symbol", "trade_date"], keep="first"))
    px.attrs["fno"] = fno
    return px


def fwd_returns(ev: pd.DataFrame, px: pd.DataFrame) -> pd.DataFrame:
    px["trade_date"] = pd.to_datetime(px["trade_date"])
    cal = np.array(sorted(px["trade_date"].unique()))
    by_sym = {s: g.set_index("trade_date") for s, g in px.groupby("symbol")}
    out = []
    for _, e in ev.iterrows():
        g = by_sym.get(e["ticker"])
        if g is None:
            continue
        # entry = first trading day AFTER the event day (announcements land all
        # day incl. post-market; next-open is the first honest fill)
        nxt = cal[cal > e["day"]]
        if not len(nxt):
            continue
        d_entry = nxt[0]
        if d_entry not in g.index:
            continue
        entry = g.loc[d_entry, "open_price"]
        if not entry or entry != entry:
            continue
        # ENTRY GAP — how much the stock ALREADY moved between the last pre-news
        # close and your first fillable price (next open). News during market hours
        # → reference = that day's prev_close (the pre-news close); news after
        # 15:30 → reference = that day's close. This is the user's "already up 5%
        # = no purpose buying / only up 1-2% = maybe unpriced" filter, measured.
        t = e["ts"].time()
        ref_day = e["day"] if (t >= _dt.time(15, 30) or t <= _dt.time(9, 0)) else None
        if ref_day is not None and ref_day in g.index:
            ref_px = g.loc[ref_day, "close_price"]
        else:
            ref_px = g.loc[e["day"], "prev_close"] if e["day"] in g.index else (
                g.loc[d_entry, "prev_close"])
        gap = (entry / ref_px - 1.0) * 100.0 if (ref_px and ref_px == ref_px) else np.nan
        rec = {"ticker": e["ticker"], "event_type": e["event_type"],
               "score": e["score"], "day": e["day"], "entry_day": d_entry,
               "entry": entry, "gap_entry": gap,
               "fno": e["ticker"] in px.attrs["fno"],
               "series": g.loc[d_entry, "series"]}
        fwd = cal[cal >= d_entry]
        for h in HORIZONS:
            if len(fwd) > h - 1:
                d_h = fwd[h - 1]                      # h-th trading day close
                if d_h in g.index:
                    c = g.loc[d_h, "close_price"]
                    rec[f"ret{h}"] = (c / entry - 1.0) * 100.0 if c == c else np.nan
        out.append(rec)
    return pd.DataFrame(out)


def block(title: str, df: pd.DataFrame) -> None:
    print(f"\n  {title}  (n={len(df)})")
    if not len(df):
        print("    —")
        return
    for h in HORIZONS:
        col = f"ret{h}"
        r = df[col].dropna()
        if not len(r):
            continue
        t = r.mean() / (r.std(ddof=1) / np.sqrt(len(r))) if len(r) > 2 and r.std() else float("nan")
        print(f"    +{h}d  n={len(r):3d}  mean {r.mean():+6.2f}%  median {r.median():+6.2f}%  "
              f"down(win for short) {(r < 0).mean():5.1%}  t={t:+.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", type=int, default=None,
                    help="score cutoff (default -7 for neg, +6 for pos)")
    ap.add_argument("--side", choices=("neg", "pos"), default="neg")
    a = ap.parse_args()
    neg = a.side == "neg"
    th = a.score if a.score is not None else (-7 if neg else 6)
    ev = load_events(th, a.side)
    if ev.empty:
        print("no captured events"); return
    cats = SEVERE if neg else SEVERE_POS
    sev = ev[ev["event_type"].isin(cats)]
    word = "NEGATIVE" if neg else "POSITIVE"
    win = "short wins when return < 0" if neg else "long wins when return > 0"
    print("=" * 78)
    print(f"  SEVERE-{word} EVENT → FORWARD DRIFT   events {ev['day'].min():%Y-%m-%d} → "
          f"{ev['day'].max():%Y-%m-%d}")
    print(f"  stock events past cutoff {th:+d}: {len(ev)}  ·  severe categories: {len(sev)} "
          f"({ev['ticker'].nunique()} tickers)")
    print("=" * 78)
    px = load_prices(sorted(ev["ticker"].unique()))
    r = fwd_returns(sev, px)
    if r.empty:
        print("no price joins — ticker mismatch vs daily_data?"); return
    block(f"ALL severe {word.lower()}s (next-open entry, {win})", r)
    block("F&O names only (futures-tradeable)", r[r["fno"]])
    block("non-F&O", r[~r["fno"]])
    for et, g in r.groupby("event_type"):
        block(f"[{et}]", g)
    # ── ENTRY-GAP conditioning (the "already moved?" filter) ─────────────────────
    # gap_entry = pre-news close → your entry open. Thesis: a big gap = the news is
    # consumed (no purpose entering); a small gap = possibly unpriced.
    print("\n  " + "─" * 70)
    print("  ENTRY-GAP BUCKETS — how much it already moved by your first fill")
    gp = r.dropna(subset=["gap_entry"])
    if neg:
        cuts = [("gap >= -1% (barely reacted)", gp[gp.gap_entry >= -1]),
                ("-3% < gap < -1%", gp[(gp.gap_entry < -1) & (gp.gap_entry > -3)]),
                ("gap <= -3% (already smashed)", gp[gp.gap_entry <= -3])]
    else:
        cuts = [("gap <= +1% (barely reacted)", gp[gp.gap_entry <= 1]),
                ("+1% < gap < +3%", gp[(gp.gap_entry > 1) & (gp.gap_entry < 3)]),
                ("gap >= +3% (already popped)", gp[gp.gap_entry >= 3])]
    for name, g in cuts:
        block(name, g)
    print(f"\n  join rate: {len(r)}/{len(sev)} severe events priced  ·  "
          f"F&O members: {int(r['fno'].sum())}  ·  BE/BZ (T2T/circuit): "
          f"{(r['series'] != 'EQ').sum()}")
    print("  CAVEAT: ~5 weeks of capture, overlapping windows, one regime — a first read,")
    print("  NOT a validated edge. Wire nothing to auto-action off this alone.")


if __name__ == "__main__":
    main()
