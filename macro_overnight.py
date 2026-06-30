"""
macro_overnight.py — NEXT-DAY macro briefing for the Indian market + accuracy backtest.

Causal, lookahead-free by construction: global markets trade OVERNIGHT while India is
closed (US session, UST yield, DXY, Brent all settle ~02:00 IST, before India's 09:15
open). So the global move on day d-1 is KNOWN before India opens day d and legitimately
predicts India's day-d gap / direction. This is the documented "global cues" link.

Signal (overnight, lagged 1 day vs India):
  risk_on_for_india = +S&P500  - DXY  - UST10Y-yield  - Brent
  (US equities up = risk-on; a stronger dollar / rising US yields / dearer oil all hurt
   an EM oil-importer like India). Net sign -> expected India direction next day.

Backtest reports, on Indian data, HOW OFTEN it works (hit %): per factor, the composite,
on the GAP (open vs prev close) and the full day (close vs prev close), by year. Honest
split: the GAP is highly predictable (and largely priced by GIFT Nifty) = PREP/context;
the intraday-after-gap is the hard part. Then prints the LATEST reading = what the last
global session implies for India's next session.

    .venv\\Scripts\\python.exe macro_overnight.py
"""
from __future__ import annotations

import sys
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import yfinance as yf

START = "2021-06-01"
# (ticker, sign for India risk-on)  — yield/dollar/oil up = risk-off for India
FACTORS = {"S&P500": ("^GSPC", +1), "UST10Y": ("^TNX", -1),
           "DXY": ("DX-Y.NYB", -1), "Brent": ("BZ=F", -1)}
NIFTY = "^NSEI"


def _close(tkr):
    d = yf.download(tkr, start=START, progress=False, auto_adjust=True)
    if d is None or len(d) == 0:
        return None
    s = d["Close"]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    return s.rename(tkr)


def build():
    nif = _close(NIFTY)
    nopen = yf.download(NIFTY, start=START, progress=False, auto_adjust=True)["Open"]
    if isinstance(nopen, pd.DataFrame):
        nopen = nopen.iloc[:, 0]
    nopen.index = pd.to_datetime(nopen.index).tz_localize(None).normalize()
    df = pd.DataFrame({"nif_close": nif, "nif_open": nopen}).dropna()
    df["gap"] = df["nif_open"] / df["nif_close"].shift(1) - 1
    df["dayret"] = df["nif_close"] / df["nif_close"].shift(1) - 1
    df["intra"] = df["nif_close"] / df["nif_open"] - 1
    # overnight global signal: global return on the PRIOR session (lag 1 vs India date)
    score = pd.Series(0.0, index=df.index)
    for name, (tkr, sgn) in FACTORS.items():
        s = _close(tkr)
        ret = (s / s.shift(1) - 1).reindex(df.index).ffill(limit=1)
        df[f"g_{name}"] = ret.shift(1) * sgn        # shift(1): yesterday's global move
        score = score.add(np.sign(df[f"g_{name}"]).fillna(0), fill_value=0)
    df["score"] = score
    return df.dropna(subset=["gap", "dayret"])


def hit(pred_sign, actual):
    m = (pred_sign != 0) & actual.notna()
    if m.sum() == 0:
        return float("nan"), 0
    return 100 * (np.sign(actual[m]) == pred_sign[m]).mean(), int(m.sum())


def main():
    print("MACRO OVERNIGHT -> INDIA NEXT-DAY  (accuracy on Indian data)")
    print("=" * 76)
    df = build()
    print(f"  {len(df)} India sessions  {df.index.min():%Y-%m-%d} .. {df.index.max():%Y-%m-%d}")
    base_up = 100 * (df["dayret"] > 0).mean()
    print(f"  baseline: India up-day rate {base_up:.0f}%  (beat this to add info)\n")

    print("  PER-FACTOR hit% (sign of overnight move -> India next day)")
    print(f"     {'factor':8s} {'gap':>10} {'dayret':>12}")
    for name in FACTORS:
        ps = np.sign(df[f"g_{name}"]).fillna(0)
        hg, ng = hit(ps, df["gap"]); hd, nd = hit(ps, df["dayret"])
        print(f"     {name:8s} {hg:6.0f}% (n{ng}) {hd:6.0f}% (n{nd})")

    print("\n  COMPOSITE score (sign of summed votes)")
    ps = np.sign(df["score"])
    for tgt in ("gap", "dayret", "intra"):
        h, n = hit(ps, df[tgt])
        print(f"     {tgt:8s} hit {h:.0f}%  (n={n})")

    print("\n  COMPOSITE on GAP & DAYRET by year")
    df["yr"] = df.index.year
    for y, s in df.groupby("yr"):
        psy = np.sign(s["score"])
        hg, ng = hit(psy, s["gap"]); hd, nd = hit(psy, s["dayret"])
        # strong-signal subset (|score|>=3 = most factors agree)
        strong = s[s["score"].abs() >= 3]
        hs, ns = hit(np.sign(strong["score"]), strong["dayret"]) if len(strong) else (float("nan"), 0)
        print(f"     {y}  gap {hg:3.0f}% | dayret {hd:3.0f}% (n{nd}) | "
              f"strong(|s|>=3) dayret {hs:3.0f}% (n{ns})")

    print("\n  STRONG-AGREEMENT subset (|score|>=3, most factors aligned)")
    strong = df[df["score"].abs() >= 3]
    for tgt in ("gap", "dayret", "intra"):
        h, n = hit(np.sign(strong["score"]), strong[tgt])
        print(f"     {tgt:8s} hit {h:.0f}%  (n={n}, {100*len(strong)/len(df):.0f}% of days)")

    # ── tomorrow's reading ────────────────────────────────────────────────────
    last = df.iloc[-1]
    print("\n" + "=" * 76)
    print(f"  LATEST READING (globals dated up to {df.index.max():%Y-%m-%d} -> next India session)")
    votes = []
    for name in FACTORS:
        v = last[f"g_{name}"]
        votes.append(f"{name} {'+' if v>0 else '-' if v<0 else '0'}")
    sc = last["score"]
    lean = "RISK-ON / lean UP" if sc > 0 else "RISK-OFF / lean DOWN" if sc < 0 else "MIXED / flat"
    print(f"     votes: {'  '.join(votes)}   score={sc:+.0f}  ->  {lean}")
    print("     NOTE: gap is largely priced by GIFT Nifty (PREP/context, not free alpha);")
    print("     the intraday-after-gap is the unpredictable part. Use for regime/risk, not a trade.")


if __name__ == "__main__":
    main()
