"""
LEVEL-BREAK regime detection — does a CLOSE beyond a horizontal support/resistance
identify a sustained Nifty regime change (the user's chart thesis: 24637 support breaks
-> Feb-Mar 2026 downtrend; 26323 resistance caps rallies)?

Tested on clean 4yr Nifty50 daily (Tradebot parquet, multi-regime). All signals causal.
Compares against the earlier EMA/200-DMA/death-cross finding (those LAGGED). Question:
is a horizontal-level BREAKDOWN a LEADING regime trigger those missed?

Level proxy (parameter-light, causal): Donchian channel — a "support break" = daily CLOSE
below the lowest low of the trailing W days (a new W-day low = broke the range floor); a
"resistance break" = CLOSE above the trailing W-day high. W in {20,40,60} ~ 1/2/3-month
structure. Also a MARGIN-confirmed variant (close must clear by >0.5*ATR to cut whipsaw).

Outputs: forward 5/10/20/40d return after each break, continuation hit-rate, whipsaw
(break then reverse within 5d), regime-by-year, and the exact Feb-Mar 2026 episode trace.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, r"d:/Python Projects/Tradebot")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import numpy as np, pandas as pd

P = r"d:/Python Projects/Tradebot/data/historical/daily/NSE_NIFTY50_INDEX_daily.parquet"

def load():
    d = pd.read_parquet(P)
    d["date"] = pd.to_datetime(d["ts"]).dt.normalize()
    return d.set_index("date")[["open","high","low","close"]].sort_index()

def fret(c, n):
    lr = np.log(c); return (np.exp(lr.shift(-n)-lr)-1)*100

if __name__ == "__main__":
    d = load()
    c, h, l = d["close"], d["high"], d["low"]
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    for n in (5,10,20,40):
        d[f"f{n}"] = fret(c, n)

    print(f"Nifty50 {d.index.min().date()} -> {d.index.max().date()}  ({len(d)} bars)\n")

    print("="*98)
    print("A) SUPPORT BREAK (close < trailing W-day low) → forward return. Thesis: continues DOWN.")
    print("   RESISTANCE BREAK (close > trailing W-day high) → forward. Thesis: continues UP.")
    print("="*98)
    print(f"  {'signal':26s} {'%days':>6s} {'f5':>7s} {'f10':>7s} {'f20':>7s} {'f40':>7s} {'P(f20<0)':>9s} {'n':>5s}")
    rows = {}
    for W in (20,40,60):
        prior_lo = l.shift(1).rolling(W).min()
        prior_hi = h.shift(1).rolling(W).max()
        sup_break = c < prior_lo                     # new W-day low on close
        res_break = c > prior_hi                     # new W-day high on close
        rows[("sup",W)] = sup_break; rows[("res",W)] = res_break
        for name, sig, thesis in [(f"support break {W}d", sup_break, "down"),
                                   (f"resistance break {W}d", res_break, "up")]:
            s = d[sig].dropna(subset=["f20"]) if False else d.loc[sig.fillna(False)]
            s = s.dropna(subset=["f20"])
            if len(s) < 10:
                print(f"  {name:26s}  thin"); continue
            pneg = (s["f20"]<0).mean()*100
            print(f"  {name:26s} {sig.mean()*100:5.1f}% {s['f5'].mean():+6.2f}% {s['f10'].mean():+6.2f}% "
                  f"{s['f20'].mean():+6.2f}% {s['f40'].mean():+6.2f}% {pneg:8.0f}% {len(s):5d}")
        print("  " + "-"*94)

    print("\n" + "="*98)
    print("B) MARGIN-confirmed 40d break (close clears level by >0.5*ATR) — cuts whipsaw")
    print("="*98)
    plo = l.shift(1).rolling(40).min(); phi = h.shift(1).rolling(40).max()
    sup_m = c < plo - 0.5*atr; res_m = c > phi + 0.5*atr
    for name, sig in [("support break 40d +margin", sup_m), ("resistance break 40d +margin", res_m)]:
        s = d.loc[sig.fillna(False)].dropna(subset=["f20"])
        if len(s) < 10: print(f"  {name:30s} thin"); continue
        print(f"  {name:30s} f10 {s['f10'].mean():+.2f}%  f20 {s['f20'].mean():+.2f}%  f40 {s['f40'].mean():+.2f}%  "
              f"P(f20<0) {(s['f20']<0).mean()*100:.0f}%  n {len(s)}")

    print("\n" + "="*98)
    print("C) FIRST break only (new signal after >=10 quiet days) — the REGIME-TURN trigger,")
    print("   not every day inside an existing break. This is what 'regime change' means.")
    print("="*98)
    for lab, sig in [("support 40d", rows[("sup",40)]), ("resistance 40d", rows[("res",40)])]:
        s = sig.fillna(False)
        first = s & ~s.shift(1).fillna(False).rolling(10).max().astype(bool)  # first ON after quiet
        ss = d.loc[first].dropna(subset=["f20"])
        if len(ss) < 5: print(f"  {lab:16s} thin ({len(ss)})"); continue
        print(f"  first {lab:16s} f10 {ss['f10'].mean():+.2f}%  f20 {ss['f20'].mean():+.2f}%  "
              f"f40 {ss['f40'].mean():+.2f}%  P(f20<0) {(ss['f20']<0).mean()*100:.0f}%  n {len(ss)}  "
              f"dates {[x.date().isoformat() for x in ss.index]}")

    print("\n" + "="*98)
    print("D) WHIPSAW — of support breaks, how many reverse back ABOVE the level within 5d?")
    print("="*98)
    for W in (20,40,60):
        sb = rows[("sup",W)].fillna(False); plo = l.shift(1).rolling(W).min()
        # reversed if close back above the broken level within 5 bars
        back = pd.Series(False, index=d.index)
        for i in np.where(sb.values)[0]:
            lvl = plo.iloc[i]
            if i+5 < len(d) and (c.iloc[i+1:i+6] > lvl).any(): back.iloc[i] = True
        n = int(sb.sum()); fp = back[sb].mean()*100 if n else np.nan
        print(f"  support break {W}d: {n} signals, {fp:.0f}% snapped back above within 5d (whipsaw)")

    print("\n" + "="*98)
    print("E) THE Feb-Mar 2026 EPISODE (user's 24637 break) — did a level break flag it early?")
    print("="*98)
    seg = d.loc["2026-01-15":"2026-04-10"]
    plo40 = l.shift(1).rolling(40).min()
    for dt, row in seg.iterrows():
        sb = row["close"] < plo40.loc[dt]
        if sb or dt.day % 7 == 0:
            tag = "  <-- SUPPORT BREAK" if sb else ""
            f20 = row["f20"]
            print(f"  {dt.date()}  close {row['close']:8.0f}  40d-low {plo40.loc[dt]:8.0f}  "
                  f"fwd20 {f20:+5.1f}%{tag}")
