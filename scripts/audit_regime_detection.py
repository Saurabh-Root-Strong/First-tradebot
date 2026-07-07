"""
REGIME DETECTION — how to identify a SUSTAINED (1-2 month) Nifty downtrend, and catch
the regime TURNING early. World-best-desk method: don't trust one index line — combine
index STRUCTURE (200-DMA, death cross) with market BREADTH (how many of the 50 members
are individually broken) and a DURATION gate (kill whipsaw).

Panel: Tradebot 211-name F&O daily 2022-06 -> 2026-06 (multi-regime) + NIFTY50 index.
All signals causal (own past only); forward index returns are separate outcome cols.

Signals compared (each a candidate "downtrend ON" rule):
  R1 short_stack   px < EMA20 < EMA50           (current engine — fast/short)
  R2 below_200dma  px < 200-DMA                  (classic bull/bear line)
  R3 death_cross   EMA50 < EMA200                (slow, confirmed structural bear)
  R4 breadth_bear  <40% of members above own 50-DMA
  R5 breadth_div   index 20d UP but breadth (%>50DMA) falling 10d  (EARLY WARNING / topping)
  R6 combo_confirm below_200dma AND breadth_bear, held >= 5 days   (sustained, low-whipsaw)

Ground truth for "sustained 1-2mo downtrend" = forward-40d index return (fwd40).
Also lead-time: how many days each early rule fires BEFORE the death cross of each episode.
"""
from __future__ import annotations
import sys, glob, os
sys.path.insert(0, r"d:/Python Projects/Tradebot")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import numpy as np, pandas as pd

DAILY = r"d:/Python Projects/Tradebot/data/historical/daily"

def load_closes() -> pd.DataFrame:
    rows = []
    for f in glob.glob(os.path.join(DAILY, "*_EQ_daily.parquet")):
        sym = os.path.basename(f).replace("NSE_", "").replace("_EQ_daily.parquet", "")
        d = pd.read_parquet(f)[["ts", "close"]].copy(); d["symbol"] = sym
        rows.append(d)
    df = pd.concat(rows, ignore_index=True)
    df["date"] = pd.to_datetime(df["ts"]).dt.normalize()
    return df.pivot_table("close", "date", "symbol").sort_index()

def load_index() -> pd.Series:
    d = pd.read_parquet(os.path.join(DAILY, "NSE_NIFTY50_INDEX_daily.parquet"))
    d["date"] = pd.to_datetime(d["ts"]).dt.normalize()
    return d.set_index("date")["close"].sort_index()

def fret(s, n):
    lr = np.log(s); return (np.exp(lr.shift(-n) - lr) - 1) * 100

if __name__ == "__main__":
    px = load_closes()                     # date x symbol
    idx = load_index()
    common = idx.index.intersection(px.index)
    idx = idx.reindex(common); px = px.reindex(common)

    # ── breadth (causal): % of members above own 50-DMA / 200-DMA ────────────────
    sma50  = px.rolling(50).mean()
    sma200 = px.rolling(200).mean()
    # count a member only on days it has a valid MA (>=50 / >=200 of its own bars); causal
    breadth50  = (px > sma50).sum(axis=1)  / sma50.notna().sum(axis=1).replace(0, np.nan)  * 100
    breadth200 = (px > sma200).sum(axis=1) / sma200.notna().sum(axis=1).replace(0, np.nan) * 100

    # ── index structure (causal) ─────────────────────────────────────────────────
    d = pd.DataFrame(index=common)
    d["idx"] = idx
    d["ema20"] = idx.ewm(span=20, adjust=False).mean()
    d["ema50"] = idx.ewm(span=50, adjust=False).mean()
    d["ema200"] = idx.ewm(span=200, adjust=False).mean()
    d["sma200"] = idx.rolling(200).mean()
    d["r20"] = (np.exp(np.log(idx) - np.log(idx).shift(20)) - 1) * 100
    d["b50"] = breadth50
    d["b200"] = breadth200
    d["b50_chg10"] = breadth50 - breadth50.shift(10)
    for h in (20, 40, 60):
        d[f"fwd{h}"] = fret(idx, h)

    # ── candidate rules (causal booleans) ────────────────────────────────────────
    R = pd.DataFrame(index=common)
    R["R1_short_stack"]  = (d["idx"] < d["ema20"]) & (d["ema20"] < d["ema50"])
    R["R2_below_200dma"] = d["idx"] < d["sma200"]
    R["R3_death_cross"]  = d["ema50"] < d["ema200"]
    R["R4_breadth_bear"] = d["b50"] < 40
    R["R5_breadth_div"]  = (d["r20"] > 0) & (d["b50_chg10"] < 0) & (d["b50"] < 55)
    confirm = (d["idx"] < d["sma200"]) & (d["b50"] < 40)
    R["R6_combo_5d"] = confirm & confirm.shift(1).fillna(False) & confirm.shift(2).fillna(False) \
                       & confirm.shift(3).fillna(False) & confirm.shift(4).fillna(False)

    valid = d["sma200"].notna() & d["fwd40"].notna()
    print(f"panel {common.min().date()} -> {common.max().date()} | eval days {int(valid.sum())} "
          f"(200-DMA available) | members {px.shape[1]}\n")

    print("="*100)
    print("A) EACH RULE vs forward index return — does 'ON' actually mean a sustained downtrend?")
    print("   fwd40 = next-40d (≈2mo) Nifty %. Good downtrend rule: ON fwd40<<0 and P(down) high.")
    print("="*100)
    print(f"  {'rule':18s} {'%days ON':>8s} {'fwd20 ON':>9s} {'fwd40 ON':>9s} {'fwd60 ON':>9s} "
          f"{'fwd40 OFF':>9s} {'P(fwd40<0)':>10s}")
    base40 = d.loc[valid, "fwd40"].mean()
    for c in R.columns:
        m = R[c] & valid
        off = (~R[c]) & valid
        if m.sum() < 20:
            print(f"  {c:18s}  thin ({int(m.sum())})"); continue
        pdown = (d.loc[m, "fwd40"] < 0).mean() * 100
        print(f"  {c:18s} {m.sum()/valid.sum()*100:7.1f}% {d.loc[m,'fwd20'].mean():+8.2f}% "
              f"{d.loc[m,'fwd40'].mean():+8.2f}% {d.loc[m,'fwd60'].mean():+8.2f}% "
              f"{d.loc[off,'fwd40'].mean():+8.2f}% {pdown:9.0f}%")
    print(f"  {'(baseline all)':18s} {'100.0%':>8s} {'':9s} {base40:+8.2f}%")

    print("\n" + "="*100)
    print("B) LEAD/LAG — when a confirmed bear (R3 death cross) episode starts, how many trading")
    print("   days EARLIER did each early rule fire? (+ = earlier warning, - = later/lagging)")
    print("="*100)
    dc = R["R3_death_cross"].astype(int)
    starts = list(np.where((dc.diff() == 1).values)[0])          # death-cross onsets
    print(f"  death-cross episodes: {len(starts)}  at "
          f"{[common[i].date().isoformat() for i in starts]}")
    for c in ["R1_short_stack", "R2_below_200dma", "R4_breadth_bear", "R5_breadth_div", "R6_combo_5d"]:
        arr = R[c].values; leads = []
        for s in starts:
            lo = max(0, s - 60)                                  # look back up to 60d
            fired = np.where(arr[lo:s+1])[0]
            if len(fired): leads.append(s - (lo + fired[0]))     # days before onset
        if leads:
            print(f"  {c:18s} fired before {len(leads)}/{len(starts)} episodes | "
                  f"median lead {int(np.median(leads))}d  (range {min(leads)}..{max(leads)})")
        else:
            print(f"  {c:18s} never fired before an episode")

    print("\n" + "="*100)
    print("C) FALSE-ALARM check — of the days each rule is ON, how many were NOT followed by a")
    print("   real decline (fwd40 >= 0)? Lower FP = more reliable. (early rules trade FP for lead)")
    print("="*100)
    print(f"  {'rule':18s} {'ON days':>8s} {'false-alarm %':>13s}  {'note'}")
    notes = {"R1_short_stack":"fast, whippy","R2_below_200dma":"classic line","R3_death_cross":"confirmed/slow",
             "R4_breadth_bear":"breadth","R5_breadth_div":"early warning","R6_combo_5d":"sustained combo"}
    for c in R.columns:
        m = R[c] & valid
        if m.sum() < 20: continue
        fp = (d.loc[m, "fwd40"] >= 0).mean() * 100
        print(f"  {c:18s} {int(m.sum()):8d} {fp:12.0f}%  {notes.get(c,'')}")

    print("\n" + "="*100)
    print("D) YEAR-BY-YEAR robustness of the two anchors (R6 sustained combo, R5 early warning)")
    print("="*100)
    yr = pd.Series(common.year, index=common)
    for c in ["R5_breadth_div", "R6_combo_5d"]:
        print(f"  {c}:")
        for y in sorted(set(common.year)):
            m = R[c] & valid & (yr == y)
            if m.sum() < 8:
                print(f"    {y}: thin"); continue
            print(f"    {y}: ON {int(m.sum()):3d}d  fwd40 {d.loc[m,'fwd40'].mean():+6.2f}%  "
                  f"P(down) {(d.loc[m,'fwd40']<0).mean()*100:3.0f}%")
