"""
REGIME-CONDITIONAL stress test of the DCM 1-2wk Forward Tilt — on the 4yr F&O panel.

WHY: the shipped tilt (src/analytics/sector_forward_tilt.py) is validated on ONE
broadly-bull DCM window (2024-12 -> 2026-07). It is long-only. Its regime read is
display-only (confidence_mult ~1.0 everywhere; only REVERSAL gets 0.80). The open
question the user raised: "beautiful in an uptrend — how does it behave in a
downtrend / crash, and is there a smart-money-accumulation long (2-3mo) angle?"

This panel (Tradebot data/historical/daily, 211 F&O names, 2022-05 -> 2026-06) DOES
contain downtrends / high-vol / the 2022 bear, so it can stress the tilt out-of-regime.

The factor is reconstructed faithfully at STOCK level (the sector tilt is the same
cross-sectional relative-momentum factor, aggregated): composite = 0.70*rank(rs_2w)
+ 0.30*rank(rs_1w), rs = stock momentum - Nifty momentum. Delivery/accumulation
breadth is DCM-only (no delivery in this parquet) so the "smart money in crash"
thesis is tested with a PRICE PROXY (relative strength surviving the fall), stated
honestly as a proxy.

Regime label = faithful replica of engine _market_regime on Nifty50 close, PLUS a
medium-term (1-2 month) trend axis for the divergence / bull-trap test.

Outputs an actionable POSTURE matrix: per regime, OW absolute vs relative forward
return (10d / 40d / 60d), the momentum-crash check, the divergence states, and the
resilient-names recovery test.
"""
from __future__ import annotations
import sys, glob, os
sys.path.insert(0, r"d:/Python Projects/Tradebot")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import numpy as np, pandas as pd

DAILY = r"d:/Python Projects/Tradebot/data/historical/daily"
SMAP  = r"C:/Users/HP/AppData/Local/Temp/claude/d--Python-Projects-Tradebot/00fb1329-476b-4454-94e7-cefc7a5aa9d8/scratchpad/sector_map.csv"

# ── load ─────────────────────────────────────────────────────────────────────
def load_panel() -> pd.DataFrame:
    rows = []
    for f in glob.glob(os.path.join(DAILY, "*_EQ_daily.parquet")):
        sym = os.path.basename(f).replace("NSE_", "").replace("_EQ_daily.parquet", "")
        d = pd.read_parquet(f)[["ts", "close"]].copy()
        d["symbol"] = sym
        rows.append(d)
    df = pd.concat(rows, ignore_index=True)
    df["date"] = pd.to_datetime(df["ts"]).dt.normalize()
    df = df.sort_values(["symbol", "date"]).drop_duplicates(["symbol", "date"])
    return df[["symbol", "date", "close"]]

def load_nifty() -> pd.DataFrame:
    d = pd.read_parquet(os.path.join(DAILY, "NSE_NIFTY50_INDEX_daily.parquet"))
    d["date"] = pd.to_datetime(d["ts"]).dt.normalize()
    d = d.sort_values("date")[["date", "close"]].rename(columns={"close": "nclose"})
    d["nret"] = d["nclose"].pct_change() * 100
    return d

def cret(s: pd.Series, n: int) -> pd.Series:
    """trailing n-bar % return (compounded via log)."""
    lr = np.log(s)
    return (np.exp(lr - lr.shift(n)) - 1) * 100

def fret(s: pd.Series, n: int) -> pd.Series:
    """forward n-bar % return."""
    lr = np.log(s)
    return (np.exp(lr.shift(-n) - lr) - 1) * 100

# ── regime label (faithful engine replica + medium-term axis) ────────────────
def label_regimes(nf: pd.DataFrame) -> pd.DataFrame:
    nf = nf.copy().reset_index(drop=True)
    c = nf["nclose"]; r = nf["nret"] / 100.0
    ema20 = c.ewm(span=20, adjust=False).mean()
    ema50 = c.ewm(span=50, adjust=False).mean()
    vol20 = r.rolling(20).std()
    volpct = pd.Series([(vol20.iloc[:i+1] <= vol20.iloc[i]).mean() if i > 20 else np.nan
                        for i in range(len(vol20))])
    ret5  = cret(c, 5); ret20 = cret(c, 20)
    ret40 = cret(c, 40); ret60 = cret(c, 60)          # 1-2 month trend axis
    ema20s = ema20.diff(10)                            # medium EMA slope
    lab = []
    for i in range(len(nf)):
        if i < 60 or not np.isfinite(vol20.iloc[i]):
            lab.append("UNKNOWN"); continue
        px = c.iloc[i]
        if ret5.iloc[i] <= -3 and ret20.iloc[i] > 0:            lab.append("REVERSAL")
        elif px < ema20.iloc[i] < ema50.iloc[i]:                lab.append("TRENDING_DOWN")
        elif np.isfinite(volpct.iloc[i]) and volpct.iloc[i]>=0.80: lab.append("HIGH_VOL")
        elif px > ema20.iloc[i] > ema50.iloc[i]:                lab.append("TRENDING_UP")
        else:                                                   lab.append("CHOPPY")
    nf["regime"] = lab
    # medium-term (1-2mo) direction + divergence states
    nf["med_up"] = (ret40 > 0) & (ema20s > 0)
    nf["med_dn"] = (ret40 < 0) & (ema20s < 0)
    nf["short_up"] = ret5 > 0
    def dv(i):
        if not np.isfinite(ret40.iloc[i]): return "n/a"
        su, mu, md = nf["short_up"].iloc[i], nf["med_up"].iloc[i], nf["med_dn"].iloc[i]
        if su and md:  return "BULLTRAP"     # 1-2wk up but 1-2mo down
        if (not su) and mu: return "DIP_IN_UP"  # 1-2wk down but 1-2mo up (buyable dip)
        if su and mu:  return "ALIGNED_UP"
        if (not su) and md: return "ALIGNED_DN"
        return "MIXED"
    nf["diverg"] = [dv(i) for i in range(len(nf))]
    nf["ret40"] = ret40; nf["ret60"] = ret60
    return nf

# ── build cross-sectional factor + forwards ──────────────────────────────────
def build(df: pd.DataFrame, nf: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    g = df.groupby("symbol")["close"]
    df["m10"] = g.transform(lambda s: cret(s, 10))
    df["m5"]  = g.transform(lambda s: cret(s, 5))
    for h in (10, 40, 60):
        df[f"f{h}"] = g.transform(lambda s: fret(s, h))
    nfx = nf[["date", "nclose", "regime", "diverg", "ret40", "ret60"]].copy()
    nfx["nm10"] = cret(nfx["nclose"], 10); nfx["nm5"] = cret(nfx["nclose"], 5)
    for h in (10, 40, 60):
        nfx[f"nf{h}"] = fret(nfx["nclose"], h)
    df = df.merge(nfx, on="date", how="inner")
    df["rs2w"] = df["m10"] - df["nm10"]
    df["rs1w"] = df["m5"]  - df["nm5"]
    # relative forwards (vs Nifty = true rotation P&L for a long-only book's alpha)
    for h in (10, 40, 60):
        df[f"rel{h}"] = df[f"f{h}"] - df[f"nf{h}"]
    # cross-sectional composite rank per day (faithful momentum-led factor)
    d = df.dropna(subset=["rs2w", "rs1w"]).copy()
    gg = d.groupby("date")
    d["comp"] = (0.70 * gg["rs2w"].transform(lambda x: x.rank(pct=True))
               + 0.30 * gg["rs1w"].transform(lambda x: x.rank(pct=True)))
    d["crank"] = d.groupby("date")["comp"].rank(pct=True)
    for h in (10, 40, 60):
        d[f"med{h}"] = d.groupby("date")[f"f{h}"].transform("median")
    return d

def tstat_nonoverlap(s: pd.Series, dates: pd.Series, h: int) -> tuple:
    """mean + non-overlapping t: one obs per h days (date-collapsed daily means)."""
    daily = pd.DataFrame({"date": dates, "v": s}).groupby("date")["v"].mean().dropna()
    if len(daily) < h + 5: return (daily.mean(), np.nan, len(daily))
    sub = daily.iloc[::h]
    t = sub.mean() / (sub.std() / np.sqrt(len(sub))) if len(sub) >= 4 else np.nan
    return (daily.mean(), t, len(sub))

# ── run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("loading 4yr F&O panel ...")
    df = build(load_panel(), label_regimes(load_nifty()))
    smap = pd.read_csv(SMAP).set_index("symbol")["sector"]
    df["sector"] = df["symbol"].map(smap)
    print(f"panel: {df['symbol'].nunique()} names x {df['date'].nunique()} days "
          f"| {df['date'].min().date()} -> {df['date'].max().date()}  ({len(df):,} obs)")

    print("\n" + "="*100)
    print("0) REGIME BASE RATES (share of trading days) — is each regime sampled enough?")
    print("="*100)
    perday = df.drop_duplicates("date")[["date", "regime", "diverg"]]
    print("  regime :", (perday["regime"].value_counts()/len(perday)*100).round(1).to_dict())
    print("  diverg :", (perday["diverg"].value_counts()/len(perday)*100).round(1).to_dict())

    OW = df["crank"] >= 0.75; UW = df["crank"] <= 0.25

    print("\n" + "="*100)
    print("1) OW long-only P&L BY REGIME — ABSOLUTE fwd-10d (market beta) vs RELATIVE (alpha)")
    print("   The tilt is long-only. ABSOLUTE<0 => you lose money even if the RELATIVE call is right.")
    print("="*100)
    print(f"  {'regime':15s} {'OW abs%':>9s} {'OW rel%':>9s} {'OW-UW rel':>10s} {'t(OW-UW)':>9s} {'days':>6s}")
    for reg in ["TRENDING_UP","CHOPPY","HIGH_VOL","TRENDING_DOWN","REVERSAL"]:
        s = df[(df["regime"]==reg)].dropna(subset=["f10"])
        if s.empty: continue
        ow = s[s["crank"]>=0.75]; uw = s[s["crank"]<=0.25]
        abs_ow = ow["f10"].mean(); rel_ow = ow["rel10"].mean()
        spread = ow["rel10"].mean() - uw["rel10"].mean()
        # non-overlap t on OW-UW daily spread
        sp = (s[s["crank"]>=0.75].groupby("date")["rel10"].mean()
              - s[s["crank"]<=0.25].groupby("date")["rel10"].mean()).dropna()
        t = sp.iloc[::10].mean()/(sp.iloc[::10].std()/np.sqrt(len(sp.iloc[::10]))) if len(sp)>=40 else np.nan
        print(f"  {reg:15s} {abs_ow:+8.2f}% {rel_ow:+8.2f}% {spread:+9.2f}% {t:+8.1f} {s['date'].nunique():6d}")

    print("\n" + "="*100)
    print("2) MOMENTUM-CRASH check — in REVERSAL/HIGH_VOL does high momentum (OW) UNDERPERFORM?")
    print("   Quintile relative fwd-10d. If Q5<Q1 => momentum inverts (avoid the tilt).")
    print("="*100)
    for reg in ["TRENDING_UP","CHOPPY","HIGH_VOL","TRENDING_DOWN","REVERSAL"]:
        s = df[df["regime"]==reg].dropna(subset=["rel10"]).copy()
        if len(s) < 200:
            print(f"  {reg:15s} thin (n={len(s)})"); continue
        s["Q"] = pd.qcut(s["crank"], 5, labels=[1,2,3,4,5], duplicates="drop")
        m = s.groupby("Q", observed=True)["rel10"].mean()
        print(f"  {reg:15s} Q1{m.get(1,np.nan):+.2f} Q2{m.get(2,np.nan):+.2f} Q3{m.get(3,np.nan):+.2f} "
              f"Q4{m.get(4,np.nan):+.2f} Q5{m.get(5,np.nan):+.2f} | Q5-Q1 {m.get(5,0)-m.get(1,0):+.2f}%")

    print("\n" + "="*100)
    print("3) DIVERGENCE / BULL-TRAP — 1-2wk vs 1-2mo Nifty trend. Forward Nifty AND OW abs.")
    print("   BULLTRAP = short-up but medium-down (the 'looks bullish 1-2wk, bearish 1-2mo' state).")
    print("="*100)
    print(f"  {'state':11s} {'nifty f10':>10s} {'nifty f40':>10s} {'OW abs f10':>11s} {'OW abs f40':>11s} {'days':>6s}")
    for st in ["ALIGNED_UP","DIP_IN_UP","BULLTRAP","ALIGNED_DN","MIXED"]:
        s = df[df["diverg"]==st]
        if s["date"].nunique() < 8:
            print(f"  {st:11s} thin"); continue
        nf10 = s.drop_duplicates("date")["nf10"].mean(); nf40 = s.drop_duplicates("date")["nf40"].mean()
        ow = s[s["crank"]>=0.75]
        print(f"  {st:11s} {nf10:+9.2f}% {nf40:+9.2f}% {ow['f10'].mean():+10.2f}% {ow['f40'].mean():+10.2f}% "
              f"{s['date'].nunique():6d}")

    print("\n" + "="*100)
    print("4) SMART-MONEY-IN-CRASH (price proxy) — in DOWN regimes, do RESILIENT names")
    print("   (top relative-strength quartile) lead the 2-3mo recovery? abs+rel fwd-40 & 60.")
    print("="*100)
    down = df[df["regime"].isin(["TRENDING_DOWN","REVERSAL","HIGH_VOL"])]
    print(f"  {'bucket':22s} {'abs f40':>9s} {'rel f40':>9s} {'abs f60':>9s} {'rel f60':>9s} {'n':>7s}")
    for name, mask in [("resilient (crank>=0.75)", down["crank"]>=0.75),
                       ("weak (crank<=0.25)",     down["crank"]<=0.25),
                       ("all down-regime",         down["crank"].notna())]:
        s = down[mask].dropna(subset=["f40"])
        print(f"  {name:22s} {s['f40'].mean():+8.2f}% {s['rel40'].mean():+8.2f}% "
              f"{s['f60'].mean():+8.2f}% {s['rel60'].mean():+8.2f}% {len(s):7d}")

    print("\n" + "="*100)
    print("5) SECTOR view in DOWN regimes — which sectors' resilient names recover best (rel f40)")
    print("="*100)
    ds = down[(down["crank"]>=0.60)].dropna(subset=["rel40","sector"]).copy()
    tab = ds.groupby("sector").agg(n=("rel40","size"), rel40=("rel40","mean"),
                                    rel60=("rel60","mean"))
    tab = tab[tab["n"]>=40].sort_values("rel40", ascending=False)
    print(tab.round(2).to_string())
