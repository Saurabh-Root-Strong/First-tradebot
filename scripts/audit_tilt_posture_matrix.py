"""
POSTURE MATRIX — the canonical scenario -> action table for the DCM forward tilt.

Consolidates the 4yr OOS regime stress test into ONE decision surface: for every
(market regime x short-vs-medium divergence) cell, measure the forward outcome and
derive a single VERDICT (ACT / SELECTIVE / STAND-ASIDE), a SIZE hint (0..1), and the
one-line action the UI should tell the user. The displayed posture becomes literally
the backtested cell — no hand-waving.

Metrics per cell (fwd-10d, 211-name F&O panel 2022-26, non-overlap t):
  ow_abs   = OVERWEIGHT basket absolute return   (long-only P&L incl. market beta)
  ow_uw    = OW - UW relative spread             (the TILT's alpha; the sign that matters)
  t_ow_uw  = non-overlapping t of ow_uw
  days     = sample size

Verdict rule (derived below, then frozen into the engine):
  STAND-ASIDE  if ow_uw <= -0.25% (tilt inverts — leaders are the wrong side)
  ACT          if ow_uw >= +0.30% and ow_abs > 0 (alpha present AND long-only viable)
  SELECTIVE    otherwise (edge ~0 or positive-but-fragile: half size, top names only)
Size hint:  ACT 1.0 | SELECTIVE 0.5 | STAND-ASIDE 0.0 ; BULLTRAP overlay caps at 0.5.
"""
from __future__ import annotations
import sys
sys.path.insert(0, r"d:/Python Projects/Tradebot")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import numpy as np, pandas as pd
from scripts.audit_forward_tilt_regimes import load_panel, load_nifty, label_regimes, build

def cell_stats(s: pd.DataFrame, h: int = 10) -> dict:
    ow = s[s["crank"] >= 0.75]; uw = s[s["crank"] <= 0.25]
    if len(ow) < 40 or len(uw) < 40:
        return dict(days=s["date"].nunique(), ow_abs=np.nan, ow_uw=np.nan, t=np.nan, thin=True)
    sp = (ow.groupby("date")[f"rel{h}"].mean() - uw.groupby("date")[f"rel{h}"].mean()).dropna()
    t = (sp.iloc[::h].mean() / (sp.iloc[::h].std() / np.sqrt(len(sp.iloc[::h])))
         if len(sp) >= h * 4 else np.nan)
    return dict(days=s["date"].nunique(), ow_abs=ow[f"f{h}"].mean(),
                ow_uw=ow[f"rel{h}"].mean() - uw[f"rel{h}"].mean(), t=t, thin=False)

def verdict(ow_abs, ow_uw, diverg=None):
    if not np.isfinite(ow_uw):        return ("SELECTIVE", 0.5)   # unknown -> cautious default
    if ow_uw <= -0.25:                v, sz = ("STAND-ASIDE", 0.0)
    elif ow_uw >= 0.30 and ow_abs > 0: v, sz = ("ACT", 1.0)
    else:                             v, sz = ("SELECTIVE", 0.5)
    if diverg == "BULLTRAP" and sz > 0.5: sz = 0.5                # overlay cap
    return (v, sz)

if __name__ == "__main__":
    df = build(load_panel(), label_regimes(load_nifty()))
    print(f"panel {df['date'].min().date()}->{df['date'].max().date()}  {len(df):,} obs\n")

    REG = ["TRENDING_UP", "HIGH_VOL", "CHOPPY", "TRENDING_DOWN", "REVERSAL"]
    DVS = ["ALIGNED_UP", "DIP_IN_UP", "MIXED", "BULLTRAP", "ALIGNED_DN"]

    print("="*104)
    print("A) REGIME-LEVEL POSTURE (the primary lever) — fwd-10d")
    print("="*104)
    print(f"  {'regime':15s} {'days':>5s} {'OW abs':>8s} {'OW-UW':>8s} {'t':>6s}  {'VERDICT':12s} {'size':>4s}")
    reg_verdict = {}
    for r in REG:
        c = cell_stats(df[df["regime"] == r])
        v, sz = verdict(c["ow_abs"], c["ow_uw"])
        reg_verdict[r] = (v, sz)
        aa = f"{c['ow_abs']:+7.2f}%" if np.isfinite(c['ow_abs']) else "   thin "
        uu = f"{c['ow_uw']:+7.2f}%" if np.isfinite(c['ow_uw']) else "   thin "
        tt = f"{c['t']:+5.1f}" if np.isfinite(c['t']) else "   . "
        print(f"  {r:15s} {c['days']:5d} {aa} {uu} {tt}  {v:12s} {sz:>4.1f}")

    print("\n" + "="*104)
    print("B) DIVERGENCE-LEVEL (short 1-2wk vs medium 1-2mo Nifty trend) — fwd-10d")
    print("="*104)
    print(f"  {'divergence':12s} {'days':>5s} {'OW abs':>8s} {'OW-UW':>8s} {'t':>6s}  {'VERDICT':12s} {'size':>4s}")
    for d in DVS:
        c = cell_stats(df[df["diverg"] == d])
        v, sz = verdict(c["ow_abs"], c["ow_uw"], d)
        aa = f"{c['ow_abs']:+7.2f}%" if np.isfinite(c['ow_abs']) else "   thin "
        uu = f"{c['ow_uw']:+7.2f}%" if np.isfinite(c['ow_uw']) else "   thin "
        tt = f"{c['t']:+5.1f}" if np.isfinite(c['t']) else "   . "
        print(f"  {d:12s} {c['days']:5d} {aa} {uu} {tt}  {v:12s} {sz:>4.1f}")

    print("\n" + "="*104)
    print("C) JOINT regime x divergence — OW-UW spread (blank=thin). Confirms interactions.")
    print("="*104)
    hdr = "  " + f"{'regime':15s}" + "".join(f"{d[:9]:>10s}" for d in DVS)
    print(hdr)
    for r in REG:
        line = f"  {r:15s}"
        for d in DVS:
            c = cell_stats(df[(df["regime"] == r) & (df["diverg"] == d)])
            line += (f"{c['ow_uw']:+9.2f} " if np.isfinite(c["ow_uw"]) else f"{'·':>10s}")
        print(line)

    print("\n" + "="*104)
    print("D) FINAL POSTURE MATRIX (regime primary; divergence downgrades size only)")
    print("   This is what the engine ships. STAND-ASIDE never upgraded by divergence.")
    print("="*104)
    action = {
        "ACT":         "Trade the tilt — overweights are live. Rotate into leaders.",
        "SELECTIVE":   "Half size, top-ranked only — edge is thin/fragile here.",
        "STAND-ASIDE": "No long rotation — the tilt inverts; leaders bleed. Preserve capital.",
    }
    print(f"  {'regime':15s} {'verdict':12s} {'base size':>9s}  action")
    for r in REG:
        v, sz = reg_verdict[r]
        print(f"  {r:15s} {v:12s} {sz:>9.1f}  {action[v]}")
    print("\n  overlays: BULLTRAP (1-2wk up inside 1-2mo down) caps size at 0.5 (weakest fwd state);")
    print("            DIP_IN_UP (1-2wk down inside 1-2mo up) = best entry timing (keep full size).")
    print("            low dispersion (<1.5 rs2w std) -> size x0.5 (nothing to rotate on).")
