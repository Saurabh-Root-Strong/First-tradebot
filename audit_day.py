"""
audit_day.py — episode-level audit of the scout on ONE day across 5/10/15/30m.

Complements backtest_scout (checkpoint-sampled, multi-day, CI). This is the per-day
forensic: it reconstructs the ACTUAL trades the scout would have fired that day per
index per timeframe, walks each to resolution (SL / target / open-at-close) on the
captured option premium, and diagnoses WHY the SLs hit using price_structure (fired
into resistance / a no-breakout coil?) and the CE/PE-vs-day-trend mismatch.

The 2026-06-29 run was the case study: 34 trades, 12 SL (35%), 1 target, mean
−11.5%/trade (−391% total) — 32/34 were CE on a DOWN day, so the failure was WRONG
DIRECTION (trend error), not stop placement or S/R; only 5/12 SLs were struct-flagged.

Lookahead-free: every verdict is scan_index(as_of=t) over ts<=t mirrors; the forward
premium walk is the answer key only.

    .venv\\Scripts\\python.exe audit_day.py 2026-06-29
"""
from __future__ import annotations

import datetime
import sys

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from core.constants import IST, INDEX_SYMBOLS, LABELS
from core.mirror_io import read_mirror as R
import footprint_chart as fc
import intraday_scout as scout
import price_structure as ps

DATE = sys.argv[1] if len(sys.argv) > 1 else "2026-06-29"
TFS = [5, 10, 15, 30]
SESS_OPEN = datetime.time(9, 15)
SESS_CLOSE = datetime.time(15, 30)


def _bar_grid(tf: int):
    d0 = datetime.date.fromisoformat(DATE)
    t = datetime.datetime.combine(d0, datetime.time(9, 35), tzinfo=IST)  # first tradeable
    end = datetime.datetime.combine(d0, SESS_CLOSE, tzinfo=IST)
    out = []
    while t <= end:
        out.append(t)
        t += datetime.timedelta(minutes=tf)
    return out


def _prem_series(sym, strike, side):
    """All (ts, ltp) for one strike/side today, session-bounded, ascending."""
    ch = R("chain_snapshots", DATE, None, sym)
    if ch is None or not len(ch) or "ltp" not in ch.columns:
        return None
    ch, ok = fc._filter_expiry(ch, "weekly")
    if not ok or ch is None or not len(ch):
        return None
    sub = ch[(ch["side"] == side) & (ch["strike"] == strike)].copy()
    sub = sub[(sub["ts"].dt.time >= SESS_OPEN) & (sub["ts"].dt.time <= SESS_CLOSE)]
    sub = sub[pd.notna(sub["ltp"]) & (sub["ltp"] > 0)].sort_values("ts")
    return sub[["ts", "ltp"]] if len(sub) else None


def _resolve(sym, tf, trig_t, side, strike, entry):
    """Walk premium forward from entry → first of SL / target / close. Returns dict."""
    sl_pct, t1_pct = scout._SLT.get(tf, (0.32, 0.55))
    sl, tgt = entry * (1 - sl_pct), entry * (1 + t1_pct)
    ps_ = _prem_series(sym, strike, side)
    if ps_ is None:
        return {"outcome": "no-prem", "exit": None, "net_pct": None, "held_min": None}
    fwd = ps_[ps_["ts"] > pd.Timestamp(trig_t)]
    outcome, exit_p, exit_t = "OPEN@close", None, None
    for _, r in fwd.iterrows():
        p = float(r["ltp"])
        if p <= sl:
            outcome, exit_p, exit_t = "SL", p, r["ts"]; break
        if p >= tgt:
            outcome, exit_p, exit_t = "TARGET", p, r["ts"]; break
    if exit_p is None and len(fwd):
        exit_p, exit_t = float(fwd.iloc[-1]["ltp"]), fwd.iloc[-1]["ts"]
    net = ((exit_p / entry - 1.0) * 100.0 - scout._OPT_RT_COST) if exit_p else None
    held = ((exit_t - pd.Timestamp(trig_t)).total_seconds() / 60.0) if exit_t else None
    return {"outcome": outcome, "exit": round(exit_p, 2) if exit_p else None,
            "net_pct": round(net, 1) if net is not None else None,
            "held_min": round(held) if held else None,
            "sl": round(sl, 2), "tgt": round(tgt, 2)}


def episodes(sym, tf):
    """Distinct trades today: a TRADE bar whose prior bar was not the SAME-dir trade."""
    grid = _bar_grid(tf)
    prev_dir = None
    out = []
    for t in grid:
        r = scout.scan_index(sym, tf, date=DATE, as_of=t, verdict_only=True)
        d = r.get("direction") if r.get("verdict", "").startswith("TRADE") else None
        if d and d != prev_dir:
            # confirm with the FULL path (applies warmup/settle + struct) before logging
            full = scout.scan_index(sym, tf, date=DATE, as_of=t, with_lifecycle=False)
            if full.get("verdict", "").startswith("TRADE"):
                spot = full.get("spot")
                atm = full.get("atm")
                entry = scout._opt_premium(sym, DATE, t, atm, d) if atm else None
                st = full.get("struct") or {}
                rg = full.get("regime") or {}
                veto, vr = ps.veto(st, d)
                ep = {"t": t.strftime("%H:%M"), "dir": d, "strike": atm,
                      "entry": round(entry, 2) if entry else None, "spot": spot,
                      "str": full.get("strength"), "agree": full.get("agree"),
                      "veto": veto, "brk": st.get("breakout"),
                      "coil": st.get("consolidating"),
                      "dres": st.get("dist_res_atr"), "dsup": st.get("dist_sup_atr"),
                      "trend": rg.get("trend"), "tveto": bool(full.get("trend_veto"))}
                if entry and atm:
                    ep.update(_resolve(sym, tf, t, d, atm, entry))
                else:
                    ep.update({"outcome": "no-entry-prem", "net_pct": None,
                               "held_min": None})
                out.append(ep)
        prev_dir = d
    return out


def run():
    print(f"\nSCOUT EPISODE AUDIT — {DATE}  (actual trades, walked to resolution)")
    print("=" * 92)
    grand = []
    for tf in TFS:
        print(f"\n┌─ {tf}m bars " + "─" * 78)
        tf_rows = []
        for sym in INDEX_SYMBOLS:
            eps = episodes(sym, tf)
            for e in eps:
                e["sym"] = LABELS.get(sym, sym); e["tf"] = tf
                tf_rows.append(e); grand.append(e)
                tag = "🛑" if e["outcome"] == "SL" else ("🎯" if e["outcome"] == "TARGET" else "·")
                struct = (f"trend={e['trend']}" + ("⛔" if e["tveto"] else "")
                          + f" brk={e['brk']} coil={e['coil']}")
                print(f"  {tag} {e['sym']:13s} {e['t']} {e['dir']} {e['strike']} "
                      f"@{e['entry']}  str{e['str']:+.2f} ag{e['agree']}  "
                      f"-> {e['outcome']:9s} net {e['net_pct']}%  held {e['held_min']}m  | {struct}")
        if not tf_rows:
            print("  (no trades fired)")
            continue
        df = pd.DataFrame(tf_rows)
        n = len(df)
        sl = int((df.outcome == "SL").sum())
        tg = int((df.outcome == "TARGET").sum())
        op = int((df.outcome == "OPEN@close").sum())
        net = df["net_pct"].dropna()
        wins = int((net > 0).sum())
        sl_veto = int(((df.outcome == "SL") & (df.veto)).sum())
        print(f"  └─ n={n}  🎯{tg} 🛑{sl} ·open{op}  win={wins}/{len(net)}  "
              f"mean net={net.mean():+.1f}%  median={net.median():+.1f}%  "
              f"|  SLs the struct-veto flagged: {sl_veto}/{sl}")

    # ── cross-timeframe rollup ───────────────────────────────────────────────────
    print("\n" + "=" * 92)
    print("CROSS-TIMEFRAME ROLLUP")
    g = pd.DataFrame(grand)
    if len(g):
        for tf in TFS:
            s = g[g.tf == tf]
            if not len(s):
                print(f"  {tf:>2}m  no trades"); continue
            net = s["net_pct"].dropna()
            print(f"  {tf:>2}m  trades={len(s):2d}  SL={int((s.outcome=='SL').sum())} "
                  f"TGT={int((s.outcome=='TARGET').sum())} "
                  f"open={int((s.outcome=='OPEN@close').sum())}  "
                  f"win={int((net>0).sum())}/{len(net)}  "
                  f"mean net={net.mean():+.1f}%  total net={net.sum():+.1f}%")
        net = g["net_pct"].dropna()
        sl = g[g.outcome == "SL"]
        print(f"\n  ALL TF: {len(g)} trades, {int((g.outcome=='SL').sum())} SL "
              f"({100*(g.outcome=='SL').mean():.0f}%), "
              f"{int((g.outcome=='TARGET').sum())} target, "
              f"mean net {net.mean():+.1f}%, total {net.sum():+.1f}%")
        if len(sl):
            print(f"  SL diagnosis: {int(sl.veto.sum())}/{len(sl)} fired into resistance/coil "
                  f"(struct-veto); {int(sl.tveto.sum())}/{len(sl)} fired AGAINST the day "
                  f"trend (trend-veto)")
        # ── counterfactual: what if we'd applied each veto? ──────────────────────
        print("\n  COUNTERFACTUAL — net P&L if a veto had SKIPPED those trades:")
        base = g["net_pct"].dropna()
        for name, mask in [("struct-veto", ~g.veto.fillna(False)),
                           ("trend-veto", ~g.tveto.fillna(False))]:
            kept = g[mask]["net_pct"].dropna()
            skipped = len(g) - int(mask.sum())
            print(f"    {name:11s} keeps {len(kept):2d}/{len(g)} trades (skips {skipped})  "
                  f"-> mean net {kept.mean():+.1f}%  total {kept.sum():+.1f}%  "
                  f"(vs base mean {base.mean():+.1f}% total {base.sum():+.1f}%)")
        # trend-aligned vs trend-against split (the core thesis)
        aligned = g[~g.tveto.fillna(False)]["net_pct"].dropna()
        against = g[g.tveto.fillna(False)]["net_pct"].dropna()
        print(f"\n  THESIS: trend-ALIGNED trades mean {aligned.mean():+.1f}% (n={len(aligned)})  "
              f"vs trend-AGAINST mean {against.mean():+.1f}% (n={len(against)})")


if __name__ == "__main__":
    run()
