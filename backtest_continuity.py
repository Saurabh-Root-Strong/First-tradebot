"""
backtest_continuity.py  —  Does the overnight OI-CONTINUITY read have intraday edge?

THE QUESTION (the one that gates wiring it into the live decision)
-----------------------------------------------------------------
intraday_oi_intel.analyze_continuity() reads LAST NIGHT's structure (call wall =
ceiling, put wall = floor, max pain) and diffs it against THIS MORNING's live OI
at those exact strikes:
    ceiling + more call writing -> REINFORCED  (bearish)
    ceiling + calls covering    -> LIFTING     (bullish, wall removed)
    floor   + more put writing  -> REINFORCED  (bullish)
    floor   + puts covering      -> CRUMBLING   (bearish, floor breaks)
It is computed live and SHOWN, but is *not* wired into build_recommendation /
the buy-sell call. Before wiring it in, measure it. This is the falsification.

DESIGN (lookahead-free)
-----------------------
For each captured intraday day D (parquet chain-snapshot mirrors), per index:
  * walls   = EOD floor/ceiling/max-pain from DCM fno_bhavcopy on the PRIOR
              trade date (strictly < D)  -> reuse eod_oi_range.build()
  * at each morning sample instant T, read the live strike_map (with oich) and
    spot as-of T (reuse engine_replay.ReplayFeed) -> analyze_continuity()
  * outcome = forward spot return T -> T+H  (H in {30, 60} min) and T -> EOD
A continuity "call" is BULLISH if score >= band, BEARISH if <= -band.

METRICS
  * sign-hit  : sign(call) == sign(fwd_ret) on ACTIONABLE rows, vs 50% null (boot CI)
  * IC        : Spearman(score, fwd_ret), bootstrap CI vs 0
  * by action : mean fwd_ret in LIFTING / CRUMBLING / REINFORCED buckets
  * gap split : same, restricted to gap-open days (|gap| > 0.30%) — the case the
                user cares about most (gap breaks the overnight range)

VERDICT: edge only if a CI excludes its null. Sample is small (a handful of
captured days) — treat a pass as "worth wiring + watching", not gospel.

  .venv\\Scripts\\python.exe backtest_continuity.py
  .venv\\Scripts\\python.exe backtest_continuity.py --reps 5000
"""
from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from core.constants import IST, INDEX_SYMBOLS, LABELS, NSE_NAME, DATA_DIR
import intraday_oi_intel
from eod_oi_range import build as build_walls, DCM_DB

SEED = 7
BAND = intraday_oi_intel._NEUTRAL_BAND          # actionable |score| threshold
# Chain (OI) capture only covers ~09:14-11:05 in the data, so sample continuity
# inside that window; forward returns come from full-day ticks.
SAMPLE_TIMES = ["09:45", "10:00", "10:15", "10:30", "10:45"]
DUCK_DIR = DATA_DIR / "intraday"
HORIZONS = [30, 60]                              # minutes
EOD_CUT = datetime.time(15, 20)
SESSION_END = datetime.time(15, 25)
GAP_THRESH = 0.0030                             # 0.30% open gap = "gap day"


# ── duckdb-backed feed (full-day ticks; morning-only chain) ─────────────────────
class DuckFeed:
    """Reads the canonical per-day duckdb (full session) at a virtual instant.

    The parquet mirrors are truncated mid-morning; the duckdb holds the full tick
    session, so forward returns are honest. ts is tz-typed (needs pytz to
    materialise) — we compare via epoch() to sidestep that.
    """
    def __init__(self, date: str):
        import duckdb
        self.date = date
        self.con = duckdb.connect(str(DUCK_DIR / f"{date}.duckdb"), read_only=True)
        self._te = 0.0

    def set_time(self, t: datetime.datetime) -> None:
        self._te = t.timestamp()

    def spot(self, sym: str) -> float:
        r = self.con.execute(
            "select ltp from ticks where symbol=? and epoch(ts)<=? "
            "order by ts desc limit 1", [sym, self._te]).fetchone()
        return float(r[0]) if r and r[0] is not None else 0.0

    def pa_anchor(self, sym: str) -> float:
        """Price-action anchor = the session OPEN. Indices carry no traded volume
        (tick_vol is 0), so VWAP is undefined; spot vs the day's open (green/red)
        is the robust volume-free trend reference."""
        r = self.con.execute(
            "select any_value(day_open) from ticks where symbol=? and epoch(ts)<=?",
            [sym, self._te]).fetchone()
        return float(r[0]) if r and r[0] is not None else 0.0

    def strike_map(self, sym: str) -> dict:
        ts = self.con.execute(
            "select max(epoch(ts)) from chain_snapshots where symbol=? and epoch(ts)<=?",
            [sym, self._te]).fetchone()
        if not ts or ts[0] is None:
            return {}
        rows = self.con.execute(
            "select strike, side, ltp, oi, oich, volume from chain_snapshots "
            "where symbol=? and epoch(ts)=?", [sym, ts[0]]).fetchall()
        sm: dict = {}
        for sp, side, ltp, oi, oich, vol in rows:
            sm.setdefault(float(sp), {})[side] = {
                "ltp": float(ltp or 0), "oi": float(oi or 0),
                "oich": float(oich or 0), "volume": float(vol or 0)}
        return sm

    def close(self):
        try: self.con.close()
        except Exception: pass


def _captured_days() -> list[str]:
    """Duckdb days that actually have chain_snapshots with non-zero oich."""
    import duckdb
    days = []
    for p in sorted(DUCK_DIR.glob("2026-*.duckdb")):
        date = p.name[:10]
        try:
            con = duckdb.connect(str(p), read_only=True)
            tbls = {r[0] for r in con.execute("show tables").fetchall()}
            if "chain_snapshots" in tbls:
                nz = con.execute("select count(*) from chain_snapshots "
                                 "where oich is not null and oich<>0").fetchone()[0]
                if nz > 100:
                    days.append(date)
            con.close()
        except Exception:
            pass
    return days


# ── prior-EOD walls per index, indexed by trade_date (built once) ───────────────
def _walls_tables(con) -> dict:
    """{dcm_sym: DataFrame indexed by date with floor/ceiling/magnet/spot}."""
    out = {}
    for sym in INDEX_SYMBOLS:
        dcm = NSE_NAME[sym]
        try:
            df = build_walls(dcm, con)
        except Exception:
            df = pd.DataFrame()
        if not df.empty:
            out[sym] = df.set_index("date").sort_index()
    return out


def _prior_walls(wt: pd.DataFrame, day: datetime.date):
    """Latest wall row strictly before `day` -> levels dict, or None."""
    prior = wt[wt.index < day]
    if prior.empty:
        return None, None
    r = prior.iloc[-1]
    levels = {"top_call_strike": float(r["ceiling"]),
              "top_put_strike":  float(r["floor"]),
              "max_pain_price":  float(r["magnet"])}
    return levels, float(r["spot"])             # prior close (for gap calc)


# ── stats helpers ───────────────────────────────────────────────────────────────
def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return float("nan")
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def _boot_ci(fn, *arrs, reps: int, rng) -> tuple[float, float, float]:
    base = fn(*arrs)
    n = len(arrs[0])
    if n < 3:
        return base, float("nan"), float("nan")
    vals = []
    for _ in range(reps):
        idx = rng.integers(0, n, n)
        vals.append(fn(*[a[idx] for a in arrs]))
    vals = np.array([v for v in vals if v == v])
    if not len(vals):
        return base, float("nan"), float("nan")
    return base, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# ── collect the (continuity, forward-return) rows ───────────────────────────────
def collect(days: list[str], walls: dict) -> pd.DataFrame:
    rows = []
    for date in days:
        d0 = datetime.date.fromisoformat(date)
        feed = DuckFeed(date)
        try:
            for sym in INDEX_SYMBOLS:
                wt = walls.get(sym)
                if wt is None:
                    continue
                levels, prior_close = _prior_walls(wt, d0)
                if not levels:
                    continue
                feed.set_time(datetime.datetime.combine(d0, datetime.time(9, 16), tzinfo=IST))
                open_spot = feed.spot(sym)
                gap = (open_spot / prior_close - 1.0) if (open_spot and prior_close) else 0.0

                for hhmm in SAMPLE_TIMES:
                    hh, mm = map(int, hhmm.split(":"))
                    t = datetime.datetime.combine(d0, datetime.time(hh, mm), tzinfo=IST)
                    feed.set_time(t)
                    spot = feed.spot(sym)
                    sm = feed.strike_map(sym)
                    if not spot or not sm:
                        continue
                    cont = intraday_oi_intel.analyze_continuity(sm, spot, levels)
                    if not cont.has_data:
                        continue
                    # forward returns from full-day ticks
                    fwd = {}
                    for h in HORIZONS:
                        tt = t + datetime.timedelta(minutes=h)
                        if tt.time() > SESSION_END:
                            fwd[h] = np.nan
                            continue
                        feed.set_time(tt)
                        s2 = feed.spot(sym)
                        fwd[h] = (s2 / spot - 1.0) if (s2 and s2 != spot) else np.nan
                    feed.set_time(datetime.datetime.combine(d0, EOD_CUT, tzinfo=IST))
                    s_eod = feed.spot(sym)
                    fwd_eod = (s_eod / spot - 1.0) if (s_eod and s_eod != spot) else np.nan
                    feed.set_time(t)   # restore

                    rows.append({
                        "date": date, "sym": sym, "t": hhmm, "gap": gap,
                        "score": cont.score, "verdict": cont.verdict,
                        "res_action": cont.res_action, "sup_action": cont.sup_action,
                        "fwd30": fwd.get(30, np.nan), "fwd60": fwd.get(60, np.nan),
                        "fwd_eod": fwd_eod,
                    })
        finally:
            feed.close()
    return pd.DataFrame(rows)


# ── analysis / report ───────────────────────────────────────────────────────────
def report(df: pd.DataFrame, reps: int, rng) -> None:
    print("\nOVERNIGHT OI-CONTINUITY — intraday edge test (lookahead-free)")
    print("=" * 78)
    if df.empty:
        print("  no rows — no captured days with prior-EOD walls + live oich.")
        return
    days = sorted(df.date.unique())
    print(f"  rows={len(df)}  days={len(days)} ({days[0]}..{days[-1]})  "
          f"indices={df.sym.nunique()}  gap-days rows={int((df.gap.abs()>GAP_THRESH).sum())}")
    print(f"  actionable band |score|>={BAND}")

    for label, sub in [("ALL", df), (f"GAP (|gap|>{GAP_THRESH*100:.2f}%)", df[df.gap.abs() > GAP_THRESH])]:
        print(f"\n── {label}  (n={len(sub)}) " + "-" * 40)
        if len(sub) < 5:
            print("   too few rows."); continue
        for hcol in ["fwd30", "fwd60", "fwd_eod"]:
            d = sub.dropna(subset=[hcol])
            if len(d) < 5:
                print(f"   {hcol:7}: n<5"); continue
            score = d["score"].to_numpy(float)
            ret = d[hcol].to_numpy(float)
            ic, iclo, ichi = _boot_ci(_spearman, score, ret, reps=reps, rng=rng)
            # actionable sign-hit
            act = d[d.score.abs() >= BAND]
            if len(act) >= 5:
                call = np.sign(act["score"].to_numpy())
                hit = (call == np.sign(act[hcol].to_numpy())).astype(float)
                hr, hlo, hhi = _boot_ci(lambda a: a.mean(), hit, reps=reps, rng=rng)
                hit_s = f"hit {100*hr:4.1f}% [{100*hlo:4.1f},{100*hhi:4.1f}] n={len(act)}"
                hv = "EDGE" if hlo > 0.5 else ("anti" if hhi < 0.5 else "—")
            else:
                hit_s, hv = "hit n<5", "—"
            icv = "EDGE" if iclo > 0 else ("anti" if ichi < 0 else "—")
            print(f"   {hcol:7}: IC {ic:+.3f} [{iclo:+.3f},{ichi:+.3f}] {icv:4} | {hit_s} {hv}")

    # by-action mean forward (the mechanism check)
    print("\n── mean forward return by overnight action (fwd60, bps) " + "-" * 16)
    for col in ["res_action", "sup_action"]:
        g = df.groupby(col)["fwd60"].agg(["count", "mean"]).dropna()
        if g.empty:
            continue
        print(f"   {col}:")
        for act, r in g.iterrows():
            if not act:
                continue
            print(f"      {act:13} n={int(r['count']):3}  {1e4*r['mean']:+7.1f} bps")

    print("\n" + "=" * 78)
    print("READ: 'EDGE' = bootstrap 95% CI excludes the null (IC>0 / hit>50%).")
    print("Small sample — a pass = wire it in capped + keep watching; a fail = keep")
    print("it as display-only context (same shelf as max pain).")


def main() -> None:
    ap = argparse.ArgumentParser(description="overnight OI-continuity intraday edge test")
    ap.add_argument("--reps", type=int, default=3000)
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)

    days = _captured_days()
    if not days:
        print("No captured chain-snapshot mirrors found in", LIVE_DIR)
        return
    if not DCM_DB.exists():
        print("DCM DuckDB not found:", DCM_DB)
        return

    import duckdb
    con = duckdb.connect(str(DCM_DB), read_only=True)
    walls = _walls_tables(con)
    if not walls:
        print("No EOD walls built from DCM fno_bhavcopy.")
        return
    df = collect(days, walls)
    report(df, args.reps, rng)


if __name__ == "__main__":
    main()
