"""
backtest_engine.py  —  Walk-forward backtest of the Tradebot technical engine
                       over the downloaded F&O historical data.

WHAT THIS DOES (and, honestly, what it cannot)
----------------------------------------------
The live engine (trade_setup.build_recommendation) blends 10 layers. Only TWO are
reconstructable from plain OHLCV history:

  • Layer 1  — Technical (signals._analyze)      ← fully reproduced here
  • Layer 10 — price-impulse part of the shock    ← partially (price only)

The other layers (OI flow, IV, futures basis, PCR, max-pain, velocity, FII/FAO,
daily-context) require historical OPTION-CHAIN + intraday-snapshot data we do not
store, so they are out of scope. Likewise, live trades are taken on OPTION premiums
with SL/T1/T2 as % moves on the premium (trade_setup.py:954) — we have no premium /
IV / greek history, so synthesising those would be fiction dominated by theta/IV.

Therefore this harness backtests the thing that actually carries the edge: the
engine's MULTI-TIMEFRAME TECHNICAL CONSENSUS as a directional signal on the
UNDERLYING, managed with the SAME R-based trade-management state machine the live
ledger uses (intraday_trades._apply), but with ATR-based geometry on price instead
of premium %. Positive expectancy here is necessary (not sufficient) for the full
composite to work; it isolates whether the technical backbone has a real edge across
the F&O universe — directly on-mission for "backtesting the Tradebot engine".

FIDELITY GUARANTEES
  • Indicators reuse signals.py's own _ema/_rsi/_macd/_vwap/_bollinger.
  • The bar score is a VECTORISED copy of signals._analyze, self-tested at startup
    to match signals._analyze exactly (--self-test, on by default).
  • Higher timeframes are aligned strictly causally (merge_asof on closed bars) —
    no lookahead. Entry is at the signal bar's close; SL/T1/T2 are resolved from the
    NEXT bar onward; if a bar's range spans both stop and target, the stop is taken
    first (pessimistic).

FIDELITY GAP — why this is an UPPER BOUND on live
  Live signals.fetch_ohlcv appends the FORMING (partial, still-moving) candle so the
  dashboard reacts intra-bar. This harness scores only CLOSED bars. So live takes
  some entries on partial-bar signals that later flip, which this backtest never
  sees. Read any positive expectancy here as a CEILING the live engine approaches
  from below — not a like-for-like estimate of live fills.

Usage
  .venv\\Scripts\\python.exe backtest_engine.py --symbols NIFTY               # quick single
  .venv\\Scripts\\python.exe backtest_engine.py --symbols NIFTY,RELIANCE,BANKNIFTY
  .venv\\Scripts\\python.exe backtest_engine.py                               # full universe
  .venv\\Scripts\\python.exe backtest_engine.py --min-band BUY --sl-atr 1.5 --t1-r 1.5 --t2-r 3
"""

from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

# Reuse the live engine's own indicator implementations so the backtest signal is
# computed from the identical primitives the dashboard uses.
import signals as S
from fno_universe import ALL_SYMBOLS, LABELS

DATA_DIR = Path("data") / "historical"
OUT_DIR  = Path("data") / "backtest"
from core.constants import IST   # single source of truth  # noqa: E402

# ── Consensus config — kept IN SYNC with signals.analyze_index ──────────────────
# (signals.py: TIMEFRAMES, weights at analyze_index, overall bands)
TF_KEYS      = ["5min", "15min", "60min", "daily"]
TF_WEIGHTS   = {"5min": 0.20, "15min": 0.25, "60min": 0.25, "daily": 0.30}
TF_DAYS      = {k: S.TIMEFRAMES[k]["days"] for k in TF_KEYS}       # live trailing window
TF_MINBARS   = {k: S.TIMEFRAMES[k]["min_candles"] for k in TF_KEYS}
EXEC_TF      = "5min"                                              # decision + fill grid

# Overall-band thresholds, copied verbatim from signals.analyze_index.
def _overall_band(w: float) -> str:
    if   w >=  2.5: return "STRONG BUY"
    elif w >=  1.0: return "BUY"
    elif w >=  0.3: return "MILD BUY"
    elif w <= -2.5: return "STRONG SELL"
    elif w <= -1.0: return "SELL"
    elif w <= -0.3: return "MILD SELL"
    return "NEUTRAL"

_BUY_BANDS  = {"STRONG BUY", "BUY", "MILD BUY"}
_SELL_BANDS = {"STRONG SELL", "SELL", "MILD SELL"}
_BAND_RANK  = {"MILD BUY": 1, "BUY": 2, "STRONG BUY": 3,
               "MILD SELL": 1, "SELL": 2, "STRONG SELL": 3}

SESSION_SQUAREOFF = datetime.time(15, 25)   # no overnight holds (intraday model)


# ── Vectorised score — faithful copy of signals._analyze rules 1-7 ──────────────

def vectorized_scores(df: pd.DataFrame) -> pd.Series:
    """Per-bar technical score, identical to signals._analyze run at each bar.

    Returns a float Series aligned to df.index; NaN where indicators aren't warm
    (mirrors _analyze's INSUFFICIENT-DATA guard → treated as no-signal).
    """
    close, vol = df["close"], df["volume"]
    e9, e21, e50 = S._ema(close, 9), S._ema(close, 21), S._ema(close, 50)
    m, ms, mh    = S._macd(close)
    r            = S._rsi(close)
    vw           = S._vwap(df)
    bb_lo, _, bb_hi = S._bollinger(close)
    avg_vol = vol.rolling(20).mean()

    cp   = close.shift(1)
    e9p, e21p = e9.shift(1), e21.shift(1)
    mhp  = mh.shift(1)
    rp   = r.shift(1)

    idx = df.index

    # 1. EMA 9/21 fresh cross, else alignment.  np.select keeps _analyze's if/elif order.
    s1 = pd.Series(np.select(
        [(e9p < e21p) & (e9 > e21), (e9p > e21p) & (e9 < e21), (e9 > e21)],
        [2.0, -2.0, 1.0], default=-1.0), index=idx)

    # 2. EMA50 structure
    s2 = pd.Series(np.where(close > e50, 1.0, -1.0), index=idx)

    # 3. MACD histogram (zero-cross, then expansion)
    s3 = pd.Series(np.select(
        [(mhp < 0) & (mh > 0), (mhp > 0) & (mh < 0),
         (mh > 0) & (mh > mhp), (mh < 0) & (mh < mhp)],
        [2.0, -2.0, 1.0, -1.0], default=0.0), index=idx)

    # 4. RSI
    s4 = pd.Series(np.select(
        [r < 30, r > 70, (r > 55) & (r > rp), (r < 45) & (r < rp)],
        [2.0, -2.0, 1.0, -1.0], default=0.0), index=idx)

    # 5. VWAP — score ONLY when VWAP is defined, matching _analyze's notna guard.
    # _vwap returns NaN when cumulative volume is 0 (index 5-min bars frequently have
    # zero traded volume from Fyers), and _analyze SKIPS the rule then (scores 0).
    # The old np.where scored -1 on NaN VWAP, a spurious bearish tick that diverged
    # from live by exactly +1.0 on every no-volume bar (~half of them).
    s5 = pd.Series(np.where(vw.isna(), 0.0, np.where(close > vw, 1.0, -1.0)), index=idx)

    # 6. Volume confirmation
    hv = vol > avg_vol * 1.5
    s6 = pd.Series(np.select([hv & (close > cp), hv & (close < cp)], [1.0, -1.0], default=0.0), index=idx)

    # 7. Bollinger position
    s7 = pd.Series(np.select([close > bb_hi, close < bb_lo], [-0.5, 0.5], default=0.0), index=idx)

    score = (s1 + s2 + s3 + s4 + s5 + s6 + s7).round(1)

    # Warm-up guard: any core indicator NaN → no signal.
    warm = e50.notna() & ms.notna() & r.notna() & avg_vol.notna() & cp.notna()
    return score.where(warm)


def _self_test(df: pd.DataFrame, n: int = 200) -> None:
    """Assert vectorized_scores == signals._analyze on n random end-points."""
    vec = vectorized_scores(df)
    rng = np.random.default_rng(7)
    idx = [i for i in rng.integers(60, len(df), size=min(n, max(0, len(df) - 60)))]
    bad = 0
    for i in idx:
        ref = S._analyze(df.iloc[: i + 1].reset_index(drop=True), 30)
        rs  = ref.get("score")
        vs  = vec.iloc[i]
        if ref.get("signal") == "INSUFFICIENT DATA":
            continue
        if pd.isna(vs) or abs(float(rs) - float(vs)) > 1e-6:
            bad += 1
            if bad <= 5:
                print(f"    MISMATCH @ {i}: _analyze={rs}  vectorized={vs}")
    if bad:
        raise AssertionError(f"vectorized_scores diverged from _analyze on {bad}/{len(idx)} bars")
    print(f"    self-test OK — vectorized score == _analyze on {len(idx)} sampled bars")


# ── Data loading ────────────────────────────────────────────────────────────────

def _safe(sym: str) -> str:
    return sym.replace(":", "_").replace("-", "_")


def _load_tf(fy_sym: str, tf: str) -> pd.DataFrame | None:
    p = DATA_DIR / tf / f"{_safe(fy_sym)}_{tf}.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    df["ts"] = pd.to_datetime(df["ts"]).astype("datetime64[ns]")
    return df.sort_values("ts").reset_index(drop=True)


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()   # Wilder ATR


# ── Signal construction (causal multi-timeframe consensus) ──────────────────────

def build_signal_grid(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the 5-min execution grid with weighted consensus + band per bar.

    Higher-TF scores are merged as-of on each TF's CLOSE time (ts + tf_minutes),
    so a bar only influences decisions strictly after it has closed.
    """
    tf_minutes = {"5min": 5, "15min": 15, "60min": 60, "daily": None}
    per_tf_score = {}
    for tf, df in frames.items():
        sc = vectorized_scores(df)
        close_ts = (df["ts"] + pd.to_timedelta(tf_minutes[tf], unit="m")
                    if tf_minutes[tf] else
                    # daily bar "closes" at session end of its date
                    (df["ts"].dt.normalize() + pd.Timedelta(hours=15, minutes=30)))
        close_ts = close_ts.astype("datetime64[ns]")
        per_tf_score[tf] = pd.DataFrame({"close_ts": close_ts, f"sc_{tf}": sc}) \
            .dropna(subset=[f"sc_{tf}"]).sort_values("close_ts")

    base = frames[EXEC_TF][["ts", "open", "high", "low", "close", "volume"]].copy()
    base["atr"] = _atr(frames[EXEC_TF])
    grid = base.rename(columns={"ts": "dt"}).sort_values("dt")

    for tf in TF_KEYS:
        grid = pd.merge_asof(
            grid, per_tf_score[tf].rename(columns={"close_ts": "dt"}),
            on="dt", direction="backward",
        )

    sc_cols = [f"sc_{tf}" for tf in TF_KEYS]
    w = np.array([TF_WEIGHTS[tf] for tf in TF_KEYS])
    scmat = grid[sc_cols].to_numpy(dtype="float64")
    grid["w_score"] = np.nansum(scmat * w, axis=1)
    # agreement = max(#TF score>0.5, #TF score<-0.5)
    grid["bull_tf"] = np.nansum(scmat > 0.5, axis=1)
    grid["bear_tf"] = np.nansum(scmat < -0.5, axis=1)
    grid["all_tf_ready"] = (~np.isnan(scmat)).all(axis=1)
    grid["band"] = grid["w_score"].map(_overall_band)
    return grid


# ── Trade simulation — mirrors intraday_trades._apply / maybe_open ──────────────

def simulate(grid: pd.DataFrame, label: str, fy_sym: str, cfg: dict) -> list[dict]:
    """Walk the 5-min grid, open/manage one position per direction at a time."""
    trades: list[dict] = []
    open_pos: dict | None = None       # at most one live position (no CE+PE together)
    last_exit: dict = {}               # direction -> {"ts","status","strength"}

    dt   = grid["dt"].to_numpy()
    o, h, l, c = (grid[x].to_numpy(dtype="float64") for x in ("open", "high", "low", "close"))
    atr  = grid["atr"].to_numpy(dtype="float64")
    band = grid["band"].to_numpy()
    wsc  = grid["w_score"].to_numpy(dtype="float64")
    agree = np.maximum(grid["bull_tf"].to_numpy(), grid["bear_tf"].to_numpy())
    ready = grid["all_tf_ready"].to_numpy()

    min_rank = _BAND_RANK_MIN = cfg["min_rank"]
    sl_atr, t1_r, t2_r = cfg["sl_atr"], cfg["t1_r"], cfg["t2_r"]
    cooldown = pd.Timedelta(minutes=cfg["cooldown_min"])
    min_agree = cfg["min_agree"]

    n = len(grid)
    for i in range(n):
        ts = pd.Timestamp(dt[i])
        tm = ts.time()
        # ── manage an open position on THIS bar (from the bar after entry) ──────
        if open_pos and i > open_pos["entry_i"]:
            d = open_pos
            long = d["dir"] == "CE"
            hit_sl  = (l[i] <= d["sl"]) if long else (h[i] >= d["sl"])
            hit_t2  = (h[i] >= d["t2"]) if long else (l[i] <= d["t2"])
            hit_t1  = (h[i] >= d["t1"]) if long else (l[i] <= d["t1"])
            risk = d["risk"]
            closed = None
            # Pessimistic intrabar ordering: stop before target.
            if hit_sl:
                if d["booked"]:
                    r = round(0.5 * d["r_t1"] + 0.5 * ((d["sl"] - d["entry"]) / risk *
                                                       (1 if long else -1)), 2)
                    closed = ("T1", d["sl"], r)
                else:
                    closed = ("SL", d["sl"], -1.0)
            elif hit_t2:
                base_r = (d["t2"] - d["entry"]) / risk * (1 if long else -1)
                if d["booked"]:
                    r = round(0.5 * d["r_t1"] + 0.5 * base_r, 2)
                elif hit_t1:
                    # bar spans T1→T2 in one move: realistically half books at T1 and
                    # the rest at T2 — NOT full size at T2 (which overstated by ~0.75R).
                    r_t1 = (d["t1"] - d["entry"]) / risk * (1 if long else -1)
                    r = round(0.5 * r_t1 + 0.5 * base_r, 2)
                else:
                    r = round(base_r, 2)
                closed = ("T2", d["t2"], r)
            elif (not d["booked"]) and hit_t1:
                d["booked"] = True
                d["r_t1"]   = (d["t1"] - d["entry"]) / risk * (1 if long else -1)
                d["sl"]     = d["entry"]            # lift stop to breakeven
            # square-off at session end
            if closed is None and tm >= SESSION_SQUAREOFF:
                px = c[i]
                base_r = (px - d["entry"]) / risk * (1 if long else -1)
                r = round(0.5 * d["r_t1"] + 0.5 * base_r, 2) if d["booked"] else round(base_r, 2)
                closed = ("EOD", px, r)
            if closed:
                status, xpx, r = closed
                trades.append({**d["rec"], "exit_ts": ts, "exit_px": round(xpx, 2),
                               "status": status, "r": r,
                               "outcome": "WIN" if r > 0 else "LOSS" if r < 0 else "SCRATCH",
                               "bars_held": i - d["entry_i"]})
                last_exit[d["dir"]] = {"ts": ts, "status": status, "strength": d["strength"]}
                open_pos = None

        # ── consider a new entry on this bar's close ───────────────────────────
        if tm < datetime.time(9, 20) or tm >= SESSION_SQUAREOFF:
            continue
        if not ready[i] or np.isnan(atr[i]) or atr[i] <= 0:
            continue
        b = band[i]
        want = "CE" if b in _BUY_BANDS else "PE" if b in _SELL_BANDS else None
        if want is None or _BAND_RANK.get(b, 0) < min_rank or agree[i] < min_agree:
            continue

        # direction flip — close opposite open position at this close, then re-eval
        if open_pos and open_pos["dir"] != want:
            d = open_pos; long = d["dir"] == "CE"; px = c[i]
            base_r = (px - d["entry"]) / d["risk"] * (1 if long else -1)
            r = round(0.5 * d["r_t1"] + 0.5 * base_r, 2) if d["booked"] else round(base_r, 2)
            trades.append({**d["rec"], "exit_ts": ts, "exit_px": round(px, 2),
                           "status": "FLIP", "r": r,
                           "outcome": "WIN" if r > 0 else "LOSS" if r < 0 else "SCRATCH",
                           "bars_held": i - d["entry_i"]})
            last_exit[d["dir"]] = {"ts": ts, "status": "FLIP", "strength": d["strength"]}
            open_pos = None
        if open_pos:                       # already in same direction — hold
            continue

        # cooldown + post-SL stronger-signal discipline (intraday_trades.maybe_open)
        le = last_exit.get(want)
        strength = abs(wsc[i])
        if le and le["ts"] is not None:
            if ts - le["ts"] < cooldown:
                continue
            if le["status"] == "SL" and strength <= le["strength"]:
                continue

        entry = c[i]
        dist  = sl_atr * atr[i]
        long  = want == "CE"
        sl = entry - dist if long else entry + dist
        t1 = entry + t1_r * dist if long else entry - t1_r * dist
        t2 = entry + t2_r * dist if long else entry - t2_r * dist
        open_pos = {
            "dir": want, "entry": entry, "sl": sl, "t1": t1, "t2": t2,
            "risk": dist, "booked": False, "r_t1": 0.0, "entry_i": i,
            "strength": strength,
            "rec": {"symbol": fy_sym, "label": label, "dir": want, "band": b,
                    "w_score": round(float(wsc[i]), 2), "agree": int(agree[i]),
                    "entry_ts": ts, "entry_px": round(entry, 2),
                    "sl": round(sl, 2), "t1": round(t1, 2), "t2": round(t2, 2),
                    "atr": round(float(atr[i]), 2)},
        }
    return trades


# ── Stats ───────────────────────────────────────────────────────────────────────

def summarize(tr: pd.DataFrame, scope: str) -> dict:
    if tr.empty:
        return {"scope": scope, "trades": 0}
    r = tr["r"]
    wins = tr[tr["r"] > 0]["r"]; losses = tr[tr["r"] < 0]["r"]
    gross_w, gross_l = wins.sum(), -losses.sum()
    eq = r.cumsum()
    dd = (eq - eq.cummax()).min()
    return {
        "scope": scope,
        "trades": len(tr),
        "win%": round(100 * (r > 0).mean(), 1),
        "expectancy_R": round(r.mean(), 3),
        "total_R": round(r.sum(), 1),
        "profit_factor": round(gross_w / gross_l, 2) if gross_l > 0 else float("inf"),
        "avg_win_R": round(wins.mean(), 2) if len(wins) else 0.0,
        "avg_loss_R": round(losses.mean(), 2) if len(losses) else 0.0,
        "max_dd_R": round(dd, 1),
        "avg_bars_held": round(tr["bars_held"].mean(), 1),
    }


def _print_table(rows: list[dict]) -> None:
    if not rows:
        print("  (no trades)"); return
    cols = ["scope", "trades", "win%", "expectancy_R", "total_R",
            "profit_factor", "avg_win_R", "avg_loss_R", "max_dd_R", "avg_bars_held"]
    w = {c: max(len(c), max(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    print("  " + "  ".join(c.ljust(w[c]) for c in cols))
    print("  " + "  ".join("-" * w[c] for c in cols))
    for r in rows:
        print("  " + "  ".join(str(r.get(c, "")).ljust(w[c]) for c in cols))


# ── Entry point ─────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Backtest the Tradebot technical engine")
    ap.add_argument("--symbols", default="", help="Comma list of underlyings (e.g. NIFTY,RELIANCE). Default: full universe")
    ap.add_argument("--min-band", choices=["MILD", "BUY", "STRONG"], default="BUY",
                    help="Minimum consensus band to enter. Default: BUY")
    ap.add_argument("--min-agree", type=int, default=2, help="Min timeframes agreeing. Default: 2")
    ap.add_argument("--sl-atr", type=float, default=1.5, help="Stop distance in ATRs. Default: 1.5")
    ap.add_argument("--t1-r", type=float, default=1.5, help="T1 in R multiples. Default: 1.5")
    ap.add_argument("--t2-r", type=float, default=3.0, help="T2 in R multiples. Default: 3.0")
    ap.add_argument("--cooldown-min", type=int, default=10, help="Re-entry cooldown minutes. Default: 10")
    ap.add_argument("--cost-bps", type=float, default=0.0,
                    help="Round-trip cost (spread+slippage+taxes) in bps of notional, charged per "
                         "trade and converted to R via the trade's ATR risk. 0=gross. ~3=index futures, "
                         "~5 conservative. Default: 0")
    ap.add_argument("--no-self-test", action="store_true", help="Skip the vectorized-vs-_analyze check")
    ap.add_argument("--save", action="store_true", help="Write per-trade ledger + summaries to data/backtest/")
    args = ap.parse_args()

    rank = {"MILD": 1, "BUY": 2, "STRONG": 3}[args.min_band]
    cfg = {"min_rank": rank, "min_agree": args.min_agree, "sl_atr": args.sl_atr,
           "t1_r": args.t1_r, "t2_r": args.t2_r, "cooldown_min": args.cooldown_min}

    # resolve symbols
    rev = {v: k for k, v in LABELS.items()}   # underlying label -> fyers symbol
    if args.symbols.strip():
        wanted = [s.strip().upper() for s in args.symbols.split(",")]
        fy_syms = []
        for s in wanted:
            if s in rev: fy_syms.append(rev[s])
            elif s in LABELS: fy_syms.append(s)
            else: print(f"  WARN unknown symbol '{s}' — skipping")
    else:
        fy_syms = list(ALL_SYMBOLS)

    print("=" * 70)
    print("  TRADEBOT TECHNICAL-ENGINE BACKTEST")
    print("=" * 70)
    print(f"  Symbols   : {len(fy_syms)}")
    print(f"  Entry     : band >= {args.min_band}, >= {args.min_agree} TFs agree")
    print(f"  Geometry  : SL {args.sl_atr} ATR | T1 {args.t1_r}R | T2 {args.t2_r}R | cooldown {args.cooldown_min}m | EOD square-off")
    print(f"  Signal    : multi-TF technical consensus (signals._analyze, layer 1 of 10)")
    print(f"  Cost      : {args.cost_bps:.1f} bps round-trip ({'GROSS' if args.cost_bps==0 else 'NET of cost'})")
    print("=" * 70)

    all_trades: list[pd.DataFrame] = []
    per_symbol_rows: list[dict] = []
    tested = False

    for k, fy in enumerate(fy_syms, 1):
        frames = {}
        for tf in TF_KEYS:
            df = _load_tf(fy, tf)
            if df is None or df.empty:
                frames = {}; break
            frames[tf] = df
        if not frames:
            print(f"  [{k:>3}/{len(fy_syms)}] {LABELS.get(fy, fy):<14} — missing data, skip")
            continue

        if not args.no_self_test and not tested:
            print(f"  Fidelity self-test on {LABELS.get(fy, fy)} 5-min:")
            _self_test(frames["5min"])
            tested = True

        try:
            grid = build_signal_grid(frames)
            tr = simulate(grid, LABELS.get(fy, fy), fy, cfg)
        except Exception as exc:
            print(f"  [{k:>3}/{len(fy_syms)}] {LABELS.get(fy, fy):<14} — ERROR: {exc}")
            continue
        tdf = pd.DataFrame(tr)
        if not tdf.empty and args.cost_bps > 0:
            # round-trip cost in R = (entry * bps) / risk_price, risk_price = sl_atr*ATR
            risk_price = args.sl_atr * tdf["atr"]
            cost_r = (tdf["entry_px"] * args.cost_bps / 1e4) / risk_price.replace(0, np.nan)
            tdf["r_gross"] = tdf["r"]
            tdf["r"] = (tdf["r"] - cost_r).round(3)
        if not tdf.empty:
            all_trades.append(tdf)
            per_symbol_rows.append(summarize(tdf, LABELS.get(fy, fy)))
        n = len(tdf)
        exp = round(tdf["r"].mean(), 3) if n else 0.0
        print(f"  [{k:>3}/{len(fy_syms)}] {LABELS.get(fy, fy):<14} {n:>5} trades  exp {exp:+.3f}R")

    if not all_trades:
        print("\n  No trades generated."); return
    book = pd.concat(all_trades, ignore_index=True)

    print("\n" + "=" * 70)
    print("  PER-SYMBOL  (top / bottom by expectancy)")
    print("=" * 70)
    ps = sorted(per_symbol_rows, key=lambda r: r["expectancy_R"], reverse=True)
    _print_table(ps[:10])
    if len(ps) > 12:
        print("  ...")
        _print_table(ps[-5:])

    print("\n" + "=" * 70)
    print("  BY DIRECTION")
    print("=" * 70)
    _print_table([summarize(book[book["dir"] == d], d) for d in ("CE", "PE")])

    print("\n" + "=" * 70)
    print("  BY EXIT TYPE")
    print("=" * 70)
    _print_table([summarize(book[book["status"] == s], s)
                  for s in ("SL", "T1", "T2", "FLIP", "EOD") if (book["status"] == s).any()])

    print("\n" + "=" * 70)
    print("  AGGREGATE  (whole universe)")
    print("=" * 70)
    _print_table([summarize(book, "ALL")])

    if args.save:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        book.to_parquet(OUT_DIR / "trades.parquet", index=False)
        pd.DataFrame(per_symbol_rows).to_csv(OUT_DIR / "per_symbol.csv", index=False)
        print(f"\n  Saved ledger ({len(book):,} trades) → {OUT_DIR.resolve()}\\trades.parquet")
    print()


if __name__ == "__main__":
    main()
