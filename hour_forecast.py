"""
hour_forecast.py — next-~60-minute index forecast (HONEST scaffold).

Produces, for one index at instant t, a near-term read:
    direction  + calibrated p(up)      (the UNPROVEN part — index direction has
                                        been null in every test so far)
    range band lo/hi for the next 60m  (the HONEST part — realised vol forecasts
                                        the band well even when direction doesn't)

This is a v0, deliberately TRANSPARENT model — no training on a handful of days
(that would overfit). Direction = the opening-playbook intraday composite squashed
to a probability; range = realised 1-min vol scaled to 60 min. Its only job today
is to emit a forecast every cycle so backtest_hour_forecast.py can accumulate a
real, day-block-validated track record. Swap in learned weights ONLY once the
ledger shows out-of-sample edge over the coin-flip / ATR-band nulls.

Pure + lookahead-free (reads mirrors via opening_playbook + core.mirror_io with
as_of), so it works live and replays any captured day.
"""
from __future__ import annotations

import datetime
import math
from typing import Optional

import numpy as np
import pandas as pd

from core.constants import IST
from core.mirror_io import read_mirror as _read
import opening_playbook as opb

# intraday-only factor weights (drop the EOD factor — its bridge read leaks in
# replay and it carries no measured edge). Same set the playbook backtest used.
_W = {"or": 0.20, "gap": 0.10, "oi": 0.25, "prem": 0.15, "fut": 0.10}
_WTOT = sum(_W.values())
_SIG_FLOOR = 0.02          # min 1-min sigma %
_P_K = 2.5                 # logistic steepness mapping composite -> p(up)
# Range multiplier: 1-min realised vol scaled by sqrt(60) OVER-states the 60-min
# move (microstructure bounce), so a raw 1-sigma band over-covers. 0.47 was the
# empirical 68%-coverage multiplier on the FIRST 116-row ledger — but the 2yr
# multi-horizon audit (backtest_band_horizon_hist.py, ~38k 60m samples across 495
# sessions×4 idx, CROSS-checked on 15 live-tick days) OVERTURNED it: at 0.47 the
# real 60m endpoint coverage is only ~50% (a coin flip), not 68%. The 68th-pctile of
# |move|/base-σ clusters 0.72-0.74 for all four indices, so 0.73 is the honest ±1σ
# multiplier. The L4 loop (calibration_engine) fine-tunes per-index AROUND this base.
_RANGE_M = 0.73


def _sigma1_pct(ticks: pd.DataFrame, upto) -> float | None:
    s = ticks[ticks["ts"] <= pd.Timestamp(upto)].set_index("ts")["ltp"]
    s = s.resample("1min").last().dropna()
    if len(s) < 12:
        return None
    r = s.pct_change().dropna() * 100.0
    return float(r.std()) if len(r) >= 8 and r.std() > 0 else None


def _mom_pct(ticks: pd.DataFrame, t, minutes: int) -> float:
    """Return % over the last `minutes` up to t (price momentum feature)."""
    s = ticks[ticks["ts"] <= pd.Timestamp(t)]
    if len(s) < 2:
        return 0.0
    now = float(s.iloc[-1]["ltp"])
    past = s[s["ts"] <= pd.Timestamp(t) - pd.Timedelta(minutes=minutes)]
    base = float(past.iloc[-1]["ltp"]) if len(past) else float(s.iloc[0]["ltp"])
    return (now / base - 1.0) * 100.0 if base else 0.0


def features(sym: str, as_of: Optional[datetime.datetime] = None,
             date: Optional[str] = None) -> dict:
    """Lookahead-free feature vector at t. {} until enough is captured."""
    p = opb.playbook_index(sym, as_of=as_of, date=date)
    if not p.get("has_data"):
        return {}
    ticks = _read("ticks", date, as_of, sym)
    if ticks is None or len(ticks) < 12:
        return {}
    parts = p.get("parts", {})
    sig1 = _sigma1_pct(ticks, ticks["ts"].max())
    spot = float(ticks.iloc[-1]["ltp"])
    return {
        "sym": sym, "spot": spot,
        **{f"f_{k}": float(parts.get(k, 0.0)) for k in ("or", "gap", "oi", "prem", "fut")},
        "f_mom15": _mom_pct(ticks, ticks["ts"].max(), 15),
        "f_mom30": _mom_pct(ticks, ticks["ts"].max(), 30),
        "f_pcr": float(p.get("pcr") or 0.0),
        "f_iv": float(p.get("iv_now") or 0.0),
        "sigma1": sig1 if sig1 else np.nan,
        "conviction": p.get("conviction", 0),
    }


def predict(feat: dict, m: float = _RANGE_M) -> dict:
    """v0 forecast from a feature vector. m = calibrated band half-width multiplier."""
    if not feat:
        return {"has_data": False}
    comp = sum(_W[k] * feat.get(f"f_{k}", 0.0) for k in _W) / _WTOT     # [-1,1]-ish
    p_up = 1.0 / (1.0 + math.exp(-_P_K * comp))
    spot = feat.get("spot", 0.0)
    sig1 = feat.get("sigma1", np.nan)
    sig60 = (max(sig1, _SIG_FLOOR) * math.sqrt(60)) if (sig1 == sig1) else np.nan  # raw %
    band_pct = (m * sig60) if (sig60 == sig60) else np.nan                          # calibrated band %
    band = (spot * band_pct / 100.0) if (band_pct == band_pct and spot) else np.nan
    direction = "BULLISH" if comp > 0.10 else "BEARISH" if comp < -0.10 else "NEUTRAL"
    return {
        "has_data": True, "direction": direction, "composite": round(comp, 3),
        "p_up": round(p_up, 3), "exp_move_pct": (round(band_pct, 3) if band_pct == band_pct else None),
        "lo": (round(spot - band, 1) if band == band else None),
        "hi": (round(spot + band, 1) if band == band else None),
        "spot": spot,
    }


def forecast(sym: str, as_of: Optional[datetime.datetime] = None,
             date: Optional[str] = None) -> dict:
    """Live entry point: features + predict for one index."""
    return predict(features(sym, as_of, date))


# ── HONEST band coverage — the band's REAL measured accuracy, self-updating ───────
# backtest_band_horizon.py persists per-(index,horizon) endpoint coverage to
# data/calibration/band_coverage.json. Surfacing it next to the band stops the UI
# claiming a flat "~70%" when e.g. the 15m / BANK / FIN bands measure ~50%. Cached by
# file mtime so it refreshes when the ledger is regenerated without a restart.
from core.constants import DATA_DIR  # noqa: E402
_LEDGER = DATA_DIR / "calibration" / "band_coverage.json"
_COV_CACHE: dict = {}


def band_coverage(sym: str, horizon_min: int) -> dict:
    """{'cover': pct|None, 'n': int, 'conf': tag} for this (index, horizon) from the ledger.
    Nearest available horizon if the exact one isn't logged. conf: ok/soft/low/thin/none."""
    try:
        mt = _LEDGER.stat().st_mtime_ns
    except OSError:
        return {"cover": None, "n": 0, "conf": "none"}
    led = _COV_CACHE.get(mt)
    if led is None:
        import json
        try:
            led = json.loads(_LEDGER.read_text())
        except Exception:
            led = {}
        _COV_CACHE.clear(); _COV_CACHE[mt] = led
    cells = (led.get("by_index", {}) or {}).get(sym, {})
    if not cells:
        return {"cover": None, "n": 0, "conf": "none"}
    key = str(horizon_min) if str(horizon_min) in cells else \
        min(cells, key=lambda k: abs(int(k) - horizon_min))
    c = cells.get(key, {}) or {}
    cover, n = c.get("endpoint"), int(c.get("n", 0))
    if cover is None or n < 20:
        conf = "thin"
    elif cover >= 0.66:
        conf = "ok"
    elif cover >= 0.58:
        conf = "soft"
    else:
        conf = "low"
    return {"cover": cover, "n": n, "conf": conf}


# ── LEARNED band multiplier — the L4 calibration loop's live output ───────────────
# calibration_engine.py recalibrates a per-(index,horizon) multiplier (shrunk toward 1.0
# on thin data) so each band self-tunes toward the 68% coverage target. Applied on TOP of
# the base band. 1.0 = no change / no data. Cached by mtime like the coverage ledger.
_MULT = DATA_DIR / "calibration" / "band_multipliers.json"
_MULT_CACHE: dict = {}


def band_multiplier(sym: str, horizon_min: int) -> float:
    """Learned band-width multiplier for (index, horizon); 1.0 when absent/thin."""
    try:
        mt = _MULT.stat().st_mtime_ns
    except OSError:
        return 1.0
    d = _MULT_CACHE.get(mt)
    if d is None:
        import json
        try:
            d = json.loads(_MULT.read_text())
        except Exception:
            d = {}
        _MULT_CACHE.clear(); _MULT_CACHE[mt] = d
    cells = (d.get("by_index", {}) or {}).get(sym, {})
    if not cells:
        return 1.0
    key = str(horizon_min) if str(horizon_min) in cells else \
        min(cells, key=lambda k: abs(int(k) - horizon_min))
    try:
        return float(cells.get(key, {}).get("mult", 1.0))
    except Exception:
        return 1.0


if __name__ == "__main__":
    import sys
    from core.constants import INDEX_SYMBOLS
    d = sys.argv[1] if len(sys.argv) > 1 else None
    as_of = None
    if len(sys.argv) > 2 and d:
        hh, mm = sys.argv[2].split(":")
        as_of = datetime.datetime.fromisoformat(d).replace(hour=int(hh), minute=int(mm), tzinfo=IST)
    for s in INDEX_SYMBOLS:
        f = forecast(s, as_of=as_of, date=d)
        if f.get("has_data"):
            print(f"  {s:26s} {f['direction']:8s} p_up={f['p_up']:.2f}  "
                  f"±{f['exp_move_pct']}%  [{f['lo']}, {f['hi']}]")
        else:
            print(f"  {s:26s} warming up")
