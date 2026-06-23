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


def predict(feat: dict, m: float = 1.0) -> dict:
    """v0 forecast from a feature vector. m = range half-width in sigma (1.0 ~= 68%)."""
    if not feat:
        return {"has_data": False}
    comp = sum(_W[k] * feat.get(f"f_{k}", 0.0) for k in _W) / _WTOT     # [-1,1]-ish
    p_up = 1.0 / (1.0 + math.exp(-_P_K * comp))
    spot = feat.get("spot", 0.0)
    sig1 = feat.get("sigma1", np.nan)
    sig60 = (max(sig1, _SIG_FLOOR) * math.sqrt(60)) if (sig1 == sig1) else np.nan  # %
    band = (spot * m * sig60 / 100.0) if (sig60 == sig60 and spot) else np.nan
    direction = "BULLISH" if comp > 0.10 else "BEARISH" if comp < -0.10 else "NEUTRAL"
    return {
        "has_data": True, "direction": direction, "composite": round(comp, 3),
        "p_up": round(p_up, 3), "exp_move_pct": (round(sig60, 3) if sig60 == sig60 else None),
        "lo": (round(spot - band, 1) if band == band else None),
        "hi": (round(spot + band, 1) if band == band else None),
        "spot": spot,
    }


def forecast(sym: str, as_of: Optional[datetime.datetime] = None,
             date: Optional[str] = None) -> dict:
    """Live entry point: features + predict for one index."""
    return predict(features(sym, as_of, date))


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
