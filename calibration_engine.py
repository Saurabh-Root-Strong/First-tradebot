"""
calibration_engine.py — L4 of the intraday system: the MEMORY / LEARNING loop.

The one honest form of "learn from today's data, improve tomorrow": recalibrate the
SIGNAL-BEARING knobs (never a directional arrow) so the risk map self-tunes. v1 target =
the RANGE-band multiplier per (index, horizon): the coverage ledger (backtest_band_horizon
.py) says how well each band actually covered; this nudges each band toward the 68% target.

WHY SHRINKAGE (the overfit guard): with only a few captured days per cell, the raw
"multiply the band by target/observed" correction would chase noise (an adverse trend week
would blow the band out). So the applied correction is SHRUNK toward "leave it alone" by
weight = n / (n + N0): near-zero nudge when data is thin, full correction only as the
sample grows. This is the Bayesian/James-Stein discipline a real desk uses to act on thin
data without overfitting. Cells below a hard N floor don't move at all.

Output: data/calibration/band_multipliers.json = {sym: {horizon: {mult, ...provenance}}},
read live by hour_forecast.band_multiplier() and applied on top of the base band. mult=1.0
means "no change". Idempotent + provenance-stamped so every nightly run is auditable.

    .venv\\Scripts\\python.exe calibration_engine.py            # learn from current ledger
    .venv\\Scripts\\python.exe calibration_engine.py --dry-run  # show, don't write
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from statistics import NormalDist

from core.constants import DATA_DIR, IST

_CAL_DIR = DATA_DIR / "calibration"
_LEDGER = _CAL_DIR / "band_coverage.json"          # written by backtest_band_horizon.py
_OUT = _CAL_DIR / "band_multipliers.json"          # read live by hour_forecast

TARGET_COV = 0.68        # the band's design coverage (±1σ destination)
N0 = 200                 # shrinkage prior strength — full correction needs ~N0 samples
N_FLOOR = 30             # below this many samples a cell does not move at all
MULT_CLAMP = (0.70, 1.60)  # never let one learning step distort the band beyond this
_Z = NormalDist()


def _k(cov: float) -> float:
    """Half-normal quantile: coverage c -> the σ-multiple k s.t. P(|Z|<k)=c."""
    cov = min(max(cov, 0.05), 0.95)
    return _Z.inv_cdf((1.0 + cov) / 2.0)


def learn(dry: bool = False) -> dict:
    if not _LEDGER.exists():
        print(f"no coverage ledger at {_LEDGER} — run backtest_band_horizon.py first")
        return {}
    led = json.loads(_LEDGER.read_text())
    by_index = led.get("by_index", {}) or {}
    prev = json.loads(_OUT.read_text()) if _OUT.exists() else {}
    prev_m = prev.get("by_index", {}) if isinstance(prev, dict) else {}

    k_target = _k(TARGET_COV)
    out = {"updated": dt.datetime.now(IST).isoformat(), "target_cov": TARGET_COV,
           "n0": N0, "n_floor": N_FLOOR, "source_days": led.get("days"), "by_index": {}}
    print(f"  LEARN band multipliers — target {TARGET_COV:.0%}, prior N0={N0}, "
          f"floor n≥{N_FLOOR}   (source days: {led.get('days')})")
    print(f"  {'index':12}{'H':>4}{'n':>6}{'cover':>7}{'raw×':>7}{'shrink':>7}"
          f"{'prev×':>7}{'new×':>7}")
    for sym, cells in by_index.items():
        out["by_index"].setdefault(sym, {})
        for hz, c in (cells or {}).items():
            cov, n = c.get("endpoint"), int(c.get("n", 0))
            prior_mult = float((prev_m.get(sym, {}) or {}).get(hz, {}).get("mult", 1.0))
            if cov is None or n < N_FLOOR:
                mult = prior_mult                      # hold — too thin to move
                raw = shrink = float("nan")
            else:
                raw = k_target / _k(cov)               # multiplier that would hit target now
                shrink = n / (n + N0)                   # trust ∝ sample size
                target_mult = 1.0 + shrink * (raw - 1.0)
                # blend with the standing multiplier (EWMA on prior) so nightly runs are stable
                mult = 0.5 * prior_mult + 0.5 * target_mult
                mult = min(max(mult, MULT_CLAMP[0]), MULT_CLAMP[1])
            out["by_index"][sym][hz] = {"mult": round(mult, 4), "n": n,
                                        "cover": cov, "updated": out["updated"]}
            rs = f"{raw:6.3f}" if raw == raw else "   —  "
            ss = f"{shrink:6.2f}" if shrink == shrink else "   —  "
            cv = f"{cov:6.1%}" if cov is not None else "   — "
            print(f"  {sym:12}{hz:>4}{n:>6}{cv:>7}{rs:>7}{ss:>7}"
                  f"{prior_mult:>7.3f}{mult:>7.3f}")
    if dry:
        print("\n  --dry-run: not written")
    else:
        _CAL_DIR.mkdir(parents=True, exist_ok=True)
        _OUT.write_text(json.dumps(out, indent=2))
        print(f"\n  → {_OUT}")
    return out


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    learn(dry=a.dry_run)
