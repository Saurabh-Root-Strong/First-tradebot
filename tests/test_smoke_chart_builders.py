"""Smoke test: run every chart builder against a REAL captured session.

WHY THIS EXISTS. On 2026-08-11 a NameError in `_build_series_impl` — `day` is bound in
the memoising wrapper, not the impl — passed 92 green tests and was caught only by
running the function by hand afterwards. The offline suite exercises `intraday_tf`
through monkeypatched reads and never once calls `build_series` against real parquet, so
the entire chart data path had zero end-to-end coverage. Anything that only breaks on
contact with actual captured data was invisible.

`data/` is gitignored, so there is no fixture in CI: this SKIPS when no captured day is
present and runs for real on a box that has mirrors. It deliberately picks a PAST day —
today's files are still growing and are being rewritten by `sync_from_vm`, which makes
assertions non-deterministic even though the writes themselves are atomic.

The assertions are the invariants the 2026-08-11 audit established, so this is a
regression net for that whole commit range, not a "does it return something" check.
"""
from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from core.constants import LIVE_DIR, today_iso
from core.session import CLOSE, OPEN

SYM = "NSE:NIFTY50-INDEX"
TFS = (5, 15, 60)
MIN_TICKS = 500            # below this the "day" is a stub, not a session


def _usable_day() -> str | None:
    """Newest PAST day whose ticks AND chain are both substantial."""
    today = today_iso()
    days = sorted({p.name.split("_")[0] for p in LIVE_DIR.glob("*_ticks.parquet")
                   if p.name[0].isdigit()}, reverse=True)
    for d in days:
        if d >= today:
            continue                          # never assert against a growing file
        t, c = LIVE_DIR / f"{d}_ticks.parquet", LIVE_DIR / f"{d}_chain_snapshots.parquet"
        if not (t.exists() and c.exists()):
            continue
        try:
            if len(pd.read_parquet(t)) >= MIN_TICKS and len(pd.read_parquet(c)) >= MIN_TICKS:
                return d
        except Exception:
            continue
    return None


DAY = _usable_day()
pytestmark = pytest.mark.skipif(
    DAY is None, reason="no captured session in data/intraday/live (expected in CI)")


@pytest.fixture(scope="module")
def session_range():
    """True session high/low from the clamped tick stream — computed here independently
    of the builders, so it is a real cross-check and not the same code asserting itself."""
    import footprint_chart as fc
    t = fc._read("ticks", DAY, None, SYM)
    ist = t["ts"].dt.time
    s = t[(ist >= OPEN) & (ist <= CLOSE) & (t["ts"].dt.date == dt.date.fromisoformat(DAY))]
    assert len(s), "clamped tick stream is empty — fixture day is not a real session"
    return float(s["ltp"].min()), float(s["ltp"].max())


@pytest.fixture(scope="module")
def series():
    import footprint_chart as fc
    return {tf: fc.build_series(SYM, tf, DAY, None) for tf in TFS}


# ── build_series ─────────────────────────────────────────────────────────────────
def test_builds_at_every_dropdown_timeframe(series):
    """The bug that started this: an exception on real data. Every timeframe the Charts
    dropdown can produce must build."""
    for tf, d in series.items():
        assert d.get("has_data"), f"tf={tf} returned {d.get('note')!r}"
        assert len(d["ts"]) > 0


def test_every_column_is_bar_aligned(series):
    for tf, d in series.items():
        n = len(d["ts"])
        for k in ("open", "high", "low", "close", "premium", "oi_ce", "oi_pe", "volume",
                  "spot", "iv_atm", "d_oi_ce", "d_oi_pe", "ce_act", "pe_act", "gap_after"):
            assert len(d[k]) == n, f"tf={tf} column {k} has {len(d[k])} rows, ts has {n}"


def test_no_price_from_outside_the_session(series, session_range):
    """The pre-open call auction printed 24722.5 on 2026-08-10 against a true session
    high of 24618.9, and at tf=60 it landed in the first candle."""
    lo, hi = session_range
    for tf, d in series.items():
        highs = [v for v in d["high"] if v is not None]
        lows = [v for v in d["low"] if v is not None]
        assert max(highs) <= hi + 1e-6, f"tf={tf} high {max(highs)} exceeds session high {hi}"
        assert min(lows) >= lo - 1e-6, f"tf={tf} low {min(lows)} below session low {lo}"


def test_volume_telescopes_across_timeframes(series):
    """Per-(strike,side) diff → clip → sum must be frame-independent. Summing a churning
    strike set and then diffing is not, which is why the ordering matters."""
    tot = {tf: sum(v for v in d["volume"] if v) for tf, d in series.items()}
    ref = tot[TFS[0]]
    for tf, v in tot.items():
        assert abs(v - ref) <= max(1.0, ref * 1e-4), f"volume drifted by timeframe: {tot}"


def test_closed_bar_labels_and_values_are_final(series):
    """Prefix invariance: build_series(as_of=T) must equal the prefix of the full day for
    EVERY column. `ce_act`/`pe_act` used to fail this because the deadband was a median
    over the bars currently in view."""
    import footprint_chart as fc
    full = series[15]
    mid = pd.Timestamp(full["ts"][len(full["ts"]) // 2]).to_pydatetime()
    part = fc.build_series(SYM, 15, DAY, mid)
    n = len(part["ts"]) - 1                       # drop the forming bar
    assert n > 2, "cutoff produced too few bars to be a real check"
    for k in ("ts", "open", "high", "low", "close", "premium", "oi_ce", "oi_pe",
              "volume", "d_oi_ce", "d_oi_pe", "ce_act", "pe_act", "iv_atm"):
        assert part[k][:n] == full[k][:n], f"{k} is not final on closed bars"


def test_gap_flags_match_actual_bar_spacing(series):
    for tf, d in series.items():
        ts = [pd.Timestamp(x) for x in d["ts"]]
        step = pd.Timedelta(minutes=tf)
        for i, flagged in enumerate(d["gap_after"][:-1]):
            assert flagged == (ts[i + 1] - ts[i] > step), f"tf={tf} gap flag wrong at bar {i}"
        assert d["gap_after"][-1] is False


# ── the other builders ───────────────────────────────────────────────────────────
@pytest.mark.parametrize("leg", ["near", "next", "far"])
def test_futures_builder_runs_for_every_leg(leg):
    import footprint_chart as fc
    d = fc.build_futures_series(SYM, 15, DAY, None, leg=leg)
    if not d.get("has_data"):
        pytest.skip(f"no futures capture on {DAY}: {d.get('note')}")
    n = len(d["ts"])
    for k in ("open", "high", "low", "close", "volume", "basis", "fut_act", "gap_after"):
        assert len(d[k]) == n, f"leg={leg} column {k} misaligned"
    assert d["has_vol"] is (leg != "far"), "far leg has no volume column in the capture"


def test_strike_builder_runs_on_a_captured_strike():
    import footprint_chart as fc
    anchor, ladder = fc.atm_strikes(SYM, DAY, None)
    if not ladder:
        pytest.skip(f"no strike ladder on {DAY}")
    d = fc.build_strike_series(SYM, 15, ladder[len(ladder) // 2], DAY, None)
    assert d.get("has_data"), d.get("note")
    n = len(d["ts"])
    for k in ("ce_oi", "pe_oi", "ce_prem", "pe_prem", "ce_act", "pe_act"):
        assert len(d[k]) == n


@pytest.mark.parametrize("hhmm,least", [((9, 30), 1), ((14, 0), 2)])
def test_footprint_panel_cells_are_distinct_and_in_session(session_range, hhmm, least):
    """Two cells sharing a measurement is one number counted twice, and the stack line
    reads it as timeframes agreeing. Checked early (where windows can outrun capture and
    must go pending) and mid-session (where all four should be live and distinct).

    SCOPE, honestly: a captured day need not contain the pathology. Mutation-testing this
    against 2026-08-10 showed it stays green with the session clamp removed, because that
    day has no previous-evening straggler row and OI capture starts near enough to the
    bell that the `anchor < first_ts` guard catches everything on its own. The test that
    ISOLATES the clamp is synthetic —
    test_unit_session_clamp.test_clamp_is_load_bearing_when_capture_itself_starts_pre_open
    — and it does go red when the clamp is mutated out. This one's job is end-to-end
    coverage on real parquet, not the clamp proof."""
    import intraday_tf as itf
    from core.constants import IST
    lo, hi = session_range
    asof = dt.datetime.combine(dt.date.fromisoformat(DAY),
                               dt.time(*hhmm)).replace(tzinfo=IST)
    r = itf.analyze(SYM, DAY, asof)
    assert r.get("has_data"), r.get("note")
    cells = r["cells"]
    assert len(cells) >= least, f"expected >={least} timeframes live at {hhmm}"
    tots = [c["d_tot"] for c in cells]
    assert len(tots) == len(set(tots)), f"duplicate windows collapsed onto one number: {tots}"

    # Every window must be priced off a baseline INSIDE the session. Invert the cell's
    # own arithmetic — px = (p1-p0)/p0*100, so p0 = p1/(1+px/100) — and check where it
    # landed. This is the assertion that fails on the 2026-08-11 bug: the "60m" cell was
    # priced off the previous evening's 17:56 tick, and the "15m" off the pre-open
    # auction, both far outside [lo, hi].
    import footprint_chart as fc
    t = fc._read("ticks", DAY, None, SYM)
    ist = t["ts"].dt.time
    s = t[(ist >= OPEN) & (ist <= CLOSE) & (t["ts"].dt.date == dt.date.fromisoformat(DAY))]
    p1 = float(s[s["ts"] <= asof]["ltp"].iloc[-1])
    for c in cells:
        p0 = p1 / (1 + c["px"] / 100.0)
        assert lo - 1e-6 <= p0 <= hi + 1e-6, (
            f"tf={c['tf']} is priced off {p0:.1f}, outside the session range "
            f"[{lo:.1f}, {hi:.1f}] — an overnight or pre-open baseline leaked in")
