"""
overnight_reconciliation.py  —  per-strike "last-night positioning vs today's live
OI change" engine.  The richer successor to intraday_oi_intel.analyze_continuity.

THE IDEA (user's, sharpened)
----------------------------
Last night's EOD chain is a FROZEN map of where positions were built and of what
kind. Today's tick-by-tick OI change (oich = Δ vs prev close) is the LIVE truth of
whether those exact positions are being ADDED to (conviction held) or CLOSED
(conviction abandoned). When a gap CONTRADICTS the overnight positioning and the
contradicted side is being unwound across strikes -> the trend is shifting.

Continuity (the old version) only looked at the TWO heaviest walls and only the
oich sign. This looks at the FULL set of overnight-positioned strikes, derives each
strike's overnight ROLE from EOD OI + its 2-day OI change, and reconciles it against
today's per-strike 4-quadrant (oich + premium + volume).

PER-STRIKE LOGIC
    CE strike heavy with OI = ceiling (bearish overnight positioning).
        today CE OI falling  -> ceiling ABANDONED (covering) -> BULLISH
        today CE OI rising   -> ceiling REINFORCED            -> BEARISH
    PE strike heavy with OI = floor (bullish overnight positioning).
        today PE OI falling  -> floor ABANDONED (covering)    -> BEARISH
        today PE OI rising   -> floor REINFORCED              -> BULLISH
Each contribution is weighted by the overnight OI size (bigger position = louder
when abandoned) and damped/boosted by the premium move (short-covering, oich<0 &
premium up, is the highest-conviction abandonment).

REGIME-SHIFT flag fires when: the gap direction opposes the dominant overnight
bias AND the contradicted walls are net-abandoned.

PURE module (no I/O / no Dash). `eod_baseline()` builds the frozen map from DCM
fno_bhavcopy; `analyze_reconciliation()` is the hot-path-safe analyzer.

DISCIPLINE: this is decision-support until backtest_reconciliation.py shows it
beats the null. Every positioning-direction idea in this project so far has come
back null/contrarian — so it ships display-only and gets wired only on evidence.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# ── Tunables ──────────────────────────────────────────────────────────────────
_MIN_EOD_OI      = 0.0        # set per-index at runtime (fraction of max handles it)
_REL_OI_FRACTION = 0.15       # a strike is "positioned" if its EOD OI >= 15% of the
                              # heaviest leg in the chain (adapts across indices)
_MIN_OICH        = 5_000      # today's |oich| must clear this to count as action
_NEUTRAL_BAND    = 0.5        # |score| below this = no clear edge
_GAP_MIN         = 0.0015     # |gap| above this counts as a directional open


@dataclass
class StrikeRecon:
    strike:    float
    leg:       str            # "CE" | "PE"
    eod_oi:    float          # last night's OI at this leg
    eod_chg:   float          # last night's 2-day OI change (fresh build vs old)
    today_oich: float         # today's OI change vs prev close
    today_ltpch: float        # today's premium change vs prev close
    effect:    str            # REINFORCED | ABANDONED | QUIET
    bias:      int            # +1 bullish / -1 bearish / 0
    contrib:   float          # signed weighted contribution to the net score

    def describe(self) -> str:
        arrow = {"REINFORCED": "=", "ABANDONED": "x", "QUIET": "·"}.get(self.effect, "·")
        return (f"{self.strike:,.0f}{self.leg} {arrow} {self.effect.lower()} "
                f"(EOD OI {self.eod_oi:,.0f}, today {self.today_oich:+,.0f})")


@dataclass
class ReconResult:
    spot:          float
    has_data:      bool = False
    gap:           float = 0.0
    rows:          list[StrikeRecon] = field(default_factory=list)
    score:         float = 0.0          # [-4, +4]
    regime_shift:  bool = False
    overnight_bias: int = 0             # +1 if floors > ceilings overnight, etc.
    pa_dir:        int = 0              # price-action direction at read time (+1/-1/0)
    pa_confirmed:  bool = False         # OI score agrees with price action
    verdict:       str = "NEUTRAL"
    headline:      str = ""
    note:          str = ""


# ── EOD baseline from DCM fno_bhavcopy ──────────────────────────────────────────
def eod_baseline(con, dcm_sym: str, as_of_date=None) -> dict:
    """Frozen last-night per-strike map for the front expiry as of `as_of_date`
    (default = latest available). Returns
        {strike: {"CE": (oi, chg, prem), "PE": (oi, chg, prem)}}
    chg = 2-day OI delta (bhavcopy chg_in_oi is unreliable, so diff two dates).
    """
    date_filter = ""
    params: list = [dcm_sym]
    if as_of_date is not None:
        date_filter = "AND trade_date <= ?"
        params.append(str(as_of_date))
    # two most recent trade dates on/before as_of
    dates = con.execute(
        f"SELECT DISTINCT trade_date FROM fno_bhavcopy WHERE symbol=? {date_filter} "
        f"ORDER BY trade_date DESC LIMIT 2", params).fetchall()
    if not dates:
        return {}
    d0 = dates[0][0]
    d1 = dates[1][0] if len(dates) > 1 else d0
    # front expiry on/after d0
    fe = con.execute(
        "SELECT MIN(expiry_date) FROM fno_bhavcopy WHERE symbol=? AND instrument='OPTIDX' "
        "AND trade_date=? AND expiry_date>=?", [dcm_sym, d0, d0]).fetchone()
    if not fe or fe[0] is None:
        return {}
    front = fe[0]
    rows = con.execute(
        "SELECT trade_date, strike_price, option_type, open_interest, close_price "
        "FROM fno_bhavcopy WHERE symbol=? AND instrument='OPTIDX' AND expiry_date=? "
        "AND trade_date IN (?, ?) AND option_type IN ('CE','PE')",
        [dcm_sym, front, d0, d1]).fetchall()
    cur: dict = {}
    prev: dict = {}
    for td, sp, ot, oi, prem in rows:
        tgt = cur if td == d0 else prev
        tgt[(float(sp), ot)] = (float(oi or 0), float(prem or 0))
    base: dict = {}
    for (sp, ot), (oi, prem) in cur.items():
        p_oi = prev.get((sp, ot), (0.0, 0.0))[0]
        base.setdefault(sp, {})[ot] = (oi, oi - p_oi, prem)
    return base


# ── the analyzer (pure, hot-path safe) ──────────────────────────────────────────
def _classify_effect(leg: str, today_oich: float, today_ltpch: float,
                     floor_oich: float) -> tuple[str, int]:
    """(effect, directional bias) for one positioned leg given today's action."""
    if abs(today_oich) < floor_oich:
        return "QUIET", 0
    rising = today_oich > 0
    if leg == "CE":
        # ceiling: rising = reinforced (bearish); falling = abandoned (bullish)
        return ("REINFORCED", -1) if rising else ("ABANDONED", +1)
    else:
        # floor: rising = reinforced (bullish); falling = abandoned (bearish)
        return ("REINFORCED", +1) if rising else ("ABANDONED", -1)


def analyze_reconciliation(strike_map: dict, spot: float, baseline: dict,
                           gap: float = 0.0, vwap: float = 0.0) -> ReconResult:
    """Reconcile today's live per-strike oich against last night's positioned map.

    strike_map : {strike: {"CE": {oich, ltpch, oi, volume}, "PE": {...}}}  (live)
    baseline   : output of eod_baseline()                                   (frozen)
    gap        : (open - prev_close)/prev_close, for the regime-shift gate.
    vwap       : session VWAP — price-action anchor. When given, the OI verdict
                 is flagged pa_confirmed only if spot sits the right side of VWAP
                 (above for bullish, below for bearish), and a regime shift further
                 requires that confirmation (OI saying 'shift' while price fades =
                 likely trap, so we don't confirm it).
    """
    res = ReconResult(spot=spot, gap=gap)
    if not strike_map or not spot or not baseline:
        res.note = "No live chain / EOD baseline."
        return res

    # significance bars (relative to the heaviest EOD leg)
    max_eod = max((max(v.get("CE", (0,))[0], v.get("PE", (0,))[0]) for v in baseline.values()),
                  default=0.0)
    if max_eod <= 0:
        res.note = "EOD baseline empty."
        return res
    oi_bar = _REL_OI_FRACTION * max_eod
    floor_oich = _MIN_OICH

    rows: list[StrikeRecon] = []
    ceil_w = floor_w = 0.0       # overnight positioning mass per side
    score = 0.0
    for sp, legs in baseline.items():
        live = strike_map.get(sp) or strike_map.get(float(sp)) or {}
        for leg in ("CE", "PE"):
            if leg not in legs:
                continue
            eod_oi, eod_chg, _prem = legs[leg]
            if eod_oi < oi_bar:
                continue                       # not a meaningfully positioned strike
            if leg == "CE":
                ceil_w += eod_oi
            else:
                floor_w += eod_oi
            le = live.get(leg) or {}
            t_oich = float(le.get("oich") or 0)
            t_ltpch = float(le.get("ltpch") or 0)
            effect, bias = _classify_effect(leg, t_oich, t_ltpch, floor_oich)
            if effect == "QUIET" or bias == 0:
                rows.append(StrikeRecon(sp, leg, eod_oi, eod_chg, t_oich, t_ltpch, effect, 0, 0.0))
                continue
            w = min(eod_oi / max_eod, 1.0)     # weight by overnight position size
            # No conviction multiplier: distinguishing short-covering from long
            # unwinding needs the DELTA-ADJUSTED premium residual (Δprem − delta·Δspot,
            # as in footprint_chart.build_strike_series), not raw ltpch — which is
            # delta-contaminated (CE rises on any up-move, PE on any down-move). The
            # old raw-ltpch boost was leg-asymmetric and effectively backwards on CE.
            # This panel is context-only with an unproven directional edge
            # (backtest_reconciliation CIs straddle null), so a fudge multiplier is
            # false precision. Plain bias × overnight-size weight.
            contrib = bias * w
            score += contrib
            rows.append(StrikeRecon(sp, leg, eod_oi, eod_chg, t_oich, t_ltpch, effect, bias, round(contrib, 3)))

    if not rows:
        res.note = "No positioned strikes in range / no action yet."
        return res

    res.has_data = True
    rows.sort(key=lambda r: abs(r.contrib), reverse=True)
    res.rows = rows
    res.score = round(max(-4.0, min(4.0, score * 1.5)), 2)
    res.overnight_bias = 1 if floor_w > ceil_w else (-1 if ceil_w > floor_w else 0)

    # price-action confirmation: spot vs session VWAP must agree with the OI score
    s = res.score
    if vwap and spot:
        res.pa_dir = 1 if spot > vwap else (-1 if spot < vwap else 0)
    score_dir = 1 if s >= _NEUTRAL_BAND else (-1 if s <= -_NEUTRAL_BAND else 0)
    res.pa_confirmed = bool(score_dir != 0 and res.pa_dir == score_dir)

    # regime shift: gap contradicts overnight bias AND contradicted side abandoned
    # AND (when VWAP supplied) price action confirms the new direction — no trap.
    if abs(gap) >= _GAP_MIN and res.overnight_bias != 0:
        gap_dir = 1 if gap > 0 else -1
        contradicts = gap_dir != res.overnight_bias
        abandoned_net = sum(r.contrib for r in rows if r.effect == "ABANDONED")
        pa_ok = (res.pa_dir == gap_dir) if vwap else True
        if contradicts and (abandoned_net * gap_dir) > 0 and pa_ok:
            res.regime_shift = True

    res.verdict = "BULLISH" if s >= _NEUTRAL_BAND else "BEARISH" if s <= -_NEUTRAL_BAND else "NEUTRAL"
    res.headline = _headline(res)
    return res


def _headline(r: ReconResult) -> str:
    top = r.rows[0] if r.rows else None
    bits = []
    if r.regime_shift:
        bits.append("⚡ REGIME SHIFT — gap contradicts overnight, walls abandoning")
    if top is not None and top.bias != 0:
        bits.append(f"loudest {top.strike:,.0f}{top.leg} {top.effect.lower()}")
    head = f"{r.verdict} reconciliation"
    return head + (" — " + " · ".join(bits) if bits else "")


def summary_lines(r: ReconResult) -> list[tuple[str, str]]:
    """(bias, text) lines for a dashboard panel."""
    if not r.has_data:
        return [("neut", r.note or "Overnight reconciliation warming up…")]
    out = [("bull" if r.score > 0 else "bear" if r.score < 0 else "neut", r.headline)]
    for row in r.rows[:8]:
        if row.bias == 0:
            continue
        b = "bull" if row.bias > 0 else "bear"
        out.append((b, row.describe()))
    return out
