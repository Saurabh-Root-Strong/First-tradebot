"""
intraday_oi_intel.py  --  Per-strike intraday OI-Dynamics intelligence.

THE question this answers (live, from ~9:30 once the chain has moved):
    "Which strike is seeing fresh positions CREATED, which is being COVERED /
     UNWOUND, where is the ceiling being written, where is the floor being built?"

For each strike's CALL and PUT leg we read the live Fyers chain fields:
    oich  = OI change vs previous close   (today's net buildup at this strike)
    ltpch = premium change vs prev close  (demand/supply on that option)
and classify the footprint with the institutional 4-quadrant OI-Premium matrix
(identical to the Daily_Cash_Market EOD matrix, applied intraday):

    OI up   + premium up    -> Long Buildup   (fresh BUYING of the option)
    OI up   + premium down  -> Short Buildup   = WRITING (fresh selling)
    OI down + premium up    -> Short Covering  (writers buying back)
    OI down + premium down  -> Long Unwinding  (buyers exiting)

Directional translation (the edge):
    CALL writing      -> ceiling / resistance being built          (bearish)
    CALL buying       -> bullish demand                            (bullish)
    CALL short cover   -> shorts fleeing, squeeze fuel             (bullish)
    CALL unwinding    -> upside bets abandoned                     (mild bearish)
    PUT  writing      -> floor / support being built              (bullish)
    PUT  buying       -> downside protection / bearish bets        (bearish)
    PUT  short cover   -> floor crumbling                          (bearish)
    PUT  unwinding    -> hedges removed                            (mild bullish)

This module is PURE (no Dash / no I/O) so it is trivially unit-testable; the
dashboard imports analyze_oi_dynamics() and renders OIDynamics.

NOTE on ITM strikes: deep-ITM premium moves mostly with the underlying (delta),
so the buy/cover labels there are delta-driven, not pure positioning. The ranked
map is dominated by ATM/OTM strikes (where writers and buyers actually position),
so this rarely distorts the read; ITM rows are flagged with low intensity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# ── Tunables ──────────────────────────────────────────────────────────────────
# A leg counts as "active" only if its OI moved at least this many contracts AND
# at least this fraction of the largest OI move in the chain (so we adapt across
# NIFTY / BANKNIFTY / FINNIFTY / MIDCAP without hard-coding per-index floors).
_MIN_ABS_OI_CHG   = 8_000     # absolute floor (contracts) — kills micro-noise
_REL_OI_FRACTION  = 0.12      # must be >=12% of the chain's biggest OI move
_PREM_DEADZONE    = 0.05      # |ltpch| <= this (Rs) treated as flat premium
_NEUTRAL_BAND     = 0.5       # |net_bias_score| below this = no clear OI edge

# Action labels
LB = "Long Buildup"
SB = "Writing"            # short buildup = option writing
SC = "Short Covering"
LU = "Long Unwinding"
NEUTRAL = "Neutral"

# Per-(leg, action) directional sign and a base weight (writing/buildup are the
# high-conviction footprints; covering/unwinding are secondary). +bullish/-bearish.
_BIAS = {
    ("CE", LB): (+1.0, 0.8),   # call buying      -> bullish
    ("CE", SB): (-1.0, 1.0),   # call writing     -> bearish (ceiling)
    ("CE", SC): (+1.0, 0.7),   # call short cover  -> bullish (squeeze)
    ("CE", LU): (-1.0, 0.4),   # call unwinding    -> mild bearish
    ("PE", LB): (-1.0, 0.8),   # put buying       -> bearish
    ("PE", SB): (+1.0, 1.0),   # put writing      -> bullish (floor)
    ("PE", SC): (-1.0, 0.7),   # put short cover   -> bearish (floor cracks)
    ("PE", LU): (+1.0, 0.4),   # put unwinding     -> mild bullish
}

_ACTION_TAG = {LB: "🟢 LB", SB: "🔴 WR", SC: "🔵 SC", LU: "🟠 LU", NEUTRAL: "⚪"}


@dataclass
class StrikeFlow:
    strike:    float
    leg:       str            # "CE" | "PE"
    action:    str            # LB / SB / SC / LU / NEUTRAL
    oich:      float          # OI change (contracts)
    ltpch:     float          # premium change (Rs)
    oi:        float          # current OI
    volume:    float
    bias:      int            # +1 bullish / -1 bearish / 0 neutral
    intensity: float          # 0..1 — relative size of this OI move
    moneyness: str            # "ITM" | "ATM" | "OTM"

    @property
    def tag(self) -> str:
        return _ACTION_TAG.get(self.action, "⚪")

    def describe(self) -> str:
        side = "CE" if self.leg == "CE" else "PE"
        return (f"{self.strike:,.0f} {side} {self.tag}  "
                f"OI {self.oich:+,.0f} · prem {self.ltpch:+.1f}")


@dataclass
class OIDynamics:
    spot:            float
    atm:             float
    has_data:        bool = False
    flows:           list[StrikeFlow] = field(default_factory=list)   # all active legs, ranked by |oich|
    creating:        list[StrikeFlow] = field(default_factory=list)   # OI rising (positions forming)
    closing:         list[StrikeFlow] = field(default_factory=list)   # OI falling (covering / unwinding)
    ceiling_strike:  Optional[float] = None    # heaviest fresh call writing
    floor_strike:    Optional[float] = None    # heaviest fresh put writing
    net_bias_score:  float = 0.0               # [-4, +4] — feeds composite later
    verdict:         str = "NEUTRAL"
    headline:        str = ""
    note:            str = ""


def _classify(leg: str, oich: float, ltpch: float) -> str:
    """4-quadrant OI-premium classification for one option leg."""
    oi_up = oich > 0
    pr_up = ltpch > _PREM_DEADZONE
    pr_dn = ltpch < -_PREM_DEADZONE
    if oi_up:
        if pr_up:  return LB
        if pr_dn:  return SB
        return SB            # OI up, premium flat -> writing pressure (premium capped)
    else:
        if pr_up:  return SC
        if pr_dn:  return LU
        return SC            # OI down, premium flat -> covering (no demand to lift it)


def _moneyness(strike: float, spot: float, leg: str) -> str:
    if abs(strike - spot) <= max(spot * 0.001, 1e-9):
        return "ATM"
    if leg == "CE":
        return "ITM" if strike < spot else "OTM"
    return "ITM" if strike > spot else "OTM"


def analyze_oi_dynamics(strike_map: dict, spot: float,
                        top_n: int = 6) -> OIDynamics:
    """
    Build the per-strike OI-Dynamics map from a live Fyers strike_map.

    strike_map: {strike: {"CE": <fyers entry>, "PE": <fyers entry>}}
                each entry has oich, ltpch, oi, volume.
    Returns OIDynamics (has_data=False when the chain hasn't moved yet).
    """
    res = OIDynamics(spot=spot, atm=spot)
    if not strike_map or not spot:
        res.note = "No chain / spot."
        return res

    strikes = sorted(strike_map.keys())
    atm = min(strikes, key=lambda s: abs(s - spot))
    res.atm = atm

    # Largest absolute OI move in the chain — sets the relative significance bar.
    max_abs = 0.0
    for sp in strikes:
        for leg in ("CE", "PE"):
            oc = (strike_map[sp].get(leg) or {}).get("oich") or 0
            max_abs = max(max_abs, abs(float(oc)))
    if max_abs <= 0:
        res.note = "OI unchanged — chain has not moved yet (pre-open / settled)."
        return res

    sig_floor = max(_MIN_ABS_OI_CHG, _REL_OI_FRACTION * max_abs)

    flows: list[StrikeFlow] = []
    for sp in strikes:
        for leg in ("CE", "PE"):
            e = strike_map[sp].get(leg) or {}
            oich = float(e.get("oich") or 0)
            if abs(oich) < sig_floor:
                continue
            ltpch  = float(e.get("ltpch") or 0)
            action = _classify(leg, oich, ltpch)
            sign, w = _BIAS.get((leg, action), (0.0, 0.0))
            intensity = min(abs(oich) / max_abs, 1.0)
            flows.append(StrikeFlow(
                strike=float(sp), leg=leg, action=action,
                oich=oich, ltpch=ltpch,
                oi=float(e.get("oi") or 0), volume=float(e.get("volume") or 0),
                bias=int(sign) if sign else 0,
                intensity=round(intensity, 3),
                moneyness=_moneyness(float(sp), spot, leg),
            ))

    if not flows:
        res.note = "No significant OI buildup yet — positions still light."
        return res

    res.has_data = True
    flows.sort(key=lambda f: abs(f.oich), reverse=True)
    res.flows    = flows[:top_n * 2]
    res.creating = [f for f in flows if f.oich > 0][:top_n]
    res.closing  = [f for f in flows if f.oich < 0][:top_n]

    # Ceiling / floor = heaviest FRESH writing on each side.
    ce_writes = [f for f in flows if f.leg == "CE" and f.action == SB]
    pe_writes = [f for f in flows if f.leg == "PE" and f.action == SB]
    if ce_writes:
        res.ceiling_strike = max(ce_writes, key=lambda f: f.oich).strike
    if pe_writes:
        res.floor_strike = max(pe_writes, key=lambda f: f.oich).strike

    # ── Net bias score [-4, +4] ──────────────────────────────────────────────
    # Each active leg contributes sign * weight * intensity; ITM legs are damped
    # (their premium move is delta-driven, not positioning). Normalised so a few
    # decisive writes don't saturate instantly.
    raw = 0.0
    for f in flows:
        sign, w = _BIAS.get((f.leg, f.action), (0.0, 0.0))
        damp = 0.4 if f.moneyness == "ITM" else 1.0
        raw += sign * w * f.intensity * damp
    res.net_bias_score = round(max(-4.0, min(4.0, raw * 1.5)), 2)

    # ── Verdict + headline ────────────────────────────────────────────────────
    s = res.net_bias_score
    if s >= _NEUTRAL_BAND:
        res.verdict = "BULLISH"
    elif s <= -_NEUTRAL_BAND:
        res.verdict = "BEARISH"
    else:
        res.verdict = "NEUTRAL"

    res.headline = _headline(res)
    return res


def _headline(d: OIDynamics) -> str:
    bits: list[str] = []
    if d.ceiling_strike is not None:
        bits.append(f"ceiling forming {d.ceiling_strike:,.0f}CE")
    if d.floor_strike is not None:
        bits.append(f"floor forming {d.floor_strike:,.0f}PE")
    top = d.flows[0] if d.flows else None
    if top is not None:
        verb = {LB: "buying", SB: "writing", SC: "short-covering", LU: "unwinding"}.get(top.action, "")
        bits.append(f"heaviest: {top.strike:,.0f}{top.leg} {verb} ({top.oich:+,.0f})")
    head = f"{d.verdict} OI flow"
    return head + (" — " + " · ".join(bits) if bits else "")


def summary_lines(d: OIDynamics) -> list[tuple[str, str]]:
    """(bias, text) lines for the dashboard panel. bias in {bull,bear,neut}."""
    if not d.has_data:
        return [("neut", d.note or "OI intelligence warming up…")]
    out: list[tuple[str, str]] = [
        ("bull" if d.net_bias_score > 0 else "bear" if d.net_bias_score < 0 else "neut",
         d.headline)
    ]
    for f in d.flows[:6]:
        b = "bull" if f.bias > 0 else "bear" if f.bias < 0 else "neut"
        verb = {LB: "long buildup", SB: "writing", SC: "short covering", LU: "unwinding"}.get(f.action, "")
        out.append((b, f"{f.strike:,.0f} {f.leg}: {verb}  (OI {f.oich:+,.0f} · prem {f.ltpch:+.1f})"))
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# COMPONENT B — Overnight → Morning Continuity
# ═══════════════════════════════════════════════════════════════════════════════
# Last night DCM recorded the key walls (top_call_strike = ceiling, top_put_strike
# = floor) and max pain. This morning we read the LIVE oich at those EXACT strikes:
#   ceiling + more call writing (OI up) -> REINFORCED  -> bearish follow-through
#   ceiling + call covering   (OI down) -> LIFTING     -> bullish (wall removed, squeeze)
#   floor   + more put writing (OI up)  -> REINFORCED  -> bullish follow-through
#   floor   + put covering    (OI down) -> CRUMBLING   -> bearish (floor breaks)
# This is the highest-edge morning read: are dealers defending or abandoning the
# overnight structure?

@dataclass
class ContinuitySignal:
    has_data:    bool = False
    resistance:  Optional[float] = None      # last night's call wall (ceiling)
    support:     Optional[float] = None      # last night's put wall (floor)
    max_pain:    Optional[float] = None
    res_action:  str = ""                    # REINFORCED | LIFTING | QUIET | OUT_OF_RANGE
    sup_action:  str = ""                    # REINFORCED | CRUMBLING | QUIET | OUT_OF_RANGE
    res_oich:    float = 0.0
    sup_oich:    float = 0.0
    score:       float = 0.0                 # [-4, +4]
    verdict:     str = "NEUTRAL"
    headline:    str = ""
    lines:       list = field(default_factory=list)
    note:        str = ""


def _nearest_strike(strike_map: dict, target: float) -> tuple[Optional[float], bool]:
    """Nearest chain strike to `target`; in_range=False if it's >1.5 gaps away."""
    if not strike_map or target is None:
        return None, False
    strikes = sorted(strike_map.keys())
    if not strikes:
        return None, False
    nearest = min(strikes, key=lambda s: abs(s - target))
    gap = min((strikes[i + 1] - strikes[i] for i in range(len(strikes) - 1)), default=0) or 1
    return float(nearest), abs(nearest - target) <= 1.5 * gap


def analyze_continuity(strike_map: dict, spot: float, levels: dict) -> ContinuitySignal:
    """Diff this morning's live OI against last night's DCM walls."""
    res = ContinuitySignal()
    if not strike_map or not spot or not levels:
        res.note = "No EOD reference (daily context not loaded)."
        return res

    rstrike = levels.get("top_call_strike")
    sstrike = levels.get("top_put_strike")
    mp      = levels.get("max_pain_price")
    res.resistance, res.support, res.max_pain = rstrike, sstrike, mp

    if rstrike is None and sstrike is None:
        res.note = "No EOD walls recorded for this index."
        return res

    floor_ = _MIN_ABS_OI_CHG
    score  = 0.0
    lines: list[tuple[str, str]] = []

    # ── Resistance (last night's call wall) ───────────────────────────────────
    if rstrike is not None:
        sp, in_range = _nearest_strike(strike_map, rstrike)
        ce = (strike_map.get(sp) or {}).get("CE") or {} if sp is not None else {}
        oich = float(ce.get("oich") or 0)
        res.res_oich = oich
        if not in_range:
            res.res_action = "OUT_OF_RANGE"
            lines.append(("neut", f"Overnight ceiling {rstrike:,.0f}CE now outside the chain — spot has run away from it"))
        elif oich > floor_:
            res.res_action = "REINFORCED"; score -= 1.5
            lines.append(("bear", f"Ceiling {sp:,.0f}CE REINFORCED — more call writing (+{oich:,.0f}) → wall defended, capped upside"))
        elif oich < -floor_:
            res.res_action = "LIFTING"; score += 2.0
            lines.append(("bull", f"Ceiling {sp:,.0f}CE LIFTING — calls covering ({oich:,.0f}) → overnight wall removed, room to run"))
        else:
            res.res_action = "QUIET"
            lines.append(("neut", f"Ceiling {sp:,.0f}CE quiet — overnight resistance intact, no fresh action"))

    # ── Support (last night's put wall) ───────────────────────────────────────
    if sstrike is not None:
        sp, in_range = _nearest_strike(strike_map, sstrike)
        pe = (strike_map.get(sp) or {}).get("PE") or {} if sp is not None else {}
        oich = float(pe.get("oich") or 0)
        res.sup_oich = oich
        if not in_range:
            res.sup_action = "OUT_OF_RANGE"
            lines.append(("neut", f"Overnight floor {sstrike:,.0f}PE now outside the chain — spot has run away from it"))
        elif oich > floor_:
            res.sup_action = "REINFORCED"; score += 1.5
            lines.append(("bull", f"Floor {sp:,.0f}PE REINFORCED — more put writing (+{oich:,.0f}) → support defended, cushion strong"))
        elif oich < -floor_:
            res.sup_action = "CRUMBLING"; score -= 2.0
            lines.append(("bear", f"Floor {sp:,.0f}PE CRUMBLING — puts covering/unwinding ({oich:,.0f}) → support breaking, breakdown risk"))
        else:
            res.sup_action = "QUIET"
            lines.append(("neut", f"Floor {sp:,.0f}PE quiet — overnight support intact"))

    # ── Max-pain gravity context ──────────────────────────────────────────────
    if mp and spot:
        diff = (spot - mp) / spot * 100
        if abs(diff) > 0.25:
            lines.append(("neut",
                f"Spot {spot:,.0f} {'above' if diff > 0 else 'below'} max pain {mp:,.0f} "
                f"({diff:+.2f}%) — expiry gravity pulls {'down' if diff > 0 else 'up'}"))

    res.has_data = True
    res.score    = round(max(-4.0, min(4.0, score)), 2)
    res.lines    = lines
    res.verdict, res.headline = _continuity_verdict(res)
    return res


def _continuity_verdict(c: ContinuitySignal) -> tuple[str, str]:
    s = c.score
    reversal = c.res_action == "LIFTING" or c.sup_action == "CRUMBLING"
    confirm  = c.res_action == "REINFORCED" or c.sup_action == "REINFORCED"
    if s >= _NEUTRAL_BAND:
        kind = "REVERSAL ↑" if c.res_action == "LIFTING" else "CONFIRMATION"
        verdict = f"BULLISH {kind}"
    elif s <= -_NEUTRAL_BAND:
        kind = "REVERSAL ↓" if c.sup_action == "CRUMBLING" else "CONFIRMATION"
        verdict = f"BEARISH {kind}"
    elif c.res_action == "REINFORCED" and c.sup_action == "REINFORCED":
        verdict = "RANGE — both walls held"
    else:
        verdict = "NEUTRAL"
    return verdict, f"{verdict}  (overnight structure {'fading' if reversal and not confirm else 'holding' if confirm and not reversal else 'mixed'})"


def continuity_lines(c: ContinuitySignal) -> list[tuple[str, str]]:
    if not c.has_data:
        return [("neut", c.note or "Overnight continuity warming up…")]
    head = ("bull" if c.score > 0 else "bear" if c.score < 0 else "neut", c.headline)
    return [head] + list(c.lines)
