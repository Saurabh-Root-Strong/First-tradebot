"""
trade_setup.py  --  Quant-grade intraday option trade engine.

9-layer signal system (weighted for intraday):
  Layer 1  Technical      25%   EMA 9/21/50 · MACD · RSI 14 · VWAP · Bollinger
  Layer 2  OI Flow        10%   4-quadrant OI change vs price · ATM activity · walls
  Layer 3  Velocity       18%   OI/IV/wall/PCR velocity over session history
  Layer 4  Institutional  10%   FII futures net · FAO smart-money vs retail divergence · PE
  Layer 5  Futures         8%   Live basis · term structure · institutional direction
  Layer 6  IV              6%   ATM IV level · put/call skew
  Layer 7  PCR             4%   Contrarian/momentum PCR bands
  Layer 8  Max Pain        2%   Expiry gravity (low weight for intraday)
  Layer 9  Daily Context  17%   EOD structural setup from Daily_Cash_Market DB
                                (prediction engine · HMM regime · FII trend ·
                                 market breadth · VIX regime · EOD OI footprint)

Each layer returns a score in [-4, +4] range.
Positive = bullish, Negative = bearish.
Layer 9 degrades to 0 gracefully when Daily_Cash_Market DB is unavailable.
"""

import datetime
from signals import fetch_ohlcv, _analyze, TIMEFRAMES, _fmt_oi
from market_data_bridge import get_institutional_bias
try:
    from intraday_store import oi_store, session_phase, session_strategy
    _INTRADAY_STORE_AVAILABLE = True
except Exception:
    _INTRADAY_STORE_AVAILABLE = False

try:
    from daily_context_bridge import get_bridge as _get_bridge
    _BRIDGE_AVAILABLE = True
except Exception:
    _BRIDGE_AVAILABLE = False

try:
    from intraday_shock import analyze_shock
    _SHOCK_AVAILABLE = True
except Exception:
    _SHOCK_AVAILABLE = False

IST = datetime.timezone(datetime.timedelta(hours=5, minutes=30))


# ── DB helper (non-blocking; lazy-import avoids circular deps) ────────────────

def _idb_write_setup(
    sym: str, tf: str, signal: str,
    composite: float, confidence: float, direction: str,
    l1: float, l2: float, l3: float, l4: float, l5: float,
    l6: float, l7: float, l8: float, l9: float,
    agreement: int, phase: str, spot: float, atm_iv: float,
) -> None:
    try:
        from intraday_db import idb
        idb.write_setup(
            sym, tf, signal,
            composite, int(confidence), direction,
            l1, l2, l3, l4, l5, l6, l7, l8, l9,
            agreement, phase, spot, atm_iv,
        )
    except Exception:
        pass


# ── Timeframe profiles ────────────────────────────────────────────────────────
TF_PROFILES = {
    "5min":  {"label": "5 Min  (Intraday Scalp)",    "trade": "INTRADAY SCALP",
              "sl_pct": 0.30, "t1_pct": 0.50, "t2_pct": 0.80,
              "expiry_rank": 0,   "delta_high": 0.50, "delta_mid": 0.38, "delta_low": 0.28},
    "15min": {"label": "15 Min (Intraday Swing)",    "trade": "INTRADAY SWING",
              "sl_pct": 0.32, "t1_pct": 0.55, "t2_pct": 0.90,
              "expiry_rank": 0,   "delta_high": 0.48, "delta_mid": 0.35, "delta_low": 0.26},
    "60min": {"label": "1 Hour (BTST / Positional)", "trade": "BTST",
              "sl_pct": 0.35, "t1_pct": 0.65, "t2_pct": 1.20,
              "expiry_rank": 1,   "delta_high": 0.45, "delta_mid": 0.32, "delta_low": 0.24},
    "daily": {"label": "Daily  (Swing Positional)",  "trade": "POSITIONAL",
              "sl_pct": 0.40, "t1_pct": 0.80, "t2_pct": 1.50,
              "expiry_rank": "M", "delta_high": 0.42, "delta_mid": 0.30, "delta_low": 0.22},
}

LOT_SIZES = {
    "NSE:NIFTY50-INDEX":    75,
    "NSE:NIFTYBANK-INDEX":  15,
    "NSE:FINNIFTY-INDEX":   40,
    "NSE:MIDCPNIFTY-INDEX": 50,
}

# ── Layer 2: OI Flow Analysis ─────────────────────────────────────────────────
def _oi_flow(strike_map: dict, spot: float, tech_score: float) -> tuple[float, list]:
    """
    4-quadrant OI interpretation:
      Price UP   + Call OI falling  → short covering     = bullish but weak (shorts exiting)
      Price UP   + Put  OI rising   → fresh put writing   = STRONG bullish (dealers selling puts = expect up)
      Price UP   + Put  OI falling  → put long closing    = mild bullish
      Price DOWN + Call OI rising   → fresh call writing  = STRONG bearish (dealers selling calls = expect down)
      Price DOWN + Call OI falling  → call long closing   = mild bearish
      Price DOWN + Put  OI rising   → fresh put buying    = bearish but weak
      Put OI building > Call OI building overall → net bullish writing bias
      Call OI building > Put OI building overall → net bearish writing bias
    """
    if not strike_map or not spot:
        return 0.0, []

    strikes = sorted(strike_map.keys())
    atm = min(strikes, key=lambda x: abs(x - spot))
    atm_idx = strikes.index(atm)

    # Aggregate OI changes: split by adds vs unwinds
    c_oi_add, c_oi_unwind = 0, 0   # calls
    p_oi_add, p_oi_unwind = 0, 0   # puts

    # Near-ATM zone: ±5 strikes (most relevant for intraday)
    near = strikes[max(0, atm_idx - 5): atm_idx + 6]

    for sp in strikes:
        c_ch = strike_map[sp].get("CE", {}).get("oich") or 0
        p_ch = strike_map[sp].get("PE", {}).get("oich") or 0
        if c_ch > 0: c_oi_add    += c_ch
        else:        c_oi_unwind += abs(c_ch)
        if p_ch > 0: p_oi_add    += p_ch
        else:        p_oi_unwind += abs(p_ch)

    # Near-ATM directional OI (most actionable for intraday momentum)
    near_c_add, near_p_add = 0, 0
    near_c_unwind, near_p_unwind = 0, 0
    for sp in near:
        c_ch = strike_map[sp].get("CE", {}).get("oich") or 0
        p_ch = strike_map[sp].get("PE", {}).get("oich") or 0
        if c_ch > 0: near_c_add    += c_ch
        else:        near_c_unwind += abs(c_ch)
        if p_ch > 0: near_p_add    += p_ch
        else:        near_p_unwind += abs(p_ch)

    # ATM strike specific (highest sensitivity)
    atm_c_ch = strike_map.get(atm, {}).get("CE", {}).get("oich") or 0
    atm_p_ch = strike_map.get(atm, {}).get("PE", {}).get("oich") or 0

    score = 0.0
    signals = []
    is_bull = tech_score > 0
    is_bear = tech_score < 0

    # ── Overall writing bias (are dealers writing puts or calls?) ─────────────
    if p_oi_add > 0 and c_oi_add > 0:
        ratio = p_oi_add / c_oi_add
        if ratio > 1.8:
            score += 2.5
            signals.append(("bull",
                f"Put writing {ratio:.1f}x > call writing (Put OI +{_fmt_oi(p_oi_add)} vs Call OI +{_fmt_oi(c_oi_add)}) — dealers building strong floor"))
        elif ratio > 1.2:
            score += 1.5
            signals.append(("bull",
                f"More put OI being written (+{_fmt_oi(p_oi_add)}) vs calls (+{_fmt_oi(c_oi_add)}) — net bullish OI flow"))
        elif ratio < 0.55:
            score -= 2.5
            signals.append(("bear",
                f"Call writing {1/ratio:.1f}x > put writing (Call OI +{_fmt_oi(c_oi_add)} vs Put OI +{_fmt_oi(p_oi_add)}) — dealers building hard ceiling"))
        elif ratio < 0.83:
            score -= 1.5
            signals.append(("bear",
                f"More call OI being written (+{_fmt_oi(c_oi_add)}) vs puts (+{_fmt_oi(p_oi_add)}) — net bearish OI flow"))
        else:
            signals.append(("neut",
                f"Balanced OI writing (Calls +{_fmt_oi(c_oi_add)} / Puts +{_fmt_oi(p_oi_add)}) — no clear dealer bias"))

    # ── 4-quadrant: combine price direction with OI unwind signals ────────────
    if is_bull:
        if c_oi_unwind > c_oi_add * 0.6:
            score += 1.0
            signals.append(("bull",
                f"Call short covering ({_fmt_oi(c_oi_unwind)} call OI unwinding) while price rising — momentum confirmed, shorts fleeing"))
        if p_oi_unwind > p_oi_add * 0.5:
            score -= 0.5
            signals.append(("neut",
                f"Put long unwinding ({_fmt_oi(p_oi_unwind)} put OI closing) — hedges being removed, mixed signal on continuation"))
    elif is_bear:
        if p_oi_unwind > p_oi_add * 0.6:
            score -= 1.0
            signals.append(("bear",
                f"Put short covering ({_fmt_oi(p_oi_unwind)} put OI unwinding) while price falling — bearish momentum, floors collapsing"))
        if c_oi_unwind > c_oi_add * 0.5:
            score += 0.5
            signals.append(("neut",
                f"Call long unwinding ({_fmt_oi(c_oi_unwind)} call OI closing) — upside bets exiting, mixed continuation signal"))

    # ── Near-ATM zone: most relevant for intraday momentum ────────────────────
    if near_p_add > near_c_add and near_p_add > 0:
        score += 1.0
        signals.append(("bull",
            f"Near-ATM put writing dominant (+{_fmt_oi(near_p_add)}) — strong intraday floor being built"))
    elif near_c_add > near_p_add and near_c_add > 0:
        score -= 1.0
        signals.append(("bear",
            f"Near-ATM call writing dominant (+{_fmt_oi(near_c_add)}) — strong intraday ceiling being built"))

    # ── ATM strike (highest gamma, most sensitive level) ──────────────────────
    if atm_c_ch < 0 and atm_p_ch > 0:
        score += 1.5
        signals.append(("bull",
            f"ATM {atm:,.0f}: Call OI shrinking ({_fmt_oi(atm_c_ch)}) + Put OI growing (+{_fmt_oi(atm_p_ch)}) — classic bullish ATM footprint"))
    elif atm_c_ch > 0 and atm_p_ch < 0:
        score -= 1.5
        signals.append(("bear",
            f"ATM {atm:,.0f}: Call OI growing (+{_fmt_oi(atm_c_ch)}) + Put OI shrinking ({_fmt_oi(atm_p_ch)}) — classic bearish ATM footprint"))
    elif atm_p_ch > 0 and atm_c_ch > 0:
        if atm_p_ch > atm_c_ch:
            score += 0.5
            signals.append(("bull", f"ATM {atm:,.0f}: Both sides adding OI, puts growing faster — slight bullish lean"))
        else:
            score -= 0.5
            signals.append(("bear", f"ATM {atm:,.0f}: Both sides adding OI, calls growing faster — slight bearish lean"))

    # ── OI wall proximity (how close is the ceiling/floor?) ───────────────────
    max_c_sp = max(strikes, key=lambda sp: (strike_map[sp].get("CE", {}).get("oi") or 0))
    max_p_sp = max(strikes, key=lambda sp: (strike_map[sp].get("PE", {}).get("oi") or 0))
    max_c_oi = strike_map[max_c_sp].get("CE", {}).get("oi") or 0
    max_p_oi = strike_map[max_p_sp].get("PE", {}).get("oi") or 0

    res_dist_pct = (max_c_sp - spot) / spot * 100 if spot else 999
    sup_dist_pct = (spot - max_p_sp) / spot * 100 if spot else 999

    if 0 < res_dist_pct < 0.4:
        score -= 1.5
        signals.append(("bear",
            f"Resistance wall AT DOOR: {max_c_sp:,.0f} CE (OI {_fmt_oi(max_c_oi)}) only {res_dist_pct:.2f}% above — high rejection risk"))
    elif 0 < res_dist_pct < 0.8:
        score -= 0.5
        signals.append(("neut",
            f"Resistance: {max_c_sp:,.0f} CE (OI {_fmt_oi(max_c_oi)}) — {res_dist_pct:.1f}% above, approaching supply zone"))
    elif res_dist_pct <= 0:
        score -= 2.0
        signals.append(("bear",
            f"Spot above Call wall {max_c_sp:,.0f} (OI {_fmt_oi(max_c_oi)}) — in danger zone, shorts may pile back"))
    else:
        signals.append(("neut",
            f"Resistance {max_c_sp:,.0f} CE (OI {_fmt_oi(max_c_oi)}) at +{res_dist_pct:.1f}% — room to run"))

    if 0 < sup_dist_pct < 0.4:
        score += 1.5
        signals.append(("bull",
            f"Support wall UNDERFOOT: {max_p_sp:,.0f} PE (OI {_fmt_oi(max_p_oi)}) — {sup_dist_pct:.2f}% below, strong cushion"))
    elif sup_dist_pct <= 0:
        score -= 2.0
        signals.append(("bear",
            f"Spot below Put wall {max_p_sp:,.0f} (OI {_fmt_oi(max_p_oi)}) — floor breached, breakdown risk elevated"))
    else:
        signals.append(("neut",
            f"Support {max_p_sp:,.0f} PE (OI {_fmt_oi(max_p_oi)}) at -{sup_dist_pct:.1f}% — floor intact"))

    return round(min(max(score, -4), 4), 2), signals


# ── Layer 4: IV Analysis ──────────────────────────────────────────────────────
def _iv_analysis(strike_map: dict, spot: float) -> tuple[float, list, float]:
    """
    IV level, put-call skew, and IV-direction alignment.

    High IV when buying = expensive premium, IV crush risk after event
    Low IV when buying = cheap gamma, favorable
    Put skew > call skew = market pricing downside fear (bearish)
    Call skew > put skew = market pricing upside (bullish)
    """
    if not strike_map or not spot:
        return 0.0, [], 0.0

    strikes = sorted(strike_map.keys())
    atm = min(strikes, key=lambda x: abs(x - spot))

    # ATM IV
    atm_c_iv = (strike_map.get(atm, {}).get("CE", {}).get("greeks") or {}).get("iv") or 0
    atm_p_iv = (strike_map.get(atm, {}).get("PE", {}).get("greeks") or {}).get("iv") or 0
    atm_iv   = (atm_c_iv + atm_p_iv) / 2 if (atm_c_iv and atm_p_iv) else (atm_c_iv or atm_p_iv)

    # OTM skew: 1.5% OTM strikes
    otm_p_sp = min(strikes, key=lambda x: abs(x - spot * 0.985))
    otm_c_sp = min(strikes, key=lambda x: abs(x - spot * 1.015))
    otm_p_iv = (strike_map.get(otm_p_sp, {}).get("PE", {}).get("greeks") or {}).get("iv") or 0
    otm_c_iv = (strike_map.get(otm_c_sp, {}).get("CE", {}).get("greeks") or {}).get("iv") or 0

    score = 0.0
    signals = []

    # IV level (NIFTY context: normal 10-18%, elevated 18-25%, extreme >25%)
    if atm_iv > 25:
        score -= 2.0
        signals.append(("bear",
            f"ATM IV {atm_iv:.1f}% is EXTREME — options very expensive, IV crush risk severe if buying"))
    elif atm_iv > 18:
        score -= 1.0
        signals.append(("bear",
            f"ATM IV {atm_iv:.1f}% is ELEVATED — premium inflated, consider smaller size or wait for IV to ease"))
    elif atm_iv < 10:
        score += 2.0
        signals.append(("bull",
            f"ATM IV {atm_iv:.1f}% is VERY LOW — cheap gamma, excellent R/R for option buyers"))
    elif atm_iv < 13:
        score += 1.0
        signals.append(("bull",
            f"ATM IV {atm_iv:.1f}% is LOW — fair premium, favorable buying conditions"))
    else:
        signals.append(("neut",
            f"ATM IV {atm_iv:.1f}% is NORMAL (Call {atm_c_iv:.1f}% / Put {atm_p_iv:.1f}%) — fair premium environment"))

    # IV skew: OTM put IV vs OTM call IV
    if otm_p_iv > 0 and otm_c_iv > 0:
        skew = otm_p_iv - otm_c_iv
        if skew > 4:
            score -= 2.0
            signals.append(("bear",
                f"Heavy put skew: OTM put IV {otm_p_iv:.1f}% vs call IV {otm_c_iv:.1f}% (skew +{skew:.1f}%) — market pricing serious downside fear"))
        elif skew > 2:
            score -= 1.0
            signals.append(("bear",
                f"Put skew {skew:.1f}%: OTM puts pricier than calls — dealers hedging downside, bearish risk premium"))
        elif skew < -3:
            score += 2.0
            signals.append(("bull",
                f"Call skew {abs(skew):.1f}%: OTM calls pricier than puts — market positioning for upside breakout"))
        elif skew < -1:
            score += 1.0
            signals.append(("bull",
                f"Mild call skew: OTM call IV {otm_c_iv:.1f}% vs put IV {otm_p_iv:.1f}% — demand for upside calls, bullish"))
        else:
            signals.append(("neut",
                f"IV skew balanced ({skew:+.1f}%) — no fear premium in either direction"))

    # Call IV vs Put IV at ATM (directional bias from market makers)
    if atm_c_iv > 0 and atm_p_iv > 0:
        atm_diff = atm_p_iv - atm_c_iv
        if atm_diff > 2:
            score -= 0.5
            signals.append(("bear",
                f"ATM put IV ({atm_p_iv:.1f}%) > call IV ({atm_c_iv:.1f}%) — market makers pricing more downside at ATM"))
        elif atm_diff < -2:
            score += 0.5
            signals.append(("bull",
                f"ATM call IV ({atm_c_iv:.1f}%) > put IV ({atm_p_iv:.1f}%) — market makers pricing more upside at ATM"))

    return round(min(max(score, -4), 4), 2), signals, atm_iv


# ── Layer 3: Futures Analysis ─────────────────────────────────────────────────
def _futures_analysis(futures: list, spot: float) -> tuple[float, list, str]:
    """
    Institutional positioning via futures:
    - Basis (futures - spot): positive = contango (bullish), negative = backwardation (bearish)
    - Basis magnitude: >50 pts = strong, 20-50 = moderate
    - Futures own % change: are institutions buying or selling futures?
    - Roll spread: term structure slope (contango vs backwardation)
    """
    if not futures or not spot:
        return 0.0, [], ""

    score = 0.0
    signals = []

    near   = futures[0] if len(futures) > 0 else {}
    next_f = futures[1] if len(futures) > 1 else {}

    near_ltp = near.get("ltp") or 0
    next_ltp = next_f.get("ltp") or 0

    if near_ltp:
        basis     = near_ltp - spot
        basis_pct = basis / spot * 100

        if basis > 60:
            score += 3.0
            signals.append(("bull",
                f"Futures +{basis:.0f} pts ({basis_pct:+.2f}%) STRONG CONTANGO — heavy institutional long positioning in futures"))
        elif basis > 30:
            score += 2.0
            signals.append(("bull",
                f"Futures +{basis:.0f} pts ({basis_pct:+.2f}%) CONTANGO — institutions are net long futures, bullish bias"))
        elif basis > 10:
            score += 1.0
            signals.append(("bull",
                f"Futures +{basis:.0f} pts premium — mild institutional long bias, contango intact"))
        elif basis > 0:
            signals.append(("neut",
                f"Futures +{basis:.0f} pts — thin premium, near-neutral institutional stance"))
        elif basis < -40:
            score -= 3.0
            signals.append(("bear",
                f"Futures {basis:.0f} pts ({basis_pct:+.2f}%) STRONG BACKWARDATION — institutions aggressively shorting/hedging via futures"))
        elif basis < -15:
            score -= 2.0
            signals.append(("bear",
                f"Futures {basis:.0f} pts discount BACKWARDATION — institutional SELL/HEDGE pressure via futures"))
        elif basis < 0:
            score -= 1.0
            signals.append(("bear",
                f"Futures {basis:.0f} pts discount — mild backwardation, institutions cautious"))

        # Futures' own price change (is futures market leading or lagging spot?)
        fut_chp = near.get("chp") or 0
        fut_ch  = near.get("ch")  or 0
        if abs(fut_chp) > 0.25:
            if fut_chp > 0:
                score += 0.5
                signals.append(("bull",
                    f"Futures up {fut_chp:+.2f}% ({fut_ch:+.1f} pts) — futures market itself showing bullish move"))
            else:
                score -= 0.5
                signals.append(("bear",
                    f"Futures down {fut_chp:.2f}% ({fut_ch:.1f} pts) — futures market itself showing bearish pressure"))

    # Term structure / roll spread
    if near_ltp and next_ltp:
        roll = next_ltp - near_ltp
        if roll > 30:
            score += 0.5
            signals.append(("bull",
                f"Steep contango: Next month {roll:+.0f} pts above Near — market pricing sustained upside into {next_f.get('month','')}"))
        elif roll > 0:
            signals.append(("neut",
                f"Normal roll: Next {next_f.get('month','')} +{roll:.0f} pts — healthy term structure"))
        elif roll < -20:
            score -= 0.5
            signals.append(("bear",
                f"Inverted term structure: Next month {roll:.0f} pts below Near — backwardation across curve, sustained institutional selling"))
        else:
            signals.append(("neut",
                f"Flat/inverted roll ({roll:+.0f} pts to next) — no strong carry signal"))

    context = signals[0][1] if signals else ""
    return round(min(max(score, -4), 4), 2), signals, context


# ── Layer 5: PCR Analysis ─────────────────────────────────────────────────────
def _pcr_analysis(pcr: float) -> tuple[float, list]:
    """
    PCR contrarian/momentum bands (NSE-calibrated):
    >1.5 = extreme put writing = contrarian BUY signal (crowd is too bearish)
    1.2-1.5 = heavy put writing = bullish
    1.0-1.2 = mild put bias = mildly bullish
    0.85-1.0 = neutral
    0.7-0.85 = mild call writing = mildly bearish
    0.5-0.7 = heavy call writing = bearish
    <0.5 = extreme call writing = contrarian SELL signal (crowd too bullish)
    """
    if not pcr:
        return 0.0, []

    if pcr > 1.5:
        return 2.0, [("bull",
            f"PCR {pcr:.2f} EXTREME put writing — crowd is overly bearish, contrarian BULLISH signal (max floor built)")]
    elif pcr > 1.2:
        return 1.5, [("bull",
            f"PCR {pcr:.2f} — heavy put writing, strong support floor below spot, bullish OI bias")]
    elif pcr > 1.0:
        return 0.5, [("bull",
            f"PCR {pcr:.2f} — more puts than calls written, mild bullish OI bias")]
    elif pcr > 0.85:
        return 0.0, [("neut",
            f"PCR {pcr:.2f} — balanced, no strong directional bias from overall OI")]
    elif pcr > 0.7:
        return -0.5, [("bear",
            f"PCR {pcr:.2f} — call writing dominant, mild resistance building above spot")]
    elif pcr > 0.5:
        return -1.5, [("bear",
            f"PCR {pcr:.2f} — heavy call writing, strong ceiling, bearish OI bias")]
    else:
        return -2.0, [("bear",
            f"PCR {pcr:.2f} EXTREME call writing — crowd is overly bullish, contrarian BEARISH signal (max ceiling built)")]


# ── Layer 6: Max Pain Analysis ────────────────────────────────────────────────
def _max_pain_analysis(spot: float, mp: float) -> tuple[float, list]:
    """
    Expiry gravity: Max Pain is where option writers (dealers) profit most.
    Spot above max pain → gravitational pull DOWN (relevant near expiry).
    """
    if not mp or not spot:
        return 0.0, []

    diff_pct = (spot - mp) / mp * 100
    if diff_pct > 2.0:
        return -1.0, [("bear",
            f"Spot {diff_pct:+.1f}% above Max Pain {mp:,.0f} — strong gravitational pull DOWN toward {mp:,.0f} near expiry")]
    elif diff_pct > 1.0:
        return -0.5, [("bear",
            f"Spot {diff_pct:+.1f}% above Max Pain {mp:,.0f} — mild pull toward {mp:,.0f}")]
    elif diff_pct < -2.0:
        return 1.0, [("bull",
            f"Spot {diff_pct:+.1f}% below Max Pain {mp:,.0f} — gravitational pull UP toward {mp:,.0f}")]
    elif diff_pct < -1.0:
        return 0.5, [("bull",
            f"Spot {diff_pct:+.1f}% below Max Pain {mp:,.0f} — mild upward pull toward {mp:,.0f}")]
    else:
        return 0.0, [("neut",
            f"Spot near Max Pain {mp:,.0f} ({diff_pct:+.1f}%) — range-bound expiry likely, neutral gravity")]


# ── Layer 3: Velocity (OI / IV / Wall / PCR change over session history) ──────
def _velocity_layer(sym: str, spot: float) -> tuple[float, list]:
    """
    The quant desk's edge: not what OI IS, but how fast it is CHANGING.

    Sources: OITimeSeriesStore — snapshots taken every ~3 min since 9:15.
    Lookbacks: 15 min / 30 min / 1 hr.
    Score range: [-4, +4].
    Returns (0.0, [neut msg]) when session history < 3 snapshots (~9 min).
    """
    if not _INTRADAY_STORE_AVAILABLE:
        return 0.0, [("neut", "Velocity store unavailable")]
    try:
        vel = oi_store.velocity(sym)
    except Exception:
        return 0.0, [("neut", "Velocity store unavailable")]

    if not vel.get("has_data"):
        n = vel.get("snap_count", 0)
        return 0.0, [("neut",
            f"Velocity building — {n}/3 snapshots collected (need ~9 min of session)")]

    score   = 0.0
    signals = []

    oi   = vel["oi"]
    iv   = vel["iv"]
    wal  = vel["walls"]
    pcr  = vel["pcr"]

    # ── 1. OI flow velocity over 1 hour ──────────────────────────────────────
    c1 = oi.get("call_1hr") or 0
    p1 = oi.get("put_1hr")  or 0
    if abs(c1) > 200_000 or abs(p1) > 200_000:
        if p1 > 0 and (c1 <= 0 or p1 > c1 * 1.4):
            score += 2.0
            signals.append(("bull",
                f"OI velocity 1hr: Put OI {'+' if p1>0 else ''}{_fmt_oi(p1)} vs Call {'+' if c1>0 else ''}{_fmt_oi(c1)}"
                f" — dealers building floor at scale"))
        elif c1 > 0 and (p1 <= 0 or c1 > p1 * 1.4):
            score -= 2.0
            signals.append(("bear",
                f"OI velocity 1hr: Call OI +{_fmt_oi(c1)} vs Put {'+' if p1>0 else ''}{_fmt_oi(p1)}"
                f" — dealers building ceiling at scale"))
        elif p1 < 0 and abs(p1) > abs(c1):
            score -= 1.0
            signals.append(("bear",
                f"OI velocity 1hr: Put OI unwinding {_fmt_oi(abs(p1))} — floor being dismantled, breakdown risk"))
        elif c1 < 0 and abs(c1) > abs(p1):
            score += 1.0
            signals.append(("bull",
                f"OI velocity 1hr: Call OI unwinding {_fmt_oi(abs(c1))} — ceiling collapsing, short covering underway"))

    # ── 2. Near-ATM velocity over 30 min (most actionable) ───────────────────
    nc = oi.get("near_call_30m") or 0
    np = oi.get("near_put_30m")  or 0
    if abs(nc) > 50_000 or abs(np) > 50_000:
        if np > 0 and np > abs(nc):
            score += 1.5
            signals.append(("bull",
                f"Near-ATM 30m: Put OI +{_fmt_oi(np)} vs Call {'+' if nc>0 else ''}{_fmt_oi(nc)}"
                f" — strong floor being built at current levels"))
        elif nc > 0 and nc > abs(np):
            score -= 1.5
            signals.append(("bear",
                f"Near-ATM 30m: Call OI +{_fmt_oi(nc)} vs Put {'+' if np>0 else ''}{_fmt_oi(np)}"
                f" — ceiling being capped right here"))

    # ── 3. Wall shift — THE most powerful directional signal ─────────────────
    cws = wal.get("call_shift_1hr")
    pws = wal.get("put_shift_1hr")
    cn  = wal.get("call_now", 0);  c1h = wal.get("call_1hr_ago") or cn
    pn  = wal.get("put_now",  0);  p1h = wal.get("put_1hr_ago")  or pn

    if cws is not None and abs(cws) >= 100:
        if cws > 0:
            score += 1.5
            signals.append(("bull",
                f"Call wall RISING {c1h:,.0f} → {cn:,.0f} (+{cws:.0f} pts in 1hr)"
                f" — sellers retreating, range expanding upward"))
        else:
            score -= 1.5
            signals.append(("bear",
                f"Call wall DROPPING {c1h:,.0f} → {cn:,.0f} ({cws:.0f} pts in 1hr)"
                f" — ceiling lowering, compression ahead"))

    if pws is not None and abs(pws) >= 100:
        if pws > 0:
            score += 1.5
            signals.append(("bull",
                f"Put wall RISING {p1h:,.0f} → {pn:,.0f} (+{pws:.0f} pts in 1hr)"
                f" — support floor strengthening under spot"))
        else:
            score -= 1.5
            signals.append(("bear",
                f"Put wall COLLAPSING {p1h:,.0f} → {pn:,.0f} ({pws:.0f} pts in 1hr)"
                f" — support breakdown risk elevated"))

    # ── 4. IV regime (expanding = uncertainty, contracting = stable) ──────────
    iv_ch  = iv.get("change_1hr") or 0
    sk_ch  = iv.get("skew_ch_1hr") or 0
    regime = iv.get("regime", "stable")
    iv_now = iv.get("now", 0)
    if abs(iv_ch) > 1.5:
        if regime == "expanding":
            if sk_ch > 0:
                score -= 0.5
                signals.append(("bear",
                    f"IV expanding +{iv_ch:.1f}% to {iv_now:.1f}% (1hr), put skew rising {sk_ch:+.1f}%"
                    f" — fear premium building, market pricing downside"))
            else:
                signals.append(("neut",
                    f"IV expanding +{iv_ch:.1f}% to {iv_now:.1f}% — uncertainty rising, both sides expensive"))
        else:
            score += 0.5
            signals.append(("bull",
                f"IV contracting {iv_ch:.1f}% to {iv_now:.1f}% (1hr)"
                f" — premium deflating, calm environment favours buyers"))

    # ── 5. PCR slope over 30 min ──────────────────────────────────────────────
    pcr_ch = pcr.get("change_30m") or 0
    pcr_n  = pcr.get("now", 0)
    p30    = pcr.get("30m_ago") or pcr_n
    if abs(pcr_ch) > 0.08:
        if pcr_ch > 0:
            score += 1.0
            signals.append(("bull",
                f"PCR rising {p30:.2f} → {pcr_n:.2f} (+{pcr_ch:.2f} in 30m)"
                f" — put writing accelerating, floor building fast"))
        else:
            score -= 1.0
            signals.append(("bear",
                f"PCR falling {p30:.2f} → {pcr_n:.2f} ({pcr_ch:.2f} in 30m)"
                f" — put protection being removed, call writing dominant"))

    return round(min(max(score, -4), 4), 2), signals


# ── Composite score + conviction ──────────────────────────────────────────────
def _composite(tech_score, oi_score, vel_score, inst_score,
               fut_score, iv_score, pcr_score, mp_score,
               ctx_score, shock_score=0.0) -> tuple[float, str, str]:
    """
    10-layer weighted composite. Returns (weighted_score, verdict, color).

    Weight rationale:
      Daily Context (≈16%) provides yesterday's EOD structural setup from a
      24-signal prediction engine + HMM regime + FII trend — a dense,
      pre-computed signal that anchors intraday decisions.
      Institutional reduced because FII data now also appears in Layer 9,
      avoiding double-counting.
      Shock (Layer 10, ≈7%) is the abrupt-regime modulator: sudden OI bursts,
      volume surges and price impulses inside the last few minutes. It is the
      LINEAR part of the shock's influence; the non-linear override (confidence
      cut / signal stand-down on an opposing alert) is applied separately in
      build_recommendation via apply_shock_override().
    """
    w = {
        "tech":    0.25,
        "oi":      0.10,
        "vel":     0.18,
        "inst":    0.10,
        "futures": 0.08,
        "iv":      0.06,
        "pcr":     0.04,
        "mp":      0.02,
        "ctx":     0.17,   # Daily Context bridge
        "shock":   0.08,   # Layer 10 — abrupt regime modulator
    }
    # Renormalise so the 10 weights sum to exactly 1.0 (preserves the relative
    # proportions of the original 9 layers while making room for shock).
    _tot = sum(w.values())
    w = {k: v / _tot for k, v in w.items()}

    tech_n = max(min(tech_score / 2, 4), -4)

    composite = (
        tech_n      * w["tech"]    +
        oi_score    * w["oi"]      +
        vel_score   * w["vel"]     +
        inst_score  * w["inst"]    +
        fut_score   * w["futures"] +
        iv_score    * w["iv"]      +
        pcr_score   * w["pcr"]     +
        mp_score    * w["mp"]      +
        ctx_score   * w["ctx"]     +
        shock_score * w["shock"]
    )

    if   composite >= 2.0:  return composite, "STRONG BUY",  "#22c55e"
    elif composite >= 1.0:  return composite, "BUY",          "#4ade80"
    elif composite >= 0.4:  return composite, "MILD BUY",     "#86efac"
    elif composite <= -2.0: return composite, "STRONG SELL",  "#ef4444"
    elif composite <= -1.0: return composite, "SELL",          "#f87171"
    elif composite <= -0.4: return composite, "MILD SELL",    "#fca5a5"
    else:                   return composite, "NEUTRAL",       "#94a3b8"


# ── Shock override (non-linear conviction modulator) ──────────────────────────
def apply_shock_override(composite: float, confidence: float, shock: dict
                         ) -> tuple[float, float, bool, list]:
    """
    The non-linear half of Layer 10 (the linear half is already folded into
    `composite` by _composite). Adjusts confidence — and, on a strong opposing
    alert against a weak setup, stands the signal down entirely.

    Returns (composite, confidence, forced_neutral, notes).
      forced_neutral=True  → caller routes to the NEUTRAL "Regime Shift" path.
    """
    notes: list = []
    if not shock:
        return composite, confidence, False, notes
    level = shock.get("level", "none")
    sdir  = shock.get("direction", "")        # "CE" bullish / "PE" bearish / ""
    if level == "none" or not sdir:
        return composite, confidence, False, notes

    comp_dir = "CE" if composite > 0 else "PE" if composite < 0 else ""
    aligned  = sdir == comp_dir and comp_dir != ""
    head     = shock["signals"][0][1] if shock.get("signals") else "market shock"

    if level == "alert":
        if aligned:
            confidence = min(confidence + 12, 95)
            notes.append(f"✓ SHOCK CONFIRMS — {head}")
        else:
            confidence = round(confidence * 0.40)
            notes.append(f"⚠ CONDITIONS CHANGING — {head}")
            # Weak base signal overwhelmed by a sudden opposing flow → stand aside.
            if abs(composite) < 1.0:
                notes.append("Signal stood down — opposing shock overwhelms a weak setup")
                return 0.0, min(confidence, 25), True, notes
    elif level == "watch":
        if aligned:
            confidence = min(confidence + 5, 95)
            notes.append(f"Shock leans with signal — {head}")
        else:
            confidence = round(confidence * 0.75)
            notes.append(f"⚠ Early opposing flow — {head}")

    return composite, confidence, False, notes


# ── Expiry selection ──────────────────────────────────────────────────────────
def select_expiry(expiry_data: list, tf_key: str) -> tuple[str, str]:
    if not expiry_data:
        return "", ""
    rank = TF_PROFILES[tf_key]["expiry_rank"]
    if rank == "M":
        monthly = [e for e in expiry_data if e.get("expiry_flag") == "M"]
        chosen  = monthly[0] if monthly else expiry_data[-1]
    else:
        idx    = min(rank, len(expiry_data) - 1)
        chosen = expiry_data[idx]
    return chosen["expiry"], chosen["date"]


# ── Strike selection ──────────────────────────────────────────────────────────
def select_strike(strike_map: dict, direction: str, target_delta: float) -> tuple[int, dict]:
    best_sp, best_opt, best_diff = None, {}, 999.0
    for sp, opts in strike_map.items():
        opt   = opts.get(direction, {})
        delta = abs((opt.get("greeks") or {}).get("delta") or 0)
        if delta and abs(delta - target_delta) < best_diff:
            best_diff = abs(delta - target_delta)
            best_sp   = sp
            best_opt  = opt
    return best_sp, best_opt


# ── Main recommendation engine ────────────────────────────────────────────────
def build_recommendation(
    sym:         str,
    tf_key:      str,
    spot:        float,
    strike_map:  dict,
    expiry_data: list,
    futures:     list,
    pcr:         float,
    mp:          float,
    total_c_oi:  int,
    total_p_oi:  int,
) -> dict | None:
    profile = TF_PROFILES.get(tf_key)
    if not profile or not spot or not strike_map:
        return None

    # ── Layer 1: Technical ────────────────────────────────────────────────────
    cfg        = TIMEFRAMES.get(tf_key, {})
    df         = fetch_ohlcv(sym, cfg["res"], cfg["days"])
    sig        = _analyze(df, cfg["min_candles"])
    tech_score = sig.get("score", 0)
    tech_reasons = sig.get("reasons", [])

    # ── Layer 2: OI Flow (snapshot) ───────────────────────────────────────────
    oi_score, oi_signals = _oi_flow(strike_map, spot, tech_score)

    # ── Layer 3: Velocity (session time-series) ───────────────────────────────
    vel_score, vel_signals = _velocity_layer(sym, spot)

    # ── Layer 4: Futures ──────────────────────────────────────────────────────
    fut_score, fut_signals, fut_context = _futures_analysis(futures, spot)

    # ── Layer 5: IV ───────────────────────────────────────────────────────────
    iv_score, iv_signals, atm_iv = _iv_analysis(strike_map, spot)

    # ── Layer 6: PCR ──────────────────────────────────────────────────────────
    pcr_score, pcr_signals = _pcr_analysis(pcr)

    # ── Layer 7: Max Pain ─────────────────────────────────────────────────────
    mp_score, mp_signals = _max_pain_analysis(spot, mp)

    # ── Layer 8: Institutional (FII futures + FAO smart-money + PE) ───────────
    inst_score, inst_signals = get_institutional_bias(sym, spot)

    # ── Layer 9: Daily Context (EOD structural setup from Daily_Cash_Market) ──
    if _BRIDGE_AVAILABLE:
        try:
            ctx_score, ctx_signals = _get_bridge().score_index(sym)
        except Exception:
            ctx_score, ctx_signals = 0.0, [("neut", "Daily context: error reading bridge")]
    else:
        ctx_score, ctx_signals = 0.0, [("neut", "Daily context: bridge module unavailable")]

    # ── Layer 10: Market Shock / Regime-Shift (abrupt OI/volume/price) ────────
    if _SHOCK_AVAILABLE:
        try:
            shock = analyze_shock(sym)
        except Exception:
            shock = {"level": "none", "score": 0.0, "direction": "", "signals": []}
    else:
        shock = {"level": "none", "score": 0.0, "direction": "", "signals": []}
    shock_score   = shock.get("score", 0.0)
    shock_signals = shock.get("signals", [])

    # ── Session phase / strategy profile (time-of-day trade shaping) ──────────
    if _INTRADAY_STORE_AVAILABLE:
        strat = session_strategy()
    else:
        strat = dict(phase="UNKNOWN", conf_mult=1.0, allow_new_entry=True, min_conf=0,
                     stop_mult=1.0, target_mult=1.0, reentry_cooldown_min=10,
                     bias="", caution="")
    phase_name    = strat["phase"]
    phase_mult    = strat["conf_mult"]
    phase_caution = strat["caution"]

    # ── Composite direction ───────────────────────────────────────────────────
    composite, verdict, verdict_clr = _composite(
        tech_score, oi_score, vel_score, inst_score,
        fut_score, iv_score, pcr_score, mp_score,
        ctx_score, shock_score,
    )

    # Count layer agreements (how many of the 10 layers agree with composite direction?)
    layer_scores = [tech_score / 2, oi_score, vel_score,
                    inst_score, fut_score, iv_score, pcr_score, mp_score,
                    ctx_score, shock_score]
    bull_layers  = sum(1 for s in layer_scores if s > 0.3)
    bear_layers  = sum(1 for s in layer_scores if s < -0.3)
    neut_layers  = 10 - bull_layers - bear_layers

    if composite > 0:
        agreement = bull_layers
    elif composite < 0:
        agreement = bear_layers
    else:
        agreement = neut_layers

    # Confidence from composite magnitude + layer agreement + session phase.
    # Threshold = 5.  Each layer beyond 5 agreeing adds +5%.
    # Below threshold: each missing layer subtracts 5% (reduces false conviction).
    # Clamped [0, 95] — negative confidence is meaningless and breaks UI/delta.
    raw_conf    = min(abs(composite) / 2.5 * 100, 95)
    align_bonus = (agreement - 5) * 5
    confidence  = min(max(round((raw_conf + align_bonus) * phase_mult, 0), 0), 95)
    # Hard floor: strong composite magnitude cannot override genuinely split signals.
    # Only applied when ≤2 layers agree — this is the case where one anomalously
    # high layer is dragging the composite while the rest are neutral/opposing.
    if agreement <= 2:
        confidence = min(confidence, 35)   # 2/10 agreeing = speculative at best

    # ── Shock override (non-linear half of Layer 10) ─────────────────────────
    # A sudden ALERT-level shock against the signal slashes confidence and, on a
    # weak base setup, stands the trade down. Aligned shocks add conviction.
    composite, confidence, forced_neutral, shock_notes = apply_shock_override(
        composite, confidence, shock,
    )

    # Neutral — no clear edge (or signal stood down by an opposing shock).
    # Threshold matches _composite()'s neutral band (±0.40) exactly,
    # so we never return a trade with signal="NEUTRAL".
    if forced_neutral or abs(composite) < 0.40:
        neutral_label = "NEUTRAL — Regime Shift" if forced_neutral else "NEUTRAL — No Clear Edge"
        neutral_clr   = "#f59e0b" if forced_neutral else "#94a3b8"
        neutral_warn  = "  |  ".join([n for n in shock_notes] + ([phase_caution] if phase_caution else [])) or phase_caution
        _idb_write_setup(
            sym, tf_key, "NEUTRAL",
            composite, confidence, "",
            round(tech_score / 2, 2), oi_score, vel_score, inst_score,
            fut_score, iv_score, pcr_score, mp_score, ctx_score,
            agreement, phase_name, spot, atm_iv,
        )
        return {
            "signal":     neutral_label,
            "color":      neutral_clr,
            "score":      composite,
            "confidence": confidence,
            "neutral":    True,
            "tf_label":   profile["label"],
            "phase":      phase_name,
            "warning":    neutral_warn,
            "reasons":    tech_reasons,
            "shock":      shock,
        }

    direction = "CE" if composite > 0 else "PE"
    dir_label = "CALL (CE)" if composite > 0 else "PUT (PE)"
    dir_clr   = "#4ade80" if composite > 0 else "#f87171"

    # ── Delta target from conviction ──────────────────────────────────────────
    if confidence >= 65:
        target_delta = profile["delta_high"]
        conviction   = "HIGH"
    elif confidence >= 45:
        target_delta = profile["delta_mid"]
        conviction   = "MODERATE"
    else:
        target_delta = profile["delta_low"]
        conviction   = "LOW"

    # ── Expiry + Strike ───────────────────────────────────────────────────────
    exp_epoch, exp_date = select_expiry(expiry_data, tf_key)
    best_sp, best_opt   = select_strike(strike_map, direction, target_delta)
    if not best_sp or not best_opt:
        return None

    ltp   = best_opt.get("ltp")    or 0
    iv    = (best_opt.get("greeks") or {}).get("iv")    or 0
    delta = abs((best_opt.get("greeks") or {}).get("delta") or 0)
    gamma = abs((best_opt.get("greeks") or {}).get("gamma") or 0)
    theta = (best_opt.get("greeks") or {}).get("theta") or 0
    vega  = (best_opt.get("greeks") or {}).get("vega")  or 0
    oi    = best_opt.get("oi")     or 0
    vol   = best_opt.get("volume") or 0

    if ltp < 2:
        return None

    # ── Risk management (geometry shaped by session phase) ────────────────────
    # Wider stops at the volatile open; tighter, faster targets into the
    # theta-heavy close. Multipliers come from session_strategy().
    sl_pct = profile["sl_pct"] * strat["stop_mult"]
    t1_pct = profile["t1_pct"] * strat["target_mult"]
    t2_pct = profile["t2_pct"] * strat["target_mult"]

    entry_lo  = round(ltp * 0.98, 2)
    entry_hi  = round(ltp * 1.02, 2)
    sl        = round(ltp * (1 - sl_pct), 2)
    t1        = round(ltp * (1 + t1_pct), 2)
    t2        = round(ltp * (1 + t2_pct), 2)
    rr        = round(t1_pct / sl_pct, 1)

    lot_size  = LOT_SIZES.get(sym, 50)
    loss_lot  = round((ltp - sl)  * lot_size, 0)
    profit_t1 = round((t1  - ltp) * lot_size, 0)
    profit_t2 = round((t2  - ltp) * lot_size, 0)

    sl_pts = (ltp - sl) / delta if delta > 0 else abs(best_sp - spot) * 0.5
    spot_sl = round(spot - sl_pts, 0) if direction == "CE" else round(spot + sl_pts, 0)

    # ── IV context ────────────────────────────────────────────────────────────
    if iv > 25:
        iv_context = f"IV {iv:.1f}% is EXTREME HIGH — severe IV crush risk, use smaller size"
    elif iv > 18:
        iv_context = f"IV {iv:.1f}% is ELEVATED — premium expensive, prefer selling or tight entry"
    elif iv < 10:
        iv_context = f"IV {iv:.1f}% is LOW — very cheap option, excellent buyer conditions"
    elif iv < 13:
        iv_context = f"IV {iv:.1f}% is LOW-NORMAL — good buying conditions"
    else:
        iv_context = f"IV {iv:.1f}% — normal range."

    # ── Warning generation ────────────────────────────────────────────────────
    daily_df    = fetch_ohlcv(sym, "D", 60)
    daily_sig   = _analyze(daily_df, 30) if not daily_df.empty else {}
    daily_score = daily_sig.get("score", 0)

    warnings = []
    if composite > 0 and daily_score < -1:
        warnings.append("CAUTION: Intraday BUY against DAILY SELL trend — scalp only, do NOT hold overnight, tight SL")
    elif composite < 0 and daily_score > 1:
        warnings.append("CAUTION: Intraday SELL against DAILY BUY trend — scalp only, do NOT hold overnight, tight SL")
    elif composite > 0 and daily_score > 0:
        warnings.append("Daily + intraday trend ALIGNED (bullish) — higher conviction, can hold with trailing SL")
    elif composite < 0 and daily_score < 0:
        warnings.append("Daily + intraday trend ALIGNED (bearish) — higher conviction, can hold with trailing SL")

    if agreement <= 2:
        warnings.append(f"Only {agreement}/10 signal layers agree — CONFLICTED setup, small size or skip")
    elif agreement == 3:
        warnings.append(f"{agreement}/10 layers agree — mixed conviction, use half position size")

    if atm_iv > 20:
        warnings.append(f"ATM IV {atm_iv:.1f}% elevated — buy dips in premium if possible, IV crush will erode gains fast")

    # Layered signals for "WHY THIS TRADE?" panel (combine all layers).
    # Shock signals lead when present — they are the freshest, most actionable read.
    all_opt_signals = (
        shock_signals[:2] +        # Layer 10 — abrupt regime shifts (last few minutes)
        ctx_signals[:5] +          # Daily context — structural EOD setup (B1-B8)
        vel_signals[:2] +          # Intraday velocity next
        inst_signals[:2] +
        oi_signals[:2] +
        pcr_signals +
        mp_signals +
        fut_signals[:1]
    )

    # Shock override notes lead the warnings (most time-sensitive), then phase.
    for n in reversed(shock_notes):
        warnings.insert(0, n)
    if phase_caution:
        warnings.insert(0, phase_caution)

    warning = "  |  ".join(warnings) if warnings else ""

    # ── Layer summary ─────────────────────────────────────────────────────────
    layer_summary = {
        "tech":     {"score": round(tech_score / 2, 2), "label": "Technical (25%)"},
        "oi":       {"score": oi_score,   "label": "OI Flow snapshot (10%)"},
        "velocity": {"score": vel_score,  "label": "OI/IV/Wall Velocity (18%)"},
        "inst":     {"score": inst_score, "label": "Institutional FII/FAO (10%)"},
        "futures":  {"score": fut_score,  "label": "Futures Basis (8%)"},
        "iv":       {"score": iv_score,   "label": "IV (6%)"},
        "pcr":      {"score": pcr_score,  "label": "PCR (4%)"},
        "mp":       {"score": mp_score,   "label": "Max Pain (2%)"},
        "context":  {"score": ctx_score,  "label": "Daily Context EOD (16%)"},
        "shock":    {"score": shock_score, "label": "Market Shock (7%)"},
        "agree":    f"{agreement}/10 layers aligned",
        "phase":    phase_name,
    }

    _idb_write_setup(
        sym, tf_key, verdict,
        composite, confidence, direction,
        layer_summary["tech"]["score"],
        layer_summary["oi"]["score"],
        layer_summary["velocity"]["score"],
        layer_summary["inst"]["score"],
        layer_summary["futures"]["score"],
        layer_summary["iv"]["score"],
        layer_summary["pcr"]["score"],
        layer_summary["mp"]["score"],
        layer_summary["context"]["score"],
        agreement, phase_name, spot, atm_iv,
    )

    return {
        "signal":       verdict,
        "color":        verdict_clr,
        "score":        composite,
        "confidence":   confidence,
        "conviction":   conviction,
        "neutral":      False,

        "action":       "BUY",
        "direction":    direction,
        "dir_label":    dir_label,
        "dir_clr":      dir_clr,
        "trade_type":   profile["trade"],
        "tf_label":     profile["label"],

        "strike":       best_sp,
        "option_sym":   best_opt.get("symbol"),
        "spot":         spot,
        "exp_date":     exp_date,
        "ltp":          ltp,
        "entry_lo":     entry_lo,
        "entry_hi":     entry_hi,
        "sl":           sl,
        "t1":           t1,
        "t2":           t2,
        "rr":           rr,
        "spot_sl":      spot_sl,
        "loss_lot":     loss_lot,
        "profit_t1":    profit_t1,
        "profit_t2":    profit_t2,
        "lot_size":     lot_size,

        "iv":           iv,
        "delta":        delta,
        "gamma":        gamma,
        "theta":        theta,
        "vega":         vega,
        "oi":           oi,
        "volume":       vol,

        "tech_reasons": tech_reasons,
        "opt_signals":  all_opt_signals,
        "fut_context":  fut_context,
        "fut_bias":     "bull" if fut_score > 0 else "bear" if fut_score < 0 else "neut",
        "iv_context":   iv_context,
        "warning":      warning,
        "daily_score":  daily_score,
        "layer_summary": layer_summary,
        "shock":        shock,
        "shock_level":  shock.get("level", "none"),
        # Time-of-day strategy gates (consumed by the ledger at open time).
        "phase":               phase_name,
        "phase_bias":          strat["bias"],
        "allow_new_entry":     strat["allow_new_entry"],
        "phase_min_conf":      strat["min_conf"],
        "reentry_cooldown_min": strat["reentry_cooldown_min"],
    }
