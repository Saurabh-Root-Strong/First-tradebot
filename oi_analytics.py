"""
oi_analytics.py — pure option-chain analytics (extracted from dashboard.py).

The dashboard mixed three layers in one 4,900-line file: data fetch, business
logic, and presentation. This module holds the BUSINESS-LOGIC slice for the live
option chain — max-pain, the multi-signal chain-prediction engine, and the OI
number formatters. All functions are pure (no Dash, no Fyers, no I/O), so they
are unit-testable in isolation and shared by both the dashboard and any headless
caller (engine / replay).
"""
from __future__ import annotations


def _fmt_oi(v) -> str:
    """Format an OI/volume number. Handles negative values (OI unwinding)."""
    if not v: return "—"
    sign = "-" if v < 0 else ""
    a = abs(v)
    if a >= 1_000_000: return f"{sign}{a / 1_000_000:.2f}M"
    if a >= 1_000:     return f"{sign}{a / 1_000:.1f}K"
    return f"{sign}{int(a)}"


def _fmt_futoi(v) -> str:
    """Signed futures-OI day change with ADAPTIVE units (L at index scale, K below a
    lakh, raw below 1K). Thin index futures (BANKNIFTY/FINNIFTY/MIDCPNIFTY — their
    liquidity is in options, not futures) build only ~thousands of contracts/day, so
    a whole-lakh format collapses them to '+0L'. Shows the real number instead."""
    if v is None: return ""
    a = abs(int(v)); s = "+" if v >= 0 else "-"
    if   a >= 100_000: body = f"{a / 1e5:.1f}L"
    elif a >= 1_000:   body = f"{a / 1e3:.0f}K"
    else:              body = f"{a}"
    return f"  ·  futOI {s}{body} today"


def compute_max_pain(chain: list) -> float:
    # Max pain = price minimising total ITM intrinsic loss for option WRITERS.
    # CE writer loses max(0, test - strike): ITM when test > strike.
    # PE writer loses max(0, strike - test): ITM when test < strike.
    strikes = [r["strike_price"] for r in chain]
    best, mp = float("inf"), strikes[0]
    for test in strikes:
        loss = sum(
            ((r.get("call_options") or {}).get("oi") or 0) * max(0, test - r["strike_price"]) +
            ((r.get("put_options")  or {}).get("oi") or 0) * max(0, r["strike_price"] - test)
            for r in chain
        )
        if loss < best:
            best, mp = loss, test
    return mp


def compute_prediction(strike_map: dict, spot: float, pcr: float, mp: float) -> dict:
    """
    Multi-signal option chain prediction engine.
    Signals: PCR, Max Pain gravity, OI walls, OI change direction, IV skew.
    Returns verdict, score, signals, support, resistance.
    """
    if not strike_map or not spot:
        return {}

    strikes = sorted(strike_map.keys())

    # Max OI strikes (resistance = highest call OI, support = highest put OI)
    max_c_sp = max(strikes, key=lambda sp: (strike_map[sp].get("CE", {}).get("oi") or 0))
    max_p_sp = max(strikes, key=lambda sp: (strike_map[sp].get("PE", {}).get("oi") or 0))
    max_c_oi = strike_map[max_c_sp].get("CE", {}).get("oi") or 0
    max_p_oi = strike_map[max_p_sp].get("PE", {}).get("oi") or 0

    # OI change totals (positive = fresh writing, negative = unwinding)
    tot_c_oich = sum((strike_map[sp].get("CE", {}).get("oich") or 0) for sp in strikes)
    tot_p_oich = sum((strike_map[sp].get("PE", {}).get("oich") or 0) for sp in strikes)

    # Average IV (call vs put skew)
    c_ivs = [(strike_map[sp].get("CE", {}).get("greeks") or {}).get("iv") or 0 for sp in strikes]
    p_ivs = [(strike_map[sp].get("PE", {}).get("greeks") or {}).get("iv") or 0 for sp in strikes]
    avg_c_iv = sum(c_ivs) / len(c_ivs) if c_ivs else 0
    avg_p_iv = sum(p_ivs) / len(p_ivs) if p_ivs else 0

    score    = 0
    signals  = []

    # ── Signal 1: PCR ──────────────────────────────────────────────────────────
    if pcr > 1.3:
        score += 2
        signals.append(("🟢", f"PCR {pcr:.2f} — Heavy put writing → strong floor, bullish",         "bull"))
    elif pcr > 1.0:
        score += 1
        signals.append(("🟢", f"PCR {pcr:.2f} — More puts than calls → mild bullish bias",           "bull"))
    elif pcr > 0.85:
        signals.append(("⚪", f"PCR {pcr:.2f} — Balanced OI, no clear directional edge",            "neut"))
    elif pcr > 0.7:
        score -= 1
        signals.append(("🔴", f"PCR {pcr:.2f} — Call writing dominating → mild resistance",         "bear"))
    else:
        score -= 2
        signals.append(("🔴", f"PCR {pcr:.2f} — Heavy call writing → strong ceiling, bearish",      "bear"))

    # ── Signal 2: Max Pain gravity ─────────────────────────────────────────────
    mp_diff = (spot - mp) / mp * 100 if mp else 0
    if mp_diff > 1.5:
        score -= 1
        signals.append(("🔴", f"Spot {mp_diff:+.1f}% above Max Pain {mp:,.0f} → gravitational pull down",  "bear"))
    elif mp_diff < -1.5:
        score += 1
        signals.append(("🟢", f"Spot {mp_diff:+.1f}% below Max Pain {mp:,.0f} → gravitational pull up",    "bull"))
    else:
        signals.append(("⚪", f"Spot near Max Pain {mp:,.0f} ({mp_diff:+.1f}%) → range-bound expiry likely", "neut"))

    # ── Signal 3: Call wall (resistance) proximity ─────────────────────────────
    res_pct = (max_c_sp - spot) / spot * 100 if spot else 0
    if 0 < res_pct < 0.5:
        score -= 1
        signals.append(("🔴", f"Approaching Call wall {max_c_sp:,.0f} (OI {_fmt_oi(max_c_oi)}) → strong resistance just above", "bear"))
    elif res_pct <= 0:
        score -= 1
        signals.append(("🔴", f"Spot above Call wall {max_c_sp:,.0f} → shorts covering, volatile", "bear"))
    else:
        signals.append(("⚪", f"Resistance @ {max_c_sp:,.0f} (+{res_pct:.1f}%), Call OI {_fmt_oi(max_c_oi)}", "neut"))

    # ── Signal 4: Put wall (support) proximity ─────────────────────────────────
    sup_pct = (spot - max_p_sp) / spot * 100 if spot else 0
    if 0 < sup_pct < 0.5:
        score += 1
        signals.append(("🟢", f"Near Put wall {max_p_sp:,.0f} (OI {_fmt_oi(max_p_oi)}) → strong floor just below", "bull"))
    elif sup_pct <= 0:
        score -= 1
        signals.append(("🔴", f"Spot below Put wall {max_p_sp:,.0f} → puts in pain, breakdown risk", "bear"))
    else:
        signals.append(("⚪", f"Support @ {max_p_sp:,.0f} (-{sup_pct:.1f}%), Put OI {_fmt_oi(max_p_oi)}", "neut"))

    # ── Signal 5: OI change direction (fresh build vs unwinding) ──────────────
    if tot_p_oich > 0 and tot_c_oich > 0:
        ratio = tot_p_oich / tot_c_oich if tot_c_oich else 1
        if ratio > 1.3:
            score += 1
            signals.append(("🟢", f"Put OI growing {ratio:.1f}x faster than Call OI → fresh put writing, bullish", "bull"))
        elif ratio < 0.7:
            score -= 1
            signals.append(("🔴", f"Call OI growing {1/ratio:.1f}x faster than Put OI → fresh call writing, bearish", "bear"))
        else:
            signals.append(("⚪", "OI buildup balanced across calls and puts", "neut"))
    elif tot_c_oich < 0 and tot_p_oich > 0:
        score += 1
        signals.append(("🟢", "Call OI unwinding + Put OI adding → bullish shift", "bull"))
    elif tot_p_oich < 0 and tot_c_oich > 0:
        score -= 1
        signals.append(("🔴", "Put OI unwinding + Call OI adding → bearish shift", "bear"))

    # ── Signal 6: IV skew ──────────────────────────────────────────────────────
    if avg_p_iv > 0 and avg_c_iv > 0:
        iv_skew = avg_p_iv - avg_c_iv
        if iv_skew > 2:
            score -= 1
            signals.append(("🔴", f"Put IV {avg_p_iv:.1f}% > Call IV {avg_c_iv:.1f}% → fear premium in puts, bearish skew", "bear"))
        elif iv_skew < -2:
            score += 1
            signals.append(("🟢", f"Call IV {avg_c_iv:.1f}% > Put IV {avg_p_iv:.1f}% → demand for upside calls, bullish skew", "bull"))
        else:
            signals.append(("⚪", f"IV balanced — Call {avg_c_iv:.1f}%  Put {avg_p_iv:.1f}%", "neut"))

    # ── Verdict ────────────────────────────────────────────────────────────────
    if   score >= 4: trend, clr, meter = "STRONG BULLISH",  "#22c55e", 5
    elif score >= 2: trend, clr, meter = "BULLISH",         "#4ade80", 4
    elif score >= 1: trend, clr, meter = "MILDLY BULLISH",  "#86efac", 3
    elif score <= -4:trend, clr, meter = "STRONG BEARISH",  "#ef4444", -5
    elif score <= -2:trend, clr, meter = "BEARISH",         "#f87171", -4
    elif score <= -1:trend, clr, meter = "MILDLY BEARISH",  "#fca5a5", -3
    else:            trend, clr, meter = "NEUTRAL",          "#94a3b8", 0

    return {
        "trend": trend, "color": clr, "score": score,
        "signals": signals,
        "resistance": max_c_sp, "support": max_p_sp,
    }
