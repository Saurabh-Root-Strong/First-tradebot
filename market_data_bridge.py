"""
market_data_bridge.py  --  Read institutional & cash market data from
D:\\Python Projects\\Daily_Cash_Market\\data\\market_data.duckdb

Tables available:
  fao_participant       FII / Client / Pro / DII long-short in index futures & options
  fii_derivatives_stats FII net buy/sell in NIFTY, BANKNIFTY, INDEX futures
  index_data            Daily OHLC + P/E + P/B for all NSE indices
  daily_data            Bhavcopy for all cash market stocks (prev close etc.)
  fno_bhavcopy          F&O bhavcopy: OI, volume, settle price for all contracts

Exposed API (all functions return {} / [] on any failure — never crash the dashboard):
  get_fii_positioning(index_sym)  →  dict with FII net futures, net options, bias
  get_fao_snapshot()              →  dict  FII / Client / Pro breakdown for index futures+options
  get_index_context(index_sym)    →  dict  yesterday close, PE ratio, gap-up/down vs live spot
  get_institutional_bias(index_sym) → (score, signals)  ready to plug into trade_setup.py
"""

import datetime
import sqlite3
import threading
import time
from pathlib import Path

from core.obs import warn_once   # observe silently-swallowed field-computation failures

DB_PATH   = Path(r"D:\Python Projects\Daily_Cash_Market\data\market_data.duckdb")
_LOCAL_DB = Path(__file__).parent / "data" / "tradebot_context.db"

# Map Tradebot symbol → names used in the DB
INDEX_NAMES = {
    "NSE:NIFTY50-INDEX":    {"fii_cat": "NIFTY FUTURES",    "idx_name": "Nifty 50"},
    "NSE:NIFTYBANK-INDEX":  {"fii_cat": "BANKNIFTY FUTURES","idx_name": "Nifty Bank"},
    "NSE:FINNIFTY-INDEX":   {"fii_cat": "FINNIFTY FUTURES", "idx_name": "Nifty Financial Services"},
    "NSE:MIDCPNIFTY-INDEX": {"fii_cat": "MIDCPNIFTY FUTURES","idx_name": "Nifty Midcap Select"},
}

_lock  = threading.Lock()
_cache: dict = {}
# EOD data (FII, FAO, OI) is published once after market close and never updated
# intraday.  30-min TTL matches daily_context_bridge's 4-hr TTL direction:
# both layers read from the same DuckDB, so keeping them on similar cadence
# prevents them from seeing different snapshots of the same underlying data.
_CACHE_TTL = 1800


def _db():
    """Open read-only DuckDB connection. Returns None if DB missing."""
    try:
        import duckdb
        return duckdb.connect(str(DB_PATH), read_only=True)
    except Exception:
        return None


def _cached(key: str, loader):
    """Simple TTL cache wrapper."""
    with _lock:
        c = _cache.get(key)
        if c and time.time() - c["ts"] < _CACHE_TTL:
            return c["data"]

    try:
        data = loader()
    except Exception:
        data = {}

    with _lock:
        _cache[key] = {"data": data, "ts": time.time()}
    return data


# ── FII net futures positioning (last 5 trading days trend) ──────────────────
def get_fii_positioning(index_sym: str) -> dict:
    """
    Returns:
      net_contracts     FII net today (buy - sell)
      net_value_cr      FII net Rs. crore
      oi_contracts      Total OI in this future
      net_5d            List of last-5 daily nets (most recent first)
      trend             'accumulating' | 'distributing' | 'neutral'
      bias              'bull' | 'bear' | 'neut'
      label             human-readable sentence
    """
    meta = INDEX_NAMES.get(index_sym)
    if not meta:
        return {}

    cat = meta["fii_cat"]

    def _load():
        con = _db()
        if not con:
            return {}
        rows = con.execute(
            "SELECT trade_date, buy_contracts - sell_contracts AS net, "
            "buy_value_cr - sell_value_cr AS net_val, oi_contracts "
            "FROM fii_derivatives_stats "
            "WHERE category = ? "
            "ORDER BY trade_date DESC LIMIT 5",
            [cat]
        ).fetchall()
        con.close()
        if not rows:
            return {}

        latest     = rows[0]
        net_today  = latest[1] or 0
        net_val    = round(latest[2] or 0, 2)
        oi         = latest[3] or 0
        net_5d     = [r[1] for r in rows]
        bull_days  = sum(1 for n in net_5d if n > 0)
        bear_days  = sum(1 for n in net_5d if n < 0)

        # Trend: 3+ consecutive in same direction
        if bull_days >= 3:
            trend, bias = "accumulating", "bull"
        elif bear_days >= 3:
            trend, bias = "distributing", "bear"
        else:
            trend, bias = "neutral", "neut"

        # Today's label
        sign = "+" if net_today >= 0 else ""
        if net_today > 5000:
            label = f"FII BUYING {cat}: {sign}{net_today:,} contracts (Rs.{sign}{net_val:.0f}Cr) — strong institutional LONG"
        elif net_today > 1000:
            label = f"FII buying {cat}: {sign}{net_today:,} contracts (Rs.{sign}{net_val:.0f}Cr) — mild institutional long"
        elif net_today < -5000:
            label = f"FII SELLING {cat}: {net_today:,} contracts (Rs.{net_val:.0f}Cr) — strong institutional SHORT"
        elif net_today < -1000:
            label = f"FII selling {cat}: {net_today:,} contracts (Rs.{net_val:.0f}Cr) — mild institutional short"
        else:
            label = f"FII {cat}: {sign}{net_today:,} contracts — neutral day"

        return {
            "net_contracts": net_today,
            "net_value_cr":  net_val,
            "oi_contracts":  oi,
            "net_5d":        net_5d,
            "trend":         trend,
            "bias":          bias,
            "label":         label,
            "category":      cat,
        }

    return _cached(f"fii_{index_sym}", _load)


# ── FAO participant snapshot (FII vs Retail vs Pro in index F&O) ──────────────
def get_fao_snapshot() -> dict:
    """
    Returns the most recent FAO participant OI breakdown.
    Key insight:
      FII net index futures long/short → institutional direction
      Client (retail) net → contrarian signal (retail is often wrong at extremes)
      When FII is net long AND Client is net short → strong bullish setup
      When FII is net short AND Client is net long → strong bearish setup
    """
    def _load():
        con = _db()
        if not con:
            return {}
        rows = con.execute(
            "SELECT trade_date, client_type, "
            "fut_idx_long, fut_idx_short, "
            "opt_idx_call_long, opt_idx_call_short, "
            "opt_idx_put_long, opt_idx_put_short "
            "FROM fao_participant WHERE data_type='OI' "
            "ORDER BY trade_date DESC, client_type LIMIT 8"
        ).fetchall()
        con.close()
        if not rows:
            return {}

        date    = rows[0][0]
        result  = {"date": str(date), "participants": {}}
        for r in rows:
            if str(r[0]) != str(date):
                continue   # only latest date
            ct = r[1]
            result["participants"][ct] = {
                "fut_idx_net":       (r[2] or 0) - (r[3] or 0),
                "fut_idx_long":      r[2] or 0,
                "fut_idx_short":     r[3] or 0,
                "call_idx_net":      (r[4] or 0) - (r[5] or 0),
                "put_idx_net":       (r[6] or 0) - (r[7] or 0),
            }
        return result

    return _cached("fao_snapshot", _load)


# ── Index context: yesterday's close, P/E, gap ───────────────────────────────
def get_index_context(index_sym: str, live_spot: float = 0) -> dict:
    """
    Returns:
      prev_close    last known EOD close from bhavcopy
      pe_ratio      current index P/E
      pb_ratio      current index P/B
      gap_pct       if live_spot provided: today's gap % from prev close
      gap_label     'GAP UP' | 'GAP DOWN' | 'FLAT'
      pe_signal     'expensive' | 'fair' | 'cheap'
    """
    meta = INDEX_NAMES.get(index_sym)
    if not meta:
        return {}

    idx_name = meta["idx_name"]

    def _load():
        con = _db()
        if not con:
            return {}
        row = con.execute(
            "SELECT trade_date, close_val, prev_close, pct_chg, pe_ratio, pb_ratio "
            "FROM index_data WHERE index_name = ? "
            "ORDER BY trade_date DESC LIMIT 1",
            [idx_name]
        ).fetchone()
        con.close()
        if not row:
            return {}

        pe = row[4] or 0
        if pe > 24:
            pe_signal = "expensive (PE>24, caution on new longs)"
        elif pe > 20:
            pe_signal = "fair value (PE 20-24)"
        elif pe > 0:
            pe_signal = "attractive (PE<20, supports buying dips)"
        else:
            pe_signal = "unknown"

        return {
            "date":       str(row[0]),
            "prev_close": row[2] or row[1],
            "eod_close":  row[1],
            "pct_chg":    row[3] or 0,
            "pe_ratio":   pe,
            "pb_ratio":   row[5] or 0,
            "pe_signal":  pe_signal,
        }

    data = _cached(f"idx_{index_sym}", _load)

    # Gap calculation against live spot (not cached — computed fresh)
    if live_spot and data.get("prev_close"):
        gap_pct = (live_spot - data["prev_close"]) / data["prev_close"] * 100
        if gap_pct > 0.3:
            gap_label = f"GAP UP {gap_pct:+.2f}%"
        elif gap_pct < -0.3:
            gap_label = f"GAP DOWN {gap_pct:+.2f}%"
        else:
            gap_label = f"FLAT open ({gap_pct:+.2f}%)"
        data = dict(data, gap_pct=round(gap_pct, 3), gap_label=gap_label)

    return data


# ── Composite institutional bias (score + signals, plug into trade_setup) ────
def get_institutional_bias(index_sym: str, live_spot: float = 0) -> tuple[float, list]:
    """
    Combines FII futures flow + FAO smart-money vs retail divergence + index valuation.
    Returns (score, signals) where score is in [-4, +4], same convention as other layers.

    Logic:
      FII net buy in futures  → score +1 to +3 (size-dependent)
      FII net sell in futures → score -1 to -3
      FII fut trend 3+ days   → additional ±0.5
      FII net short BUT retail net long → bearish divergence signal
      FII net long  AND retail net short → bullish confirmation
      PE expensive → mild negative for new longs
    """
    score   = 0.0
    signals = []

    # ── FII futures flow ──────────────────────────────────────────────────────
    fii = get_fii_positioning(index_sym)
    if fii:
        net = fii.get("net_contracts", 0)
        oi  = fii.get("oi_contracts", 1) or 1
        net_pct = net / oi * 100  # net as % of total OI

        if net_pct > 3:
            score += 3.0
            signals.append(("bull", fii["label"] + f" ({net_pct:+.1f}% of OI)"))
        elif net_pct > 1:
            score += 2.0
            signals.append(("bull", fii["label"] + f" ({net_pct:+.1f}% of OI)"))
        elif net_pct > 0:
            score += 0.5
            signals.append(("bull", fii["label"]))
        elif net_pct < -3:
            score -= 3.0
            signals.append(("bear", fii["label"] + f" ({net_pct:+.1f}% of OI)"))
        elif net_pct < -1:
            score -= 2.0
            signals.append(("bear", fii["label"] + f" ({net_pct:+.1f}% of OI)"))
        elif net_pct < 0:
            score -= 0.5
            signals.append(("bear", fii["label"]))
        else:
            signals.append(("neut", fii["label"]))

        # Multi-day trend bonus
        if fii.get("trend") == "accumulating":
            score += 0.5
            signals.append(("bull", f"FII futures trend: ACCUMULATING for 3+ days — sustained institutional buying"))
        elif fii.get("trend") == "distributing":
            score -= 0.5
            signals.append(("bear", f"FII futures trend: DISTRIBUTING for 3+ days — sustained institutional selling"))

    # ── FAO smart-money vs retail divergence ─────────────────────────────────
    # NSE publishes FAO participant data as ONE aggregate row covering ALL index
    # futures (NIFTY + BANKNIFTY + FINNIFTY + MIDCAP combined).  It cannot be
    # disaggregated per index.  Apply index-specific weights:
    #   NIFTY      → full weight  (NIFTY ≈ 70% of total index futures volume)
    #   BANKNIFTY  → 50% weight   (significant but not dominant share)
    #   FINNIFTY / MIDCAP → 30%   (minority share; aggregate can mislead)
    # FAO participant data = ALL index futures combined (NSE doesn't break it down).
    # NIFTY ~70% of index futures volume → aggregate is a close proxy.
    # BANKNIFTY ~25% → banking stocks strongly tracked by FII → 80% weight still valid.
    # FINNIFTY / MIDCAP → smaller share, less direct proxy → 50%.
    _FAO_WEIGHT = {
        "NSE:NIFTY50-INDEX":    1.0,
        "NSE:NIFTYBANK-INDEX":  0.8,
        "NSE:FINNIFTY-INDEX":   0.5,
        "NSE:MIDCPNIFTY-INDEX": 0.5,
    }
    fao_w   = _FAO_WEIGHT.get(index_sym, 0.7)
    agg_tag = "" if index_sym == "NSE:NIFTY50-INDEX" else " [All Index Futures — aggregate]"

    fao = get_fao_snapshot()
    parts = fao.get("participants", {})
    fii_p   = parts.get("FII",    {})
    client  = parts.get("Client", {})

    fii_fut_net    = fii_p.get("fut_idx_net",    0)
    client_fut_net = client.get("fut_idx_net",   0)

    if fii_fut_net > 0 and client_fut_net < 0:
        score += round(1.5 * fao_w, 2)
        signals.append(("bull",
            f"Smart-money DIVERGENCE: FII net LONG {fii_fut_net:,} + Retail SHORT {client_fut_net:,}"
            f"{agg_tag} — bullish setup"))
    elif fii_fut_net < 0 and client_fut_net > 0:
        score -= round(1.5 * fao_w, 2)
        signals.append(("bear",
            f"Smart-money DIVERGENCE: FII net SHORT {fii_fut_net:,} + Retail LONG {client_fut_net:,}"
            f"{agg_tag} — bearish setup"))
    elif fii_fut_net > 0 and client_fut_net > 0:
        score += round(0.5 * fao_w, 2)
        signals.append(("bull",
            f"FII and Retail both net long in index futures{agg_tag} — broad bullish consensus"))
    elif fii_fut_net < 0 and client_fut_net < 0:
        score -= round(0.5 * fao_w, 2)
        signals.append(("bear",
            f"FII and Retail both net short in index futures{agg_tag} — broad bearish consensus"))

    # FII options positioning (aggregate — same weight logic applies)
    fii_put_net  = fii_p.get("put_idx_net",  0)
    fii_call_net = fii_p.get("call_idx_net", 0)
    if fii_put_net > 100_000:
        score -= round(0.5 * fao_w, 2)
        signals.append(("bear",
            f"FII holding {fii_put_net:,} net PUT contracts{agg_tag} — heavy hedging, downside risk"))
    elif fii_call_net > 100_000:
        score += round(0.5 * fao_w, 2)
        signals.append(("bull",
            f"FII holding {fii_call_net:,} net CALL contracts{agg_tag} — institutional upside bets"))

    # ── Index valuation context ───────────────────────────────────────────────
    ctx = get_index_context(index_sym, live_spot)
    if ctx:
        pe = ctx.get("pe_ratio", 0)
        if pe > 24:
            score -= 0.5
            signals.append(("bear",
                f"Index P/E {pe:.1f} — {ctx['pe_signal']} — elevated valuation caps upside"))
        elif pe < 18 and pe > 0:
            score += 0.5
            signals.append(("bull",
                f"Index P/E {pe:.1f} — {ctx['pe_signal']} — valuation supports buying"))
        elif pe > 0:
            signals.append(("neut",
                f"Index P/E {pe:.1f} — {ctx['pe_signal']}"))

        if ctx.get("gap_label"):
            signals.append(("neut", f"Today vs prev close: {ctx['gap_label']}"))

    # ── Large-cap delivery quality (from nightly sync local snapshot) ─────────
    deliv = _get_delivery_context()
    avg_deliv  = deliv.get("avg_deliv_pct")
    high_count = deliv.get("high_deliv_count", 0) or 0
    if avg_deliv is not None:
        if avg_deliv > 65:
            score += 0.5
            signals.append(("bull",
                f"Large-cap delivery: {avg_deliv:.0f}% avg ({high_count} stocks >60%) "
                f"— institutional accumulation, holding positions overnight"))
        elif avg_deliv < 35:
            score -= 0.3
            signals.append(("bear",
                f"Large-cap delivery: {avg_deliv:.0f}% avg — "
                f"low conviction, intraday-heavy activity (no overnight holding)"))
        else:
            signals.append(("neut",
                f"Large-cap delivery: {avg_deliv:.0f}% avg — normal delivery quality"))

    return round(min(max(score, -4), 4), 2), signals


def _get_delivery_context() -> dict:
    """
    Read large-cap delivery quality from the nightly sync local snapshot.

    Accepts the most recent successful sync within the last 3 calendar days —
    NOT an exact today() match.  The sync runs at ~8:30 PM (sync_date = that
    evening's date); the live trading session is the NEXT morning, so an exact
    today() match would always miss and silently drop this sub-signal.  This
    mirrors daily_context_bridge._try_load_from_local's 3-day acceptance window.
    Returns {} when no local DB or no recent successful sync.
    """
    if not _LOCAL_DB.exists():
        return {}
    try:
        con = sqlite3.connect(str(_LOCAL_DB))
        row = con.execute("""
            SELECT mc.nifty50_avg_deliv_pct, mc.nifty50_high_deliv_count, mc.sync_date
            FROM market_context mc
            JOIN sync_metadata sm ON mc.sync_date = sm.sync_date
            WHERE sm.status = 'ok'
            ORDER BY mc.sync_date DESC
            LIMIT 1
        """).fetchone()
        con.close()
        if row and row[0] is not None:
            try:
                age_days = (datetime.date.today() -
                            datetime.date.fromisoformat(str(row[2])[:10])).days
            except Exception:
                age_days = 99
            if age_days <= 3:
                return {"avg_deliv_pct": row[0], "high_deliv_count": row[1]}
    except Exception as _e:
        warn_once(_e)
    return {}
