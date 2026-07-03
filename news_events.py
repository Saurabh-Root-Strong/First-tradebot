"""
news_events.py — Event-driven news layer (signal Layer 11).

The desk does NOT trade "news sentiment". It trades EVENT SURPRISES and the
market's reaction to them. This module follows the institutional pipeline:

    feed  →  event detection  →  impact score (−10..+10)  →  bias  →  (quant rules)

NOT  "article → LLM summary → trade"  (too slow, too noisy).

Three things only:
  1. FETCH structured events from real feeds (NSE corporate announcements; RBI
     press releases; a pluggable macro-surprise injector). Each source is wrapped
     so one dead feed degrades gracefully — same contract as nse_oi.py.
  2. SCORE each event with a deterministic keyword rule-book → an Impact Score in
     [-10, +10] and a canonical Bias. No LLM in the hot path; the rule-book is the
     audit trail ("SEBI order → −9 → BEARISH"). LLMs, if ever added, only classify
     ambiguous text into the SAME event types — never emit a trade.
  3. PERSIST to a lock-free per-day mirror (data/intraday/live/<date>_news_events
     .parquet) and expose analyze_news() / news_alerts() for the dashboard + engine.

Scope tags
  MACRO   market-wide  (RBI, CPI, Fed, crude …)   → moves all 4 indices
  SECTOR  basket-wide  (USFDA, China stimulus …)  → moves a sector basket
  STOCK   single name  (SEBI order, buyback …)    → moves one ticker

CAVEAT: NSE blocks datacenter IPs (same as nse_oi). Works from a home/office IP;
from the cloud VM the NSE source may 403 — RBI / macro injection still populate.

  .venv\\Scripts\\python.exe news_events.py            # one fetch, print top alerts
  .venv\\Scripts\\python.exe news_events.py --poll     # background capture loop
  .venv\\Scripts\\python.exe news_events.py --selftest # score-book sanity checks
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import re
import sys
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from typing import Optional

import requests

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from core.constants import IST, LIVE_DIR, today_iso as _today  # noqa: E402
from signal_types import Bias, bias_of  # noqa: E402

# ── Scope tags ──────────────────────────────────────────────────────────────────
MACRO, SECTOR, STOCK = "MACRO", "SECTOR", "STOCK"

# ── Impact score rule-book ───────────────────────────────────────────────────────
# (compiled regex, event_type, base_score, scope). Score sign = direction, magnitude
# = conviction. First match per text wins within a category; the strongest abs score
# across all matches is the event's score. Tables are the audit trail — tune here, not
# in 12 call-sites. Patterns are word-boundary-ish and case-insensitive.
def _rx(*words: str) -> "re.Pattern[str]":
    return re.compile(r"(?i)\b(?:" + "|".join(words) + r")")


# STOCK — negative (the desk's priority: avoid longs into these)
_STOCK_NEG = [
    (_rx(r"sebi\s+(order|action|ban|investigat|show.?cause|penalt)"), "SEBI action",        -9, STOCK),
    (_rx(r"\bfraud\b", r"forensic\s+audit", r"misappropriat", r"siphon"),  "Fraud allegation",   -9, STOCK),
    (_rx(r"auditor\s+resign", r"resignation\s+of\s+(\w+\s+){0,3}auditor"), "Auditor resignation",-8, STOCK),
    (_rx(r"\bcfo\b.*resign", r"resignation.*\bcfo\b",
         r"(md|ceo|chairman).*resign", r"resignation.*\b(md|ceo)\b"),       "Key-exec resignation",-7, STOCK),
    (_rx(r"credit\s+(rating\s+)?downgrade", r"downgrad", r"rating\s+cut",
         r"\bdefault\b", r"\bnpa\b\s+spike", r"insolvenc", r"\bnclt\b"),    "Credit downgrade",   -8, STOCK),
    (_rx(r"regulatory\s+ban", r"\bban\b\s+on", r"licen[sc]e\s+(cancel|revok)",
         r"product\s+recall", r"plant\s+shutdown"),                        "Regulatory ban",     -8, STOCK),
    (_rx(r"profit\s+(fell|declin|drop|down|miss)", r"earnings\s+miss",
         r"loss\s+widen", r"misses\s+estimat", r"weak\s+(results|guidance)",
         r"guidance\s+cut", r"profit\s+warning"),                          "Earnings miss",      -7, STOCK),
    (_rx(r"promoter.*(pledg|sold|stake\s+sale|offloaded)",
         r"stake\s+sale\s+by\s+promoter", r"block\s+deal.*sell"),          "Promoter selling",   -6, STOCK),
    (_rx(r"gst\s+(notice|demand)", r"tax\s+demand", r"income.?tax\s+(raid|search)",
         r"\bed\b\s+(raid|search|summon)", r"\bcbi\b\s+(raid|probe)"),     "Tax/enforcement",    -6, STOCK),
]
# STOCK — positive
_STOCK_POS = [
    (_rx(r"\bbuyback\b", r"buy.?back\s+of\s+shares"),                      "Buyback",            +8, STOCK),
    (_rx(r"large\s+order", r"order\s+win", r"bags?\s+(order|contract|deal)",
         r"awarded\s+(a\s+)?(contract|order|project)", r"\bloi\b",
         r"letter\s+of\s+(award|intent)", r"order\s+inflow"),             "Large order win",    +7, STOCK),
    (_rx(r"acqui(re|sition)", r"merger", r"to\s+buy\s+\d", r"takeover",
         r"strategic\s+stake"),                                            "Acquisition",        +7, STOCK),
    (_rx(r"promoter.*(buy|acqui|increas.*stake|infus)", r"insider\s+buying",
         r"open\s+market\s+purchase\s+by\s+promoter"),                     "Promoter buying",    +6, STOCK),
    (_rx(r"capacity\s+expansion", r"capex\s+(plan|of)", r"new\s+plant",
         r"greenfield", r"brownfield", r"commission(ed|ing)\s+of"),        "Capacity expansion", +6, STOCK),
    (_rx(r"record\s+(profit|revenue)", r"profit\s+(rose|surg|jump|up\s+\d|doubl)",
         r"beats?\s+estimat", r"strong\s+(results|earnings|guidance)",
         r"margin\s+expansion", r"revenue\s+up"),                          "Strong earnings",    +7, STOCK),
    (_rx(r"special\s+dividend", r"dividend\s+(hike|increas|of\s+\d)",
         r"bonus\s+(issue|share)", r"stock\s+split"),                      "Dividend/bonus",     +5, STOCK),
    (_rx(r"upgrade\s+by", r"rating\s+upgrade", r"credit\s+(rating\s+)?upgrade"),"Credit upgrade", +6, STOCK),
]
# SECTOR — keyword carries an implied direction for that basket
_SECTOR = [
    (_rx(r"usfda.*(observation|483|warning\s+letter|import\s+alert)",
         r"\b483\b", r"import\s+alert"),                                    "USFDA observation",  -7, SECTOR),
    (_rx(r"usfda.*(approval|clearance)", r"anda\s+approv", r"drug\s+approv"),"Drug approval",     +6, SECTOR),
    (_rx(r"china\s+stimulus", r"pboc\s+(cut|stimulus|easing)",
         r"china\s+(property|infra)\s+package"),                           "China stimulus",     +6, SECTOR),
    (_rx(r"\bcrr\b\s+(cut|reduc)", r"\bslr\b\s+cut", r"liquidity\s+boost"), "CRR/liquidity ease", +6, SECTOR),
    (_rx(r"\bnpa\b\s+(norm|provision|tighten)", r"asset\s+quality\s+review",
         r"banking\s+regulation\s+tighten"),                               "NPA/banking curb",   -5, SECTOR),
    (_rx(r"us\s+recession", r"it\s+spending\s+(cut|slowdown|weak)",
         r"discretionary\s+spend\s+weak"),                                 "IT demand weakness", -5, SECTOR),
    (_rx(r"\bai\b\s+(capex|spending)\s+(surge|boom|ramp)",
         r"deal\s+wins?\s+(surge|strong)\s+it"),                           "IT/AI demand",       +5, SECTOR),
    (_rx(r"steel\s+(price|duty)\s+(hike|safeguard)", r"metal\s+price\s+surg",
         r"commodity\s+price\s+(surge|rally)"),                            "Metals tailwind",    +5, SECTOR),
    (_rx(r"export\s+(ban|duty)", r"windfall\s+tax"),                       "Export curb",        -5, SECTOR),
]
# MACRO — headline keyword (surprise-aware scoring is separate, see score_surprise)
_MACRO = [
    (_rx(r"emergency\s+rate", r"surprise\s+(hike|rate)", r"inter.?meeting\s+hike"),"Emergency hike",-9, MACRO),
    (_rx(r"\brbi\b.*\bhike", r"repo\s+rate\s+hike", r"rate\s+hike"),       "Rate hike",          -6, MACRO),
    (_rx(r"\brbi\b.*\bcut", r"repo\s+rate\s+cut", r"rate\s+cut"),          "Rate cut",           +6, MACRO),
    (_rx(r"cpi.*(above|hotter|higher\s+than|surg)", r"inflation\s+(spike|surg|hotter)",
         r"wpi.*(surg|higher)"),                                           "Inflation hot",      -6, MACRO),
    (_rx(r"cpi.*(below|cooler|lower\s+than|eas)", r"inflation\s+(cool|eas|fell)"),"Inflation cool",+5, MACRO),
    (_rx(r"\bgdp\b.*(beat|stronger|above|surg)", r"growth\s+(beat|surg|accelerat)"),"GDP beat",   +5, MACRO),
    (_rx(r"\bgdp\b.*(miss|slow|below|contract)", r"growth\s+(miss|slow|contract)"),"GDP miss",    -6, MACRO),
    (_rx(r"crude\s+(spike|surg|jump|shock)", r"oil\s+price\s+(spike|surg|shock)",
         r"opec\s+cut"),                                                    "Crude shock",        -6, MACRO),
    (_rx(r"crude\s+(crash|fell|slump|drop)", r"oil\s+price\s+(crash|slump|fell)"),"Crude relief",  +4, MACRO),
    (_rx(r"non.?farm.*(beat|strong|above)", r"\bnfp\b.*(beat|strong)"),    "Strong NFP",         -4, MACRO),
    (_rx(r"fed.*hawkish", r"hawkish\s+fed", r"fomc.*hike"),                "Hawkish Fed",        -6, MACRO),
    (_rx(r"fed.*dovish", r"dovish\s+fed", r"fed.*(cut|pause|pivot)"),      "Dovish Fed",         +6, MACRO),
    (_rx(r"gst\s+collection.*(record|high|surg)"),                        "GST collections up", +3, MACRO),
    (_rx(r"fiscal\s+deficit.*(widen|overshoot|breach)"),                  "Fiscal slippage",    -4, MACRO),
]

_ALL_TABLES = _STOCK_NEG + _STOCK_POS + _SECTOR + _MACRO


# ── Event record ─────────────────────────────────────────────────────────────────
@dataclass
class NewsEvent:
    ts: datetime.datetime           # event time (IST, source's own time when available)
    source: str                     # NSE | RBI | MACRO | …
    scope: str                      # MACRO | SECTOR | STOCK
    ticker: str                     # NSE symbol, sector name, or "" for macro
    headline: str
    event_type: str                 # matched rule label ("SEBI action", …) or "Uncategorised"
    score: int                      # impact score, −10..+10
    url: str = ""
    uid: str = field(default="")    # stable dedupe id

    @property
    def bias(self) -> Bias:
        return Bias.BULL if self.score > 0 else Bias.BEAR if self.score < 0 else Bias.NEUTRAL

    @property
    def bucket(self) -> str:
        """Trader-facing lens for the panel tabs: BULLISH / BEARISH by score sign
        (NEUTRAL if unscored)."""
        return "BULLISH" if self.score > 0 else "BEARISH" if self.score < 0 else "NEUTRAL"


def _uid(source: str, ticker: str, headline: str) -> str:
    return hashlib.sha1(f"{source}|{ticker}|{headline}".encode("utf-8", "ignore")).hexdigest()[:16]


# ── Severe-negative lens (fraud / insolvency / audit-exit class) ─────────────────
# MEASURED 2026-07-03 (backtest_news_short.py, 5 weeks of capture, 43 priced events):
# severe negatives drift DOWN next day (−0.57% mean, 60% fall) but the drift lives
# in NON-F&O smallcaps (−0.85%, 67% fall, t≈−1.9) which have NO practical short
# route (no stock futures; SLB is 1-month-minimum and illiquid; frauds migrate to
# T2T/circuits). The F&O names — the only actually shortable ones — went UP
# (+0.86% next day, n=7: big names get dip-bought). So the honest product today is
# a DISCIPLINE tag (don't buy the dip / exit longs), NOT a short signal. The daily
# news mirrors ARE the accumulating ledger — re-run the study as n grows before
# promoting anything to a trade.
SEVERE_NEG = {"Fraud allegation", "SEBI action", "Auditor resignation",
              "Credit downgrade", "Regulatory ban", "Key-exec resignation"}
_SEVERE_TH = -7          # score at/below this + a SEVERE_NEG category = severe

# POSITIVE side, MEASURED 2026-07-03 (same study, --side pos, n=1734 priced — a real
# sample): buying the NEXT OPEN after big positive news LOSES −0.16% +1d, 57% down,
# t=−2.57 (significant) — the pop is in the opening gap, chasing it buys the fade.
# "Large order win" pops +0.5% day 1 then REVERSES to −1.6% by day 5 (78% down).
# F&O large caps: flat/negative at every horizon (priced instantly). The +5d smallcap
# mean (+0.7%, t=4.3) is a pure lottery tail — median 0.0%. So the positive badge is
# also DISCIPLINE copy: the news is real, the entry isn't — don't chase the open.
SEVERE_POS = {"Buyback", "Large order win", "Acquisition", "Promoter buying",
              "Capacity expansion", "Strong earnings"}
_SEVERE_POS_TH = 6       # score at/above this + a SEVERE_POS category = severe

_FNO_TICKERS: "set[str] | None" = None


def _fno_tickers() -> "set[str]":
    """Bare NSE tickers with stock futures (lazy; empty set if universe missing)."""
    global _FNO_TICKERS
    if _FNO_TICKERS is None:
        try:
            from fno_universe import STOCK_SYMBOLS
            _FNO_TICKERS = {s.split(":")[1].rsplit("-", 1)[0] for s in STOCK_SYMBOLS}
        except Exception:
            _FNO_TICKERS = set()
    return _FNO_TICKERS


def severe_tag(e: "NewsEvent") -> "str | None":
    """Discipline tag for severe stock events; None for everything else.
    Negative: 'FUT' (F&O name — futures exist, short measured UNSUPPORTED) or
              'AVOID' (no short route — don't buy the dip / exit longs).
    Positive: 'POS_FUT' (F&O — priced instantly, no edge measured) or
              'POS' (pop-fades — don't chase the open)."""
    if e.scope != STOCK:
        return None
    if e.score <= _SEVERE_TH and e.event_type in SEVERE_NEG:
        return "FUT" if e.ticker.upper() in _fno_tickers() else "AVOID"
    if e.score >= _SEVERE_POS_TH and e.event_type in SEVERE_POS:
        return "POS_FUT" if e.ticker.upper() in _fno_tickers() else "POS"
    return None


# ── Scoring ──────────────────────────────────────────────────────────────────────
def score_text(text: str) -> "tuple[str, int, str]":
    """Score a headline/announcement → (event_type, score, scope).

    Deterministic rule-book: scan every pattern, keep the match with the largest
    |score| (the most market-moving interpretation). Returns ("Uncategorised", 0,
    STOCK) when nothing matches — uncategorised events are stored but never alert.
    """
    text = text or ""
    best: "tuple[str, int, str] | None" = None
    for rx, etype, sc, scope in _ALL_TABLES:
        if rx.search(text) and (best is None or abs(sc) > abs(best[1])):
            best = (etype, sc, scope)
    return best or ("Uncategorised", 0, STOCK)


def score_surprise(event: str, actual: float, expected: float,
                   higher_is_bullish: bool = False) -> dict:
    """Macro surprise → structured signal. 'The surprise is more important than the
    headline.' Sign of (actual − expected) × orientation drives the bias; magnitude
    is scaled by the relative surprise and capped at ±10.

    higher_is_bullish=False (default) suits inflation / rates / deficit, where a hot
    print (actual > expected) is BEARISH for equities. Set True for GDP / IIP / GST.
    """
    surprise = actual - expected
    rel = surprise / (abs(expected) if expected else 1.0)
    raw = 10.0 * max(-1.0, min(1.0, rel * 3.0))          # ±33% surprise saturates
    if not higher_is_bullish:
        raw = -raw
    score = int(round(max(-10, min(10, raw))))
    return {"event": event, "actual": actual, "expected": expected,
            "surprise": round(surprise, 4), "score": score,
            "impact": Bias.BULL.value if score > 0 else Bias.BEAR.value if score < 0
                      else Bias.NEUTRAL.value}


# ── Feeds ────────────────────────────────────────────────────────────────────────
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/companies-listing/corporate-filings-announcements",
}
_NSE_HOME = "https://www.nseindia.com"
_NSE_ANN  = "https://www.nseindia.com/api/corporate-announcements?index=equities"
# RBI press releases + notifications (verified live RSS — the website.rbi.org.in
# path 404s; these classic *_rss.xml endpoints return 10 items each).
_RBI_RSS  = ("https://www.rbi.org.in/pressreleases_rss.xml",
             "https://www.rbi.org.in/notifications_rss.xml")

_nse_session: "requests.Session | None" = None
_lock = threading.Lock()


def _nse_new_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(_HEADERS)
    s.get(_NSE_HOME, timeout=12)                  # prime cookies (NSE needs the handshake)
    return s


def _parse_nse_dt(s: str) -> datetime.datetime:
    """NSE announcement time, e.g. '20-Jun-2026 09:57:44' → IST datetime (now on fail)."""
    for fmt in ("%d-%b-%Y %H:%M:%S", "%d-%b-%Y %H:%M", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.datetime.strptime(s.strip(), fmt).replace(tzinfo=IST)
        except Exception:
            continue
    return datetime.datetime.now(tz=IST)


def _row_to_event(d: dict) -> "Optional[NewsEvent]":
    """One NSE announcement JSON row → a scored NewsEvent (None if empty)."""
    sym  = (d.get("symbol") or "").strip()
    subj = (d.get("desc") or d.get("subject") or "").strip()
    body = (d.get("attchmntText") or d.get("smIndustry") or "").strip()
    head = (subj + " — " + body).strip(" —") if body else subj
    if not sym and not head:
        return None
    etype, sc, _scope = score_text(head)
    ts = _parse_nse_dt(d.get("an_dt") or d.get("dt") or d.get("sort_date") or "")
    return NewsEvent(ts=ts, source="NSE", scope=STOCK, ticker=sym,
                     headline=head[:300], event_type=etype, score=sc,
                     url=(d.get("attchmntFile") or ""), uid=_uid("NSE", sym, head))


def _nse_get(url: str) -> "list[dict]":
    """GET an NSE announcements URL with cookie-primed session; [] on block/fail."""
    global _nse_session
    for _ in range(2):
        try:
            if _nse_session is None:
                _nse_session = _nse_new_session()
            r = _nse_session.get(url, timeout=15)
            if r.status_code in (401, 403):
                _nse_session = None
                continue
            if r.status_code != 200:
                return []
            rows = r.json()
            return rows.get("data", rows.get("rows", [])) if isinstance(rows, dict) else (rows or [])
        except Exception:
            _nse_session = None
            time.sleep(0.8)
    return []


def fetch_nse_announcements() -> "list[NewsEvent]":
    """NSE corporate announcements (live filings, today). Scored STOCK events; [] on block."""
    return [e for e in (_row_to_event(d) for d in _nse_get(_NSE_ANN)) if e]


def backfill_nse(days: int = 14) -> dict:
    """Pull NSE announcements over the last `days` (date-range API) and write per-day
    mirrors bucketed by each filing's own date — so the date-nav has real history to
    scroll today, instead of waiting for daily capture to accumulate. Idempotent
    (uid-deduped per day). Returns {days_written, events}."""
    to_d   = datetime.datetime.now(tz=IST).date()
    from_d = to_d - datetime.timedelta(days=days)
    url = (f"{_NSE_ANN}&from_date={from_d:%d-%m-%Y}&to_date={to_d:%d-%m-%Y}")
    rows = _nse_get(url)
    by_day: "dict[str, list[NewsEvent]]" = {}
    for d in rows:
        e = _row_to_event(d)
        if e:
            by_day.setdefault(e.ts.date().isoformat(), []).append(e)
    written = 0
    for day, evs in by_day.items():
        if _persist_for(day, evs):
            written += 1
    return {"days_written": written, "events": sum(len(v) for v in by_day.values()),
            "dates": sorted(by_day)}


def fetch_rbi() -> "list[NewsEvent]":
    """RBI press releases + notifications RSS → scored MACRO events. [] on any failure."""
    out: "list[NewsEvent]" = []
    seen: set[str] = set()
    for url in _RBI_RSS:
        try:
            r = requests.get(url, headers={"User-Agent": _HEADERS["User-Agent"]}, timeout=12)
            if r.status_code != 200 or not r.content:
                continue
            root = ET.fromstring(r.content)
        except Exception:
            continue
        for item in root.iter("item"):
            title = (item.findtext("title") or "").strip()
            link  = (item.findtext("link") or "").strip()
            pub   = (item.findtext("pubDate") or "").strip()
            uid   = _uid("RBI", "", title)
            if not title or uid in seen:
                continue
            seen.add(uid)
            etype, sc, scope = score_text(title)
            try:
                ts = datetime.datetime.strptime(pub[:25], "%a, %d %b %Y %H:%M:%S").replace(tzinfo=IST)
            except Exception:
                ts = datetime.datetime.now(tz=IST)
            out.append(NewsEvent(
                ts=ts, source="RBI", scope=MACRO if scope == STOCK else scope, ticker="",
                headline=title[:300], event_type=etype, score=sc, url=link, uid=uid))
    return out


def inject_macro_surprise(event: str, actual: float, expected: float,
                          higher_is_bullish: bool = False) -> NewsEvent:
    """Record a macro economic-calendar surprise (CPI/GDP/Fed/NFP/…) as a scored MACRO
    event. Use when a print lands — the surprise vs consensus is what moves the tape."""
    s = score_surprise(event, actual, expected, higher_is_bullish)
    head = f"{event}: actual {actual} vs est {expected} (surprise {s['surprise']:+})"
    ev = NewsEvent(ts=datetime.datetime.now(tz=IST), source="MACRO", scope=MACRO, ticker="",
                   headline=head, event_type=event, score=s["score"],
                   uid=_uid("MACRO", event, head))
    _persist([ev])
    return ev


# ── Persistence (lock-free per-day mirror, same pattern as nse_oi) ───────────────
def _persist_for(date: str, events: "list[NewsEvent]") -> int:
    """Append new (uid-deduped) events to <date>_news_events.parquet. Returns rows added."""
    if not events:
        return 0
    import pandas as pd
    p = LIVE_DIR / f"{date}_news_events.parquet"
    with _lock:
        LIVE_DIR.mkdir(parents=True, exist_ok=True)
        old = None
        seen: set[str] = set()
        if p.exists():
            try:
                old = pd.read_parquet(p)
                seen = set(old["uid"].astype(str))
            except Exception:
                old = None
        fresh = [e for e in events if e.uid not in seen]
        if not fresh:
            return 0
        df = pd.DataFrame([asdict(e) for e in fresh])
        df["scope"] = df["scope"].astype(str)
        if old is not None:
            df = pd.concat([old, df], ignore_index=True)
        df.to_parquet(p, index=False)
    return len(fresh)


def _persist(events: "list[NewsEvent]") -> int:
    """Persist live events into today's mirror (back-compat wrapper)."""
    return _persist_for(_today(), events)


def record() -> int:
    """Fetch all live feeds once, score, persist deduped. Returns new rows added."""
    events: "list[NewsEvent]" = []
    events += fetch_nse_announcements()
    events += fetch_rbi()
    return _persist(events)


# ── Read / analyze (dashboard + engine consumers) ────────────────────────────────
def available_dates() -> "list[str]":
    """ISO dates (newest first) for which a news mirror exists — drives date nav."""
    out = sorted((p.name[:10] for p in LIVE_DIR.glob("*_news_events.parquet")
                  if p.stat().st_size >= 100), reverse=True)
    today = _today()
    if today not in out:                              # always offer today (may be empty)
        out = [today] + out
    return out


def _load_day(date: "str | None" = None) -> "list[NewsEvent]":
    import pandas as pd
    p = LIVE_DIR / f"{date or _today()}_news_events.parquet"
    if not p.exists() or p.stat().st_size < 100:
        return []
    try:
        df = pd.read_parquet(p)
    except Exception:
        return []
    out: "list[NewsEvent]" = []
    for _, r in df.iterrows():
        ts = pd.to_datetime(r["ts"])
        ts = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else r["ts"]
        out.append(NewsEvent(
            ts=ts, source=str(r["source"]), scope=str(r["scope"]), ticker=str(r.get("ticker") or ""),
            headline=str(r["headline"]), event_type=str(r["event_type"]),
            score=int(r["score"]), url=str(r.get("url") or ""), uid=str(r.get("uid") or "")))
    return out


def _dedup_stock(alerts: "list[NewsEvent]") -> "tuple[list[NewsEvent], dict[str, int]]":
    """Collapse repeated STOCK filings: one (ticker, event_type) shows ONCE — its
    FIRST sighting (a company re-files the same NCLT/downgrade story many times a
    day; the desk cares when the story STARTED and when it CHANGES, not each
    re-filing). Returns (kept, repeat-count by uid). MACRO/SECTOR left untouched
    (distinct headlines carry distinct content there)."""
    kept: "list[NewsEvent]" = []
    reps: "dict[str, int]" = {}
    first: "dict[tuple, NewsEvent]" = {}
    for e in sorted(alerts, key=lambda e: e.ts):
        if e.scope == STOCK and e.ticker:
            k = (e.ticker.upper(), e.event_type)
            if k in first:
                reps[first[k].uid] = reps.get(first[k].uid, 1) + 1
                continue
            first[k] = e
        kept.append(e)
    return kept, reps


def analyze_news(min_abs: int = 5, limit: int = 25, date: "str | None" = None,
                 order: str = "impact", dedup: bool = True) -> dict:
    """A day's scored events → the cockpit view-model (default today; date= for nav).

    Returns {alerts, macro_bias, by_scope, n_total, date, dates} where `alerts` is the
    market-moving subset (|score| >= min_abs) and `macro_bias` aggregates MACRO+SECTOR
    events into one market-wide tilt.

    order  : "impact" — most-impactful-then-recent first, capped at `limit` (cockpit
             glance default); "time" — CHRONOLOGICAL, uncapped: the full session
             tape, review a whole day start-to-finish.
    dedup  : collapse repeated STOCK (ticker, event_type) filings to the FIRST
             sighting; each alert carries n_rep (how many times it re-filed).
    """
    day = date or _today()
    evs = _load_day(day)
    alerts = [e for e in evs if abs(e.score) >= min_abs]
    reps: "dict[str, int]" = {}
    if dedup:
        alerts, reps = _dedup_stock(alerts)
    if order == "time":
        alerts.sort(key=lambda e: e.ts)               # chronological session tape
        limit = 0                                     # full day, no cap
    else:
        # Rank: impact magnitude first, then recency.
        alerts.sort(key=lambda e: (abs(e.score), e.ts), reverse=True)
    macro = [e for e in evs if e.scope in (MACRO, SECTOR) and e.score]
    net = sum(e.score for e in macro)
    macro_bias = (Bias.BULL if net > 2 else Bias.BEAR if net < -2 else Bias.NEUTRAL).value
    by_scope = {sc: sum(1 for e in alerts if e.scope == sc) for sc in (MACRO, SECTOR, STOCK)}
    shown = alerts[:limit] if limit else alerts
    by_bucket = {b: sum(1 for e in shown if e.bucket == b)
                 for b in ("BULLISH", "BEARISH")}
    return {
        "alerts": [asdict(e) | {"bias": e.bias.value, "bucket": e.bucket,
                                "severe": severe_tag(e),
                                "n_rep": reps.get(e.uid, 1)} for e in shown],
        "macro_bias": macro_bias, "macro_net": net,
        "by_scope": by_scope, "by_bucket": by_bucket,
        "n_total": len(evs), "n_alerts": len(alerts),
        "date": day, "dates": available_dates(),
        "as_of": datetime.datetime.now(tz=IST).strftime("%H:%M:%S"),
    }


def ticker_alert(ticker: str, min_abs: int = 6) -> "Optional[dict]":
    """Strongest market-moving event for one NSE ticker today (for trade-time veto).
    e.g. a −9 SEBI order on the symbol you're about to go long → block it."""
    best = None
    for e in _load_day():
        if e.ticker.upper() == ticker.upper() and abs(e.score) >= min_abs:
            if best is None or abs(e.score) > abs(best.score):
                best = e
    return (asdict(best) | {"bias": best.bias.value}) if best else None


# ── Stock-news → index tilt (constituent aggregation) ────────────────────────────
# Indices are baskets: a cluster of bad bank-stock filings SHOULD tilt BANKNIFTY.
# Weights are approximate index weights (cap-weighted) so a heavyweight dominates and
# a minor name can't single-handedly veto an index — critical for live-money safety.
# Stable ~2x/year; ticker = NSE symbol as it appears in corporate-announcements.
_BANK_W = {                                   # NIFTY BANK (≈ sums to 1.0)
    "HDFCBANK": 0.28, "ICICIBANK": 0.24, "SBIN": 0.09, "AXISBANK": 0.09,
    "KOTAKBANK": 0.08, "INDUSINDBK": 0.05, "PNB": 0.03, "BANKBARODA": 0.03,
    "AUBANK": 0.03, "FEDERALBNK": 0.03, "IDFCFIRSTB": 0.03, "BANDHANBNK": 0.02,
}
_FIN_W = {                                    # NIFTY FINANCIAL SERVICES
    "HDFCBANK": 0.30, "ICICIBANK": 0.20, "AXISBANK": 0.08, "SBIN": 0.08,
    "KOTAKBANK": 0.07, "BAJFINANCE": 0.07, "BAJAJFINSV": 0.04, "SBILIFE": 0.03,
    "HDFCLIFE": 0.03, "SHRIRAMFIN": 0.03, "PFC": 0.02, "RECLTD": 0.02,
    "CHOLAFIN": 0.02, "ICICIPRULI": 0.01,
}
_INDEX_CONSTITUENTS = {
    "NSE:NIFTYBANK-INDEX": _BANK_W,
    "NSE:FINNIFTY-INDEX":  _FIN_W,
}


def index_news_tilt(fyers_sym: str, date: "str | None" = None, min_abs: int = 6) -> dict:
    """Aggregate today's STOCK events for an index's constituents → one weighted tilt
    in ≈[-10, +10] (same scale as a single event score), plus the dominant name.

    Weighted by approximate index weight so a heavyweight's −9 SEBI order tilts the
    basket while a micro-name's noise does not. NIFTY (no constituent map) uses a
    broad-market net of all strong stock events, heavily damped (diffuse signal).
    Returns {net, n, top}: top = (ticker, score, event_type) or None.
    """
    weights = _INDEX_CONSTITUENTS.get(fyers_sym)
    if not weights:
        # NIFTY/MIDCAP: no clean basket here. Deliberately NOT a broad-market net of
        # all filings — corporate announcements self-select as good news (buybacks /
        # order wins), so an "average filing" tilt is structurally bullish = a
        # confound, not signal. These indices ride MACRO only. (User scope: BANK/FIN.)
        return {"net": 0.0, "n": 0, "top": None}
    evs = [e for e in _load_day(date) if e.scope == STOCK and abs(e.score) >= min_abs]
    contrib = [(weights[e.ticker.upper()] * e.score, e.ticker, e.score, e.event_type)
               for e in evs if e.ticker.upper() in weights]
    net = max(-10.0, min(10.0, sum(c[0] for c in contrib)))
    top = max(contrib, key=lambda x: abs(x[0]))[1:] if contrib else None
    return {"net": round(net, 1), "n": len(contrib), "top": top}


# ── Background poller (dashboard wires this as a daemon thread) ──────────────────
def poll_loop(interval: int = 120, market_hours_only: bool = False) -> None:
    """Capture loop — record() every `interval`s. Default runs all day (filings and RBI
    releases land pre-open and post-close too, unlike intraday OI)."""
    while True:
        now = datetime.datetime.now(tz=IST)
        in_window = datetime.time(8, 0) <= now.time() <= datetime.time(18, 0) and now.weekday() < 5
        if (not market_hours_only) or in_window:
            try:
                n = record()
                if n:
                    print(f"[news] {now:%H:%M:%S} +{n} events")
            except Exception as exc:
                print(f"[news] {now:%H:%M:%S} error: {exc}")
        time.sleep(interval)


# ── CLI ──────────────────────────────────────────────────────────────────────────
def _selftest() -> None:
    cases = [
        ("Board approves buyback of equity shares", "Buyback", 8),
        ("Company receives SEBI order in matter of disclosure", "SEBI action", -9),
        ("Resignation of Statutory Auditor", "Auditor resignation", -8),
        ("Bags order worth Rs 2400 cr from NHAI", "Large order win", 7),
        ("Q4 profit misses estimates, guidance cut", "Earnings miss", -7),
        ("RBI keeps repo rate unchanged; dovish Fed expected", "Dovish Fed", 6),
        ("USFDA issues Form 483 with 5 observations", "USFDA observation", -7),
        ("Just an ordinary intimation of board meeting", "Uncategorised", 0),
    ]
    ok = 0
    for text, want_type, want_score in cases:
        etype, sc, _ = score_text(text)
        good = (etype == want_type and sc == want_score)
        ok += good
        print(f"  [{'ok' if good else 'XX'}] {sc:+3d} {etype:22} | {text[:48]}")
    # surprise scorer
    s = score_surprise("US CPI", 3.5, 3.2)          # hot inflation → bearish
    assert s["score"] < 0, s
    s2 = score_surprise("India GDP", 7.8, 7.0, higher_is_bullish=True)
    assert s2["score"] > 0, s2
    print(f"  surprise CPI {s['score']:+d} (bearish ok), GDP {s2['score']:+d} (bullish ok)")
    print(f"\n  {ok}/{len(cases)} score-book cases passed")


def main() -> None:
    ap = argparse.ArgumentParser(description="Event-driven news layer")
    ap.add_argument("--poll", action="store_true", help="run the capture loop")
    ap.add_argument("--selftest", action="store_true", help="rule-book sanity checks")
    ap.add_argument("--backfill", type=int, metavar="DAYS",
                    help="backfill NSE announcements over the last DAYS into per-day mirrors")
    ap.add_argument("--interval", type=int, default=60)
    args = ap.parse_args()
    if args.selftest:
        _selftest(); return
    if args.backfill is not None:
        r = backfill_nse(args.backfill)
        print(f"Backfilled {r['events']} events across {r['days_written']} days: {r['dates']}")
        return
    if args.poll:
        poll_loop(args.interval); return
    n = record()
    print(f"Captured {n} new events.\n")
    a = analyze_news()
    print(f"Macro tilt: {a['macro_bias']} (net {a['macro_net']:+d})   "
          f"{a['n_alerts']} alerts / {a['n_total']} events   as of {a['as_of']}")
    for e in a["alerts"][:15]:
        tk = e["ticker"] or e["scope"]
        print(f"  {e['score']:+3d} {e['bias']:8} {tk:12} {e['event_type']:20} | {e['headline'][:60]}")
    if not a["alerts"]:
        print("  (no market-moving alerts — feeds may be blocked from this IP, or quiet)")


if __name__ == "__main__":
    main()
