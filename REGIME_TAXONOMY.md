# Intraday Market Regime Taxonomy — NSE Index Futures & Options

> ## v2 — SHARPENED (the version that survives scrutiny)
>
> The 14-cell lattice below (v1) is the *map*. An automated engine that works runs on far
> less. After grounding on 2yr NIFTY 5m, the taxonomy collapses to **2 axes / 3 states**:
>
> **The engine is one asymmetric-loss binary: "is it safe to be short gamma for the next H?"**
> because the only cost-surviving intraday edge in NSE index options is the **Variance Risk
> Premium** (retail overpays → IV > RV). Direction is dead at the cost floor; the harvest is
> non-directional.
>
> **Axis 1 — RV-state (harvest detector).** Trailing realised-vol percentile + stability.
> **Axis 2 — Catalyst (jump veto).** Scheduled event / expiry / news-shock proximity.
> Everything else (direction, dealer-gamma proxy, skew, breadth) = secondary **sizing context**.
>
> **States:** `HARVEST` (sell gamma) · `DANGER` (flatten / stand aside) · `VETO` (event jump-risk).
>
> **Grounding (2yr NIFTY 5m, this repo's data):**
> - Vol persistence splits the chop **hard**: low-RV chop forward blow-up tail P(>0.40%/hr) =
>   **1.6%** vs high-RV chop **13.6%** — an **8.5×** separation. Trailing RV is the feature.
> - **High-RV chop (13.6%) is more dangerous than a trend day (7.6%)** — the whipsaw cell is the
>   single worst place to be short gamma. The "both-side SL hunt" is the #1 killer, quantified.
> - Reversion ≈ 50% in every bucket → the edge is **containment, not direction** (fade stays
>   dead; only a premium *seller* monetises a contained move).
> - Escapes from the harvest cell are **97% ramps, 1% jumps**; 83% give ≥2 bars lead → a
>   trailing-RV kill-switch catches them. The residual 1% jump tail = the event VETO's job.
>
> **The real target is not "low vol" — it's "high VRP."** The safest cell (lowest RV) also has
> the smallest premium, so cost (2-leg straddle ≈ 6% RT) eats it. Optimal harvest =
> **max( IV − forecast_RV )**: IV richly above a *confidently-low* realised forecast. The engine
> forecasts RV (persistence, proven doable above) and compares to captured chain IV.
>
> **The one untested leg:** containment (realised) is proven predictable. The **premium leg
> (is IV − RV fat enough to pay after the 1.6% tail + 6% RT cost?)** needs captured intraday
> option IV — ~6 fat days exist (`chain_snapshots.iv`). That single test decides if this is a
> live product. It is the next research dollar — not a better CE/PE arrow.

---

**Status:** design spec (foundation for an automated regime engine).
**Author seat:** Head of Quantitative Research — systematic index F&O.
**Scope:** NIFTY, BANKNIFTY, FINNIFTY, MIDCPNIFTY — intraday (09:15–15:30 IST).

---

## 0. The thesis, stated plainly

A regime is **not a label**. It is a **point in a multi-axis state space**, carried with a
**confidence** and a **time-of-day/expiry context**, that selects which *strategy family*
is allowed to trade and at what size.

The single biggest defect in the current system (`regime_classifier.py`: Kaufman ER →
BIG/SMALL trend or `CONSOLIDATION`) is that **`CONSOLIDATION` collapses four regimes that
require opposite actions**:

| looks like "chop" on price | true regime | correct action |
|---|---|---|
| tight range, **low** realised vol, dealers **long gamma** | **Quiet Range / Gamma Pin** | **SELL premium** (theta) |
| range, **high** realised vol, dealers **short gamma** | **Volatile Whipsaw** | **STAND ASIDE** (both-side SL hunt) |
| range **tightening**, vol compressing, vol-of-vol rising | **Compression Coil** | **BUY cheap gamma** (straddle) |
| range, **IV elevated/rising** into a catalyst | **Event Vacuum** | **STAND ASIDE / sell only if IV rich** |

Price-structure (ER/ADX) alone cannot separate these — they are identical on the tape. Only
the **volatility axis** and the **options-structure axis** split them. That split is the
entire reason to go multi-axis, and it is where index-option money actually is: **the system
has spent its whole life hunting the directional arrow (proven dead at the cost floor) while
never targeting the theta-sell and gamma-buy regimes that don't need a direction at all.**

---

## 1. Design principles

1. **Orthogonal axes, composed.** Classify each axis independently; the named regime is the
   *signature* across axes. Axes must be as uncorrelated as possible (direction ⊥ vol level ⊥
   gamma sign ⊥ liquidity ⊥ catalyst).
2. **Confidence is first-class.** Every axis emits `(state, confidence)`. Composite confidence
   falls when axes **conflict** or when we sit in a **transition zone**. A low-confidence
   regime maps to *stand aside / range-only*, never to leverage.
3. **Causal, no lookahead.** Same discipline as the rest of the repo (`read_mirror` ts≤as_of,
   drop the forming bar). Every estimator uses only closed, past data. The engine must run
   identically live (t=now) and in replay (t=scrubbed past).
4. **Hysteresis / anti-whipsaw.** Regime switches require persistence (N confirming bars or a
   z-threshold, cf. `swing_classifier._Z_CONFIRM=2σ`). The regime engine itself must not chop.
5. **Multi-timeframe nesting.** The *dominant* regime is set on the higher TF (60m/day); the
   lower TF (5/15m) is for timing within it. A 5m trend inside a 60m range is a fade, not a
   breakout.
6. **Honesty about tradability.** Most cells map to *don't trade directionally*. The engine's
   value is **regime-conditional risk sizing + identifying the few exploitable cells**, not a
   permission slip to trade every bar. The ~3% option round-trip + theta floor still rules
   every directional cell (see `project_intraday_scout` COST-FLOOR LAW).

---

## 2. The axes (the engine's primary inputs)

Each axis lists: **states**, **estimators**, **NSE calibration**, **data status** in *this* repo.

### Axis A — Directional Structure (trend vs balance)
- **States:** `TREND_UP_BIG`, `TREND_UP_SMALL`, `BALANCED/RANGE`, `TREND_DOWN_SMALL`,
  `TREND_DOWN_BIG`, `TRANSITION`.
- **Estimators:** Kaufman Efficiency Ratio (have); ADX/DI; Hurst exponent or variance-ratio
  (trend vs mean-revert discriminator); net drift in ATR units; higher-high/higher-low count;
  distance from VWAP and VWAP slope.
- **NSE calibration:** intraday ER runs low/noisy — calibrate per index (BANKNIFTY trends
  cleaner than MIDCAP). Window must be warm by ~late-morning (`_MOOD_WIN=10`).
- **Data:** ✅ built (`regime_classifier`, `price_structure.regime`).

### Axis B — Volatility State (level + direction)
- **States:** `COMPRESSED`, `NORMAL`, `ELEVATED`, `SHOCK`; crossed with `EXPANDING` /
  `STABLE` / `CONTRACTING`.
- **Estimators:** realised vol (close-to-close + Parkinson/Garman-Klass on OHLC) as a
  *percentile vs trailing intraday distribution*; ATR ratio (fast/slow); **vol-of-vol** (rolling
  std of realised vol) — the coil tell; gap size at open.
- **NSE calibration:** percentile bands per index and **per session-phase** (open vol ≫ lunch
  vol — never compare raw). India VIX as the cross-check / overnight anchor.
- **Data:** ✅ realised from OHLC/ticks; India VIX available.

### Axis C — Options Structure (the index-option edge axis)
- **Sub-states:**
  - **Dealer gamma sign:** `LONG_GAMMA` (dealers stabilise → pin / mean-revert) vs
    `SHORT_GAMMA` (dealers chase → momentum / whipsaw accelerant). The single most useful
    options-regime variable and the one nobody on the desk is reading yet.
  - **IV level:** `IV_LOW / MID / HIGH` (ATM IV percentile).
  - **Skew:** put-call IV skew steepness & sign (steep put skew = downside fear / crash-prone;
    flat/call skew = melt-up/complacency).
  - **Term:** front vs next IV (backwardation = acute stress) — **limited: we capture one
    expiry intraday**, so term is EOD-only for now.
- **Estimators / proxies (from `chain_snapshots`: per-strike `delta`, `iv`, `oi`, `oich`):**
  - Gamma proxy = Σ over strikes of `gamma(strike) × OI × sign(dealer)`; sign from the
    standard assumption (dealers short calls/puts that customers buy). Build **GEX-style net
    gamma vs spot**; the zero-gamma flip level is the intraday pin/repel pivot.
  - ATM IV = interpolate `iv` at the ATM strike; percentile vs trailing.
  - Skew = `iv(25Δ put) − iv(25Δ call)` using the `delta` column.
- **NSE calibration:** weekly NIFTY (Tue expiry post-2024) shows the strongest pin; BANKNIFTY/
  FIN/MIDCAP are **monthly-only** → pin only near monthly expiry. FINNIFTY chain is thin (~50%
  strikes no OI) → gamma proxy unreliable there (flag, low confidence).
- **Data:** ✅ greeks now captured (`chain_snapshots.delta/iv`, fix 73168f1). ⚠ single expiry,
  ⚠ FINNIFTY thin. Gamma proxy = **new module to build**.

### Axis D — Participation / Liquidity / Breadth
- **States:** `DEEP/ACTIVE`, `NORMAL`, `THIN/ILLIQUID`; plus `COHERENT` vs `DISPERSED` across
  the 4 indices.
- **Estimators:** volume vs time-of-day baseline; futures **basis** stability (have); bid-ask /
  depth where available; cross-index coherence (`regime_forecast.market_forecast` already does
  this); advance-decline breadth (DCM EOD; intraday sector basket).
- **NSE calibration:** hard session-phase baselines — open (huge), **lunch lull ~12:00–13:30**
  (thin, fade size), pre-close ramp 14:30+. Thin ≠ trend even if price drifts.
- **Data:** ✅ volume/basis/coherence; breadth partial.

### Axis E — Catalyst / Time State
- **States:** `OPENING` (09:15–~09:35 forming), `MORNING`, `LUNCH`, `AFTERNOON`, `PRE_CLOSE`;
  crossed with `EXPIRY_DAY`, `PRE_EVENT` (RBI/Fed/data/results), `POST_EVENT`, `NEWS_SHOCK`.
- **Estimators:** clock + expiry calendar (data-driven next-expiry, cf. DCM bug fix); event
  calendar + `news_events` layer (have); `intraday_shock` for realised shocks.
- **NSE calibration:** **NIFTY weekly expiry = Tuesday** (NSE moved it; do not hardcode
  Thursday). Expiry-day intraday is its **own regime family** (gamma pin → unpin). Budget/RBI/
  Fed nights repriced overnight (see `macro_overnight`: gap is gift-priced, intraday fades).
- **Data:** ✅ session phase (`signal_types`/`opening_playbook`), news layer, expiry calendar.

---

## 3. The composite regime lattice (named regimes → strategy)

Each named regime = a **signature** over axes A–E. `—` = don't-care. Strategy family is the
*licensed* action; size is conditioned on confidence.

| # | Regime | A dir | B vol | C gamma/IV | D/E context | Strategy family (license) | Honest edge note |
|---|---|---|---|---|---|---|---|
| 1 | **Trending Expansion** (drive) | strong trend | elevated, **expanding** | short-gamma, IV rising | active, coherent | **Momentum / breakout ride** | the *only* cell directional pays; still thin at cost — size small |
| 2 | **Orderly Grind** | small trend | normal/contracting | mixed | low-mod participation | trend with pullback entries; pyramid | choppy; many fakeouts |
| 3 | **Quiet Range / Gamma Pin** | balanced | **compressed** | **long-gamma**, IV low | often pre-event/expiry | **SELL premium** (iron-fly / ATM straddle short), mean-revert to mid | the theta regime — *new* product, not directional |
| 4 | **Volatile Whipsaw** | balanced (no net) | **elevated** | **short-gamma** | — | **STAND ASIDE** (or tiny fade, wide stop) | the user's "both-side SL hunting"; untradeable directionally |
| 5 | **Compression Coil** | range tightening | compressed, **vol-of-vol ↑** | gamma flattening | volume drying | **BUY cheap gamma** (long straddle/strangle) ahead of break | pays when break is real; theta risk if it doesn't |
| 6 | **Expansion Breakout** | range→trend | **vol spike** | gamma flips short | basis/flow confirm | breakout entry **on confirmation** | beware the false break (most are) |
| 7 | **Event Vacuum / Pre-Event Freeze** | balanced | flat realised, **IV rising** | long-gamma, IV high | `PRE_EVENT` | **STAND ASIDE**; sell only if IV ≫ realised | direction is a coin flip; vol is rich for a reason |
| 8 | **Post-Event Vol Crush** | unclear→ | shock then **collapse** | **IV crush** | `POST_EVENT` first 5–15m | **SELL vol** after the spike; avoid first-minutes direction | the repricing is instant & gift-priced |
| 9 | **Expiry Pin** (NSE Tue NIFTY) | balanced, pinned | compressed intraday | **strong long-gamma**, pin to max-pain/round strike | `EXPIRY_DAY` | **fade to pin / short ATM straddle into close** | very NSE-specific; high hit-rate, tail risk on unpin |
| 10 | **Expiry Unpin / Squeeze** | sudden trend | spike | gamma flips short late | `EXPIRY_DAY` PM | momentum chase the break (rare) | dangerous; small, fast |
| 11 | **Liquidation Cascade** (trend-day down) | **big trend down** | shock-expanding | short-gamma + **steep put skew** | broad, coherent | **directional puts** (the asymmetric cell) | matches the BIG_TREND_DOWN edge in backtest_regime — the one directional cell with real history |
| 12 | **Opening (forming)** | undetermined | high | unstable | 09:15–~09:35 | **STAND ASIDE** until OR resolves | scout already cools to 09:35–45 |
| 13 | **Lunch Lull** | drift/range | low | long-gamma-ish | 12:00–13:30 thin | theta only, **cut size** | thin drift ≠ trend |
| 14 | **Risk-Off / Global Stress** | gap + trend | elevated | put-bid, IV high | overnight macro | context + **gap-fade asymmetry** | gap is gift-priced (`macro_overnight`) |

**The asymmetry that recurs in our own data:** downside regimes (11, steep put skew, short
gamma) are where directional edge actually shows (PE/short = regime beta in the technical
backtest). Up-trends mean-revert faster (call skew/complacency). The taxonomy must treat
**UP and DOWN as different regimes, not mirror images** — they are not symmetric on NSE.

---

## 4. Confidence & transition framework

```
regime_confidence = base_axis_confidence
                  × axis_agreement_factor      # axes telling a consistent story
                  − conflict_penalty           # e.g. trend(A) but thin(D) → demote
                  − transition_penalty         # sitting between two cells
                  × persistence_factor         # bars the regime has held (hysteresis)
```

- **Agreement vs conflict.** A clean regime = all axes corroborate (e.g. Trending Expansion:
  A strong + B expanding + C short-gamma + D coherent). Conflicts demote to *low confidence →
  stand aside*. Example conflict: A says trend-up but C shows long-gamma pin + D thin → it's a
  **fakeout grind (regime 2/13)**, not a drive — do not chase.
- **Transition detection.** Borrow `regime_forecast`'s short-vs-long divergence as a
  *regime-change pressure* gauge; high pressure ⇒ widen stops, cut size, *do not* initiate.
- **Hysteresis.** Require k consecutive confirming classifications (or a vol-scaled z-move) to
  flip the dominant regime. Pin a `regime_since` timestamp; freshly-flipped regimes trade
  smaller until they persist.
- **MTF nesting.** Dominant = 60m/day; tactical = 5/15m. Publish both; the tactical action must
  be *consistent with* the dominant (no 5m breakout longs inside a 60m Quiet Pin).

---

## 5. Strategy-selection matrix (regime → action, the point of the whole thing)

| Strategy family | Regimes that license it | Instrument | Risk posture |
|---|---|---|---|
| **Momentum / breakout** | 1, 6, 10, 11 | futures / directional ATM option (small) | trend stop, scale out; cost-aware |
| **Mean-reversion / fade** | 3, 9, (4 tiny) | fade to VWAP/mid/pin | tight invalidation at range edge |
| **Premium selling (theta)** | 3, 7, 8, 9, 13 | short straddle/strangle, iron-fly | defined risk, gamma/event aware |
| **Premium buying (gamma)** | 5, pre-6 | long straddle/strangle | theta-bleed budget, time-boxed |
| **STAND ASIDE** | 4, 7, 12, low-confidence anything | — | capital preservation = the default |

**Default action = STAND ASIDE.** This is consistent with the standing strategy pivot: the
system is a discipline tool. The regime engine's first job is to *say no* in regimes 4/7/12
(where the board currently bleeds the 41 stops), and to *route to vol strategies* in 3/5/8/9
(which the directional system never even attempted, and which don't fight the cost floor the
same way a naked directional bet does).

---

## 6. Implementation architecture (fits this repo)

```
core/regime/
  axis_direction.py   # ER/ADX/VR/Hurst  (wrap regime_classifier + price_structure)
  axis_volatility.py  # realised-vol percentile + vol-of-vol + gap   (new)
  axis_options.py     # GEX/dealer-gamma sign, ATM-IV pct, 25Δ skew  (new, from chain greeks)
  axis_liquidity.py   # volume-vs-baseline, basis, cross-index coherence  (wrap regime_forecast)
  axis_catalyst.py    # session phase + expiry + event/news state    (wrap signal_types/news_events)
  engine.py           # compose axes → named regime + confidence + transition + MTF nesting
  strategy_map.py     # named regime → licensed strategy family + size
```

- **Output contract:** `RegimeState{ dominant, tactical, axes:{A..E:(state,conf)}, confidence,
  regime_since, transition_pressure, licensed_strategies, size_mult, reasons[] }` — causal,
  `as_of`-safe, JSON-able for the dashboard and replay.
- **Validation:** extend `backtest_regime.py` to stratify by the **full vector** (not just the
  ER mood), and `backtest_scout.py`'s option-P&L grade per named regime — especially proving
  the **theta-sell cells (3/9) clear cost**, which is the genuinely new, untested frontier.
- **Phasing:**
  1. Axis B (volatility state) — cheap, high value, splits `CONSOLIDATION` immediately.
  2. Axis C (dealer gamma / skew) — the index-option edge; needs the GEX module on captured greeks.
  3. Compose engine + confidence + strategy map; wire as **display-first** (like the mood gate),
     enforce per cell only after its own option-CI clears.

---

## 7. Honest caveats (the part most regime decks omit)

- **Most cells are not tradable directionally.** The taxonomy's payoff is *risk routing*, not a
  signal multiplier. Expect 60–70% of session-time in regimes 3/4/12/13 where the answer is
  *sell theta or stand aside*.
- **Data gaps:** intraday **futures OI** missing (NSE oi-spurts blocks us) — Axis A/C lose the
  positioning leg intraday; **single captured expiry** ⇒ no intraday IV term structure; FINNIFTY
  chain too thin for a trustworthy gamma read. Build around these, don't pretend.
- **The new frontier is vol, not direction.** The most defensible *new* edge this taxonomy
  exposes is **premium-selling in long-gamma pin/event regimes (3/7/9)** — a non-directional
  product the system has never targeted, and the one most likely to survive the cost floor that
  killed every directional arrow. That, not a better CE/PE call, is where I'd point the firm's
  next research dollar.
```
