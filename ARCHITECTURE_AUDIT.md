# Tradebot — World-Class Architecture Audit

> Author: architecture pass, 2026-06-23. Scope: every Python module, every store, every layer.
> Goal: a structure where changing one thing touches one place, and the live app stays fast.

---

## 0. Verdict

The **quant logic is sound** (validated repeatedly in prior audits). The **software architecture is not** — it is a flat pile of 60+ root modules with one 249 KB god-file (`dashboard.py`) that *is* the application: auth + broker + websocket + capture + DB writes + every signal poller + every render + every callback in a single file.

The good news: the *direction* is already proven. `core/` killed the IST/symbol/path duplication that was smeared across 19 files. `engine.py` lifted the trading loop out into a clean feed-driven core. **The pattern works — it was just never carried through.** This audit finishes the job: a 7-layer package, one broker boundary, one storage contract, and guardrails so it can't rot back.

---

## 1. Current state (measured, not remembered)

| Metric | Value |
|---|---|
| Python modules (non-venv) | 67 |
| Total LOC | ~24,100 |
| `dashboard.py` | 249 KB — 90 top-level defs, 22 callbacks |
| `daily_context_bridge.py` | 60 KB (1330 lines) |
| `trade_setup.py` | 55 KB |
| `nightly_sync.py` | 37 KB |
| Modules that build a Fyers client directly | 5+ (`dashboard`, `signals`, `supervise`, `download_historical`, + auth) |
| Packaging (`pyproject`/`setup.py`) | **none** |
| Test config (`pytest.ini`/`conftest`) | **none** (despite ~12 `test_*`/`backtest_*` files) |
| Storage paradigms | **3** — per-day DuckDB, parquet mirrors, SQLite |

### What's already right
- **`core/`** (constants, `mirror_io`, `ta`) — leaf package, no local imports, single source of truth. IST duplication is gone (only a stray test still hardcodes it). *This is the model to replicate.*
- **`engine.py`** — headless, feed-driven, duck-typed `Feed`. Same code runs live or replay. Textbook.
- **`signal_types.py`**, **`market_snapshot.py`**, **`cost_model.py`** — the partial L1/L2 layering scaffold.

### What's wrong (ranked by blast radius)

**P0 — `dashboard.py` is the whole app.** Its `def` list shows auth (`_get_auth`, `_validate_token`, `_run_auth`), broker fetch (`fetch_option_chain`, `fetch_futures`, `_fetch_quotes`), websocket lifecycle + four pollers (`_oi_background_poller`, `_trade_tracker_poller`, `_auto_signal_poller`, `_heartbeat_writer`), DB writes (`_idb_write_futures`), and ~40 `_render_*`/callback functions. `engine.py` extracted the *trading loop* but `LiveFeed` and the ingestion threads still live here. Result: you cannot run capture without booting the UI; one edit risks the whole live session; it's the OOM/`code -9` crash surface.

**P1 — No broker boundary.** Five modules construct Fyers clients and read `access_token.txt` independently. Token logic, rate-limit handling, and the REST option-chain quota that *kills chain capture ~11am daily* are scattered, not owned by one adapter.

**P2 — Two overlapping "bridges."** `market_data_bridge` (442 lines, institutional/cash from SQLite) and `daily_context_bridge` (1330 lines, EOD structural context) both read context and re-derive flow primitives. Prior audits already flagged "no canonical flow primitive, re-derived across engines." Two doors into the same data.

**P3 — Storage is three paradigms with no contract.** Per-day DuckDB (durable, **single-writer file lock** — confirmed: live PID holds it), parquet mirrors (lock-free live reads via `core.mirror_io`), SQLite (`tradebot_context.db`, `intraday_trades.db`). The dual-write (`intraday_store` → mirrors, `intraday_db` → duckdb) is reasonable, but there's no single module that *owns* "where does X live and who may write it."

**P4 — Flat namespace.** 60 modules in root means no import tells you which layer it's in. `backtest_*`, `test_*`, runbooks (`.md`), batch files, and logs all sit beside production code. No `pyproject.toml`, so it's a script pile, not an installable package — imports are position-dependent and there's no dependency pinning boundary between dev and VM.

**P5 — Tests aren't a suite.** `test_*.py` are ad-hoc scripts; `backtest_*.py` are one-shot research harnesses. No `pytest` runner, no CI gate. The parity proofs (replay == live) that the architecture *depends on* aren't enforced automatically.

---

## 2. Target architecture

A single installable package, `tradebot/`, with **7 layers**. The rule that makes it world-class: **imports only ever point downward.** A linter enforces it (§4).

```
tradebot/
├── pyproject.toml              # installable; pins; entry points; ruff+import-linter config
│
├── core/                       # L0 — leaf. constants, mirror_io, ta, clock, types.
│                               #   imports nothing local. (EXISTS — keep, extend.)
│
├── adapters/                   # L1 — the ONLY code that talks to the outside world.
│   ├── broker/                 #   Fyers behind one interface: auth, token, rate-limit,
│   │   ├── fyers.py            #   REST quota, websocket. Everything else depends on the
│   │   └── feed.py             #   interface, never on fyers_apiv3. (LiveFeed lands here.)
│   ├── nse.py                  #   nse_oi
│   └── news.py                 #   RBI/NSE feeds
│
├── storage/                    # L2 — the ONLY code that reads/writes a store.
│   ├── intraday.py             #   DuckDB per-day (owns the writer lock) + parquet mirrors
│   ├── context.py              #   SQLite context (tradebot_context.db)
│   ├── trades.py               #   SQLite paper trades
│   └── schema.py               #   every table DDL in one place + migrations
│
├── data/                       # L3 — domain reads. Turns stores into clean frames.
│   ├── snapshot.py             #   market_snapshot (primitives once)
│   ├── flow.py                 #   THE canonical flow primitive (kills the 2 bridges)
│   ├── chain.py                #   option-chain shaping
│   └── context.py              #   daily/EOD context (merges the two bridge readers)
│
├── signals/                    # L4 — pure analytics. frames in → scores out. No I/O.
│   ├── technical.py            #   signals.py core
│   ├── oi_intel.py  oi.py  shock.py  regime.py  playbook.py  conductor.py
│   └── forecast.py             #   hour_forecast
│
├── engine/                     # L5 — decisions + lifecycle. feed-driven. (EXISTS.)
│   ├── trade_setup.py          #   build_recommendation
│   ├── engine.py  cost_model.py  ledger.py
│   └── replay.py               #   engine_replay/session_replay
│
├── app/                        # L6 — Dash UI ONLY. render + callbacks. zero logic.
│   ├── server.py               #   app = Dash(...); wires pollers→engine; serves
│   ├── components/             #   the ~40 _render_* split by panel
│   └── callbacks/
│
├── jobs/                       # L7 — scheduled/batch. eod_sync, nightly_sync, morning_*
│
├── research/                   # backtests & one-shot studies. NOT imported by prod.
│   └── backtest_*.py  *_validate.py
│
├── tests/                      # pytest suite. parity proofs become CI gates.
└── ops/                        # Dockerfile, Caddyfile, runbooks, .bat/.ps1, compose
```

**Layer law:** `app → engine → signals → data → storage → adapters → core`. Never sideways, never up. `core` imports nothing local; `app` imports anything; nobody imports `app`.

This directly dissolves P0–P5: the god-file splits along its own seams (its defs already cluster into adapter/storage/render groups); the broker gets one home; the two bridges merge into `data/flow.py` + `data/context.py`; storage ownership becomes explicit; the flat namespace becomes a typed dependency graph; `research/` and `tests/` leave the production path.

---

## 3. Speed / performance findings

Architecture *is* speed here — the same moves that clean structure also cut latency:

1. **Per-panel recompute (biggest win).** Prior audit noted callbacks recompute primitives per panel. `market_snapshot` was built to compute-once; **finish wiring every render to read the snapshot Store**, not re-derive. In a 22-callback Dash app this is the dominant cost.
2. **DuckDB single-writer lock.** One writer thread owns the connection (move into `storage/intraday.py`); all readers use parquet mirrors via `core.mirror_io`. Today writes are scattered across 9 modules — contention + the lock error we hit live.
3. **REST option-chain quota death ~11am.** Centralize chain fetching in `adapters/broker` with a budget + backoff; read DuckDB (not truncated parquet) when REST is exhausted. This is a known daily failure (chain capture dies mid-session).
4. **OOM / `code -9` on t3.micro.** The monolith loads UI + capture + all engines in one process. Splitting capture (`adapters`+`storage`) into a headless process from the Dash UI lets each be sized/restarted independently — the real fix the swappiness band-aid is masking.
5. **`pandas==3.0.3` / `numpy` dance.** Pin once in `pyproject` with environment markers (dev py3.14 vs VM py3.12/numba) instead of the comment-in-`requirements.txt` workaround.

---

## 4. Guardrails (so it can't rot back)

1. **`import-linter` contract** in CI: declares the 7 layers; build fails on an upward/sideways import. This is what makes the structure *durable* rather than aspirational.
2. **`ruff`** for lint/format, one config, pre-commit hook.
3. **`pytest` suite** with the replay==live parity proof as a gate (the architecture already promises this — enforce it).
4. **One module per concern, hard size budget** (~600 LOC soft cap). A file approaching it is a missing boundary.
5. **`adapters/` is the only place** `import fyers_apiv3`, `requests`, or a DB driver may appear. Grep-able invariant.

---

## 5. Migration plan — phased, safe for a LIVE trading system

Do **not** big-bang move 60 files. Each phase is independently shippable and leaves the app runnable. Order chosen so the highest-risk live path (`dashboard.py`) moves last, on top of stable foundations.

- **Phase A — Package skeleton.** Add `pyproject.toml`, create the empty layer dirs, move `research/`, `tests/`, `ops/` out of the production path. No prod import changes. Zero behavioral risk.
- **Phase B — Broker boundary (P1).** Create `adapters/broker`. Move auth/token/fetch/WS there behind one interface. Repoint the 5 client-builders. Kills the daily-quota and token-scatter problems.
- **Phase C — Storage ownership (P3).** Create `storage/`. One DuckDB writer, schema in `schema.py`, readers on mirrors. Repoint the 9 writers.
- **Phase D — Merge the bridges (P2).** `data/flow.py` canonical flow primitive + `data/context.py`. Retire `market_data_bridge`/`daily_context_bridge` once parity-tested.
- **Phase E — Split `dashboard.py` (P0).** `app/server.py` + `components/` + `callbacks/`; pollers call `engine`. The 90 defs already group cleanly — mechanical, but do it last on a stable base, with the running session as the parity oracle.
- **Phase F — Guardrails on (§4).** import-linter + ruff + pytest in CI. Lock it in.

Each phase: branch, move, repoint imports, run the app + parity backtest, commit. Memory (`project_tradebook_layering.md`) gets updated per phase so the next session knows where we are.

---

## 6. One-line summary

> The logic is a quant's; the layout is a script pile. `core/` and `engine.py` already prove the fix works — extend that pattern into a 7-layer installable package with one broker boundary, one storage owner, and an import-linter gate, migrating `dashboard.py` last. Same code, cleaner seams, faster app, no rot.
