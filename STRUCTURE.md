# Tradebot — Structure Map (where every module belongs)

Canonical home for each current flat module under the target 7-layer package
(`tradebot/`). Use this when adding or moving code: find the layer, obey the law
(**imports point down only**: app → engine → signals → data → storage → adapters → core).

Status legend: ✅ already there · 🔜 to move (phase) · ⏳ deferred

| Current flat module | Target home | Layer | Phase |
|---|---|---|---|
| `core/` (constants, mirror_io, ta) | `tradebot/core/` | L0 | ✅ |
| `signal_types.py` | `tradebot/core/types.py` | L0 | 🔜 E |
| `fyers_auth*.py`, `dashboard` auth/fetch/WS, `LiveFeed` | `tradebot/adapters/broker/` | L1 | 🔜 B |
| `nse_oi.py` | `tradebot/adapters/nse.py` | L1 | 🔜 B |
| `news_events.py` (feed I/O part) | `tradebot/adapters/news.py` | L1 | 🔜 B |
| `intraday_db.py`, `intraday_store.py` | `tradebot/storage/intraday.py` | L2 | 🔜 C |
| context/trades SQLite access | `tradebot/storage/context.py`, `trades.py` | L2 | 🔜 C |
| (table DDL, scattered) | `tradebot/storage/schema.py` | L2 | 🔜 C |
| `market_data_bridge.py`, `daily_context_bridge.py` | `tradebot/data/flow.py` + `data/context.py` | L3 | 🔜 D |
| `market_snapshot.py` | `tradebot/data/snapshot.py` | L3 | 🔜 D |
| `signals.py` | `tradebot/signals/technical.py` | L4 | 🔜 E |
| `intraday_oi_intel.py`, `oi_analytics.py` | `tradebot/signals/oi.py` | L4 | 🔜 E |
| `intraday_shock.py` | `tradebot/signals/shock.py` | L4 | 🔜 E |
| `regime_forecast.py` | `tradebot/signals/regime.py` | L4 | 🔜 E |
| `opening_playbook.py` | `tradebot/signals/playbook.py` | L4 | 🔜 E |
| `session_conductor.py` | `tradebot/signals/conductor.py` | L4 | 🔜 E |
| `hour_forecast.py` | `tradebot/signals/forecast.py` | L4 | 🔜 E |
| `timeframe_delta.py`, `trend_matrix.py`, `smart_money.py` | `tradebot/signals/` | L4 | 🔜 E |
| `trade_setup.py` | `tradebot/engine/trade_setup.py` | L5 | 🔜 E |
| `engine.py` | `tradebot/engine/engine.py` | L5 | 🔜 E |
| `cost_model.py`, `cost_overlay.py` | `tradebot/engine/cost.py` | L5 | 🔜 E |
| `engine_replay.py`, `session_replay.py`, `conductor_replay.py` | `tradebot/engine/replay.py` | L5 | 🔜 E |
| `dashboard.py` (UI half) | `tradebot/app/server.py` + `components/` + `callbacks/` | L6 | 🔜 E |
| `eod_sync.py`, `nightly_sync.py`, `sync_from_vm.py` | `tradebot/jobs/` | L7 | 🔜 C/E |
| `morning_read.py`, `download_historical.py`, `supervise.py` | `tradebot/jobs/` | L7 | 🔜 E |
| `fno_universe.py` | `tradebot/core/universe.py` | L0 | 🔜 E |
| `backtest_*.py`, `*_validate.py`, `analyze_backtest.py` | `research/` | — | ⏳ (needs pkg installed) |
| `test_*.py` | `tests/` | — | ✅ A |
| `Dockerfile`, `Caddyfile`, `docker-compose.yml`, `*.bat`, `*.ps1`, `*.sh`, `*.md` runbooks | `ops/` | — | ⏳ (deploy-path-referenced) |
| `legacy/` | `legacy/` (quarantined, never imported) | — | ✅ |

## Known coupling debt to break during migration
- `eod_sync.py:239` imports `backtest_hour_forecast` → a prod job depends on research code.
  The hour-forecast ledger-refresh must become an engine/job function before backtests
  can move to `research/`. (Phase E.)
- `dashboard.py` builds `LiveFeed` AND owns the WS + 4 pollers → split capture (adapters
  + storage, headless process) from UI (app) — this is also the real OOM fix. (Phase B+E.)
