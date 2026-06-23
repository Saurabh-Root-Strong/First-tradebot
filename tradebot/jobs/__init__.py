"""L7 jobs — scheduled / batch entry points.

eod_sync, nightly_sync, morning_token, download_historical. May import any layer.
NOTE (Phase-E debt): eod_sync currently imports backtest_hour_forecast (research) —
the ledger-refresh logic must move into a job/engine module to break that coupling.
"""
