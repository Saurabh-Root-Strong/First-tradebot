"""L2 storage — the ONLY layer that reads/writes a store.

Owns the three stores and their contract: per-day DuckDB (single writer thread
holds the lock), parquet mirrors (lock-free live reads), SQLite context/trades.
schema.py holds every table DDL + migrations in one place. The 9 scattered
DuckDB writers collapse into intraday.py here.
"""
