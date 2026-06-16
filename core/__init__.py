"""core — shared primitives for the Tradebot stack.

Single source of truth for things that were copy-pasted across many modules:
constants (IST, index symbols, labels, paths), the lock-free parquet-mirror reader,
and small TA helpers. Leaf package — imports nothing local — so it can be used
everywhere without cycles.
"""
