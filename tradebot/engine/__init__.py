"""L5 engine — decisions + trade lifecycle. Feed-driven.

build_recommendation, the open/track/resolve ledger, cost_model, replay. Takes a
duck-typed Feed (LiveFeed = prod, ReplayFeed = backtest) so the same code runs
live or replayed, lookahead-free. engine.py already embodies this — extend it.
"""
