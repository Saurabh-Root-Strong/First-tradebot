"""L6 app — Dash UI ONLY. render + callbacks. ZERO business logic.

server.py builds the Dash app, wires pollers → engine, serves. components/ holds
the ~40 _render_* split by panel; callbacks/ holds the 22 callbacks. Nobody imports
app. This is where dashboard.py's UI half lands; its ingestion half goes to adapters.
"""
