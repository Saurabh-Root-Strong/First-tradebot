"""tradebot — the layered trading stack.

Layer law (imports point DOWN only, never sideways, never up):

    app → engine → signals → data → storage → adapters → core

See ARCHITECTURE_AUDIT.md for the full design + migration phases.
Enforced by import-linter (pyproject.toml [tool.importlinter]) once code lands here.
"""
