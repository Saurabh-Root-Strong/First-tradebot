"""Unit — architectural boundaries on the FLAT top-level modules.

import-linter (pyproject) only governs the tradebot/ package, which is ~575 of ~36k LOC
— so its "contract kept" says nothing about the real code. These tests enforce, on the
FLAT modules where the logic actually lives, the three boundaries that genuinely matter:

  1. legacy/ is QUARANTINED dead code -> no live module may import it (resurrection guard;
     legacy.indicators was retired in the 2026 audit and must stay retired).
  2. core/ is a LEAF -> it must not import any top-level business module (constants /
     mirror_io / market_calendar are the base of the graph; a core->business import is a
     dependency inversion / cycle risk).
  3. the LIVE-SERVE path (the trading + dashboard process) must not import backtest_* /
     audit_* / edge_board -> heavy analysis code must never load into the live process
     (wrong dependency direction; keeps the trading loop lean). Nightly JOBS (eod_sync)
     may use backtests for calibration and are intentionally excluded.

Pure-stdlib ast — parses source, never imports the modules (fast, side-effect-free, works
even for modules whose import needs a broker token / network).
"""
import ast
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _imported_top_names(path: pathlib.Path) -> set[str]:
    """Top-level names of every ABSOLUTE import in the file (module-level AND function-local,
    via ast.walk). Relative imports (from .x) are intra-package and skipped."""
    names: set[str] = set()
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                names.add(a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


def _is_analysis(mod: str) -> bool:
    return mod.startswith("backtest_") or mod.startswith("audit_") or mod == "edge_board"


def test_legacy_is_dead_code():
    """No live/core/tradebot module may import the quarantined legacy/ package."""
    scan = (list(ROOT.glob("*.py")) + list(ROOT.glob("core/*.py"))
            + list(ROOT.glob("tradebot/**/*.py")))
    offenders = [str(p.relative_to(ROOT)) for p in scan
                 if "legacy" not in p.parts and "legacy" in _imported_top_names(p)]
    assert not offenders, f"live code imports quarantined legacy/: {offenders}"


def test_core_is_a_leaf():
    """core/ imports stdlib / third-party / intra-core only — never a business module."""
    business = {p.stem for p in ROOT.glob("*.py")} - {"conftest"}
    offenders = {p.name: sorted(_imported_top_names(p) & business)
                 for p in ROOT.glob("core/*.py")
                 if _imported_top_names(p) & business}
    assert not offenders, f"core/ is not a leaf — imports business modules: {offenders}"


def test_live_serve_free_of_analysis_code():
    """The live trading + dashboard process must not pull backtest_/audit_/edge_board in."""
    live = ["dashboard", "trade_setup", "intraday_scout", "signals", "intraday_store",
            "intraday_db", "footprint_chart", "session_conductor", "supervise"]
    offenders = {}
    for name in live:
        p = ROOT / f"{name}.py"
        if not p.exists():
            continue
        bad = sorted(m for m in _imported_top_names(p) if _is_analysis(m))
        if bad:
            offenders[name] = bad
    assert not offenders, f"live-serve path imports analysis code: {offenders}"
