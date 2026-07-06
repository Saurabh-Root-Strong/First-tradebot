"""Import smoke test — the cheap regression net over the decision spine.

Importing these modules exercises their top-level (constants, table compilation,
dataclass defs). A syntax error or a broken cross-import from a refactor fails HERE
in <1s instead of at dashboard boot. Offline: no module may need a token/network at
import time (if one does, THAT is the finding).
"""
import importlib

import pytest

CORE_MODULES = [
    "cost_model", "news_events", "macro_radar", "hour_forecast",
    "calibration_engine", "regime_classifier", "intraday_scout",
    "intraday_db", "intraday_store", "feature_engine", "opening_playbook",
    "session_conductor",
]


@pytest.mark.parametrize("mod", CORE_MODULES)
def test_module_imports_offline(mod):
    importlib.import_module(mod)
