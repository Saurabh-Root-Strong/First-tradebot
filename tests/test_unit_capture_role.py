"""The capturer/viewer rule, and the guarantee that every layer agrees on it.

Three layers act on this decision: dashboard.py gates the WebSocket + pollers,
intraday_db gates every DB write (the clobber guard), supervise.py refuses to run.
When two of them disagreed about the SAME process the laptop spent a whole session
in a restart loop (2026-08-11) — so the contract under test is not just "the rule is
right" but "there is exactly one rule".
"""
from __future__ import annotations

import importlib
import os

import pytest

from core import capture_role
from core.constants import LIVE_DIR

ENV_KEYS = ("DASH_VIEWER", "TRADEBOT_CAPTURE")


@pytest.fixture
def clean_env(monkeypatch):
    for k in ENV_KEYS:
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


@pytest.fixture
def marker_dir(tmp_path):
    """A fake mirror dir; capture_role takes live_dir and looks at its PARENT."""
    live = tmp_path / "intraday" / "live"
    live.mkdir(parents=True)
    return live


# ── the rule itself ──────────────────────────────────────────────────────────────
def test_safe_by_default(clean_env, marker_dir):
    """No marker, no env → VIEWER. The dangerous role is never the fallthrough."""
    assert capture_role.is_capture_host(marker_dir) is False
    assert capture_role.resolve_role(marker_dir) == capture_role.VIEWER


def test_marker_makes_a_capturer(clean_env, marker_dir):
    capture_role.marker_path(marker_dir).touch()
    assert capture_role.is_capture_host(marker_dir) is True
    assert capture_role.resolve_role(marker_dir) == capture_role.CAPTURER


def test_env_flag_makes_a_capturer_without_a_marker(clean_env, marker_dir):
    clean_env.setenv("TRADEBOT_CAPTURE", "1")
    assert capture_role.is_capture_host(marker_dir) is True


def test_dash_viewer_overrides_both(clean_env, marker_dir):
    """dev.bat forces a viewer on a box that may carry the marker — it must win."""
    capture_role.marker_path(marker_dir).touch()
    clean_env.setenv("TRADEBOT_CAPTURE", "1")
    clean_env.setenv("DASH_VIEWER", "1")
    assert capture_role.is_capture_host(marker_dir) is False


@pytest.mark.parametrize("val", ["0", "", "true", "yes", "2"])
def test_only_exactly_one_enables(clean_env, marker_dir, val):
    """Anything other than the literal "1" is not opting in — no truthiness games on a
    flag whose wrong answer clobbers the mirrors."""
    clean_env.setenv("TRADEBOT_CAPTURE", val)
    assert capture_role.is_capture_host(marker_dir) is False


def test_why_names_the_deciding_input(clean_env, marker_dir):
    assert "no marker" in capture_role.why(marker_dir)
    clean_env.setenv("TRADEBOT_CAPTURE", "1")
    assert "TRADEBOT_CAPTURE" in capture_role.why(marker_dir)
    clean_env.setenv("DASH_VIEWER", "1")
    assert "DASH_VIEWER" in capture_role.why(marker_dir)


# ── the anti-drift contract ──────────────────────────────────────────────────────
def test_intraday_db_marker_path_matches(clean_env):
    """intraday_db resolves the marker from its own _DB_DIR. If that ever stops being
    LIVE_DIR.parent, the write guard and the launcher guard silently disagree — a viewer
    that writes IS the clobber bug (2026-07-09)."""
    import intraday_db
    assert intraday_db._DB_DIR.resolve() == LIVE_DIR.parent.resolve()
    assert (intraday_db._DB_DIR / capture_role.MARKER_NAME).resolve() == \
        capture_role.marker_path().resolve()


@pytest.mark.parametrize("env,expect", [
    ({}, False),
    ({"TRADEBOT_CAPTURE": "1"}, True),
    ({"DASH_VIEWER": "1"}, False),
    ({"TRADEBOT_CAPTURE": "1", "DASH_VIEWER": "1"}, False),
])
def test_intraday_db_agrees_with_capture_role(clean_env, env, expect):
    """The write guard must return exactly what the shared rule returns, for every
    combination — this is the assertion that would have caught the 2026-08-11 split."""
    for k, v in env.items():
        clean_env.setenv(k, v)
    import intraday_db
    assert intraday_db._is_capture_host() is capture_role.is_capture_host()
    if not capture_role.marker_path().exists():
        assert intraday_db._is_capture_host() is expect


def test_supervise_refuses_on_a_viewer_box(clean_env, monkeypatch):
    """supervise.main() must exit rather than babysit a process that cannot write the
    heartbeat it health-checks. Guard placed FIRST so no port is freed and no token is
    demanded on the way out."""
    monkeypatch.setenv("DASH_VIEWER", "1")
    supervise = importlib.import_module("supervise")
    called = {}
    monkeypatch.setattr(supervise, "launch",
                        lambda *a, **k: called.setdefault("launched", True))
    monkeypatch.setattr(supervise, "ensure_token",
                        lambda *a, **k: called.setdefault("token", True))
    with pytest.raises(SystemExit) as exc:
        supervise.main()
    assert exc.value.code == supervise.EXIT_NOT_CAPTURE_HOST
    assert not called, "a viewer refusal must not launch or touch the token"


def test_supervise_still_runs_on_the_capture_host(clean_env, monkeypatch):
    """THE VM MUST NOT REGRESS. The role gate is a refusal for viewers only — on the
    capture host supervise has to fall straight through to its normal start-up. Proven by
    letting the gate pass and stopping at the very next step (the single-instance check /
    token gate), i.e. the gate itself never fires."""
    monkeypatch.setenv("TRADEBOT_CAPTURE", "1")
    supervise = importlib.import_module("supervise")
    reached = {}
    monkeypatch.setattr(supervise, "_other_supervisors",
                        lambda: reached.setdefault("past_role_gate", True) and [999])
    with pytest.raises(SystemExit) as exc:
        supervise.main()
        raise SystemExit(0)          # main() returns on "already running"
    assert reached.get("past_role_gate"), "role gate wrongly refused the capture host"
    assert exc.value.code != supervise.EXIT_NOT_CAPTURE_HOST


def test_ensure_capture_noops_on_a_viewer_box(clean_env, monkeypatch):
    monkeypatch.setenv("DASH_VIEWER", "1")
    ec = importlib.import_module("ensure_capture")
    monkeypatch.setattr(ec, "_supervisor_running",
                        lambda: pytest.fail("must not even scan on a viewer"))
    ec.main()          # returns silently


def test_this_box_is_a_viewer(clean_env):
    """Standing topology decision: the laptop is ALWAYS viewer-only, one capturer = the
    VM. If this ever fails on a dev box, someone created .capture_host locally and the
    laptop is about to fight the VM for the single Fyers socket."""
    if os.environ.get("CI"):
        pytest.skip("role depends on the checkout, not meaningful in CI")
    assert capture_role.marker_path().exists() is False
