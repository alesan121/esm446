"""Verification of T3: the periodic status heartbeat.

Before this, STATUS.txt was written only in `main()`'s `finally: clean_shutdown()` -- a
faithful record of a clean end and no record at all of any other one. An OOM kill, a power
cut, or `kill -9` all leave it saying "RUNNING" forever. These tests exist to check the thing
that actually matters: that a stale `last_heartbeat` timestamp is enough, on its own, to date
when an unattended session actually stopped -- which is the whole point of writing it
periodically instead of once at the end.
"""

from __future__ import annotations

import calendar
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import ModuleType

import pytest


def _load_module(status_dir: Path) -> ModuleType:
    """Load overnight_survey.py fresh, with OUTPUT_ROOT/STATUS_PATH redirected to a temp dir.

    Module-scope code creates OUTPUT_ROOT and derives STATUS_PATH from it at import time, so
    this patches both after loading rather than trying to intercept the real ~/esm446_overnight.
    """
    path = Path(__file__).resolve().parent.parent / "scripts" / "overnight_survey.py"
    spec = importlib.util.spec_from_file_location("overnight_survey", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["overnight_survey"] = module
    spec.loader.exec_module(module)
    module.OUTPUT_ROOT = status_dir
    module.STATUS_PATH = status_dir / "STATUS.json"
    return module


@pytest.fixture
def mod(tmp_path: Path) -> ModuleType:
    return _load_module(tmp_path)


def test_write_status_is_atomic_no_tmp_file_left_behind(mod, tmp_path: Path) -> None:
    mod.write_status(state="running", phase="c2_gain_sweep")

    files = sorted(p.name for p in tmp_path.glob("STATUS*"))
    assert files == ["STATUS.json"], "a .tmp file was left behind"


def test_write_status_carries_the_required_fields(mod) -> None:
    mod.write_status(
        state="running",
        phase="c3_long_run",
        phase_progress="chunk 5 starting",
        started_at="2026-08-16T10:00:00Z",
        chunks_completed=5,
        overruns_total=2,
    )

    record = json.loads(mod.STATUS_PATH.read_text())
    assert record["state"] == "running"
    assert record["phase"] == "c3_long_run"
    assert record["phase_progress"] == "chunk 5 starting"
    assert record["started_at"] == "2026-08-16T10:00:00Z"
    assert record["chunks_completed"] == 5
    assert record["overruns_total"] == 2
    assert "pid" in record
    assert "disk_free_gb" in record
    assert "last_heartbeat" in record


def test_successive_writes_advance_the_heartbeat_timestamp(mod) -> None:
    mod.write_status(state="running", phase="c2_gain_sweep", chunks_completed=1)
    first = json.loads(mod.STATUS_PATH.read_text())["last_heartbeat"]

    time.sleep(1.1)
    mod.write_status(state="running", phase="c2_gain_sweep", chunks_completed=2)
    second = json.loads(mod.STATUS_PATH.read_text())["last_heartbeat"]

    assert second != first, "the heartbeat did not move between two writes a second apart"


def test_an_abrupt_death_is_datable_from_the_last_heartbeat(mod) -> None:
    """The scenario this task exists for: the process is killed mid-session, no clean exit.

    Simulated by writing one heartbeat and then simply not writing another -- exactly what an
    OOM kill or a power cut looks like from STATUS.json's point of view. A reader who comes
    back later must be able to tell *when* the process stopped, not just that it isn't
    running now.
    """
    mod.write_status(state="running", phase="c3_long_run", chunks_completed=42, overruns_total=7)
    death_time = time.time()

    # Time passes with nobody updating the file -- the process is gone.
    time.sleep(1.2)

    record = json.loads(mod.STATUS_PATH.read_text())
    # timegm, not mktime: the timestamp string was built from gmtime() (UTC), and mktime
    # would silently reinterpret it in the local zone -- exactly wrong by the UTC offset.
    last_heartbeat = calendar.timegm(time.strptime(record["last_heartbeat"], "%Y-%m-%dT%H:%M:%SZ"))
    staleness_s = time.time() - last_heartbeat

    assert record["state"] == "running", "still claims to be running -- this is the failure mode"
    assert staleness_s >= 1.0, "the last heartbeat should date the moment writes stopped"
    assert (
        abs(last_heartbeat - death_time) < 2.0
    ), "heartbeat timestamp does not match when it stopped"
    # What makes this recoverable: the last real progress is still on disk, not lost with the
    # process.
    assert record["chunks_completed"] == 42
    assert record["overruns_total"] == 7


def test_terminal_states_are_distinct_from_running(mod) -> None:
    """completed / aborted_phase_X / disk_guard_tripped must not be confusable with RUNNING."""
    for state in ("completed", "aborted_phase_c3_long_run", "disk_guard_tripped"):
        mod.write_status(state=state, phase="done")
        record = json.loads(mod.STATUS_PATH.read_text())
        assert record["state"] == state
        assert record["state"] != "running"
