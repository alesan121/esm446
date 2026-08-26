"""T1: a single measured-results file, written by the tests that make each measurement.

Why this exists
----------------
Six figures in this project's history have had to be retracted because a measured number was
hand-transcribed into a docstring, a doc, or a README, and nothing kept that copy in sync when
a better measurement superseded it. The fix is not writing more carefully -- it is removing
the transcription step: a test that computes a figure calls `record_measurement` once, and
every document that would otherwise quote the number instead references the key.

`record_measurement` deliberately requires nothing to be transcribed by hand that could drift:

- ``source_test`` is read from ``PYTEST_CURRENT_TEST`` (which pytest sets for the duration of
  every test), not passed as a string -- a hand-typed test name is exactly the kind of copy
  that goes stale the moment the test is renamed. One consequence, accepted deliberately:
  this can only be called from inside a running pytest test, so `esm446.vv`'s CLI-triggered
  figure builders (``poetry run esm446-vv``, which run outside pytest) cannot call it. That
  unification is real follow-up work, not part of this module.
- ``commit`` is two fields, not one: ``first_seen_commit`` (written only when the recorded
  content changes) and ``last_verified_commit`` (written on every run that reaches this key).
  A single field cannot distinguish a value that was reverified an hour ago from one that has
  not run in forty commits and may be broken -- exactly the shape of drift this module exists
  to stop.

Where the file lives
---------------------
``results/measured.json`` is **not committed** -- it is in ``.gitignore``. Unlike
``docs/figures/results.json`` (opt-in, via ``poetry run esm446-vv``), this file is touched by
every ordinary ``pytest`` run, because the ``record_measurement`` calls live inside tests that
run on every ``make test`` and every CI push. Versioning something that changes on every test
run is what produces noisy diffs -- not writing it at all is what removes them by construction,
rather than by a write policy trying to guess when a diff would be "noisy". CI publishes it as
a build artefact after the suite runs, so a future consistency check has something to read
without depending on a local run.
"""

from __future__ import annotations

import json
import os
import subprocess  # nosec -- fixed argv only, no shell, used only for git rev-parse
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

MEASURED_PATH = Path("results/measured.json")

#: Substrings that mark `units` as an *absolute* RF level (depends on receiver gain), which
#: therefore needs its gain point recorded -- the same lesson as v0's OFFSET_CAL, which nobody
#: could reproduce because it was never written down alongside the gains it was measured at.
#:
#: Deliberately narrower than "any unit containing dB": a bare "dB" or "dBc" is usually a
#: *ratio* (rejection, gain, SNR cost, carrier-relative spur level) computed between two
#: quantities in the same capture, and a ratio cancels the gain it was measured at -- ask for
#: lna_db/vga_db on one of those and the requirement would be not just unnecessary but
#: actively misleading, implying a dependency the figure structurally does not have. Found
#: while migrating the first real figure in Fase 2 (92.9 dB adjacent-channel rejection, a
#: ratio) against the first version of this list, which included bare "db" and would have
#: forced a meaningless gain point onto it.
_RF_UNIT_HINTS = ("dbfs", "dbm")

#: Fields compared to decide whether a measurement's *content* changed, distinct from the
#: provenance fields (first_seen_*/last_verified_*) that are never part of the comparison.
_COMPARABLE_FIELDS = ("value", "units", "conditions", "status", "reason", "source_test")


class MeasurementError(ValueError):
    """Raised when `record_measurement`'s own requirements are not met.

    Not a bug in the caller's arithmetic -- a bug in how the measurement was *reported*: a
    missing description, a missing gain point, or an attempt to erase a real result. Each of
    these is exactly the shape of mistake that produced this project's past retractions, moved
    from "wrote something misleading" to "the tool refused to write it".
    """


def _current_commit() -> str:
    result = subprocess.run(  # nosec -- fixed argv, no shell, no untrusted input
        ["git", "rev-parse", "--short=12", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _current_source_test() -> str:
    """The running test's node ID, from pytest's own environment variable.

    Format is ``"<nodeid> (<phase>)"``, e.g. ``"tests/test_x.py::test_y (call)"``. Only the
    node ID is kept -- the phase suffix is stripped by removing the last space-delimited
    token, which is safe even if the node ID itself contains a space (an unusual but legal
    parametrize ID), because pytest always appends the phase as the final token.
    """
    current = os.environ.get("PYTEST_CURRENT_TEST")
    if not current:
        raise RuntimeError(
            "record_measurement() was called outside a running pytest test "
            "(PYTEST_CURRENT_TEST is not set). This is deliberate, not a missing feature -- "
            "see esm446/measured.py's module docstring."
        )
    return current.rsplit(" ", 1)[0]


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _looks_like_rf_level(units: str) -> bool:
    normalised = units.lower()
    return any(hint in normalised for hint in _RF_UNIT_HINTS)


def record_measurement(
    key: str,
    value: float | int | None,
    *,
    units: str,
    conditions: Mapping[str, Any],
    status: Literal["measured", "pending"] = "measured",
    reason: str | None = None,
    force: bool = False,
    path: Path = MEASURED_PATH,
) -> None:
    """Record one measured (or pending) figure under `key` in `path`.

    Args:
        key: Dotted identifier, e.g. ``"false_alarm_rate.ambient_high_gain_tracked"``.
        value: The measured figure, or `None` when `status="pending"`.
        units: Physical units. Values that look like an RF level (dB/dBFS/dBm/dBc) require
            `conditions` to include `lna_db` and `vga_db`.
        conditions: Must include `"description"` -- a human sentence -- plus whatever else
            the caller has that would let someone reconstruct the operating point without
            reading the source test.
        status: `"measured"` (default) or `"pending"` -- a test that cannot run today (no
            hardware, no capture file) should still register its key as pending rather than
            leave it silently absent.
        reason: Why the measurement is pending. Ignored for `status="measured"`.
        force: Required to let a `"pending"` call overwrite an existing `"measured"` entry --
            without it, that call raises. See `MeasurementError`.
        path: Where to read/write. Overridden in tests; production code should rely on the
            default.

    Raises:
        MeasurementError: `conditions` is missing `"description"`, an RF-level measurement is
            missing its gain point, `status`/`value` are inconsistent, or a `"pending"` call
            would silently erase an existing `"measured"` entry without `force=True`.
        RuntimeError: called outside a running pytest test.
    """
    if "description" not in conditions:
        raise MeasurementError(
            f"record_measurement({key!r}): conditions must include 'description' -- a figure "
            "without its conditions is exactly the problem this module exists to eliminate."
        )
    if _looks_like_rf_level(units) and not {"lna_db", "vga_db"} <= conditions.keys():
        raise MeasurementError(
            f"record_measurement({key!r}): units {units!r} look like an RF level, so "
            "conditions must include 'lna_db' and 'vga_db'. A power without its gains is not "
            "calibratable after the fact -- this is why v0's OFFSET_CAL was irreproducible."
        )
    if status == "pending" and value is not None:
        raise MeasurementError(
            f"record_measurement({key!r}): status='pending' means no value exists yet -- "
            "pass value=None."
        )
    if status == "measured" and value is None:
        raise MeasurementError(
            f"record_measurement({key!r}): status='measured' requires a real value."
        )

    if value is not None:
        # DSP figures are routinely numpy scalars (float32 from a complex64 pipeline); json
        # cannot serialise those. Coercing here, once, is cheaper than every call site
        # remembering it -- found while migrating the first Fase 2 figure.
        value = float(value)

    source_test = _current_source_test()
    entries = _load(path)
    prior = entries.get(key)

    if (
        prior is not None
        and prior.get("status") == "measured"
        and status == "pending"
        and not force
    ):
        raise MeasurementError(
            f"record_measurement({key!r}): already measured; refusing to downgrade to "
            "pending without force=True. A good measurement silently replaced by nothing is "
            "exactly the failure this module exists to prevent."
        )

    new_content = {
        "value": value,
        "units": units,
        "conditions": dict(conditions),
        "status": status,
        "reason": reason,
        "source_test": source_test,
    }
    unchanged = prior is not None and all(
        prior.get(field) == new_content[field] for field in _COMPARABLE_FIELDS
    )

    commit = _current_commit()
    now = _now_iso()

    if unchanged:
        # Confirmed again, but nothing about the recorded content is new: first_seen_* stays
        # exactly as it was, only the "still true as of" fields move.
        entry = dict(prior)
    else:
        entry = {**new_content, "first_seen_commit": commit, "last_changed": now}
    entry["last_verified_commit"] = commit
    entry["last_verified_at"] = now

    entries[key] = entry
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, indent=2) + "\n")
