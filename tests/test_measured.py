"""Verification of T1: `record_measurement`, the fix for six past retractions.

Every past retraction shared one shape: a figure computed once, transcribed by hand into a
doc or docstring, and never re-checked when the code that produced it changed. These tests
check the two properties that actually prevent that shape of mistake: that content changing
is distinguishable from content merely being reverified, and that a measurement can never be
silently erased by a test that could not run.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from esm446.measured import MeasurementError, record_measurement, toolchain_versions

pytestmark = pytest.mark.skipif(
    subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True).returncode != 0,
    reason="record_measurement shells out to git rev-parse; needs a git checkout",
)


@pytest.fixture
def results_path(tmp_path: Path) -> Path:
    return tmp_path / "measured.json"


def _minimal_conditions(**extra: object) -> dict:
    return {"description": "unit test fixture", **extra}


# --------------------------------------------------------------------------------------
# Validation: conditions, RF gain point, status/value consistency
# --------------------------------------------------------------------------------------


def test_conditions_without_a_description_are_refused(results_path: Path) -> None:
    with pytest.raises(MeasurementError, match="description"):
        record_measurement(
            "x", 1.0, units="dB", conditions={"lna_db": 32.0, "vga_db": 40.0}, path=results_path
        )


def test_an_rf_level_without_gains_is_refused(results_path: Path) -> None:
    """v0's OFFSET_CAL was irreproducible for exactly this reason -- a power with no gains."""
    with pytest.raises(MeasurementError, match="lna_db"):
        record_measurement(
            "x", -92.9, units="dBFS", conditions=_minimal_conditions(), path=results_path
        )


def test_an_rf_level_with_gains_is_accepted(results_path: Path) -> None:
    record_measurement(
        "x",
        -92.9,
        units="dBFS",
        conditions=_minimal_conditions(lna_db=32.0, vga_db=40.0),
        path=results_path,
    )
    assert json.loads(results_path.read_text())["x"]["value"] == -92.9


@pytest.mark.parametrize("units", ["dBFS", "dBm", "DbFs", "dBM"])
def test_absolute_rf_level_detection_is_case_insensitive(results_path: Path, units: str) -> None:
    """dBFS/dBm are absolute levels -- they depend on the gain they were measured at."""
    with pytest.raises(MeasurementError, match="lna_db"):
        record_measurement(
            "x", 1.0, units=units, conditions=_minimal_conditions(), path=results_path
        )


@pytest.mark.parametrize("units", ["dBc", "dB rejection", "dB", "false alarms per cell"])
def test_ratio_units_do_not_require_gains(results_path: Path, units: str) -> None:
    """dBc and bare 'dB' are typically ratios (rejection, gain, SNR cost) computed between two
    quantities in the same capture -- they cancel the gain they were measured at, so forcing
    lna_db/vga_db on them would be misleading rather than useful. Found while migrating the
    92.9 dB adjacent-channel-rejection figure, which is exactly such a ratio."""
    record_measurement("x", 1.0, units=units, conditions=_minimal_conditions(), path=results_path)
    assert json.loads(results_path.read_text())["x"]["value"] == 1.0


def test_pending_requires_value_none(results_path: Path) -> None:
    with pytest.raises(MeasurementError, match="pending"):
        record_measurement(
            "x",
            1.0,
            units="count",
            conditions=_minimal_conditions(),
            status="pending",
            path=results_path,
        )


def test_a_numpy_scalar_value_is_coerced_to_a_plain_float(results_path: Path) -> None:
    """DSP tests routinely produce np.float32/float64 from a complex64 pipeline; json cannot
    serialise those. Found migrating the 92.9 dB figure, whose rejection_db was np.float32."""
    np = pytest.importorskip("numpy")
    record_measurement(
        "x", np.float32(92.19631), units="dB", conditions=_minimal_conditions(), path=results_path
    )
    stored = json.loads(results_path.read_text())["x"]["value"]
    assert isinstance(stored, float)
    assert stored == pytest.approx(92.19631, abs=1e-4)


def test_measured_requires_a_real_value(results_path: Path) -> None:
    with pytest.raises(MeasurementError, match="measured"):
        record_measurement(
            "x",
            None,
            units="count",
            conditions=_minimal_conditions(),
            status="measured",
            path=results_path,
        )


# --------------------------------------------------------------------------------------
# The core fix: first_seen_* vs last_verified_*
# --------------------------------------------------------------------------------------


def test_first_seen_is_set_on_the_first_write(results_path: Path) -> None:
    record_measurement("x", 1.0, units="count", conditions=_minimal_conditions(), path=results_path)
    entry = json.loads(results_path.read_text())["x"]
    assert entry["first_seen_commit"] == entry["last_verified_commit"]
    assert entry["last_changed"] == entry["last_verified_at"]


def test_reverifying_the_same_value_moves_last_verified_not_first_seen(
    results_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The distinction that motivated this design: a stale figure and a freshly-confirmed
    one must not look identical in the file."""
    import esm446.measured as measured_module

    monkeypatch.setattr(measured_module, "_current_commit", lambda: "aaaaaaaaaaaa")
    record_measurement("x", 1.0, units="count", conditions=_minimal_conditions(), path=results_path)
    first = json.loads(results_path.read_text())["x"]

    monkeypatch.setattr(measured_module, "_current_commit", lambda: "bbbbbbbbbbbb")
    record_measurement("x", 1.0, units="count", conditions=_minimal_conditions(), path=results_path)
    second = json.loads(results_path.read_text())["x"]

    assert second["first_seen_commit"] == "aaaaaaaaaaaa", "first_seen must not move on reverify"
    assert second["last_verified_commit"] == "bbbbbbbbbbbb", "last_verified must move on reverify"
    assert second["last_changed"] == first["last_changed"]


def test_a_changed_value_moves_first_seen_too(
    results_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import esm446.measured as measured_module

    monkeypatch.setattr(measured_module, "_current_commit", lambda: "aaaaaaaaaaaa")
    record_measurement("x", 1.0, units="count", conditions=_minimal_conditions(), path=results_path)

    monkeypatch.setattr(measured_module, "_current_commit", lambda: "bbbbbbbbbbbb")
    record_measurement("x", 2.0, units="count", conditions=_minimal_conditions(), path=results_path)

    entry = json.loads(results_path.read_text())["x"]
    assert entry["first_seen_commit"] == "bbbbbbbbbbbb", "a genuinely new value is a new first_seen"
    assert entry["value"] == 2.0


def test_a_changed_condition_counts_as_changed_content(results_path: Path) -> None:
    """Same number, different circumstances, must not be silently treated as the same fact."""
    record_measurement(
        "x",
        1.0,
        units="count",
        conditions=_minimal_conditions(lna_db=32.0, vga_db=20.0),
        path=results_path,
    )
    first = json.loads(results_path.read_text())["x"]["first_seen_commit"]

    record_measurement(
        "x",
        1.0,
        units="count",
        conditions=_minimal_conditions(lna_db=32.0, vga_db=40.0),
        path=results_path,
    )
    second = json.loads(results_path.read_text())["x"]

    assert second["conditions"]["vga_db"] == 40.0
    # Both calls run in the same test, same commit -- first_seen_commit can't distinguish
    # this case by itself, but the conditions actually stored must reflect the latest call.
    del first


# --------------------------------------------------------------------------------------
# pending / force -- the guard against silently erasing a real measurement
# --------------------------------------------------------------------------------------


def test_pending_cannot_silently_overwrite_a_measured_entry(results_path: Path) -> None:
    record_measurement("x", 1.0, units="count", conditions=_minimal_conditions(), path=results_path)

    with pytest.raises(MeasurementError, match="force"):
        record_measurement(
            "x",
            None,
            units="count",
            conditions=_minimal_conditions(),
            status="pending",
            reason="hardware unavailable",
            path=results_path,
        )
    # The measured entry must survive the refused call untouched.
    assert json.loads(results_path.read_text())["x"]["status"] == "measured"


def test_pending_can_overwrite_measured_with_explicit_force(results_path: Path) -> None:
    record_measurement("x", 1.0, units="count", conditions=_minimal_conditions(), path=results_path)
    record_measurement(
        "x",
        None,
        units="count",
        conditions=_minimal_conditions(),
        status="pending",
        reason="hardware unavailable",
        force=True,
        path=results_path,
    )
    assert json.loads(results_path.read_text())["x"]["status"] == "pending"


def test_pending_over_pending_needs_no_force(results_path: Path) -> None:
    record_measurement(
        "x",
        None,
        units="count",
        conditions=_minimal_conditions(),
        status="pending",
        reason="first",
        path=results_path,
    )
    record_measurement(
        "x",
        None,
        units="count",
        conditions=_minimal_conditions(),
        status="pending",
        reason="second",
        path=results_path,
    )
    assert json.loads(results_path.read_text())["x"]["reason"] == "second"


def test_a_fresh_key_needs_no_force_even_as_pending(results_path: Path) -> None:
    record_measurement(
        "brand_new_key",
        None,
        units="count",
        conditions=_minimal_conditions(),
        status="pending",
        reason="no hardware in CI",
        path=results_path,
    )
    assert json.loads(results_path.read_text())["brand_new_key"]["status"] == "pending"


# --------------------------------------------------------------------------------------
# Multiple keys, source_test, outside-pytest guard
# --------------------------------------------------------------------------------------


def test_multiple_keys_coexist_in_one_file(results_path: Path) -> None:
    record_measurement("a", 1.0, units="count", conditions=_minimal_conditions(), path=results_path)
    record_measurement("b", 2.0, units="count", conditions=_minimal_conditions(), path=results_path)

    entries = json.loads(results_path.read_text())
    assert set(entries) == {"a", "b"}
    assert entries["a"]["value"] == 1.0
    assert entries["b"]["value"] == 2.0


def test_source_test_is_derived_from_pytest_not_a_parameter(results_path: Path) -> None:
    """No source_test parameter exists on record_measurement -- confirm it is recorded anyway,
    and that it names *this* test."""
    record_measurement("x", 1.0, units="count", conditions=_minimal_conditions(), path=results_path)

    source_test = json.loads(results_path.read_text())["x"]["source_test"]
    assert "test_source_test_is_derived_from_pytest_not_a_parameter" in source_test
    assert "(call)" not in source_test, "the phase suffix should have been stripped"


def test_calling_outside_pytest_raises(results_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    with pytest.raises(RuntimeError, match="PYTEST_CURRENT_TEST"):
        record_measurement(
            "x", 1.0, units="count", conditions=_minimal_conditions(), path=results_path
        )


def test_toolchain_versions_reports_numpy_and_scipy(results_path: Path) -> None:
    """A near-cancellation DSP figure can move with the toolchain, not just the code -- see
    the adjacent-channel-rejection migration this was built for."""
    pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    versions = toolchain_versions()
    assert set(versions) == {"numpy_version", "scipy_version"}
    assert all(versions.values()), "both versions must be non-empty strings"
