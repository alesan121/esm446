"""Verification of the dBFS-to-dBm power calibration.

The behaviour under test is mostly about refusal. A calibration that extrapolates past its
fitted range, or that applies an offset measured at one gain setting to samples taken at
another, produces a number that looks like a measurement and is not. Most of these tests
check that the module declines to do that.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from esm446.core.calibration import (
    CalibrationPoint,
    PowerCalibration,
    estimate_dynamic_range_db,
    fit_calibration,
    linear_to_dbfs,
    required_sweep_levels,
)
from esm446.core.rfchain import quantise_gains

GAINS = quantise_gains(32.0, 20.0)
OTHER_GAINS = quantise_gains(16.0, 20.0)


def linear_sweep(
    offset_db: float = -50.0,
    count: int = 13,
    step_db: float = 5.0,
    top_dbfs: float = -10.0,
):
    """A perfectly linear receiver, for checking the fit recovers what it was given.

    Constructed from the digital side outwards: pick the dBFS the receiver reads, then the
    input power that would have produced it under ``dbm = dbfs + offset_db``. The fit must
    recover ``offset_db`` and a slope of exactly 1.
    """
    points = []
    for i in range(count):
        measured_dbfs = top_dbfs - i * step_db
        input_dbm = measured_dbfs + offset_db
        points.append(
            CalibrationPoint(
                source_dbm=0.0,
                attenuation_db=-input_dbm,
                measured_dbfs=measured_dbfs,
            )
        )
    return points


# --------------------------------------------------------------------------------------
# Units
# --------------------------------------------------------------------------------------


def test_full_scale_reads_zero_dbfs() -> None:
    assert linear_to_dbfs(1.0) == pytest.approx(0.0)


def test_half_power_reads_minus_three_dbfs() -> None:
    assert linear_to_dbfs(0.5) == pytest.approx(-3.0103, abs=1e-4)


def test_zero_power_is_floored_rather_than_infinite() -> None:
    assert np.isfinite(linear_to_dbfs(0.0))


# --------------------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------------------


def test_fit_recovers_unit_slope_from_a_linear_receiver() -> None:
    """A linear receiver must give one dB out per dB in; the slope is the sanity check."""
    calibration = fit_calibration(GAINS, linear_sweep())

    assert calibration.slope == pytest.approx(1.0, abs=1e-6)
    assert calibration.r_squared > 0.999
    assert calibration.residual_rms_db < 1e-6


def test_fit_recovers_the_offset_it_was_given() -> None:
    calibration = fit_calibration(GAINS, linear_sweep(offset_db=-50.0))

    assert calibration.offset_db == pytest.approx(-50.0, abs=1e-6)
    assert calibration.to_dbm(-40.0) == pytest.approx(-90.0, abs=1e-6)


def test_fit_discards_compressed_points_at_the_top_of_the_sweep() -> None:
    """Isolating the linear region is the point of the iterative discard.

    The three strongest points are made to compress by 6 dB. They must be dropped, and the
    surviving validity range must stop below them -- which is exactly the information a
    single-point calibration cannot provide.
    """
    points = linear_sweep(count=13, step_db=5.0, top_dbfs=-10.0)
    compressed = [
        (
            CalibrationPoint(p.source_dbm, p.attenuation_db, p.measured_dbfs)
            if index >= 3
            else CalibrationPoint(p.source_dbm, p.attenuation_db, p.measured_dbfs - 6.0)
        )
        for index, p in enumerate(points)
    ]

    calibration = fit_calibration(GAINS, compressed)

    assert calibration.num_points < len(points)
    assert calibration.slope == pytest.approx(1.0, abs=0.05)
    assert calibration.valid_dbfs_range[1] < -10.0


def test_fit_rejects_too_few_points() -> None:
    with pytest.raises(ValueError, match="at least 3"):
        fit_calibration(GAINS, linear_sweep(count=2))


def test_dynamic_range_reflects_the_linear_span() -> None:
    assert estimate_dynamic_range_db(linear_sweep(count=13, step_db=5.0)) == pytest.approx(60.0)


def test_sweep_levels_descend_from_the_maximum() -> None:
    levels = required_sweep_levels(max_input_dbm=-20.0, dynamic_range_db=60.0, step_db=5.0)
    assert levels[0] == -20.0
    assert levels[-1] == -80.0
    assert len(levels) == 13


# --------------------------------------------------------------------------------------
# Refusals: the behaviour that keeps an uncalibrated number from looking measured
# --------------------------------------------------------------------------------------


def test_uncalibrated_instance_reports_itself_as_such() -> None:
    assert PowerCalibration().is_calibrated is False


def test_uncalibrated_conversion_returns_none_rather_than_a_guess() -> None:
    assert PowerCalibration().to_dbm(-60.0, GAINS) is None


def test_conversion_refuses_a_gain_configuration_it_was_not_measured_at() -> None:
    """The flaw that made v0's OFFSET_CAL meaningless: an offset is gain-specific."""
    calibration = PowerCalibration()
    calibration.add(fit_calibration(GAINS, linear_sweep()))

    assert calibration.to_dbm(-60.0, GAINS) is not None
    assert calibration.to_dbm(-60.0, OTHER_GAINS) is None


def test_conversion_refuses_to_extrapolate_past_the_fitted_range() -> None:
    """Extrapolating is how a compressed reading becomes a confident overestimate."""
    calibration = PowerCalibration()
    calibration.add(fit_calibration(GAINS, linear_sweep(count=13, step_db=5.0)))
    low, high = calibration.get(GAINS).valid_dbfs_range

    assert calibration.to_dbm((low + high) / 2, GAINS) is not None
    assert calibration.to_dbm(high + 10.0, GAINS) is None
    assert calibration.to_dbm(low - 10.0, GAINS) is None


# --------------------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------------------


def test_calibration_survives_a_save_and_load_round_trip(tmp_path: Path) -> None:
    original = PowerCalibration()
    original.add(fit_calibration(GAINS, linear_sweep()))
    path = tmp_path / "calibration.yaml"
    original.save(path)

    loaded = PowerCalibration.load(path)

    assert loaded.is_calibrated
    assert loaded.to_dbm(-60.0, GAINS) == pytest.approx(original.to_dbm(-60.0, GAINS))


def test_missing_calibration_file_yields_an_uncalibrated_instance(tmp_path: Path) -> None:
    """A node with no calibration must start and run, reporting dBFS only."""
    loaded = PowerCalibration.load(tmp_path / "does-not-exist.yaml")
    assert loaded.is_calibrated is False


def test_saved_calibration_records_the_gains_it_was_measured_at(tmp_path: Path) -> None:
    calibration = PowerCalibration()
    calibration.add(fit_calibration(GAINS, linear_sweep()))
    path = tmp_path / "calibration.yaml"
    calibration.save(path)

    assert "lna32" in path.read_text()
    assert "vga20" in path.read_text()
