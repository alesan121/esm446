"""Verification of the antenna plausibility bound and band coverage.

Antenna gain multiplies straight through the link budget, so these tests guard the point
where an unphysical specification would enter the system and produce a confident wrong
answer about detection range.
"""

from __future__ import annotations

import pytest

from esm446.core import bands
from esm446.core.antenna import (
    CATALOGUE,
    DIPOLE_GAIN_DBI,
    Antenna,
    audit,
    max_plausible_gain_dbi,
    required_dimension_m,
    usable_at,
    wavelength,
)

PMR = bands.DEFAULT_CENTRE_HZ


def test_wavelength_at_pmr446() -> None:
    assert wavelength(PMR) == pytest.approx(0.672, abs=0.001)


# --------------------------------------------------------------------------------------
# The aperture bound
# --------------------------------------------------------------------------------------


def test_bound_and_required_dimension_are_inverses() -> None:
    for dimension in (0.1, 0.5, 2.0):
        gain = max_plausible_gain_dbi(PMR, dimension)
        assert required_dimension_m(PMR, gain) == pytest.approx(dimension, rel=1e-9)


def test_bound_scales_with_aperture_and_frequency() -> None:
    """Doubling the aperture dimension is 6 dB; doubling frequency is another 6 dB."""
    base = max_plausible_gain_dbi(PMR, 0.5)
    assert max_plausible_gain_dbi(PMR, 1.0) == pytest.approx(base + 6.02, abs=0.05)
    assert max_plausible_gain_dbi(2 * PMR, 0.5) == pytest.approx(base + 6.02, abs=0.05)


def test_a_dipole_sized_antenna_lands_near_dipole_gain() -> None:
    """Sanity anchor: a half-wave element should bound out in the low single digits dBi."""
    half_wave = wavelength(PMR) / 2
    bound = max_plausible_gain_dbi(PMR, half_wave)
    assert DIPOLE_GAIN_DBI - 3.0 < bound < DIPOLE_GAIN_DBI + 4.0


def test_thirtyfive_dbi_at_446_mhz_would_need_a_twelve_metre_aperture() -> None:
    """The claim, converted into the physical object it would have to be."""
    needed = required_dimension_m(PMR, 35.0)
    assert needed > 10.0


# --------------------------------------------------------------------------------------
# Plausibility of the actual catalogue
# --------------------------------------------------------------------------------------


def test_inflated_gain_claim_is_rejected() -> None:
    antenna = Antenna("panel", 700e6, 2700e6, claimed_gain_dbi=35.0, largest_dimension_m=0.3)
    problem = antenna.plausibility(900e6)
    assert problem is not None
    assert "35 dBi claimed" in problem


def test_modest_gain_claim_is_accepted() -> None:
    """The bound must not reject ordinary antennas, or it is useless as a filter."""
    antenna = Antenna("yagi", 400e6, 500e6, claimed_gain_dbi=9.0, largest_dimension_m=1.2)
    assert antenna.plausibility(PMR) is None


def test_no_claim_means_nothing_to_object_to() -> None:
    antenna = Antenna("whip", 40e6, 860e6, largest_dimension_m=1.0)
    assert antenna.plausibility(PMR) is None


# --------------------------------------------------------------------------------------
# Coverage: the check that comes before gain
# --------------------------------------------------------------------------------------


def test_only_the_wideband_antennas_cover_pmr446() -> None:
    """Three of the five are out of band at 446 MHz, gain figures notwithstanding."""
    usable = {antenna.name for antenna in usable_at(PMR)}
    assert usable == {"wideband 40-860 MHz", "wideband 40 MHz - 6 GHz"}


def test_a_700_mhz_antenna_does_not_cover_446_mhz() -> None:
    antenna = Antenna("panel", 700e6, 2700e6, claimed_gain_dbi=12.0)
    assert not antenna.covers(PMR)
    assert antenna.covers(1_800e6)


def test_audit_reports_coverage_before_gain() -> None:
    findings = audit(PMR)
    out_of_band = [f for f in findings if "does not cover" in f]
    assert len(out_of_band) == 3
    assert not any(
        "dBi claimed" in f for f in findings
    ), "an out-of-band antenna should be reported as out of band, not argued about on gain"


def test_the_gain_claim_is_still_caught_inside_its_own_band() -> None:
    findings = audit(1_800e6)
    assert any("35 dBi claimed" in f for f in findings)


# --------------------------------------------------------------------------------------
# What reaches the link budget
# --------------------------------------------------------------------------------------


def test_claimed_gain_never_reaches_the_link_budget() -> None:
    """An unmeasured antenna contributes an assumption, and the budget should say so."""
    antenna = Antenna("panel", 700e6, 2700e6, claimed_gain_dbi=35.0, largest_dimension_m=0.3)
    assert antenna.effective_gain_dbi(1_800e6) == 0.0
    assert antenna.effective_gain_dbi(1_800e6, fallback_dbi=2.0) == 2.0


def test_measured_gain_is_used_when_available() -> None:
    antenna = Antenna("whip", 40e6, 860e6, claimed_gain_dbi=9.0, measured_gain_dbi=1.8)
    assert antenna.effective_gain_dbi(PMR) == 1.8


def test_catalogue_carries_no_measured_gains_yet() -> None:
    """Guards against a placeholder becoming a fact. Phase 4 fills these in."""
    assert all(antenna.measured_gain_dbi is None for antenna in CATALOGUE)
