"""Verification of range estimation and its uncertainty.

Anyone can print a distance. The tests that matter here are the ones about the interval
around it: that it is skewed the way the physics is skewed, that it widens when an assumption
is loosened, and above all that a ring said to contain the emitter 95 % of the time actually
does. The last one is what separates an uncertainty estimate from a number with a
plus-or-minus after it.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from esm446.core.geolocation import (
    DEFAULT_DRAWS,
    PropagationPrior,
    estimate_from_report,
    estimate_range,
    free_space_loss_db,
    path_loss_db,
)
from esm446.core.node import EmissionReport

FREQUENCY = 446_093_750.0


def report(estimated_dbm: float | None, calibrated: bool = False) -> EmissionReport:
    return EmissionReport(
        timestamp=1_786_950_000.0,
        frequency_hz=FREQUENCY,
        pmr_channel=8,
        bin_index=9,
        duration_s=4.0,
        peak_power_dbfs=-22.0,
        snr_db=40.0,
        estimated_dbm=estimated_dbm,
        calibrated=calibrated,
        ctcss_tone_hz=114.8,
        classification="FRIEND",
        offset_s=0.0,
        peak_deviation_hz=1_350.0,
        gains={},
    )


# --------------------------------------------------------------------------------------
# The forward model
# --------------------------------------------------------------------------------------


def test_free_space_loss_matches_the_textbook_value() -> None:
    """32.45 + 20log10(f_MHz) + 20log10(d_km), the form every link budget is written in."""
    expected = 32.44778 + 20 * math.log10(446.09375) + 20 * math.log10(1.0)
    assert free_space_loss_db(FREQUENCY, 1_000.0) == pytest.approx(expected, abs=0.01)


def test_path_loss_grows_by_ten_n_per_decade() -> None:
    """The definition of the log-distance exponent, which is easy to write down wrong."""
    near = path_loss_db(100.0, FREQUENCY, 3.5)
    far = path_loss_db(1_000.0, FREQUENCY, 3.5)

    assert far - near == pytest.approx(35.0, abs=0.001)


def test_an_exponent_of_two_is_free_space() -> None:
    assert path_loss_db(500.0, FREQUENCY, 2.0) == pytest.approx(
        free_space_loss_db(FREQUENCY, 500.0), abs=0.001
    )


# --------------------------------------------------------------------------------------
# The estimate
# --------------------------------------------------------------------------------------


def test_a_weaker_signal_is_estimated_further_away() -> None:
    near = estimate_range(-80.0, FREQUENCY, seed=1)
    far = estimate_range(-100.0, FREQUENCY, seed=1)

    assert far.median_m > near.median_m


def test_the_median_matches_the_closed_form_when_nothing_is_uncertain() -> None:
    """With every uncertainty at zero the Monte Carlo must reproduce the algebra exactly."""
    prior = PropagationPrior(
        path_loss_exponent_sigma=0.0,
        shadowing_sigma_db=0.0,
        eirp_sigma_db=0.0,
        calibration_sigma_db=0.0,
    )
    estimate = estimate_range(-95.0, FREQUENCY, prior=prior, draws=1_000, seed=7)

    loss_db = prior.eirp_dbm - (-95.0)
    expected = 10 ** ((loss_db - free_space_loss_db(FREQUENCY)) / (10 * prior.path_loss_exponent))
    assert estimate.median_m == pytest.approx(expected, rel=1e-9)


def test_the_interval_is_skewed_not_symmetric() -> None:
    """The whole reason for Monte Carlo: d +/- k*sigma cannot describe this distribution.

    Distance is exponential in path loss, so the upper half of the interval is much longer
    than the lower half. The v0 estimator's symmetric sigma puts both rings in the wrong
    place.
    """
    estimate = estimate_range(-95.0, FREQUENCY, seed=3)
    below = estimate.median_m - estimate.ring(5)
    above = estimate.ring(95) - estimate.median_m

    assert above > 3 * below


def test_the_percentiles_are_ordered() -> None:
    estimate = estimate_range(-95.0, FREQUENCY, seed=5)
    values = [estimate.ring(p) for p in sorted(estimate.percentiles)]

    assert values == sorted(values)


def only(**overrides: float) -> PropagationPrior:
    """A prior with every uncertainty at zero except the ones named."""
    settings: dict[str, float] = {
        "path_loss_exponent_sigma": 0.0,
        "shadowing_sigma_db": 0.0,
        "eirp_sigma_db": 0.0,
        "calibration_sigma_db": 0.0,
    }
    settings.update(overrides)
    return PropagationPrior(**settings)


def test_loosening_an_assumption_widens_the_interval() -> None:
    """An uncertainty that does not move the answer is not being propagated.

    Measured at 1.6x for shadowing going from 2 dB to 12 dB, rather than the 6x the change
    itself suggests, because the exponent's contribution is already there and dominates. That
    is the point of the next test.
    """
    tight = estimate_range(
        -95.0, FREQUENCY, prior=PropagationPrior(shadowing_sigma_db=2.0), seed=11
    )
    loose = estimate_range(
        -95.0, FREQUENCY, prior=PropagationPrior(shadowing_sigma_db=12.0), seed=11
    )

    assert loose.ring(95) > 1.4 * tight.ring(95)


def test_the_path_loss_exponent_dominates_the_width() -> None:
    """Measured, and worth stating because it decides where effort is worth spending.

    Taking each uncertainty on its own, the 5-95 span is a factor of 24 for the exponent,
    5.8 for shadowing, 1.9 for the emitter's power and 1.5 for the calibration. Narrowing the
    calibration to improve a range estimate would be wasted work while the exponent is a
    guess -- which is a conclusion about priorities, from arithmetic, rather than an opinion.
    """

    def span(**overrides: float) -> float:
        estimate = estimate_range(-95.0, FREQUENCY, prior=only(**overrides), seed=17)
        return estimate.ring(95) / estimate.ring(5)

    exponent = span(path_loss_exponent_sigma=0.5)
    shadowing = span(shadowing_sigma_db=8.0)
    calibration = span(calibration_sigma_db=2.0)

    assert exponent > 3 * shadowing
    assert exponent > 10 * calibration


def test_the_exponent_is_not_allowed_below_free_space() -> None:
    """A propagation exponent under 2 describes a waveguide, not an outdoor path.

    Without the truncation, the tail of a wide prior on the exponent generates distances of
    hundreds of kilometres from a signal a handset produced, and those draws set the upper
    rings. With it, the widest the exponent alone can make the estimate is exactly the
    free-space distance for the assumed power.
    """
    wild = estimate_range(
        -95.0,
        FREQUENCY,
        prior=only(path_loss_exponent=2.2, path_loss_exponent_sigma=1.5),
        seed=23,
    )
    free_space_bound = 10 ** (
        (PropagationPrior().eirp_dbm + 95.0 - free_space_loss_db(FREQUENCY)) / 20
    )

    assert wild.ring(95) == pytest.approx(free_space_bound, rel=1e-9)


# --------------------------------------------------------------------------------------
# Coverage: does the ring contain the truth as often as it says
# --------------------------------------------------------------------------------------


def test_the_ninety_five_percent_ring_contains_the_truth_that_often() -> None:
    """The check that makes the interval mean something.

    Realisations are generated through the same forward model the estimator inverts: a true
    exponent, a true radiated power, a shadowing draw and a calibration error, all from the
    prior. If the inversion is right, the truth falls inside the 95 % ring 95 % of the time.

    Measured over 400 realisations: the 50 % ring covers 50.0 %, the 68 % ring 70.0 %, the
    95 % ring 94.8 %.

    This verifies the arithmetic, not the model. The realisations come from the prior the
    estimator assumes, so coverage would be just as good if that prior were wrong about the
    environment. Establishing that needs measured distances against measured power, which
    needs a calibration -- see #41.
    """
    prior = PropagationPrior()
    rng = np.random.default_rng(2024)
    trials = 400
    contained = 0

    for trial in range(trials):
        exponent = max(rng.normal(prior.path_loss_exponent, prior.path_loss_exponent_sigma), 2.0)
        eirp_dbm = rng.normal(prior.eirp_dbm, prior.eirp_sigma_db)
        shadowing_db = rng.normal(0.0, prior.shadowing_sigma_db)
        calibration_db = rng.normal(0.0, prior.calibration_sigma_db)

        true_distance_m = float(rng.uniform(50.0, 2_000.0))
        received_dbm = (
            eirp_dbm
            + prior.receiver_gain_dbi
            - path_loss_db(true_distance_m, FREQUENCY, exponent)
            - shadowing_db
            + calibration_db
        )

        estimate = estimate_range(
            received_dbm, FREQUENCY, prior=prior, draws=4_000, seed=trial, calibrated=True
        )
        contained += true_distance_m <= estimate.ring(95)

    coverage = contained / trials
    # Measured at 0.95 over these realisations; the binomial standard error at 400 trials is
    # about 1.1 %, so the band is three of those either side.
    assert 0.915 < coverage < 0.985, f"95 % ring covered the truth {coverage:.1%} of the time"


def test_a_narrower_ring_covers_less() -> None:
    """Coverage has to track the percentile, or the rings are decoration."""
    prior = PropagationPrior()
    rng = np.random.default_rng(99)
    trials = 300
    covered = {50: 0, 95: 0}

    for trial in range(trials):
        exponent = max(rng.normal(prior.path_loss_exponent, prior.path_loss_exponent_sigma), 2.0)
        eirp_dbm = rng.normal(prior.eirp_dbm, prior.eirp_sigma_db)
        shadowing_db = rng.normal(0.0, prior.shadowing_sigma_db)
        true_distance_m = float(rng.uniform(50.0, 2_000.0))
        received_dbm = eirp_dbm - path_loss_db(true_distance_m, FREQUENCY, exponent) - shadowing_db

        estimate = estimate_range(
            received_dbm, FREQUENCY, prior=prior, draws=4_000, seed=trial, calibrated=True
        )
        for percentile in covered:
            covered[percentile] += true_distance_m <= estimate.ring(percentile)

    assert covered[50] / trials < 0.7
    assert covered[95] / trials > 0.85
    assert covered[50] < covered[95]


# --------------------------------------------------------------------------------------
# Refusing to estimate
# --------------------------------------------------------------------------------------


def test_an_uncalibrated_report_produces_no_range() -> None:
    """Every report this system currently produces. Blocked on #41, and it says so.

    Inventing the one input the calculation rests on would turn a blocked measurement into a
    confident-looking number, which is the failure the whole calibration flag exists to stop.
    """
    assert estimate_from_report(report(estimated_dbm=None)) is None


def test_a_calibrated_report_produces_a_range() -> None:
    estimate = estimate_from_report(report(estimated_dbm=-95.0, calibrated=True), seed=1)

    assert estimate is not None
    assert estimate.calibrated is True
    assert estimate.median_m > 0.0


def test_an_uncalibrated_estimate_is_flagged_all_the_way_through() -> None:
    """The flag is not decoration: it is what stops a model output being read as a measurement."""
    estimate = estimate_range(-95.0, FREQUENCY, calibrated=False, seed=1)

    assert estimate.calibrated is False
    assert estimate.as_dict()["calibrated"] is False
    assert "UNCALIBRATED" in estimate.describe()


def test_the_output_states_that_bearing_was_not_measured() -> None:
    """One omnidirectional antenna gives a range. Anything more would be a stronger claim."""
    estimate = estimate_range(-95.0, FREQUENCY, calibrated=True, seed=1)

    assert "bearing" in estimate.describe()
    assert "bearing" in estimate.as_dict()["geometry"]


# --------------------------------------------------------------------------------------
# Arguments
# --------------------------------------------------------------------------------------


def test_the_default_draw_count_is_enough_to_settle_the_upper_ring() -> None:
    """Two independent runs must agree to well inside the width they are reporting."""
    first = estimate_range(-95.0, FREQUENCY, draws=DEFAULT_DRAWS, seed=1)
    second = estimate_range(-95.0, FREQUENCY, draws=DEFAULT_DRAWS, seed=2)

    assert first.ring(95) == pytest.approx(second.ring(95), rel=0.15)


def test_a_seed_makes_the_answer_reproducible() -> None:
    assert estimate_range(-95.0, FREQUENCY, seed=42).percentiles == (
        estimate_range(-95.0, FREQUENCY, seed=42).percentiles
    )


def test_zero_draws_is_refused() -> None:
    with pytest.raises(ValueError, match="draws"):
        estimate_range(-95.0, FREQUENCY, draws=0)


def test_a_percentile_outside_the_range_is_refused() -> None:
    with pytest.raises(ValueError, match="percentile"):
        estimate_range(-95.0, FREQUENCY, percentiles=(0, 95))


def test_a_negative_uncertainty_is_refused() -> None:
    with pytest.raises(ValueError, match="negative"):
        PropagationPrior(shadowing_sigma_db=-1.0)


def test_the_median_is_always_available_even_if_not_requested() -> None:
    """It is the estimate; a set of rings with no centre is not an answer."""
    estimate = estimate_range(-95.0, FREQUENCY, percentiles=(90,), seed=1)

    assert 50 in estimate.percentiles
    assert estimate.median_m > 0.0


# --------------------------------------------------------------------------------------
# What happens when the model is wrong, which the coverage test cannot see
# --------------------------------------------------------------------------------------


def coverage_at(true_exponent: float, percentile: int, trials: int = 150) -> float:
    """Fraction of realisations the ring contains, when the truth obeys a different exponent.

    The coverage test draws its realisations from the estimator's own prior, so it verifies
    the arithmetic and says nothing about the environment. This fixes the true exponent away
    from the assumed one and asks what the rings then achieve.
    """
    prior = PropagationPrior()
    rng = np.random.default_rng(7)
    contained = 0
    for trial in range(trials):
        shadowing_db = rng.normal(0.0, prior.shadowing_sigma_db)
        truth_m = float(rng.uniform(50.0, 2_000.0))
        received_dbm = (
            prior.eirp_dbm - path_loss_db(truth_m, FREQUENCY, true_exponent) - shadowing_db
        )
        estimate = estimate_range(
            received_dbm, FREQUENCY, prior=prior, draws=3_000, seed=trial, calibrated=True
        )
        contained += truth_m <= estimate.ring(percentile)
    return contained / trials


def test_the_rings_stay_safe_when_the_environment_is_worse_than_assumed() -> None:
    """The direction that does not hurt, and the reason the failure mode is worth naming.

    If the real path is more obstructed than the prior assumes, the estimator places the
    emitter further out than it is and the rings still contain it. Over-covering is a loss of
    precision, not a wrong answer.
    """
    assert coverage_at(4.0, 95) >= 0.95
    assert coverage_at(4.5, 95) >= 0.95


def test_the_rings_undercover_badly_when_the_environment_is_clearer() -> None:
    """The direction that does hurt, quantified rather than left as a caveat.

    A clearer path than assumed means the emitter is much further away than the model allows
    for, and the ring simply does not reach it. At an exponent of 2.5 -- near free space, which
    is an open field or a rooftop -- the 95 % ring contains the truth about a third of the
    time. Deployed there, four rings in five that claim to hold the emitter do not.

    This is why REQ-CAL-005 is PARTIAL: the arithmetic is verified, the prior is not, and this
    is the size of the exposure that leaves.
    """
    assert coverage_at(2.5, 95) < 0.5
    assert coverage_at(3.0, 95) < 0.98


def test_the_safe_direction_is_to_assume_more_obstruction_not_less() -> None:
    """The operational conclusion, as a test so it cannot be softened by editing prose."""
    clearer = coverage_at(3.0, 95)
    obstructed = coverage_at(4.0, 95)

    assert (
        obstructed > clearer
    ), "assuming more obstruction than there is must be the conservative error"
