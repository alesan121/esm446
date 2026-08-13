"""Verification of detection scoring.

The point of the matcher is to keep apart failures that a single accuracy number would
collapse: a missed emission, a fabricated one, one transmission reported as several, and
several reported as one. Each says something different about which stage is misbehaving, so
each is tested separately.
"""

from __future__ import annotations

import pytest

from esm446.core.node import EmissionReport
from esm446.sim.metrics import score
from esm446.sim.scenario import TruthEmission

FREQUENCY = 446_068_750.0


def truth(
    start: float = 1.0,
    stop: float = 4.0,
    frequency: float = FREQUENCY,
    channel: int | None = 6,
    ctcss: float | None = 114.8,
    snr_db: float = 30.0,
) -> TruthEmission:
    return TruthEmission(
        emitter="alpha",
        start_s=start,
        stop_s=stop,
        frequency_hz=frequency,
        pmr_channel=channel,
        ctcss_hz=ctcss,
        received_dbm=-90.0,
        snr_db=snr_db,
    )


def report(
    start: float = 1.0,
    duration: float = 3.0,
    frequency: float = FREQUENCY,
    channel: int | None = 6,
    ctcss: float | None = 114.8,
) -> EmissionReport:
    return EmissionReport(
        timestamp=start,
        frequency_hz=frequency,
        pmr_channel=channel,
        bin_index=158,
        duration_s=duration,
        peak_power_dbfs=-30.0,
        snr_db=30.0,
        estimated_dbm=None,
        calibrated=False,
        ctcss_tone_hz=ctcss,
        classification="FRIEND",
        offset_s=0.0,
        peak_deviation_hz=750.0,
        gains={},
    )


# --------------------------------------------------------------------------------------
# The four outcomes
# --------------------------------------------------------------------------------------


def test_a_clean_match_scores_perfectly() -> None:
    result = score([report()], [truth()], scene_duration_s=10.0, scene_start=0.0)

    assert result.probability_of_detection == 1.0
    assert result.spurious == []
    assert result.num_fragmented == 0
    assert result.channel_accuracy == 1.0
    assert result.ctcss_accuracy == 1.0
    assert result.duration_rmse_s == pytest.approx(0.0)


def test_a_missed_emission_lowers_detection_without_adding_a_false_alarm() -> None:
    result = score([], [truth()], scene_duration_s=10.0, scene_start=0.0)

    assert result.probability_of_detection == 0.0
    assert result.spurious == []


def test_a_report_matching_nothing_is_spurious() -> None:
    """Against an empty scene this is the end-to-end false alarm rate."""
    result = score([report()], [], scene_duration_s=10.0, scene_start=0.0)

    assert len(result.spurious) == 1
    assert result.false_alarms_per_second == pytest.approx(0.1)


def test_one_transmission_reported_twice_is_fragmented_not_spurious() -> None:
    """The failure a hit-or-miss score cannot see: the tracker hangover is too short."""
    reports = [report(start=1.0, duration=1.0), report(start=2.5, duration=1.4)]
    result = score(reports, [truth()], scene_duration_s=10.0, scene_start=0.0)

    assert result.probability_of_detection == 1.0
    assert result.num_fragmented == 1
    assert result.spurious == []


def test_one_report_spanning_two_transmissions_is_merged() -> None:
    """The opposite failure: the hangover is too long and two overs ran together."""
    truths = [truth(start=1.0, stop=2.0), truth(start=3.0, stop=4.0)]
    result = score(
        [report(start=1.0, duration=3.0)], truths, scene_duration_s=10.0, scene_start=0.0
    )

    assert result.merged == 1


# --------------------------------------------------------------------------------------
# Frequency and channel
# --------------------------------------------------------------------------------------


def test_a_report_a_channel_away_does_not_match() -> None:
    result = score(
        [report(frequency=FREQUENCY + 12_500.0)], [truth()], scene_duration_s=10.0, scene_start=0.0
    )
    assert result.probability_of_detection == 0.0
    assert len(result.spurious) == 1


def test_a_report_a_few_hundred_hertz_off_still_matches() -> None:
    """Handsets drift; the matcher should not treat crystal error as a different emitter."""
    result = score(
        [report(frequency=FREQUENCY + 350.0)], [truth()], scene_duration_s=10.0, scene_start=0.0
    )
    assert result.probability_of_detection == 1.0


def test_snapping_an_off_grid_emitter_to_a_channel_counts_as_wrong() -> None:
    """Reporting an off-grid emission on its nearest channel is a wrong answer.

    Detecting emissions that are not on the channel plan is a large part of why the band is
    surveyed at all, so this must not be scored as a rounding.
    """
    off_grid = truth(frequency=446_162_500.0, channel=None)
    snapped = report(frequency=446_162_500.0, channel=14)

    result = score([snapped], [off_grid], scene_duration_s=10.0, scene_start=0.0)
    assert result.probability_of_detection == 1.0
    assert result.channel_accuracy == 0.0


def test_correctly_reporting_off_grid_scores_full_marks() -> None:
    off_grid = truth(frequency=446_162_500.0, channel=None)
    result = score(
        [report(frequency=446_162_499.0, channel=None)],
        [off_grid],
        scene_duration_s=10.0,
        scene_start=0.0,
    )
    assert result.channel_accuracy == 1.0


# --------------------------------------------------------------------------------------
# Identification
# --------------------------------------------------------------------------------------


def test_the_wrong_tone_is_scored_wrong() -> None:
    result = score(
        [report(ctcss=141.3)], [truth(ctcss=114.8)], scene_duration_s=10.0, scene_start=0.0
    )
    assert result.ctcss_accuracy == 0.0


def test_correctly_reporting_no_tone_scores_full_marks() -> None:
    """Absence identified as absence is a correct answer, not a failure to identify."""
    result = score(
        [report(ctcss=None)], [truth(ctcss=None)], scene_duration_s=10.0, scene_start=0.0
    )
    assert result.ctcss_accuracy == 1.0


def test_missing_a_tone_that_was_transmitted_is_scored_wrong() -> None:
    result = score(
        [report(ctcss=None)], [truth(ctcss=114.8)], scene_duration_s=10.0, scene_start=0.0
    )
    assert result.ctcss_accuracy == 0.0


# --------------------------------------------------------------------------------------
# Aggregates
# --------------------------------------------------------------------------------------


def test_duration_error_is_measured_against_the_truth_extent() -> None:
    result = score(
        [report(duration=2.0)], [truth(start=1.0, stop=4.0)], scene_duration_s=10.0, scene_start=0.0
    )
    assert result.duration_rmse_s == pytest.approx(1.0)


def test_detection_is_binned_by_the_snr_it_was_generated_at() -> None:
    """The curve that says what the node can actually hear."""
    truths = [
        truth(start=1.0, stop=2.0, snr_db=6.0),
        truth(start=3.0, stop=4.0, snr_db=7.0),
        truth(start=5.0, stop=6.0, snr_db=22.0),
    ]
    reports = [report(start=5.0, duration=1.0)]

    rows = score(reports, truths, scene_duration_s=10.0, scene_start=0.0).detection_by_snr()
    by_bin = {low: (probability, count) for low, probability, count in rows}

    assert by_bin[5.0] == (0.0, 2)
    assert by_bin[20.0] == (1.0, 1)


def test_empty_scene_and_empty_output_does_not_divide_by_zero() -> None:
    result = score([], [], scene_duration_s=10.0, scene_start=0.0)

    assert result.probability_of_detection == 0.0
    assert result.channel_accuracy == 0.0
    assert result.duration_rmse_s == 0.0
    assert "transmissions          0" in result.describe()


def test_summary_dictionary_carries_the_headline_figures() -> None:
    payload = score([report()], [truth()], scene_duration_s=10.0, scene_start=0.0).as_dict()

    assert payload["transmissions"] == 1
    assert payload["probability_of_detection"] == 1.0
    assert payload["spurious"] == 0
