"""Verification of the Electronic Order of Battle.

Occupancy and burst statistics are arithmetic and are tested as such. Emitter grouping is
inference, and the tests that matter there are the ones about what it refuses to claim: a
count it cannot justify, a separation the recording does not support.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from esm446.analysis.eob import (
    cluster_emitters,
    compute_occupancy,
    describe,
    summarise,
)
from esm446.cli import eob as eob_cli
from esm446.cli.demo import run_demo
from esm446.core.node import EmissionReport
from esm446.io.sinks import SqliteSink
from esm446.sim.scenario import Emitter, Propagation, Scenario, Transmission

#: 2026-08-13 09:00:00 UTC, so hour binning has a value to land on.
BASE_TIME = 1_786_950_000.0


def emission(
    at: float = 0.0,
    duration: float = 4.0,
    frequency: float = 446_093_750.0,
    channel: int | None = 8,
    tone: float | None = 114.8,
    power: float = -22.0,
    deviation: float = 1_350.0,
) -> EmissionReport:
    return EmissionReport(
        timestamp=BASE_TIME + at,
        frequency_hz=frequency,
        pmr_channel=channel,
        bin_index=9,
        duration_s=duration,
        peak_power_dbfs=power,
        snr_db=40.0,
        estimated_dbm=None,
        calibrated=False,
        ctcss_tone_hz=tone,
        classification="FRIEND",
        offset_s=at,
        peak_deviation_hz=deviation,
        gains={"lna_db": 0.0, "vga_db": 0.0, "amp_enabled": False},
    )


# --------------------------------------------------------------------------------------
# Grouping: what it claims
# --------------------------------------------------------------------------------------


def test_repeated_transmissions_from_one_radio_group_together() -> None:
    reports = [emission(at=0.0), emission(at=20.0), emission(at=40.0)]
    profiles = cluster_emitters(reports)

    assert len(profiles) == 1
    assert profiles[0].transmission_count == 3
    assert profiles[0].pmr_channel == 8


def test_different_channels_are_different_emitters() -> None:
    reports = [
        emission(at=0.0, frequency=446_093_750.0, channel=8),
        emission(at=10.0, frequency=446_031_250.0, channel=3),
    ]
    assert len(cluster_emitters(reports)) == 2


def test_the_same_channel_with_different_tones_is_two_emitters() -> None:
    """The case the hardware session produced: two handsets, one channel apart, two codes."""
    reports = [
        emission(at=0.0, tone=114.8),
        emission(at=10.0, tone=141.3),
    ]
    profiles = cluster_emitters(reports)

    assert len(profiles) == 2
    assert {p.ctcss_tone_hz for p in profiles} == {114.8, 141.3}


def test_absence_of_a_tone_is_its_own_identity() -> None:
    """An emitter with no CTCSS is not the same as one whose tone happened to be missed."""
    reports = [emission(at=0.0, tone=None), emission(at=10.0, tone=114.8)]
    assert len(cluster_emitters(reports)) == 2


def test_a_drifting_handset_is_still_one_emitter() -> None:
    """Crystal drift of a few hundred hertz must not fragment an emitter across a session."""
    reports = [
        emission(at=0.0, frequency=446_093_729.0),
        emission(at=20.0, frequency=446_093_761.0),
        emission(at=40.0, frequency=446_093_755.0),
    ]
    profiles = cluster_emitters(reports)

    assert len(profiles) == 1
    assert profiles[0].frequency_spread_hz < 100.0


# --------------------------------------------------------------------------------------
# Grouping: what it refuses to claim
# --------------------------------------------------------------------------------------


def test_a_count_is_a_lower_bound_unless_transmissions_overlap() -> None:
    """The sharp limit, stated rather than papered over.

    Three transmissions on one channel with one tone, none overlapping, are consistent with
    one radio taking turns and with three radios sharing a channel and a code. The recording
    contains nothing that separates those, so the profile says so.
    """
    reports = [emission(at=0.0), emission(at=20.0), emission(at=40.0)]
    profile = cluster_emitters(reports)[0]

    assert profile.proven_multiple is False
    assert profile.count_is_lower_bound is True


def test_overlapping_transmissions_prove_more_than_one_transmitter() -> None:
    """Simultaneity is the one observation that forces a second radio."""
    reports = [emission(at=0.0, duration=10.0), emission(at=5.0, duration=10.0)]
    profile = cluster_emitters(reports)[0]

    assert profile.proven_multiple is True
    assert profile.count_is_lower_bound is False


def test_the_report_states_the_limitation_in_words() -> None:
    """A caveat only in a dataclass field is a caveat nobody reads."""
    text = describe([emission(at=0.0), emission(at=20.0)])
    assert ">= 1" in text
    assert "lower bound" in text


# --------------------------------------------------------------------------------------
# Burst statistics
# --------------------------------------------------------------------------------------


def test_airtime_and_duty_cycle() -> None:
    reports = [emission(at=0.0, duration=4.0), emission(at=10.0, duration=6.0)]
    profile = cluster_emitters(reports)[0]

    assert profile.total_airtime_s == pytest.approx(10.0)
    assert profile.duty_cycle == pytest.approx(10.0 / 16.0)


def test_median_gap_measures_the_silence_between_overs() -> None:
    reports = [
        emission(at=0.0, duration=4.0),
        emission(at=10.0, duration=4.0),
        emission(at=20.0, duration=4.0),
    ]
    assert cluster_emitters(reports)[0].median_gap_s == pytest.approx(6.0)


def test_median_duration_distinguishes_traffic_types() -> None:
    """A talker, a repeater and a data link differ here more than anywhere else."""
    talker = cluster_emitters([emission(at=0.0, duration=5.0), emission(at=20.0, duration=4.0)])
    burst = cluster_emitters(
        [
            emission(at=0.0, duration=0.2, frequency=446_031_250.0, channel=3),
            emission(at=5.0, duration=0.2, frequency=446_031_250.0, channel=3),
        ]
    )
    assert talker[0].median_duration_s > 10 * burst[0].median_duration_s


def test_power_spread_is_reported_but_does_not_split_emitters() -> None:
    """Power tracks distance, not identity, so it is evidence and not a decision."""
    reports = [emission(at=0.0, power=-22.0), emission(at=20.0, power=-40.0)]
    profiles = cluster_emitters(reports)

    assert len(profiles) == 1
    assert profiles[0].power_spread_db > 10.0


# --------------------------------------------------------------------------------------
# Occupancy
# --------------------------------------------------------------------------------------


def test_occupancy_bins_by_channel_and_hour() -> None:
    reports = [emission(at=0.0, duration=4.0), emission(at=3_600.0, duration=6.0)]
    occupancy = compute_occupancy(reports)

    hours = {hour for _, hour in occupancy.airtime_s}
    assert len(hours) == 2, "emissions an hour apart must land in different hour bins"
    assert occupancy.total_airtime_s == pytest.approx(10.0)


def test_occupancy_keeps_off_grid_emissions() -> None:
    """Finding emissions off the channel plan is a large part of why the band is surveyed."""
    reports = [emission(at=0.0, channel=None, frequency=446_000_000.0)]
    occupancy = compute_occupancy(reports)

    assert (None, 9) in occupancy.airtime_s or any(c is None for c, _ in occupancy.airtime_s)
    assert occupancy.busiest_channels()[0][0] is None


def test_busiest_channel_is_the_one_with_most_airtime_not_most_bursts() -> None:
    """Twenty short bursts are less occupancy than one long transmission."""
    reports = [emission(at=0.0, duration=60.0, channel=8, frequency=446_093_750.0)]
    reports += [
        emission(at=float(i), duration=0.5, channel=3, frequency=446_031_250.0) for i in range(20)
    ]
    assert compute_occupancy(reports).busiest_channels()[0][0] == 8


def test_band_load_can_exceed_one_with_simultaneous_users() -> None:
    """Deliberate: the figure measures load, not the probability the band is busy."""
    reports = [
        emission(at=0.0, duration=10.0),
        emission(at=0.0, duration=10.0, channel=3, frequency=446_031_250.0),
    ]
    assert compute_occupancy(reports).band_duty_cycle > 1.0


# --------------------------------------------------------------------------------------
# Edges
# --------------------------------------------------------------------------------------


def test_an_empty_band_produces_an_empty_order_of_battle() -> None:
    assert cluster_emitters([]) == []
    assert compute_occupancy([]).total_airtime_s == 0.0
    assert describe([]) == "no emissions"


def test_a_single_emission_does_not_divide_by_zero() -> None:
    profile = cluster_emitters([emission(at=0.0)])[0]

    assert profile.median_gap_s == 0.0
    assert profile.frequency_spread_hz == 0.0
    assert profile.duty_cycle >= 0.0


def test_summary_serialises_to_json_friendly_types() -> None:
    import json

    payload = summarise([emission(at=0.0), emission(at=20.0)])
    round_tripped = json.loads(json.dumps(payload))

    assert round_tripped["emissions"] == 2
    assert round_tripped["emitters"][0]["count_is_lower_bound"] is True


# --------------------------------------------------------------------------------------
# Command line, over a real store
# --------------------------------------------------------------------------------------


def _store(path: Path) -> Path:
    """Write a two-emitter capture to a store, the way the node would."""
    with SqliteSink(path) as sink:
        sink.write(
            [
                emission(at=0.0, tone=114.8),
                emission(at=20.0, tone=114.8),
                emission(at=10.0, tone=141.3, frequency=446_031_250.0, channel=3),
            ]
        )
    return path


def test_the_cli_renders_an_order_of_battle_from_a_store(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert eob_cli.main([str(_store(tmp_path / "e.db"))]) == 0

    out = capsys.readouterr().out
    assert "PMR8/114.8Hz" in out
    assert "PMR3/141.3Hz" in out
    assert "lower bound" in out


def test_the_cli_emits_json_on_request(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    import json

    assert eob_cli.main([str(_store(tmp_path / "e.db")), "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["emissions"] == 3
    assert len(payload["emitters"]) == 2


def test_the_cli_filters_by_channel(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    import json

    assert eob_cli.main([str(_store(tmp_path / "e.db")), "--json", "--channel", "3"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["emissions"] == 1
    assert payload["emitters"][0]["pmr_channel"] == 3


def test_the_cli_filters_by_time(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    import json

    since = str(BASE_TIME + 5.0)
    assert eob_cli.main([str(_store(tmp_path / "e.db")), "--json", "--since", since]) == 0

    assert json.loads(capsys.readouterr().out)["emissions"] == 2


def test_the_cli_reports_a_missing_store_rather_than_traceback(tmp_path: Path) -> None:
    assert eob_cli.main([str(tmp_path / "never-captured.db")]) == 1


# --------------------------------------------------------------------------------------
# Over a scenario whose emitter count is known
# --------------------------------------------------------------------------------------


def test_a_scene_with_two_emitters_is_reported_as_two_emitters() -> None:
    """The whole chain against ground truth: two radios, one of which speaks twice.

    Everything above works on hand-built reports. This one starts from synthetic IQ and goes
    through the channeliser, the detector, the tracker and the tone identification, so a
    grouping failure caused by a measurement error upstream — a frequency estimate off by
    more than the clustering tolerance, a tone missed on one over and found on the next —
    shows up here and nowhere else.
    """
    scenario = Scenario(
        name="two-emitters",
        duration_s=8.0,
        noise_floor_dbm=-110.0,
        seed=11,
        propagation=Propagation(path_loss_exponent=3.5, shadowing_sigma_db=0.0),
        emitters=[
            Emitter(
                name="alpha",
                channel=4,
                distance_m=400.0,
                ctcss_hz=114.8,
                transmissions=[Transmission(0.5, 2.0), Transmission(4.0, 5.5)],
            ),
            Emitter(
                name="bravo",
                channel=12,
                distance_m=600.0,
                ctcss_hz=141.3,
                transmissions=[Transmission(2.3, 3.8)],
            ),
        ],
    )
    reports, _, _, _ = run_demo(scenario)
    profiles = cluster_emitters(reports)

    assert len(profiles) == 2, [p.label for p in profiles]

    by_channel = {p.pmr_channel: p for p in profiles}
    assert by_channel[4].transmission_count == 2
    assert by_channel[12].transmission_count == 1
    # Nothing in the scene overlaps, so neither count may be asserted as exact.
    assert all(p.count_is_lower_bound for p in profiles)
