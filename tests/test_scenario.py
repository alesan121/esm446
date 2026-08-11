"""Verification of the scenario simulator.

The simulator is the reference the whole testbench is measured against, so it has to be
correct in ways that are easy to get subtly wrong: reproducible from its seed, honest about
the signal-to-noise ratio it actually produced, and placing emitters where it says it did.
A simulator that quietly generates 22 dB more SNR than it records makes every
probability-of-detection curve wrong without anything failing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from esm446.core import bands
from esm446.core.channelizer import ChannelizerConfig, PolyphaseChannelizer
from esm446.sim.scenario import Emitter, Propagation, Scenario, Transmission


def make_scenario(**overrides) -> Scenario:
    defaults = {
        "name": "test",
        "duration_s": 2.0,
        "sample_rate": 2_000_000.0,
        "centre_frequency": float(bands.DEFAULT_CENTRE_HZ),
        "noise_floor_dbm": -100.0,
        "seed": 7,
        "emitters": [
            Emitter(
                name="alpha",
                channel=6,
                eirp_dbm=29.0,
                distance_m=500.0,
                ctcss_hz=114.8,
                transmissions=[Transmission(0.3, 1.5)],
            )
        ],
    }
    return Scenario(**{**defaults, **overrides})


# --------------------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------------------


def test_same_seed_produces_identical_iq() -> None:
    first, _ = make_scenario().generate()
    second, _ = make_scenario().generate()
    np.testing.assert_array_equal(first, second)


def test_different_seed_produces_different_iq() -> None:
    first, _ = make_scenario(seed=1).generate()
    second, _ = make_scenario(seed=2).generate()
    assert not np.array_equal(first, second)


def test_scene_length_matches_the_requested_duration() -> None:
    scenario = make_scenario(duration_s=1.5)
    iq, _ = scenario.generate()
    assert iq.size == int(1.5 * scenario.sample_rate)
    assert iq.dtype == np.complex64


# --------------------------------------------------------------------------------------
# Ground truth honesty
# --------------------------------------------------------------------------------------


def test_truth_snr_is_in_channel_not_wideband() -> None:
    """The 22 dB trap.

    Scene noise spans the full 2 MHz while an emission occupies 12.5 kHz, so channelising
    lifts every emitter by the ratio. Recording the wideband figure as ground truth would
    make every detection curve wrong by 22 dB — and produce a perfectly plausible plot.
    """
    scenario = make_scenario()
    assert scenario.processing_gain_db == pytest.approx(22.04, abs=0.01)

    _, truth = scenario.generate()
    wideband = truth[0].received_dbm - scenario.noise_floor_dbm
    assert truth[0].snr_db == pytest.approx(wideband + scenario.processing_gain_db, abs=0.01)


def test_truth_snr_matches_what_the_channeliser_measures() -> None:
    """The simulator's claimed SNR must survive contact with the actual filter bank."""
    scenario = make_scenario(
        noise_floor_dbm=-100.0,
        propagation=Propagation(path_loss_exponent=3.5, shadowing_sigma_db=0.0),
    )
    iq, truth = scenario.generate()

    bank = PolyphaseChannelizer(
        ChannelizerConfig(sample_rate=scenario.sample_rate, num_channels=160, decimation=80)
    )
    power = np.abs(bank.process(iq)) ** 2
    bin_index = bands.channel_bin_index(6, scenario.centre_frequency, scenario.sample_rate, 160)

    # Frames well inside the transmission, against frames well outside it.
    frames_per_second = scenario.sample_rate / 80
    signal = power[int(0.6 * frames_per_second) : int(1.2 * frames_per_second), bin_index].mean()
    noise = power[: int(0.2 * frames_per_second), bin_index].mean()
    measured_snr = 10 * np.log10(signal / noise)

    assert measured_snr == pytest.approx(truth[0].snr_db, abs=3.0)


def test_truth_records_the_frequency_error() -> None:
    scenario = make_scenario(
        emitters=[
            Emitter(
                name="drifty",
                channel=6,
                frequency_error_hz=350.0,
                transmissions=[Transmission(0.2, 1.0)],
            )
        ]
    )
    _, truth = scenario.generate()
    assert truth[0].frequency_hz == pytest.approx(bands.channel_frequency(6) + 350.0)


def test_off_grid_emitter_is_recorded_as_off_grid() -> None:
    """An emitter between channels must not be labelled with the nearest one."""
    scenario = make_scenario(
        emitters=[
            Emitter(
                name="rogue",
                frequency_hz=446_162_500.0,
                transmissions=[Transmission(0.2, 1.0)],
            )
        ]
    )
    _, truth = scenario.generate()
    assert truth[0].pmr_channel is None


def test_truth_is_ordered_by_start_time() -> None:
    scenario = make_scenario(
        emitters=[
            Emitter(name="late", channel=4, transmissions=[Transmission(1.0, 1.5)]),
            Emitter(name="early", channel=9, transmissions=[Transmission(0.1, 0.5)]),
        ]
    )
    _, truth = scenario.generate()
    assert [item.emitter for item in truth] == ["early", "late"]


# --------------------------------------------------------------------------------------
# Propagation
# --------------------------------------------------------------------------------------


def test_received_power_falls_with_distance() -> None:
    close = make_scenario(
        emitters=[
            Emitter(name="a", channel=6, distance_m=100.0, transmissions=[Transmission(0.2, 1.0)])
        ]
    ).generate()[1][0]
    far = make_scenario(
        emitters=[
            Emitter(name="a", channel=6, distance_m=1000.0, transmissions=[Transmission(0.2, 1.0)])
        ]
    ).generate()[1][0]

    # Ten times the distance at exponent 3.5 is 35 dB.
    assert close.received_dbm - far.received_dbm == pytest.approx(35.0, abs=0.5)


def test_free_space_exponent_gives_inverse_square_law() -> None:
    propagation = Propagation(path_loss_exponent=2.0)
    near = propagation.path_loss_db(446e6, 100.0)
    far = propagation.path_loss_db(446e6, 200.0)
    assert far - near == pytest.approx(6.02, abs=0.01)


def test_emitter_outside_the_captured_band_is_rejected() -> None:
    scenario = make_scenario(
        emitters=[
            Emitter(
                name="way-off", frequency_hz=450_000_000.0, transmissions=[Transmission(0.2, 1.0)]
            )
        ]
    )
    with pytest.raises(ValueError, match="outside"):
        scenario.generate()


def test_emitter_without_channel_or_frequency_is_rejected() -> None:
    with pytest.raises(ValueError, match="neither channel nor frequency"):
        Emitter(name="nowhere").centre_frequency()


# --------------------------------------------------------------------------------------
# Keying
# --------------------------------------------------------------------------------------


def test_keying_ramp_removes_the_step_at_the_start_of_a_transmission() -> None:
    """A hard gate is a phase step, and a step through an FM discriminator is a huge spike."""

    def envelope_ratio(ramp_ms: float) -> float:
        """Amplitude just after key-up, relative to steady state."""
        iq, _ = make_scenario(
            # Noise off: with it on, the sample-to-sample variation at these levels is the
            # noise floor rather than the keying, and the test measures nothing.
            noise_floor_dbm=-400.0,
            emitters=[
                Emitter(
                    name="a",
                    channel=6,
                    distance_m=100.0,
                    ramp_ms=ramp_ms,
                    transmissions=[Transmission(0.5, 1.5)],
                )
            ],
        ).generate()
        onset = int(0.5 * 2_000_000)
        just_after = float(np.abs(iq[onset + 100]))
        steady = float(np.abs(iq[onset + 400_000]))
        return just_after / steady

    # A hard gate is at full amplitude one sample in. A 5 ms raised cosine has barely begun
    # 100 samples in, which is 50 microseconds of a 5 millisecond rise.
    assert envelope_ratio(0.0) == pytest.approx(1.0, abs=0.05)
    assert envelope_ratio(5.0) < 0.01


# --------------------------------------------------------------------------------------
# Serialisation
# --------------------------------------------------------------------------------------


def test_scenario_survives_a_yaml_round_trip(tmp_path: Path) -> None:
    original = make_scenario()
    path = tmp_path / "scenario.yaml"
    original.save(path)
    loaded = Scenario.load(path)

    assert loaded.name == original.name
    assert loaded.seed == original.seed
    assert len(loaded.emitters) == len(original.emitters)
    assert loaded.emitters[0].ctcss_hz == original.emitters[0].ctcss_hz
    np.testing.assert_array_equal(loaded.generate()[0], original.generate()[0])


def test_shipped_demo_scenario_loads_and_is_well_formed() -> None:
    scenario = Scenario.load(Path("scenarios/demo.yaml"))

    assert scenario.name == "demo"
    assert len(scenario.emitters) == 5
    names = {emitter.name for emitter in scenario.emitters}
    assert names == {"alpha", "bravo", "charlie", "delta", "echo"}
    # The scenario exists to exercise distinct cases, so it must contain an off-grid emitter
    # and at least one emitter with no tone.
    assert any(e.frequency_hz is not None for e in scenario.emitters)
    assert any(e.ctcss_hz is None for e in scenario.emitters)
