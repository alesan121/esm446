"""Verification of the wideband STFT survey."""

from __future__ import annotations

import numpy as np
import pytest

from esm446.core import bands
from esm446.core.survey import SpectrumSurvey, SurveyConfig

SAMPLE_RATE = 2_000_000.0


def tone(frequency_offset_hz: float, num_samples: int, amplitude: float = 1.0) -> np.ndarray:
    t = np.arange(num_samples) / SAMPLE_RATE
    return (amplitude * np.exp(2j * np.pi * frequency_offset_hz * t)).astype(np.complex64)


@pytest.fixture
def survey() -> SpectrumSurvey:
    return SpectrumSurvey(SAMPLE_RATE, bands.DEFAULT_CENTRE_HZ)


def test_resolution_and_coverage(survey: SpectrumSurvey) -> None:
    assert survey.resolution_hz == pytest.approx(1953.125)
    span = survey._frequencies[-1] - survey._frequencies[0]
    assert span == pytest.approx(SAMPLE_RATE - survey.resolution_hz)


def test_locates_a_tone_at_the_right_absolute_frequency(survey: SpectrumSurvey) -> None:
    """The survey reports absolute frequencies, so an off-grid emitter can be logged as such."""
    offset = 300_000.0
    result = survey.analyse(tone(offset, 200_000))
    peak_hz, _ = result.peak()
    assert peak_hz == pytest.approx(bands.DEFAULT_CENTRE_HZ + offset, abs=survey.resolution_hz)


def test_full_scale_tone_reads_zero_dbfs(survey: SpectrumSurvey) -> None:
    """Coherent-gain normalisation: changing the window must not rescale reported power."""
    result = survey.analyse(tone(0.0, 200_000, amplitude=1.0))
    assert result.power_db.max() == pytest.approx(0.0, abs=0.5)


def test_window_choice_does_not_change_reported_power() -> None:
    signal = tone(100_000.0, 200_000)
    peaks = []
    for window in ("hann", "blackmanharris", "hamming"):
        result = SpectrumSurvey(
            SAMPLE_RATE, bands.DEFAULT_CENTRE_HZ, SurveyConfig(window=window)
        ).analyse(signal)
        peaks.append(result.peak()[1])
    assert max(peaks) - min(peaks) < 0.5


@pytest.mark.parametrize("offset_hz", [10_000.0, 20_000.0, 50_000.0])
def test_blackmanharris_keeps_a_strong_emitter_from_burying_its_neighbours(
    offset_hz: float,
) -> None:
    """The reason the survey does not use a resolution-first window.

    A strong local emitter must not raise the apparent floor over its neighbours. The
    offsets probed here — 10 to 50 kHz — are the distances that actually occur in PMR446,
    where channels are 12.5 kHz apart and the whole allocation spans 200 kHz.

    Measured near-in advantage over Hann: 45 dB at 10 kHz, 28 dB at 20 kHz, 29 dB at
    50 kHz. Hann does overtake beyond roughly 300 kHz because its sidelobes roll off
    faster asymptotically, but by then both are below -130 dBFS, approaching the numerical
    floor of complex64 arithmetic, and nothing in this band is that far away anyway.
    """
    signal = tone(200_000.0, 200_000, amplitude=1.0)
    probe_hz = bands.DEFAULT_CENTRE_HZ + 200_000.0 + offset_hz

    levels = {}
    for window in ("hann", "blackmanharris"):
        result = SpectrumSurvey(
            SAMPLE_RATE, bands.DEFAULT_CENTRE_HZ, SurveyConfig(window=window)
        ).analyse(signal)
        index = int(np.argmin(np.abs(result.frequencies_hz - probe_hz)))
        levels[window] = result.power_db[index]

    assert levels["blackmanharris"] < levels["hann"] - 20.0, (
        f"at {offset_hz / 1e3:.0f} kHz: blackmanharris {levels['blackmanharris']:.1f} dB "
        f"vs hann {levels['hann']:.1f} dB"
    )


def test_noise_floor_is_robust_to_occupied_channels(survey: SpectrumSurvey) -> None:
    """Median, not mean: several active emitters must not drag the floor estimate upward."""
    rng = np.random.default_rng(0)
    n = 200_000
    noise = (0.001 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))).astype(np.complex64)

    quiet = survey.analyse(noise).noise_floor_db
    busy = survey.analyse(
        noise + sum(tone(f, n) for f in (-50_000.0, 0.0, 37_500.0, 100_000.0))
    ).noise_floor_db

    assert busy == pytest.approx(quiet, abs=1.0)


def test_occupancy_flags_only_the_active_bins(survey: SpectrumSurvey) -> None:
    rng = np.random.default_rng(1)
    n = 200_000
    noise = (0.001 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))).astype(np.complex64)
    result = survey.analyse(noise + tone(125_000.0, n, amplitude=0.5))

    occupied = np.flatnonzero(result.occupancy(threshold_db=20.0))
    assert occupied.size > 0
    centre = bands.DEFAULT_CENTRE_HZ + 125_000.0
    assert np.all(np.abs(result.frequencies_hz[occupied] - centre) < 20_000.0)


def test_spectrogram_shape_follows_the_hop(survey: SpectrumSurvey) -> None:
    spectrogram = survey.spectrogram(tone(0.0, 100_000))
    expected = (100_000 - survey.config.fft_size) // survey.config.hop + 1
    assert spectrogram.shape == (expected, survey.config.fft_size)


def test_block_shorter_than_the_transform_yields_nothing(survey: SpectrumSurvey) -> None:
    assert survey.spectrogram(tone(0.0, 100)).shape[0] == 0
    assert survey.analyse(tone(0.0, 100)).num_frames == 0


def test_rejects_a_non_power_of_two_transform() -> None:
    with pytest.raises(ValueError, match="power of two"):
        SurveyConfig(fft_size=1000)
