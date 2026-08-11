"""Verification of NFM demodulation and CTCSS identification.

Most of these tests run the real chain — synthesise an FM emission, demodulate it, identify
its tone — rather than feeding the tone detector a bare sinusoid. That is deliberate: the
two v0 defects this module exists to fix (a rounded Goertzel bin and a 1 % sample-rate
error) both survive a bare-sinusoid test and only show up once a real sub-audible tone has
to be picked out from under voice content.
"""

from __future__ import annotations

import numpy as np
import pytest

from esm446.core.ctcss import CTCSS_TONES_HZ, CtcssConfig, CtcssDetector, goertzel
from esm446.core.demod import MAX_DEVIATION_HZ, NfmDemodulator, discriminate

CHANNEL_RATE = 25_000.0

#: Typical CTCSS deviation is 10-20 % of the channel maximum; voice takes the rest.
CTCSS_DEVIATION_HZ = 400.0
VOICE_DEVIATION_HZ = 1_500.0


def synthesise_emission(
    duration_s: float,
    ctcss_hz: float | None,
    *,
    voice: bool = True,
    snr_db: float | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Build a realistic NFM emission: sub-audible tone plus voice-band content."""
    rng = np.random.default_rng(seed)
    n = int(duration_s * CHANNEL_RATE)
    t = np.arange(n) / CHANNEL_RATE

    deviation = np.zeros(n)
    if ctcss_hz is not None:
        deviation += CTCSS_DEVIATION_HZ * np.sin(2 * np.pi * ctcss_hz * t)
    if voice:
        # Two voice-band tones plus band-limited noise stands in for speech.
        deviation += VOICE_DEVIATION_HZ * 0.5 * np.sin(2 * np.pi * 640.0 * t)
        deviation += VOICE_DEVIATION_HZ * 0.3 * np.sin(2 * np.pi * 1750.0 * t)
        deviation += VOICE_DEVIATION_HZ * 0.2 * rng.standard_normal(n)

    phase = 2 * np.pi * np.cumsum(deviation) / CHANNEL_RATE
    iq = np.exp(1j * phase).astype(np.complex64)

    if snr_db is not None:
        noise_power = 10 ** (-snr_db / 10.0)
        noise = np.sqrt(noise_power / 2) * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
        iq = (iq + noise).astype(np.complex64)
    return iq


# --------------------------------------------------------------------------------------
# Generalised Goertzel
# --------------------------------------------------------------------------------------


def test_goertzel_measures_exactly_at_non_integer_bins() -> None:
    """114.8 Hz over a 1000-sample window sits at k = 114.8; the generalised form keeps it there."""
    rate, n = 1000.0, 1000
    t = np.arange(n) / rate
    signal = np.sin(2 * np.pi * 114.8 * t)

    exact = goertzel(signal, np.array([114.8]), rate)[0]
    assert exact == pytest.approx(1.0, rel=0.02)


def test_quantifies_the_two_v0_defects_separately() -> None:
    """Attributes the v0 identification loss to its two causes, with numbers.

    Bin rounding on its own is a mild 0.2-bin offset. The sample-rate error is the
    expensive one, and the two compound: v0 asked for 115.0 Hz in a stream it believed was
    12000 Hz but which was really 12121.2 Hz, so the 114.8 Hz tone presented at an
    apparent 113.65 Hz — over a bin away, where the window response has collapsed.

    The V&V report cites these figures, so the test measures them rather than asserting a
    vague degradation.
    """
    rate, n = 1000.0, 1000
    t = np.arange(n) / rate
    signal = np.sin(2 * np.pi * 114.8 * t)

    exact = goertzel(signal, np.array([114.8]), rate)[0]
    rounded_only = goertzel(signal, np.array([115.0]), rate)[0]
    # v0's true operating point: rounded target, and the stream 1.01x off its assumed rate.
    v0_apparent = 115.0 * (12_000.0 / (800_000.0 / 66))
    v0_combined = goertzel(signal, np.array([v0_apparent]), rate)[0]

    rounding_loss_db = -20 * np.log10(rounded_only / exact)
    combined_loss_db = -20 * np.log10(v0_combined / exact)

    assert rounding_loss_db == pytest.approx(0.58, abs=0.1), "rounding alone is a mild loss"
    assert (
        combined_loss_db > 12.0
    ), f"combined v0 error should be severe, measured {combined_loss_db:.1f} dB"


def test_goertzel_amplitude_is_calibrated() -> None:
    rate, n = 1000.0, 1000
    t = np.arange(n) / rate
    for amplitude in (0.1, 1.0, 5.0):
        measured = goertzel(amplitude * np.sin(2 * np.pi * 100.0 * t), np.array([100.0]), rate)[0]
        assert measured == pytest.approx(amplitude, rel=0.02)


def test_goertzel_separates_the_closest_tone_pair() -> None:
    """The tightest spacing in the CTCSS table has to be resolvable in one window."""
    spacings = np.diff(CTCSS_TONES_HZ)
    tightest = int(np.argmin(spacings))
    low, high = CTCSS_TONES_HZ[tightest], CTCSS_TONES_HZ[tightest + 1]

    rate, n = 1000.0, 1000
    t = np.arange(n) / rate
    signal = np.sin(2 * np.pi * low * t)
    magnitudes = goertzel(signal, np.array([low, high]), rate)
    assert magnitudes[0] > 5.0 * magnitudes[1], f"{low} Hz leaked into the adjacent {high} Hz slot"


# --------------------------------------------------------------------------------------
# Demodulation
# --------------------------------------------------------------------------------------


def test_discriminator_recovers_a_known_deviation() -> None:
    n = 25_000
    t = np.arange(n) / CHANNEL_RATE
    deviation = 1_200.0 * np.sin(2 * np.pi * 300.0 * t)
    iq = np.exp(1j * 2 * np.pi * np.cumsum(deviation) / CHANNEL_RATE).astype(np.complex64)

    recovered = discriminate(iq, CHANNEL_RATE)
    assert np.abs(recovered).max() == pytest.approx(1_200.0, rel=0.02)


def test_demodulator_reports_modulation_index() -> None:
    iq = synthesise_emission(2.0, 114.8)
    result = NfmDemodulator(CHANNEL_RATE).demodulate(iq)
    assert 0.0 < result.modulation_index < 2.0
    assert result.rms_deviation_hz < result.peak_deviation_hz
    assert result.peak_deviation_hz == pytest.approx(
        result.modulation_index * MAX_DEVIATION_HZ, rel=1e-6
    )


def test_discriminator_is_insensitive_to_amplitude() -> None:
    """FM carries nothing in amplitude; fading must not reach the demodulated output."""
    iq = synthesise_emission(1.0, 100.0, seed=2)
    faded = iq * np.linspace(1.0, 0.05, len(iq)).astype(np.complex64)
    np.testing.assert_allclose(
        discriminate(iq, CHANNEL_RATE), discriminate(faded, CHANNEL_RATE), atol=1.0
    )


# --------------------------------------------------------------------------------------
# End-to-end identification
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("tone", [67.0, 100.0, 114.8, 179.9, 250.3])
def test_identifies_every_kind_of_tone_through_the_full_chain(tone: float) -> None:
    iq = synthesise_emission(4.0, tone)
    audio = NfmDemodulator(CHANNEL_RATE).demodulate(iq).audio
    result = CtcssDetector(CHANNEL_RATE).detect(audio)

    assert result.identified
    assert result.tone_hz == pytest.approx(tone)


def test_reports_no_tone_when_none_is_present() -> None:
    iq = synthesise_emission(4.0, None)
    audio = NfmDemodulator(CHANNEL_RATE).demodulate(iq).audio
    result = CtcssDetector(CHANNEL_RATE).detect(audio)
    assert not result.identified


def test_survives_noise() -> None:
    iq = synthesise_emission(4.0, 114.8, snr_db=10.0, seed=9)
    audio = NfmDemodulator(CHANNEL_RATE).demodulate(iq).audio
    result = CtcssDetector(CHANNEL_RATE).detect(audio)
    assert result.identified
    assert result.tone_hz == pytest.approx(114.8)


def test_classification_requires_the_configured_tone() -> None:
    """A different tone is information, but it is not evidence of a friendly emitter."""
    friendly = CtcssDetector(CHANNEL_RATE).detect(
        NfmDemodulator(CHANNEL_RATE).demodulate(synthesise_emission(4.0, 114.8)).audio
    )
    other = CtcssDetector(CHANNEL_RATE).detect(
        NfmDemodulator(CHANNEL_RATE).demodulate(synthesise_emission(4.0, 141.3, seed=4)).audio
    )

    assert friendly.classify(114.8) == "FRIEND"
    assert other.identified and other.tone_hz == pytest.approx(141.3)
    assert other.classify(114.8) == "UNKNOWN"
    assert friendly.classify(None) == "UNKNOWN"


def test_short_emission_is_not_decided_on() -> None:
    """Below the minimum window count the detector must abstain, not guess."""
    iq = synthesise_emission(0.5, 114.8)
    audio = NfmDemodulator(CHANNEL_RATE).demodulate(iq).audio
    result = CtcssDetector(CHANNEL_RATE, CtcssConfig(min_windows=2)).detect(audio)
    assert not result.identified
