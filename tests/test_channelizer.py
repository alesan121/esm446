"""Verification of the polyphase analysis filter bank.

These tests are the difference between "implemented a PFB" and "verified a PFB". Each one
pins down a property that, if wrong, produces output that still *looks* plausible:
magnitudes can be right while phase is wrong, a bank can be correct on one block and drift
across block boundaries, and channels can be half a bin off the band plan without anything
obviously failing.
"""

from __future__ import annotations

import numpy as np
import pytest

from esm446.core import bands
from esm446.core.channelizer import ChannelizerConfig, PolyphaseChannelizer

SAMPLE_RATE = 2_000_000.0
NUM_CHANNELS = 160
DECIMATION = 80


@pytest.fixture
def config() -> ChannelizerConfig:
    return ChannelizerConfig(
        sample_rate=SAMPLE_RATE,
        num_channels=NUM_CHANNELS,
        decimation=DECIMATION,
    )


@pytest.fixture
def channelizer(config: ChannelizerConfig) -> PolyphaseChannelizer:
    return PolyphaseChannelizer(config)


def make_tone(frequency_hz: float, num_samples: int, amplitude: float = 1.0) -> np.ndarray:
    """Unit-amplitude complex exponential at a baseband offset from the receiver centre."""
    t = np.arange(num_samples) / SAMPLE_RATE
    return (amplitude * np.exp(2j * np.pi * frequency_hz * t)).astype(np.complex64)


# --------------------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------------------


def test_config_geometry(config: ChannelizerConfig) -> None:
    assert config.channel_spacing == bands.CHANNEL_SPACING_HZ
    assert config.channel_rate == 25_000.0
    assert config.oversampling == 2.0


def test_decimation_must_divide_channel_count() -> None:
    with pytest.raises(ValueError, match="divide"):
        ChannelizerConfig(sample_rate=SAMPLE_RATE, num_channels=160, decimation=7)


def test_prototype_has_unit_dc_gain(channelizer: PolyphaseChannelizer) -> None:
    """Unit DC gain is what makes bin magnitude directly interpretable as signal amplitude."""
    assert channelizer.prototype.sum() == pytest.approx(1.0, abs=1e-6)


# --------------------------------------------------------------------------------------
# Band plan alignment
# --------------------------------------------------------------------------------------


def test_every_pmr_channel_lands_exactly_on_a_bin() -> None:
    """Grid alignment: every channel must land exactly on a bin, and none on bin 0."""
    seen = {}
    for channel in range(1, bands.CHANNEL_COUNT + 1):
        index = bands.channel_bin_index(channel, bands.DEFAULT_CENTRE_HZ, SAMPLE_RATE, NUM_CHANNELS)
        seen[channel] = index
    assert len(set(seen.values())) == bands.CHANNEL_COUNT, "channels collided onto one bin"
    # Bin 0 is DC, where the receiver's own local-oscillator leakage lands. No channel may
    # occupy it, which is why the centre frequency is offset-tuned out of the allocation.
    assert 0 not in seen.values(), "a channel sits on the DC bin, where the LO spur lives"


def test_the_dc_bin_is_not_a_pmr_channel() -> None:
    """Offset tuning, checked where it matters: at bin 0.

    A direct-conversion receiver puts a spur at its own centre frequency -- measured at
    +31 dB above the noise floor on a HackRF One. If that frequency is a nominal channel,
    the node reports a phantom emitter on it forever.
    """
    frequencies = bands.bin_frequencies(bands.DEFAULT_CENTRE_HZ, SAMPLE_RATE, NUM_CHANNELS)
    assert bands.channel_at(frequencies[0]) is None


def test_every_iq_image_falls_outside_the_allocation() -> None:
    """IQ imbalance mirrors each signal about DC; the mirrors must not land on channels.

    With the centre inside the allocation they did: channel 9 mirrored onto channel 7.
    """
    for channel in range(1, bands.CHANNEL_COUNT + 1):
        image = bands.image_frequency(bands.channel_frequency(channel), bands.DEFAULT_CENTRE_HZ)
        assert (
            bands.channel_at(image) is None
        ), f"channel {channel} mirrors onto channel {bands.channel_at(image)}"


def test_a_channel_is_refused_as_a_centre_frequency() -> None:
    with pytest.raises(ValueError, match="phantom emitter"):
        bands.assert_centre_is_usable(bands.channel_frequency(8))


def test_the_default_centre_is_accepted() -> None:
    bands.assert_centre_is_usable(bands.DEFAULT_CENTRE_HZ)


def test_band_midpoint_is_rejected_as_centre_frequency() -> None:
    """446.1 MHz is half a bin off the grid; the configuration must fail loudly."""
    with pytest.raises(ValueError, match="not an integer"):
        bands.channel_bin_index(1, 446_100_000, SAMPLE_RATE, NUM_CHANNELS)


def test_bin_frequency_matches_band_plan() -> None:
    frequencies = bands.bin_frequencies(bands.DEFAULT_CENTRE_HZ, SAMPLE_RATE, NUM_CHANNELS)
    for channel in range(1, bands.CHANNEL_COUNT + 1):
        index = bands.channel_bin_index(channel, bands.DEFAULT_CENTRE_HZ, SAMPLE_RATE, NUM_CHANNELS)
        assert frequencies[index] == pytest.approx(bands.channel_frequency(channel))


# --------------------------------------------------------------------------------------
# Core filter bank behaviour
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("bin_index", [0, 1, 8, 79, 81, 153, 159])
def test_tone_appears_in_its_own_bin_at_unit_magnitude(
    channelizer: PolyphaseChannelizer, bin_index: int
) -> None:
    offset = np.fft.fftfreq(NUM_CHANNELS, d=1.0 / SAMPLE_RATE)[bin_index]
    spectra = channelizer.process(make_tone(offset, 200_000))
    # Discard frames still filling the filter window.
    settled = spectra[20:]
    magnitudes = np.abs(settled).mean(axis=0)

    assert magnitudes[bin_index] == pytest.approx(1.0, rel=0.02)
    others = np.delete(magnitudes, bin_index)
    assert others.max() < 1e-3, f"leakage {20 * np.log10(others.max()):.1f} dBc is too high"


def test_tone_phase_is_constant_across_frames(channelizer: PolyphaseChannelizer) -> None:
    """Catches a missing or wrong commutator de-rotation.

    Magnitudes are identical with and without the correction, so only phase reveals the
    error — and it is precisely the phase that NFM demodulation depends on.
    """
    bin_index = 8
    offset = np.fft.fftfreq(NUM_CHANNELS, d=1.0 / SAMPLE_RATE)[bin_index]
    spectra = channelizer.process(make_tone(offset, 200_000))
    phases = np.angle(spectra[20:, bin_index])
    unwrapped = np.unwrap(phases)
    drift = np.abs(unwrapped - unwrapped[0]).max()
    assert drift < 1e-3, f"phase drifted {drift:.4f} rad; commutator de-rotation is wrong"


def test_adjacent_channel_rejection(channelizer: PolyphaseChannelizer) -> None:
    """A tone one channel away must be attenuated in the neighbouring bin.

    12.5 kHz-spaced channels share a transition band by construction, so the figure to
    check is rejection at the *adjacent channel centre*, which is where a real interferer
    sits.
    """
    victim_bin = 8
    interferer_offset = np.fft.fftfreq(NUM_CHANNELS, d=1.0 / SAMPLE_RATE)[victim_bin + 1]
    spectra = channelizer.process(make_tone(interferer_offset, 200_000))
    settled = spectra[20:]
    leaked = np.abs(settled[:, victim_bin]).mean()
    rejection_db = -20.0 * np.log10(leaked)
    assert rejection_db > 60.0, f"adjacent-channel rejection only {rejection_db:.1f} dB"


def test_modulated_emitter_does_not_leak_into_adjacent_bins(
    channelizer: PolyphaseChannelizer,
) -> None:
    """Pins the figure that drove the prototype cutoff back to the channel edge.

    A pure tone at an adjacent bin centre sits exactly at the stopband edge and is rejected
    well under almost any cutoff, so it hides the defect. A *modulated* emitter has skirts
    that reach into the neighbour's transition band, and that is where a badly placed
    cutoff shows up.

    With the -6 dB point at the midpoint between the channel edge and the alias limit, this
    measured 49.5 dB and a single emitter produced spurious detections in both neighbouring
    bins. With the cutoff at the channel edge it is over 90 dB.
    """
    bin_index = 8
    offset = np.fft.fftfreq(NUM_CHANNELS, d=1.0 / SAMPLE_RATE)[bin_index]
    num_samples = 400_000
    t = np.arange(num_samples) / SAMPLE_RATE

    # Narrowband FM, 750 Hz deviation on a 700 Hz tone: a realistic PMR446 emission.
    deviation = 750.0 * np.sin(2 * np.pi * 700.0 * t)
    phase = 2 * np.pi * np.cumsum(deviation) / SAMPLE_RATE
    signal = np.exp(1j * (2 * np.pi * offset * t + phase)).astype(np.complex64)

    power = np.abs(channelizer.process(signal)[50:]) ** 2
    wanted = power[:, bin_index].mean()
    neighbours = max(power[:, bin_index - 1].mean(), power[:, bin_index + 1].mean())
    rejection_db = 10.0 * np.log10(wanted / neighbours)

    assert rejection_db > 85.0, (
        f"modulated adjacent-channel rejection only {rejection_db:.1f} dB; "
        f"a strong emitter will produce spurious detections either side of it"
    )


def test_streaming_matches_single_block(config: ChannelizerConfig) -> None:
    """Filter state must carry across `process` calls, or every block boundary rings."""
    signal = (
        make_tone(37_500.0, 120_000)
        + 0.3 * make_tone(-125_000.0, 120_000)
        + 0.01 * (np.random.default_rng(0).standard_normal(120_000)).astype(np.complex64)
    )

    whole = PolyphaseChannelizer(config).process(signal)

    streamed_bank = PolyphaseChannelizer(config)
    chunks = [streamed_bank.process(part) for part in np.array_split(signal, 17)]
    streamed = np.concatenate([c for c in chunks if c.size])

    assert streamed.shape == whole.shape
    np.testing.assert_allclose(streamed, whole, atol=1e-5)


def test_short_block_is_buffered_not_dropped(channelizer: PolyphaseChannelizer) -> None:
    """A block shorter than the filter window yields no frames but loses no samples."""
    tiny = make_tone(0.0, 100)
    assert channelizer.process(tiny).shape == (0, NUM_CHANNELS)
    rest = channelizer.process(make_tone(0.0, 100_000))
    assert rest.shape[0] > 0


def test_bin_power_tracks_input_power(channelizer: PolyphaseChannelizer) -> None:
    """Power scaling must be linear, or the dBFS-to-dBm calibration has no meaning."""
    offset = np.fft.fftfreq(NUM_CHANNELS, d=1.0 / SAMPLE_RATE)[8]
    for amplitude in (1.0, 0.1, 0.01):
        channelizer.reset()
        spectra = channelizer.process(make_tone(offset, 100_000, amplitude=amplitude))
        measured = np.abs(spectra[20:, 8]).mean()
        assert measured == pytest.approx(amplitude, rel=0.02)


def test_two_emitters_are_separated(channelizer: PolyphaseChannelizer) -> None:
    """The end-to-end property that matters: simultaneous emitters land in distinct bins."""
    frequencies = np.fft.fftfreq(NUM_CHANNELS, d=1.0 / SAMPLE_RATE)
    signal = make_tone(frequencies[8], 200_000) + 0.25 * make_tone(frequencies[153], 200_000)
    spectra = channelizer.process(signal)
    magnitudes = np.abs(spectra[20:]).mean(axis=0)

    assert magnitudes[8] == pytest.approx(1.0, rel=0.02)
    assert magnitudes[153] == pytest.approx(0.25, rel=0.02)
    quiet = np.delete(magnitudes, [8, 153])
    assert quiet.max() < 1e-3


# --------------------------------------------------------------------------------------
# Sensitivity against offset from a bin centre
# --------------------------------------------------------------------------------------


def bin_response_db(offset_bins: float) -> tuple[float, float]:
    """Peak-bin and summed-pair response to a tone offset from a bin centre.

    Args:
        offset_bins: Offset from the bin centre, in bins.

    Returns:
        ``(peak_bin_db, pair_sum_db)`` relative to a tone exactly on centre.
    """
    config = ChannelizerConfig(sample_rate=2_000_000.0, num_channels=160, decimation=80)
    samples = 1 << 17
    t = np.arange(samples) / config.sample_rate

    def bins(offset: float) -> np.ndarray:
        frequency = (20 + offset) * config.channel_spacing
        tone = np.exp(2j * np.pi * frequency * t).astype(np.complex64)
        spectra = PolyphaseChannelizer(config).process(tone)
        return (np.abs(spectra[spectra.shape[0] // 4 :]) ** 2).mean(axis=0)

    reference = bins(0.0).max()
    power = bins(offset_bins)
    ordered = np.sort(power)[::-1]
    return (
        float(10 * np.log10(ordered[0] / reference)),
        float(10 * np.log10((ordered[0] + ordered[1]) / reference)),
    )


def test_the_response_is_flat_across_most_of_a_bin() -> None:
    """A sharp prototype filter buys a flat top, and that is worth stating as a figure.

    This is not the gentle sag of a windowed FFT's scalloping loss. Out to 0.3 bins the
    response does not move at the hundredth of a decibel.
    """
    for offset in (0.0, 0.1, 0.2, 0.3):
        peak, _ = bin_response_db(offset)
        assert abs(peak) < 0.05, f"{offset} bins off centre cost {peak:.3f} dB"


def test_the_worst_case_is_six_decibels_at_the_bin_edge() -> None:
    """Halfway between two bins is the worst place an emitter can sit, and it costs 6 dB.

    That is the -6 dB point the prototype puts at the channel edge: this loss and the
    adjacent-channel rejection are the same number seen from either side, so it cannot be
    reduced without giving rejection back.
    """
    peak, _ = bin_response_db(0.5)

    assert peak == pytest.approx(-6.02, abs=0.15)


def test_the_loss_appears_only_in_the_last_fifth_of_the_bin() -> None:
    """Where the ripple lives decides whether it matters, so it is measured, not assumed."""
    assert bin_response_db(0.40)[0] == pytest.approx(-0.79, abs=0.15)
    assert bin_response_db(0.45)[0] == pytest.approx(-2.53, abs=0.20)


def test_summing_the_pair_halves_the_loss_in_decibels() -> None:
    """And this is the measurement that shows why it buys less than it looks like it should.

    The split energy comes back -- 6.02 dB becomes 3.01 dB -- but the noise in the second bin
    comes with it, so the ratio of one to the other is unchanged. What the sum actually buys
    is lower variance in the test statistic; see `esm446.core.detector`.
    """
    peak, pair = bin_response_db(0.5)

    assert peak == pytest.approx(-6.02, abs=0.15)
    assert pair == pytest.approx(-3.01, abs=0.15)
