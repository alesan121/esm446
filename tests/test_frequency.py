"""Verification of the frequency-error measurement.

The method is checked against synthetic references whose true frequency is known exactly,
which is the only way to establish that it measures what it claims: with a real transmitter
the answer is precisely what is unknown.
"""

from __future__ import annotations

import numpy as np
import pytest

from esm446.core.frequency import (
    DVBT_OCCUPIED_HZ,
    average_spectrum,
    measure_centre,
    measure_frequency_error,
    nearest_uhf_channel,
)

#: Captured wider than the 8 MHz channel, so the guard bands either side are visible.
#: Capturing exactly the channel width leaves the edge finder nothing to stand on.
RATE = 10_000_000.0


def multiplex(centre_offset_hz: float, samples: int = 1 << 19, seed: int = 0) -> np.ndarray:
    """A synthetic OFDM-like block: flat, steep-edged, symmetric about its centre.

    Built in the frequency domain so its occupied bandwidth is exact, which is what makes it
    usable as ground truth for an estimator that works from the band edges.
    """
    rng = np.random.default_rng(seed)
    spectrum = np.zeros(samples, dtype=np.complex128)
    frequencies = np.fft.fftfreq(samples, 1.0 / RATE)
    occupied = np.abs(frequencies - centre_offset_hz) <= DVBT_OCCUPIED_HZ / 2.0
    spectrum[occupied] = rng.normal(size=occupied.sum()) + 1j * rng.normal(size=occupied.sum())
    block = np.fft.ifft(spectrum)
    noise = 1e-4 * (rng.standard_normal(samples) + 1j * rng.standard_normal(samples))
    return (block / np.abs(block).max() + noise).astype(np.complex64)


# --------------------------------------------------------------------------------------
# The raster
# --------------------------------------------------------------------------------------


def test_the_television_raster_is_the_european_one() -> None:
    """Channel N sits at 306 + 8N MHz. Channel 21 is 474 MHz."""
    assert nearest_uhf_channel(474_000_000.0) == (21, 474_000_000.0)
    assert nearest_uhf_channel(594_000_000.0) == (36, 594_000_000.0)


def test_a_frequency_between_channels_snaps_to_the_nearer() -> None:
    assert nearest_uhf_channel(595_500_000.0)[0] == 36
    assert nearest_uhf_channel(599_000_000.0)[0] == 37


def test_the_raster_does_not_run_off_the_band() -> None:
    assert nearest_uhf_channel(100_000_000.0)[0] == 21
    assert nearest_uhf_channel(3_000_000_000.0)[0] == 48


# --------------------------------------------------------------------------------------
# Finding the centre
# --------------------------------------------------------------------------------------


def test_a_centred_multiplex_is_measured_where_it_is() -> None:
    centre, half_width = measure_centre(multiplex(0.0), RATE, 594_000_000.0)

    assert centre == pytest.approx(594_000_000.0, abs=5_000.0)
    assert half_width == pytest.approx(DVBT_OCCUPIED_HZ / 2.0, rel=0.05)


@pytest.mark.parametrize("offset_hz", [-40_000.0, -5_000.0, 5_000.0, 40_000.0])
def test_an_offset_multiplex_is_measured_at_its_offset(offset_hz: float) -> None:
    """The measurement has to track the offset, which is the whole point of it."""
    centre, _ = measure_centre(multiplex(offset_hz), RATE, 594_000_000.0)

    assert centre - 594_000_000.0 == pytest.approx(offset_hz, abs=6_000.0)


def test_the_half_width_reports_the_occupied_bandwidth() -> None:
    """It is the check on the estimate: a symmetric signal read cleanly gives its own width."""
    _, half_width = measure_centre(multiplex(20_000.0), RATE, 594_000_000.0)

    assert 2 * half_width == pytest.approx(DVBT_OCCUPIED_HZ, rel=0.05)


def test_an_empty_band_is_refused_rather_than_answered() -> None:
    """A band with nothing in it has no edges, and inventing a centre for it would be worse
    than failing. This is what happens when no reference is receivable."""
    rng = np.random.default_rng(1)
    noise = (rng.standard_normal(1 << 18) + 1j * rng.standard_normal(1 << 18)).astype(np.complex64)

    with pytest.raises(ValueError, match="occupies"):
        measure_centre(noise, RATE, 594_000_000.0)


# --------------------------------------------------------------------------------------
# The error itself
# --------------------------------------------------------------------------------------


def test_a_receiver_tuning_high_reports_a_negative_offset() -> None:
    """A reference appearing below where it belongs means the receiver tuned above it."""
    error = measure_frequency_error(multiplex(-30_000.0), RATE, 594_000_000.0)

    assert error.offset_hz < 0
    assert error.ppm == pytest.approx(1e6 * error.offset_hz / 594_000_000.0, rel=1e-9)


def test_the_error_transfers_to_the_operating_frequency() -> None:
    """Parts per million is the figure that carries; hertz at 594 MHz does not."""
    error = measure_frequency_error(multiplex(59_400.0), RATE, 594_000_000.0)

    assert error.ppm == pytest.approx(100.0, rel=0.2)
    assert error.error_at(446_093_750.0) == pytest.approx(44_609.0, rel=0.2)


def test_the_reference_identifies_itself_from_the_raster() -> None:
    """Capturing the wrong channel by accident should be visible, not silent."""
    error = measure_frequency_error(multiplex(0.0), RATE, 594_000_000.0)

    assert error.nominal_hz == 594_000_000.0


def test_a_known_reference_can_be_given_instead_of_the_raster() -> None:
    """The raster only covers television. Any other reference is passed in."""
    error = measure_frequency_error(multiplex(0.0), RATE, 594_000_000.0, nominal_hz=594_010_000.0)

    assert error.offset_hz == pytest.approx(-10_000.0, abs=6_000.0)


def test_the_description_states_what_it_means_at_the_operating_frequency() -> None:
    text = measure_frequency_error(multiplex(20_000.0), RATE, 594_000_000.0).describe()

    assert "ppm" in text
    assert "446.09375" in text


def test_the_spectrum_average_needs_a_whole_transform() -> None:
    with pytest.raises(ValueError, match="at least"):
        average_spectrum(np.zeros(100, dtype=np.complex64), RATE, 594_000_000.0)
