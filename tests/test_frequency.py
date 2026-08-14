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
    measure_notch_centre,
    average_spectrum,
    measure_centre,
    measure_frequency_error,
    measure_notch_error,
    nearest_carrier,
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


# --------------------------------------------------------------------------------------
# The estimator that actually measured the crystal, and the bias it had to survive
# --------------------------------------------------------------------------------------

NOTCH_RATE = 20_000_000.0
NOTCH_NOMINAL = 816_000_000.0
NOTCH_LO = 818_500_000.0


def lte_like(
    shift_hz: float = 0.0,
    tilt_db: float = 0.0,
    samples: int = 1 << 21,
    lo_hz: float = NOTCH_LO,
) -> np.ndarray:
    """A flat carrier with an unused centre subcarrier, optionally shifted and tilted.

    Built in the frequency domain so the notch sits exactly where it is asked to, which is
    what makes it usable as ground truth for an estimator whose whole job is finding it. The
    notch is placed at the baseband offset a real capture would put it at -- the carrier's
    licensed centre seen from an offset-tuned receiver -- because measuring it at baseband DC
    would test a case the receiver's own leakage makes unusable in practice.
    """
    rng = np.random.default_rng(4)
    baseband = NOTCH_NOMINAL - lo_hz
    frequencies = np.fft.fftfreq(samples, 1.0 / NOTCH_RATE)
    spectrum = rng.normal(size=samples) + 1j * rng.normal(size=samples)
    spectrum[np.abs(frequencies - baseband) > 4.5e6] = 0.0

    offset = frequencies - baseband - shift_hz
    notch = 1.0 - 0.93 * np.exp(-((offset / 7.5e3) ** 2))
    # The asymmetry that biased the real measurement is local to the notch -- the
    # synchronisation and broadcast channels sit in the central megahertz and are not flat --
    # so it is modelled as a slope across the measurement window that levels off outside it,
    # not as a tilt across the whole carrier, which at these gradients would be unphysical.
    across = np.clip((frequencies - baseband) / 40e3, -1.0, 1.0)
    tilt = 10.0 ** ((tilt_db * across) / 20.0)
    return np.fft.ifft(spectrum * notch * tilt).astype(np.complex64)


def test_the_notch_is_found_where_it_was_put() -> None:
    offset = measure_notch_centre(lte_like(), NOTCH_RATE, NOTCH_LO, NOTCH_NOMINAL)

    assert abs(offset) < 60.0, f"a centred notch measured {offset:+.0f} Hz off centre"


@pytest.mark.parametrize("shift", [-800.0, -300.0, 300.0, 800.0])
def test_a_shifted_notch_is_measured_at_its_shift(shift: float) -> None:
    """The measurement has to track the offset; that is the whole quantity being reported."""
    offset = measure_notch_centre(lte_like(shift), NOTCH_RATE, NOTCH_LO, NOTCH_NOMINAL)

    assert offset == pytest.approx(shift, abs=150.0)


@pytest.mark.parametrize("tilt_db", [-5.0, -2.0, 2.0, 5.0])
def test_spectral_tilt_does_not_move_the_answer(tilt_db: float) -> None:
    """The bias that made two real carriers disagree by a part per million.

    An undetrended centroid moves 1638 Hz for one decibel of tilt across the window, which at
    816 MHz is two parts per million -- five times the quantity being measured. Detrending in
    decibels against the notch's own flanks removes it, and this pins that: several decibels
    of tilt must not move the answer by more than the measurement's own scatter.
    """
    offset = measure_notch_centre(
        lte_like(tilt_db=tilt_db), NOTCH_RATE, 818_500_000.0, NOTCH_NOMINAL
    )

    assert abs(offset) < 150.0, f"{tilt_db:+.0f} dB of tilt moved the answer {offset:+.0f} Hz"


def test_tilt_and_shift_together_still_recover_the_shift() -> None:
    """Tilt must not scale the measurement either, only fail to displace it."""
    for tilt in (-4.0, 0.0, 4.0):
        offset = measure_notch_centre(
            lte_like(500.0, tilt), NOTCH_RATE, 818_500_000.0, NOTCH_NOMINAL
        )
        assert offset == pytest.approx(500.0, abs=200.0), f"tilt {tilt:+.0f} dB gave {offset:+.0f}"


def test_a_band_with_no_carrier_centre_is_refused() -> None:
    """Noise has no notch, and returning a confident number for it is the failure to avoid.

    The first guard tried was depth, and it failed in the dangerous direction: noise reaches
    23 dB below its own fitted line while the real carriers measured 11 to 12, so a depth test
    accepts noise and rejects signal. Width separates them by a factor of forty.
    """
    rng = np.random.default_rng(9)
    noise = (rng.standard_normal(1 << 21) + 1j * rng.standard_normal(1 << 21)).astype(np.complex64)

    with pytest.raises(ValueError, match="no carrier centre"):
        measure_notch_centre(noise, NOTCH_RATE, NOTCH_LO, NOTCH_NOMINAL)


def test_a_capture_that_misses_the_carrier_is_refused() -> None:
    with pytest.raises(ValueError, match="does not span"):
        measure_notch_centre(lte_like(), NOTCH_RATE, 700_000_000.0, NOTCH_NOMINAL)


# --------------------------------------------------------------------------------------
# Combining captures: the carrier raster, the ensemble, and what it refuses
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("measured", "expected"),
    [
        (816_000_236.0, 816_000_000.0),
        (815_999_811.0, 816_000_000.0),
        (806_000_189.0, 806_000_000.0),
        (816_049_000.0, 816_000_000.0),
        (816_051_000.0, 816_100_000.0),
    ],
)
def test_a_measured_carrier_snaps_to_the_licensed_raster(measured: float, expected: float) -> None:
    assert nearest_carrier(measured) == pytest.approx(expected)


def test_the_ensemble_recovers_a_shift_every_capture_shares() -> None:
    """A real error is common to every local oscillator, so it must survive averaging."""
    captures = [(lte_like(400.0, lo_hz=lo), lo) for lo in (818.5e6, 817.0e6, 814.5e6)]
    error = measure_notch_error(captures, NOTCH_RATE, NOTCH_NOMINAL)

    assert error.offset_hz == pytest.approx(400.0, abs=150.0)
    assert error.nominal_hz == NOTCH_NOMINAL
    assert error.ppm == pytest.approx(1e6 * 400.0 / NOTCH_NOMINAL, abs=0.2)


def test_the_confidence_is_the_scatter_between_local_oscillators() -> None:
    """The quoted uncertainty must come from the captures disagreeing, not from a constant."""
    error = measure_notch_error(
        [(lte_like(400.0, lo_hz=lo), lo) for lo in (818.5e6, 817.0e6, 814.5e6)],
        NOTCH_RATE,
        NOTCH_NOMINAL,
    )
    assert 0.0 < error.confidence_hz < 200.0
    assert "3 local oscillators" in error.basis
    assert "3 local oscillators" in error.describe()


def test_one_capture_is_measurable_but_declares_no_scatter() -> None:
    """A single capture cannot disagree with itself, and must not pretend to a confidence."""
    error = measure_notch_error([(lte_like(400.0), NOTCH_LO)], NOTCH_RATE, NOTCH_NOMINAL)
    assert error.offset_hz == pytest.approx(400.0, abs=150.0)
    assert error.confidence_hz == 0.0


def test_the_strongest_bin_is_not_where_the_carrier_centre_is() -> None:
    """Why the carrier must be supplied: an OFDM block is flat, so its peak is arbitrary.

    This is the check behind the decision not to infer the carrier. If the strongest bin
    landed near the centre, inferring it would be safe; it does not, and snapping that bin to
    the raster would put the reference megahertz away from the truth.
    """
    frequencies, power = average_spectrum(lte_like(), NOTCH_RATE, NOTCH_LO, fft_size=1 << 18)
    strongest = float(frequencies[int(np.argmax(power))])
    assert abs(nearest_carrier(strongest) - NOTCH_NOMINAL) > 100_000.0


def test_a_capture_with_no_notch_is_dropped_rather_than_averaged_in() -> None:
    """One bad capture must not drag the ensemble; it is skipped and the rest still answer."""
    rng = np.random.default_rng(11)
    noise = (rng.normal(size=1 << 21) + 1j * rng.normal(size=1 << 21)).astype(np.complex64)
    captures = [
        (lte_like(400.0, lo_hz=818.5e6), 818.5e6),
        (noise, 817.0e6),
        (lte_like(400.0, lo_hz=814.5e6), 814.5e6),
    ]

    error = measure_notch_error(captures, NOTCH_RATE, NOTCH_NOMINAL)

    assert error.offset_hz == pytest.approx(400.0, abs=150.0)
    assert "2 local oscillators" in error.basis


def test_captures_that_all_fail_are_refused_rather_than_answered() -> None:
    rng = np.random.default_rng(12)
    noise = (rng.normal(size=1 << 21) + 1j * rng.normal(size=1 << 21)).astype(np.complex64)
    with pytest.raises(ValueError, match="no capture contained a measurable notch"):
        measure_notch_error([(noise, 817.0e6)], NOTCH_RATE, NOTCH_NOMINAL)


def test_no_captures_at_all_is_refused() -> None:
    with pytest.raises(ValueError, match="no captures given"):
        measure_notch_error([], NOTCH_RATE, NOTCH_NOMINAL)
