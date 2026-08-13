"""Measure the receiver's frequency error against a broadcast transmitter.

Every frequency this system reports inherits the error of the HackRF's crystal, and nothing
in the project had ever established what that error is. Reported frequencies were consistent
between captures to within tens of hertz, which says the oscillator is stable and says
nothing at all about whether it is *right*: a crystal several parts per million out is stable
and wrong, and at 446 MHz a few parts per million is a few hundred hertz — comparable to the
whole spread the emitter-grouping tolerance is built around.

Why a DVB-T multiplex is the reference
--------------------------------------
The measurement needs a transmitter whose frequency is known better than the receiver's. The
options available without buying anything are poor except this one:

- An FM broadcast carrier is frequency-modulated, so there is no discrete carrier to measure;
  only its long-term average is the nominal frequency, and programme material is not
  symmetric enough to average out to the accuracy needed.
- The project's own handsets are crystals of unknown error. Calibrating one uncalibrated
  oscillator against another measures nothing.
- A terrestrial television multiplex is transmitted from a single-frequency network, which
  can only work if every transmitter in it is locked to a common reference — in practice GPS.
  Its spectrum is an OFDM block: flat-topped, steep-edged and symmetric by construction, so
  its centre can be found from the shape alone without demodulating anything.

The centre is estimated from the two band edges rather than from a peak, because an OFDM
block has no peak. Averaging the spectrum over a second reduces the fading that would
otherwise move each edge independently.

What this does not do
---------------------
It measures the receiver against one transmitter. If that transmitter is off frequency the
error is attributed to the receiver, and nothing here can tell the difference. The mitigation
is that a single-frequency network cannot tolerate that error and remain a network, which is
an argument about the transmitter's engineering rather than a measurement of it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

#: European terrestrial television raster: channel N is centred at 306 + 8N MHz, N = 21..48.
UHF_CHANNEL_BASE_HZ = 306_000_000
UHF_CHANNEL_STEP_HZ = 8_000_000
UHF_CHANNEL_RANGE = (21, 48)

#: Occupied bandwidth of a DVB-T signal inside its 8 MHz channel.
DVBT_OCCUPIED_HZ = 7_610_000

#: How far the measured occupied bandwidth may differ from the reference's own before the
#: measurement is refused.
#:
#: This is the guard that matters, and it is a statement about shape rather than about level.
#: Levels do not separate the two cases: a multiplex built from noise inside its own band has
#: the same spectral statistics as noise, and measured on an 8 MHz capture of an 8 MHz channel
#: the peak stands barely a decibel above the median in either case. The width does separate
#: them completely. Measured on a 10 MHz capture, a real block returns 7.607 MHz against an
#: expected 7.610; pure noise returns 9.994 MHz, which is the whole span -- the edge finder
#: walked to the ends of the capture because there were no edges to find.
#:
#: So the question asked is not "is something loud here" but "does the thing I measured have
#: the bandwidth a reference has". Fifteen per cent is far wider than the 0.04 % a real
#: reference achieves and far narrower than the 31 % noise produced.
OCCUPIED_TOLERANCE = 0.15


@dataclass(frozen=True)
class FrequencyError:
    """The receiver's frequency error, measured against one transmitter.

    Attributes:
        measured_hz: Where the reference appeared.
        nominal_hz: Where it should have been.
        offset_hz: Measured minus nominal. Positive means the receiver tunes low, so signals
            appear higher than they are.
        ppm: The same error as parts per million, which is the figure that transfers to any
            other frequency.
        confidence_hz: Half the disagreement between the two band edges, which is the
            sharpest available check on the estimate: a symmetric signal measured cleanly
            gives the same centre from either side.
    """

    measured_hz: float
    nominal_hz: float
    offset_hz: float
    ppm: float
    confidence_hz: float

    def error_at(self, frequency_hz: float) -> float:
        """The error this implies at another frequency, in hertz.

        Args:
            frequency_hz: Where the system actually operates.

        Returns:
            Expected error in hertz, signed the same way as ``offset_hz``.
        """
        return frequency_hz * self.ppm * 1e-6

    def describe(self) -> str:
        """Render the measurement, including what it means at 446 MHz."""
        return (
            f"reference at {self.nominal_hz / 1e6:.3f} MHz measured at "
            f"{self.measured_hz / 1e6:.6f} MHz\n"
            f"  offset      {self.offset_hz:+.0f} Hz  ({self.ppm:+.2f} ppm)\n"
            f"  edge agreement +/-{self.confidence_hz:.0f} Hz\n"
            f"  implied error at 446.09375 MHz: {self.error_at(446_093_750.0):+.0f} Hz"
        )


def nearest_uhf_channel(frequency_hz: float) -> tuple[int, float]:
    """Closest terrestrial television channel to a frequency.

    Args:
        frequency_hz: A measured centre.

    Returns:
        ``(channel_number, nominal_centre_hz)``.
    """
    channel = round((frequency_hz - UHF_CHANNEL_BASE_HZ) / UHF_CHANNEL_STEP_HZ)
    channel = max(UHF_CHANNEL_RANGE[0], min(UHF_CHANNEL_RANGE[1], channel))
    return channel, float(UHF_CHANNEL_BASE_HZ + channel * UHF_CHANNEL_STEP_HZ)


def average_spectrum(
    iq: np.ndarray, sample_rate: float, centre_hz: float, fft_size: int = 8192
) -> tuple[np.ndarray, np.ndarray]:
    """Power spectrum averaged over the whole capture.

    Args:
        iq: Complex baseband.
        sample_rate: Its sample rate.
        centre_hz: The frequency it was captured at.
        fft_size: Transform length. 8192 at 2 MS/s is 244 Hz per bin.

    Returns:
        ``(frequencies_hz, power)``, both fftshifted so frequency increases.

    Raises:
        ValueError: If the capture is shorter than one transform.
    """
    frames = iq.size // fft_size
    if frames < 1:
        raise ValueError(f"need at least {fft_size} samples, got {iq.size}")

    block = iq[: frames * fft_size].reshape(frames, fft_size)
    window = np.hanning(fft_size).astype(np.float32)
    spectra = np.abs(np.fft.fft(block * window, axis=1)) ** 2
    power = np.fft.fftshift(spectra.mean(axis=0))
    frequencies = np.fft.fftshift(np.fft.fftfreq(fft_size, 1.0 / sample_rate)) + centre_hz
    return frequencies, power


def _edge(frequencies: np.ndarray, power_db: np.ndarray, level_db: float, rising: bool) -> float:
    """Where the spectrum crosses a level, interpolated between bins.

    Args:
        frequencies: Bin frequencies, increasing.
        power_db: Power in dB, same length.
        level_db: The level to find.
        rising: Search the low-frequency side if true, the high side if false.

    Returns:
        The crossing frequency.

    Raises:
        ValueError: If the spectrum never crosses the level on that side.
    """
    peak = int(np.argmax(power_db))
    indices = range(peak, 0, -1) if rising else range(peak, len(power_db) - 1)

    for i in indices:
        step = -1 if rising else 1
        if power_db[i] >= level_db > power_db[i + step]:
            # Linear interpolation in dB against frequency, which is close enough over one
            # bin of a roll-off this steep.
            span = power_db[i] - power_db[i + step]
            fraction = (power_db[i] - level_db) / span if span else 0.0
            return float(frequencies[i] + fraction * (frequencies[i + step] - frequencies[i]))
    raise ValueError("the spectrum does not cross that level on this side")


def measure_centre(
    iq: np.ndarray,
    sample_rate: float,
    centre_hz: float,
    edge_below_peak_db: float = 10.0,
    expected_occupied_hz: float = DVBT_OCCUPIED_HZ,
) -> tuple[float, float]:
    """Centre of a flat-topped, symmetric signal, from its two edges.

    Args:
        iq: Complex baseband containing the reference.
        sample_rate: Its sample rate.
        centre_hz: The frequency it was captured at.
        edge_below_peak_db: How far down the roll-off to place the edges. Ten decibels is
            below the ripple of the flat top and above the noise floor.
        expected_occupied_hz: Occupied bandwidth the reference is known to have. The
            measurement is refused if what was found does not have it.

    Returns:
        ``(centre_hz, half_width_hz)``. The half width is the sharpest available check: for a
        symmetric signal it should match the reference's known occupied bandwidth.

    Raises:
        ValueError: If nothing in the capture stands far enough above the noise to be a
            reference. Measuring an empty band would produce a number, and that number would
            be the shape of the noise rather than the frequency of anything.
    """
    frequencies, power = average_spectrum(iq, sample_rate, centre_hz)
    with np.errstate(divide="ignore"):
        power_db = 10.0 * np.log10(np.maximum(power, 1e-30))

    # Smooth over a few bins: the flat top of an OFDM block is not flat frame to frame, and
    # the edge finder should follow the shape rather than one bin's realisation.
    kernel = np.ones(9) / 9.0
    smoothed = np.convolve(power_db, kernel, mode="same")

    level = smoothed.max() - edge_below_peak_db
    low = _edge(frequencies, smoothed, level, rising=True)
    high = _edge(frequencies, smoothed, level, rising=False)
    occupied = high - low

    if abs(occupied - expected_occupied_hz) > OCCUPIED_TOLERANCE * expected_occupied_hz:
        raise ValueError(
            f"no reference in this capture: what was measured occupies "
            f"{occupied / 1e6:.3f} MHz, and a reference of this kind occupies "
            f"{expected_occupied_hz / 1e6:.3f} MHz. A band with nothing in it returns the "
            f"width of the capture, because there are no edges to find. Point the receiver "
            f"at a transmitter that is actually receivable, and capture wider than the "
            f"channel so the guard bands are visible."
        )

    return 0.5 * (low + high), 0.5 * occupied


def measure_frequency_error(
    iq: np.ndarray,
    sample_rate: float,
    centre_hz: float,
    nominal_hz: float | None = None,
) -> FrequencyError:
    """Measure the receiver's frequency error against a broadcast reference.

    Args:
        iq: Complex baseband containing the reference signal.
        sample_rate: Its sample rate.
        centre_hz: The frequency the receiver was tuned to.
        nominal_hz: Where the reference should be. Defaults to the nearest television
            channel centre, which also identifies which channel was captured.

    Returns:
        The measured error.
    """
    measured, half_width = measure_centre(iq, sample_rate, centre_hz)
    if nominal_hz is None:
        channel, nominal_hz = nearest_uhf_channel(measured)
        logger.info("frequency: reference identified as television channel %d", channel)

    offset = measured - nominal_hz
    # The two edges are found independently, so how far the measured half width departs from
    # the reference's own occupied bandwidth bounds how far the centre can be out.
    confidence = abs(half_width - DVBT_OCCUPIED_HZ / 2.0)

    error = FrequencyError(
        measured_hz=measured,
        nominal_hz=nominal_hz,
        offset_hz=offset,
        ppm=1e6 * offset / nominal_hz,
        confidence_hz=confidence,
    )
    logger.info("frequency: %+.0f Hz (%+.2f ppm)", error.offset_hz, error.ppm)
    return error
