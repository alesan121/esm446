"""Measure the receiver's frequency error against a broadcast transmitter.

Every frequency this system reports inherits the error of the HackRF's crystal, and nothing
in the project had ever established what that error is. Reported frequencies were consistent
between captures to within tens of hertz, which says the oscillator is stable and says
nothing at all about whether it is *right*: a crystal several parts per million out is stable
and wrong, and at 446 MHz a few parts per million is a few hundred hertz — comparable to the
whole spread the emitter-grouping tolerance is built around.

Why an LTE carrier's unused centre subcarrier is the reference
--------------------------------------------------------------
The measurement that succeeded uses neither of the methods this module was first written
around, and the route to it is worth recording because every step eliminated something.

An LTE downlink does not transmit its DC subcarrier. That leaves a notch about 15 kHz wide at
**exactly** the carrier centre, and the carrier centre is on a 100 kHz raster set by the
operator's licence. The notch is a *local* feature, so unlike a band edge it cannot be pulled
about by a neighbouring carrier -- which matters, because band 20 carries two adjacent
10 MHz blocks whose skirts overlap where a band edge would be looked for.

Base stations hold frequency to 0.05 ppm, forty times tighter than the error being measured,
so the notch is a reference and not a guess.

**The estimator has to be immune to spectral tilt, and the obvious one is not.** Taking the
centroid of the power deficit across the notch was measured against a synthetic notch under
known tilt: one decibel of slope across a 40 kHz window moves the answer by 1638 Hz, which at
816 MHz is two parts per million -- five times the quantity being measured. Two real carriers
measured that way disagreed by 1.0 ppm against a combined uncertainty of 0.29, which is how
the bias was found.

Removing a straight line fitted to the notch's flanks, in decibels, before taking the centroid
reduces the tilt sensitivity from 4460 Hz to 5 Hz across the same range. The two carriers then
agree to 0.002 ppm against a combined uncertainty of 0.042.

**Average the whole capture, not a slice of it.** A single capture's notch centre carries a
few hundred hertz of random error -- measured at -452, -249 and -173 Hz on one capture from a
fifth, four fifths and the whole of its length. Averaging too little leaves enough of that in
the ensemble mean to look like a disagreement between carriers, which is exactly how an
earlier version of this measurement invented a systematic that was not there. The scatter
falls as the square root of the samples averaged, as it should.

Why the other candidates were eliminated
----------------------------------------
The measurement needs a transmitter whose frequency is known better than the receiver's. The
options available without buying anything are poor except this one:

- A GSM carrier's FCCH burst is a pure tone at exactly Rb/4 = 67 708.33 Hz above the carrier,
  and a tone is the ideal thing to measure: a phase-slope estimator recovers it exactly in the
  absence of noise, and to 0.38 Hz when fifty bursts are averaged at 20 dB. It was abandoned
  because the estimator collapses below about 10 dB of burst SNR -- phase unwrapping slips
  cycles, and at 5 dB the error is 1.4 kHz -- and no GSM carrier receivable here reached that.
- An FM broadcast carrier is frequency-modulated, so there is no discrete carrier to measure;
  only its long-term average is the nominal frequency, and programme material is not
  symmetric enough to average out to the accuracy needed.
- The project's own handsets are crystals of unknown error. Calibrating one uncalibrated
  oscillator against another measures nothing.
- A terrestrial television multiplex is locked to a common reference across its
  single-frequency network, and its OFDM block is flat-topped and symmetric enough that a
  centre can be found from its two band edges alone. This is what :func:`measure_centre` and
  :func:`measure_frequency_error` implement, and it is kept because it needs no cellular
  coverage. It is not what produced the published figure: no multiplex was receivable at
  adequate strength on the antenna available here, and the width guard correctly refused to
  measure what was captured rather than returning a number from noise.

Which function to use
---------------------
:func:`measure_notch_error` is the one to reach for, and the one the CLI defaults to.
:func:`measure_frequency_error` remains for the television method where cellular coverage is
absent.

What this does not do
---------------------
It measures the receiver against one transmitter. If that transmitter is off frequency the
error is attributed to the receiver, and nothing here can tell the difference. The mitigation
is that a single-frequency network cannot tolerate that error and remain a network, which is
an argument about the transmitter's engineering rather than a measurement of it.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
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

#: LTE carriers are placed on a 100 kHz raster, so a measured centre can be snapped to its
#: licensed value without knowing the band or the operator. The receiver's error is around a
#: tenth of a kilohertz, a thousand times finer than the step, so the snap is unambiguous.
LTE_RASTER_HZ = 100_000.0


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
        confidence_hz: How far the estimate can be out, by whatever check the method
            supports. What that check is differs between methods, so ``basis`` names it
            rather than leaving the reader to assume.
        basis: What ``confidence_hz`` was derived from, for the report.
    """

    measured_hz: float
    nominal_hz: float
    offset_hz: float
    ppm: float
    confidence_hz: float
    basis: str = "edge agreement"

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
            f"  {self.basis} +/-{self.confidence_hz:.0f} Hz\n"
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


def measure_notch_centre(
    iq: np.ndarray,
    sample_rate: float,
    centre_hz: float,
    nominal_hz: float,
    half_window_hz: float = 20_000.0,
    minimum_width_hz: float = 5_000.0,
) -> float:
    """Locate an LTE carrier's unused centre subcarrier, in hertz.

    The notch sits at exactly the carrier frequency and is a local feature, so a neighbouring
    carrier cannot move it the way it moves a band edge.

    A straight line fitted to the notch's flanks is removed in decibels before the centroid is
    taken. That step is not tidiness: measured against a synthetic notch, one decibel of tilt
    across the window moves an undetrended centroid by 1638 Hz, and detrending reduces the same
    case to 5 Hz. Without it, two real carriers disagreed by 1.0 ppm.

    Args:
        iq: Complex baseband containing the carrier.
        sample_rate: Its sample rate.
        centre_hz: Where the receiver was tuned. Must not equal ``nominal_hz``, or the
            receiver's own local-oscillator leakage lands on the notch being measured.
        nominal_hz: The carrier's licensed centre, on the 100 kHz raster.
        half_window_hz: Half-width of the window fitted. Twenty kilohertz is wide enough for
            the flanks and narrow enough that the receiver's own response is straight across
            it.
        minimum_width_hz: How wide the depression must be to count as a carrier centre.

            Width rather than depth, and the reason is the opposite of the intuition: on real
            captures the notch is 11 to 12 dB deep, while pure noise reaches 23 to 24 dB
            below its own fitted line, because a single noise bin can be very low while a
            notch is a smooth broad feature. Depth therefore fails as a test and fails in the
            dangerous direction -- it accepts noise and would reject the signal.

            Measured, the contiguous run more than 3 dB below the fitted line is 13.6 to
            14.6 kHz on the two real carriers, matching LTE's 15 kHz subcarrier, and 0.31 kHz
            on noise. Five kilohertz sits an order of magnitude from each.

    Returns:
        Measured notch position minus ``nominal_hz``, in hertz.

    Raises:
        ValueError: If no notch of the required depth is present, which is what an empty band
            or a mistuned capture looks like.
    """
    frequencies, power = average_spectrum(iq, sample_rate, centre_hz, fft_size=1 << 18)
    selected = np.abs(frequencies - nominal_hz) < half_window_hz
    if selected.sum() < 50:
        raise ValueError("the capture does not span the carrier's centre")

    offsets = frequencies[selected] - nominal_hz
    with np.errstate(divide="ignore"):
        decibels = 10.0 * np.log10(np.maximum(power[selected], 1e-30))
    decibels = np.convolve(decibels, np.ones(9) / 9.0, mode="same")

    # The line is fitted to the flanks only: including the notch would let it tilt the fit.
    flanks = np.abs(offsets) > half_window_hz / 2.0
    flattened = decibels - np.polyval(np.polyfit(offsets[flanks], decibels[flanks], 1), offsets)

    resolution = sample_rate / (1 << 18)
    below = flattened < -3.0
    longest = run = 0
    for is_below in below:
        run = run + 1 if is_below else 0
        longest = max(longest, run)
    width = longest * resolution
    if width < minimum_width_hz:
        raise ValueError(
            f"no carrier centre here: the widest depression is {width / 1e3:.1f} kHz across "
            f"and {minimum_width_hz / 1e3:.0f} kHz is required. Noise produces deep but "
            f"narrow dips; a carrier's unused centre subcarrier is broad and smooth."
        )

    deficit = np.maximum(-flattened, 0.0)
    total = deficit.sum()
    if total <= 0.0:
        raise ValueError("nothing dips below the fitted line; there is no notch here")
    return float((offsets * deficit).sum() / total)


def nearest_carrier(frequency_hz: float, raster_hz: float = LTE_RASTER_HZ) -> float:
    """Snap a measured frequency onto the licensed channel raster.

    A cellular carrier's centre is not arbitrary: it sits on a raster fixed by the standard,
    100 kHz for LTE. Snapping to it recovers the nominal frequency without needing to know
    which operator or band the carrier belongs to, and it is safe here because the receiver's
    error is three orders of magnitude smaller than the raster step.

    Args:
        frequency_hz: The measured carrier centre.
        raster_hz: Raster step. Defaults to LTE's 100 kHz.

    Returns:
        The nominal carrier centre.
    """
    return round(frequency_hz / raster_hz) * raster_hz


def measure_notch_error(
    captures: Iterable[tuple[np.ndarray, float]],
    sample_rate: float,
    nominal_hz: float,
) -> FrequencyError:
    """Measure the receiver's frequency error against a cellular carrier's unused centre.

    This is the method that produced the figure in ``docs/04_link_budget.md``, and the one to
    reach for. It needs no equipment beyond an antenna: a base station's frequency is
    disciplined to 0.05 ppm, two orders of magnitude better than the receiver being measured,
    so it serves as a reference wherever there is coverage.

    Every capture should be taken at a *different* local oscillator. That is not a refinement,
    it is the control: the receiver's own artefacts sit at fixed baseband offsets and would
    otherwise be indistinguishable from the carrier, whereas a real emission stays put in
    absolute frequency while the local oscillator moves under it. Retuning between captures
    both proves the signal is external and averages down whatever depends on where in the
    passband the notch happened to land.

    Args:
        captures: Pairs of complex baseband and the frequency each was tuned to. Consumed
            lazily and one at a time, so a generator that opens each capture as it is reached
            keeps only one in memory; a full set of wideband captures does not fit otherwise.
        sample_rate: Sample rate shared by the captures.
        nominal_hz: The carrier's licensed centre. This is required rather than inferred:
            picking the strongest bin and snapping it to the raster looks like it would work
            and does not, because an OFDM carrier is flat across its occupied bandwidth and
            its strongest bin falls at random within several megahertz. Take the carrier from
            a survey and pass it, or snap a value with :func:`nearest_carrier`.

    Returns:
        The measured error, with the scatter across captures as its confidence.

    Raises:
        ValueError: If no capture is given, or none contains a measurable notch.
    """
    seen = 0
    offsets: list[float] = []
    for iq, centre_hz in captures:
        seen += 1
        try:
            offsets.append(measure_notch_centre(iq, sample_rate, centre_hz, nominal_hz))
        except ValueError as problem:
            logger.warning("frequency: skipping a capture, %s", problem)

    if not seen:
        raise ValueError("no captures given")

    if not offsets:
        raise ValueError(
            "no capture contained a measurable notch. The carrier must be strong enough for "
            "its unused centre subcarrier to show against the noise"
        )

    offset = float(np.mean(offsets))
    # The scatter between local oscillators bounds what the method has not removed, and it is
    # consistently larger than the error of any single capture. Quoting the smaller number
    # would be quoting precision in place of accuracy.
    spread = float(np.std(offsets, ddof=1) / np.sqrt(len(offsets))) if len(offsets) > 1 else 0.0

    error = FrequencyError(
        measured_hz=nominal_hz + offset,
        nominal_hz=nominal_hz,
        offset_hz=offset,
        ppm=1e6 * offset / nominal_hz,
        confidence_hz=spread,
        basis=f"scatter over {len(offsets)} local oscillators",
    )
    logger.info("frequency: %+.0f Hz (%+.3f ppm)", error.offset_hz, error.ppm)
    return error


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
