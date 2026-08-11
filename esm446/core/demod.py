"""Narrowband FM demodulation of a single channeliser output.

The discriminator is the same one v0 used — the argument of the product of each sample with
the conjugate of its predecessor, which is the phase increment and therefore proportional
to instantaneous frequency. It is the standard quadrature discriminator and it was correct.

What is added here is the scaling and the filtering around it. v0 emitted raw phase
increments in radians and left ``ffmpeg`` to apply a fixed +20 dB of gain, which meant the
output level depended on the deviation of whoever happened to be transmitting. Converting
the phase increment to Hz of deviation, then normalising by the deviation the standard
actually permits, gives an output whose full scale means something: 1.0 is a fully deviated
signal under ETSI EN 300 296.

That matters downstream. The CTCSS detector's contrast metric is a ratio and so survives
arbitrary gain, but the emission metadata records peak deviation, and deviation is a real
emitter discriminant — it is one of the few parameters that distinguishes two radios on the
same channel with the same tone.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal as dsp

#: Maximum frequency deviation permitted for 12.5 kHz channels (Hz), ETSI EN 300 296.
MAX_DEVIATION_HZ = 2_500.0

#: Audio passband of a PMR446 emission (Hz).
AUDIO_LOW_HZ = 250.0
AUDIO_HIGH_HZ = 3_000.0


def discriminate(iq: np.ndarray, sample_rate: float) -> np.ndarray:
    """Quadrature FM discriminator, returning instantaneous deviation in Hz.

    Args:
        iq: Complex baseband samples of one channel.
        sample_rate: Sample rate of ``iq`` in Hz.

    Returns:
        Instantaneous frequency deviation in Hz, one sample shorter than the input.
    """
    if len(iq) < 2:
        return np.zeros(0, dtype=np.float32)
    # Limiting first: FM carries no information in amplitude, and normalising removes the
    # amplitude modulation that fading and AGC would otherwise leak into the phase estimate.
    limited = iq / (np.abs(iq) + 1e-12)
    phase_increment = np.angle(limited[1:] * np.conj(limited[:-1]))
    return (phase_increment * sample_rate / (2.0 * np.pi)).astype(np.float32)


@dataclass
class DemodResult:
    """Demodulated audio and the emission parameters measured while demodulating."""

    audio: np.ndarray
    sample_rate: float
    peak_deviation_hz: float
    rms_deviation_hz: float

    @property
    def modulation_index(self) -> float:
        """Peak deviation as a fraction of the ETSI limit. Above 1.0 is over-deviation."""
        return self.peak_deviation_hz / MAX_DEVIATION_HZ


class NfmDemodulator:
    """Narrowband FM demodulator for one channeliser output.

    Produces two things from the same discriminator output: audio filtered to the voice
    band for listening and encoding, and the *unfiltered* deviation statistics used as
    emitter metadata. Keeping them separate matters — the sub-audible CTCSS tone sits below
    the voice passband, so filtering for audio would destroy the very component the
    identification stage needs.
    """

    def __init__(self, sample_rate: float) -> None:
        self.sample_rate = sample_rate
        nyquist = sample_rate / 2.0
        if AUDIO_HIGH_HZ >= nyquist:
            raise ValueError(
                f"sample_rate {sample_rate} Hz is too low for a {AUDIO_HIGH_HZ} Hz audio passband"
            )
        self._voice_filter = dsp.firwin(
            127, [AUDIO_LOW_HZ, AUDIO_HIGH_HZ], fs=sample_rate, pass_zero=False
        ).astype(np.float64)

    def demodulate(self, iq: np.ndarray) -> DemodResult:
        """Demodulate one channel's IQ.

        Returns the full-bandwidth discriminator output as ``audio`` — not voice-filtered —
        so that the CTCSS stage still sees the sub-audible tone. Use `voice_audio` for
        anything intended to be listened to.
        """
        deviation = discriminate(iq, self.sample_rate)
        if deviation.size == 0:
            return DemodResult(deviation, self.sample_rate, 0.0, 0.0)
        return DemodResult(
            audio=deviation / MAX_DEVIATION_HZ,
            sample_rate=self.sample_rate,
            peak_deviation_hz=float(np.abs(deviation).max()),
            rms_deviation_hz=float(np.sqrt(np.mean(deviation**2))),
        )

    def voice_audio(self, result: DemodResult) -> np.ndarray:
        """Band-limit demodulated audio to the voice passband.

        Only used in the validation modes described in ``docs/06_legal_ethics.md``; the
        node's normal operating mode never calls it.
        """
        return dsp.lfilter(self._voice_filter, 1.0, result.audio).astype(np.float32)
