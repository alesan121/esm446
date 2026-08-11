"""CTCSS sub-audible tone identification by generalised Goertzel.

What this actually is
---------------------
v0 called this "IFF" and printed ALIADO or DESCONOCIDO. It is not IFF. There is no
challenge, no response, no cryptography and no way to tell a cooperating radio from anyone
who happens to have set the same tone. What it really is: **cooperative identification by a
pre-shared sub-audible key**, which is a useful discriminator and an honest thing to call
it. Describing it as IFF in a portfolio read by radar and EW engineers invites exactly the
question you do not want. The operational ALIADO/DESCONOCIDO output is kept; only the claim
attached to it is corrected.

Two defects inherited from v0
-----------------------------
**Rounded Goertzel bin.** v0 computed ``k = round(N * f / rate)`` and used the *rounded*
``k`` in the recurrence, so asking for 114.8 Hz actually measured 115.0 Hz. On its own this
is mild — a 0.2-bin offset costs 0.58 dB — but it costs nothing to avoid: the generalised
Goertzel keeps ``k`` real-valued, changing only the initial coefficient.

**Sample rate error.** This is the expensive one. v0's channeliser produced 12121.2 Hz
while this detector assumed 12000 Hz, so the 114.8 Hz tone presented at an apparent
113.65 Hz. Combined with the rounding, the analysis sat more than a full bin off the tone,
where the window response has collapsed: over 12 dB of loss, measured in
``tests/test_ctcss.py``. Here the audio rate is passed in explicitly and the decimation
factor derived from it, so the two cannot drift apart in the first place.

Why decimate first
------------------
CTCSS tones live between 67 and 250 Hz. Running the analysis at the channel rate wastes
almost all of the work on empty spectrum, and it sets the trade-off badly: Goertzel costs
``num_tones * N`` operations against an FFT's ``N log2 N``, so at 38 tones and N = 12000 the
FFT would actually be cheaper. Decimating to 1 kHz first changes the answer. A 1-second
window is then 1000 samples, the whole 38-tone bank costs 38k operations, and Goertzel is
comfortably the right tool — while also placing each analysis frequency exactly where we
want it, which an FFT cannot do without interpolation. Choosing the right transform is
downstream of choosing the right sample rate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal as dsp

#: The 38 standard CTCSS tones in Hz (EIA/TIA-603).
CTCSS_TONES_HZ: tuple[float, ...] = (
    67.0,
    71.9,
    74.4,
    77.0,
    79.7,
    82.5,
    85.4,
    88.5,
    91.5,
    94.8,
    97.4,
    100.0,
    103.5,
    107.2,
    110.9,
    114.8,
    118.8,
    123.0,
    127.3,
    131.8,
    136.5,
    141.3,
    146.2,
    151.4,
    156.7,
    162.2,
    167.9,
    173.8,
    179.9,
    186.2,
    192.8,
    203.5,
    210.7,
    218.1,
    225.7,
    233.6,
    241.8,
    250.3,
)

#: Sample rate the tone analysis runs at, after decimation (Hz).
ANALYSIS_RATE_HZ = 1000.0

#: Analysis window length in seconds. The closest tone spacing in the table is 3.5 Hz
#: (203.5 to 210.7 is wider, but 67.0 to 71.9 and the dense middle run tighter), so a
#: 1-second window giving 1 Hz resolution separates every pair with margin.
WINDOW_SECONDS = 1.0


def goertzel(samples: np.ndarray, frequencies: np.ndarray, sample_rate: float) -> np.ndarray:
    """Generalised Goertzel magnitude at arbitrary, non-integer-bin frequencies.

    The classic Goertzel assumes the target lands on an exact DFT bin. The generalised form
    simply keeps ``k = N * f / rate`` real instead of rounding it, which makes the analysis
    frequency exact for any target. The recurrence itself is unchanged.

    Args:
        samples: Real-valued input, shape ``(N,)``.
        frequencies: Target frequencies in Hz, shape ``(T,)``.
        sample_rate: Sample rate of ``samples`` in Hz.

    Returns:
        Magnitude at each target frequency, shape ``(T,)``.
    """
    n = len(samples)
    if n == 0:
        return np.zeros(len(frequencies))

    omega = 2.0 * np.pi * np.asarray(frequencies, dtype=np.float64) / sample_rate
    coefficient = 2.0 * np.cos(omega)

    # The recurrence is sequential in samples but independent across tones, so the tone
    # axis is what gets vectorised.
    q1 = np.zeros(len(omega))
    q2 = np.zeros(len(omega))
    for sample in samples:
        q0 = coefficient * q1 - q2 + sample
        q2 = q1
        q1 = q0

    real = q1 - q2 * np.cos(omega)
    imaginary = q2 * np.sin(omega)
    return np.hypot(real, imaginary) * (2.0 / n)


@dataclass(frozen=True)
class CtcssConfig:
    """Decision thresholds for tone identification.

    Attributes:
        contrast_threshold: How far the winning tone must stand above the median of the
            others, as a linear ratio. The median is used rather than the mean because
            with 38 candidates it is unmoved by the winner itself and by any harmonic
            relationship that lights up a second tone.
        vote_ratio: Fraction of analysis windows that must agree before a tone is declared.
            Guards against a single window catching a transient.
        min_windows: Minimum number of complete windows required to decide at all.
    """

    contrast_threshold: float = 8.0
    vote_ratio: float = 0.6
    min_windows: int = 2


@dataclass
class CtcssResult:
    """Outcome of tone identification over one captured emission."""

    tone_hz: float | None
    contrast: float
    windows_analysed: int
    windows_agreeing: int

    @property
    def identified(self) -> bool:
        """True when a CTCSS tone was positively identified."""
        return self.tone_hz is not None

    def classify(self, expected_tone_hz: float | None) -> str:
        """Operational identification against the pre-shared tone for this deployment.

        Returns ``"FRIEND"`` only for a positive match with the configured tone. Everything
        else is ``"UNKNOWN"`` — including a positively identified *different* tone, which
        is information worth recording but is not evidence of anything friendly.
        """
        if expected_tone_hz is None or self.tone_hz is None:
            return "UNKNOWN"
        return "FRIEND" if abs(self.tone_hz - expected_tone_hz) < 0.5 else "UNKNOWN"


class CtcssDetector:
    """Identify the CTCSS tone present in a demodulated audio stream."""

    def __init__(self, audio_rate: float, config: CtcssConfig | None = None) -> None:
        if audio_rate < 2.0 * ANALYSIS_RATE_HZ:
            raise ValueError(
                f"audio_rate {audio_rate} Hz is too low to decimate to {ANALYSIS_RATE_HZ} Hz"
            )
        self.audio_rate = audio_rate
        self.config = config or CtcssConfig()
        self.decimation = int(round(audio_rate / ANALYSIS_RATE_HZ))
        self.analysis_rate = audio_rate / self.decimation
        self.window_samples = int(round(WINDOW_SECONDS * self.analysis_rate))
        self._tones = np.asarray(CTCSS_TONES_HZ)
        # Anti-alias ahead of decimation. 300 Hz passes every CTCSS tone with margin while
        # rejecting the voice band, which would otherwise fold straight onto the tones.
        self._decimation_filter = dsp.firwin(
            129, 300.0, fs=audio_rate, window=("kaiser", 8.0)
        ).astype(np.float64)

    def _to_analysis_rate(self, audio: np.ndarray) -> np.ndarray:
        filtered = dsp.lfilter(self._decimation_filter, 1.0, audio.astype(np.float64))
        return filtered[:: self.decimation]

    def analyse_window(self, window: np.ndarray) -> tuple[float, float]:
        """Return ``(winning_tone_hz, contrast)`` for one analysis window."""
        magnitudes = goertzel(window, self._tones, self.analysis_rate)
        winner = int(np.argmax(magnitudes))
        others = np.delete(magnitudes, winner)
        contrast = float(magnitudes[winner] / (np.median(others) + 1e-12))
        return float(self._tones[winner]), contrast

    def detect(self, audio: np.ndarray) -> CtcssResult:
        """Identify the CTCSS tone in a demodulated emission.

        Args:
            audio: Real-valued demodulated audio at ``audio_rate``.

        Returns:
            The identified tone, or ``tone_hz=None`` when no tone met the decision criteria.
        """
        baseband = self._to_analysis_rate(audio)
        num_windows = len(baseband) // self.window_samples

        if num_windows < self.config.min_windows:
            return CtcssResult(None, 0.0, num_windows, 0)

        votes: dict[float, int] = {}
        contrasts: dict[float, list[float]] = {}
        for index in range(num_windows):
            start = index * self.window_samples
            tone, contrast = self.analyse_window(baseband[start : start + self.window_samples])
            if contrast >= self.config.contrast_threshold:
                votes[tone] = votes.get(tone, 0) + 1
                contrasts.setdefault(tone, []).append(contrast)

        if not votes:
            return CtcssResult(None, 0.0, num_windows, 0)

        best_tone = max(votes, key=lambda t: votes[t])
        agreeing = votes[best_tone]
        if agreeing / num_windows < self.config.vote_ratio:
            return CtcssResult(None, 0.0, num_windows, agreeing)

        return CtcssResult(
            tone_hz=best_tone,
            contrast=float(np.median(contrasts[best_tone])),
            windows_analysed=num_windows,
            windows_agreeing=agreeing,
        )
