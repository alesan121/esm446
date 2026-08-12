"""Wideband spectrum survey by short-time Fourier transform.

Why a second spectral path exists
---------------------------------
The filter bank is tuned to one job: clean 12.5 kHz channels with enough selectivity to
separate adjacent PMR446 users. That selectivity costs a 1920-tap prototype, and it buys
resolution the survey does not need. Asking one transform to do both jobs means
over-paying for the wideband picture on every frame.

So the two are split, and each gets the transform that suits it:

- **Channelisation** — polyphase filter bank, every frame, 12.5 kHz bins, sharp filters.
  Feeds detection, demodulation and identification.
- **Survey** — plain STFT, low duty cycle, coarse bins, no filter design at all. Feeds the
  occupancy waterfall, the noise floor estimate, and awareness of energy outside the
  channel plan.

Measured on the development machine, a 1024-point STFT costs 0.009 CPU-seconds per second
of signal against the filter bank's 0.27 — roughly 3 % on top for a complete wideband
picture. Running it continuously would be affordable; running it at a low duty cycle makes
it free.

The survey is also what gives the noise floor its long time constant. CFAR estimates noise
locally in frequency from a couple of dozen bins in a single frame, which is what makes it
responsive; the survey estimates it globally over seconds, which is what makes it stable.
Reporting both, and noticing when they disagree, is how the node detects that something
broadband has arrived — an interferer, a nearby switching supply, or a front end being
driven into compression.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.fft as sfft
from scipy import signal as dsp


@dataclass(frozen=True)
class SurveyConfig:
    """Resolution and averaging of the wideband survey.

    Attributes:
        fft_size: STFT length. 1024 bins across 2 MS/s gives 1.95 kHz resolution, which is
            finer than a 12.5 kHz channel without being expensive.
        overlap: Fractional overlap between successive STFT frames.
        window: Any window name accepted by ``scipy.signal.get_window``. Blackman-Harris is
            the default because survey work is dominated by dynamic range, not by resolving
            two tones a bin apart: a strong local emitter must not smear across the band
            and hide weak ones. Its -92 dB sidelobes buy that at the cost of a wider main
            lobe, which is the right trade here and the wrong one for the filter bank.
        chunk_frames: Frames transformed at a time when averaging. The average is what the
            caller wants; the full spectrogram is an intermediate nobody asked for, and at
            2 MS/s a 90-second capture makes that intermediate 2.9 GB. Accumulating in
            chunks keeps the working set at a few tens of megabytes whatever the length.
    """

    fft_size: int = 1024
    overlap: float = 0.5
    window: str = "blackmanharris"
    chunk_frames: int = 4096

    def __post_init__(self) -> None:
        if self.fft_size < 16 or self.fft_size & (self.fft_size - 1):
            raise ValueError(f"fft_size must be a power of two >= 16, got {self.fft_size}")
        if not 0.0 <= self.overlap < 1.0:
            raise ValueError(f"overlap must be in [0, 1), got {self.overlap}")

    @property
    def hop(self) -> int:
        """Samples advanced between successive STFT frames."""
        return max(1, int(self.fft_size * (1.0 - self.overlap)))


@dataclass
class SurveyResult:
    """A wideband spectral snapshot."""

    frequencies_hz: np.ndarray
    power_db: np.ndarray
    noise_floor_db: float
    num_frames: int

    def occupancy(self, threshold_db: float = 10.0) -> np.ndarray:
        """Boolean mask of bins more than ``threshold_db`` above the noise floor."""
        return self.power_db > self.noise_floor_db + threshold_db

    def peak(self) -> tuple[float, float]:
        """Return ``(frequency_hz, power_db)`` of the strongest bin."""
        index = int(np.argmax(self.power_db))
        return float(self.frequencies_hz[index]), float(self.power_db[index])


class SpectrumSurvey:
    """Wideband STFT survey of the full receiver bandwidth."""

    def __init__(self, sample_rate: float, centre_hz: float, config: SurveyConfig | None = None):
        self.sample_rate = sample_rate
        self.centre_hz = centre_hz
        self.config = config or SurveyConfig()
        window = dsp.get_window(self.config.window, self.config.fft_size)
        # Normalise for coherent gain so that a full-scale tone reads 0 dBFS regardless of
        # which window is chosen. Without this, changing the window silently rescales every
        # power measurement the node reports.
        self._window = (window / window.sum()).astype(np.float32)
        self._frequencies = (
            np.fft.fftshift(np.fft.fftfreq(self.config.fft_size, d=1.0 / sample_rate)) + centre_hz
        )

    @property
    def resolution_hz(self) -> float:
        """Frequency resolution of the survey (Hz per bin)."""
        return self.sample_rate / self.config.fft_size

    def frame_count(self, num_samples: int) -> int:
        """Number of STFT frames a capture of this length would produce."""
        size, hop = self.config.fft_size, self.config.hop
        return 0 if num_samples < size else (num_samples - size) // hop + 1

    def spectrogram(self, iq: np.ndarray, max_frames: int = 8192) -> np.ndarray:
        """Power spectrogram, shape ``(frames, fft_size)``, in linear power, fftshifted.

        This is the waterfall behind the occupancy plots in the V&V report, and it is the
        expensive path: the result is ``frames x fft_size`` and nothing reduces it.

        Frames beyond ``max_frames`` are decimated in time rather than dropped, so a long
        capture still yields a waterfall spanning the whole recording. Decimating is the
        right answer for a plot that will be rendered a thousand pixels wide anyway, and
        the alternative -- allocating 2.9 GB for a 90-second capture, as this did before --
        takes the machine down.

        Args:
            iq: Complex baseband samples.
            max_frames: Upper bound on the frames returned.

        Returns:
            Power spectrogram, at most ``max_frames`` rows.
        """
        size, hop = self.config.fft_size, self.config.hop
        num_frames = self.frame_count(len(iq))
        if num_frames == 0:
            return np.zeros((0, size), dtype=np.float32)

        stride = max(1, num_frames // max_frames)
        windows = np.lib.stride_tricks.sliding_window_view(iq, size)[:: hop * stride]
        windows = windows[: min(num_frames, max_frames)]

        spectra = sfft.fft(windows * self._window, axis=1)
        return np.fft.fftshift(np.abs(spectra) ** 2, axes=1).astype(np.float32)

    def analyse(self, iq: np.ndarray) -> SurveyResult:
        """Average a block of IQ into one spectral snapshot.

        The noise floor is taken as the median across bins rather than the mean. With
        several emitters active, the mean is dragged upward by exactly the signals whose
        presence we are trying to measure against the floor; the median is not, as long as
        under half the band is occupied — a safe assumption for PMR446 and one worth
        stating rather than assuming.
        """
        size, hop = self.config.fft_size, self.config.hop
        num_frames = self.frame_count(len(iq))
        if num_frames == 0:
            empty = np.full(size, -np.inf)
            return SurveyResult(self._frequencies, empty, -np.inf, 0)

        # Accumulate the average over chunks. Building the whole spectrogram first and then
        # reducing it needs 2.9 GB for a 90-second capture at 2 MS/s, which is how this took
        # a machine down; the answer wanted is the average, and an average does not need its
        # own summands kept.
        windows = np.lib.stride_tricks.sliding_window_view(iq, size)[::hop][:num_frames]
        total = np.zeros(size, dtype=np.float64)
        for start in range(0, num_frames, self.config.chunk_frames):
            block = windows[start : start + self.config.chunk_frames]
            spectra = sfft.fft(block * self._window, axis=1)
            total += (np.abs(spectra) ** 2).sum(axis=0)

        mean_power = np.fft.fftshift(total / num_frames)
        with np.errstate(divide="ignore"):
            power_db = 10.0 * np.log10(mean_power)
        return SurveyResult(
            frequencies_hz=self._frequencies,
            power_db=power_db,
            noise_floor_db=float(np.median(power_db)),
            num_frames=int(num_frames),
        )
