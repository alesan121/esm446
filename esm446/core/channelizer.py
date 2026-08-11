"""Polyphase analysis filter bank.

Splits a wideband IQ stream into ``M`` uniformly spaced channels using one FFT per output
frame instead of one mixer-and-filter chain per channel.

Why this replaces the v0 approach
---------------------------------
The v0 prototype channelised by brute force: for each of 57 channels it built a 101-tap FIR
with ``firwin``, mixed the *entire* input block down by that channel's offset, and ran a
full-length ``lfilter``. Cost grows linearly with channel count, and measured on the
development machine it took 2.14 s to process 0.328 s of signal — 6.5x slower than real
time, meaning roughly 85 % of the incoming signal was never examined.

A polyphase filter bank computes every channel from a single prototype filter and a single
size-``M`` FFT per frame. The prototype is designed once at construction. Cost per output
frame is ``M*K`` multiply-accumulates for the polyphase fold plus ``O(M log M)`` for the
FFT — essentially independent of how many channels you actually care about, which is what
makes surveying all 160 bins as cheap as surveying the 16 nominal ones.

Structure
---------
Each output frame ``n`` consumes ``D`` new input samples and reads a window of ``L = M*K``
samples starting at ``n*D``::

    y[n, l] = h[l] * x[n*D + l]                 windowing by the prototype
    a[n, m] = sum_j y[n, j*M + m]               fold K blocks down to M points
    X[n, k] = FFT_M(a[n, :])[k]                 all M channels at once
    Y[n, k] = X[n, k] * exp(-2j*pi*k*n*D/M)     de-rotate the sliding commutator

The final de-rotation is the step that is easy to omit and easy to get wrong: because the
analysis window advances by ``D`` samples while the FFT basis has period ``M``, each bin
picks up a frame-dependent phase ramp unless ``D`` is a multiple of ``M``. Without the
correction, magnitudes look correct while phase — and therefore any subsequent demodulation
— is wrong. `tests/test_channelizer.py` pins this down with a constant-phase assertion.

Oversampling
------------
With ``D == M`` the bank is critically sampled: output rate equals channel spacing, and the
filter's transition band folds back onto itself. Running ``D == M//2`` (2x oversampled)
gives each channel an output rate of twice its spacing, which leaves room for a real
transition band between the 6.25 kHz channel edge and the 12.5 kHz alias limit. The cost is
2x the output data rate, which is cheap here and worth it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import scipy.fft as sfft
from scipy import signal as dsp

#: Frames processed per fold pass. The accumulator for a whole 0.13 s block is ~16 MB and
#: streams out of cache on every one of the K passes; chunking it keeps the working set
#: resident. Measured: 0.198 -> 0.161 CPU-s per signal second at 2 MS/s.
_FOLD_CHUNK_FRAMES = 1024


@dataclass(frozen=True)
class ChannelizerConfig:
    """Geometry of the filter bank.

    Attributes:
        sample_rate: Input IQ sample rate in Hz.
        num_channels: Number of output channels ``M``. Channel spacing is
            ``sample_rate / num_channels``.
        decimation: Input samples consumed per output frame ``D``. Must divide
            ``num_channels``. Use ``num_channels // 2`` for a 2x oversampled bank.
        taps_per_phase: Prototype filter length in units of ``num_channels`` (``K``).
            Longer means a sharper channel filter at linear cost.
        stopband_atten_db: Target stopband attenuation of the prototype filter, used to
            pick the Kaiser window parameter.
    """

    sample_rate: float
    num_channels: int
    decimation: int
    taps_per_phase: int = 12
    stopband_atten_db: float = 80.0

    def __post_init__(self) -> None:
        if self.num_channels <= 0 or self.num_channels % 2 != 0:
            raise ValueError(f"num_channels must be positive and even, got {self.num_channels}")
        if self.decimation <= 0 or self.num_channels % self.decimation != 0:
            raise ValueError(
                f"decimation {self.decimation} must be positive and divide "
                f"num_channels {self.num_channels}"
            )
        if self.taps_per_phase < 2:
            raise ValueError(f"taps_per_phase must be >= 2, got {self.taps_per_phase}")

    @property
    def channel_spacing(self) -> float:
        """Frequency spacing between adjacent channel centres (Hz)."""
        return self.sample_rate / self.num_channels

    @property
    def channel_rate(self) -> float:
        """Output sample rate of each individual channel (Hz)."""
        return self.sample_rate / self.decimation

    @property
    def oversampling(self) -> float:
        """Output rate as a multiple of channel spacing. 1.0 is critically sampled."""
        return self.num_channels / self.decimation

    @property
    def num_taps(self) -> int:
        """Total length ``L`` of the prototype filter."""
        return self.num_channels * self.taps_per_phase


def design_prototype(config: ChannelizerConfig) -> np.ndarray:
    """Design the prototype lowpass filter for a channeliser.

    The passband edge is half the channel spacing. The stopband edge is placed at the alias
    limit — half the per-channel output rate — rather than at the passband edge, because
    energy beyond that point is what actually folds into the channel output. Oversampling
    therefore buys a genuine transition band instead of forcing a brick wall.

    Designed once, at construction. The v0 prototype rebuilt its filter inside the per-
    channel loop on every block.
    """
    nyquist = config.sample_rate / 2.0
    passband_edge = config.channel_spacing / 2.0
    alias_limit = config.channel_rate / 2.0
    # Place the -6 dB point midway through the available transition region.
    cutoff = (passband_edge + alias_limit) / 2.0 / nyquist

    beta = dsp.kaiser_beta(config.stopband_atten_db)
    taps = dsp.firwin(
        config.num_taps,
        cutoff,
        window=("kaiser", beta),
        scale=True,
    )
    # Normalise DC gain to unity so that a full-scale tone at a bin centre reads back as
    # unit magnitude in that bin. This is what makes bin power directly interpretable as
    # dBFS, and therefore what makes the power calibration meaningful.
    return (taps / taps.sum()).astype(np.float32)


class PolyphaseChannelizer:
    """Streaming polyphase analysis filter bank.

    Call `process` repeatedly with consecutive blocks of IQ; filter state carries across
    calls, so block boundaries introduce no transient. Not thread-safe.
    """

    def __init__(self, config: ChannelizerConfig, prototype: np.ndarray | None = None) -> None:
        self.config = config
        self.prototype = design_prototype(config) if prototype is None else prototype
        if len(self.prototype) != config.num_taps:
            raise ValueError(
                f"prototype has {len(self.prototype)} taps, expected {config.num_taps}"
            )
        # Prototype reshaped as (K, P, D) to match the fold's block layout, where
        # P = M/D sub-blocks of D samples make up one polyphase segment.
        self._sub_blocks = config.num_channels // config.decimation
        self._phases = self.prototype.reshape(
            config.taps_per_phase, self._sub_blocks, config.decimation
        )
        self._tail = np.zeros(0, dtype=np.complex64)
        self._frame_index = 0
        # exp(-2j*pi*k*n*D/M) repeats in n with this period, so the correction table is
        # small and can be precomputed once.
        self._phase_period = config.num_channels // math.gcd(config.decimation, config.num_channels)
        n = np.arange(self._phase_period)[:, None]
        k = np.arange(config.num_channels)[None, :]
        self._derotation = np.exp(
            -2j * np.pi * n * k * config.decimation / config.num_channels
        ).astype(np.complex64)

    def reset(self) -> None:
        """Clear filter state and frame counter."""
        self._tail = np.zeros(0, dtype=np.complex64)
        self._frame_index = 0

    @property
    def latency_samples(self) -> int:
        """Group delay of the bank in input samples (linear-phase prototype)."""
        return (self.config.num_taps - 1) // 2

    def process(self, samples: np.ndarray) -> np.ndarray:
        """Channelise a block of IQ.

        Args:
            samples: Complex IQ, any length. Leftover samples are retained internally.

        Returns:
            Array of shape ``(frames, num_channels)``, complex64. Column ``k`` is channel
            ``k`` in FFT bin order; use `esm446.core.bands.bin_frequencies` to map bins to
            absolute frequencies. May have zero rows if the block was shorter than the
            filter window.
        """
        cfg = self.config
        x = samples.astype(np.complex64, copy=False)
        if self._tail.size:
            x = np.concatenate([self._tail, x])

        n_frames = 0 if x.size < cfg.num_taps else (x.size - cfg.num_taps) // cfg.decimation + 1
        if n_frames == 0:
            self._tail = x
            return np.zeros((0, cfg.num_channels), dtype=np.complex64)

        # Fold the K polyphase segments of every frame down to (frames, M).
        #
        # Because frames advance by exactly D samples, viewing the input as contiguous
        # D-sample blocks makes every term a plain row slice with unit stride, rather than
        # the overlapping strided window a naive formulation produces. Same arithmetic,
        # ~25 % less time, because the reads stop fighting the cache.
        p_count, d = self._sub_blocks, cfg.decimation
        num_blocks = x.size // d
        blocks = x[: num_blocks * d].reshape(num_blocks, d)
        folded = np.empty((n_frames, p_count, d), dtype=np.complex64)

        for lo in range(0, n_frames, _FOLD_CHUNK_FRAMES):
            hi = min(lo + _FOLD_CHUNK_FRAMES, n_frames)
            span = hi - lo
            for p in range(p_count):
                base = lo + p
                acc = blocks[base : base + span] * self._phases[0, p]
                for j in range(1, cfg.taps_per_phase):
                    offset = base + j * p_count
                    acc += blocks[offset : offset + span] * self._phases[j, p]
                folded[lo:hi, p, :] = acc

        # scipy.fft keeps single precision end to end; numpy.fft always computes in double
        # and forces a cast back, which measured 2x slower here for no accuracy benefit.
        spectra = sfft.fft(folded.reshape(n_frames, cfg.num_channels), axis=1)

        # De-rotate the sliding commutator (see module docstring).
        frame_numbers = (self._frame_index + np.arange(n_frames)) % self._phase_period
        spectra *= self._derotation[frame_numbers]

        self._frame_index += n_frames
        self._tail = x[n_frames * cfg.decimation :]
        return spectra

    def frequency_response(self, num_points: int = 8192) -> tuple[np.ndarray, np.ndarray]:
        """Prototype filter response, for verification and documentation plots.

        Returns:
            ``(frequencies_hz, magnitude_db)`` for baseband offsets from a channel centre.
        """
        w, h = dsp.freqz(self.prototype, worN=num_points, fs=self.config.sample_rate)
        with np.errstate(divide="ignore"):
            magnitude_db = 20.0 * np.log10(np.abs(h))
        return w, magnitude_db
