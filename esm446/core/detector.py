"""Constant false alarm rate detection across channeliser bins.

Replaces v0's fixed thresholds::

    MIN_POWER  = 0.065
    SNR_FACTOR = 2.0

Those two numbers encode a specific antenna, a specific receiver gain setting and the noise
floor of one particular afternoon. Move the node indoors, change the LNA, or run it in a
noisier band and they are wrong, silently — either the detector goes deaf or it fires
constantly. Neither failure announces itself.

CFAR inverts the problem. Instead of fixing the threshold and letting the false alarm rate
float with the environment, it estimates the local noise power from reference cells
surrounding each cell under test and scales it by a factor derived from the false alarm
probability you asked for. The threshold moves with the noise floor; ``P_fa`` stays where
you put it. That makes it a design parameter you can state, justify and verify, which is
the whole point.

Two estimators are provided:

**CA-CFAR** (cell averaging) is optimal when the reference cells contain noise only. Its
threshold factor follows in closed form: for ``N`` reference cells of exponentially
distributed power, ``alpha = N * (P_fa**(-1/N) - 1)``.

**OS-CFAR** (ordered statistic) takes the ``k``-th smallest reference cell instead of the
mean. It matters here specifically because PMR446 channels are adjacent by design: when two
emitters are active on neighbouring channels, a strong one sitting inside the reference
window inflates the CA-CFAR noise estimate and masks the weaker one. An order statistic
below the interferer's rank simply ignores it. The price is roughly 0.5 dB of detection
loss in pure noise, which is a good trade in a band whose normal condition is several
simultaneous users.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize


@dataclass(frozen=True)
class CfarConfig:
    """CFAR geometry and false alarm design point.

    Attributes:
        num_reference: Total reference cells, split evenly either side of the cell under
            test. More cells mean a lower-variance noise estimate but less ability to
            track a sloping noise floor.
        num_guard: Guard cells either side, excluded from the estimate. They keep energy
            that has leaked out of the cell under test from raising its own threshold.
        pfa: Design probability of false alarm per cell per frame.
        method: ``"ca"`` for cell averaging, ``"os"`` for ordered statistic.
        os_rank_fraction: For OS-CFAR, which order statistic to use as a fraction of
            ``num_reference``. 0.75 is the usual robust choice: high enough to be a good
            noise estimate, low enough to reject a few interferers.
    """

    num_reference: int = 24
    num_guard: int = 2
    pfa: float = 1e-4
    method: str = "os"
    os_rank_fraction: float = 0.75

    def __post_init__(self) -> None:
        if self.num_reference < 2 or self.num_reference % 2 != 0:
            raise ValueError(f"num_reference must be even and >= 2, got {self.num_reference}")
        if self.num_guard < 0:
            raise ValueError(f"num_guard must be >= 0, got {self.num_guard}")
        if not 0.0 < self.pfa < 1.0:
            raise ValueError(f"pfa must be in (0, 1), got {self.pfa}")
        if self.method not in ("ca", "os"):
            raise ValueError(f"method must be 'ca' or 'os', got {self.method!r}")

    @property
    def os_rank(self) -> int:
        """1-based rank of the order statistic used by OS-CFAR."""
        return max(1, min(self.num_reference, round(self.os_rank_fraction * self.num_reference)))

    @property
    def window_size(self) -> int:
        """Total cells spanned by the CFAR window, including the cell under test."""
        return self.num_reference + 2 * self.num_guard + 1


def ca_threshold_factor(num_reference: int, pfa: float) -> float:
    """Closed-form CA-CFAR threshold factor for exponentially distributed cell power."""
    return num_reference * (pfa ** (-1.0 / num_reference) - 1.0)


def os_threshold_factor(num_reference: int, rank: int, pfa: float) -> float:
    """OS-CFAR threshold factor, solved numerically.

    For ``N`` i.i.d. exponential reference cells and the ``k``-th order statistic, the false
    alarm probability has the closed form

        P_fa(alpha) = prod_{i=0}^{k-1} (N - i) / (N - i + alpha)

    which cannot be inverted in elementary functions, so the factor is found by root
    finding. Solving in ``log(alpha)`` keeps the bracket well conditioned across the many
    orders of magnitude ``alpha`` spans as ``P_fa`` tightens.
    """
    indices = np.arange(rank)

    def log_pfa(log_alpha: float) -> float:
        alpha = np.exp(log_alpha)
        terms = (num_reference - indices) / (num_reference - indices + alpha)
        return float(np.sum(np.log(terms)) - np.log(pfa))

    return float(np.exp(optimize.brentq(log_pfa, -20.0, 30.0, xtol=1e-12)))


@dataclass
class Detection:
    """One emitter detection in one CFAR frame."""

    bin_index: int
    power: float
    noise_estimate: float
    threshold: float

    @property
    def snr_db(self) -> float:
        """Signal-to-noise ratio in dB, relative to the locally estimated noise floor."""
        return 10.0 * np.log10(self.power / self.noise_estimate)


class CfarDetector:
    """Sliding-window CFAR over the channeliser's bin power spectrum.

    The spectrum is treated as circular. That is correct rather than convenient: the bins
    are the FFT of a contiguous band, so the highest bin really is adjacent to the lowest,
    and wrapping avoids the edge bins getting a truncated, biased reference window.
    """

    def __init__(self, config: CfarConfig | None = None) -> None:
        self.config = config or CfarConfig()
        if self.config.method == "ca":
            self.threshold_factor = ca_threshold_factor(self.config.num_reference, self.config.pfa)
        else:
            self.threshold_factor = os_threshold_factor(
                self.config.num_reference, self.config.os_rank, self.config.pfa
            )

    def noise_estimate(self, power: np.ndarray) -> np.ndarray:
        """Estimate noise power local to each bin, excluding the cell under test and guards."""
        cfg = self.config
        num_bins = power.shape[-1]
        if num_bins < cfg.window_size:
            raise ValueError(
                f"need at least {cfg.window_size} bins for this CFAR window, got {num_bins}"
            )

        half = cfg.num_reference // 2
        # Offsets of the reference cells relative to the cell under test, guards removed.
        offsets = np.concatenate(
            [
                np.arange(-cfg.num_guard - half, -cfg.num_guard),
                np.arange(cfg.num_guard + 1, cfg.num_guard + half + 1),
            ]
        )
        indices = (np.arange(num_bins)[:, None] + offsets[None, :]) % num_bins
        reference = power[..., indices]

        if cfg.method == "ca":
            return reference.mean(axis=-1)
        # np.partition puts the k-th smallest in place without a full sort.
        return np.partition(reference, cfg.os_rank - 1, axis=-1)[..., cfg.os_rank - 1]

    def threshold(self, power: np.ndarray) -> np.ndarray:
        """Detection threshold for each bin."""
        return self.noise_estimate(power) * self.threshold_factor

    def detect(self, power: np.ndarray) -> list[Detection]:
        """Detect emitters in a single frame's power spectrum.

        Args:
            power: Non-negative power per bin, shape ``(num_bins,)``.

        Returns:
            Detections ordered by descending SNR.
        """
        if power.ndim != 1:
            raise ValueError(f"expected a 1-D power spectrum, got shape {power.shape}")
        noise = self.noise_estimate(power)
        threshold = noise * self.threshold_factor
        hits = np.flatnonzero(power > threshold)
        detections = [
            Detection(
                bin_index=int(index),
                power=float(power[index]),
                noise_estimate=float(noise[index]),
                threshold=float(threshold[index]),
            )
            for index in hits
        ]
        return sorted(detections, key=lambda d: d.snr_db, reverse=True)

    def detection_mask(self, power: np.ndarray) -> np.ndarray:
        """Boolean mask of detections, for whole blocks of frames at once.

        Accepts shape ``(frames, bins)`` and returns the same shape, which is how the
        measured false alarm rate is checked against the design point in the test suite.
        """
        return power > self.threshold(power)
