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

The pair test, and why it is off by default
-------------------------------------------
An emitter halfway between two bins loses 6.02 dB (see `esm446.core.channelizer`). Testing
adjacent bins as a summed cell recovers part of that, and `os_pair_threshold_factor` derives
the threshold for it exactly. Measured, as the SNR at which half of frames detect:

===========  ==========  ========  ========
offset       single-bin  + pairs   change
===========  ==========  ========  ========
0.00 bins    14.81 dB    15.19 dB  -0.38 dB
0.25 bins    14.81 dB    15.06 dB  -0.25 dB
0.40 bins    15.69 dB    15.94 dB  -0.25 dB
0.50 bins    20.19 dB    18.94 dB  **+1.25 dB**
===========  ==========  ========  ========

Worst-case ripple falls from 5.38 dB to 3.75 dB, paid for with 0.25 to 0.38 dB everywhere
else -- the cost of splitting the false alarm budget across two tests.

Averaged over offsets that is roughly a wash, and on the recorded two-emitter capture it was
worse than a wash: the 0.38 dB pushed a genuine sideband detection at 6.6 dB SNR below the
threshold, which broke the symmetric-pair arithmetic `esm446.analysis.artefacts` uses to
attribute splatter, while adding several detections at negative single-bin SNR whose
frequency estimates are too poor to attribute. Measured on synthetic tones the test is a
modest win; measured on the real captures it made the system's output worse.

So it ships implemented, verified, and **off**. Turn it on where the worst-case off-grid
emitter matters more than 0.3 dB everywhere and more than the quality of weak detections --
which is a real operating point, just not this one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize

#: How far above its own held estimate a bin may sit and still count towards the frame's
#: noise level. Eight is about 9 dB: comfortably above what noise reaches at these false alarm
#: rates, comfortably below the tens of decibels a real emission stands at.
_LEVEL_EXCLUSION_FACTOR = 8.0

#: Guards the level ratio against a division by zero on an all-zero frame, which a file
#: source can produce at the end of a capture.
_TINY = 1e-300


@dataclass(frozen=True)
class CfarConfig:
    """CFAR geometry and false alarm design point.

    Attributes:
        num_reference: Total reference cells, split evenly either side of the cell under
            test. More cells mean a lower-variance noise estimate but less ability to
            track a sloping noise floor.
        num_guard: Guard cells either side, excluded from the estimate. They keep energy
            that has leaked out of the cell under test from raising its own threshold.
        pfa: Design probability of false alarm **per cell per frame**.

            The unit is what makes the number counter-intuitive. Radar texts routinely quote
            1e-4 or 1e-6, but the sensible value depends entirely on how many cells pass
            through the detector per second, and here that is 160 bins at a 25 kHz frame
            rate: four million. At 1e-4 that is 400 false alarms every second across the
            band; at 1e-8 it is effectively none.

            The cost of tightening from 1e-4 to 1e-8 is 4.3 dB of threshold, which is a
            straightforward trade for removing four hundred spurious detections a second.
            It matters more than it sounds: the tracker's quarter-second hangover will glue
            unrelated false alarms in one bin into something long enough to look like a
            transmission, so a loose P_fa does not merely add noise to the output, it
            fabricates emissions.
        method: ``"ca"`` for cell averaging, ``"os"`` for ordered statistic.
        os_rank_fraction: For OS-CFAR, which order statistic to use as a fraction of
            ``num_reference``. 0.75 is the usual robust choice: high enough to be a good
            noise estimate, low enough to reject a few interferers.
        track_level: Whether to correct the held noise estimate to each frame's own noise
            level. On synthetic noise this changes nothing, because synthetic noise is
            stationary; on real receiver noise it is worth a factor of 64 in false alarm
            rate. Off only to reproduce the behaviour that made the defect visible.
        detect_pairs: Whether to run a second test on the sum of each adjacent bin pair,
            which recovers about 2 dB for an emitter sitting between two bins. The false
            alarm budget is split between the two tests, so the design ``pfa`` still bounds
            the pair of them together and enabling it costs 0.3 dB on the single-bin test.
        update_interval: Frames between noise-floor estimates. The estimate is held for
            the frames in between.

            This is what makes OS-CFAR affordable. Estimating per frame builds a
            ``(frames, bins, references)`` array — 101 MB for one 131 ms block at
            2 MS/s — and then takes an order statistic across all of it, which measured
            4.93 CPU-seconds per signal second against the channeliser's 0.19. The node
            was 4.8x slower than real time, and none of it was the filter bank.

            Holding the estimate is not a compromise, it is the physics: at a 25 kHz frame
            rate a frame is 40 microseconds, while a thermal noise floor is stationary over
            milliseconds at least. The default of 64 frames re-estimates every 2.6 ms,
            which is still far faster than any real change in the noise environment, and
            costs 64x less. The false alarm rate is unaffected, and the test suite measures
            that rather than assuming it.
    """

    num_reference: int = 24
    num_guard: int = 2
    pfa: float = 1e-8
    method: str = "os"
    os_rank_fraction: float = 0.75
    update_interval: int = 64
    track_level: bool = True
    detect_pairs: bool = False

    def __post_init__(self) -> None:
        if self.num_reference < 2 or self.num_reference % 2 != 0:
            raise ValueError(f"num_reference must be even and >= 2, got {self.num_reference}")
        if self.num_guard < 0:
            raise ValueError(f"num_guard must be >= 0, got {self.num_guard}")
        if not 0.0 < self.pfa < 1.0:
            raise ValueError(f"pfa must be in (0, 1), got {self.pfa}")
        if self.method not in ("ca", "os"):
            raise ValueError(f"method must be 'ca' or 'os', got {self.method!r}")
        if self.update_interval < 1:
            raise ValueError(f"update_interval must be >= 1, got {self.update_interval}")

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


def ca_pair_threshold_factor(num_reference: int, pfa: float) -> float:
    """CA-CFAR threshold factor for the sum of two adjacent cells.

    The companion to `os_pair_threshold_factor`, derived the same way but against the cell
    average rather than an order statistic. With ``S`` the mean of ``N`` exponential
    reference cells and the test statistic gamma-distributed with two degrees of freedom,

        P_fa(alpha) = M(beta) * (1 + beta / (1 + beta/N)),  beta = 2*alpha
        M(beta)     = (1 + beta/N)**-N

    which is closed form and needs only a root find to invert.

    Args:
        num_reference: Reference cells in the CFAR window.
        pfa: Design false alarm probability for this test.

    Returns:
        The threshold factor, applied to a single cell's noise estimate.
    """

    def log_pfa(log_alpha: float) -> float:
        beta = 2.0 * np.exp(log_alpha)
        ratio = 1.0 + beta / num_reference
        return float(-num_reference * np.log(ratio) + np.log1p(beta / ratio) - np.log(pfa))

    return float(np.exp(optimize.brentq(log_pfa, -20.0, 30.0, xtol=1e-12)))


def os_pair_threshold_factor(num_reference: int, rank: int, pfa: float) -> float:
    """OS-CFAR threshold factor for the sum of two adjacent cells.

    An emitter is under no obligation to land on a bin centre, and one halfway between two
    bins puts half its energy in each. Testing the pair as one cell is the obvious response,
    and it works -- but not for the obvious reason, which is worth setting down because the
    intuition is wrong in a way that would lead to overstating the gain.

    Summing two bins does **not** recover the split energy in any useful sense: the noise in
    the two bins sums along with the signal, so the ratio of one to the other is exactly what
    it was in a single bin. What the sum does buy is *variance*. One exponential cell has a
    standard deviation equal to its mean; the sum of two has a relative spread smaller by a
    factor of sqrt(2), so the threshold that achieves a given false alarm rate sits closer to
    the mean. That is non-coherent integration gain, and measured here it is about 2 dB.

    The false alarm probability follows the same route as `os_threshold_factor`. With ``U``
    the ``k``-th order statistic of ``N`` reference cells normalised by the true mean, and the
    test statistic gamma-distributed with two degrees of freedom,

        P_fa(alpha) = E[ exp(-2*alpha*U) * (1 + 2*alpha*U) ]
                    = M(2*alpha) * (1 + 2*alpha * sum_i 1/(N - i + 2*alpha))

    where ``M(beta) = prod_i (N - i) / (N - i + beta)`` is the same expectation the
    single-cell case reduces to. The factor of two inside comes from the pair's noise
    estimate being twice a single cell's.

    Args:
        num_reference: Reference cells in the CFAR window.
        rank: Order statistic used as the noise estimate.
        pfa: Design false alarm probability for this test.

    Returns:
        The threshold factor, applied to a single cell's noise estimate.
    """
    indices = np.arange(rank)

    def log_pfa(log_alpha: float) -> float:
        beta = 2.0 * np.exp(log_alpha)
        log_m = np.sum(np.log((num_reference - indices) / (num_reference - indices + beta)))
        correction = np.log1p(beta * np.sum(1.0 / (num_reference - indices + beta)))
        return float(log_m + correction - np.log(pfa))

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

        # Two tests run on every cell when pair detection is on, so the design false alarm
        # rate is split between them and their union still respects it. The union bound is
        # what matters operationally: a false alarm is a false alarm whichever test raised
        # it. Splitting costs 0.3 dB, which is cheap against the 2 dB the pair test returns.
        per_test_pfa = self.config.pfa / 2.0 if self.config.detect_pairs else self.config.pfa

        if self.config.method == "ca":
            self.threshold_factor = ca_threshold_factor(self.config.num_reference, per_test_pfa)
        else:
            self.threshold_factor = os_threshold_factor(
                self.config.num_reference, self.config.os_rank, per_test_pfa
            )

        if not self.config.detect_pairs:
            self.pair_threshold_factor = float("inf")
        elif self.config.method == "ca":
            self.pair_threshold_factor = ca_pair_threshold_factor(
                self.config.num_reference, per_test_pfa
            )
        else:
            self.pair_threshold_factor = os_pair_threshold_factor(
                self.config.num_reference, self.config.os_rank, per_test_pfa
            )

    def pair_mask(
        self,
        power: np.ndarray,
        noise: np.ndarray,
        exclude: np.ndarray | None = None,
    ) -> np.ndarray:
        """Flag bins belonging to an adjacent pair whose summed power crosses the threshold.

        Reuses the single-bin noise estimate rather than computing a second one. The noise in
        a pair is the sum of the noise in its two bins, which on a locally flat floor is twice
        one bin's -- and the OS estimate is the expensive part of the detector, so not
        repeating it is what makes this test essentially free.

        Both bins of a crossing pair are flagged. Reporting them as one emitter is already
        handled downstream by the adjacent-bin merge, which exists for exactly this geometry.

        Args:
            power: Per-bin power, shape ``(frames, bins)`` or ``(bins,)``.
            noise: Noise estimate of the same shape.
            exclude: Optional per-bin mask of bins that carry something other than signal --
                the local oscillator leakage at DC, for instance. Any pair containing one is
                not tested. Without this the spur pairs with its neighbour and flags it,
                which turns one excluded bin into an emission on the next channel along.

        Returns:
            Boolean mask of the same shape.
        """
        if not self.config.detect_pairs:
            return np.zeros_like(power, dtype=bool)

        # Pair k is bins k and k+1, wrapping at the band edge as the rest of the detector does.
        paired_power = power + np.roll(power, -1, axis=-1)
        paired_noise = noise + np.roll(noise, -1, axis=-1)
        crossed = paired_power > paired_noise * self.pair_threshold_factor
        if exclude is not None:
            crossed &= ~(exclude | np.roll(exclude, -1, axis=-1))
        return crossed | np.roll(crossed, 1, axis=-1)

    def noise_estimate(self, power: np.ndarray) -> np.ndarray:
        """Estimate noise power local to each bin, excluding the cell under test and guards.

        The estimate is separated into a shape and a level, because the two change on
        different timescales and only one of them is expensive.

        The **shape** across bins is set by the analogue filters and the channeliser, and it
        does not move: it is computed every ``update_interval`` frames and held in between.
        The **level** moves with receiver gain and local-oscillator behaviour on a timescale
        far shorter than that hold, and it is recomputed for every frame by
        `_level_correction`, which costs almost nothing.

        Holding both was the original design and it is why the node reported eight
        twenty-second emissions on the first ambient capture ever put through it. Measured on
        receiver noise, holding the level for 64 frames gives a false alarm rate of
        3.3e-3 against a 1e-8 design point; tracking it per frame gives 5.2e-5, a
        sixty-fourfold improvement, for 0.03 CPU-seconds per signal second.

        None of this appears on synthetic noise, which is stationary by construction: there
        the two are indistinguishable and the test suite measured them as such for months.

        Args:
            power: Per-bin power, shape ``(bins,)`` or ``(frames, bins)``.

        Returns:
            Noise estimate with the same shape as ``power``.
        """
        interval = self.config.update_interval
        if power.ndim > 1 and interval > 1 and power.shape[0] > interval:
            frames = power.shape[0]
            sampled = self._noise_estimate_exact(power[::interval])
            held = np.repeat(sampled, interval, axis=0)[:frames]
            if self.config.track_level:
                levels = self._frame_levels(power, held)
                # Each frame is compared against the level of the frame its estimate was
                # built from, not against the estimate itself. That matters: the estimate is
                # an order statistic and sits about 1.4 dB above the mean it is derived from,
                # so dividing by it would bias every threshold down by that much. Comparing
                # like with like makes the correction exactly 1.0 on stationary noise.
                anchors = np.repeat(levels[::interval], interval)[:frames]
                held = held * (levels / np.maximum(anchors, _TINY))[:, None]
            return held
        return self._noise_estimate_exact(power)

    def _frame_levels(self, power: np.ndarray, held: np.ndarray) -> np.ndarray:
        """Noise level of each frame, in the units of the power itself.

        The ratio of the current mean bin power to the held one, taken over the bins that are
        **not** carrying a signal. Robustness is the whole difficulty here, and the naive
        version is a trap worth describing because it was measured and nearly shipped.

        A plain mean over all bins is the cheapest statistic and gave the best false alarm
        rate of anything tried, 1.9e-5. It also raised the threshold by up to 26 dB whenever a
        strong emitter was present, because one bin 40 dB above the floor moves the mean of
        160 bins by a factor of sixty. Measured on the recorded two-emitter capture, detection
        of the two real carriers fell from 74 % of frames to 13 % and 7 %. It would have
        traded the system's actual job for a better number on a page.

        Excluding the bins that exceed their own held estimate by more than
        `_LEVEL_EXCLUSION_FACTOR` fixes that: the emitters exclude themselves, the correction
        stays within a factor of three, and detection of the two carriers is bit-for-bit what
        it was without any correction at all.

        A median would also be robust, and was measured: it costs three times as much and
        gives a worse false alarm rate, 4.0e-4 against 5.2e-5. The mean is the better estimator
        of a mean, which is what a power level is.

        Args:
            power: Per-bin power, shape ``(frames, bins)``.
            held: The held estimate, used only to decide which bins are carrying signal.

        Returns:
            Mean quiet-bin power per frame, shape ``(frames,)``.
        """
        quiet = power < held * _LEVEL_EXCLUSION_FACTOR
        counted = quiet.sum(axis=1)
        # Every bin occupied at once is possible in principle and must not divide by zero.
        # Falling back to the plain mean there is the least wrong option available, and it
        # cannot desensitise anything that is not already fully occupied.
        return np.where(
            counted > 0,
            (power * quiet).sum(axis=1) / np.maximum(counted, 1),
            power.mean(axis=1),
        )

    def _noise_estimate_exact(self, power: np.ndarray) -> np.ndarray:
        """Estimate the noise floor for every frame given, with no time decimation."""
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

    def detection_mask(self, power: np.ndarray, exclude: np.ndarray | None = None) -> np.ndarray:
        """Boolean mask of detections, for whole blocks of frames at once.

        Both tests, because both are what the detector does: a cell crossing on its own, and
        a cell belonging to an adjacent pair that crosses together. The design ``pfa`` bounds
        their union, so this is also the mask whose measured false alarm rate the test suite
        compares against the design point -- measuring only one of two tests would report half
        the rate the system actually produces.

        Args:
            power: Per-bin power, shape ``(frames, bins)`` or ``(bins,)``.
            exclude: Bins that must not take part in a pair. See `pair_mask`.

        Returns:
            Boolean mask of the same shape.
        """
        noise = self.noise_estimate(power)
        mask = power > noise * self.threshold_factor
        return mask | self.pair_mask(power, noise, exclude)
