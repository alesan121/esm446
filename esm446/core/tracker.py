"""Group per-frame detections into emissions.

CFAR answers "is there energy in bin k during frame n". An emission is something else: a
contiguous burst of energy in one channel, with a start, a duration, and enough baseband
samples behind it to demodulate and identify. This module turns the first into the second.

Replacing the v0 coupling
-------------------------
v0 did this across two processes and a directory. The channeliser decided a burst had ended,
wrote ``raw_<ts>.s16`` and ``freq_<ts>.txt`` into ``/tmp``, and a Bash loop polled every
300 ms for files to process. Three problems went away by moving the logic in-process rather
than by fixing them individually:

- **The race.** The Bash loop tested `[ -f "$RAW_FILE" ]` — that the raw file *exists*, not
  that the writer had finished. A burst caught mid-write was processed truncated, and
  nothing reported an error.
- **The polling.** A 300 ms sleep bounded latency for no reason, and burned a wakeup per
  cycle whether or not anything had happened.
- **The lost state.** Everything the channeliser knew about an emission beyond frequency and
  power — its SNR, its exact frame extent, the noise floor it stood against — could not
  cross the filesystem boundary, so it was discarded.

Hangover
--------
A transmission does not present as a continuous run of detections. The carrier stays up for
the whole over, but its received power does not: multipath fading puts nulls in it, and while
the signal is in a null the detector loses it. Closing an emission on the first quiet frame
shreds one transmission into dozens of fragments, each too short for the CTCSS stage to
decide on.

So an emission stays open through up to ``hangover_frames`` quiet frames and closes only when
the gap exceeds that.

The default is 6250 frames, which at the 25 kHz channel rate is **250 ms**. That number comes
from the fading, not from the modulation. An earlier default of 75 frames — 3 ms — was chosen
against modulation nulls, which was the wrong physical quantity: at walking pace and 446 MHz
the Doppler spread is a couple of hertz, so a fade lasts hundreds of milliseconds, not
microseconds. Measured against a scenario with Rayleigh fading, 3 ms turned 12 transmissions
into 63 reports with 6 fragmented; 250 ms recovers them as 12.

The upper bound is set by how close together two separate overs can be. A quarter of a second
of dead air is longer than any fade and shorter than the gap between one person releasing the
key and another pressing it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Emission:
    """One completed burst of energy in a single channel.

    Attributes:
        bin_index: Channeliser bin the emission occupied.
        start_frame: Absolute frame index of the first detected frame.
        end_frame: Absolute frame index of the last detected frame.
        peak_power: Highest per-frame power seen, in linear full-scale units.
        mean_power: Mean power across the emission, in linear full-scale units.
        noise_estimate: Mean CFAR noise estimate over the emission.
        iq: Baseband samples of the channel across the emission.
        left_power: Mean power in the bin below, used to refine the frequency estimate.
        right_power: Mean power in the bin above.
        detected_frames: Frames in which the channel was above the detection threshold, as
            opposed to the total span the emission covers.
    """

    bin_index: int
    start_frame: int
    end_frame: int
    peak_power: float
    mean_power: float
    noise_estimate: float
    iq: np.ndarray
    left_power: float = 0.0
    right_power: float = 0.0
    detected_frames: int = 0

    @property
    def bin_offset(self) -> float:
        """Fractional bin offset of the emitter from its bin centre, in [-0.5, 0.5].

        A filter bank quantises frequency to the bin grid, and an emitter sitting between
        two bins is reported at whichever centre wins. That is fine until the winning centre
        happens to be a nominal PMR446 channel, at which point a genuinely off-grid emitter
        is confidently misreported as on-grid — and detecting off-grid emissions is a large
        part of why the band is surveyed at all.

        Parabolic interpolation over the three bins in decibels recovers the offset. The
        energy of a signal halfway between two bins splits between them, and the imbalance
        says which way and by how much.

        The guard matters as much as the formula. Parabolic interpolation assumes the middle
        sample is a peak, and in a band of adjacent channels it frequently is not: a strong
        emitter one channel away makes its neighbour's "left" bin larger than its centre, the
        parabola opens the wrong way, and the result saturates at half a bin. Observed on a
        weak emitter sitting beside one 37 dB stronger, which was pushed 6.3 kHz off its true
        frequency and then failed to match its own ground truth. Where the centre bin is not
        the local maximum there is no peak to interpolate, and the bin centre is the better
        answer.
        """
        if self.left_power <= 0.0 or self.right_power <= 0.0 or self.mean_power <= 0.0:
            return 0.0
        if self.mean_power <= self.left_power or self.mean_power <= self.right_power:
            return 0.0
        left = 10.0 * np.log10(self.left_power)
        centre = 10.0 * np.log10(self.mean_power)
        right = 10.0 * np.log10(self.right_power)
        denominator = left - 2.0 * centre + right
        if abs(denominator) < 1e-12:
            return 0.0
        return float(np.clip(0.5 * (left - right) / denominator, -0.5, 0.5))

    @property
    def occupancy(self) -> float:
        """Fraction of the emission's span in which the channel was actually detected.

        A real transmission holds its carrier up: detection is near-continuous, broken only
        by fades. A chain of unrelated false alarms in the same bin is not — it is a handful
        of isolated hits spread across the span the hangover glued together.

        Duration alone cannot tell those apart, which is what makes this the companion of a
        long hangover rather than an optional extra.
        """
        return self.detected_frames / self.frame_count if self.frame_count else 0.0

    @property
    def frame_count(self) -> int:
        """Number of frames spanned, inclusive of both ends."""
        return self.end_frame - self.start_frame + 1

    @property
    def snr_db(self) -> float:
        """Mean power over the estimated noise floor, in dB."""
        return float(10.0 * np.log10(self.mean_power / max(self.noise_estimate, 1e-30)))

    @property
    def peak_power_dbfs(self) -> float:
        """Peak power relative to full scale."""
        return float(10.0 * np.log10(max(self.peak_power, 1e-30)))

    def duration_seconds(self, channel_rate: float) -> float:
        """Duration of the emission in seconds.

        Args:
            channel_rate: Per-channel output sample rate in Hz.

        Returns:
            Duration in seconds.
        """
        return self.frame_count / channel_rate


@dataclass
class _OpenEmission:
    """An emission still accumulating frames."""

    bin_index: int
    start_frame: int
    last_detected_frame: int
    peak_power: float = 0.0
    power_sum: float = 0.0
    noise_sum: float = 0.0
    left_sum: float = 0.0
    right_sum: float = 0.0
    frames: int = 0
    chunks: list[np.ndarray] = field(default_factory=list)


class EmissionTracker:
    """Assemble per-frame CFAR detections into per-channel emissions.

    State carries across `update` calls, so an emission spanning a block boundary is a
    single emission rather than two fragments.
    """

    def __init__(
        self,
        hangover_frames: int = 6_250,
        min_frames: int = 2_500,
        max_frames: int = 500_000,
        min_occupancy: float = 0.35,
    ) -> None:
        """Initialise the tracker.

        Args:
            hangover_frames: Quiet frames tolerated inside one emission before it closes.
                At the 25 kHz channel rate the default is 250 ms; see the module docstring
                for why fading rather than modulation sets it.
            min_frames: Emissions shorter than this are discarded as transients. At the
                25 kHz channel rate the default is 100 ms, which is below the shortest
                deliberate PMR446 over and above any keying transient.
            max_frames: Hard cap on emission length, so a stuck carrier or an interferer
                cannot grow a buffer without bound. The default is 20 seconds.
            min_occupancy: Minimum fraction of an emission's span that must have been
                detected. A quarter-second hangover will happily glue unrelated false
                alarms in one bin into something long enough to pass ``min_frames``; an
                occupancy floor rejects them, because a real carrier is present for most of
                its own emission and a chain of noise hits is not.
        """
        if hangover_frames < 0:
            raise ValueError(f"hangover_frames must be >= 0, got {hangover_frames}")
        if min_frames < 1:
            raise ValueError(f"min_frames must be >= 1, got {min_frames}")
        if not 0.0 <= min_occupancy <= 1.0:
            raise ValueError(f"min_occupancy must be in [0, 1], got {min_occupancy}")
        self.hangover_frames = hangover_frames
        self.min_frames = min_frames
        self.max_frames = max_frames
        self.min_occupancy = min_occupancy
        self._open: dict[int, _OpenEmission] = {}
        self._frame_offset = 0

    @property
    def open_count(self) -> int:
        """Number of emissions currently open."""
        return len(self._open)

    def update(
        self, spectra: np.ndarray, power: np.ndarray, mask: np.ndarray, noise: np.ndarray
    ) -> list[Emission]:
        """Feed one block of channelised frames and collect any emissions that completed.

        Args:
            spectra: Channelised IQ, shape ``(frames, bins)``.
            power: Per-bin power, shape ``(frames, bins)``.
            mask: CFAR detections, shape ``(frames, bins)``.
            noise: CFAR noise estimate, shape ``(frames, bins)``.

        Returns:
            Emissions that ended within this block, in the order they closed.
        """
        if spectra.shape[0] == 0:
            return []

        completed: list[Emission] = []
        # Only bins that are already open or that fired at least once can change state, and
        # in a quiet band that is a handful out of 160.
        candidates = set(np.flatnonzero(mask.any(axis=0)).tolist()) | set(self._open)

        num_bins = power.shape[1]
        for bin_index in sorted(candidates):
            completed.extend(
                self._update_bin(
                    bin_index,
                    spectra[:, bin_index],
                    power[:, bin_index],
                    mask[:, bin_index],
                    noise[:, bin_index],
                    # Wrapping is correct: the bins are the FFT of a contiguous band, so the
                    # highest really is adjacent to the lowest.
                    power[:, (bin_index - 1) % num_bins],
                    power[:, (bin_index + 1) % num_bins],
                )
            )

        self._frame_offset += spectra.shape[0]
        return completed

    def _update_bin(
        self,
        bin_index: int,
        channel_iq: np.ndarray,
        power: np.ndarray,
        mask: np.ndarray,
        noise: np.ndarray,
        left_power: np.ndarray,
        right_power: np.ndarray,
    ) -> list[Emission]:
        """Advance one bin's state machine across a block of frames."""
        completed: list[Emission] = []
        detected = np.flatnonzero(mask)

        if detected.size == 0:
            # Nothing here this block. An open emission survives only if the accumulated
            # gap is still inside the hangover.
            open_emission = self._open.get(bin_index)
            if open_emission is not None:
                gap = self._frame_offset + mask.size - open_emission.last_detected_frame
                if gap > self.hangover_frames:
                    completed.append(self._close(bin_index))
                else:
                    self._accumulate(
                        open_emission,
                        channel_iq,
                        power,
                        noise,
                        left_power,
                        right_power,
                        count_frames=False,
                    )
            return completed

        # Split the detected frames into runs separated by gaps longer than the hangover.
        gaps = np.diff(detected)
        breaks = np.flatnonzero(gaps > self.hangover_frames)
        run_starts = np.concatenate([[detected[0]], detected[breaks + 1]])
        run_ends = np.concatenate([detected[breaks], [detected[-1]]])

        for index, (start, end) in enumerate(zip(run_starts, run_ends, strict=True)):
            open_emission = self._open.get(bin_index)

            if open_emission is not None and index == 0:
                gap = self._frame_offset + int(start) - open_emission.last_detected_frame
                if gap > self.hangover_frames:
                    completed.append(self._close(bin_index))
                    open_emission = None

            if open_emission is None:
                open_emission = _OpenEmission(
                    bin_index=bin_index,
                    start_frame=self._frame_offset + int(start),
                    last_detected_frame=self._frame_offset + int(start),
                )
                self._open[bin_index] = open_emission

            segment = slice(int(start), int(end) + 1)
            self._accumulate(
                open_emission,
                channel_iq[segment],
                power[segment],
                noise[segment],
                left_power[segment],
                right_power[segment],
            )
            open_emission.last_detected_frame = self._frame_offset + int(end)

            is_last_run = index == len(run_starts) - 1
            # A run that is not the last one is followed, by construction, by a gap longer
            # than the hangover. The last run has to be checked against the frames that
            # remain in this block, or an emission that clearly ended mid-block would stay
            # open until the next one arrived.
            trailing_gap = mask.size - int(end) - 1
            over_cap = open_emission.frames >= self.max_frames
            if not is_last_run or trailing_gap > self.hangover_frames or over_cap:
                if over_cap:
                    logger.warning(
                        "tracker: bin %d hit the %d-frame cap; closing it early",
                        bin_index,
                        self.max_frames,
                    )
                completed.append(self._close(bin_index))

        return completed

    @staticmethod
    def _accumulate(
        emission: _OpenEmission,
        channel_iq: np.ndarray,
        power: np.ndarray,
        noise: np.ndarray,
        left_power: np.ndarray,
        right_power: np.ndarray,
        count_frames: bool = True,
    ) -> None:
        """Append samples and statistics to an open emission."""
        if channel_iq.size == 0:
            return
        emission.chunks.append(channel_iq)
        if count_frames:
            emission.frames += int(power.size)
            emission.power_sum += float(power.sum())
            emission.noise_sum += float(noise.sum())
            emission.left_sum += float(left_power.sum())
            emission.right_sum += float(right_power.sum())
            emission.peak_power = max(emission.peak_power, float(power.max()))

    def _close(self, bin_index: int) -> Emission:
        """Finalise an open emission and remove it from the tracker."""
        open_emission = self._open.pop(bin_index)
        frames = max(open_emission.frames, 1)
        emission = Emission(
            bin_index=bin_index,
            start_frame=open_emission.start_frame,
            end_frame=open_emission.last_detected_frame,
            peak_power=open_emission.peak_power,
            mean_power=open_emission.power_sum / frames,
            noise_estimate=open_emission.noise_sum / frames,
            iq=np.concatenate(open_emission.chunks) if open_emission.chunks else np.empty(0),
            left_power=open_emission.left_sum / frames,
            right_power=open_emission.right_sum / frames,
            detected_frames=open_emission.frames,
        )
        logger.debug(
            "tracker: closed bin %d, %d frames, SNR %.1f dB",
            bin_index,
            emission.frame_count,
            emission.snr_db,
        )
        return emission

    def flush(self) -> list[Emission]:
        """Close every open emission. Call at end of stream.

        Without this, a transmission still in progress when the recording ends would be
        dropped entirely rather than reported as what was captured of it.
        """
        return [self._close(bin_index) for bin_index in sorted(self._open)]

    def filter_short(self, emissions: list[Emission]) -> list[Emission]:
        """Drop emissions that are too short, or too sparse, to be a real transmission.

        Args:
            emissions: Candidate emissions.

        Returns:
            Those spanning at least ``min_frames`` frames with at least ``min_occupancy``
            of that span actually detected.
        """
        return [
            e
            for e in emissions
            if e.frame_count >= self.min_frames and e.occupancy >= self.min_occupancy
        ]
