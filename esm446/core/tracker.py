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
Speech is not continuous. A talker pauses, a syllable ends, and for a few frames the channel
falls below the detection threshold without the transmission having stopped. Closing an
emission on the first quiet frame would shred one transmission into dozens of fragments, and
each fragment would be too short for the CTCSS stage to decide on.

So an emission stays open through up to ``hangover_frames`` quiet frames and closes only
when the gap exceeds that. At the 25 kHz channel rate a frame is 40 microseconds, so the
default of 75 frames is a 3 ms tolerance — long enough to bridge modulation nulls, short
enough that two separate overs do not merge into one.
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
    """

    bin_index: int
    start_frame: int
    end_frame: int
    peak_power: float
    mean_power: float
    noise_estimate: float
    iq: np.ndarray

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
    frames: int = 0
    chunks: list[np.ndarray] = field(default_factory=list)


class EmissionTracker:
    """Assemble per-frame CFAR detections into per-channel emissions.

    State carries across `update` calls, so an emission spanning a block boundary is a
    single emission rather than two fragments.
    """

    def __init__(
        self,
        hangover_frames: int = 75,
        min_frames: int = 250,
        max_frames: int = 500_000,
    ) -> None:
        """Initialise the tracker.

        Args:
            hangover_frames: Quiet frames tolerated inside one emission before it closes.
            min_frames: Emissions shorter than this are discarded as transients. At the
                25 kHz channel rate the default is 10 ms.
            max_frames: Hard cap on emission length, so a stuck carrier or an interferer
                cannot grow a buffer without bound. The default is 20 seconds.
        """
        if hangover_frames < 0:
            raise ValueError(f"hangover_frames must be >= 0, got {hangover_frames}")
        if min_frames < 1:
            raise ValueError(f"min_frames must be >= 1, got {min_frames}")
        self.hangover_frames = hangover_frames
        self.min_frames = min_frames
        self.max_frames = max_frames
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

        for bin_index in sorted(candidates):
            completed.extend(
                self._update_bin(
                    bin_index,
                    spectra[:, bin_index],
                    power[:, bin_index],
                    mask[:, bin_index],
                    noise[:, bin_index],
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
                    self._accumulate(open_emission, channel_iq, power, noise, count_frames=False)
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
            self._accumulate(open_emission, channel_iq[segment], power[segment], noise[segment])
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
        """Drop emissions too short to be a real transmission.

        Args:
            emissions: Candidate emissions.

        Returns:
            Those spanning at least ``min_frames`` frames.
        """
        return [e for e in emissions if e.frame_count >= self.min_frames]
