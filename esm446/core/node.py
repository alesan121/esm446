"""The ESM node: the pipeline that turns IQ into emitter metadata.

This is what replaces `EW_FoF_Scanner.sh`. The shell script coordinated a Python channeliser,
`ffmpeg`, a CTCSS detector and a range estimator by writing files into `/tmp` and polling for
them. Here it is one process holding one pipeline, which removes an entire class of failure
rather than repairing instances of it: there is no partially written file to read, no polling
interval to tune, and no information lost crossing a filesystem boundary.

Pipeline
--------
::

    IQSource ─► PolyphaseChannelizer ─► CfarDetector ─► EmissionTracker
                                                              │
                                              ┌───────────────┘
                                              ▼
                               NfmDemodulator ─► CtcssDetector ─► EmissionReport

Each stage is a plain object taking arrays and returning arrays, so the node is assembled
from them rather than being a monolith, and each is tested independently.

Metadata only
-------------
The node emits `EmissionReport` records: when, where in the spectrum, how strong, how long,
which sub-audible code. Demodulated audio exists only inside `_identify`, only long enough to
run tone detection, and is never returned or written. That is the policy in
`docs/06_legal_ethics.md` expressed as control flow rather than as a promise.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from esm446.core import bands
from esm446.core.calibration import PowerCalibration
from esm446.core.channelizer import ChannelizerConfig, PolyphaseChannelizer
from esm446.core.ctcss import CtcssDetector
from esm446.core.demod import NfmDemodulator
from esm446.core.detector import CfarConfig, CfarDetector
from esm446.core.rfchain import HackrfGains
from esm446.core.source import IQSource
from esm446.core.tracker import Emission, EmissionTracker

logger = logging.getLogger(__name__)

#: Samples read from the source per iteration. At 2 MS/s this is 131 ms of signal, which is
#: long enough to amortise per-block overhead and short enough to keep latency sane.
DEFAULT_BLOCK_SIZE = 262_144


@dataclass
class EmissionReport:
    """Metadata for one detected emission. The node's only output.

    Attributes:
        timestamp: Unix time at which the emission started.
        frequency_hz: Absolute centre frequency of the occupied channel.
        pmr_channel: PMR446 channel number, or ``None`` if the emission is off-grid.
        bin_index: Channeliser bin the emission occupied.
        duration_s: Length of the emission in seconds.
        peak_power_dbfs: Peak power relative to receiver full scale.
        snr_db: Mean power over the locally estimated noise floor.
        estimated_dbm: Absolute received power, or ``None`` when no valid calibration
            applies. Never a guess.
        calibrated: Whether ``estimated_dbm`` is backed by a measured calibration.
        ctcss_tone_hz: Identified sub-audible tone, or ``None``.
        classification: ``FRIEND`` or ``UNKNOWN`` against the configured pre-shared tone.
        offset_s: Seconds from the start of the capture. Carried alongside the absolute
            time so a detection can be located in a recording without arithmetic, which is
            what two test vectors were cut from the wrong place for want of.
        peak_deviation_hz: Peak FM deviation, an emitter discriminant.
        gains: Receiver gain configuration in force, without which the power figures
            cannot be calibrated after the fact.
    """

    timestamp: float
    frequency_hz: float
    pmr_channel: int | None
    bin_index: int
    duration_s: float
    peak_power_dbfs: float
    snr_db: float
    estimated_dbm: float | None
    calibrated: bool
    ctcss_tone_hz: float | None
    classification: str
    offset_s: float
    peak_deviation_hz: float
    gains: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Return the report as a plain dictionary, ready for JSON serialisation."""
        return asdict(self)


class EsmNode:
    """Wire the signal chain together and run it over a source."""

    def __init__(
        self,
        channelizer_config: ChannelizerConfig,
        centre_frequency: float,
        cfar_config: CfarConfig | None = None,
        calibration: PowerCalibration | None = None,
        gains: HackrfGains | None = None,
        expected_ctcss_hz: float | None = None,
        tracker: EmissionTracker | None = None,
        dc_guard_bins: int = 1,
    ) -> None:
        self.config = channelizer_config
        self.centre_frequency = centre_frequency
        self.channelizer = PolyphaseChannelizer(channelizer_config)
        self.detector = CfarDetector(cfar_config)
        self.tracker = tracker or EmissionTracker()
        self.demodulator = NfmDemodulator(channelizer_config.channel_rate)
        self.ctcss = CtcssDetector(channelizer_config.channel_rate)
        self.calibration = calibration or PowerCalibration()
        self.gains = gains
        self.expected_ctcss_hz = expected_ctcss_hz
        self.dc_guard_bins = dc_guard_bins
        self._bin_frequencies = bands.bin_frequencies(
            centre_frequency, channelizer_config.sample_rate, channelizer_config.num_channels
        )
        self._start_time = 0.0
        self.frames_processed = 0

        logger.info(
            "node: %d channels of %.1f Hz at %.6f MHz, CFAR %s with P_fa %.1e",
            channelizer_config.num_channels,
            channelizer_config.channel_spacing,
            centre_frequency / 1e6,
            self.detector.config.method,
            self.detector.config.pfa,
        )
        if not self.calibration.is_calibrated:
            logger.warning(
                "node: no power calibration loaded; power is reported in dBFS and every "
                "dBm figure will be null"
            )

    def process_block(self, samples: np.ndarray) -> list[EmissionReport]:
        """Push one block of IQ through the pipeline.

        Args:
            samples: Complex baseband samples at the configured sample rate.

        Returns:
            Reports for emissions that completed within this block.
        """
        spectra = self.channelizer.process(samples)
        if spectra.shape[0] == 0:
            return []

        power = (np.abs(spectra) ** 2).astype(np.float64)
        noise = self.detector.noise_estimate(power)
        mask = power > noise * self.detector.threshold_factor

        # Bin 0 is DC, and a direct-conversion receiver leaks its own local oscillator
        # there. Measured on a HackRF One that spur runs 31 dB above the noise floor and is
        # present for as long as the receiver is on, so without this the node reports one
        # permanent emission of unlimited duration. The bin carries no external signal by
        # construction, and offset tuning has already placed it outside the allocation, so
        # excluding it costs nothing.
        if self.dc_guard_bins > 0:
            mask[:, : self.dc_guard_bins] = False
            if self.dc_guard_bins > 1:
                mask[:, -(self.dc_guard_bins - 1) :] = False

        emissions = self.tracker.update(spectra, power, mask, noise)
        self.frames_processed += spectra.shape[0]
        return self._report_all(self.tracker.filter_short(emissions))

    def flush(self) -> list[EmissionReport]:
        """Close and report any emission still open at end of stream."""
        return self._report_all(self.tracker.filter_short(self.tracker.flush()))

    def _report_all(self, emissions: list[Emission]) -> list[EmissionReport]:
        """Merge emissions that are one emitter seen twice, then report each group."""
        return [self._report_group(group) for group in self._group_adjacent(emissions)]

    def _group_adjacent(self, emissions: list[Emission]) -> list[list[Emission]]:
        """Group emissions in adjacent bins that overlap in time.

        A transmitter is under no obligation to sit on a bin centre. One landing halfway
        between two bins splits its energy evenly, both bins cross the detection threshold,
        and the tracker — which works per bin — closes two emissions for one emitter. The
        `delta` emitter in the demonstration scenario does exactly this.

        Adjacency plus time overlap is the discriminator. Two genuinely different emitters on
        adjacent channels transmitting simultaneously would be merged by this rule, which is
        the cost; against it, a single off-grid emitter reported twice, once at a channel it
        was never on, is the more damaging error in a system whose job includes finding
        emissions that are not on the channel plan.
        """
        groups: list[list[Emission]] = []
        for emission in sorted(emissions, key=lambda e: (e.start_frame, e.bin_index)):
            for group in groups:
                if any(self._is_adjacent(emission, other) for other in group):
                    group.append(emission)
                    break
            else:
                groups.append([emission])
        return groups

    def _is_adjacent(self, first: Emission, second: Emission) -> bool:
        """Whether two emissions are one bin apart and overlap in time."""
        num_bins = self.config.num_channels
        gap = min(
            (first.bin_index - second.bin_index) % num_bins,
            (second.bin_index - first.bin_index) % num_bins,
        )
        overlaps = first.start_frame <= second.end_frame and second.start_frame <= first.end_frame
        return gap == 1 and overlaps

    def _group_frequency(self, group: list[Emission]) -> float:
        """Estimate the emitter's frequency from a group of bins.

        For a single bin, parabolic interpolation against its neighbours. For a split
        emitter, the power-weighted centroid of the bins it landed in, which returns the
        midpoint when the split is even and the dominant bin when it is not — exactly the
        cases parabolic interpolation cannot resolve from one bin's point of view, because
        neither bin is a local maximum.
        """
        spacing = self.config.channel_spacing
        if len(group) == 1:
            emission = group[0]
            return float(self._bin_frequencies[emission.bin_index] + emission.bin_offset * spacing)

        # Unwrap bin indices relative to the strongest, so a group straddling bin 0 does not
        # average to the opposite side of the band.
        strongest = max(group, key=lambda e: e.mean_power)
        num_bins = self.config.num_channels
        total = 0.0
        weighted = 0.0
        for emission in group:
            offset = (emission.bin_index - strongest.bin_index + num_bins // 2) % num_bins - (
                num_bins // 2
            )
            weighted += emission.mean_power * offset
            total += emission.mean_power
        centroid = weighted / total if total else 0.0
        return float(self._bin_frequencies[strongest.bin_index] + centroid * spacing)

    def run(
        self,
        source: IQSource,
        block_size: int = DEFAULT_BLOCK_SIZE,
        sink: Any | None = None,
    ) -> list[EmissionReport]:
        """Consume a source to exhaustion and return every emission found.

        Args:
            source: Where the IQ comes from. The node cannot tell live capture from replay.
            block_size: Complex samples read per iteration.
            sink: Optional `esm446.io.sinks.EmissionSink`. Records are written as they
                complete rather than at the end, so a capture killed after an hour keeps
                the hour rather than losing it.

        Returns:
            Every emission report produced, in completion order.
        """
        # The capture's own start time, not the clock now. Replaying a recording must not
        # relabel it with when it was analysed.
        self._start_time = getattr(source, "start_time", None) or time.time()
        reports: list[EmissionReport] = []
        started = time.perf_counter()

        with source:
            while True:
                block = source.read(block_size)
                if block is None:
                    break
                if block.size:
                    batch = self.process_block(block)
                    if batch:
                        reports.extend(batch)
                        if sink is not None:
                            sink.write(batch)

        final = self.flush()
        if final:
            reports.extend(final)
            if sink is not None:
                sink.write(final)

        elapsed = time.perf_counter() - started
        signal_seconds = self.frames_processed / self.config.channel_rate
        logger.info(
            "node: %.1f s of signal in %.1f s of CPU (%.3f cpu-s/s), %d emissions",
            signal_seconds,
            elapsed,
            elapsed / max(signal_seconds, 1e-9),
            len(reports),
        )
        return reports

    def _identify(self, emission: Emission) -> tuple[float | None, str, float]:
        """Demodulate an emission and identify its sub-audible tone.

        The demodulated audio is local to this method and is never returned or stored.

        Args:
            emission: The completed emission.

        Returns:
            ``(ctcss_tone_hz, classification, peak_deviation_hz)``.
        """
        if emission.iq.size < 2:
            return None, "UNKNOWN", 0.0

        demodulated = self.demodulator.demodulate(emission.iq)
        result = self.ctcss.detect(demodulated.audio)
        return (
            result.tone_hz,
            result.classify(self.expected_ctcss_hz),
            demodulated.peak_deviation_hz,
        )

    def _report_group(self, group: list[Emission]) -> EmissionReport:
        """Turn one emitter's emissions -- usually one, sometimes a split pair -- into a record."""
        emission = max(group, key=lambda e: e.mean_power)
        tone_hz, classification, deviation_hz = self._identify(emission)

        # Refine the frequency past the bin grid before deciding anything about it. Nothing
        # obliges a transmitter to sit on a channel centre, and reporting an off-grid emitter
        # at the nearest bin would state it was on a channel it was never on.
        frequency_hz = self._group_frequency(group)
        peak_dbfs = max(e.peak_power_dbfs for e in group)
        estimated_dbm = (
            self.calibration.to_dbm(peak_dbfs, self.gains) if self.gains is not None else None
        )

        offset_s = emission.start_frame / self.config.channel_rate
        report = EmissionReport(
            timestamp=self._start_time + offset_s,
            offset_s=offset_s,
            frequency_hz=frequency_hz,
            pmr_channel=bands.channel_at(frequency_hz),
            bin_index=emission.bin_index,
            duration_s=emission.duration_seconds(self.config.channel_rate),
            peak_power_dbfs=peak_dbfs,
            snr_db=emission.snr_db,
            estimated_dbm=estimated_dbm,
            calibrated=estimated_dbm is not None,
            ctcss_tone_hz=tone_hz,
            classification=classification,
            peak_deviation_hz=deviation_hz,
            gains=self.gains.as_dict() if self.gains else {},
        )

        logger.info(
            "node: %.6f MHz (%s) %.2f s SNR %.1f dB CTCSS %s -> %s",
            frequency_hz / 1e6,
            f"PMR{report.pmr_channel}" if report.pmr_channel else "off-grid",
            report.duration_s,
            report.snr_db,
            f"{tone_hz:.1f} Hz" if tone_hz else "none",
            classification,
        )
        return report
