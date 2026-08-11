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
from esm446.core.calibration import PowerCalibration, linear_to_dbfs
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

        emissions = self.tracker.update(spectra, power, mask, noise)
        self.frames_processed += spectra.shape[0]
        return [self._report(e) for e in self.tracker.filter_short(emissions)]

    def flush(self) -> list[EmissionReport]:
        """Close and report any emission still open at end of stream."""
        return [self._report(e) for e in self.tracker.filter_short(self.tracker.flush())]

    def run(self, source: IQSource, block_size: int = DEFAULT_BLOCK_SIZE) -> list[EmissionReport]:
        """Consume a source to exhaustion and return every emission found.

        Args:
            source: Where the IQ comes from. The node cannot tell live capture from replay.
            block_size: Complex samples read per iteration.

        Returns:
            Every emission report produced, in completion order.
        """
        self._start_time = time.time()
        reports: list[EmissionReport] = []
        started = time.perf_counter()

        with source:
            while True:
                block = source.read(block_size)
                if block is None:
                    break
                if block.size:
                    reports.extend(self.process_block(block))

        reports.extend(self.flush())

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

    def _report(self, emission: Emission) -> EmissionReport:
        """Turn a completed emission into its metadata record."""
        tone_hz, classification, deviation_hz = self._identify(emission)

        frequency_hz = float(self._bin_frequencies[emission.bin_index])
        peak_dbfs = emission.peak_power_dbfs
        estimated_dbm = (
            self.calibration.to_dbm(peak_dbfs, self.gains) if self.gains is not None else None
        )

        report = EmissionReport(
            timestamp=self._start_time + emission.start_frame / self.config.channel_rate,
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


def power_to_dbfs(power_linear: np.ndarray) -> np.ndarray:
    """Convert linear bin power to dBFS. Thin re-export for callers of this module."""
    return linear_to_dbfs(power_linear)
