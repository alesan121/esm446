"""Score node output against scenario ground truth.

A detection is not simply right or wrong. It can be the right emitter reported with the wrong
extent, one transmission split into two reports, or two overs merged into one — and each of
those means something different about which stage is misbehaving. Collapsing them into a
single accuracy figure hides exactly the information worth having, so the matcher keeps them
apart.

What the outcomes mean
----------------------
- **Matched**: a report overlapping a truth emission in time and sitting within half a channel
  of its frequency.
- **Missed**: a truth emission with no report. Detection sensitivity, so a function of SNR.
- **Spurious**: a report matching no truth emission. Against a scene with no emitters this is
  the end-to-end false alarm rate, which is the CFAR design point measured through the whole
  pipeline rather than in the detector's own unit tests.
- **Fragmented**: several reports against one truth emission. The tracker's hangover is too
  short, so pauses in speech are ending transmissions.
- **Merged**: one report spanning several truth emissions. The hangover is too long, so
  separate overs are running together.

Fragmentation and merging are the two failures a plain hit-or-miss score cannot see, and they
are the ones that move when the hangover is tuned.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from esm446.core.node import EmissionReport
from esm446.sim.scenario import TruthEmission

logger = logging.getLogger(__name__)

#: Frequency tolerance when matching a report to a truth emission, in Hz. Half a channel:
#: anything further away is a different channel, not a mismeasurement.
FREQUENCY_TOLERANCE_HZ = 6_250.0


@dataclass
class Match:
    """One truth emission and the reports that were matched to it."""

    truth: TruthEmission
    reports: list[EmissionReport] = field(default_factory=list)

    @property
    def detected(self) -> bool:
        """Whether anything was reported for this emission."""
        return bool(self.reports)

    @property
    def fragmented(self) -> bool:
        """Whether one transmission was reported as several."""
        return len(self.reports) > 1

    @property
    def duration_error_s(self) -> float:
        """Reported extent minus true extent, in seconds. Zero if undetected."""
        if not self.reports:
            return 0.0
        reported = sum(report.duration_s for report in self.reports)
        return reported - self.truth.duration_s

    @property
    def ctcss_correct(self) -> bool:
        """Whether the identified tone matches the transmitted one, including both absent."""
        if not self.reports:
            return False
        identified = self.reports[0].ctcss_tone_hz
        if self.truth.ctcss_hz is None:
            return identified is None
        return identified is not None and abs(identified - self.truth.ctcss_hz) < 0.5


@dataclass
class Score:
    """The result of scoring one run against one scenario."""

    matches: list[Match]
    spurious: list[EmissionReport]
    merged: int
    scene_duration_s: float

    @property
    def num_truth(self) -> int:
        """Number of transmissions in the scene."""
        return len(self.matches)

    @property
    def num_detected(self) -> int:
        """Number of transmissions with at least one report."""
        return sum(1 for match in self.matches if match.detected)

    @property
    def probability_of_detection(self) -> float:
        """Fraction of transmissions detected. Zero-safe on an empty scene."""
        return self.num_detected / self.num_truth if self.num_truth else 0.0

    @property
    def false_alarms_per_second(self) -> float:
        """Spurious reports per second of scene.

        Against a scene with no emitters this is the end-to-end false alarm rate.
        """
        return len(self.spurious) / self.scene_duration_s if self.scene_duration_s else 0.0

    @property
    def num_fragmented(self) -> int:
        """Transmissions reported as more than one emission."""
        return sum(1 for match in self.matches if match.fragmented)

    @property
    def channel_accuracy(self) -> float:
        """Fraction of detected transmissions assigned the correct PMR channel.

        An off-grid emitter counts as correct only when reported as off-grid. Snapping it to
        the nearest nominal channel is a wrong answer, not a rounding.
        """
        detected = [m for m in self.matches if m.detected]
        if not detected:
            return 0.0
        correct = sum(1 for m in detected if m.reports[0].pmr_channel == m.truth.pmr_channel)
        return correct / len(detected)

    @property
    def ctcss_accuracy(self) -> float:
        """Fraction of detected transmissions whose tone was identified correctly."""
        detected = [m for m in self.matches if m.detected]
        if not detected:
            return 0.0
        return sum(1 for m in detected if m.ctcss_correct) / len(detected)

    @property
    def duration_rmse_s(self) -> float:
        """Root mean square error of reported duration, over detected transmissions."""
        errors = [m.duration_error_s for m in self.matches if m.detected]
        return float(np.sqrt(np.mean(np.square(errors)))) if errors else 0.0

    def detection_by_snr(self, edges: list[float] | None = None) -> list[tuple[float, float, int]]:
        """Probability of detection binned by the SNR the emission was generated at.

        This is the curve that says what the node can actually hear, and it is the reason
        the scenario records the SNR it used rather than only the amplitude.

        Args:
            edges: Bin edges in dB. Defaults to 5 dB bins from 0 to 40.

        Returns:
            ``(low_db, probability, count)`` per bin, skipping empty bins.
        """
        edges = edges or [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]
        rows = []
        for low, high in zip(edges[:-1], edges[1:], strict=True):
            in_bin = [m for m in self.matches if low <= m.truth.snr_db < high]
            if in_bin:
                detected = sum(1 for m in in_bin if m.detected)
                rows.append((low, detected / len(in_bin), len(in_bin)))
        return rows

    def as_dict(self) -> dict[str, Any]:
        """Return the headline figures as a plain dictionary."""
        return {
            "transmissions": self.num_truth,
            "detected": self.num_detected,
            "probability_of_detection": round(self.probability_of_detection, 4),
            "spurious": len(self.spurious),
            "false_alarms_per_second": round(self.false_alarms_per_second, 4),
            "fragmented": self.num_fragmented,
            "merged": self.merged,
            "channel_accuracy": round(self.channel_accuracy, 4),
            "ctcss_accuracy": round(self.ctcss_accuracy, 4),
            "duration_rmse_s": round(self.duration_rmse_s, 4),
        }

    def describe(self) -> str:
        """Human-readable summary for the demonstration and the V&V report."""
        lines = [
            f"transmissions          {self.num_truth}",
            f"detected               {self.num_detected}"
            f"  (Pd {self.probability_of_detection:.1%})",
            f"spurious reports       {len(self.spurious)}"
            f"  ({self.false_alarms_per_second:.3f}/s)",
            f"fragmented / merged    {self.num_fragmented} / {self.merged}",
            f"channel accuracy       {self.channel_accuracy:.1%}",
            f"CTCSS accuracy         {self.ctcss_accuracy:.1%}",
            f"duration RMSE          {self.duration_rmse_s:.3f} s",
        ]
        by_snr = self.detection_by_snr()
        if by_snr:
            lines.append("")
            lines.append("detection by SNR:")
            for low, probability, count in by_snr:
                lines.append(f"  {low:>5.0f} dB   {probability:>6.1%}   n={count}")
        return "\n".join(lines)


def _overlaps(report: EmissionReport, truth: TruthEmission, scene_start: float) -> bool:
    """Whether a report overlaps a truth emission in both time and frequency."""
    if abs(report.frequency_hz - truth.frequency_hz) > FREQUENCY_TOLERANCE_HZ:
        return False
    report_start = report.timestamp - scene_start
    report_stop = report_start + report.duration_s
    return report_start < truth.stop_s and report_stop > truth.start_s


def score(
    reports: list[EmissionReport],
    truth: list[TruthEmission],
    scene_duration_s: float,
    scene_start: float | None = None,
) -> Score:
    """Match reports against ground truth and compute the figures.

    Args:
        reports: What the node reported.
        truth: What the scenario transmitted.
        scene_duration_s: Length of the scene, used for the false alarm rate.
        scene_start: Unix time the run began, used to place report timestamps on the
            scenario's timeline. Defaults to the earliest report's timestamp.

    Returns:
        The scored result.
    """
    if scene_start is None:
        scene_start = min((r.timestamp for r in reports), default=0.0)

    matches = [Match(truth=item) for item in truth]
    unmatched = []

    for report in reports:
        hits = [match for match in matches if _overlaps(report, match.truth, scene_start)]
        if not hits:
            unmatched.append(report)
            continue
        # Attribute to the truth emission with the greatest time overlap.
        best = max(
            hits,
            key=lambda m: min(report.timestamp - scene_start + report.duration_s, m.truth.stop_s)
            - max(report.timestamp - scene_start, m.truth.start_s),
        )
        best.reports.append(report)

    merged = sum(
        1
        for report in reports
        if sum(1 for match in matches if _overlaps(report, match.truth, scene_start)) > 1
    )

    result = Score(
        matches=matches,
        spurious=unmatched,
        merged=merged,
        scene_duration_s=scene_duration_s,
    )
    logger.info(
        "metrics: Pd %.1f%%, %d spurious, %d fragmented",
        100.0 * result.probability_of_detection,
        len(result.spurious),
        result.num_fragmented,
    )
    return result
