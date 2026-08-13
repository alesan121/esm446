"""Electronic Order of Battle: what is on the band, when, and how many of them.

Individual detections are not intelligence. A list of frequencies and times is what a scanner
produces. What makes it an order of battle is aggregation — which channels are used, at what
hours, by how many distinct emitters, with what regularity — and that is the product this
whole system exists to make.

Grouping is inference, and says so
----------------------------------
Occupancy and burst statistics are arithmetic over measurements. Emitter grouping is not: it
is a guess about how many radios produced a set of emissions, made from the features a
receiver can observe.

The honest limit is sharp and worth stating plainly. **Two emitters that never transmit at
the same time cannot be distinguished from one emitter that paused.** No amount of feature
engineering escapes that, because the evidence needed does not exist in the recording. So
a cluster reports its count as a lower bound, and marks itself `proven_multiple` only when
two of its emissions actually overlap in time — which is the one observation that forces
more than one transmitter.

Features used, and why those
----------------------------
- **Frequency**, to within a fraction of a channel. The strongest discriminant, and the one
  the channeliser measures best: emissions from a single handset land within tens of hertz
  across a session.
- **CTCSS tone.** Decisive when two emitters differ, useless when they agree — and they did
  agree in the hardware session that motivated this, which is precisely why it cannot be the
  only feature.
- **Peak deviation.** How hard the transmitter drives its modulator. Varies between units of
  the same model.
- **Received power.** Weak evidence on its own, since it tracks distance rather than
  identity, but a tight power distribution across a session suggests a stationary emitter.

What is deliberately not used is the spurious signature at +/-37.5 kHz measured in
`docs/04_link_budget.md`, which distinguishes the two handsets by about 2 dB. It is the
strongest identity feature found so far and it needs the detector to attribute sidebands to
their emitter first, which is issue #26.

Known overcount on real captures
--------------------------------
Run over the two committed hardware vectors, this reports **eleven** emitters where two
radios transmitted. Every extra one is that same unattributed splatter: the sidebands land on
neighbouring channels and off the grid, arrive with no recoverable sub-audible tone, and are
therefore grouped as emitters in their own right. Their deviation figures give them away --
5 to 12 kHz against the 1.1 to 1.3 kHz of the real transmissions, because a sideband is not
an FM carrier and the discriminator reading is meaningless for it.

The count is a lower bound in the direction stated above and, until #26 attributes sidebands
to the emitter that produced them, an over-count in the other. Filtering here on deviation
would hide the symptom and lose the measurement; the fix belongs in the detector.
"""

from __future__ import annotations

import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from esm446.core.node import EmissionReport

logger = logging.getLogger(__name__)

#: How far apart two emissions may be in frequency and still be attributed to one emitter.
#: A handset drifts by a few hundred hertz across a session; half a channel would let two
#: genuinely different channels merge.
FREQUENCY_TOLERANCE_HZ = 3_000.0

#: How far apart two CTCSS tones may be and still count as the same code. The tightest
#: spacing in the standard table is 3.5 Hz, so anything looser would merge adjacent codes.
TONE_TOLERANCE_HZ = 1.0


def _median(values: list[float]) -> float:
    """Median of a list, zero when empty."""
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def _spread(values: list[float]) -> float:
    """Interquartile range, used instead of standard deviation.

    A session usually contains a handful of transmissions, and one anomalous burst moves a
    standard deviation far more than it should. The interquartile range describes the bulk of
    the distribution, which is what a spread is being asked for here.
    """
    if len(values) < 2:
        return 0.0
    ordered = sorted(values)
    lower = ordered[len(ordered) // 4]
    upper = ordered[(3 * len(ordered)) // 4 - (0 if len(ordered) % 4 else 1)]
    return abs(upper - lower)


@dataclass
class EmitterProfile:
    """One inferred emitter and everything observed about it.

    Attributes:
        label: Human-readable identifier, derived from the channel and tone.
        frequency_hz: Median frequency across the emissions attributed to it.
        pmr_channel: PMR446 channel, or ``None`` when off-grid.
        ctcss_tone_hz: Sub-audible tone, or ``None`` when none was identified.
        emissions: The emissions attributed to this emitter.
        proven_multiple: Whether two of those emissions overlap in time, which is the only
            observation that forces more than one transmitter behind them.
    """

    label: str
    frequency_hz: float
    pmr_channel: int | None
    ctcss_tone_hz: float | None
    emissions: list[EmissionReport] = field(default_factory=list)
    proven_multiple: bool = False

    @property
    def transmission_count(self) -> int:
        """Number of emissions attributed to this emitter."""
        return len(self.emissions)

    @property
    def total_airtime_s(self) -> float:
        """Total time this emitter held a carrier up."""
        return sum(e.duration_s for e in self.emissions)

    @property
    def first_seen(self) -> float:
        """Unix time of the earliest emission."""
        return min(e.timestamp for e in self.emissions)

    @property
    def last_seen(self) -> float:
        """Unix time at which the latest emission ended."""
        return max(e.timestamp + e.duration_s for e in self.emissions)

    @property
    def duty_cycle(self) -> float:
        """Fraction of its own active span this emitter was transmitting.

        The span runs from its first transmission to the end of its last, not the length of
        the whole capture, so an emitter that appears for a minute of an hour is described by
        what it did in that minute. The consequence is that a single transmission always
        gives 1.0, which is arithmetic rather than information; `describe` omits the figure in
        that case rather than printing a hundred per cent that means nothing.
        """
        window = self.last_seen - self.first_seen
        return self.total_airtime_s / window if window > 0 else 0.0

    @property
    def median_duration_s(self) -> float:
        """Median transmission length. A talker, a repeater and a data link differ here."""
        return _median([e.duration_s for e in self.emissions])

    @property
    def median_gap_s(self) -> float:
        """Median silence between consecutive transmissions."""
        ordered = sorted(self.emissions, key=lambda e: e.timestamp)
        gaps = [
            later.timestamp - (earlier.timestamp + earlier.duration_s)
            for earlier, later in zip(ordered[:-1], ordered[1:], strict=True)
        ]
        return _median([g for g in gaps if g >= 0])

    @property
    def median_power_dbfs(self) -> float:
        """Median received power. Tracks distance rather than identity."""
        return _median([e.peak_power_dbfs for e in self.emissions])

    @property
    def power_spread_db(self) -> float:
        """Spread of received power. A tight spread suggests a stationary emitter."""
        return _spread([e.peak_power_dbfs for e in self.emissions])

    @property
    def median_deviation_hz(self) -> float:
        """Median peak FM deviation. How hard this transmitter drives its modulator."""
        return _median([e.peak_deviation_hz for e in self.emissions])

    @property
    def frequency_spread_hz(self) -> float:
        """Spread of measured frequency, which bounds the emitter's stability."""
        return _spread([e.frequency_hz for e in self.emissions])

    @property
    def count_is_lower_bound(self) -> bool:
        """Whether this profile might conceal more than one transmitter.

        True unless two of its emissions overlap. A cluster of non-overlapping emissions is
        consistent with one radio taking turns and with several radios sharing a channel and
        a tone, and the recording contains nothing that separates those.
        """
        return not self.proven_multiple

    def as_dict(self) -> dict[str, Any]:
        """Return the profile's summary figures."""
        return {
            "label": self.label,
            "frequency_hz": round(self.frequency_hz, 1),
            "pmr_channel": self.pmr_channel,
            "ctcss_tone_hz": self.ctcss_tone_hz,
            "transmissions": self.transmission_count,
            "total_airtime_s": round(self.total_airtime_s, 2),
            "duty_cycle": round(self.duty_cycle, 4),
            "median_duration_s": round(self.median_duration_s, 2),
            "median_gap_s": round(self.median_gap_s, 2),
            "median_power_dbfs": round(self.median_power_dbfs, 1),
            "power_spread_db": round(self.power_spread_db, 1),
            "median_deviation_hz": round(self.median_deviation_hz, 1),
            "frequency_spread_hz": round(self.frequency_spread_hz, 1),
            "proven_multiple": self.proven_multiple,
            "count_is_lower_bound": self.count_is_lower_bound,
        }


def _overlaps(first: EmissionReport, second: EmissionReport) -> bool:
    """Whether two emissions were on air at the same time."""
    return (
        first.timestamp < second.timestamp + second.duration_s
        and second.timestamp < first.timestamp + first.duration_s
    )


def cluster_emitters(
    reports: list[EmissionReport],
    frequency_tolerance_hz: float = FREQUENCY_TOLERANCE_HZ,
) -> list[EmitterProfile]:
    """Group emissions into inferred emitters.

    Grouping is by frequency and sub-audible tone, which are the two features a receiver
    measures well and which a single radio holds constant. Power and deviation are recorded
    on the profile rather than used to split, because both vary with things other than
    identity — power with distance, deviation with how loudly somebody speaks.

    Args:
        reports: Emissions to group.
        frequency_tolerance_hz: How far apart two emissions may be and still be attributed
            to one emitter.

    Returns:
        Profiles ordered by total airtime, busiest first.
    """
    profiles: list[EmitterProfile] = []

    for report in sorted(reports, key=lambda r: r.timestamp):
        for profile in profiles:
            same_frequency = abs(report.frequency_hz - profile.frequency_hz) <= (
                frequency_tolerance_hz
            )
            same_tone = _tones_match(report.ctcss_tone_hz, profile.ctcss_tone_hz)
            if same_frequency and same_tone:
                if any(_overlaps(report, other) for other in profile.emissions):
                    profile.proven_multiple = True
                profile.emissions.append(report)
                profile.frequency_hz = _median([e.frequency_hz for e in profile.emissions])
                break
        else:
            profiles.append(
                EmitterProfile(
                    label=_label(report),
                    frequency_hz=report.frequency_hz,
                    pmr_channel=report.pmr_channel,
                    ctcss_tone_hz=report.ctcss_tone_hz,
                    emissions=[report],
                )
            )

    profiles.sort(key=lambda p: p.total_airtime_s, reverse=True)
    logger.info(
        "eob: %d emissions grouped into %d emitters, %d of which are proven multiple",
        len(reports),
        len(profiles),
        sum(1 for p in profiles if p.proven_multiple),
    )
    return profiles


def _tones_match(first: float | None, second: float | None) -> bool:
    """Whether two CTCSS readings are the same code, treating absence as its own value."""
    if first is None or second is None:
        return first is None and second is None
    return abs(first - second) <= TONE_TOLERANCE_HZ


def _label(report: EmissionReport) -> str:
    """Readable name for an emitter, from what identifies it."""
    where = f"PMR{report.pmr_channel}" if report.pmr_channel else f"{report.frequency_hz / 1e6:.5f}"
    tone = f"{report.ctcss_tone_hz:.1f}Hz" if report.ctcss_tone_hz else "no-tone"
    return f"{where}/{tone}"


@dataclass
class Occupancy:
    """Channel activity across the hours of the day.

    Attributes:
        airtime_s: Seconds of carrier per ``(channel, hour)``. Channel ``None`` collects
            off-grid emissions, which are counted rather than discarded because finding them
            is a large part of why the band is surveyed.
        counts: Emissions per ``(channel, hour)``.
        window_s: Length of the observed period.
    """

    airtime_s: dict[tuple[int | None, int], float] = field(default_factory=dict)
    counts: dict[tuple[int | None, int], int] = field(default_factory=dict)
    window_s: float = 0.0

    def busiest_channels(self, limit: int = 5) -> list[tuple[int | None, float]]:
        """Channels with the most airtime, busiest first."""
        totals: dict[int | None, float] = defaultdict(float)
        for (channel, _), seconds in self.airtime_s.items():
            totals[channel] += seconds
        return sorted(totals.items(), key=lambda item: item[1], reverse=True)[:limit]

    def busiest_hours(self, limit: int = 5) -> list[tuple[int, float]]:
        """Hours of the day with the most airtime.

        This is the pattern-of-life view: a working band has a shape over the day, and a
        band that does not is either unused or being watched over too short a window.
        """
        totals: dict[int, float] = defaultdict(float)
        for (_, hour), seconds in self.airtime_s.items():
            totals[hour] += seconds
        return sorted(totals.items(), key=lambda item: item[1], reverse=True)[:limit]

    @property
    def total_airtime_s(self) -> float:
        """Total carrier time observed across all channels."""
        return sum(self.airtime_s.values())

    @property
    def band_duty_cycle(self) -> float:
        """Fraction of the observed window with at least something transmitting.

        Overlapping emissions are counted once each, so this exceeds 1.0 on a band with
        simultaneous users. That is the intended reading: it measures load, not the
        probability the band is busy.
        """
        return self.total_airtime_s / self.window_s if self.window_s > 0 else 0.0


def compute_occupancy(reports: list[EmissionReport]) -> Occupancy:
    """Bin emissions by channel and by hour of day.

    Hours come from the emission's timestamp in UTC. That is only meaningful because
    timestamps are the capture's own time rather than the analysis time — see issue #37,
    where they were not, and this would have binned by when somebody chose to run the
    analysis.

    Args:
        reports: Emissions to bin.

    Returns:
        The occupancy matrix and the window it covers.
    """
    occupancy = Occupancy()
    if not reports:
        return occupancy

    for report in reports:
        hour = datetime.fromtimestamp(report.timestamp, tz=timezone.utc).hour
        key = (report.pmr_channel, hour)
        occupancy.airtime_s[key] = occupancy.airtime_s.get(key, 0.0) + report.duration_s
        occupancy.counts[key] = occupancy.counts.get(key, 0) + 1

    start = min(r.timestamp for r in reports)
    end = max(r.timestamp + r.duration_s for r in reports)
    occupancy.window_s = end - start
    return occupancy


def describe(reports: list[EmissionReport]) -> str:
    """Render an order of battle as text, for the command line and the V&V report.

    Args:
        reports: Emissions to summarise.

    Returns:
        A human-readable report.
    """
    if not reports:
        return "no emissions"

    occupancy = compute_occupancy(reports)
    profiles = cluster_emitters(reports)

    lines = [
        f"{len(reports)} emissions over {occupancy.window_s / 60:.1f} minutes",
        f"{occupancy.total_airtime_s:.1f} s of carrier, band load {occupancy.band_duty_cycle:.1%}",
        "",
        "Emitters, busiest first:",
        f"  {'label':<20} {'n':>3} {'airtime':>8} {'duty':>6} {'median':>7} {'dev Hz':>7} {'±Hz':>6}",
    ]
    for profile in profiles:
        marker = "" if profile.proven_multiple else "  (>= 1)"
        # A duty cycle over a single transmission is 1.0 by construction, so it is left blank
        # rather than printed as a hundred per cent somebody might read as a busy emitter.
        duty = f"{profile.duty_cycle:.1%}" if profile.transmission_count > 1 else "-"
        lines.append(
            f"  {profile.label:<20} {profile.transmission_count:>3} "
            f"{profile.total_airtime_s:>7.1f}s {duty:>6} "
            f"{profile.median_duration_s:>6.1f}s {profile.median_deviation_hz:>7.0f} "
            f"{profile.frequency_spread_hz:>6.0f}{marker}"
        )

    lines.append("")
    lines.append("Busiest channels:")
    for channel, seconds in occupancy.busiest_channels():
        name = f"PMR{channel}" if channel else "off-grid"
        lines.append(f"  {name:<10} {seconds:>7.1f} s")

    hours = occupancy.busiest_hours()
    if len(hours) > 1:
        lines.append("")
        lines.append("Busiest hours (UTC):")
        for hour, seconds in hours:
            lines.append(f"  {hour:02d}:00      {seconds:>7.1f} s")

    lines.append("")
    lines.append(
        "Emitter counts marked (>= 1) are lower bounds: none of that group's emissions\n"
        "overlap in time, and a recording cannot separate one radio taking turns from\n"
        "several sharing a channel and a tone."
    )
    return "\n".join(lines)


def summarise(reports: list[EmissionReport]) -> dict[str, Any]:
    """Return the order of battle as a dictionary, for JSON output and for tests."""
    occupancy = compute_occupancy(reports)
    profiles = cluster_emitters(reports)
    return {
        "emissions": len(reports),
        "window_s": round(occupancy.window_s, 2),
        "total_airtime_s": round(occupancy.total_airtime_s, 2),
        "band_duty_cycle": round(occupancy.band_duty_cycle, 4),
        "emitters": [p.as_dict() for p in profiles],
        "busiest_channels": [
            {"pmr_channel": c, "airtime_s": round(s, 2)} for c, s in occupancy.busiest_channels()
        ],
        "generated_at": math.floor(time.time()),
    }
