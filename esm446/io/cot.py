"""Cursor-on-Target messages: turning an emission into something a TAK client can draw.

The node's output is metadata. This is where metadata becomes a picture on somebody's screen,
and the whole value of that depends on the picture being honest about what was measured.

Where the emitter is placed, and why there
------------------------------------------
A CoT event needs a position. This system does not know one. A single omnidirectional antenna
measures how strongly a signal arrives and nothing about where it came from, so any point on
a map would be a claim the measurement does not support.

CoT already has the right field for this. ``<point>`` carries ``ce``, the circular error in
metres: the position is the receiver's, and ``ce`` is the radius inside which the emitter
lies. That is exactly the geometry -- range without bearing -- and it renders in TAK as an
accuracy circle rather than as a confident pin.

When there is no range estimate at all, ``ce`` is ``9999999.0``, the CoT convention for
unknown. That is the state of every emission this system currently produces, because absolute
power needs a calibration and there is no attenuator to make one with (#41). A default of zero
would have rendered a pinpoint fix on the receiver.

Types
-----
``a-f-G`` for an emitter whose sub-audible tone matched the pre-shared code, ``a-u-G``
otherwise: affiliation friend or unknown, battle dimension ground. The affiliation is a
statement about a **cooperative identification**, not about intent or hostility, and the
remarks say so -- an ESM node observing a tone cannot tell a friend from somebody who knows
the code.

``u-d-r`` for the uncertainty rings, one per percentile, each linked back to the track it
belongs to so a client can associate them.

Cadence and staleness
---------------------
One event per completed emission, published as it completes. ``start`` is when the emission
began, ``time`` is when the message was formed, and ``stale`` is the end of the emission plus
a hold. The hold exists because an emitter that has stopped transmitting has not stopped
existing; it stops being *current* after a while, and that while is a policy rather than a
measurement. `docs/03_icd_cot.md` is the interface control document for all of this.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

#: CoT schema version this producer emits.
COT_VERSION = "2.0"

#: Prefix of every UID this system produces, so a TAK operator can tell where a track
#: came from and filter on it.
UID_PREFIX = "ESM446"

#: Type of an emitter identified by the pre-shared sub-audible tone: friend, ground.
TYPE_FRIEND = "a-f-G"

#: Type of an emitter that carried no matching tone: unknown, ground. Unknown is the correct
#: affiliation for an emission that was heard and not identified. It is not "hostile".
TYPE_UNKNOWN = "a-u-G"

#: Type of a drawn range ring.
TYPE_RANGE_RING = "u-d-r"

#: ``how`` for a machine-derived position: machine, GPS-derived is wrong here, so
#: ``m-g`` (machine, GIGO/derived) is used for the rings and ``m-c`` for a calculated track.
HOW_CALCULATED = "m-c"
HOW_DERIVED = "m-g"

#: CoT's sentinel for an unknown circular or linear error, in metres.
UNKNOWN_ERROR_M = 9_999_999.0

#: How long after an emission ends its track stays current, in seconds. A policy, not a
#: measurement: an emitter that stopped transmitting has not stopped existing.
DEFAULT_STALE_S = 300.0

#: Percentile whose ring becomes the track's own circular error. The 95 % ring is the one a
#: viewer would reasonably read as "it is in there".
CE_PERCENTILE = 95


def _iso(when: float) -> str:
    """Format a Unix time as the ISO-8601 form CoT expects.

    Args:
        when: Unix time in seconds.

    Returns:
        ``YYYY-MM-DDTHH:MM:SS.ssZ``.
    """
    moment = datetime.fromtimestamp(when, tz=timezone.utc)
    return moment.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-4] + "Z"


@dataclass(frozen=True)
class ReceiverSite:
    """Where the receiver is, which is the only position this system knows.

    Attributes:
        latitude: Degrees north.
        longitude: Degrees east.
        altitude_m: Height above the ellipsoid. CoT's ``hae``.
        callsign: Name the receiver appears under.
    """

    latitude: float = 0.0
    longitude: float = 0.0
    altitude_m: float = 0.0
    callsign: str = "ESM-446"

    def __post_init__(self) -> None:
        if not -90.0 <= self.latitude <= 90.0:
            raise ValueError(f"latitude {self.latitude} is not a latitude")
        if not -180.0 <= self.longitude <= 180.0:
            raise ValueError(f"longitude {self.longitude} is not a longitude")


def emitter_uid(report: Any) -> str:
    """Build a stable UID for an emitter.

    Derived from what identifies the emitter rather than from the emission, so successive
    transmissions from one radio update one track instead of littering the map with a new
    one every time somebody keys up.

    Args:
        report: An `esm446.core.node.EmissionReport`.

    Returns:
        The UID.
    """
    where = f"PMR{report.pmr_channel}" if report.pmr_channel else f"{report.frequency_hz:.0f}"
    tone = f"{report.ctcss_tone_hz:.1f}" if report.ctcss_tone_hz else "notone"
    return f"{UID_PREFIX}.{where}.{tone}"


def _remarks(report: Any, estimate: Any | None) -> str:
    """Compose the human-readable line a TAK operator actually reads.

    Everything a viewer needs to judge the track: what was measured, and what was not.
    """
    parts = [
        f"{report.frequency_hz / 1e6:.5f} MHz",
        f"PMR{report.pmr_channel}" if report.pmr_channel else "off-grid",
        f"{report.duration_s:.1f} s",
        f"SNR {report.snr_db:.1f} dB",
        f"peak {report.peak_power_dbfs:.1f} dBFS",
        f"dev {report.peak_deviation_hz:.0f} Hz",
    ]
    if report.ctcss_tone_hz:
        parts.append(f"CTCSS {report.ctcss_tone_hz:.1f} Hz")
    else:
        parts.append("no CTCSS")

    if report.attribution:
        parts.append(
            f"{report.attribution} of an emission at "
            f"{(report.attributed_to_hz or 0.0) / 1e6:.5f} MHz"
        )

    if estimate is None:
        parts.append("RANGE UNKNOWN: power is uncalibrated")
    elif not estimate.calibrated:
        parts.append(
            f"RANGE UNCALIBRATED: {estimate.median_m:.0f} m from an assumed model, "
            f"not a measurement"
        )
    else:
        parts.append(f"range {estimate.median_m:.0f} m (median)")

    parts.append("bearing not measured: single omnidirectional sensor")
    return " | ".join(parts)


def emitter_track(
    report: Any,
    site: ReceiverSite | None = None,
    estimate: Any | None = None,
    stale_s: float = DEFAULT_STALE_S,
    now: float | None = None,
) -> str:
    """Build the CoT event for one emission.

    Args:
        report: An `esm446.core.node.EmissionReport`.
        site: Where the receiver is. Defaults to the null island origin, which is what an
            unconfigured node knows about its own position.
        estimate: An `esm446.core.geolocation.RangeEstimate`, or ``None`` when no range
            could be computed.
        stale_s: How long after the emission ends the track stays current.
        now: Unix time the message was formed. Defaults to the emission's own end, which
            keeps a replayed recording reproducible.

    Returns:
        The event as XML.
    """
    site = site or ReceiverSite()
    ends = report.timestamp + report.duration_s
    formed = ends if now is None else now

    circular_error = UNKNOWN_ERROR_M
    if estimate is not None and CE_PERCENTILE in estimate.percentiles:
        circular_error = estimate.ring(CE_PERCENTILE)

    attributes = {
        "version": COT_VERSION,
        "uid": emitter_uid(report),
        "type": TYPE_FRIEND if report.classification == "FRIEND" else TYPE_UNKNOWN,
        "how": HOW_CALCULATED,
        "time": _iso(formed),
        "start": _iso(report.timestamp),
        "stale": _iso(ends + stale_s),
    }
    point = _tag(
        "point",
        {
            "lat": f"{site.latitude:.7f}",
            "lon": f"{site.longitude:.7f}",
            "hae": f"{site.altitude_m:.1f}",
            # The position is the receiver's and ce is the radius the emitter lies within.
            # That is the geometry a range-only sensor produces, stated in the field CoT
            # provides for it rather than implied by a pin somebody will misread.
            "ce": f"{circular_error:.1f}",
            "le": f"{UNKNOWN_ERROR_M:.1f}",
        },
    )
    detail = _tag(
        "detail",
        children=[
            _tag("contact", {"callsign": _callsign(report)}),
            _tag("remarks", text=_remarks(report, estimate)),
            _tag(
                "__esm446",
                {
                    "frequency_hz": f"{report.frequency_hz:.1f}",
                    "pmr_channel": str(report.pmr_channel) if report.pmr_channel else "",
                    "snr_db": f"{report.snr_db:.2f}",
                    "peak_power_dbfs": f"{report.peak_power_dbfs:.2f}",
                    "peak_deviation_hz": f"{report.peak_deviation_hz:.1f}",
                    "ctcss_tone_hz": (
                        f"{report.ctcss_tone_hz:.1f}" if report.ctcss_tone_hz else ""
                    ),
                    "duration_s": f"{report.duration_s:.3f}",
                    "calibrated": "true" if report.calibrated else "false",
                    "attribution": report.attribution or "",
                },
            ),
        ],
    )
    return _document(_tag("event", attributes, children=[point, detail]))


def uncertainty_rings(
    report: Any,
    estimate: Any,
    site: ReceiverSite | None = None,
    stale_s: float = DEFAULT_STALE_S,
    now: float | None = None,
) -> list[str]:
    """Build one drawn ring per credible percentile.

    Each ring links back to the track it belongs to, and each carries its own percentile in
    the remarks. A ring on a map with no percentile attached is a number somebody will
    interpret as certainty, so the meaning travels with the drawing.

    Args:
        report: The emission.
        estimate: An `esm446.core.geolocation.RangeEstimate`.
        site: Where the receiver is.
        stale_s: How long the rings stay current after the emission ends.
        now: Unix time the messages were formed.

    Returns:
        One XML event per percentile, in ascending order.
    """
    site = site or ReceiverSite()
    ends = report.timestamp + report.duration_s
    formed = ends if now is None else now
    parent = emitter_uid(report)
    caveat = "" if estimate.calibrated else " UNCALIBRATED: model output, not a measurement."

    events: list[str] = []
    for percentile in sorted(estimate.percentiles):
        radius_m = estimate.ring(percentile)
        point = _tag(
            "point",
            {
                "lat": f"{site.latitude:.7f}",
                "lon": f"{site.longitude:.7f}",
                "hae": f"{site.altitude_m:.1f}",
                "ce": f"{radius_m:.1f}",
                "le": f"{UNKNOWN_ERROR_M:.1f}",
            },
        )
        detail = _tag(
            "detail",
            children=[
                _tag(
                    "remarks",
                    text=(
                        f"{percentile}% credible range for {_callsign(report)}: "
                        f"{radius_m:,.0f} m.{caveat} Range only; bearing not measured."
                    ),
                ),
                # Links the ring to its track so a client can group them, rather than
                # leaving a scatter of unrelated circles on the map.
                _tag(
                    "link",
                    {
                        "uid": parent,
                        "type": TYPE_FRIEND,
                        "relation": "p-p",
                        "production_time": _iso(formed),
                    },
                ),
                _tag(
                    "shape",
                    children=[
                        _tag(
                            "ellipse",
                            {
                                "major": f"{radius_m:.1f}",
                                "minor": f"{radius_m:.1f}",
                                "angle": "0",
                            },
                        )
                    ],
                ),
                # The colour carries the confidence: tighter rings are drawn hotter. The v0
                # code computed a colour map and then emitted a fixed argb of -1, so nothing
                # rendered.
                _tag("color", {"argb": str(_ring_colour(percentile))}),
            ],
        )
        events.append(
            _document(
                _tag(
                    "event",
                    {
                        "version": COT_VERSION,
                        "uid": f"{parent}.p{percentile}",
                        "type": TYPE_RANGE_RING,
                        "how": HOW_DERIVED,
                        "time": _iso(formed),
                        "start": _iso(report.timestamp),
                        "stale": _iso(ends + stale_s),
                    },
                    children=[point, detail],
                )
            )
        )
    return events


def events_for(
    report: Any,
    site: ReceiverSite | None = None,
    estimate: Any | None = None,
    stale_s: float = DEFAULT_STALE_S,
    now: float | None = None,
) -> list[str]:
    """Build every message an emission produces: the track, and its rings if there are any.

    Args:
        report: The emission.
        site: Where the receiver is.
        estimate: The range estimate, or ``None``.
        stale_s: Staleness policy.
        now: Unix time the messages were formed.

    Returns:
        The track first, then any rings.
    """
    events = [emitter_track(report, site, estimate, stale_s, now)]
    if estimate is not None:
        events.extend(uncertainty_rings(report, estimate, site, stale_s, now))
    return events


def _callsign(report: Any) -> str:
    """Readable name for the emitter, matching the label the order of battle uses."""
    where = f"PMR{report.pmr_channel}" if report.pmr_channel else f"{report.frequency_hz / 1e6:.5f}"
    tone = f"{report.ctcss_tone_hz:.1f}Hz" if report.ctcss_tone_hz else "no-tone"
    return f"{where}/{tone}"


#: Ring colours as signed 32-bit ARGB, tightest first. Semi-transparent so overlapping rings
#: stay readable.
_RING_COLOURS = {
    5: 0x8000FF00,
    50: 0x80FFFF00,
    68: 0x80FFA500,
    90: 0x80FF4500,
    95: 0x80FF0000,
}


def _ring_colour(percentile: int) -> int:
    """ARGB for a percentile ring, as the signed integer TAK expects."""
    argb = _RING_COLOURS.get(percentile, 0x80FFFFFF)
    return argb - (1 << 32) if argb >= (1 << 31) else argb


#: The five characters XML gives a meaning to, and what they must be written as. Applied to
#: attribute values and to text alike, which is stricter than the specification requires and
#: is one rule instead of two.
_XML_ESCAPES = (
    ("&", "&amp;"),
    ("<", "&lt;"),
    (">", "&gt;"),
    ('"', "&quot;"),
    ("'", "&apos;"),
)


def _escape(value: str) -> str:
    """Replace the characters XML reserves.

    The ampersand goes first: escaping it after the others would rewrite the ampersands they
    just introduced, and ``<`` would come out as ``&amp;lt;``.

    Args:
        value: Raw text.

    Returns:
        Text safe to place in an attribute or between tags.
    """
    for character, replacement in _XML_ESCAPES:
        value = value.replace(character, replacement)
    return value


def _tag(
    name: str,
    attributes: dict[str, str] | None = None,
    text: str | None = None,
    children: list[str] | None = None,
) -> str:
    """Render one XML element.

    Built, never parsed. This module only ever *writes* XML, and an XML parser in a process
    that has no XML to read is an attack surface for nothing: entity expansion and external
    entities are problems of parsing. What is genuinely needed is correct escaping, because a
    callsign or a remark containing ``&`` or ``<`` would otherwise produce a document the
    consumer cannot read -- so `_escape` does that, and `tests/test_cot.py` checks it by
    parsing the result with a real parser and comparing against the original string.

    Args:
        name: Element name.
        attributes: Attribute names and values. Values are quoted and escaped.
        text: Character content, escaped.
        children: Already-rendered child elements.

    Returns:
        The element as XML.
    """
    rendered = "".join(f' {key}="{_escape(value)}"' for key, value in (attributes or {}).items())
    body = (_escape(text) if text else "") + "".join(children or [])
    if not body:
        return f"<{name}{rendered}/>"
    return f"<{name}{rendered}>{body}</{name}>"


def _document(root: str) -> str:
    """Wrap a rendered element as a standalone document, which is what goes on the wire."""
    return f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n{root}'


def stale_after(report: Any, stale_s: float = DEFAULT_STALE_S) -> float:
    """Unix time at which a track for this emission stops being current.

    Args:
        report: The emission.
        stale_s: The hold applied after the emission ends.

    Returns:
        Unix time.
    """
    return report.timestamp + report.duration_s + stale_s
