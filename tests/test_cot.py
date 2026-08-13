"""Verification of the Cursor-on-Target messages.

The interface is what somebody else's system depends on, so the tests here are mostly about
the promises `docs/03_icd_cot.md` makes: the schema, the units, the stale policy, and above
all the two claims the message must never make -- that it knows where the emitter is, and
that a sub-audible tone is an identification.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import pytest

from esm446.core.geolocation import estimate_range
from esm446.core.node import EmissionReport
from esm446.io.cot import (
    TYPE_FRIEND,
    TYPE_RANGE_RING,
    TYPE_UNKNOWN,
    UNKNOWN_ERROR_M,
    ReceiverSite,
    emitter_track,
    emitter_uid,
    _tag,
    events_for,
    uncertainty_rings,
)

SCHEMA = Path("schemas/cot-event.xsd")
#: 2026-08-17 07:00:00 UTC, a fixed instant so the messages are reproducible.
BASE_TIME = 1_786_950_000.0
MADRID = ReceiverSite(latitude=40.4168, longitude=-3.7038, altitude_m=650.0)


def emission(
    classification: str = "FRIEND",
    tone: float | None = 114.8,
    channel: int | None = 8,
    at: float = 0.0,
    duration: float = 4.2,
    estimated_dbm: float | None = None,
    calibrated: bool = False,
    attribution: str | None = None,
) -> EmissionReport:
    return EmissionReport(
        timestamp=BASE_TIME + at,
        frequency_hz=446_093_757.0,
        pmr_channel=channel,
        bin_index=9,
        duration_s=duration,
        peak_power_dbfs=-0.6,
        snr_db=40.9,
        estimated_dbm=estimated_dbm,
        calibrated=calibrated,
        ctcss_tone_hz=tone,
        classification=classification,
        offset_s=at,
        peak_deviation_hz=1_347.0,
        gains={"lna_db": 0.0, "vga_db": 0.0, "amp_enabled": False},
        attributed_to_hz=446_093_757.0 if attribution else None,
        attribution=attribution,
    )


def parse(event_xml: str) -> ET.Element:
    return ET.fromstring(event_xml)


def point(event_xml: str) -> dict[str, str]:
    return parse(event_xml).find("point").attrib


def remarks(event_xml: str) -> str:
    return parse(event_xml).find("detail/remarks").text


def stale_of(event_xml: str) -> float:
    """Unix time of an event's stale attribute, parsed independently of the producer."""
    stale = parse(event_xml).get("stale")
    moment = datetime.strptime(stale, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)
    return moment.timestamp()


# --------------------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def schema():
    xmlschema = pytest.importorskip("xmlschema")
    return xmlschema.XMLSchema(str(SCHEMA))


def test_every_event_validates_against_the_schema(schema) -> None:
    """The ICD claims schema validity, so the claim is checked rather than asserted."""
    estimate = estimate_range(-95.0, 446_093_757.0, calibrated=True, draws=1_000, seed=1)
    events = events_for(emission(), MADRID, estimate)

    assert len(events) == 6, "one track and five rings"
    for event in events:
        schema.validate(event)


def test_an_event_without_a_range_still_validates(schema) -> None:
    schema.validate(emitter_track(emission(), MADRID))


# --------------------------------------------------------------------------------------
# Geometry: the claim the message must not make
# --------------------------------------------------------------------------------------


def test_the_position_is_the_receivers_not_the_emitters() -> None:
    """The only position this system knows. Anything else would be an invented measurement."""
    attributes = point(emitter_track(emission(), MADRID))

    assert float(attributes["lat"]) == pytest.approx(MADRID.latitude)
    assert float(attributes["lon"]) == pytest.approx(MADRID.longitude)


def test_an_uncalibrated_emission_carries_an_unknown_circular_error() -> None:
    """The state of every emission this system currently produces.

    Zero would have rendered a pinpoint fix on the receiver, which is the worst available
    answer: precise and wrong. 9999999.0 is CoT's sentinel for unknown.
    """
    attributes = point(emitter_track(emission(), MADRID))

    assert float(attributes["ce"]) == UNKNOWN_ERROR_M
    assert "RANGE UNKNOWN" in remarks(emitter_track(emission(), MADRID))


def test_a_calibrated_estimate_becomes_the_circular_error() -> None:
    estimate = estimate_range(-95.0, 446_093_757.0, calibrated=True, draws=4_000, seed=1)
    attributes = point(emitter_track(emission(), MADRID, estimate))

    # ce is serialised to a tenth of a metre, which is far finer than the estimate.
    assert float(attributes["ce"]) == pytest.approx(estimate.ring(95), abs=0.05)


def test_height_is_never_claimed() -> None:
    """Nothing in the chain estimates altitude, so the linear error stays unknown."""
    assert float(point(emitter_track(emission(), MADRID))["le"]) == UNKNOWN_ERROR_M


def test_the_remarks_say_bearing_was_not_measured() -> None:
    """The caveat has to reach the person looking at the screen, not just the ICD."""
    assert "bearing not measured" in remarks(emitter_track(emission(), MADRID))


# --------------------------------------------------------------------------------------
# Identification
# --------------------------------------------------------------------------------------


def test_a_matched_tone_produces_a_friendly_track() -> None:
    assert parse(emitter_track(emission(classification="FRIEND"))).get("type") == TYPE_FRIEND


def test_anything_else_is_unknown_never_hostile() -> None:
    """An ESM node observes emissions. It has no basis for declaring intent."""
    event = parse(emitter_track(emission(classification="UNKNOWN")))

    assert event.get("type") == TYPE_UNKNOWN
    assert "h" not in event.get("type").split("-")[1], "affiliation must never be hostile"


def test_two_transmissions_from_one_emitter_share_a_uid() -> None:
    """Otherwise every over from one radio litters the map with a new pin."""
    first = emission(at=0.0)
    second = emission(at=600.0)

    assert emitter_uid(first) == emitter_uid(second)
    assert parse(emitter_track(first)).get("uid") == parse(emitter_track(second)).get("uid")


def test_different_tones_on_one_channel_are_different_tracks() -> None:
    assert emitter_uid(emission(tone=114.8)) != emitter_uid(emission(tone=141.3))


def test_an_off_grid_emitter_gets_a_uid_from_its_frequency() -> None:
    """Nothing obliges a transmitter to sit on the channel plan, and the UID must not assume."""
    uid = emitter_uid(emission(channel=None))

    assert "446093757" in uid
    assert "PMRNone" not in uid


# --------------------------------------------------------------------------------------
# Time
# --------------------------------------------------------------------------------------


def test_start_is_when_the_emission_began() -> None:
    event = parse(emitter_track(emission(at=0.0, duration=4.2), MADRID))

    assert event.get("start").startswith("2026-08-17T")
    assert event.get("start").endswith("Z")


def test_stale_is_the_end_of_the_emission_plus_the_hold() -> None:
    """The policy the ICD states, checked as arithmetic rather than trusted."""
    stale = stale_of(emitter_track(emission(at=0.0, duration=4.2), MADRID, stale_s=300.0))
    expected = BASE_TIME + 4.2 + 300.0

    assert stale == pytest.approx(expected, abs=0.01)


def test_the_message_time_defaults_to_the_emission_not_the_clock() -> None:
    """A replayed recording must produce the same bytes every time it is replayed."""
    first = emitter_track(emission(), MADRID)
    second = emitter_track(emission(), MADRID)

    assert first == second


# --------------------------------------------------------------------------------------
# Rings
# --------------------------------------------------------------------------------------


def test_no_estimate_means_no_rings() -> None:
    """A ring nobody measured is worse than no ring."""
    assert len(events_for(emission(), MADRID, None)) == 1


def test_every_ring_names_its_percentile() -> None:
    """A circle on a map with no percentile is read as certainty."""
    estimate = estimate_range(-95.0, 446_093_757.0, calibrated=True, draws=1_000, seed=1)
    rings = uncertainty_rings(emission(), estimate, MADRID)

    for percentile in sorted(estimate.percentiles):
        assert any(f"{percentile}% credible range" in remarks(r) for r in rings)


def test_rings_are_typed_as_drawings_and_link_to_their_track() -> None:
    estimate = estimate_range(-95.0, 446_093_757.0, calibrated=True, draws=1_000, seed=1)
    ring = uncertainty_rings(emission(), estimate, MADRID)[0]

    assert parse(ring).get("type") == TYPE_RANGE_RING
    assert parse(ring).find("detail/link").get("uid") == emitter_uid(emission())


def test_ring_radii_increase_with_the_percentile() -> None:
    estimate = estimate_range(-95.0, 446_093_757.0, calibrated=True, draws=4_000, seed=1)
    radii = [float(point(r)["ce"]) for r in uncertainty_rings(emission(), estimate, MADRID)]

    assert radii == sorted(radii)


def test_each_ring_carries_a_colour() -> None:
    """v0 computed a colour map and then emitted a fixed argb of -1, so nothing rendered."""
    estimate = estimate_range(-95.0, 446_093_757.0, calibrated=True, draws=1_000, seed=1)
    colours = {
        parse(r).find("detail/color").get("argb")
        for r in uncertainty_rings(emission(), estimate, MADRID)
    }

    assert len(colours) > 1, "every ring the same colour is the v0 defect"


def test_uncalibrated_rings_say_so_on_every_one() -> None:
    """A viewer may only ever see one ring, so the caveat cannot live on just the first."""
    estimate = estimate_range(-95.0, 446_093_757.0, calibrated=False, draws=1_000, seed=1)

    for ring in uncertainty_rings(emission(), estimate, MADRID):
        assert "UNCALIBRATED" in remarks(ring)


# --------------------------------------------------------------------------------------
# Measurement carried machine-readably
# --------------------------------------------------------------------------------------


def test_the_measurement_travels_in_a_structured_element() -> None:
    """Remarks are for people. A consumer that wants the numbers should not parse prose."""
    detail = parse(emitter_track(emission(), MADRID)).find("detail/__esm446")

    assert float(detail.get("frequency_hz")) == pytest.approx(446_093_757.0)
    assert detail.get("pmr_channel") == "8"
    assert float(detail.get("peak_deviation_hz")) == pytest.approx(1_347.0)
    assert detail.get("calibrated") == "false"


def test_an_attributed_by_product_says_what_it_is() -> None:
    """A consumer must not count a splatter sideband as an independent emitter."""
    event = emitter_track(emission(attribution="SPLATTER"), MADRID)

    assert "SPLATTER" in remarks(event)
    assert parse(event).find("detail/__esm446").get("attribution") == "SPLATTER"


def test_the_callsign_matches_the_order_of_battle_label() -> None:
    """One name for an emitter across the whole system, or two reports cannot be reconciled."""
    contact = parse(emitter_track(emission(), MADRID)).find("detail/contact")

    assert contact.get("callsign") == "PMR8/114.8Hz"


# --------------------------------------------------------------------------------------
# The site
# --------------------------------------------------------------------------------------


def test_an_impossible_latitude_is_refused() -> None:
    with pytest.raises(ValueError, match="latitude"):
        ReceiverSite(latitude=91.0)


def test_an_impossible_longitude_is_refused() -> None:
    with pytest.raises(ValueError, match="longitude"):
        ReceiverSite(longitude=181.0)


# --------------------------------------------------------------------------------------
# Escaping
# --------------------------------------------------------------------------------------


def test_reserved_characters_survive_a_round_trip() -> None:
    """The producer writes XML without a parser, so escaping is checked with a real one.

    A callsign or a remark carrying ``&`` or ``<`` would otherwise produce a document the
    consumer cannot read -- and the failure would appear at the far end, in somebody else's
    system, which is the worst place to find it.
    """
    hostile = "A&B <tag> \"quoted\" 'single'"
    site = ReceiverSite(latitude=40.0, longitude=-3.0, callsign=hostile)
    event = _tag("event", {"uid": hostile}, children=[_tag("remarks", text=hostile)])

    parsed = ET.fromstring(event)
    assert parsed.get("uid") == hostile
    assert parsed.find("remarks").text == hostile
    assert site.callsign == hostile


def test_an_ampersand_is_not_escaped_twice() -> None:
    """The ordering trap: escaping & after < turns &lt; into &amp;lt;."""
    parsed = ET.fromstring(_tag("remarks", text="a < b & c"))

    assert parsed.text == "a < b & c"
