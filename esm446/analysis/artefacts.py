"""Attribute a strong emitter's by-products to the emitter that produced them.

A handset transmitting on one channel puts detectable energy on others. Those detections are
real -- the energy is there, the detector is right about it -- but they are not separate
emitters, and counting them as such is what made the order of battle report eleven emitters
for two radios.

The detector cannot see this. CFAR asks one question about one cell, is this above the local
noise floor, and by that test these are detections. What marks them as by-products is
information a single cell does not contain: that they are simultaneous with a much stronger
emission and stand in an exact arithmetic relation to it.

Two relations, both measured
----------------------------
**Splatter**, a pair symmetric about one carrier. One handset on channel 8 produces detections
on channels 5 and 11, at -26 dBc, and a weaker pair a channel further out::

    446 055 734 + 446 131 991 - 2 x 446 093 757 = +211 Hz
    446 043 750 + 446 143 750 - 2 x 446 093 757 =  -14 Hz

Both hold to a few hundred hertz, against a channel spacing of 12.5 kHz. This is the
transmitter's own spectral splatter: the ratio to the carrier stayed within 0.4 dB across
38 dB of receiver level, so nothing in the receiver produces it. See issue #26 for the
measurement that settled that, and `docs/04_link_budget.md` for the profile.

**Third-order intermodulation**, from two carriers at once. With handsets on channels 3 and 8,
62.5 kHz apart, detections appear at ``2*f1 - f2`` and ``2*f2 - f1``::

    2 x 446 031 272 - 446 093 731 = 445 968 813, measured 445 968 917  (+104 Hz)
    2 x 446 093 731 - 446 031 272 = 446 156 190, measured 446 156 089  (-101 Hz)

Each product carries the sub-audible tone of the carrier whose frequency is doubled, which is
what the mixing implies and is a useful confirmation that the attribution is right.

What this does not do
---------------------
It marks; it never deletes. Two real emitters can sit symmetrically about a third, and a node
that silently suppressed them would be hiding emissions rather than explaining them. The
splatter is also a measurement of the transmitter's spectral purity, which is a discriminant
worth keeping -- two units of the same model differ by about 2 dB in it.

So an attributed report keeps every figure it had and gains two fields saying what it is
believed to be a by-product of. Whoever reads it can disagree, and the arithmetic that
produced the claim is right there to disagree with.
"""

from __future__ import annotations

import logging
from dataclasses import replace

from esm446.core.node import EmissionReport

logger = logging.getLogger(__name__)

#: How far a detection may sit from the frequency an arithmetic relation predicts and still be
#: attributed to it. The measured errors are under 250 Hz; a channel is 12.5 kHz wide, so this
#: is loose enough for the frequency estimator's own scatter and far too tight to catch a
#: neighbouring channel by accident.
PRODUCT_TOLERANCE_HZ = 2_000.0

#: How far below its parent a detection must be before it can be called a by-product. A
#: product cannot exceed what produced it, and the measured ones run 22 to 30 dB down. The
#: margin exists to stop two comparable emitters being explained as each other's sidebands.
MIN_LEVEL_MARGIN_DB = 10.0

#: A pair symmetric about a single carrier: the transmitter's own splatter.
SPLATTER = "SPLATTER"

#: A product of two carriers mixing, at ``2*f1 - f2``.
INTERMOD3 = "INTERMOD3"


def _overlaps(first: EmissionReport, second: EmissionReport) -> bool:
    """Whether two emissions were on air at the same time."""
    return (
        first.timestamp < second.timestamp + second.duration_s
        and second.timestamp < first.timestamp + first.duration_s
    )


def attribute_products(
    reports: list[EmissionReport],
    tolerance_hz: float = PRODUCT_TOLERANCE_HZ,
    level_margin_db: float = MIN_LEVEL_MARGIN_DB,
) -> list[EmissionReport]:
    """Mark detections that are by-products of a stronger simultaneous emission.

    Each report is tested against the stronger emissions it overlaps in time. A detection is
    attributed when it sits, within ``tolerance_hz``, at a frequency one of the two measured
    relations predicts. Nothing is removed and no other field is changed.

    The inputs are left alone and copies are returned. Attribution is a judgement about a set
    of detections, and the same detection examined beside a different set may be judged
    differently; quietly rewriting the caller's records would make that judgement look like a
    property of the measurement.

    Args:
        reports: Detections to examine.
        tolerance_hz: How far from the predicted frequency a detection may sit.
        level_margin_db: How far below its parent a detection must be to be attributable.

    Returns:
        Copies of the reports, in the order given, with `EmissionReport.attributed_to_hz` and
        `EmissionReport.attribution` set on those explained.
    """
    reports = [replace(report) for report in reports]
    ordered = sorted(reports, key=lambda r: r.peak_power_dbfs, reverse=True)

    for index, weak in enumerate(ordered):
        # Only stronger emissions on air at the same time can have produced this one.
        parents = [
            strong
            for strong in ordered[:index]
            if _overlaps(strong, weak)
            and strong.peak_power_dbfs - weak.peak_power_dbfs >= level_margin_db
        ]
        if not parents:
            continue

        attribution = _match_splatter(weak, parents, ordered, tolerance_hz, level_margin_db)
        if attribution is None:
            attribution = _match_intermod(weak, parents, tolerance_hz)
        if attribution is None:
            continue

        parent, kind = attribution
        weak.attributed_to_hz = parent.frequency_hz
        weak.attribution = kind

    attributed = [r for r in reports if r.attribution is not None]
    if attributed:
        logger.info(
            "artefacts: %d of %d detections attributed to a stronger emission (%d splatter, "
            "%d intermodulation)",
            len(attributed),
            len(reports),
            sum(1 for r in attributed if r.attribution == SPLATTER),
            sum(1 for r in attributed if r.attribution == INTERMOD3),
        )
    return reports


def _match_splatter(
    weak: EmissionReport,
    parents: list[EmissionReport],
    everything: list[EmissionReport],
    tolerance_hz: float,
    level_margin_db: float,
) -> tuple[EmissionReport, str] | None:
    """Find a parent this detection is a symmetric sideband of.

    The evidence is the pair: a sideband on its own is just a weak detection at an offset,
    while two equidistant either side of a carrier, both far below it and both on air with it,
    is the signature of a transmitter splattering.

    The mirror partner has to be weak as well. Without that condition the third-order product
    of two carriers matches this test trivially -- one carrier is the exact mirror of the
    product about the other -- and the mechanism would be misreported.
    """
    for parent in parents:
        mirror_hz = 2.0 * parent.frequency_hz - weak.frequency_hz
        for other in everything:
            if other is weak or other is parent:
                continue
            if abs(other.frequency_hz - mirror_hz) > tolerance_hz:
                continue
            if parent.peak_power_dbfs - other.peak_power_dbfs < level_margin_db:
                continue
            if _overlaps(other, parent):
                return parent, SPLATTER
    return None


def _match_intermod(
    weak: EmissionReport,
    parents: list[EmissionReport],
    tolerance_hz: float,
) -> tuple[EmissionReport, str] | None:
    """Find two parents whose third-order product lands on this detection.

    Attribution goes to the carrier whose frequency is doubled, which is the one that
    contributes twice as much to the product and, measurably, the one whose sub-audible tone
    the product carries.
    """
    for first in parents:
        for second in parents:
            if first is second:
                continue
            predicted_hz = 2.0 * first.frequency_hz - second.frequency_hz
            if abs(weak.frequency_hz - predicted_hz) <= tolerance_hz:
                return first, INTERMOD3
    return None
