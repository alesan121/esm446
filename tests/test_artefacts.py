"""Verification of by-product attribution.

The claim being tested is narrow and has to stay narrow: this detection is explained by that
stronger one. The tests that matter are therefore as much about what it declines to explain --
a comparable emitter, one that was not on air at the same time, one whose frequency holds no
relation to anything -- as about what it does.
"""

from __future__ import annotations

import pytest

from esm446.analysis.artefacts import (
    INTERMOD3,
    SPLATTER,
    attribute_products,
)
from esm446.core.node import EmissionReport

BASE_TIME = 1_786_950_000.0

#: The measured carriers of `tests/data/pmr446_two_emitters.cs8`, to the hertz.
CARRIER_A = 446_031_272.0
CARRIER_B = 446_093_731.0


def emission(
    frequency: float,
    power: float = -5.0,
    at: float = 0.0,
    duration: float = 4.0,
    tone: float | None = None,
) -> EmissionReport:
    return EmissionReport(
        timestamp=BASE_TIME + at,
        frequency_hz=frequency,
        pmr_channel=None,
        bin_index=0,
        duration_s=duration,
        peak_power_dbfs=power,
        snr_db=30.0,
        estimated_dbm=None,
        calibrated=False,
        ctcss_tone_hz=tone,
        classification="UNKNOWN",
        offset_s=at,
        peak_deviation_hz=1_300.0,
        gains={},
    )


# --------------------------------------------------------------------------------------
# Splatter: a symmetric pair about one carrier
# --------------------------------------------------------------------------------------


def test_a_symmetric_pair_is_attributed_to_the_carrier_between_them() -> None:
    """The measured case: one handset on channel 8, detections on 5 and 11."""
    carrier = emission(446_093_757.0, power=-0.6)
    reports = [
        carrier,
        emission(446_055_734.0, power=-26.7),
        emission(446_131_991.0, power=-26.1),
    ]
    reports = attribute_products(reports)

    assert reports[0].attribution is None, "the carrier is not a by-product of anything"
    assert [r.attribution for r in reports[1:]] == [SPLATTER, SPLATTER]
    assert all(r.attributed_to_hz == carrier.frequency_hz for r in reports[1:])


def test_symmetry_is_measured_against_the_carrier_not_assumed() -> None:
    """Both measured pairs hold to a few hundred hertz; a kilohertz of slack is plenty."""
    reports = [
        emission(446_093_757.0, power=-0.6),
        emission(446_043_750.0, power=-30.3),
        emission(446_143_750.0, power=-30.2),
    ]
    reports = attribute_products(reports)

    assert all(r.attribution == SPLATTER for r in reports[1:])


def test_a_lone_sideband_is_not_attributed() -> None:
    """One weak detection at an offset is just a weak detection. The pair is the evidence."""
    reports = [
        emission(446_093_757.0, power=-0.6),
        emission(446_055_734.0, power=-26.7),
    ]
    reports = attribute_products(reports)

    assert reports[1].attribution is None


def test_an_asymmetric_pair_is_not_attributed() -> None:
    reports = [
        emission(446_093_757.0, power=-0.6),
        emission(446_055_734.0, power=-26.7),
        emission(446_125_000.0, power=-26.1),
    ]
    reports = attribute_products(reports)

    assert all(r.attribution is None for r in reports)


# --------------------------------------------------------------------------------------
# Intermodulation: two carriers mixing
# --------------------------------------------------------------------------------------


def test_third_order_products_of_two_carriers_are_attributed() -> None:
    """The measured case: handsets 62.5 kHz apart put products at 2*f1 - f2 and 2*f2 - f1."""
    reports = [
        emission(CARRIER_A, power=-4.5, tone=141.3),
        emission(CARRIER_B, power=-7.2, tone=114.8),
        emission(445_968_917.0, power=-26.5, tone=141.3),
        emission(446_156_089.0, power=-29.0, tone=114.8),
    ]
    reports = attribute_products(reports)

    assert reports[2].attribution == INTERMOD3
    assert reports[3].attribution == INTERMOD3


def test_a_product_is_attributed_to_the_carrier_whose_frequency_is_doubled() -> None:
    """Which is also, measurably, the carrier whose sub-audible tone the product carries.

    2*A - B contains twice as much of A as of B, and the recording bears that out: the
    product below A carries A's 141.3 Hz tone, the one above B carries B's 114.8 Hz.
    """
    _, _, below, above = attribute_products(
        [
            emission(CARRIER_A, power=-4.5),
            emission(CARRIER_B, power=-7.2),
            emission(445_968_917.0, power=-26.5, tone=141.3),
            emission(446_156_089.0, power=-29.0, tone=114.8),
        ]
    )

    assert below.attributed_to_hz == CARRIER_A
    assert above.attributed_to_hz == CARRIER_B


def test_a_product_is_not_mistaken_for_splatter() -> None:
    """The trap this has to avoid, and the reason the mirror partner must itself be weak.

    A third-order product satisfies the symmetry test trivially: 2*A - B is the exact mirror
    of B about A. Without requiring the partner to be far below the carrier, every
    intermodulation product would be reported as splatter and the mechanism misdiagnosed.
    """
    *_, product = attribute_products(
        [
            emission(CARRIER_A, power=-4.5),
            emission(CARRIER_B, power=-7.2),
            emission(445_968_917.0, power=-26.5),
        ]
    )

    assert product.attribution == INTERMOD3


# --------------------------------------------------------------------------------------
# What it refuses to explain
# --------------------------------------------------------------------------------------


def test_two_comparable_emitters_are_never_each_others_by_products() -> None:
    """A by-product cannot rival what produced it, whatever arithmetic the frequencies obey."""
    reports = [
        emission(CARRIER_A, power=-4.5),
        emission(CARRIER_B, power=-7.2),
        emission(445_968_917.0, power=-9.0),
    ]
    reports = attribute_products(reports)

    assert all(r.attribution is None for r in reports)


def test_an_emission_at_another_time_is_not_a_by_product() -> None:
    """Splatter exists only while its carrier is on air; that is what makes it splatter."""
    reports = [
        emission(446_093_757.0, power=-0.6, at=0.0, duration=4.0),
        emission(446_055_734.0, power=-26.7, at=100.0),
        emission(446_131_991.0, power=-26.1, at=100.0),
    ]
    reports = attribute_products(reports)

    assert reports[1].attribution is None
    assert reports[2].attribution is None


def test_an_unrelated_weak_emission_is_left_alone() -> None:
    """The honest outcome for a detection nothing explains: it stays an emission."""
    *_, unexplained = attribute_products(
        [
            emission(CARRIER_A, power=-4.5),
            emission(CARRIER_B, power=-7.2),
            emission(446_066_700.0, power=-29.3, duration=0.4),
        ]
    )

    assert unexplained.attribution is None


def test_nothing_is_ever_removed() -> None:
    """Marking, not deleting. Two real emitters can sit symmetrically about a third."""
    reports = [
        emission(446_093_757.0, power=-0.6),
        emission(446_055_734.0, power=-26.7),
        emission(446_131_991.0, power=-26.1),
    ]
    attributed = attribute_products(reports)

    assert len(attributed) == 3
    assert [r.frequency_hz for r in attributed] == [r.frequency_hz for r in reports]
    assert all(r.snr_db == 30.0 for r in attributed), "no other field may be touched"


def test_the_caller_s_records_are_left_alone() -> None:
    """The same detection beside a different set may be judged differently.

    Rewriting the caller's records in place would make a judgement about one set of
    detections look like a property of the measurement itself.
    """
    reports = [
        emission(446_093_757.0, power=-0.6),
        emission(446_055_734.0, power=-26.7),
        emission(446_131_991.0, power=-26.1),
    ]
    attributed = attribute_products(reports)

    assert all(r.attribution is None for r in reports)
    assert any(r.attribution == SPLATTER for r in attributed)


def test_an_empty_input_is_not_an_error() -> None:
    assert attribute_products([]) == []


def test_a_single_emission_explains_nothing() -> None:
    (lone,) = attribute_products([emission(446_093_757.0)])

    assert lone.attribution is None
    assert lone.attributed_to_hz is None


def test_the_level_margin_is_configurable_and_respected() -> None:
    """Tightening the margin must exclude, not include: the parameter has to bite."""
    reports = [
        emission(446_093_757.0, power=-0.6),
        emission(446_055_734.0, power=-26.7),
        emission(446_131_991.0, power=-26.1),
    ]
    reports = attribute_products(reports, level_margin_db=40.0)

    assert all(r.attribution is None for r in reports)


def test_the_frequency_tolerance_is_configurable_and_respected() -> None:
    reports = [
        emission(446_093_757.0, power=-0.6),
        emission(446_055_734.0, power=-26.7),
        emission(446_131_991.0, power=-26.1),
    ]
    reports = attribute_products(reports, tolerance_hz=10.0)

    assert all(r.attribution is None for r in reports), "the pair is 211 Hz from symmetric"


def test_attribution_defaults_to_nothing_claimed() -> None:
    """A report the analysis never saw must not look like one it examined and cleared."""
    fresh = emission(446_093_757.0)

    assert fresh.attribution is None
    assert fresh.attributed_to_hz is None


@pytest.mark.parametrize("kind", [SPLATTER, INTERMOD3])
def test_the_kinds_are_distinct_strings(kind: str) -> None:
    """They end up in an archive and in a report, so they are part of the interface."""
    assert kind.isupper()
    assert SPLATTER != INTERMOD3
