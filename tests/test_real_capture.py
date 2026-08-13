"""Verification against a recorded transmission, not a simulated one.

`tests/data/pmr446_ctcss114_8.cs8` is a real PMR446 transmission made by the project's own
operator, so its parameters are ground truth in the strongest sense available: channel 8,
CTCSS 114.8 Hz, handset at minimum power three metres from the receiver.

What this adds over the synthetic scenarios is the transmitter. A simulator has to assume a
deviation, a tone amplitude, a keying shape and a frequency error; a recording has whatever
the radio actually does. Everything the simulator gets wrong about a real handset shows up
here and nowhere else.

Recording somebody else's traffic and committing it would be a different matter entirely --
see `docs/06_legal_ethics.md`. This is the operator's own signal, which is why it can exist.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from esm446.analysis.artefacts import INTERMOD3, SPLATTER, attribute_products
from esm446.analysis.eob import cluster_emitters
from esm446.core import bands
from esm446.core.channelizer import ChannelizerConfig
from esm446.core.node import EsmNode
from esm446.core.rfchain import quantise_gains
from esm446.core.source import FileSource

VECTOR = Path("tests/data/pmr446_ctcss114_8.cs8")
METADATA = Path("tests/data/pmr446_ctcss114_8.json")

pytestmark = pytest.mark.skipif(not VECTOR.exists(), reason="test vector not present")


@pytest.fixture(scope="module")
def metadata() -> dict:
    return json.loads(METADATA.read_text())


@pytest.fixture(scope="module")
def reports(metadata: dict) -> list:
    node = EsmNode(
        channelizer_config=ChannelizerConfig(
            sample_rate=metadata["sample_rate_hz"],
            num_channels=metadata["num_channels"],
            decimation=metadata["num_channels"] // 2,
        ),
        centre_frequency=metadata["centre_hz"],
        gains=quantise_gains(0.0, 0.0),
        expected_ctcss_hz=114.8,
    )
    return node.run(FileSource(VECTOR, metadata["sample_rate_hz"], metadata["centre_hz"], "cs8"))


def strongest(reports: list):
    return max(reports, key=lambda r: r.snr_db)


def test_the_transmission_is_detected(reports: list) -> None:
    assert reports, "nothing detected in a recording containing a 40 dB signal"
    assert strongest(reports).snr_db > 30.0


def test_it_lands_on_channel_8(reports: list) -> None:
    """The handset was set to 446.095; its display resolves 1 kHz and its step is 5 kHz."""
    report = strongest(reports)
    assert report.pmr_channel == 8
    assert report.frequency_hz == pytest.approx(bands.channel_frequency(8), abs=2_000.0)


def test_the_ctcss_tone_is_identified(reports: list) -> None:
    """The full cooperative-identification path, on a tone a real radio generated."""
    report = strongest(reports)
    assert report.ctcss_tone_hz == pytest.approx(114.8)
    assert report.classification == "FRIEND"


def test_the_duration_matches_the_recorded_carrier(reports: list) -> None:
    """The carrier drops about 4.15 s into the extract, and the tracker must find that edge."""
    assert strongest(reports).duration_s == pytest.approx(4.15, abs=0.3)


def test_deviation_is_within_the_etsi_limit(reports: list) -> None:
    """A type-approved-equivalent narrowband emission stays under 2.5 kHz of deviation.

    This is measured on the real signal rather than assumed, which is the one thing a
    recording can settle and a simulator cannot.
    """
    assert 200.0 < strongest(reports).peak_deviation_hz < 2_500.0


def test_power_is_not_reported_in_dbm(reports: list) -> None:
    """No calibration exists, so every absolute figure must stay null."""
    assert all(r.estimated_dbm is None for r in reports)
    assert all(r.calibrated is False for r in reports)


def test_the_vector_is_small_enough_to_commit() -> None:
    """The pre-commit hook rejects anything over 5 MB, and a repository has to stay cloneable."""
    assert VECTOR.stat().st_size < 5 * 1024 * 1024


def test_the_vector_is_not_saturated() -> None:
    """A clipped recording would make every power figure taken from it fiction."""
    raw = np.fromfile(VECTOR, dtype=np.int8, count=2_000_000)
    assert np.abs(raw).max() <= 127


# --------------------------------------------------------------------------------------
# Two simultaneous emitters
# --------------------------------------------------------------------------------------

TWO = Path("tests/data/pmr446_two_emitters.cs8")
TWO_METADATA = Path("tests/data/pmr446_two_emitters.json")


@pytest.fixture(scope="module")
def two_emitter_all_reports() -> list:
    """Everything the node reports on the two-emitter vector, splatter included."""
    if not TWO.exists():
        pytest.skip("two-emitter vector not present")
    metadata = json.loads(TWO_METADATA.read_text())
    node = EsmNode(
        channelizer_config=ChannelizerConfig(
            sample_rate=metadata["sample_rate_hz"],
            num_channels=metadata["num_channels"],
            decimation=metadata["num_channels"] // 2,
        ),
        centre_frequency=metadata["centre_hz"],
        gains=quantise_gains(0.0, 0.0),
        expected_ctcss_hz=114.8,
    )
    return node.run(FileSource(TWO, metadata["sample_rate_hz"], metadata["centre_hz"], "cs8"))


@pytest.fixture(scope="module")
def two_emitter_reports(two_emitter_all_reports: list) -> list:
    """Only the carriers. The weaker reports are the transmitters' own sidebands, #26."""
    return [r for r in two_emitter_all_reports if r.snr_db > 15.0]


def test_both_emitters_are_separated(two_emitter_reports: list) -> None:
    """Two handsets transmitting at once, on channels five apart, recorded together.

    Everything before this was one emitter at a time or a simulation. This is the case the
    channeliser exists for.
    """
    channels = {r.pmr_channel for r in two_emitter_reports}
    assert channels == {3, 8}


def test_neither_emitter_masks_the_other(two_emitter_reports: list) -> None:
    """The reason OS-CFAR is the default rather than CA-CFAR, on real signals.

    A strong emitter inside the reference window inflates a cell-averaged noise estimate and
    hides its neighbour. Both are detected here with comparable SNR, so neither was masked.
    """
    assert len(two_emitter_reports) == 2
    assert all(r.snr_db > 25.0 for r in two_emitter_reports)


def test_they_overlap_in_time(two_emitter_reports: list) -> None:
    """Simultaneity is the point; two emissions in sequence would prove nothing."""
    first, second = sorted(two_emitter_reports, key=lambda r: r.timestamp)
    overlap = min(first.timestamp + first.duration_s, second.timestamp + second.duration_s) - max(
        first.timestamp, second.timestamp
    )
    assert overlap > 2.0


def test_the_tones_distinguish_friend_from_unknown(two_emitter_reports: list) -> None:
    """Cooperative identification with two real emitters carrying different tones.

    The 141.3 Hz tone was identified by the node before the operator confirmed what the
    second handset had been set to, which is the only way this counts as a test rather than a
    check that the answer was known in advance.
    """
    by_channel = {r.pmr_channel: r for r in two_emitter_reports}

    assert by_channel[8].ctcss_tone_hz == pytest.approx(114.8)
    assert by_channel[8].classification == "FRIEND"

    assert by_channel[3].ctcss_tone_hz == pytest.approx(141.3)
    assert by_channel[3].classification == "UNKNOWN"


# --------------------------------------------------------------------------------------
# Order of battle over the recording
# --------------------------------------------------------------------------------------


def test_the_order_of_battle_finds_the_two_carriers(two_emitter_reports: list) -> None:
    """Two radios, two profiles, each carrying the tone that identifies it."""
    profiles = cluster_emitters(two_emitter_reports)

    assert len(profiles) == 2
    by_channel = {p.pmr_channel: p for p in profiles}
    assert set(by_channel) == {3, 8}
    assert by_channel[8].ctcss_tone_hz == pytest.approx(114.8)
    assert by_channel[3].ctcss_tone_hz == pytest.approx(141.3)


def test_without_attribution_two_handsets_become_more_than_two_emitters(
    two_emitter_all_reports: list,
) -> None:
    """The defect #26 describes, kept as a test so the fix below is measured against it.

    Grouped over every detection the node produces, the transmitters' own by-products land on
    neighbouring channels with no recoverable tone and are counted as emitters in their own
    right.
    """
    profiles = cluster_emitters(two_emitter_all_reports)

    assert len(profiles) > 2


def test_attribution_reduces_them_to_the_two_that_transmitted(
    two_emitter_all_reports: list,
) -> None:
    """The same detections, after each by-product is attributed to the carrier that made it.

    One weak detection is left over: 0.36 s at 2.9 dB SNR, 35.4 kHz from channel 3, with no
    symmetric partner and no arithmetic relation to either carrier. Nothing explains it, so
    nothing claims to, and it is counted as what it is.
    """
    profiles = cluster_emitters(attribute_products(list(two_emitter_all_reports)))

    carriers = [p for p in profiles if p.transmission_count and p.median_deviation_hz < 2_000.0]
    assert len(carriers) == 2
    assert {p.pmr_channel for p in carriers} == {3, 8}
    assert sum(len(p.products) for p in profiles) == len(two_emitter_all_reports) - len(profiles)


def test_the_products_are_third_order_intermodulation_of_the_two_carriers(
    two_emitter_all_reports: list,
) -> None:
    """Two carriers 62.5 kHz apart put products at 2*f1 - f2 and 2*f2 - f1, and they are there.

    This is the relation that separates the two-emitter case from the single-handset splatter
    measured in `docs/04_link_budget.md`: the products are not symmetric about either carrier,
    they are symmetric about the pair.
    """
    attributed = attribute_products(list(two_emitter_all_reports))
    products = [r for r in attributed if r.attribution == INTERMOD3]
    assert products, "the vector is expected to contain the two carriers' products"

    carriers = sorted(
        (r for r in attributed if r.attribution is None and r.snr_db > 15.0),
        key=lambda r: r.frequency_hz,
    )
    first, second = carriers[0].frequency_hz, carriers[1].frequency_hz
    predicted = (2 * first - second, 2 * second - first)

    for product in products:
        assert min(abs(product.frequency_hz - p) for p in predicted) < 2_000.0


def test_a_by_product_is_recognisable_by_its_deviation(two_emitter_all_reports: list) -> None:
    """Corroboration from a different measurement than the one used to attribute.

    A discriminator reading taken on a by-product is not a modulation index; it is what the
    discriminator does when handed something that is not an FM carrier, and the figures come
    out several times those of the real transmissions. Attribution does not use this -- it
    uses frequency and simultaneity -- so the agreement between the two is evidence rather
    than tautology.
    """
    attributed = attribute_products(list(two_emitter_all_reports))
    carriers = [r for r in attributed if r.snr_db > 15.0]
    products = [r for r in attributed if r.attribution is not None]
    assert products, "the vector is expected to contain the transmitters' by-products"

    assert max(r.peak_deviation_hz for r in carriers) < 2_000.0
    assert min(r.peak_deviation_hz for r in products) > 3_000.0


def test_the_single_handset_splatter_is_attributed_as_splatter(reports: list) -> None:
    """One transmitter, no second carrier to mix with, so the pairs are its own splatter."""
    attributed = attribute_products(list(reports))
    products = [r for r in attributed if r.attribution is not None]

    assert products
    assert all(r.attribution == SPLATTER for r in products)
    assert len(cluster_emitters(attributed)) == 1


def test_the_two_emitter_vector_is_small_enough_to_commit() -> None:
    if not TWO.exists():
        pytest.skip("two-emitter vector not present")
    assert TWO.stat().st_size < 5 * 1024 * 1024
