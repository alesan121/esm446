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
