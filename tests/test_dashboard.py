"""Verification of the band picture and the dashboard.

These are what somebody looks at before deciding whether to read anything else, so the tests
are about them being correct and self-contained rather than about them being pretty.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from esm446.core.node import EmissionReport
from esm446.dashboard import dashboard, waterfall

BASE_TIME = 1_786_950_000.0


def emission(
    channel: int | None = 8,
    frequency: float = 446_093_750.0,
    tone: float | None = 114.8,
    at: float = 1.0,
    duration: float = 2.0,
    attribution: str | None = None,
) -> EmissionReport:
    return EmissionReport(
        timestamp=BASE_TIME + at,
        frequency_hz=frequency,
        pmr_channel=channel,
        bin_index=9,
        duration_s=duration,
        peak_power_dbfs=-22.0,
        snr_db=40.0,
        estimated_dbm=None,
        calibrated=False,
        ctcss_tone_hz=tone,
        classification="FRIEND",
        offset_s=at,
        peak_deviation_hz=1_350.0,
        gains={},
        attributed_to_hz=446_093_750.0 if attribution else None,
        attribution=attribution,
    )


# --------------------------------------------------------------------------------------
# The dashboard
# --------------------------------------------------------------------------------------


def test_the_page_is_self_contained() -> None:
    """It has to survive being emailed, and it must not be able to phone anywhere.

    For a page about somebody's radio traffic that is the correct property rather than a
    convenience: no request leaves the machine it is opened on.
    """
    page = dashboard([emission()])

    for external in ("<script", "src=", "http://", "https://", "@import", "<link"):
        assert external not in page, f"the page references {external!r}"


def test_the_page_reports_the_emitters_it_was_given() -> None:
    page = dashboard(
        [emission(channel=8, tone=114.8), emission(channel=3, tone=141.3, frequency=446_031_250.0)]
    )

    assert "PMR8/114.8Hz" in page
    assert "PMR3/141.3Hz" in page


def test_by_products_are_shown_as_by_products_not_emitters() -> None:
    """The distinction the whole order of battle turns on, carried into the picture."""
    reports = [
        emission(at=0.0, duration=4.0),
        emission(at=0.5, duration=1.0, channel=11, frequency=446_131_250.0, tone=None),
        emission(at=0.5, duration=1.0, channel=5, frequency=446_056_250.0, tone=None),
    ]
    page = dashboard(reports)

    assert "by-products" in page
    assert "lower bound" in page


def test_the_metadata_only_policy_is_on_the_page() -> None:
    """Anybody shown this should not have to read the repository to learn what it records."""
    assert "never communication content" in dashboard([emission()])


def test_an_empty_band_produces_a_page_rather_than_an_error() -> None:
    page = dashboard([])

    assert "<html" in page
    assert "0" in page


def test_the_page_escapes_what_it_is_given() -> None:
    """Emitter labels are derived from measurements, but the title is not."""
    page = dashboard([emission()], title='<script>alert("x")</script>')

    assert "<script>alert" not in page
    assert "&lt;script&gt;" in page


# --------------------------------------------------------------------------------------
# The waterfall
# --------------------------------------------------------------------------------------


def test_the_waterfall_renders_a_capture(tmp_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")

    rate, centre = 2_000_000.0, 446_593_750.0
    samples = int(0.5 * rate)
    rng = np.random.default_rng(0)
    t = np.arange(samples) / rate
    iq = (1e-3 * (rng.standard_normal(samples) + 1j * rng.standard_normal(samples))).astype(
        np.complex64
    )
    iq += (0.05 * np.exp(2j * np.pi * (446_093_750.0 - centre) * t)).astype(np.complex64)

    path = waterfall(iq, rate, centre, [emission(at=0.0, duration=0.5)], tmp_path / "w.png")

    assert path.exists()
    assert path.stat().st_size > 10_000


def test_the_waterfall_titles_itself_with_the_real_duration(tmp_path: Path) -> None:
    """A caption that disagrees with the axis beside it is worse than no caption."""
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.pyplot.close("all")

    rate, centre = 2_000_000.0, 446_593_750.0
    iq = np.zeros(int(0.25 * rate), dtype=np.complex64)

    waterfall(iq, rate, centre, [], tmp_path / "w.png")

    figures = matplotlib.pyplot.get_fignums()
    assert not figures, "the figure was left open, which leaks memory across a long run"
