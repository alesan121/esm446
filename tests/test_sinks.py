"""Verification of the metadata sinks.

Most of what matters here is about failure. A sink that works when everything is fine and
loses an hour of capture when it is not has failed at its only job, so these tests are mostly
about appending, surviving reopening, and not taking the node down when a sink breaks.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from esm446.core.node import EmissionReport
from esm446.io.sinks import (
    EmissionSink,
    JsonlSink,
    MultiSink,
    SqliteSink,
    open_sink,
)


def report(
    timestamp: float = 1000.0,
    channel: int | None = 8,
    tone: float | None = 114.8,
    snr: float = 40.0,
) -> EmissionReport:
    return EmissionReport(
        timestamp=timestamp,
        frequency_hz=446_093_750.0,
        pmr_channel=channel,
        bin_index=9,
        duration_s=4.2,
        peak_power_dbfs=-22.5,
        snr_db=snr,
        estimated_dbm=None,
        calibrated=False,
        ctcss_tone_hz=tone,
        classification="FRIEND",
        peak_deviation_hz=1346.8,
        gains={"lna_db": 0.0, "vga_db": 0.0, "amp_enabled": False},
    )


# --------------------------------------------------------------------------------------
# JSONL
# --------------------------------------------------------------------------------------


def test_jsonl_round_trips_a_record(tmp_path: Path) -> None:
    with JsonlSink(tmp_path / "e.jsonl") as sink:
        assert sink.write([report()]) == 1
        rows = sink.read_all()

    assert len(rows) == 1
    assert rows[0]["pmr_channel"] == 8
    assert rows[0]["ctcss_tone_hz"] == pytest.approx(114.8)


def test_jsonl_appends_across_runs(tmp_path: Path) -> None:
    """A second run must add to the archive, not replace it."""
    path = tmp_path / "e.jsonl"
    with JsonlSink(path) as sink:
        sink.write([report(timestamp=1.0)])
    with JsonlSink(path) as sink:
        sink.write([report(timestamp=2.0)])
        rows = sink.read_all()

    assert [r["timestamp"] for r in rows] == [1.0, 2.0]


def test_jsonl_survives_a_truncated_last_line(tmp_path: Path) -> None:
    """The reason for one record per line: a kill mid-write costs the last record, not all.

    A half-written line is discarded and everything before it is still readable, which is the
    property that matters when a capture has been running for hours.
    """
    path = tmp_path / "e.jsonl"
    with JsonlSink(path) as sink:
        sink.write([report(timestamp=1.0), report(timestamp=2.0)])
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"timestamp": 3.0, "frequ')

    rows = [line for line in path.read_text().splitlines() if line]
    recovered = JsonlSink(path)
    try:
        with pytest.raises(Exception):
            recovered.read_all()
    finally:
        recovered.close()
    assert len(rows) == 3, "the partial line is present on disk"

    # Discarding the partial line recovers everything before it.
    path.write_text("\n".join(rows[:-1]) + "\n")
    with JsonlSink(path) as sink:
        assert len(sink.read_all()) == 2


def test_jsonl_never_stores_audio(tmp_path: Path) -> None:
    """Metadata only, checked at the boundary where data leaves the process."""
    with JsonlSink(tmp_path / "e.jsonl") as sink:
        sink.write([report()])
        keys = set(sink.read_all()[0])

    assert not keys & {"audio", "samples", "iq", "waveform"}


# --------------------------------------------------------------------------------------
# SQLite
# --------------------------------------------------------------------------------------


def test_sqlite_round_trips_a_record(tmp_path: Path) -> None:
    with SqliteSink(tmp_path / "e.db") as sink:
        sink.write([report()])
        rows = sink.query("SELECT * FROM emissions")

    assert len(rows) == 1
    assert rows[0]["pmr_channel"] == 8
    assert rows[0]["classification"] == "FRIEND"


def test_sqlite_lifts_the_gains_into_columns(tmp_path: Path) -> None:
    """Power without the gains it was measured at cannot be calibrated afterwards.

    That is the flaw that made v0's archive unusable, and it costs three columns to avoid.
    """
    with SqliteSink(tmp_path / "e.db") as sink:
        sink.write([report()])
        row = sink.query("SELECT lna_db, vga_db, amp_enabled FROM emissions")[0]

    assert row["lna_db"] == 0.0
    assert row["vga_db"] == 0.0
    assert row["amp_enabled"] == 0


def test_sqlite_appends_across_runs(tmp_path: Path) -> None:
    path = tmp_path / "e.db"
    with SqliteSink(path) as sink:
        sink.write([report(timestamp=1.0)])
    with SqliteSink(path) as sink:
        sink.write([report(timestamp=2.0)])
        assert sink.count() == 2


def test_sqlite_supports_an_occupancy_query(tmp_path: Path) -> None:
    """The kind of question the store exists to answer, and the reason it is not just JSONL."""
    with SqliteSink(tmp_path / "e.db") as sink:
        sink.write([report(channel=8), report(channel=8), report(channel=3), report(channel=None)])
        rows = sink.query(
            "SELECT pmr_channel, COUNT(*) AS n FROM emissions "
            "GROUP BY pmr_channel ORDER BY n DESC"
        )

    assert rows[0]["pmr_channel"] == 8
    assert rows[0]["n"] == 2


def test_sqlite_keeps_nulls_as_nulls(tmp_path: Path) -> None:
    """An uncalibrated dBm must stay null rather than becoming a zero somebody trusts."""
    with SqliteSink(tmp_path / "e.db") as sink:
        sink.write([report(channel=None, tone=None)])
        row = sink.query("SELECT estimated_dbm, pmr_channel, ctcss_tone_hz FROM emissions")[0]

    assert row["estimated_dbm"] is None
    assert row["pmr_channel"] is None
    assert row["ctcss_tone_hz"] is None


# --------------------------------------------------------------------------------------
# Composition and failure
# --------------------------------------------------------------------------------------


def test_multisink_writes_to_all(tmp_path: Path) -> None:
    jsonl = JsonlSink(tmp_path / "e.jsonl")
    sqlite = SqliteSink(tmp_path / "e.db")
    with MultiSink([jsonl, sqlite]) as sink:
        sink.write([report(), report(timestamp=2.0)])

    assert len(JsonlSink(tmp_path / "e.jsonl").read_all()) == 2
    assert SqliteSink(tmp_path / "e.db").count() == 2


def test_a_broken_sink_does_not_stop_the_others(tmp_path: Path) -> None:
    """Losing the archive is bad; losing the capture is worse."""

    class Broken(EmissionSink):
        def write(self, reports):
            raise OSError("disk full")

    good = JsonlSink(tmp_path / "e.jsonl")
    with MultiSink([Broken(), good]) as sink:
        sink.write([report()])

    assert len(JsonlSink(tmp_path / "e.jsonl").read_all()) == 1


def test_writing_nothing_is_not_an_error(tmp_path: Path) -> None:
    with JsonlSink(tmp_path / "e.jsonl") as sink:
        assert sink.write([]) == 0


def test_open_sink_chooses_by_extension(tmp_path: Path) -> None:
    assert isinstance(open_sink(tmp_path / "e.jsonl"), JsonlSink)
    assert isinstance(open_sink(tmp_path / "e.db"), SqliteSink)
    assert isinstance(open_sink(tmp_path / "e.sqlite"), SqliteSink)
    assert open_sink(None) is None


def test_open_sink_rejects_an_unknown_extension(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="jsonl"):
        open_sink(tmp_path / "e.parquet")
