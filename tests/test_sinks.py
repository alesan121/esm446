"""Verification of the metadata sinks.

Most of what matters here is about failure. A sink that works when everything is fine and
loses an hour of capture when it is not has failed at its only job, so these tests are mostly
about appending, surviving reopening, and not taking the node down when a sink breaks.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from esm446.core.node import EmissionReport
from esm446.io.sinks import (
    _COLUMNS,
    _INSERT_SQL,
    EmissionSink,
    JsonlSink,
    MultiSink,
    SqliteSink,
    open_sink,
    read_reports,
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
        offset_s=0.0,
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


# --------------------------------------------------------------------------------------
# Reading back
# --------------------------------------------------------------------------------------


def test_reports_survive_a_round_trip_through_jsonl(tmp_path: Path) -> None:
    original = report()
    with JsonlSink(tmp_path / "e.jsonl") as sink:
        sink.write([original])

    assert read_reports(tmp_path / "e.jsonl") == [original]


def test_reports_survive_a_round_trip_through_sqlite(tmp_path: Path) -> None:
    """Including the gains, which are stored as three columns and have to be reassembled."""
    original = report()
    with SqliteSink(tmp_path / "e.db") as sink:
        sink.write([original])

    recovered = read_reports(tmp_path / "e.db")

    assert recovered == [original]
    assert recovered[0].gains == {"lna_db": 0.0, "vga_db": 0.0, "amp_enabled": False}


def test_reading_returns_emissions_in_time_order(tmp_path: Path) -> None:
    """Analysis assumes chronology; a store written out of order must not break it."""
    with JsonlSink(tmp_path / "e.jsonl") as sink:
        sink.write([report(timestamp=3000.0), report(timestamp=1000.0), report(timestamp=2000.0)])

    assert [r.timestamp for r in read_reports(tmp_path / "e.jsonl")] == [1000.0, 2000.0, 3000.0]


def test_reading_a_missing_store_says_so(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_reports(tmp_path / "never-captured.jsonl")


def test_reading_an_unknown_extension_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "e.parquet").write_text("")
    with pytest.raises(ValueError, match="jsonl"):
        read_reports(tmp_path / "e.parquet")


def test_an_empty_store_reads_as_no_emissions(tmp_path: Path) -> None:
    with SqliteSink(tmp_path / "e.db"):
        pass

    assert read_reports(tmp_path / "e.db") == []


# --------------------------------------------------------------------------------------
# Schema migration
# --------------------------------------------------------------------------------------


def test_an_archive_from_an_earlier_version_is_brought_up_to_date(tmp_path: Path) -> None:
    """CREATE TABLE IF NOT EXISTS sees a table and stops, so the columns must be added.

    An archive is the one thing here that cannot be regenerated. Refusing to open an older
    one, or opening it and failing on the first insert, would lose the capture it holds.
    """
    path = tmp_path / "old.db"
    connection = sqlite3.connect(path)
    connection.executescript("""
        CREATE TABLE emissions (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp         REAL    NOT NULL,
            frequency_hz      REAL    NOT NULL,
            pmr_channel       INTEGER,
            bin_index         INTEGER NOT NULL,
            duration_s        REAL    NOT NULL,
            peak_power_dbfs   REAL    NOT NULL,
            snr_db            REAL    NOT NULL,
            estimated_dbm     REAL,
            calibrated        INTEGER NOT NULL,
            ctcss_tone_hz     REAL,
            classification    TEXT    NOT NULL,
            offset_s          REAL    NOT NULL,
            peak_deviation_hz REAL    NOT NULL,
            lna_db            REAL,
            vga_db            REAL,
            amp_enabled       INTEGER
        );
        INSERT INTO emissions (
            timestamp, frequency_hz, pmr_channel, bin_index, duration_s, peak_power_dbfs,
            snr_db, calibrated, classification, offset_s, peak_deviation_hz
        ) VALUES (500.0, 446093750.0, 8, 9, 1.0, -30.0, 20.0, 0, 'UNKNOWN', 0.0, 1200.0);
        """)
    connection.commit()
    connection.close()

    with SqliteSink(path) as sink:
        assert sink.write([report()]) == 1
        assert sink.count() == 2

    recovered = read_reports(path)
    assert len(recovered) == 2
    assert recovered[0].timestamp == 500.0
    assert recovered[0].attribution is None, "never examined is not the same as examined"


def test_the_insert_statement_matches_the_column_tuple() -> None:
    """The insert is written literally rather than assembled, so it can drift from _COLUMNS.

    Bandit is right to dislike SQL built by interpolation even where every input is a
    compile-time constant, so the statement is spelled out -- and this is what stops the two
    definitions disagreeing silently, which would misfile every column after the first
    mismatch.
    """
    inside = _INSERT_SQL.split("(", 1)[1].split(")", 1)[0]
    named = tuple(name.strip() for name in inside.replace("\n", " ").split(",") if name.strip())

    assert named == _COLUMNS
    assert _INSERT_SQL.count("?") == len(_COLUMNS)
