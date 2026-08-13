"""Persist emission metadata so it can be queried later.

The node produces one record per emission and, until now, printed it and forgot it. Nothing
downstream — occupancy over time, emitter grouping, pattern of life — can be built on output
that only exists in a terminal.

Two sinks, because they answer different questions
--------------------------------------------------
**JSONL** is append-only, one record per line. That format is chosen for what happens when
things go wrong: a process killed mid-write loses at most the last line, the file stays
readable, and it can be inspected with tools already present on any machine. A long capture
is worth more if a crash costs a second of it rather than all of it.

**SQLite** is for asking questions across time — which channels were busy at which hours,
how many distinct tones appeared, how burst durations are distributed. Those are queries, and
writing a query engine over JSONL would be rebuilding a database badly.

Both implement the same interface, so the node writes to whichever is configured without
knowing which, and to both at once when that is useful.

What is stored and why
----------------------
Everything in `EmissionReport`, and in particular the receiver gain configuration. An archive
of power readings without the gains they were taken at cannot be calibrated afterwards: the
information needed to fix it later has to be written down at the time. That is exactly the
flaw that made v0's `OFFSET_CAL` unusable in retrospect, and it costs three columns to avoid.

Never audio. The node is metadata-only; see ``docs/06_legal_ethics.md``.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path
from types import TracebackType
from typing import Any

from esm446.core.node import EmissionReport

logger = logging.getLogger(__name__)

#: Columns of the emissions table, in order. Kept explicit rather than derived from the
#: dataclass so that a change to the record is a deliberate schema decision.
_COLUMNS = (
    "timestamp",
    "frequency_hz",
    "pmr_channel",
    "bin_index",
    "duration_s",
    "peak_power_dbfs",
    "snr_db",
    "estimated_dbm",
    "calibrated",
    "ctcss_tone_hz",
    "classification",
    "offset_s",
    "peak_deviation_hz",
    "lna_db",
    "vga_db",
    "amp_enabled",
    "attributed_to_hz",
    "attribution",
)

#: The insert statement, written out rather than assembled from `_COLUMNS`.
#:
#: Building SQL by interpolation is a habit worth not having even where the inputs are
#: compile-time constants, because the next person to touch it may pass something that is
#: not. Writing it literally removes the construct entirely instead of annotating it away,
#: and `test_sinks.py` asserts that the statement and `_COLUMNS` agree so the two cannot
#: drift apart.
_INSERT_SQL = """
    INSERT INTO emissions (
        timestamp, frequency_hz, pmr_channel, bin_index, duration_s, peak_power_dbfs,
        snr_db, estimated_dbm, calibrated, ctcss_tone_hz, classification, offset_s,
        peak_deviation_hz, lna_db, vga_db, amp_enabled, attributed_to_hz, attribution
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

#: Columns added after the first release, each with the statement that adds it. A database
#: written by an earlier version is missing them, and ``CREATE TABLE IF NOT EXISTS`` will not
#: notice: it sees a table and stops. Losing an archive to a schema change would defeat the
#: point of keeping one, so each is added to an existing table instead.
#:
#: The statements are spelled out for the same reason `_INSERT_SQL` is -- see #36. Assembling
#: them from the column name would put SQL back together by interpolation, and a rule that
#: gets worked around stops being a rule.
_ADDED_COLUMNS = {
    "attributed_to_hz": "ALTER TABLE emissions ADD COLUMN attributed_to_hz REAL",
    "attribution": "ALTER TABLE emissions ADD COLUMN attribution TEXT",
}


def _flatten(report: EmissionReport) -> dict[str, Any]:
    """Flatten a report into one row, lifting the gains out of their nested dictionary.

    Args:
        report: The emission to flatten.

    Returns:
        A dictionary keyed by `_COLUMNS`.
    """
    payload = report.as_dict()
    gains = payload.pop("gains", {}) or {}
    payload["lna_db"] = gains.get("lna_db")
    payload["vga_db"] = gains.get("vga_db")
    payload["amp_enabled"] = gains.get("amp_enabled")
    return payload


class EmissionSink(ABC):
    """Somewhere emission records go."""

    @abstractmethod
    def write(self, reports: list[EmissionReport]) -> int:
        """Append records.

        Args:
            reports: Emissions to store.

        Returns:
            Number of records written.
        """

    def close(self) -> None:
        """Release any resource held. Safe to call more than once."""

    def __enter__(self) -> EmissionSink:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


class JsonlSink(EmissionSink):
    """Append records to a newline-delimited JSON file.

    Opened in append mode and flushed after each batch. A capture that dies partway through
    leaves a file that is still valid up to its last complete line, which is the property
    that matters when a run is hours long.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a", encoding="utf-8")
        logger.info("sink: appending emissions to %s", self.path)

    def write(self, reports: list[EmissionReport]) -> int:
        if not reports:
            return 0
        for report in reports:
            self._handle.write(json.dumps(report.as_dict()) + "\n")
        self._handle.flush()
        return len(reports)

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.close()

    def read_all(self) -> list[dict[str, Any]]:
        """Read every record back, for tests and for feeding the analysis."""
        if not self.path.exists():
            return []
        return [
            json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line
        ]


class SqliteSink(EmissionSink):
    """Append records to a SQLite database.

    The schema is created on first use and is additive only: an existing database from an
    earlier run is opened and appended to, never replaced. Losing an archive to a schema
    change would defeat the point of keeping one.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path)
        self._connection.row_factory = sqlite3.Row
        self._create_schema()
        logger.info("sink: appending emissions to %s", self.path)

    def _create_schema(self) -> None:
        """Create the table and its indexes if they are not already there."""
        self._connection.executescript("""
            CREATE TABLE IF NOT EXISTS emissions (
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
                amp_enabled       INTEGER,
                attributed_to_hz  REAL,
                attribution       TEXT
            );
            -- Occupancy queries scan by time and group by channel, so both are indexed.
            CREATE INDEX IF NOT EXISTS idx_emissions_time    ON emissions(timestamp);
            CREATE INDEX IF NOT EXISTS idx_emissions_channel ON emissions(pmr_channel);
            """)
        self._add_missing_columns()
        self._connection.commit()

    def _add_missing_columns(self) -> None:
        """Bring a database written by an earlier version up to the current schema.

        Additive only, and each column is nullable, so an older archive keeps every record it
        has and simply reports ``NULL`` for what was never measured.
        """
        present = {row["name"] for row in self._connection.execute("PRAGMA table_info(emissions)")}
        for name, statement in _ADDED_COLUMNS.items():
            if name not in present:
                self._connection.execute(statement)
                logger.info("sink: added column %s to an existing archive", name)

    def write(self, reports: list[EmissionReport]) -> int:
        if not reports:
            return 0
        rows = [tuple(_flatten(r).get(c) for c in _COLUMNS) for r in reports]
        self._connection.executemany(_INSERT_SQL, rows)
        self._connection.commit()
        return len(rows)

    def query(self, sql: str, parameters: tuple = ()) -> list[sqlite3.Row]:
        """Run a read-only query against the store.

        Args:
            sql: The statement.
            parameters: Bound parameters.

        Returns:
            The rows returned.
        """
        return list(self._connection.execute(sql, parameters))

    def count(self) -> int:
        """Number of emissions stored."""
        return int(self._connection.execute("SELECT COUNT(*) FROM emissions").fetchone()[0])

    def close(self) -> None:
        self._connection.close()


class MultiSink(EmissionSink):
    """Write to several sinks at once.

    A failing sink must not take the others with it, and must not take the node with it
    either: losing the archive is bad, losing the capture is worse. Failures are logged and
    the remaining sinks still receive the batch.
    """

    def __init__(self, sinks: list[EmissionSink]) -> None:
        self.sinks = sinks

    def write(self, reports: list[EmissionReport]) -> int:
        written = 0
        for sink in self.sinks:
            try:
                written = max(written, sink.write(reports))
            except Exception:  # noqa: BLE001 -- a broken sink must not stop the capture
                logger.exception("sink: %s failed to write, continuing", type(sink).__name__)
        return written

    def close(self) -> None:
        for sink in self.sinks:
            try:
                sink.close()
            except Exception:  # noqa: BLE001
                logger.exception("sink: %s failed to close", type(sink).__name__)


def _to_report(row: dict[str, Any]) -> EmissionReport:
    """Rebuild a report from a stored row.

    The gains are reassembled from their three flattened columns, which is the whole reason
    they were stored flat: a column per gain is queryable, a JSON blob is not.

    Args:
        row: One stored record, from either sink.

    Returns:
        The report it came from.
    """
    gains = row.get("gains")
    if gains is None:
        gains = {
            "lna_db": row.get("lna_db"),
            "vga_db": row.get("vga_db"),
            "amp_enabled": bool(row["amp_enabled"]) if row.get("amp_enabled") is not None else None,
        }
    return EmissionReport(
        timestamp=float(row["timestamp"]),
        frequency_hz=float(row["frequency_hz"]),
        pmr_channel=row["pmr_channel"],
        bin_index=int(row["bin_index"]),
        duration_s=float(row["duration_s"]),
        peak_power_dbfs=float(row["peak_power_dbfs"]),
        snr_db=float(row["snr_db"]),
        estimated_dbm=row["estimated_dbm"],
        calibrated=bool(row["calibrated"]),
        ctcss_tone_hz=row["ctcss_tone_hz"],
        classification=str(row["classification"]),
        offset_s=float(row["offset_s"]),
        peak_deviation_hz=float(row["peak_deviation_hz"]),
        gains=gains,
        # Absent from an archive written before attribution existed, which is a record that
        # was never examined rather than one examined and found to be an emission.
        attributed_to_hz=row.get("attributed_to_hz"),
        attribution=row.get("attribution"),
    )


def read_reports(path: Path) -> list[EmissionReport]:
    """Read every emission back out of a store.

    The counterpart to `open_sink`, and deliberately symmetric with it: analysis reads a path
    and gets reports, without caring which of the two formats the capture happened to use.

    Args:
        path: A ``.jsonl``, ``.db`` or ``.sqlite`` store.

    Returns:
        The stored emissions, oldest first.

    Raises:
        ValueError: If the extension is not one this module writes.
        FileNotFoundError: If the store does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no store at {path}")

    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows: list[dict[str, Any]] = [
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line
        ]
    elif suffix in (".db", ".sqlite"):
        connection = sqlite3.connect(path)
        connection.row_factory = sqlite3.Row
        try:
            rows = [
                dict(r) for r in connection.execute("SELECT * FROM emissions ORDER BY timestamp")
            ]
        finally:
            connection.close()
    else:
        raise ValueError(f"cannot tell what store {path} is; use .jsonl, .db or .sqlite")

    reports = [_to_report(row) for row in rows]
    reports.sort(key=lambda r: r.timestamp)
    logger.info("sink: read %d emissions from %s", len(reports), path)
    return reports


def open_sink(path: Path | None) -> EmissionSink | None:
    """Choose a sink from a path's extension.

    Args:
        path: Destination, or ``None`` for no persistence.

    Returns:
        The sink, or ``None`` when no path was given.

    Raises:
        ValueError: If the extension is not one of ``.jsonl``, ``.db`` or ``.sqlite``.
    """
    if path is None:
        return None
    suffix = Path(path).suffix.lower()
    if suffix == ".jsonl":
        return JsonlSink(path)
    if suffix in (".db", ".sqlite"):
        return SqliteSink(path)
    raise ValueError(f"cannot tell what sink {path} should be; use .jsonl, .db or .sqlite")
