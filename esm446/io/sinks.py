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
    "peak_deviation_hz",
    "lna_db",
    "vga_db",
    "amp_enabled",
)


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
                peak_deviation_hz REAL    NOT NULL,
                lna_db            REAL,
                vga_db            REAL,
                amp_enabled       INTEGER
            );
            -- Occupancy queries scan by time and group by channel, so both are indexed.
            CREATE INDEX IF NOT EXISTS idx_emissions_time    ON emissions(timestamp);
            CREATE INDEX IF NOT EXISTS idx_emissions_channel ON emissions(pmr_channel);
            """)
        self._connection.commit()

    def write(self, reports: list[EmissionReport]) -> int:
        if not reports:
            return 0
        rows = [tuple(_flatten(r).get(c) for c in _COLUMNS) for r in reports]
        placeholders = ", ".join("?" * len(_COLUMNS))
        self._connection.executemany(
            f"INSERT INTO emissions ({', '.join(_COLUMNS)}) VALUES ({placeholders})", rows
        )
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
