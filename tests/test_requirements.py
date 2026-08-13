"""Verification that the requirements document describes this system and not another one.

A requirements document is worth having only if it is true, and the way it stops being true
is gradual: a test is renamed, a requirement is dropped, a status stays MET after the thing
it verified was deleted. None of that announces itself.

So the document is parsed and checked. Every requirement must name a test, every named test
must exist, and any requirement not verified must say why. That makes the traceability matrix
a build artefact rather than a claim, which is the only version of it worth putting in front
of somebody who has read a few.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

REQUIREMENTS = Path("docs/01_requirements.md")
TESTS_DIR = Path("tests")

#: A row of a requirements table: | ID | text | method | verified by | status |
_ROW = re.compile(
    r"^\|\s*(REQ-[A-Z]+-\d+)\s*\|(.+?)\|\s*([ATID, ]+?)\s*\|(.+?)\|\s*(.+?)\s*\|\s*$",
    re.MULTILINE,
)

#: Test names inside the "verified by" cell, which wraps them in backticks.
_TEST_NAME = re.compile(r"`(test_[a-z0-9_]+)`")

VALID_METHODS = {"A", "T", "I", "D"}
VALID_STATUSES = ("MET", "PARTIAL", "BLOCKED")


class Requirement:
    """One parsed row of the requirements document."""

    def __init__(self, identifier: str, text: str, methods: str, verified: str, status: str):
        self.id = identifier
        self.text = text.strip()
        self.methods = {m.strip() for m in methods.split(",") if m.strip()}
        self.tests = _TEST_NAME.findall(verified)
        # Emphasis markers are presentation, not content: BLOCKED and **BLOCKED** are the
        # same status and the parser must not care which the author reached for.
        self.status = status.strip().replace("*", "").strip()

    def __repr__(self) -> str:
        return f"<{self.id} {self.status}>"


@pytest.fixture(scope="module")
def requirements() -> list[Requirement]:
    """Every requirement in the document."""
    if not REQUIREMENTS.exists():
        pytest.skip("requirements document not present")
    return [Requirement(*match) for match in _ROW.findall(REQUIREMENTS.read_text(encoding="utf-8"))]


@pytest.fixture(scope="module")
def defined_tests() -> set[str]:
    """Every test function defined under `tests/`.

    Read from the syntax tree rather than by importing, so collecting the names costs nothing
    and cannot be affected by an import failure elsewhere in the suite.
    """
    names: set[str] = set()
    for path in TESTS_DIR.rglob("test_*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                names.add(node.name)
    return names


# --------------------------------------------------------------------------------------
# The document parses at all
# --------------------------------------------------------------------------------------


def test_the_document_contains_requirements(requirements: list[Requirement]) -> None:
    assert len(requirements) > 30, f"only {len(requirements)} requirements parsed"


def test_every_identifier_is_unique(requirements: list[Requirement]) -> None:
    """A duplicated ID makes every reference to it ambiguous, including in a review."""
    identifiers = [r.id for r in requirements]

    assert len(identifiers) == len(set(identifiers)), "duplicate requirement identifiers"


def test_every_requirement_states_a_verification_method(requirements: list[Requirement]) -> None:
    for requirement in requirements:
        assert requirement.methods, f"{requirement.id} has no verification method"
        unknown = requirement.methods - VALID_METHODS
        assert not unknown, f"{requirement.id} uses unknown method(s) {unknown}"


def test_every_requirement_has_a_recognised_status(requirements: list[Requirement]) -> None:
    for requirement in requirements:
        assert requirement.status.startswith(VALID_STATUSES), (
            f"{requirement.id} has status {requirement.status!r}, which is not one of "
            f"{VALID_STATUSES}"
        )


# --------------------------------------------------------------------------------------
# Traceability: the part that rots if nobody checks it
# --------------------------------------------------------------------------------------


def test_no_requirement_is_orphaned(requirements: list[Requirement]) -> None:
    """Every requirement is verified, or says plainly why it cannot be.

    An unverified requirement with no explanation is the failure mode this whole document
    exists to avoid: a statement that reads as a commitment and is backed by nothing.
    """
    for requirement in requirements:
        if requirement.tests:
            continue
        assert requirement.status.startswith(
            "BLOCKED"
        ), f"{requirement.id} names no verifying test and is not marked BLOCKED"
        assert (
            "#" in requirement.status or "see" in requirement.status.lower()
        ), f"{requirement.id} is BLOCKED without naming the blocker"


def test_every_named_test_exists(requirements: list[Requirement], defined_tests: set[str]) -> None:
    """The check that catches a rename, which is how a matrix silently stops being true."""
    missing = {
        requirement.id: [t for t in requirement.tests if t not in defined_tests]
        for requirement in requirements
        if any(t not in defined_tests for t in requirement.tests)
    }

    assert not missing, f"requirements naming tests that do not exist: {missing}"


def test_a_partial_status_states_what_is_missing(requirements: list[Requirement]) -> None:
    """ "Partial" without a shortfall is just "met" with a hedge in front of it."""
    for requirement in requirements:
        if requirement.status.startswith("PARTIAL"):
            assert (
                len(requirement.status) > len("PARTIAL") + 10
            ), f"{requirement.id} is PARTIAL without saying what is missing"


def test_a_measured_claim_carries_its_figure(requirements: list[Requirement]) -> None:
    """A performance requirement met "because it is" is not verified, it is asserted."""
    performance = [r for r in requirements if r.id.startswith("REQ-PER")]
    assert performance, "no performance requirements found"

    for requirement in performance:
        if requirement.status.startswith("MET") and "measured" not in requirement.status:
            # A requirement about behaviour rather than a figure is allowed to omit one.
            assert (
                "shall not" in requirement.text or "shall not change" in requirement.text
            ), f"{requirement.id} claims MET without quoting the measured figure"


# --------------------------------------------------------------------------------------
# Coverage of the document by the requirements, rather than the other way round
# --------------------------------------------------------------------------------------


def test_every_category_is_represented(requirements: list[Requirement]) -> None:
    """The categories the plan calls for, so a whole class cannot be quietly dropped."""
    prefixes = {r.id.rsplit("-", 1)[0] for r in requirements}

    assert {"REQ-FUN", "REQ-PER", "REQ-IF", "REQ-LEG"} <= prefixes


def test_no_document_names_a_test_that_does_not_exist(defined_tests: set[str]) -> None:
    """The same rot, one level out: the architecture and V&V documents cite tests too.

    A renamed test leaves a document quietly claiming verification that no longer happens,
    and nothing about reading the document reveals it.
    """
    stale: dict[str, list[str]] = {}
    for document in sorted(Path("docs").glob("*.md")):
        cited = set(_TEST_NAME.findall(document.read_text(encoding="utf-8")))
        missing = sorted(name for name in cited if name not in defined_tests)
        if missing:
            stale[document.name] = missing

    assert not stale, f"documents citing tests that do not exist: {stale}"


def test_the_blocked_requirement_is_the_one_we_expect(requirements: list[Requirement]) -> None:
    """One blocker, named, and it is the calibration. If another appears it should be argued.

    This test exists to make adding a second BLOCKED requirement a deliberate act rather than
    a quiet one.
    """
    blocked = [r.id for r in requirements if r.status.startswith("BLOCKED")]

    assert blocked == ["REQ-CAL-004"], f"unexpected blocked requirements: {blocked}"


# --------------------------------------------------------------------------------------
# The V&V report agrees with the requirements it reports on
# --------------------------------------------------------------------------------------

VV_REPORT = Path("docs/05_vv_report.md")


def test_the_vv_report_totals_match_the_requirements(requirements: list[Requirement]) -> None:
    """Two documents counting the same thing will disagree eventually unless something checks.

    The V&V report opens with a count and closes with a table of counts. Both are read from
    the requirements document by a human, which is exactly the kind of transcription that
    goes stale the first time a requirement is added.
    """
    if not VV_REPORT.exists():
        pytest.skip("V&V report not present")
    report = VV_REPORT.read_text(encoding="utf-8")

    counts = {status: 0 for status in VALID_STATUSES}
    for requirement in requirements:
        counts[requirement.status.split()[0].split("—")[0].strip()] += 1

    assert f"**{len(requirements)} requirements**" in report or (
        f"{len(requirements)} requirements" in report
    ), f"the report does not state the true total of {len(requirements)}"
    for status, count in counts.items():
        assert (
            f"{count} {status}" in report
        ), f"the report does not state {count} {status} requirements"


def test_the_vv_report_totals_row_adds_up(requirements: list[Requirement]) -> None:
    """The summary table's total row, checked against the rows above it."""
    if not VV_REPORT.exists():
        pytest.skip("V&V report not present")
    report = VV_REPORT.read_text(encoding="utf-8")

    match = re.search(r"\|\s*\*\*Total\*\*\s*\|\s*\*\*(\d+)\*\*", report)
    assert match, "no total row found in the traceability summary"

    assert int(match.group(1)) == len(requirements)
