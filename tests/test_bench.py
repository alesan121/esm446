"""Verification of the channeliser throughput benchmark.

The figure this produces is quoted in the README and the V&V report, so the arithmetic
turning elapsed time into a real-time margin is worth pinning down. A benchmark that
reports the wrong direction is worse than no benchmark.
"""

from __future__ import annotations

import json

import pytest

from esm446.bench import Result, benchmark_pfb, main


def test_ratio_below_one_means_the_node_keeps_up() -> None:
    result = Result("fast", 2e6, 160, signal_seconds=1.0, cpu_seconds=0.2)

    assert result.realtime_ratio == pytest.approx(0.2)
    assert result.realtime_margin == pytest.approx(5.0)
    assert "keeps up" in result.describe()


def test_ratio_above_one_is_reported_as_dropping_signal() -> None:
    """The v0 condition: more CPU time than signal time means samples are never examined."""
    result = Result("slow", 8e5, 57, signal_seconds=1.0, cpu_seconds=8.9)

    assert result.realtime_ratio > 1.0
    assert result.realtime_margin < 1.0
    assert "DROPS SIGNAL" in result.describe()


def test_margin_and_ratio_are_reciprocal() -> None:
    result = Result("x", 2e6, 160, signal_seconds=3.0, cpu_seconds=0.6)
    assert result.realtime_ratio * result.realtime_margin == pytest.approx(1.0)


def test_pfb_benchmark_keeps_up_with_real_time() -> None:
    """The performance requirement itself, exercised on the shipping configuration."""
    result = benchmark_pfb(seconds=0.5)

    assert result.sample_rate == 2_000_000
    assert result.num_channels == 160
    assert result.realtime_ratio < 1.0


def test_main_fails_when_the_budget_is_exceeded() -> None:
    """CI gates on this exit code, so an impossible budget must be a non-zero exit."""
    assert main(["--skip-v0", "--max-ratio", "1e-9"]) == 1


def test_main_succeeds_within_a_generous_budget() -> None:
    assert main(["--skip-v0", "--max-ratio", "1.0"]) == 0


def test_json_output_is_machine_readable(capsys: pytest.CaptureFixture[str]) -> None:
    main(["--skip-v0", "--max-ratio", "1.0", "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["num_channels"] == 160
    assert payload[0]["realtime_ratio"] > 0.0
