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


# --------------------------------------------------------------------------------------
# The verification figures
# --------------------------------------------------------------------------------------


def test_the_vv_figures_regenerate_from_scratch(tmp_path) -> None:
    """A figure nobody can regenerate is a claim, not evidence.

    Runs the whole set into a temporary directory, which is also the only way to know the
    report's plots still come from a system that behaves the way the report says.
    """
    import matplotlib

    matplotlib.use("Agg")

    from esm446.vv import generate

    results = generate(tmp_path)

    expected = {
        "adjacent_channel_rejection_db",
        "worst_case_scalloping_db",
        "coverage",
        "node_cpu_s_per_s",
    }
    assert expected <= set(results)
    assert len(list(tmp_path.glob("*.png"))) == 7
    assert (tmp_path / "results.json").exists()


def test_the_generated_figures_still_meet_the_requirements(tmp_path) -> None:
    """The report quotes these. If the system regresses, the quoted figures must fail here.

    Loose bounds on purpose: this guards against a regression, not against the last decimal,
    which the dedicated tests in test_channelizer.py and test_geolocation.py pin properly.
    """
    import matplotlib

    matplotlib.use("Agg")

    from esm446.vv import generate

    results = generate(tmp_path)

    assert results["adjacent_channel_rejection_db"] > 60.0, "REQ-FUN-002"
    assert results["worst_case_scalloping_db"] == pytest.approx(-6.02, abs=0.2), "REQ-FUN-004"
    assert results["node_cpu_s_per_s"] < 0.5, "REQ-PER-001"
    assert 90.0 < results["coverage"]["95"] < 99.0, "REQ-CAL-005"
