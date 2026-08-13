"""Verification of the command-line plumbing: argument handling and failure messages.

The logic underneath each of these is tested thoroughly elsewhere. What was not tested is the
layer somebody actually types at — whether a missing file produces a sentence or a traceback,
whether a flag reaches the thing it names, whether an exit status means what it should. It is
shallow by design and it covers the paths a first-time user hits hardest.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from esm446.cli import simulate

SCENARIO = Path("scenarios/demo.yaml")


def small_scenario(tmp_path: Path) -> Path:
    """A two-emitter scene short enough to generate in a test."""
    path = tmp_path / "small.yaml"
    path.write_text(
        "name: small\n"
        "duration_s: 2.0\n"
        "sample_rate: 2000000.0\n"
        "centre_frequency: 446593750.0\n"
        "noise_floor_dbm: -110.0\n"
        "seed: 3\n"
        "propagation:\n"
        "  path_loss_exponent: 3.5\n"
        "  shadowing_sigma_db: 0.0\n"
        "emitters:\n"
        "  - name: near\n"
        "    channel: 4\n"
        "    distance_m: 400.0\n"
        "    ctcss_hz: 114.8\n"
        "    transmissions:\n"
        "      - {start_s: 0.4, stop_s: 1.4}\n",
        encoding="utf-8",
    )
    return path


# --------------------------------------------------------------------------------------
# Scenario generation
# --------------------------------------------------------------------------------------


def test_simulate_writes_the_scene_and_its_ground_truth(tmp_path: Path) -> None:
    scenario = small_scenario(tmp_path)

    assert simulate.main([str(scenario), "--output", str(tmp_path / "scene")]) == 0

    assert (tmp_path / "scene.cf32").stat().st_size > 0
    truth = json.loads((tmp_path / "scene.truth.json").read_text())
    assert truth["scenario"] == "small"


def test_simulate_reports_a_missing_scenario_without_a_traceback(tmp_path: Path) -> None:
    assert simulate.main([str(tmp_path / "absent.yaml")]) == 1


def test_simulate_honours_a_seed_override(tmp_path: Path) -> None:
    """Two runs of one scenario with different seeds must not produce identical samples."""
    scenario = small_scenario(tmp_path)

    simulate.main([str(scenario), "--output", str(tmp_path / "a"), "--seed", "1"])
    simulate.main([str(scenario), "--output", str(tmp_path / "b"), "--seed", "2"])

    assert (tmp_path / "a.cf32").read_bytes() != (tmp_path / "b.cf32").read_bytes()


def test_simulate_writes_the_format_it_was_asked_for(tmp_path: Path) -> None:
    """cs16 is half the size of cf32 and is what a recorder actually produces."""
    scenario = small_scenario(tmp_path)

    simulate.main([str(scenario), "--output", str(tmp_path / "c"), "--format", "cs16"])

    assert (tmp_path / "c.cs16").exists()


def test_simulate_rejects_a_format_it_cannot_write(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        simulate.main([str(small_scenario(tmp_path)), "--format", "cs8"])


# --------------------------------------------------------------------------------------
# The dashboard
# --------------------------------------------------------------------------------------


def test_the_dashboard_builds_a_page_from_a_store(tmp_path: Path) -> None:
    from esm446.core.node import EmissionReport
    from esm446.dashboard import main as dashboard_main
    from esm446.io.sinks import SqliteSink

    store = tmp_path / "emissions.db"
    with SqliteSink(store) as sink:
        sink.write(
            [
                EmissionReport(
                    timestamp=1_786_950_000.0,
                    frequency_hz=446_093_750.0,
                    pmr_channel=8,
                    bin_index=9,
                    duration_s=3.0,
                    peak_power_dbfs=-20.0,
                    snr_db=35.0,
                    estimated_dbm=None,
                    calibrated=False,
                    ctcss_tone_hz=114.8,
                    classification="FRIEND",
                    offset_s=0.0,
                    peak_deviation_hz=1_300.0,
                    gains={},
                )
            ]
        )

    output = tmp_path / "page.html"
    assert dashboard_main([str(store), "--output", str(output)]) == 0
    assert "PMR8/114.8Hz" in output.read_text(encoding="utf-8")


def test_the_dashboard_reports_a_missing_store_without_a_traceback(tmp_path: Path) -> None:
    from esm446.dashboard import main as dashboard_main

    assert dashboard_main([str(tmp_path / "absent.db")]) == 1


# --------------------------------------------------------------------------------------
# The order of battle
# --------------------------------------------------------------------------------------


def test_the_order_of_battle_reports_an_unreadable_store_without_a_traceback(
    tmp_path: Path,
) -> None:
    from esm446.cli.eob import main as eob_main

    unreadable = tmp_path / "emissions.parquet"
    unreadable.write_text("not a store")

    assert eob_main([str(unreadable)]) == 1
