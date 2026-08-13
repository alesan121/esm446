"""Verification of the demonstration and simulation entry points.

The demonstration is the first thing anyone evaluating this repository runs, so it is worth a
test that it works on a clean checkout with no SDR and no arguments — and that the scores it
prints are the ones the pipeline actually achieves, not placeholders.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from esm446.cli.demo import run_demo
from esm446.cli.simulate import write_scene
from esm446.sim.scenario import Emitter, Propagation, Scenario, Transmission

DEMO_SCENARIO = Path("scenarios/demo.yaml")


def small_scenario() -> Scenario:
    """A short two-emitter scene, so the entry-point tests stay quick."""
    return Scenario(
        name="small",
        duration_s=4.0,
        noise_floor_dbm=-110.0,
        seed=3,
        propagation=Propagation(path_loss_exponent=3.5, shadowing_sigma_db=0.0),
        emitters=[
            Emitter(
                name="near",
                channel=4,
                distance_m=400.0,
                ctcss_hz=114.8,
                transmissions=[Transmission(0.5, 2.0)],
            ),
            Emitter(
                name="offgrid",
                frequency_hz=446_162_500.0,
                distance_m=600.0,
                transmissions=[Transmission(2.3, 3.8)],
            ),
        ],
    )


# --------------------------------------------------------------------------------------
# Simulation entry point
# --------------------------------------------------------------------------------------


def test_simulate_writes_iq_and_truth(tmp_path: Path) -> None:
    iq_path, truth_path = write_scene(small_scenario(), tmp_path / "scene")

    assert iq_path.exists() and truth_path.exists()
    payload = json.loads(truth_path.read_text())
    assert payload["scenario"] == "small"
    assert len(payload["emissions"]) == 2
    assert payload["emissions"][0]["snr_db"] > 0.0


def test_simulate_cs16_uses_the_full_scale(tmp_path: Path) -> None:
    """Quantising a weak scene without scaling first buries it in the bottom few bits."""
    iq_path, _ = write_scene(small_scenario(), tmp_path / "scene", sample_format="cs16")

    samples = np.fromfile(iq_path, dtype=np.int16)
    assert np.abs(samples).max() > 30_000


def test_simulated_file_replays_through_the_node(tmp_path: Path) -> None:
    """The loop that makes the testbench useful: generate, write, read back, detect."""
    from esm446.core.channelizer import ChannelizerConfig
    from esm446.core.node import EsmNode
    from esm446.core.source import FileSource

    scenario = small_scenario()
    iq_path, _ = write_scene(scenario, tmp_path / "scene")

    node = EsmNode(
        channelizer_config=ChannelizerConfig(
            sample_rate=scenario.sample_rate, num_channels=160, decimation=80
        ),
        centre_frequency=scenario.centre_frequency,
    )
    reports = node.run(FileSource(iq_path, scenario.sample_rate, scenario.centre_frequency, "cf32"))
    assert len(reports) == 2


# --------------------------------------------------------------------------------------
# Demonstration
# --------------------------------------------------------------------------------------


def test_demo_detects_both_emitters_and_scores_them() -> None:
    _, truth, result, _ = run_demo(small_scenario(), expected_ctcss_hz=114.8)

    assert len(truth) == 2
    assert result.probability_of_detection == 1.0
    assert result.spurious == []
    assert result.channel_accuracy == 1.0


def test_demo_reports_the_off_grid_emitter_as_off_grid() -> None:
    """Nothing obliges a transmitter to sit on the channel plan, and this one does not."""
    reports, _, _, _ = run_demo(small_scenario())

    off_grid = [r for r in reports if r.pmr_channel is None]
    assert len(off_grid) == 1
    assert off_grid[0].frequency_hz == pytest.approx(446_162_500.0, abs=500.0)


def test_demo_classifies_against_the_pre_shared_tone() -> None:
    reports, _, _, _ = run_demo(small_scenario(), expected_ctcss_hz=114.8)

    classifications = {r.classification for r in reports}
    assert "FRIEND" in classifications
    assert "UNKNOWN" in classifications


def test_demo_reports_no_dbm_without_a_calibration() -> None:
    reports, _, _, _ = run_demo(small_scenario())
    assert all(r.estimated_dbm is None for r in reports)


def test_shipped_demo_scenario_exists_for_the_make_target() -> None:
    """`make demo` depends on this file being present in a clean checkout."""
    assert DEMO_SCENARIO.exists()
    assert Scenario.load(DEMO_SCENARIO).duration_s > 0


def test_the_shipped_scenario_is_not_tuned_to_a_channel() -> None:
    """A scenario centred on a nominal channel has the node's DC guard erase that channel.

    The shipped scenario was tuned to channel 8, which is where its own `charlie` emitter
    transmits, so both of that emitter's transmissions were silently discarded and the
    demonstration reported Pd 0.83. The receiver refuses such a centre at startup; the
    scenario bypassed that because the demo takes its centre from the file.
    """
    from esm446.core import bands

    scenario = Scenario.load(DEMO_SCENARIO)
    bands.assert_centre_is_usable(scenario.centre_frequency)


def test_the_shipped_scenario_detects_everything_it_transmits() -> None:
    """The demonstration is the first thing a reviewer runs, so it has to be right."""
    scenario = Scenario.load(DEMO_SCENARIO)
    _, truth, result, _ = run_demo(scenario)

    assert (
        result.probability_of_detection == 1.0
    ), f"the shipped demonstration misses {len(truth) - result.num_detected} of {len(truth)}"
    assert result.spurious == []
