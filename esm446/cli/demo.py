"""Hardware-free demonstration: generate a scene, run the node, score the result.

One command, no SDR, no configuration. Someone evaluating this repository can see the system
do its job before deciding whether to read any of it.

Usage::

    esm446-demo
    esm446-demo --scenario scenarios/demo.yaml --json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from esm446.core.channelizer import ChannelizerConfig
from esm446.core.detector import CfarConfig
from esm446.core.node import EsmNode
from esm446.core.rfchain import RfChain, quantise_gains
from esm446.core.source import ArraySource
from esm446.sim.metrics import score
from esm446.sim.scenario import Scenario

logger = logging.getLogger(__name__)

DEFAULT_SCENARIO = Path("scenarios/demo.yaml")

#: Pre-shared tone the demonstration treats as cooperative identification.
DEMO_CTCSS_HZ = 114.8


def run_demo(scenario: Scenario, expected_ctcss_hz: float | None = DEMO_CTCSS_HZ):
    """Generate a scenario, run the node over it and score the output.

    Args:
        scenario: The scene to generate.
        expected_ctcss_hz: Pre-shared tone, or ``None`` to classify everything as unknown.

    Returns:
        ``(reports, truth, result, elapsed_s)``.
    """
    iq, truth = scenario.generate()

    node = EsmNode(
        channelizer_config=ChannelizerConfig(
            sample_rate=scenario.sample_rate, num_channels=160, decimation=80
        ),
        centre_frequency=scenario.centre_frequency,
        cfar_config=CfarConfig(),
        gains=quantise_gains(32.0, 20.0),
        expected_ctcss_hz=expected_ctcss_hz,
    )

    started = time.perf_counter()
    scene_start = time.time()
    reports = node.run(ArraySource(iq, scenario.sample_rate, scenario.centre_frequency))
    elapsed = time.perf_counter() - started

    result = score(reports, truth, scenario.duration_s, scene_start=scene_start)
    return reports, truth, result, elapsed


def print_report(scenario: Scenario, reports, truth, result, elapsed: float) -> None:
    """Print the demonstration output."""
    chain = RfChain.deployed()
    print(f"\n{'=' * 78}")
    print(f"ESM-446 demonstration  —  scenario '{scenario.name}'")
    print(f"{'=' * 78}\n")

    print("Receiver")
    print(f"  centre           {scenario.centre_frequency / 1e6:.6f} MHz (PMR446 channel 8)")
    print(f"  sample rate      {scenario.sample_rate / 1e6:.3f} MS/s, 160 channels of 12.5 kHz")
    print(f"  noise figure     {chain.noise_figure_db:.2f} dB with a 20 dB external LNA")
    print(f"  MDS              {chain.minimum_detectable_signal_dbm(12_500.0):.1f} dBm\n")

    print(
        f"Scene   {scenario.duration_s:.0f} s, {len(scenario.emitters)} emitters, "
        f"{len(truth)} transmissions"
    )
    for emitter in scenario.emitters:
        transmitted = [t for t in truth if t.emitter == emitter.name]
        if not transmitted:
            continue
        snr = sum(t.snr_db for t in transmitted) / len(transmitted)
        tone = f"{emitter.ctcss_hz:.1f} Hz" if emitter.ctcss_hz else "none"
        print(
            f"  {emitter.name:<9} {transmitted[0].frequency_hz / 1e6:.6f} MHz  "
            f"{'PMR' + str(transmitted[0].pmr_channel) if transmitted[0].pmr_channel else 'off-grid':<9} "
            f"{emitter.distance_m:>6.0f} m  SNR {snr:>5.1f} dB  CTCSS {tone}"
        )

    print(
        f"\nDetected  {len(reports)} emissions in {elapsed:.1f} s of CPU "
        f"({elapsed / scenario.duration_s:.3f} cpu-s/s)"
    )
    for report in reports[:12]:
        channel = f"PMR{report.pmr_channel}" if report.pmr_channel else "off-grid"
        tone = f"{report.ctcss_tone_hz:.1f} Hz" if report.ctcss_tone_hz else "none"
        print(
            f"  {report.frequency_hz / 1e6:.6f} MHz  {channel:<9} "
            f"{report.duration_s:>5.2f} s  SNR {report.snr_db:>5.1f} dB  "
            f"CTCSS {tone:<9} {report.classification}"
        )
    if len(reports) > 12:
        print(f"  ... and {len(reports) - 12} more")

    print(f"\nScore against ground truth\n{result.describe()}")
    print(
        "\nNote: estimated_dbm is null throughout. Absolute power is reported only where a\n"
        "calibration exists for that receiver gain configuration and level range.\n"
    )


def main(argv: list[str] | None = None) -> int:
    """Run the demonstration.

    Args:
        argv: Command line arguments, or ``None`` to read from ``sys.argv``.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario", type=Path, default=DEFAULT_SCENARIO, help="scenario YAML to run"
    )
    parser.add_argument("--json", action="store_true", help="emit the score as JSON only")
    parser.add_argument("--quiet", action="store_true", help="suppress progress logging")
    args = parser.parse_args(argv)

    if not args.quiet and not args.json:
        logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", stream=sys.stderr)
        logging.getLogger().setLevel(logging.WARNING)

    if not args.scenario.exists():
        print(f"scenario not found: {args.scenario}", file=sys.stderr)
        return 1

    scenario = Scenario.load(args.scenario)
    reports, truth, result, elapsed = run_demo(scenario)

    if args.json:
        print(json.dumps(result.as_dict(), indent=2))
    else:
        print_report(scenario, reports, truth, result, elapsed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
