"""Generate IQ and ground truth from a scenario file.

Writes what the node consumes and what the metrics score against, so a scene can be produced
once and reused by a test, the benchmark and a demonstration rather than being rebuilt inline
each time.

Usage::

    esm446-sim scenarios/demo.yaml --output out/demo
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from esm446.sim.scenario import Scenario

logger = logging.getLogger(__name__)


def write_scene(scenario: Scenario, output: Path, sample_format: str = "cf32") -> tuple[Path, Path]:
    """Generate a scenario and write its IQ and ground truth.

    Args:
        scenario: The scene to generate.
        output: Output path without extension.
        sample_format: ``cf32`` or ``cs16``.

    Returns:
        ``(iq_path, truth_path)``.
    """
    iq, truth = scenario.generate()
    output.parent.mkdir(parents=True, exist_ok=True)

    iq_path = output.with_suffix(f".{sample_format}")
    if sample_format == "cs16":
        # Scale to full scale before quantising, or a weak scene lands entirely in the
        # bottom few bits and the quantisation noise swamps everything in it.
        peak = float(np.abs(iq).max()) or 1.0
        interleaved = (iq / peak * 32767.0).view(np.float32).astype(np.int16)
        interleaved.tofile(iq_path)
    else:
        iq.view(np.float32).tofile(iq_path)

    truth_path = output.with_suffix(".truth.json")
    truth_path.write_text(
        json.dumps(
            {
                "scenario": scenario.name,
                "duration_s": scenario.duration_s,
                "sample_rate": scenario.sample_rate,
                "centre_frequency": scenario.centre_frequency,
                "seed": scenario.seed,
                "emissions": [emission.as_dict() for emission in truth],
            },
            indent=2,
        )
    )
    return iq_path, truth_path


def main(argv: list[str] | None = None) -> int:
    """Generate a scenario from the command line.

    Args:
        argv: Command line arguments, or ``None`` to read from ``sys.argv``.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", type=Path, help="scenario YAML to generate")
    parser.add_argument(
        "--output", type=Path, default=Path("out/scene"), help="output path without extension"
    )
    parser.add_argument("--format", default="cf32", choices=["cf32", "cs16"])
    parser.add_argument("--seed", type=int, help="override the scenario's seed")
    args = parser.parse_args(argv)

    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", stream=sys.stderr)
    logging.getLogger().setLevel(logging.INFO)

    if not args.scenario.exists():
        logger.error("simulate: scenario not found: %s", args.scenario)
        return 1

    scenario = Scenario.load(args.scenario)
    if args.seed is not None:
        scenario.seed = args.seed

    iq_path, truth_path = write_scene(scenario, args.output, args.format)
    size_mb = iq_path.stat().st_size / 1e6
    logger.info("simulate: wrote %s (%.1f MB) and %s", iq_path, size_mb, truth_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
