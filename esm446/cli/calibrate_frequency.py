"""Measure the receiver's frequency error against a broadcast transmitter.

Reads a capture of a reference signal and reports how far the receiver's crystal is out, in
hertz and in parts per million, and what that implies at the frequency the node operates on.

The reference has to be better than the receiver, which rules out most of what is on the air.
See `esm446.core.frequency` for why a terrestrial television multiplex is the one usable
reference available without buying anything, and what it costs if none is receivable.

Usage::

    hackrf_transfer -r dvbt.cs8 -f 594000000 -s 8000000 -n 16000000 -l 24 -g 24
    esm446-calibrate-frequency dvbt.cs8 --centre 594e6 --rate 8e6
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from esm446.core.frequency import measure_frequency_error
from esm446.core.source import SAMPLE_FORMATS

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    """Measure the frequency error from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path, help="IQ capture containing the reference")
    parser.add_argument("--centre", type=float, required=True, help="frequency it was taken at")
    parser.add_argument("--rate", type=float, required=True, help="sample rate of the capture")
    parser.add_argument("--format", default="cs8", choices=sorted(SAMPLE_FORMATS))
    parser.add_argument(
        "--nominal",
        type=float,
        help="true frequency of the reference; defaults to the nearest television channel",
    )
    parser.add_argument("--json", action="store_true", help="emit the measurement as JSON")
    args = parser.parse_args(argv)

    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", stream=sys.stderr)
    logging.getLogger().setLevel(logging.WARNING if args.json else logging.INFO)

    if not args.capture.exists():
        logger.error("calibrate: no capture at %s", args.capture)
        return 1

    dtype, full_scale = SAMPLE_FORMATS[args.format]
    raw = np.fromfile(args.capture, dtype=dtype)
    if raw.size % 2:
        raw = raw[:-1]
    iq = (raw.astype(np.float32) / full_scale).view(np.complex64)

    try:
        error = measure_frequency_error(iq, args.rate, args.centre, args.nominal)
    except ValueError as problem:
        logger.error(
            "calibrate: %s. The capture must contain a strong, flat-topped reference; a band "
            "with nothing in it cannot be measured against.",
            problem,
        )
        return 1

    if args.json:
        print(
            json.dumps(
                {
                    "measured_hz": error.measured_hz,
                    "nominal_hz": error.nominal_hz,
                    "offset_hz": error.offset_hz,
                    "ppm": error.ppm,
                    "confidence_hz": error.confidence_hz,
                    "error_at_446_hz": error.error_at(446_093_750.0),
                },
                indent=2,
            )
        )
    else:
        print(error.describe())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
