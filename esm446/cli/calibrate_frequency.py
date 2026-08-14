"""Measure the receiver's frequency error against a transmitter better than it is.

Reads captures of a reference signal and reports how far the receiver's crystal is out, in
hertz and in parts per million, and what that implies at the frequency the node operates on.

Two references are supported and they are not equivalent. The default, ``--reference lte``,
locates the unused centre subcarrier of a cellular downlink; it is what produced the figure
in ``docs/04_link_budget.md`` and it needs only an antenna and coverage. ``--reference dvbt``
finds the centre of a television multiplex from its band edges, for sites with no cellular
service. See `esm446.core.frequency` for why the other candidates were eliminated.

Give several captures taken at *different* local oscillators. That is the control, not a
refinement: a receiver artefact sits at a fixed baseband offset and would masquerade as a
carrier, whereas a real emission stays put in absolute frequency as the oscillator moves
under it. The scatter between them is what the tool reports as its confidence.

Usage::

    for f in 815200000 815600000 816400000 816800000; do
        hackrf_transfer -r lte_$f.cs8 -f $f -s 2000000 -n 4000000 -l 32 -g 20
    done
    esm446-calibrate-frequency lte_*.cs8 --centre 815.2e6 815.6e6 816.4e6 816.8e6 --rate 2e6
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from esm446.core.frequency import measure_frequency_error, measure_notch_error
from esm446.core.source import SAMPLE_FORMATS

logger = logging.getLogger(__name__)


#: How many complex samples to read from each capture. The estimator averages periodograms,
#: so accuracy improves with the square root of this and the returns flatten quickly, whereas
#: memory does not: a capture held as complex64 costs eight bytes a sample, and reading a set
#: of wideband captures whole exhausts this machine. Four million samples is a fifth of a
#: second at 20 MS/s, sixteen averages of the transform, and 32 MB.
DEFAULT_SAMPLES = 4_000_000


def _load(path: Path, sample_format: str, samples: int) -> np.ndarray:
    """Read the leading part of an interleaved IQ capture into complex baseband."""
    dtype, full_scale = SAMPLE_FORMATS[sample_format]
    raw = np.fromfile(path, dtype=dtype, count=2 * samples)
    if raw.size % 2:
        raw = raw[:-1]
    return (raw.astype(np.float32) / full_scale).view(np.complex64)


def main(argv: list[str] | None = None) -> int:
    """Measure the frequency error from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path, nargs="+", help="IQ captures of the reference")
    parser.add_argument(
        "--centre",
        type=float,
        nargs="+",
        required=True,
        help="frequency each capture was taken at, or one value shared by all of them",
    )
    parser.add_argument("--rate", type=float, required=True, help="sample rate of the captures")
    parser.add_argument("--format", default="cs8", choices=sorted(SAMPLE_FORMATS))
    parser.add_argument(
        "--reference",
        default="lte",
        choices=("lte", "dvbt"),
        help="which transmitter to measure against; see the module docstring",
    )
    parser.add_argument(
        "--nominal",
        type=float,
        help=(
            "the carrier's licensed centre. Required for --reference lte; the television "
            "method takes the nearest channel from the raster if it is omitted"
        ),
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_SAMPLES,
        help=f"complex samples to read from each capture (default {DEFAULT_SAMPLES})",
    )
    parser.add_argument("--json", action="store_true", help="emit the measurement as JSON")
    args = parser.parse_args(argv)

    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", stream=sys.stderr)
    logging.getLogger().setLevel(logging.WARNING if args.json else logging.INFO)

    missing = [path for path in args.capture if not path.exists()]
    if missing:
        logger.error("calibrate: no capture at %s", ", ".join(str(path) for path in missing))
        return 1

    centres = args.centre
    if len(centres) == 1:
        centres = centres * len(args.capture)
    if len(centres) != len(args.capture):
        logger.error(
            "calibrate: %d captures but %d centre frequencies. Give one per capture, or one "
            "shared by all of them.",
            len(args.capture),
            len(args.centre),
        )
        return 1

    if args.reference == "lte" and args.nominal is None:
        logger.error(
            "calibrate: --nominal is required for --reference lte. The carrier centre cannot "
            "be inferred: an OFDM carrier is flat across its occupied bandwidth, so its "
            "strongest bin falls at random within several megahertz of the centre. Read the "
            "carrier off a survey and pass it; it will be on the 100 kHz raster."
        )
        return 1

    if args.reference == "dvbt" and len(args.capture) > 1:
        logger.warning("calibrate: the television method reads one capture; using the first")

    try:
        if args.reference == "lte":
            # A generator, so each capture is read, measured and released before the next
            # is opened. Building a list here exhausts memory on a wideband set.
            captures = (
                (_load(path, args.format, args.samples), centre)
                for path, centre in zip(args.capture, centres, strict=True)
            )
            error = measure_notch_error(captures, args.rate, args.nominal)
        else:
            iq = _load(args.capture[0], args.format, args.samples)
            error = measure_frequency_error(iq, args.rate, centres[0], args.nominal)
    except ValueError as problem:
        logger.error(
            "calibrate: %s. The captures must contain the reference at usable strength; a "
            "band with nothing in it cannot be measured against.",
            problem,
        )
        return 1

    if args.json:
        print(
            json.dumps(
                {
                    "reference": args.reference,
                    "captures": len(args.capture),
                    "measured_hz": error.measured_hz,
                    "nominal_hz": error.nominal_hz,
                    "offset_hz": error.offset_hz,
                    "ppm": error.ppm,
                    "confidence_hz": error.confidence_hz,
                    "confidence_basis": error.basis,
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
