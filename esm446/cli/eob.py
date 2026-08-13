"""Render an Electronic Order of Battle from a store of emissions.

The node writes one record per emission and stops there. This is the step that turns that
archive into the product: which emitters are on the band, how many of them the evidence
actually supports, which channels carry the traffic, and at what hours.

Text output is for reading; ``--json`` is for the V&V report and for anything downstream that
wants the figures rather than the prose. Both come from the same functions, so the numbers in
a report and the numbers on a terminal cannot disagree.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from esm446.analysis.eob import describe, summarise
from esm446.io.sinks import read_reports

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    """Render an order of battle from the command line.

    Args:
        argv: Command-line arguments, or ``None`` to read ``sys.argv``.

    Returns:
        Process exit status. ``1`` when the store cannot be read.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("store", type=Path, help="emission store written by esm446-node")
    parser.add_argument("--json", action="store_true", help="emit the figures as JSON")
    parser.add_argument(
        "--since",
        type=float,
        default=None,
        help="ignore emissions before this Unix time",
    )
    parser.add_argument(
        "--channel",
        type=int,
        default=None,
        help="restrict to one PMR446 channel",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", stream=sys.stderr)
    # The analysis logs how many emissions became how many emitters, which is a claim worth
    # seeing; on stderr, so that --json stays pipeable.
    logging.getLogger().setLevel(logging.WARNING if args.json else logging.INFO)

    try:
        reports = read_reports(args.store)
    except (FileNotFoundError, ValueError) as error:
        logger.error("eob: %s", error)
        return 1

    if args.since is not None:
        reports = [r for r in reports if r.timestamp >= args.since]
    if args.channel is not None:
        reports = [r for r in reports if r.pmr_channel == args.channel]

    if args.json:
        print(json.dumps(summarise(reports), indent=2))
    else:
        print(describe(reports))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
