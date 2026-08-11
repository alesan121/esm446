"""Command line entry point for the ESM-446 node.

Runs the pipeline over either a live SDR or a recorded IQ file and writes one JSON object
per emission to stdout. The two modes go through the same code path; only the `IQSource`
differs, which is what lets the whole system be exercised without hardware.

Usage::

    esm446-node --file capture.cs16 --format cs16
    esm446-node --sdr
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from esm446.config.config import settings
from esm446.core.calibration import PowerCalibration
from esm446.core.channelizer import ChannelizerConfig
from esm446.core.detector import CfarConfig
from esm446.core.node import DEFAULT_BLOCK_SIZE, EsmNode
from esm446.core.rfchain import quantise_gains
from esm446.core.source import FileSource, IQSource

logger = logging.getLogger(__name__)


def configure_logging(level: str) -> None:
    """Initialise root logging once, at startup.

    Args:
        level: Logging level name, already validated by `Settings`.
    """
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        stream=sys.stderr,
    )
    logging.getLogger().setLevel(level)


def build_node() -> EsmNode:
    """Assemble the node from validated settings.

    Returns:
        A node wired with the configured channeliser geometry, CFAR design point,
        receiver gains and power calibration.
    """
    channelizer_config = ChannelizerConfig(
        sample_rate=settings.SDR_SAMPLE_RATE_HZ,
        num_channels=settings.CHANNELIZER_NUM_CHANNELS,
        decimation=settings.CHANNELIZER_DECIMATION,
        taps_per_phase=settings.CHANNELIZER_TAPS_PER_PHASE,
    )
    return EsmNode(
        channelizer_config=channelizer_config,
        centre_frequency=settings.SDR_CENTRE_FREQ_HZ,
        cfar_config=CfarConfig(pfa=settings.CFAR_PFA, method=settings.CFAR_METHOD),
        calibration=PowerCalibration.load(Path(settings.CALIBRATION_PATH)),
        gains=quantise_gains(
            settings.SDR_LNA_GAIN_DB, settings.SDR_VGA_GAIN_DB, settings.SDR_AMP_ENABLED
        ),
        expected_ctcss_hz=settings.CTCSS_EXPECTED_TONE_HZ,
    )


def build_source(args: argparse.Namespace) -> IQSource:
    """Open the IQ source the arguments select.

    Args:
        args: Parsed command line arguments.

    Returns:
        A file replay source, or a live SDR source.
    """
    if args.file:
        return FileSource(
            path=Path(args.file),
            sample_rate=settings.SDR_SAMPLE_RATE_HZ,
            centre_frequency=settings.SDR_CENTRE_FREQ_HZ,
            sample_format=args.format,
        )

    # Imported here rather than at module scope so that replaying a file needs no SDR
    # stack installed at all.
    from esm446.core.source import SoapySource

    return SoapySource(
        sample_rate=settings.SDR_SAMPLE_RATE_HZ,
        centre_frequency=settings.SDR_CENTRE_FREQ_HZ,
        lna_gain_db=settings.SDR_LNA_GAIN_DB,
        vga_gain_db=settings.SDR_VGA_GAIN_DB,
        amp_enabled=settings.SDR_AMP_ENABLED,
        driver=settings.SDR_DRIVER,
    )


def main(argv: list[str] | None = None) -> int:
    """Run the node.

    Args:
        argv: Command line arguments, or ``None`` to read from ``sys.argv``.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--file", help="replay IQ from a recorded file")
    mode.add_argument("--sdr", action="store_true", help="capture live from the configured SDR")
    parser.add_argument(
        "--format",
        default="cf32",
        choices=["cf32", "cs16"],
        help="sample format of --file (default: cf32)",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=DEFAULT_BLOCK_SIZE,
        help=f"complex samples per read (default: {DEFAULT_BLOCK_SIZE})",
    )
    args = parser.parse_args(argv)

    configure_logging(settings.LOG_LEVEL)

    try:
        source = build_source(args)
    except (FileNotFoundError, ValueError, ImportError) as error:
        logger.error("node: cannot open source: %s", error)
        return 1

    node = build_node()
    for report in node.run(source, block_size=args.block_size):
        print(json.dumps(report.as_dict()), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
