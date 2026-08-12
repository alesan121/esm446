"""Extract a committable test vector from a wideband capture.

A raw capture is 4 MB per second at 2 MS/s, so a useful recording is hundreds of megabytes
and cannot live in a repository that people are expected to clone. This cuts one out: a short
span around the PMR446 allocation, frequency-shifted and decimated, small enough to commit and
still carrying everything that matters about a real transmitter.

What survives the cut and what does not
---------------------------------------
Decimating to 500 kS/s keeps all sixteen channels and the emitter's own behaviour — its
deviation, its sub-audible tone, the shape of its keying transients, its frequency error.
Those are exactly what a simulator has to assume and a recording can prove.

What is lost is the wideband geometry: 40 bins instead of 160, so the vector exercises
detection, tracking, demodulation and identification but not the full channel plan. That is
the right way round. The channeliser's geometry is verified analytically and against
synthetic tones, where the expected answer is known exactly; a real recording cannot improve
on that. What a recording adds is a transmitter nobody modelled.

Centre frequency of the extract
-------------------------------
The same two constraints as the receiver apply, for the same reasons: the centre must be an
integer number of 12.5 kHz steps from channel 1 so the channels land on bins, and it must not
*be* a channel, or the extract's own DC bin would sit on one. 445.98125 MHz is two steps
below channel 1 and satisfies both, while keeping all sixteen channels inside the retained
500 kHz.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from scipy import signal as dsp

from esm446.core import bands
from esm446.core.source import SAMPLE_FORMATS

logger = logging.getLogger(__name__)

#: Centre frequency of the extracted vector (Hz). Two channel steps below PMR446 channel 1:
#: on the grid, not on a channel, and far enough from the allocation that the extract's DC
#: bin carries nothing of interest.
VECTOR_CENTRE_HZ = 445_981_250

#: Sample rate of the extracted vector (Hz). 40 bins of 12.5 kHz spans 500 kHz, which holds
#: the whole 200 kHz allocation with room either side.
VECTOR_SAMPLE_RATE_HZ = 500_000


def extract(
    source_path: Path,
    start_s: float,
    duration_s: float,
    source_centre_hz: float,
    source_rate_hz: float = 2_000_000.0,
    source_format: str = "cs8",
    target_centre_hz: float = VECTOR_CENTRE_HZ,
    target_rate_hz: float = VECTOR_SAMPLE_RATE_HZ,
) -> np.ndarray:
    """Cut, shift and decimate a span of a capture.

    Args:
        source_path: The wideband capture.
        start_s: Offset into the capture at which to start.
        duration_s: Length to extract.
        source_centre_hz: Centre frequency the capture was taken at.
        source_rate_hz: Sample rate of the capture.
        source_format: Sample format of the capture.
        target_centre_hz: Centre frequency of the extract.
        target_rate_hz: Sample rate of the extract.

    Returns:
        Complex64 samples at ``target_rate_hz``.

    Raises:
        ValueError: If the decimation is not an integer, or the requested span is not in the
            file.
    """
    ratio = source_rate_hz / target_rate_hz
    if abs(ratio - round(ratio)) > 1e-9:
        raise ValueError(f"{source_rate_hz} / {target_rate_hz} is not an integer decimation")
    decimation = int(round(ratio))

    dtype, full_scale = SAMPLE_FORMATS[source_format]
    itemsize = dtype.itemsize * 2
    offset = int(start_s * source_rate_hz) * itemsize
    count = int(duration_s * source_rate_hz) * 2

    raw = np.fromfile(source_path, dtype=dtype, count=count, offset=offset)
    if raw.size < count:
        raise ValueError(
            f"asked for {duration_s} s from {start_s} s but the file holds "
            f"{raw.size / 2 / source_rate_hz + start_s:.1f} s in total"
        )

    iq = (raw.astype(np.float32) / full_scale).view(np.complex64)

    # Remove the receiver's DC offset before anything else. The local-oscillator leakage of a
    # direct-conversion receiver sits at exactly the centre frequency, which in the recorded
    # baseband is exactly DC, so it is precisely the mean of the samples and subtracting that
    # removes it exactly.
    #
    # Skipping this is not a cosmetic loss. Decimating folds everything beyond the new Nyquist
    # back into the retained band, and the spur is strong: 31 dB above the noise floor. With
    # the first geometry tried here it aliased to +112.5 kHz, which is exactly where PMR446
    # channel 8 sits -- the anti-alias filter attenuated it but not enough, and the extract
    # ended up with the receiver's own artefact sitting on top of the transmission it was
    # supposed to preserve.
    iq -= iq.mean()

    # Shift the target centre down to DC before decimating, or everything outside the new
    # Nyquist would fold back on top of the band being kept.
    offset_hz = target_centre_hz - source_centre_hz
    t = np.arange(iq.size, dtype=np.float64) / source_rate_hz
    shifted = iq * np.exp(-2j * np.pi * offset_hz * t).astype(np.complex64)

    decimated = dsp.resample_poly(shifted, up=1, down=decimation, window=("kaiser", 9.0))
    return decimated.astype(np.complex64)


def write_vector(samples: np.ndarray, path: Path, metadata: dict) -> tuple[Path, Path]:
    """Write the extract as cs8 alongside a JSON description of it.

    cs8 halves the file against cs16 and loses nothing: the HackRF's converter is 8-bit, so
    the extra bits would be quantisation noise the recording never had. The samples are
    scaled to full scale first, because writing a weak extract without normalising buries it
    in the bottom few codes.

    Args:
        samples: Complex baseband to write.
        path: Destination, without extension.
        metadata: What the vector is, written beside it.

    Returns:
        ``(iq_path, metadata_path)``.
    """
    peak = float(np.abs(samples).max()) or 1.0
    scaled = np.clip(samples / peak * 127.0, -127, 127)
    iq_path = path.with_suffix(".cs8")
    scaled.view(np.float32).astype(np.int8).tofile(iq_path)

    metadata_path = path.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata | {"scale_peak": peak}, indent=2))
    return iq_path, metadata_path


def main(argv: list[str] | None = None) -> int:
    """Extract a test vector from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path, help="wideband capture to cut from")
    parser.add_argument("--output", type=Path, required=True, help="output path, no extension")
    parser.add_argument("--start", type=float, default=0.0, help="offset into the capture (s)")
    parser.add_argument("--duration", type=float, default=8.0, help="length to extract (s)")
    parser.add_argument("--source-centre", type=float, default=float(bands.DEFAULT_CENTRE_HZ))
    parser.add_argument("--source-rate", type=float, default=2_000_000.0)
    parser.add_argument("--format", default="cs8", choices=sorted(SAMPLE_FORMATS))
    parser.add_argument("--note", default="", help="description stored in the metadata")
    args = parser.parse_args(argv)

    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", stream=sys.stderr)
    logging.getLogger().setLevel(logging.INFO)

    samples = extract(
        source_path=args.capture,
        start_s=args.start,
        duration_s=args.duration,
        source_centre_hz=args.source_centre,
        source_rate_hz=args.source_rate,
        source_format=args.format,
    )

    iq_path, metadata_path = write_vector(
        samples,
        args.output,
        {
            "note": args.note,
            "source": args.capture.name,
            "source_centre_hz": args.source_centre,
            "source_rate_hz": args.source_rate,
            "start_s": args.start,
            "duration_s": args.duration,
            "centre_hz": VECTOR_CENTRE_HZ,
            "sample_rate_hz": VECTOR_SAMPLE_RATE_HZ,
            "num_channels": 40,
            "format": "cs8",
        },
    )
    logger.info(
        "extract: wrote %s (%.1f MB) and %s",
        iq_path,
        iq_path.stat().st_size / 1e6,
        metadata_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
