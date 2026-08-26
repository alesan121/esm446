#!/usr/bin/env python3
"""T4a: attenuator-linearity sweep, one radiated capture per call.

Run with the machine's own poetry environment::

    poetry run python scripts/t4a_atten_sweep.py <label> <atten_db> [seconds]

Not part of the esm446 package on purpose -- it is an operator tool for one specific
measurement campaign, not a library import anyone else's code should depend on. Same
convention as scripts/overnight_survey.py, and reuses its `capture()` (hackrf_transfer,
not SoapySource -- see that module's docstring for why).

Protocol: PMR446 handset with its own antenna, HackRF receiving on its own antenna with an
attenuator combo in series between antenna and HackRF. Fixed setup between points -- handset,
HackRF, receive antenna, cable, operator position unchanged; only the attenuator combo (and,
across separate sweeps, the handset's TX power setting) changes. Check the receive antenna
connector's torque before EVERY point, not just at the start of the sweep -- two SMA
disconnects mid-sweep on 2026-08-26 were traced to mechanical tension from the cable run, not
bad luck (see docs/data/t4a_atten_sweep_2026-08-26.json for the incident record).

Streaming, not single-shot
---------------------------
An earlier version of this script decoded the whole capture to one complex64 array and ran a
single `channelizer.process()` call over it, holding the full spectra array in memory at
once -- roughly 2 GB for a 60 s capture at this configuration, on a machine with 7.6 GB of
RAM. That drove the system into swap and hung on shutdown. This version reads and processes
the file in fixed-size chunks, using `PolyphaseChannelizer`'s tested cross-call streaming
continuity (state carries across `process()` calls -- `tests/test_channelizer.py`'s
`test_streaming_matches_single_block` is what that guarantee rests on), and keeps only the
running accumulators below (`ChannelPowerSelector`, `BinStats`) between chunks -- a handful of
floats, not the whole capture.

Two passes over the same file, because bin selection needs to see every frame before either
class knows which single bin to focus its second pass on:

1. `ChannelPowerSelector` accumulates total power per bin across all chunks, then reports
   the strongest bin outside the DC and Nyquist-edge guards (same convention as
   `esm446.core.node.EsmNode`).
2. `BinStats` accumulates just that one bin's per-frame power across a second pass, and
   reports mean dBFS (linear-domain mean, converted once -- not a mean of dB values), peak,
   min, and spread (the PTT-continuity / connector-stability diagnostic).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from esm446.core import bands  # noqa: E402
from esm446.core.channelizer import ChannelizerConfig, PolyphaseChannelizer  # noqa: E402
from scripts.overnight_survey import capture  # noqa: E402

LNA_DB = 32.0
VGA_DB = 40.0
SAMPLE_RATE = 2_000_000.0
NUM_CHANNELS = 160
DECIMATION = 80
SETTLE_FRAMES = 50  # filter-bank group delay, same margin used throughout tests/
DC_GUARD_BINS = 1  # same default as EsmNode
EDGE_GUARD_BINS = max(1, NUM_CHANNELS // 20)  # same 5% convention as EsmNode
CHUNK_SECONDS = 1.0  # ~4 MB raw + ~8 MB decoded per chunk, not ~2 GB for a whole 60 s capture
SPREAD_WARNING_DB = 10.0

OUTPUT_ROOT = Path.home() / "esm446_t4a"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


def excluded_bin_mask(
    num_channels: int = NUM_CHANNELS,
    dc_guard_bins: int = DC_GUARD_BINS,
    edge_guard_bins: int = EDGE_GUARD_BINS,
) -> np.ndarray:
    """Bins to exclude from "strongest bin" selection: DC leakage and the Nyquist edge.

    Bin 0 carries the receiver's own LO leakage (measured elsewhere in this project at
    31 dB over the floor) -- picking it as the "signal" bin would silently track the
    receiver's own artefact instead of the handset at every attenuation point. The edge
    guard excludes the anti-alias roll-off around Nyquist, which is neither flat nor
    stationary (see esm446.core.node.EsmNode's own edge_guard_bins for the same reasoning).
    """
    excluded = np.zeros(num_channels, dtype=bool)
    excluded[:dc_guard_bins] = True
    if dc_guard_bins > 1:
        excluded[-(dc_guard_bins - 1) :] = True
    nyquist = num_channels // 2
    excluded[nyquist - edge_guard_bins : nyquist + edge_guard_bins] = True
    return excluded


class ChannelPowerSelector:
    """Pass 1: accumulate total power per bin across chunks, then pick the strongest.

    Streaming equivalent of ``power.mean(axis=0)`` over a full-capture spectra array --
    never holds more than one chunk's spectra at a time.
    """

    def __init__(
        self, num_channels: int = NUM_CHANNELS, settle_frames: int = SETTLE_FRAMES
    ) -> None:
        self._sum = np.zeros(num_channels, dtype=np.float64)
        self._frames_to_drop = settle_frames
        self._frames_seen = 0

    def update(self, spectra: np.ndarray) -> None:
        """Fold in one chunk's worth of channeliser output frames."""
        if spectra.shape[0] == 0:
            return
        if self._frames_to_drop > 0:
            drop = min(self._frames_to_drop, spectra.shape[0])
            spectra = spectra[drop:]
            self._frames_to_drop -= drop
            if spectra.shape[0] == 0:
                return
        self._sum += (np.abs(spectra) ** 2).sum(axis=0)
        self._frames_seen += spectra.shape[0]

    @property
    def n_frames_seen(self) -> int:
        return self._frames_seen

    def select_bin(self, excluded: np.ndarray) -> int:
        """The strongest accumulated bin outside `excluded`."""
        candidate = np.where(excluded, -np.inf, self._sum)
        return int(np.argmax(candidate))


class BinStats:
    """Pass 2: accumulate mean/peak/min power for one specific bin across chunks."""

    def __init__(self, bin_index: int, settle_frames: int = SETTLE_FRAMES) -> None:
        self._bin_index = bin_index
        self._frames_to_drop = settle_frames
        self._sum = 0.0
        self._count = 0
        self._max_db = -np.inf
        self._min_db = np.inf

    def update(self, spectra: np.ndarray) -> None:
        """Fold in one chunk's worth of channeliser output frames."""
        if spectra.shape[0] == 0:
            return
        if self._frames_to_drop > 0:
            drop = min(self._frames_to_drop, spectra.shape[0])
            spectra = spectra[drop:]
            self._frames_to_drop -= drop
            if spectra.shape[0] == 0:
                return
        power = np.abs(spectra[:, self._bin_index]) ** 2
        power_db = 10.0 * np.log10(np.maximum(power, 1e-30))
        self._sum += float(power.sum())
        self._count += power.shape[0]
        self._max_db = max(self._max_db, float(power_db.max()))
        self._min_db = min(self._min_db, float(power_db.min()))

    @property
    def n_frames(self) -> int:
        return self._count

    def result(self) -> dict:
        if self._count == 0:
            return {
                "n_frames": 0,
                "mean_dbfs": None,
                "peak_dbfs": None,
                "min_dbfs": None,
                "spread_db": None,
            }
        mean_db = float(10.0 * np.log10(max(self._sum / self._count, 1e-30)))
        spread_db = self._max_db - self._min_db
        return {
            "n_frames": self._count,
            "mean_dbfs": round(mean_db, 2),
            "peak_dbfs": round(self._max_db, 2),
            "min_dbfs": round(self._min_db, 2),
            "spread_db": round(spread_db, 2),
        }


@dataclass
class SweepConfig:
    lna_db: float = LNA_DB
    vga_db: float = VGA_DB
    sample_rate: float = SAMPLE_RATE
    num_channels: int = NUM_CHANNELS
    decimation: int = DECIMATION
    chunk_seconds: float = CHUNK_SECONDS


def _chunks(path: Path, bytes_per_chunk: int):
    with path.open("rb") as f:
        while True:
            raw_bytes = f.read(bytes_per_chunk)
            if not raw_bytes:
                return
            raw = np.frombuffer(raw_bytes, dtype=np.int8)
            yield (raw.astype(np.float32) / 128.0).view(np.complex64)


def measure(
    label: str, atten_db: float, seconds: float = 10.0, config: SweepConfig | None = None
) -> dict:
    """Run one capture at the given attenuation and report the strongest-bin power stats."""
    cfg = config or SweepConfig()
    path = OUTPUT_ROOT / f"{label}.cs8"
    outcome = capture(path, seconds, cfg.lna_db, cfg.vga_db)
    if not outcome["completed"]:
        print(f"{label}: CAPTURE FAILED: {outcome}")
        return {"label": label, "atten_db": atten_db, "seconds": seconds, "failed": True, **outcome}

    channelizer_config = ChannelizerConfig(
        sample_rate=cfg.sample_rate, num_channels=cfg.num_channels, decimation=cfg.decimation
    )
    bytes_per_chunk = int(cfg.chunk_seconds * cfg.sample_rate) * 2  # interleaved I/Q, 1 byte each

    selector = ChannelPowerSelector(cfg.num_channels)
    channelizer = PolyphaseChannelizer(channelizer_config)
    for iq in _chunks(path, bytes_per_chunk):
        selector.update(channelizer.process(iq))

    if selector.n_frames_seen == 0:
        print(f"{label}: no frames survived settle-drop -- capture too short")
        return {"label": label, "atten_db": atten_db, "seconds": seconds, "failed": True}

    excluded = excluded_bin_mask(cfg.num_channels)
    bin_index = selector.select_bin(excluded)

    stats = BinStats(bin_index)
    channelizer2 = PolyphaseChannelizer(channelizer_config)
    for iq in _chunks(path, bytes_per_chunk):
        stats.update(channelizer2.process(iq))

    outcome_stats = stats.result()
    if outcome_stats["spread_db"] is not None and outcome_stats["spread_db"] > SPREAD_WARNING_DB:
        print(
            f"WARNING: {outcome_stats['spread_db']:.1f} dB spread within one capture -- "
            "PTT/connector may not have been stable; check this point"
        )

    bin_freqs = bands.bin_frequencies(bands.DEFAULT_CENTRE_HZ, cfg.sample_rate, cfg.num_channels)
    result = {
        "label": label,
        "atten_db": atten_db,
        "seconds": seconds,
        "bin_index": bin_index,
        "bin_freq_hz": float(bin_freqs[bin_index]),
        "overruns": outcome["overruns"],
        **outcome_stats,
    }
    print(result)
    path.unlink()
    return result


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        print("usage: t4a_atten_sweep.py <label> <atten_db> [seconds]")
        raise SystemExit(1)
    secs = float(sys.argv[3]) if len(sys.argv) == 4 else 10.0
    measure(sys.argv[1], float(sys.argv[2]), secs)
