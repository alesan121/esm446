"""Channeliser throughput benchmark: v0 prototype versus polyphase filter bank.

Reports CPU seconds per second of captured signal — the metric that decides whether a
receiver keeps up with its own front end. A ratio above 1.0 means the node is dropping
signal.

The comparison is deliberately unfavourable to the new implementation: v0 is measured on
its own configuration (800 kHz, 57 channels) while the PFB is measured on the configuration
it actually ships with (2 MS/s, 160 channels), which is 2.5x the bandwidth and 2.8x the
channel count. Normalising by signal duration is what makes the two comparable at all.

Run with ``python -m esm446.bench`` or ``esm446-bench``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass

import numpy as np
from scipy import signal as dsp

from esm446.core import bands
from esm446.core.channelizer import ChannelizerConfig, PolyphaseChannelizer

# v0 configuration, reproduced from legacy/channelizer.py.
V0_SAMPLE_RATE = 800_000
V0_BLOCK_SIZE = 262_144
V0_AUDIO_RATE = 12_000
V0_NUM_CHANNELS = 57

# Shipping configuration.
SAMPLE_RATE = 2_000_000
NUM_CHANNELS = 160
DECIMATION = 80


@dataclass
class Result:
    """One benchmark measurement."""

    label: str
    sample_rate: float
    num_channels: int
    signal_seconds: float
    cpu_seconds: float

    @property
    def realtime_ratio(self) -> float:
        """CPU seconds per second of signal. Below 1.0 keeps up; above 1.0 drops signal."""
        return self.cpu_seconds / self.signal_seconds

    @property
    def realtime_margin(self) -> float:
        """How many times faster than real time. Below 1.0 means it cannot keep up."""
        return self.signal_seconds / self.cpu_seconds

    def describe(self) -> str:
        verdict = "keeps up" if self.realtime_ratio < 1.0 else "DROPS SIGNAL"
        return (
            f"{self.label:<34} {self.sample_rate / 1e6:>5.2f} MS/s "
            f"{self.num_channels:>4d} ch  "
            f"{self.realtime_ratio:>8.4f} cpu-s/s  "
            f"{self.realtime_margin:>7.1f}x real time  {verdict}"
        )


def _v0_get_channel_iq(
    iq: np.ndarray, ch_freq: float, centre: float, samp_rate: float, audio_rate: float
) -> np.ndarray:
    """Verbatim reproduction of legacy/channelizer.py::get_channel_iq.

    Kept byte-for-byte faithful, including rebuilding the FIR on every call, because the
    point of the benchmark is to measure what v0 actually did.
    """
    offset = ch_freq - centre
    t = np.arange(len(iq)) / samp_rate
    shifted = iq * np.exp(-2j * np.pi * offset * t).astype(np.complex64)
    dec = int(samp_rate / audio_rate)
    lp = dsp.firwin(101, audio_rate / samp_rate)
    filt = dsp.lfilter(lp, 1.0, shifted)
    return filt[::dec]


def benchmark_v0(blocks: int = 3) -> Result:
    """Measure the v0 per-channel mixer-and-filter channeliser."""
    rng = np.random.default_rng(0)
    iq = (rng.standard_normal(V0_BLOCK_SIZE) + 1j * rng.standard_normal(V0_BLOCK_SIZE)).astype(
        np.complex64
    ) * 0.01
    centre = 446_093_500
    channels = [centre - 350_000 + i * 12_500 for i in range(V0_NUM_CHANNELS)]

    start = time.perf_counter()
    for _ in range(blocks):
        for ch_freq in channels:
            _v0_get_channel_iq(iq, ch_freq, centre, V0_SAMPLE_RATE, V0_AUDIO_RATE)
    cpu = time.perf_counter() - start

    return Result(
        label="v0 per-channel mixer + lfilter",
        sample_rate=V0_SAMPLE_RATE,
        num_channels=V0_NUM_CHANNELS,
        signal_seconds=blocks * V0_BLOCK_SIZE / V0_SAMPLE_RATE,
        cpu_seconds=cpu,
    )


def benchmark_pfb(seconds: float = 2.0) -> Result:
    """Measure the polyphase filter bank on the shipping configuration."""
    config = ChannelizerConfig(
        sample_rate=SAMPLE_RATE, num_channels=NUM_CHANNELS, decimation=DECIMATION
    )
    bank = PolyphaseChannelizer(config)

    block_size = 262_144
    num_blocks = max(1, int(seconds * SAMPLE_RATE / block_size))
    rng = np.random.default_rng(0)
    block = (rng.standard_normal(block_size) + 1j * rng.standard_normal(block_size)).astype(
        np.complex64
    ) * 0.01

    bank.process(block)  # prime the filter state; excluded from the measurement
    start = time.perf_counter()
    for _ in range(num_blocks):
        bank.process(block)
    cpu = time.perf_counter() - start

    return Result(
        label="polyphase filter bank",
        sample_rate=SAMPLE_RATE,
        num_channels=NUM_CHANNELS,
        signal_seconds=num_blocks * block_size / SAMPLE_RATE,
        cpu_seconds=cpu,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-ratio",
        type=float,
        default=0.1,
        help="fail if the PFB exceeds this many CPU seconds per signal second (default: 0.1)",
    )
    parser.add_argument("--skip-v0", action="store_true", help="skip the slow v0 baseline")
    parser.add_argument("--json", action="store_true", help="emit machine-readable results")
    args = parser.parse_args(argv)

    results = []
    if not args.skip_v0:
        results.append(benchmark_v0())
    pfb = benchmark_pfb()
    results.append(pfb)

    if args.json:
        print(json.dumps([asdict(r) | {"realtime_ratio": r.realtime_ratio} for r in results]))
    else:
        print(f"Receiver centre {bands.DEFAULT_CENTRE_HZ / 1e6:.6f} MHz (PMR446 channel 8)\n")
        for result in results:
            print(result.describe())
        if len(results) == 2:
            speedup = results[0].realtime_ratio / results[1].realtime_ratio
            print(f"\nSpeedup, normalised by signal duration: {speedup:.0f}x")

    if pfb.realtime_ratio > args.max_ratio:
        print(
            f"\nFAIL: {pfb.realtime_ratio:.4f} cpu-s/s exceeds budget {args.max_ratio}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
