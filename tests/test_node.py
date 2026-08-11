"""End-to-end verification of the node and its IQ sources.

These tests build a synthetic PMR446 scene, push it through the whole pipeline, and check
the reports against ground truth that was known before the signal was generated. Nothing
here touches an SDR: that is the property the source abstraction exists to provide, and it
is what lets this run in continuous integration.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from esm446.core import bands
from esm446.core.channelizer import ChannelizerConfig
from esm446.core.detector import CfarConfig
from esm446.core.node import EsmNode
from esm446.core.rfchain import quantise_gains
from esm446.core.source import ArraySource, FileSource
from esm446.core.tracker import EmissionTracker

SAMPLE_RATE = 2_000_000.0
NUM_CHANNELS = 160
DECIMATION = 80
CENTRE = bands.DEFAULT_CENTRE_HZ

CTCSS_DEVIATION_HZ = 400.0
VOICE_DEVIATION_HZ = 1_500.0


def make_config() -> ChannelizerConfig:
    return ChannelizerConfig(
        sample_rate=SAMPLE_RATE, num_channels=NUM_CHANNELS, decimation=DECIMATION
    )


def synthesise_scene(
    emitters: list[dict],
    duration_s: float = 1.5,
    noise_amplitude: float = 3e-4,
    seed: int = 0,
) -> np.ndarray:
    """Build a wideband scene containing several bursty NFM emitters.

    Args:
        emitters: One dict per emitter with keys ``channel``, ``amplitude``, ``ctcss_hz``,
            ``start_s`` and ``stop_s``.
        duration_s: Total length of the scene.
        noise_amplitude: Per-component standard deviation of the additive noise.
        seed: Seed for reproducibility.

    Returns:
        Complex baseband samples at ``SAMPLE_RATE``.
    """
    rng = np.random.default_rng(seed)
    total = int(duration_s * SAMPLE_RATE)
    t = np.arange(total) / SAMPLE_RATE

    scene = (
        noise_amplitude * (rng.standard_normal(total) + 1j * rng.standard_normal(total))
    ).astype(np.complex64)

    for emitter in emitters:
        offset_hz = bands.channel_frequency(emitter["channel"]) - CENTRE

        deviation = VOICE_DEVIATION_HZ * 0.5 * np.sin(2 * np.pi * 700.0 * t)
        if emitter.get("ctcss_hz"):
            deviation += CTCSS_DEVIATION_HZ * np.sin(2 * np.pi * emitter["ctcss_hz"] * t)

        phase = 2 * np.pi * np.cumsum(deviation) / SAMPLE_RATE
        carrier = np.exp(1j * (2 * np.pi * offset_hz * t + phase))

        gate = np.zeros(total)
        start = int(emitter["start_s"] * SAMPLE_RATE)
        stop = int(emitter["stop_s"] * SAMPLE_RATE)
        gate[start:stop] = 1.0

        scene += (emitter["amplitude"] * gate * carrier).astype(np.complex64)

    return scene


def build_node(expected_ctcss_hz: float | None = None) -> EsmNode:
    return EsmNode(
        channelizer_config=make_config(),
        centre_frequency=CENTRE,
        cfar_config=CfarConfig(pfa=1e-6, method="os"),
        gains=quantise_gains(32.0, 20.0),
        expected_ctcss_hz=expected_ctcss_hz,
        tracker=EmissionTracker(hangover_frames=200, min_frames=2_000),
    )


# --------------------------------------------------------------------------------------
# Sources
# --------------------------------------------------------------------------------------


def test_array_source_serves_a_signal_in_blocks() -> None:
    samples = np.arange(1000, dtype=np.complex64)
    source = ArraySource(samples, SAMPLE_RATE, CENTRE)

    blocks = []
    while (block := source.read(256)) is not None:
        blocks.append(block)

    assert sum(b.size for b in blocks) == 1000
    np.testing.assert_array_equal(np.concatenate(blocks), samples)


def test_file_source_round_trips_cf32(tmp_path: Path) -> None:
    samples = (np.random.default_rng(0).standard_normal(2000).astype(np.float32)).view(np.complex64)
    path = tmp_path / "capture.cf32"
    samples.view(np.float32).tofile(path)

    with FileSource(path, SAMPLE_RATE, CENTRE, "cf32") as source:
        read = source.read(10_000)

    np.testing.assert_allclose(read, samples, rtol=1e-6)


def test_file_source_scales_cs16_to_unit_full_scale(tmp_path: Path) -> None:
    """Getting this scaling wrong is silent: everything runs and every power is 90 dB off."""
    raw = np.array([32767, 0, -32768, 0], dtype=np.int16)
    path = tmp_path / "capture.cs16"
    raw.tofile(path)

    with FileSource(path, SAMPLE_RATE, CENTRE, "cs16") as source:
        read = source.read(10)

    assert read[0].real == pytest.approx(1.0, abs=1e-4)
    assert read[1].real == pytest.approx(-1.0, abs=1e-4)


def test_file_source_reports_its_duration(tmp_path: Path) -> None:
    path = tmp_path / "capture.cf32"
    np.zeros(int(SAMPLE_RATE), dtype=np.complex64).view(np.float32).tofile(path)

    with FileSource(path, SAMPLE_RATE, CENTRE, "cf32") as source:
        assert source.duration_seconds == pytest.approx(1.0)


def test_file_source_rejects_an_unknown_format(tmp_path: Path) -> None:
    path = tmp_path / "capture.bin"
    path.write_bytes(b"\x00" * 16)
    with pytest.raises(ValueError, match="sample_format"):
        FileSource(path, SAMPLE_RATE, CENTRE, "cs8")


def test_file_source_rejects_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        FileSource(tmp_path / "nope.cf32", SAMPLE_RATE, CENTRE)


# --------------------------------------------------------------------------------------
# End to end
# --------------------------------------------------------------------------------------


def test_node_finds_a_single_emitter_on_the_right_channel() -> None:
    scene = synthesise_scene(
        [{"channel": 3, "amplitude": 0.05, "ctcss_hz": None, "start_s": 0.3, "stop_s": 1.1}]
    )
    reports = build_node().run(ArraySource(scene, SAMPLE_RATE, CENTRE))

    assert len(reports) == 1
    assert reports[0].pmr_channel == 3
    assert reports[0].frequency_hz == pytest.approx(bands.channel_frequency(3))
    assert reports[0].duration_s == pytest.approx(0.8, abs=0.1)


def test_node_separates_two_simultaneous_emitters() -> None:
    """Two overlapping transmissions on different channels must be two reports."""
    scene = synthesise_scene(
        [
            {"channel": 2, "amplitude": 0.05, "ctcss_hz": None, "start_s": 0.2, "stop_s": 1.2},
            {"channel": 11, "amplitude": 0.02, "ctcss_hz": None, "start_s": 0.5, "stop_s": 1.3},
        ]
    )
    reports = build_node().run(ArraySource(scene, SAMPLE_RATE, CENTRE))

    assert {r.pmr_channel for r in reports} == {2, 11}


def test_node_identifies_the_pre_shared_tone_and_classifies_accordingly() -> None:
    """The full cooperative-identification path, with ground truth known in advance."""
    scene = synthesise_scene(
        [
            {"channel": 4, "amplitude": 0.06, "ctcss_hz": 114.8, "start_s": 0.1, "stop_s": 1.4},
            {"channel": 12, "amplitude": 0.06, "ctcss_hz": None, "start_s": 0.1, "stop_s": 1.4},
        ],
        duration_s=1.5,
    )
    reports = build_node(expected_ctcss_hz=114.8).run(ArraySource(scene, SAMPLE_RATE, CENTRE))

    by_channel = {r.pmr_channel: r for r in reports}
    assert by_channel[4].ctcss_tone_hz == pytest.approx(114.8)
    assert by_channel[4].classification == "FRIEND"
    assert by_channel[12].classification == "UNKNOWN"


def test_node_reports_no_dbm_without_a_calibration() -> None:
    """An uncalibrated node must not emit a number that looks like a measurement."""
    scene = synthesise_scene(
        [{"channel": 6, "amplitude": 0.05, "ctcss_hz": None, "start_s": 0.2, "stop_s": 1.2}]
    )
    reports = build_node().run(ArraySource(scene, SAMPLE_RATE, CENTRE))

    assert reports[0].estimated_dbm is None
    assert reports[0].calibrated is False


def test_node_records_the_gains_with_every_report() -> None:
    """Without these, the archive cannot be calibrated retrospectively -- the v0 flaw."""
    scene = synthesise_scene(
        [{"channel": 6, "amplitude": 0.05, "ctcss_hz": None, "start_s": 0.2, "stop_s": 1.2}]
    )
    reports = build_node().run(ArraySource(scene, SAMPLE_RATE, CENTRE))

    assert reports[0].gains["lna_db"] == 32.0
    assert reports[0].gains["vga_db"] == 20.0


def test_node_reports_nothing_on_noise_alone() -> None:
    """The CFAR design point made concrete: an empty band produces an empty report set."""
    scene = synthesise_scene([], duration_s=1.5)
    assert build_node().run(ArraySource(scene, SAMPLE_RATE, CENTRE)) == []


def test_node_report_serialises_to_json() -> None:
    scene = synthesise_scene(
        [{"channel": 6, "amplitude": 0.05, "ctcss_hz": None, "start_s": 0.2, "stop_s": 1.2}]
    )
    reports = build_node().run(ArraySource(scene, SAMPLE_RATE, CENTRE))

    payload = reports[0].as_dict()
    assert payload["pmr_channel"] == 6
    assert "peak_power_dbfs" in payload
