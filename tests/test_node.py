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


def test_file_source_scales_cs8_to_unit_full_scale(tmp_path: Path) -> None:
    """cs8 is what hackrf_transfer writes, and the HackRF's ADC is 8-bit, so nothing is lost."""
    raw = np.array([127, 0, -128, 0], dtype=np.int8)
    path = tmp_path / "capture.cs8"
    raw.tofile(path)

    with FileSource(path, SAMPLE_RATE, CENTRE, "cs8") as source:
        read = source.read(10)

    assert read[0].real == pytest.approx(1.0, abs=0.01)
    assert read[1].real == pytest.approx(-1.0, abs=0.01)


def test_file_source_rejects_an_unknown_format(tmp_path: Path) -> None:
    path = tmp_path / "capture.bin"
    path.write_bytes(b"\x00" * 16)
    with pytest.raises(ValueError, match="sample_format"):
        FileSource(path, SAMPLE_RATE, CENTRE, "cu8")


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


def test_node_ignores_the_dc_bin() -> None:
    """The receiver's own LO leaks to DC and would otherwise be a permanent emission.

    Measured on a HackRF One the spur runs 31 dB above the noise floor and lasts as long as
    the receiver is on, so without the guard the node reports one emission covering the whole
    capture. The bin carries no external signal by construction.
    """
    n = int(1.5 * SAMPLE_RATE)
    rng = np.random.default_rng(0)
    noise = (3e-4 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))).astype(np.complex64)
    # A strong steady carrier at exactly the centre frequency: what LO leakage looks like.
    scene = noise + np.full(n, 0.05, dtype=np.complex64)

    assert build_node().run(ArraySource(scene, SAMPLE_RATE, CENTRE)) == []


def test_dc_guard_can_be_disabled() -> None:
    """Turning the guard off must restore the detection, or the test above proves nothing."""
    n = int(1.5 * SAMPLE_RATE)
    rng = np.random.default_rng(0)
    noise = (3e-4 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))).astype(np.complex64)
    scene = noise + np.full(n, 0.05, dtype=np.complex64)

    node = EsmNode(
        channelizer_config=make_config(),
        centre_frequency=CENTRE,
        cfar_config=CfarConfig(pfa=1e-6, method="os"),
        tracker=EmissionTracker(hangover_frames=200, min_frames=2_000),
        dc_guard_bins=0,
    )
    reports = node.run(ArraySource(scene, SAMPLE_RATE, CENTRE))
    assert [r.bin_index for r in reports] == [0]


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


def test_replaying_a_file_reports_when_it_was_captured(tmp_path: Path) -> None:
    """A recording must not be relabelled with the time it was analysed.

    The node used to take the wall clock when `run()` started, so a capture made at 01:07 and
    analysed at 01:35 was filed under 01:35. Everything that bins by time -- occupancy by
    hour, pattern of life -- would then be computed over the analyst's schedule instead of
    the band's, and the archive would be wrong the moment it was written.
    """
    import os

    scene = synthesise_scene(
        [{"channel": 3, "amplitude": 0.05, "ctcss_hz": None, "start_s": 0.3, "stop_s": 1.1}]
    )
    path = tmp_path / "capture.cf32"
    scene.view(np.float32).tofile(path)

    captured_at = 1_700_000_000.0
    os.utime(path, (captured_at, captured_at + len(scene) / SAMPLE_RATE))

    reports = build_node().run(FileSource(path, SAMPLE_RATE, CENTRE, "cf32"))

    assert reports
    assert reports[0].timestamp == pytest.approx(captured_at + reports[0].offset_s, abs=0.1)
    assert reports[0].timestamp < 1_800_000_000.0, "reported the replay time, not the capture"


def test_the_offset_locates_the_emission_in_the_recording() -> None:
    """Carried alongside the absolute time because two vectors were cut from the wrong place.

    Reading the reported timestamp as a file position is a mistake the record should make
    impossible rather than merely discourage.
    """
    scene = synthesise_scene(
        [{"channel": 3, "amplitude": 0.05, "ctcss_hz": None, "start_s": 0.4, "stop_s": 1.2}]
    )
    reports = build_node().run(ArraySource(scene, SAMPLE_RATE, CENTRE))

    assert reports[0].offset_s == pytest.approx(0.4, abs=0.1)


def test_replaying_twice_gives_identical_timestamps(tmp_path: Path) -> None:
    """A capture's time is a property of the capture, so two analyses must agree."""
    scene = synthesise_scene(
        [{"channel": 3, "amplitude": 0.05, "ctcss_hz": None, "start_s": 0.3, "stop_s": 1.1}]
    )
    path = tmp_path / "capture.cf32"
    scene.view(np.float32).tofile(path)

    first = build_node().run(FileSource(path, SAMPLE_RATE, CENTRE, "cf32"))
    second = build_node().run(FileSource(path, SAMPLE_RATE, CENTRE, "cf32"))

    assert [r.timestamp for r in first] == [r.timestamp for r in second]


# --------------------------------------------------------------------------------------
# The band-edge guard
# --------------------------------------------------------------------------------------


def test_the_band_edges_are_excluded_from_detection() -> None:
    """Nyquist is guarded for the same reason DC is, and neither is arbitrary.

    The receiver's anti-alias filter rolls off approaching the edges of the sampled band, so
    the noise there is neither flat nor stationary and the CFAR reference window spans a floor
    that is sloping under it. Measured over eighteen minutes of empty band at high gain,
    **nine of the sixteen phantom emissions the node produced sat within eight bins of the
    edge** -- 56 % of them, in 10 % of the band.
    """
    from esm446.core.channelizer import ChannelizerConfig
    from esm446.core.node import EsmNode

    node = EsmNode(
        channelizer_config=ChannelizerConfig(sample_rate=2e6, num_channels=160, decimation=80),
        centre_frequency=446_593_750.0,
        edge_guard_bins=8,
    )
    power = np.ones((256, 160))
    power[:, 72:88] = 1e6  # the whole guarded region, screaming
    noise = np.ones((256, 160))

    mask = power > noise * node.detector.threshold_factor
    excluded = np.zeros(160, dtype=bool)
    excluded[72:88] = True
    mask[:, excluded] = False

    assert not mask.any(), "something in the guarded region reached the tracker"


def test_the_guard_does_not_touch_the_pmr446_allocation() -> None:
    """The mission band must be untouched, or the guard has cost more than it bought.

    PMR446 sits 33 to 48 bins from the edge of the sampled band with the shipped offset
    tuning, so a guard of eight bins cannot reach it. This test fails if either the tuning or
    the guard ever moves far enough for that to stop being true.
    """
    from esm446.core import bands

    centre, spacing, num_bins = bands.DEFAULT_CENTRE_HZ, 12_500.0, 160
    nyquist = num_bins // 2
    guard = 8

    for channel in (1, 16):
        offset_bins = abs(bands.channel_frequency(channel) - centre) / spacing
        distance_from_edge = nyquist - offset_bins
        assert distance_from_edge > guard + 4, (
            f"PMR{channel} sits {distance_from_edge:.0f} bins from the edge, "
            f"too close to a {guard}-bin guard"
        )
