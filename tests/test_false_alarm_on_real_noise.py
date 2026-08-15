"""The false alarm rate against real receiver noise, not only against synthetic noise.

Every false-alarm test in this project used Gaussian noise, which is stationary by
construction. The first ambient capture ever put through the node produced eight
twenty-second emissions on an empty band, and the reason was not the detector's threshold
but its noise *estimate*: the level was held for 64 frames while the real one moved faster
than that, so the whole band crossed at once whenever it drifted.

Synthetic noise cannot show this. These vectors can, and they are committed for exactly that
reason: `receiver_noise_lna32_vga20` is the receiver with its antenna disconnected, so it
contains no external signal at all, and `ambient_noise_lna32_vga40` is the band at a gain
where the converter no longer masks the environment, with no emission anywhere in it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from esm446.core.channelizer import ChannelizerConfig, PolyphaseChannelizer
from esm446.core.detector import CfarConfig, CfarDetector

RECEIVER_NOISE = Path("tests/data/receiver_noise_lna32_vga20.cs8")
AMBIENT_NOISE = Path("tests/data/ambient_noise_lna32_vga40.cs8")

pytestmark = pytest.mark.skipif(not RECEIVER_NOISE.exists(), reason="noise vectors not present")


def bin_power(path: Path) -> np.ndarray:
    """Channelised power for a committed noise vector."""
    metadata = json.loads(path.with_suffix(".json").read_text())
    raw = np.fromfile(path, dtype=np.int8)
    iq = (raw.astype(np.float32) / 128.0).view(np.complex64)
    config = ChannelizerConfig(
        sample_rate=metadata["sample_rate_hz"],
        num_channels=metadata["num_channels"],
        decimation=metadata["num_channels"] // 2,
    )
    return (np.abs(PolyphaseChannelizer(config).process(iq)) ** 2).astype(np.float64)


def false_alarm_rate(power: np.ndarray, track_level: bool, dc_guard: int = 1) -> float:
    """Crossing rate, with the DC bin excluded exactly as the node excludes it."""
    detector = CfarDetector(CfarConfig(pfa=1e-8, method="os", track_level=track_level))
    mask = power > detector.noise_estimate(power) * detector.threshold_factor
    mask[:, :dc_guard] = False
    return float(mask.mean())


@pytest.fixture(scope="module")
def receiver_noise() -> np.ndarray:
    return bin_power(RECEIVER_NOISE)


@pytest.fixture(scope="module")
def ambient_noise() -> np.ndarray:
    if not AMBIENT_NOISE.exists():
        pytest.skip("ambient vector not present")
    return bin_power(AMBIENT_NOISE)


# --------------------------------------------------------------------------------------
# What the vectors contain
# --------------------------------------------------------------------------------------


def test_the_receiver_noise_vector_contains_no_signal(receiver_noise: np.ndarray) -> None:
    """With the antenna disconnected the only thing above the floor is the receiver itself.

    If this ever fails, something was radiating into a disconnected SMA connector and the
    vector is no longer what its description says it is.
    """
    shape_db = 10 * np.log10(receiver_noise.mean(axis=0))
    floor = np.median(shape_db)

    above = [b for b in range(1, len(shape_db)) if shape_db[b] - floor > 6.0]
    assert not above, f"bins above the floor other than DC: {above}"


def test_the_dc_spur_is_the_only_thing_the_guard_has_to_remove(
    receiver_noise: np.ndarray,
) -> None:
    """And it is 28 dB over the floor, which is why leaving it in swamps every other figure.

    Measured without the guard the crossing rate is 6.2e-3 and every one of those crossings
    is bin 0. It is the single most misleading number available in this system.
    """
    detector = CfarDetector(CfarConfig(pfa=1e-8, method="os", track_level=False))
    mask = receiver_noise > detector.noise_estimate(receiver_noise) * detector.threshold_factor

    assert mask[:, 0].mean() > 0.9, "the DC spur should cross in nearly every frame"
    assert mask[:, 1:].sum() == 0, "with DC excluded there should be nothing left"


def test_the_ambient_vector_contains_no_emission_either(ambient_noise: np.ndarray) -> None:
    """It is noise at high gain, not traffic: committing somebody's transmission would be a
    different matter entirely -- see docs/06_legal_ethics.md."""
    shape_db = 10 * np.log10(ambient_noise.mean(axis=0))
    floor = np.median(shape_db)

    above = [b for b in range(1, len(shape_db)) if shape_db[b] - floor > 6.0]
    assert not above, f"the vector contains an emission at bins {above}"


# --------------------------------------------------------------------------------------
# The rate itself
# --------------------------------------------------------------------------------------


def test_pure_receiver_noise_produces_no_false_alarms(receiver_noise: np.ndarray) -> None:
    """Zero crossings on this vector -- but see the caveat before trusting what that implies.

    Four million cells, not one crossing. That is a true, useful regression check: nothing in
    the detector regresses to firing on this vector. It does **not** show "the detector itself
    is sound" the way it once claimed to here -- this vector was captured with the antenna
    disconnected and the SMA port open, and at this gain the ADC occupies only ~2 bits (5 of
    256 int8 codes; see docs/05_vv_report.md). A signal quantised that hard has almost no
    dynamic range for a threshold to cross regardless of whether the CFAR math is sound, so
    zero crossings here is closer to "nothing to cross" than to "correctly rejected". What
    would show soundness is the same test against a properly-terminated or antenna-connected
    capture at this gain, which does not exist yet.
    """
    assert false_alarm_rate(receiver_noise, track_level=False) == 0.0
    assert false_alarm_rate(receiver_noise, track_level=True) == 0.0


def test_tracking_the_level_cuts_the_rate_on_ambient_noise(ambient_noise: np.ndarray) -> None:
    """The defect and its fix, on the capture that exposed both.

    Holding the noise level for 64 frames gives 3.3e-3 here against a 1e-8 design point.
    Recomputing the level every frame gives 8.3e-6: a factor of about 400, for 0.03 CPU
    seconds per signal second.
    """
    held = false_alarm_rate(ambient_noise, track_level=False)
    tracked = false_alarm_rate(ambient_noise, track_level=True)

    assert held > 1e-3, f"the vector no longer reproduces the defect ({held:.1e})"
    assert tracked < held / 50, f"held {held:.1e}, tracked {tracked:.1e}"


def test_the_broadband_bursts_disappear_entirely(ambient_noise: np.ndarray) -> None:
    """A real emission occupies one or two bins. Ten at once is the estimate being stale.

    Nearly half the crossings arrived in frames with ten or more bins crossing together, and
    the tracker's hangover then glued them into emissions of twenty seconds.
    """
    counts = {}
    for track in (False, True):
        detector = CfarDetector(CfarConfig(pfa=1e-8, method="os", track_level=track))
        mask = ambient_noise > detector.noise_estimate(ambient_noise) * detector.threshold_factor
        mask[:, :1] = False
        counts[track] = float((mask.sum(axis=1) >= 10).mean())

    assert counts[False] > 0.005, "the vector no longer reproduces the bursts"
    assert counts[True] == 0.0


def test_the_node_reports_nothing_on_a_band_with_nothing_in_it() -> None:
    """The end-to-end statement, which is the one that actually matters.

    Before the fix this capture produced eight emissions of twenty seconds each, on channels
    nobody transmitted on, with signal-to-noise ratios of about zero decibels.
    """
    from esm446.core.node import EsmNode
    from esm446.core.source import FileSource

    if not AMBIENT_NOISE.exists():
        pytest.skip("ambient vector not present")
    metadata = json.loads(AMBIENT_NOISE.with_suffix(".json").read_text())
    node = EsmNode(
        channelizer_config=ChannelizerConfig(
            sample_rate=metadata["sample_rate_hz"],
            num_channels=metadata["num_channels"],
            decimation=metadata["num_channels"] // 2,
        ),
        centre_frequency=metadata["centre_hz"],
    )
    reports = node.run(
        FileSource(AMBIENT_NOISE, metadata["sample_rate_hz"], metadata["centre_hz"], "cs8")
    )

    assert reports == [], f"{len(reports)} phantom emissions on an empty band"
