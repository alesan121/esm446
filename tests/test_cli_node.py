"""Verification of the node command line entry point.

Exercises the path a reviewer actually takes on a fresh clone: point the node at a recorded
file and read JSON off stdout. No SDR is involved, which is the whole point of the source
abstraction.

Following the project testing convention, settings-dependent modules are imported inside the
tests rather than at module level.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from esm446.core import bands

SAMPLE_RATE = 2_000_000.0


def write_scene(path: Path, channel: int, duration_s: float = 1.5) -> Path:
    """Write a synthetic single-emitter scene to a cf32 file.

    Args:
        path: Destination file.
        channel: PMR446 channel the emitter occupies.
        duration_s: Length of the scene.

    Returns:
        The path written.
    """
    rng = np.random.default_rng(0)
    total = int(duration_s * SAMPLE_RATE)
    t = np.arange(total) / SAMPLE_RATE
    offset = bands.channel_frequency(channel) - bands.DEFAULT_CENTRE_HZ

    deviation = 750.0 * np.sin(2 * np.pi * 700.0 * t)
    phase = 2 * np.pi * np.cumsum(deviation) / SAMPLE_RATE
    gate = np.zeros(total)
    gate[int(0.2 * SAMPLE_RATE) : int(1.2 * SAMPLE_RATE)] = 1.0

    scene = (
        3e-4 * (rng.standard_normal(total) + 1j * rng.standard_normal(total))
        + 0.05 * gate * np.exp(1j * (2 * np.pi * offset * t + phase))
    ).astype(np.complex64)

    scene.view(np.float32).tofile(path)
    return path


def test_cli_replays_a_file_and_prints_one_json_object_per_emission(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from esm446.cli.node import main

    path = write_scene(tmp_path / "scene.cf32", channel=7)
    assert main(["--file", str(path)]) == 0

    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert len(lines) == 1

    report = json.loads(lines[0])
    assert report["pmr_channel"] == 7
    assert report["duration_s"] == pytest.approx(1.0, abs=0.15)
    assert report["estimated_dbm"] is None, "no calibration is loaded, so dBm must be null"
    assert report["gains"]["lna_db"] == 32.0


def test_cli_reports_a_missing_file_without_a_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from esm446.cli.node import main

    assert main(["--file", str(tmp_path / "absent.cf32")]) == 1
    assert capsys.readouterr().out == ""


def test_cli_rejects_an_unknown_sample_format(tmp_path: Path) -> None:
    from esm446.cli.node import main

    path = tmp_path / "scene.cf32"
    path.write_bytes(b"\x00" * 64)
    with pytest.raises(SystemExit):
        main(["--file", str(path), "--format", "cs8"])


def test_cli_requires_a_mode(tmp_path: Path) -> None:
    """--file and --sdr are mutually exclusive and one is required."""
    from esm446.cli.node import main

    with pytest.raises(SystemExit):
        main([])


def test_cli_surfaces_a_missing_sdr_stack_as_an_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """SoapySDR is not installed here, and asking for --sdr must fail cleanly."""
    from esm446.cli.node import main

    pytest.importorskip
    try:
        import SoapySDR  # noqa: F401
    except ImportError:
        assert main(["--sdr"]) == 1
        assert capsys.readouterr().out == ""
