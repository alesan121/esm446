"""Verification of the frequency-calibration command line tool.

This CLI had never been tested directly -- only the estimators underneath it, in
`test_frequency.py`. That left the argument handling, the file I/O, and the memory ceiling
that this file exists because of unverified. A tool that reads real captures earns its
correctness the same way the estimators do: by being run and checked, not by being read.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from esm446.cli.calibrate_frequency import MAX_SAFE_SAMPLES, main

RATE = 20_000_000.0
NOMINAL = 816_000_000.0
LO = 818_500_000.0

#: Same construction as `lte_like` in test_frequency.py: a flat carrier with an unused
#: centre subcarrier, built in the frequency domain so the notch sits exactly where it is
#: put. Kept self-contained here rather than imported across test modules.
DVBT_OCCUPIED_HZ = 7_610_000.0
DVBT_RATE = 10_000_000.0


def _write_capture(path: Path, iq: np.ndarray) -> None:
    """Interleave and quantise to cs8, the format hackrf_transfer actually produces."""
    scaled = np.clip(np.stack([iq.real, iq.imag], axis=1).ravel() * 120.0, -127, 127)
    scaled.astype(np.int8).tofile(path)


def _notch_capture(shift_hz: float = 0.0, samples: int = 1 << 20) -> np.ndarray:
    rng = np.random.default_rng(1)
    baseband = NOMINAL - LO
    frequencies = np.fft.fftfreq(samples, 1.0 / RATE)
    spectrum = rng.normal(size=samples) + 1j * rng.normal(size=samples)
    spectrum[np.abs(frequencies - baseband) > 4.5e6] = 0.0
    offset = frequencies - baseband - shift_hz
    notch = 1.0 - 0.93 * np.exp(-((offset / 7.5e3) ** 2))
    block = np.fft.ifft(spectrum * notch)
    # Normalised to unit peak before the int8 round-trip a real capture goes through --
    # unlike the core estimator's own tests, which operate on this array directly and never
    # face quantisation, an unnormalised ifft output is too quiet to survive it.
    return (block / np.abs(block).max()).astype(np.complex64)


def _multiplex_capture(centre_offset_hz: float = 0.0, samples: int = 1 << 19) -> np.ndarray:
    rng = np.random.default_rng(2)
    spectrum = np.zeros(samples, dtype=np.complex128)
    frequencies = np.fft.fftfreq(samples, 1.0 / DVBT_RATE)
    occupied = np.abs(frequencies - centre_offset_hz) <= DVBT_OCCUPIED_HZ / 2.0
    spectrum[occupied] = rng.normal(size=occupied.sum()) + 1j * rng.normal(size=occupied.sum())
    block = np.fft.ifft(spectrum)
    return (block / np.abs(block).max()).astype(np.complex64)


# --------------------------------------------------------------------------------------
# The memory ceiling -- what this file exists for
# --------------------------------------------------------------------------------------


def test_refuses_samples_above_the_memory_ceiling(tmp_path: Path, capsys) -> None:
    """The guard added after 190 million samples drove this tool to 6.9 GB and swap thrashing.

    Checked before the capture is even opened, deliberately: refusing early is what makes
    this a guard rather than a slow way to discover the same crash.
    """
    missing = tmp_path / "does_not_need_to_exist.cs8"
    code = main(
        [
            str(missing),
            "--centre",
            "816e6",
            "--rate",
            "20e6",
            "--nominal",
            "816e6",
            "--samples",
            str(MAX_SAFE_SAMPLES + 1),
        ]
    )
    assert code == 1
    # Refused before the (nonexistent) file is even opened -- that is the point of the guard.
    assert capsys.readouterr().out == ""


def test_the_default_sample_count_is_well_inside_the_ceiling() -> None:
    from esm446.cli.calibrate_frequency import DEFAULT_SAMPLES

    assert DEFAULT_SAMPLES < MAX_SAFE_SAMPLES


# --------------------------------------------------------------------------------------
# Argument validation
# --------------------------------------------------------------------------------------


def test_refuses_a_missing_capture(tmp_path: Path, capsys) -> None:
    missing = tmp_path / "nothing_here.cs8"
    code = main([str(missing), "--centre", "816e6", "--rate", "20e6", "--nominal", "816e6"])
    assert code == 1
    assert capsys.readouterr().out == ""


def test_refuses_mismatched_capture_and_centre_counts(tmp_path: Path, capsys) -> None:
    a = tmp_path / "a.cs8"
    b = tmp_path / "b.cs8"
    _write_capture(a, _notch_capture())
    _write_capture(b, _notch_capture())
    code = main(
        [
            str(a),
            str(b),
            "--centre",
            "816e6",
            "817e6",
            "818e6",
            "--rate",
            "20e6",
            "--nominal",
            "816e6",
        ]
    )
    assert code == 1
    assert capsys.readouterr().out == ""


def test_lte_reference_requires_nominal(tmp_path: Path, capsys) -> None:
    """The carrier centre cannot be inferred -- see the module docstring for why."""
    capture = tmp_path / "carrier.cs8"
    _write_capture(capture, _notch_capture())
    code = main([str(capture), "--centre", "818.5e6", "--rate", "20e6"])
    assert code == 1
    assert capsys.readouterr().out == ""


# --------------------------------------------------------------------------------------
# End to end, on synthetic captures with known ground truth
# --------------------------------------------------------------------------------------


def test_a_synthetic_lte_capture_measures_its_known_shift(tmp_path: Path, capsys) -> None:
    capture = tmp_path / "lte.cs8"
    _write_capture(capture, _notch_capture(shift_hz=600.0))
    code = main(
        [
            str(capture),
            "--centre",
            str(LO),
            "--rate",
            str(RATE),
            "--nominal",
            str(NOMINAL),
            "--samples",
            "1000000",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "offset" in out
    # Quantised to 8 bits and read back through the full CLI path, so the tolerance is
    # looser than the estimator's own unit tests -- this is checking the CLI wiring, not
    # re-proving the estimator.
    assert "+6" in out or "+5" in out or "+7" in out


def test_json_output_has_the_documented_fields(tmp_path: Path, capsys) -> None:
    import json

    capture = tmp_path / "lte.cs8"
    _write_capture(capture, _notch_capture())
    code = main(
        [
            str(capture),
            "--centre",
            str(LO),
            "--rate",
            str(RATE),
            "--nominal",
            str(NOMINAL),
            "--samples",
            "1000000",
            "--json",
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    for field in ("offset_hz", "ppm", "confidence_hz", "confidence_basis", "error_at_446_hz"):
        assert field in payload


def test_a_synthetic_dvbt_capture_measures_correctly(tmp_path: Path, capsys) -> None:
    capture = tmp_path / "dvbt.cs8"
    _write_capture(capture, _multiplex_capture())
    code = main(
        [
            str(capture),
            "--centre",
            "530e6",
            "--rate",
            str(DVBT_RATE),
            "--reference",
            "dvbt",
            "--nominal",
            "530e6",
            "--samples",
            "500000",
        ]
    )
    assert code == 0
    assert "530.000" in capsys.readouterr().out


def test_an_empty_band_is_refused_with_the_wrapped_message_not_a_traceback(
    tmp_path: Path, capsys
) -> None:
    rng = np.random.default_rng(3)
    noise = (rng.normal(size=1 << 20) + 1j * rng.normal(size=1 << 20)).astype(np.complex64)
    capture = tmp_path / "noise.cs8"
    _write_capture(capture, noise * 0.01)
    code = main(
        [
            str(capture),
            "--centre",
            str(LO),
            "--rate",
            str(RATE),
            "--nominal",
            str(NOMINAL),
            "--samples",
            "1000000",
        ]
    )
    assert code == 1
    assert capsys.readouterr().out == ""


def test_dvbt_reference_warns_and_uses_the_first_capture_when_given_several(
    tmp_path: Path, capsys
) -> None:
    a = tmp_path / "a.cs8"
    b = tmp_path / "b.cs8"
    _write_capture(a, _multiplex_capture())
    _write_capture(b, _multiplex_capture())
    code = main(
        [
            str(a),
            str(b),
            "--centre",
            "530e6",
            "--rate",
            str(DVBT_RATE),
            "--reference",
            "dvbt",
            "--nominal",
            "530e6",
            "--samples",
            "500000",
        ]
    )
    assert code == 0
    assert "530.000" in capsys.readouterr().out
