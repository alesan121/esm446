"""Verification of the inline BIT accumulator and per-chunk sidecar writer.

`scripts/overnight_survey.py` is an operator tool, not a package module -- it lives outside
`esm446/` deliberately, and imports SoapySDR-free hardware calls at module scope that make no
sense to expose as a library import. It is loaded here by file path rather than `import
scripts.overnight_survey`, which is how the script itself is actually run.

These tests exist because a session ran for eight hours on code that had never been checked:
the backoff/circuit-breaker gap that let ~4000 failed captures log inside a minute, found only
because someone was watching. The BIT accumulator and sidecar writer are the direct response to
the next thing that gap allowed -- eight hours of raw IQ deleted with no way to check afterwards
whether the antenna was ever disconnected. They get tests before they get trusted with another
unattended run.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest


def _load_module() -> ModuleType:
    path = Path(__file__).resolve().parent.parent / "scripts" / "overnight_survey.py"
    spec = importlib.util.spec_from_file_location("overnight_survey", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["overnight_survey"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod() -> ModuleType:
    return _load_module()


def _tone_block(amplitude: float, n: int = 4096) -> np.ndarray:
    t = np.arange(n)
    return (amplitude * np.exp(2j * np.pi * 0.01 * t)).astype(np.complex64)


def _quantised_block(codes: list[int], repeats: int = 1000) -> np.ndarray:
    """A block whose I and Q only ever take the given int8 codes, decoded like FileSource does."""
    values = np.array(codes * repeats, dtype=np.float32) / 128.0
    return (values + 1j * values).astype(np.complex64)


# --------------------------------------------------------------------------------------
# BitAccumulator
# --------------------------------------------------------------------------------------


def test_an_empty_accumulator_reports_zero_samples(mod) -> None:
    result = mod.BitAccumulator().result()

    assert result["n_samples"] == 0
    assert result["occupied_codes"] == 0
    assert result["peak_dbfs"] is None


def test_occupied_codes_counts_distinct_int8_values_across_i_and_q(mod) -> None:
    """The quantisation this project already retracted a figure over: 5 codes, ~2 bits."""
    acc = mod.BitAccumulator()
    acc.update(_quantised_block([-2, -1, 0, 1, 2]))

    result = acc.result()
    assert result["occupied_codes"] == 5
    assert result["approx_bits"] == pytest.approx(2.32, abs=0.01)


def test_full_scale_gives_the_full_code_range(mod) -> None:
    codes = list(range(-128, 128))
    acc = mod.BitAccumulator()
    acc.update(_quantised_block(codes, repeats=10))

    result = acc.result()
    assert result["occupied_codes"] == 256
    assert result["approx_bits"] == pytest.approx(8.0, abs=0.01)


def test_accumulates_across_multiple_update_calls(mod) -> None:
    """The actual usage pattern: one call per block read from FileSource, not one big array."""
    acc = mod.BitAccumulator()
    acc.update(_quantised_block([-2, -1]))
    acc.update(_quantised_block([0]))
    acc.update(_quantised_block([1, 2]))

    result = acc.result()
    assert result["occupied_codes"] == 5


def test_peak_dbfs_reflects_the_largest_code_seen(mod) -> None:
    acc = mod.BitAccumulator()
    acc.update(_tone_block(amplitude=1.0))  # full-scale tone, |I|,|Q| approach 127

    result = acc.result()
    assert result["peak_dbfs"] > -0.5


def test_a_weak_signal_has_a_correspondingly_low_peak(mod) -> None:
    acc = mod.BitAccumulator()
    acc.update(_quantised_block([-2, -1, 0, 1, 2]))

    result = acc.result()
    # peak code 2 of 127 full scale
    assert result["peak_dbfs"] == pytest.approx(20 * np.log10(2 / 127), abs=0.1)


def test_mean_and_std_are_computed_per_component_not_combined(mod) -> None:
    """A DC offset only on I must not be hidden by averaging it with a clean Q."""
    acc = mod.BitAccumulator()
    n = 2048
    biased_i = (np.full(n, 10.0, dtype=np.float32) / 128.0).astype(np.float32)
    clean_q = np.zeros(n, dtype=np.float32)
    block = (biased_i + 1j * clean_q).astype(np.complex64)
    acc.update(block)

    result = acc.result()
    assert result["mean_i"] == pytest.approx(10.0, abs=0.01)
    assert result["mean_q"] == pytest.approx(0.0, abs=0.01)
    assert result["std_i"] == pytest.approx(0.0, abs=0.01)


def test_a_block_with_no_samples_does_not_crash_the_accumulator(mod) -> None:
    acc = mod.BitAccumulator()
    acc.update(np.array([], dtype=np.complex64))

    assert acc.result()["n_samples"] == 0


# --------------------------------------------------------------------------------------
# write_chunk_sidecar
# --------------------------------------------------------------------------------------


def test_the_sidecar_is_written_atomically_no_tmp_file_left_behind(mod, tmp_path: Path) -> None:
    outcome = {"completed": True, "overruns": 0, "returncode": 0, "bytes_written": 960_000_000}
    bit_metrics = mod.BitAccumulator().result()

    mod.write_chunk_sidecar(tmp_path, 0, 1000.0, 1240.0, outcome, bit_metrics)

    sidecar_dir = tmp_path / "chunk_sidecars"
    files = list(sidecar_dir.glob("*"))
    assert [f.name for f in files] == ["chunk_00000.json"], "a .tmp file was left behind"


def test_the_sidecar_declares_the_gain_source_as_requested_not_confirmed(
    mod, tmp_path: Path
) -> None:
    """hackrf_transfer has no gain-readback path, unlike SoapySource -- said explicitly."""
    outcome = {"completed": True, "overruns": 0, "returncode": 0, "bytes_written": 1}
    bit_metrics = mod.BitAccumulator().result()

    mod.write_chunk_sidecar(tmp_path, 0, 1000.0, 1001.0, outcome, bit_metrics)

    record = json.loads((tmp_path / "chunk_sidecars" / "chunk_00000.json").read_text())
    assert record["gain_source"] == "requested"


def test_the_sidecar_carries_the_bit_metrics_and_chunk_timing(mod, tmp_path: Path) -> None:
    outcome = {"completed": True, "overruns": 3, "returncode": 0, "bytes_written": 1}
    acc = mod.BitAccumulator()
    acc.update(_quantised_block([-2, -1, 0, 1, 2]))

    mod.write_chunk_sidecar(tmp_path, 7, 1000.0, 1240.5, outcome, acc.result())

    record = json.loads((tmp_path / "chunk_sidecars" / "chunk_00007.json").read_text())
    assert record["chunk_index"] == 7
    assert record["chunk_duration_s"] == pytest.approx(240.5, abs=0.01)
    assert record["capture_overruns"] == 3
    assert record["bit"]["occupied_codes"] == 5
    assert record["lna_db"] == mod.C3_LNA_DB
    assert record["vga_db"] == mod.C3_VGA_DB


def test_successive_chunks_get_distinct_zero_padded_filenames(mod, tmp_path: Path) -> None:
    outcome = {"completed": True, "overruns": 0, "returncode": 0, "bytes_written": 1}
    bit_metrics = mod.BitAccumulator().result()

    for index in (0, 1, 99, 100):
        mod.write_chunk_sidecar(tmp_path, index, 0.0, 1.0, outcome, bit_metrics)

    names = sorted(p.name for p in (tmp_path / "chunk_sidecars").glob("*.json"))
    assert names == ["chunk_00000.json", "chunk_00001.json", "chunk_00099.json", "chunk_00100.json"]


# --------------------------------------------------------------------------------------
# run_c2's circuit breaker -- the gap found by re-reading the code after C3's own
# breaker was added: run_c2 had none, and only luck (the disconnect landing on the
# second-to-last combination) kept it from repeating C3's ~4000-failed-attempts incident.
# --------------------------------------------------------------------------------------


def _failing_capture(*args, **kwargs) -> dict:
    return {"completed": False, "overruns": 0, "returncode": 1, "bytes_written": 0}


def test_run_c2_backs_off_and_gives_up_after_repeated_capture_failures(
    mod, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The scenario this test exists for: hardware disconnected partway through the sweep.

    Before this fix, every one of the 36 combinations would have called `capture()` and
    failed in rapid succession with no delay -- exactly the pattern that produced ~4000
    failed attempts in under a minute in run_c3, before its own breaker was added.
    """
    monkeypatch.setattr(mod, "capture", _failing_capture)
    sleeps: list[float] = []
    monkeypatch.setattr(mod.time, "sleep", lambda s: sleeps.append(s))

    result = mod.run_c2(tmp_path)

    assert result["aborted"] == "device unavailable"
    assert result["points"] == []
    # Gives up at the 20th consecutive failure, checked before sleeping again -- so 19
    # backoff delays happen (after failures 1-19), not 20. Same structure as run_c3's own
    # breaker: no point sleeping right before giving up.
    assert len(sleeps) == 19
    assert all(s > 0 for s in sleeps), "no backoff delay between failed attempts"


def test_run_c2_resets_the_failure_count_after_a_success(
    mod, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A single bad point must not count against the 20-failure budget forever after."""
    calls = {"n": 0}

    def flaky_capture(path, seconds, lna_db, vga_db) -> dict:
        calls["n"] += 1
        if calls["n"] <= 3:
            return {"completed": False, "overruns": 0, "returncode": 1, "bytes_written": 0}
        # Write a minimal valid cs8 file so the analysis step downstream does not choke.
        np.zeros(2000, dtype=np.int8).tofile(path)
        return {"completed": True, "overruns": 0, "returncode": 0, "bytes_written": 2000}

    monkeypatch.setattr(mod, "capture", flaky_capture)
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)

    result = mod.run_c2(tmp_path)

    assert result["aborted"] is None
    assert len(result["points"]) == len(mod.C2_LNA_VALUES) * len(mod.C2_VGA_VALUES) - 3
