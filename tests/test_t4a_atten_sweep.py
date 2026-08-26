"""Verification of the T4a streaming accumulators.

`scripts/t4a_atten_sweep.py` is an operator tool, not a package module -- same convention as
`scripts/overnight_survey.py`, loaded here by file path rather than `import
scripts.t4a_atten_sweep`.

These tests exist because the single-shot predecessor of this script (decode the whole
capture, run one `channelizer.process()` call, keep the whole spectra array) drove a 7.6 GB
RAM machine into swap and hung on shutdown for a 60 s capture. `ChannelPowerSelector` and
`BinStats` are the fix -- streaming accumulators that never hold more than one chunk's worth
of data -- and a design that changes how memory is used gets tests before it gets trusted with
another long capture, same reasoning `test_overnight_survey.py` gives for the BIT accumulator.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest


def _load_module() -> ModuleType:
    path = Path(__file__).resolve().parent.parent / "scripts" / "t4a_atten_sweep.py"
    spec = importlib.util.spec_from_file_location("t4a_atten_sweep", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["t4a_atten_sweep"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod() -> ModuleType:
    return _load_module()


def _spectra(num_frames: int, num_channels: int, bin_powers: dict[int, float]) -> np.ndarray:
    """A synthetic (frames, channels) spectra array: bin `k` holds constant magnitude
    sqrt(bin_powers[k]) in every frame, zero elsewhere."""
    out = np.zeros((num_frames, num_channels), dtype=np.complex64)
    for bin_index, power in bin_powers.items():
        out[:, bin_index] = np.sqrt(power)
    return out


# --------------------------------------------------------------------------------------
# excluded_bin_mask
# --------------------------------------------------------------------------------------


def test_dc_bin_is_excluded(mod: ModuleType) -> None:
    mask = mod.excluded_bin_mask(num_channels=160, dc_guard_bins=1, edge_guard_bins=0)
    assert mask[0]
    assert not mask.any() or mask.sum() == 1


def test_edge_guard_excludes_both_sides_of_nyquist(mod: ModuleType) -> None:
    mask = mod.excluded_bin_mask(num_channels=160, dc_guard_bins=0, edge_guard_bins=8)
    nyquist = 80
    assert mask[nyquist - 8 : nyquist + 8].all()
    assert not mask[0]
    assert not mask[nyquist - 9]
    assert not mask[nyquist + 8]


def test_a_bin_well_away_from_dc_and_the_edge_is_not_excluded(mod: ModuleType) -> None:
    mask = mod.excluded_bin_mask(num_channels=160, dc_guard_bins=1, edge_guard_bins=8)
    assert not mask[120]


# --------------------------------------------------------------------------------------
# ChannelPowerSelector
# --------------------------------------------------------------------------------------


def test_selector_picks_the_strongest_unexcluded_bin(mod: ModuleType) -> None:
    selector = mod.ChannelPowerSelector(num_channels=160, settle_frames=0)
    selector.update(_spectra(100, 160, {120: 1.0, 130: 0.5}))
    excluded = mod.excluded_bin_mask(num_channels=160)
    assert selector.select_bin(excluded) == 120


def test_selector_ignores_a_stronger_but_excluded_dc_spur(mod: ModuleType) -> None:
    """This is exactly the bug the first version of this script had: a smoke-test capture
    with nothing transmitting picked bin 0 (LO leakage) as "the signal"."""
    selector = mod.ChannelPowerSelector(num_channels=160, settle_frames=0)
    selector.update(_spectra(100, 160, {0: 100.0, 120: 1.0}))
    excluded = mod.excluded_bin_mask(num_channels=160)
    assert selector.select_bin(excluded) == 120


def test_selector_accumulates_across_multiple_chunks(mod: ModuleType) -> None:
    """Streaming property: splitting the same total power across several update() calls
    must give the same answer as one call with everything at once."""
    selector_one_call = mod.ChannelPowerSelector(num_channels=160, settle_frames=0)
    selector_one_call.update(_spectra(300, 160, {120: 1.0, 130: 0.9}))

    selector_chunked = mod.ChannelPowerSelector(num_channels=160, settle_frames=0)
    for _ in range(3):
        selector_chunked.update(_spectra(100, 160, {120: 1.0, 130: 0.9}))

    excluded = mod.excluded_bin_mask(num_channels=160)
    assert selector_one_call.select_bin(excluded) == selector_chunked.select_bin(excluded) == 120


def test_settle_frames_are_dropped_only_once_across_chunk_boundaries(mod: ModuleType) -> None:
    """The settle-drop budget must be spent once at the very start of the capture, even when
    it spans more than one chunk -- not re-applied at the start of every chunk, which would
    silently discard real data throughout a long streaming capture."""
    selector = mod.ChannelPowerSelector(num_channels=160, settle_frames=15)
    # First chunk shorter than the settle budget: all 10 frames must be dropped, none counted.
    selector.update(_spectra(10, 160, {120: 100.0}))
    assert selector.n_frames_seen == 0
    # Second chunk: only the remaining 5 frames of settle budget are dropped from this one.
    selector.update(_spectra(20, 160, {120: 100.0}))
    assert selector.n_frames_seen == 15
    # Third chunk: settle budget already exhausted, every frame counts.
    selector.update(_spectra(10, 160, {120: 100.0}))
    assert selector.n_frames_seen == 25


# --------------------------------------------------------------------------------------
# BinStats
# --------------------------------------------------------------------------------------


def test_bin_stats_reports_mean_of_linear_power_not_mean_of_db(mod: ModuleType) -> None:
    """Averaging in dB and then reporting that as "the mean power" is a real, easy mistake --
    Jensen's inequality means it is not the same number as the dB of the mean power, and the
    two can disagree by a couple of dB for a signal with real amplitude variation."""
    stats = mod.BinStats(bin_index=0, settle_frames=0)
    spectra = np.zeros((2, 1), dtype=np.complex64)
    spectra[0, 0] = 1.0  # power 1.0 -> 0 dBFS
    spectra[1, 0] = 0.1  # power 0.01 -> -20 dBFS
    stats.update(spectra)
    result = stats.result()
    # Mean linear power = (1.0 + 0.01) / 2 = 0.505 -> -2.966 dBFS.
    # Mean of the two dB values would instead give -10.0 dBFS -- a very different number.
    assert result["mean_dbfs"] == pytest.approx(-2.97, abs=0.01)


def test_bin_stats_peak_min_and_spread(mod: ModuleType) -> None:
    stats = mod.BinStats(bin_index=5, settle_frames=0)
    spectra = np.zeros((3, 10), dtype=np.complex64)
    spectra[0, 5] = 1.0  # 0 dBFS
    spectra[1, 5] = 0.1  # -20 dBFS
    spectra[2, 5] = 0.5  # -6.02 dBFS
    stats.update(spectra)
    result = stats.result()
    assert result["peak_dbfs"] == pytest.approx(0.0, abs=0.01)
    assert result["min_dbfs"] == pytest.approx(-20.0, abs=0.01)
    assert result["spread_db"] == pytest.approx(20.0, abs=0.01)


def test_bin_stats_accumulates_across_multiple_chunks(mod: ModuleType) -> None:
    stats_one_call = mod.BinStats(bin_index=0, settle_frames=0)
    stats_one_call.update(_spectra(4, 1, {0: 1.0}))
    result_one_call = stats_one_call.result()

    stats_chunked = mod.BinStats(bin_index=0, settle_frames=0)
    stats_chunked.update(_spectra(2, 1, {0: 1.0}))
    stats_chunked.update(_spectra(2, 1, {0: 1.0}))
    result_chunked = stats_chunked.result()

    assert result_one_call == result_chunked


def test_bin_stats_with_no_frames_reports_none_rather_than_crashing(mod: ModuleType) -> None:
    stats = mod.BinStats(bin_index=0, settle_frames=50)
    stats.update(_spectra(10, 160, {0: 1.0}))  # all 10 dropped by the settle budget
    result = stats.result()
    assert result["n_frames"] == 0
    assert result["mean_dbfs"] is None
