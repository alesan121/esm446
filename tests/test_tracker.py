"""Verification of emission tracking.

The tracker is where v0's architecture failed rather than its arithmetic, so these tests are
about state: an emission that spans a block boundary must be one emission, a modulation null
must not split a transmission, and a real gap must not merge two.
"""

from __future__ import annotations

import numpy as np
import pytest

from esm446.core.tracker import EmissionTracker

NUM_BINS = 160
CHANNEL_RATE = 25_000.0


def make_block(
    frames: int, active: dict[int, tuple[int, int]] | None = None, power_level: float = 100.0
):
    """Build a block where the given bins are detected over the given frame ranges.

    Args:
        frames: Number of frames in the block.
        active: Mapping of bin index to an inclusive ``(start, end)`` frame range.
        power_level: Linear power assigned to detected frames.

    Returns:
        ``(spectra, power, mask, noise)`` shaped ``(frames, NUM_BINS)``.
    """
    power = np.ones((frames, NUM_BINS)) * 0.01
    mask = np.zeros((frames, NUM_BINS), dtype=bool)
    noise = np.ones((frames, NUM_BINS)) * 0.01

    for bin_index, (start, end) in (active or {}).items():
        power[start : end + 1, bin_index] = power_level
        mask[start : end + 1, bin_index] = True

    spectra = np.sqrt(power).astype(np.complex64)
    return spectra, power, mask, noise


def test_a_simple_burst_is_one_emission() -> None:
    tracker = EmissionTracker(hangover_frames=5, min_frames=1)
    emissions = tracker.update(*make_block(1000, {40: (100, 599)}))

    assert len(emissions) == 1
    assert emissions[0].bin_index == 40
    assert emissions[0].frame_count == 500


def test_emission_stays_open_until_the_gap_exceeds_the_hangover() -> None:
    """A burst still in progress at the end of a block must not be reported yet."""
    tracker = EmissionTracker(hangover_frames=50, min_frames=1)
    emissions = tracker.update(*make_block(1000, {40: (100, 999)}))

    assert emissions == []
    assert tracker.open_count == 1


def test_emission_spanning_two_blocks_is_a_single_emission() -> None:
    """The property v0's file coupling could not have: state survives the block boundary."""
    tracker = EmissionTracker(hangover_frames=10, min_frames=1)

    assert tracker.update(*make_block(1000, {40: (500, 999)})) == []
    emissions = tracker.update(*make_block(1000, {40: (0, 499)}))

    assert len(emissions) == 1
    assert emissions[0].start_frame == 500
    assert emissions[0].end_frame == 1499
    assert emissions[0].frame_count == 1000


def test_short_gap_does_not_split_a_transmission() -> None:
    """Speech pauses. Closing on the first quiet frame would shred one over into fragments."""
    tracker = EmissionTracker(hangover_frames=100, min_frames=1)
    spectra, power, mask, noise = make_block(1000, {40: (0, 399)})
    mask[450:499, 40] = True
    power[450:499, 40] = 100.0
    emissions = tracker.update(spectra, power, mask, noise) + tracker.flush()
    assert len(emissions) == 1, "a 50-frame gap under a 100-frame hangover must not split"


def test_long_gap_separates_two_transmissions() -> None:
    tracker = EmissionTracker(hangover_frames=20, min_frames=1)
    spectra, power, mask, noise = make_block(1000, {40: (0, 199)})
    mask[600:799, 40] = True
    power[600:799, 40] = 100.0

    emissions = tracker.update(spectra, power, mask, noise) + tracker.flush()
    assert len(emissions) == 2
    assert emissions[0].end_frame == 199
    assert emissions[1].start_frame == 600


def test_simultaneous_emitters_are_tracked_independently() -> None:
    tracker = EmissionTracker(hangover_frames=10, min_frames=1)
    emissions = tracker.update(*make_block(1000, {40: (0, 499), 90: (200, 699)}))

    assert {e.bin_index for e in emissions} == {40, 90}
    assert next(e for e in emissions if e.bin_index == 40).frame_count == 500
    assert next(e for e in emissions if e.bin_index == 90).frame_count == 500


def test_flush_reports_an_emission_cut_off_by_end_of_stream() -> None:
    """Without this, a transmission in progress when a recording ends vanishes entirely."""
    tracker = EmissionTracker(hangover_frames=1000, min_frames=1)
    tracker.update(*make_block(500, {40: (0, 499)}))

    assert tracker.open_count == 1
    emissions = tracker.flush()
    assert len(emissions) == 1
    assert tracker.open_count == 0


def test_transients_are_filtered_out() -> None:
    tracker = EmissionTracker(hangover_frames=5, min_frames=250)
    emissions = tracker.update(*make_block(1000, {40: (100, 149)}))

    assert len(emissions) == 1
    assert tracker.filter_short(emissions) == []


def test_a_stuck_carrier_cannot_grow_without_bound() -> None:
    """The cap exists so an interferer cannot exhaust memory over a long capture."""
    tracker = EmissionTracker(hangover_frames=10, min_frames=1, max_frames=600)
    emissions = tracker.update(*make_block(1000, {40: (0, 999)}))

    assert len(emissions) == 1
    assert emissions[0].frame_count <= 1000


def test_emission_carries_the_channel_samples() -> None:
    """Identification needs the baseband samples, so the tracker must retain them."""
    tracker = EmissionTracker(hangover_frames=5, min_frames=1)
    emissions = tracker.update(*make_block(1000, {40: (100, 599)}))

    assert emissions[0].iq.size == 500
    assert emissions[0].iq.dtype == np.complex64


def test_emission_statistics_are_measured_not_assumed() -> None:
    tracker = EmissionTracker(hangover_frames=5, min_frames=1)
    emissions = tracker.update(*make_block(1000, {40: (100, 599)}, power_level=100.0))
    emission = emissions[0]

    assert emission.peak_power == pytest.approx(100.0)
    assert emission.mean_power == pytest.approx(100.0)
    assert emission.snr_db == pytest.approx(40.0, abs=0.1)
    assert emission.peak_power_dbfs == pytest.approx(20.0, abs=0.1)
    assert emission.duration_seconds(CHANNEL_RATE) == pytest.approx(0.02)


def test_empty_block_is_a_no_op() -> None:
    tracker = EmissionTracker()
    empty = np.zeros((0, NUM_BINS))
    assert tracker.update(empty.astype(np.complex64), empty, empty.astype(bool), empty) == []


def test_rejects_a_negative_hangover() -> None:
    with pytest.raises(ValueError, match="hangover_frames"):
        EmissionTracker(hangover_frames=-1)
