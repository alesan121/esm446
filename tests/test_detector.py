"""Verification of the CFAR detector.

The claim CFAR makes is specific and falsifiable: whatever the noise level, the false alarm
rate equals the design point. So the central test here does not check that the detector
finds a signal — that is easy and proves little. It feeds pure noise, counts the false
alarms, and compares the measured rate against the requested P_fa. A detector that fires
ten times too often still "works" in every casual sense.
"""

from __future__ import annotations

import numpy as np
import pytest

from esm446.core.detector import (
    CfarConfig,
    CfarDetector,
    ca_threshold_factor,
    os_threshold_factor,
)

NUM_BINS = 160


def exponential_noise(rng: np.random.Generator, shape: tuple[int, ...], mean: float = 1.0):
    """Power of complex Gaussian noise is exponentially distributed — the CFAR assumption."""
    return rng.exponential(scale=mean, size=shape)


# --------------------------------------------------------------------------------------
# Threshold factors
# --------------------------------------------------------------------------------------


def test_ca_threshold_factor_matches_closed_form() -> None:
    assert ca_threshold_factor(24, 1e-4) == pytest.approx(24 * (1e-4 ** (-1 / 24) - 1))


def test_os_threshold_factor_inverts_its_own_pfa_expression() -> None:
    """The numerical root find must reproduce the P_fa it was asked for."""
    n, k, pfa = 24, 18, 1e-4
    alpha = os_threshold_factor(n, k, pfa)
    indices = np.arange(k)
    recovered = np.prod((n - indices) / (n - indices + alpha))
    assert recovered == pytest.approx(pfa, rel=1e-6)


def test_tighter_pfa_needs_a_higher_threshold() -> None:
    assert ca_threshold_factor(24, 1e-6) > ca_threshold_factor(24, 1e-3)


# --------------------------------------------------------------------------------------
# The central claim: measured false alarm rate matches the design point
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["ca", "os"])
@pytest.mark.parametrize("noise_level", [1e-6, 1.0, 1e3])
def test_false_alarm_rate_holds_across_noise_levels(method: str, noise_level: float) -> None:
    """This is the property v0's fixed threshold could not have.

    The same detector, unchanged, is given noise nine orders of magnitude apart in power.
    The false alarm rate must not move.
    """
    rng = np.random.default_rng(1234)
    detector = CfarDetector(CfarConfig(num_reference=24, num_guard=2, pfa=1e-3, method=method))
    power = exponential_noise(rng, (4000, NUM_BINS), mean=noise_level)

    measured = detector.detection_mask(power).mean()
    assert measured == pytest.approx(1e-3, rel=0.45), (
        f"{method}-CFAR measured P_fa {measured:.2e} against a 1e-3 design point"
    )


def test_fixed_threshold_fails_where_cfar_holds() -> None:
    """Demonstrates the v0 failure mode rather than merely asserting it.

    A threshold tuned on one noise level goes blind at a lower one and saturates at a
    higher one. This test exists so the V&V report can cite a measurement, not an opinion.
    """
    rng = np.random.default_rng(7)
    tuned_threshold = 6.9  # ~1e-3 false alarm rate for unit-mean exponential noise
    rates = [
        (exponential_noise(rng, (2000, NUM_BINS), mean=level) > tuned_threshold).mean()
        for level in (0.01, 1.0, 100.0)
    ]
    assert rates[0] == 0.0, "fixed threshold should go completely deaf in quiet noise"
    assert rates[2] > 0.9, "fixed threshold should saturate in loud noise"


# --------------------------------------------------------------------------------------
# Detection behaviour
# --------------------------------------------------------------------------------------


def test_strong_emitter_is_detected_with_correct_snr() -> None:
    rng = np.random.default_rng(0)
    detector = CfarDetector(CfarConfig(pfa=1e-4))
    power = exponential_noise(rng, (NUM_BINS,), mean=1.0)
    power[40] = 1000.0

    detections = detector.detect(power)
    assert detections[0].bin_index == 40
    assert detections[0].snr_db > 25.0


def test_os_cfar_survives_an_adjacent_interferer_that_blinds_ca_cfar() -> None:
    """The reason OS-CFAR is the default in a band of adjacent 12.5 kHz channels.

    A strong emitter inside the reference window inflates the cell-averaged noise estimate
    and masks a weaker neighbour. An order statistic below the interferer's rank ignores it.
    """
    rng = np.random.default_rng(3)
    power = exponential_noise(rng, (NUM_BINS,), mean=1.0)
    power[80] = 5.0e4  # strong emitter
    power[84] = 60.0  # weak emitter, inside the strong one's reference window

    ca_hits = {d.bin_index for d in CfarDetector(CfarConfig(method="ca")).detect(power)}
    os_hits = {d.bin_index for d in CfarDetector(CfarConfig(method="os")).detect(power)}

    assert 84 not in ca_hits, "CA-CFAR was expected to be masked by the interferer"
    assert 84 in os_hits, "OS-CFAR should still see the weak emitter"


def test_detections_are_ordered_by_snr() -> None:
    rng = np.random.default_rng(5)
    power = exponential_noise(rng, (NUM_BINS,), mean=1.0)
    power[10], power[50], power[120] = 200.0, 5000.0, 800.0
    detections = CfarDetector().detect(power)
    assert [d.bin_index for d in detections[:3]] == [50, 120, 10]


def test_spectrum_wraps_so_edge_bins_are_not_biased() -> None:
    """Bin 0 and bin M-1 are physically adjacent; their reference windows must wrap."""
    rng = np.random.default_rng(11)
    detector = CfarDetector(CfarConfig(pfa=1e-3))
    power = exponential_noise(rng, (3000, NUM_BINS), mean=1.0)
    rates = detector.detection_mask(power).mean(axis=0)
    edges = np.concatenate([rates[:4], rates[-4:]])
    interior = rates[20:-20]
    assert edges.mean() == pytest.approx(interior.mean(), rel=0.6)


def test_rejects_a_spectrum_narrower_than_the_window() -> None:
    with pytest.raises(ValueError, match="at least"):
        CfarDetector(CfarConfig(num_reference=24, num_guard=2)).detect(np.ones(10))
