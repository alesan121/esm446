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
    assert measured == pytest.approx(
        1e-3, rel=0.45
    ), f"{method}-CFAR measured P_fa {measured:.2e} against a 1e-3 design point"


@pytest.mark.parametrize("interval", [1, 64, 256])
def test_holding_the_noise_estimate_does_not_change_the_false_alarm_rate(interval: int) -> None:
    """Pins the claim that makes OS-CFAR affordable.

    Re-estimating the noise floor every frame costs 4.93 CPU-seconds per signal second and
    allocates 101 MB per block. Holding the estimate for 64 frames -- 2.6 ms, far shorter
    than any real change in a thermal noise floor -- costs 64x less. That is only legitimate
    if the false alarm rate is unaffected, so it is measured here rather than argued.
    """
    rng = np.random.default_rng(99)
    detector = CfarDetector(CfarConfig(pfa=1e-3, method="os", update_interval=interval))
    power = exponential_noise(rng, (6000, NUM_BINS), mean=1.0)

    measured = detector.detection_mask(power).mean()
    assert measured == pytest.approx(
        1e-3, rel=0.45
    ), f"update_interval={interval} gave P_fa {measured:.2e} against a 1e-3 design point"


def test_noise_estimate_is_held_between_updates() -> None:
    """The estimate must actually be reused, not silently recomputed."""
    rng = np.random.default_rng(4)
    detector = CfarDetector(CfarConfig(update_interval=64))
    power = exponential_noise(rng, (256, NUM_BINS), mean=1.0)

    noise = detector.noise_estimate(power)
    assert noise.shape == power.shape
    np.testing.assert_array_equal(noise[0], noise[63])
    assert not np.array_equal(noise[0], noise[64])


def test_rejects_a_zero_update_interval() -> None:
    with pytest.raises(ValueError, match="update_interval"):
        CfarConfig(update_interval=0)


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


# --------------------------------------------------------------------------------------
# The pair test: an emitter that lands between two bins
# --------------------------------------------------------------------------------------


def test_the_pair_threshold_is_lower_than_the_single_cell_one() -> None:
    """The whole basis of the pair test, in one number.

    Summing two cells does not improve the ratio of signal to noise -- both sum. It reduces
    the spread of the statistic, so the threshold that achieves a given false alarm rate sits
    closer to the mean. That is the gain, and it is about 2 dB.
    """
    detector = CfarDetector(CfarConfig(pfa=1e-8, method="os", detect_pairs=True))
    advantage_db = 10 * np.log10(detector.threshold_factor / detector.pair_threshold_factor)

    assert advantage_db == pytest.approx(2.07, abs=0.15)


@pytest.mark.parametrize("method", ["ca", "os"])
def test_the_pair_false_alarm_rate_matches_its_closed_form(method: str) -> None:
    """The derivation is only worth having if the measured rate agrees with it.

    Measured slightly below design, by about 13 %, because adjacent pairs share reference
    cells: the two noise estimates being averaged are correlated, which trims the variance
    the derivation assumes. It errs towards fewer false alarms, and 13 % of a rate is 0.06 dB
    of threshold, so it is accepted rather than corrected.
    """
    rng = np.random.default_rng(7)
    detector = CfarDetector(
        CfarConfig(pfa=1e-3, method=method, update_interval=1, detect_pairs=True)
    )
    power = exponential_noise(rng, (20_000, NUM_BINS), mean=1.0)

    noise = detector.noise_estimate(power)
    paired = power + np.roll(power, -1, axis=-1)
    paired_noise = noise + np.roll(noise, -1, axis=-1)
    measured = (paired > paired_noise * detector.pair_threshold_factor).mean()

    # Half the budget, since the design point is split across the two tests.
    assert 0.4e-3 < measured < 0.75e-3, f"{method} pair test measured P_fa {measured:.2e}"


def test_the_budget_is_split_so_the_union_still_meets_the_design_point() -> None:
    """Two tests per cell means two chances to cry wolf, and the design P_fa bounds both."""
    with_pairs = CfarDetector(CfarConfig(pfa=1e-8, detect_pairs=True))
    without = CfarDetector(CfarConfig(pfa=1e-8, detect_pairs=False))
    cost_db = 10 * np.log10(with_pairs.threshold_factor / without.threshold_factor)

    assert cost_db == pytest.approx(0.3, abs=0.15)


def test_the_pair_test_flags_both_bins_of_a_crossing() -> None:
    """Reporting them as one emitter is the adjacent-bin merge's job, not the detector's."""
    detector = CfarDetector(CfarConfig(pfa=1e-3, detect_pairs=True))
    power = np.ones(NUM_BINS)
    power[40] = power[41] = 20.0
    noise = np.ones(NUM_BINS)

    mask = detector.pair_mask(power, noise)

    assert mask[40] and mask[41]


def test_an_excluded_bin_does_not_drag_its_neighbour_over() -> None:
    """The failure this guard exists for, and the reason the exclusion is not applied after.

    The local oscillator spur sits 31 dB above the floor in the DC bin. Blanking that bin at
    the end would leave the pair test having already flagged bin 1, reporting the receiver's
    own artefact as an emission on the next channel along.
    """
    detector = CfarDetector(CfarConfig(pfa=1e-3, detect_pairs=True))
    power = np.ones(NUM_BINS)
    power[0] = 1000.0
    noise = np.ones(NUM_BINS)

    excluded = np.zeros(NUM_BINS, dtype=bool)
    excluded[0] = True

    assert detector.pair_mask(power, noise)[1], "without the guard the spur reaches bin 1"
    assert not detector.pair_mask(power, noise, exclude=excluded)[1]


def test_pair_detection_is_off_by_default() -> None:
    """Measured on the recorded captures it made the output worse; see the module docstring.

    It cost 0.38 dB everywhere, which pushed a genuine sideband at 6.6 dB SNR below the
    threshold and broke the splatter attribution, in exchange for 1.25 dB at exactly half a
    bin. The capability ships; the default reflects what the measurement said.
    """
    detector = CfarDetector(CfarConfig())

    assert detector.config.detect_pairs is False
    assert detector.pair_threshold_factor == float("inf")
    assert not detector.pair_mask(np.ones(NUM_BINS), np.ones(NUM_BINS)).any()


def test_the_pair_test_recovers_sensitivity_at_the_worst_offset() -> None:
    """The gain it does deliver, measured through the channeliser rather than argued.

    A tone exactly halfway between two bins, at an SNR where the single-bin test is
    unreliable and the pair test is not.
    """
    from esm446.core.channelizer import ChannelizerConfig, PolyphaseChannelizer

    config = ChannelizerConfig(sample_rate=2_000_000.0, num_channels=160, decimation=80)
    samples = 1 << 17
    t = np.arange(samples) / config.sample_rate
    rng = np.random.default_rng(3)

    # 19.5 dB in-channel, which is between the two tests' half-detection points.
    amplitude = 3e-3 * 10 ** ((19.5 - 10 * np.log10(config.num_channels)) / 20)
    tone = amplitude * np.exp(2j * np.pi * 20.5 * config.channel_spacing * t)
    noise = 3e-3 * (rng.standard_normal(samples) + 1j * rng.standard_normal(samples)) / np.sqrt(2)
    spectra = PolyphaseChannelizer(config).process((tone + noise).astype(np.complex64))
    power = (np.abs(spectra[1000:]) ** 2).astype(np.float64)

    rates = {}
    for pairs in (False, True):
        detector = CfarDetector(CfarConfig(pfa=1e-8, method="os", detect_pairs=pairs))
        mask = detector.detection_mask(power)
        rates[pairs] = mask[:, 19:23].any(axis=1).mean()

    assert rates[True] > rates[False] + 0.1, f"single {rates[False]:.2f}, pair {rates[True]:.2f}"
