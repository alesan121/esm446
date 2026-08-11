"""Verification of the receiver chain model.

The sensitivity figures this module produces end up in the link budget and are compared
against measurement, so the arithmetic behind them is checked against hand-worked cases
rather than trusted.
"""

from __future__ import annotations

import numpy as np
import pytest

from esm446.core.rfchain import (
    HACKRF_LNA_STEP_DB,
    HACKRF_VGA_STEP_DB,
    RfChain,
    Stage,
    plan_gains,
    quantise_gains,
)

# --------------------------------------------------------------------------------------
# Friis cascade
# --------------------------------------------------------------------------------------


def test_single_stage_noise_figure_is_its_own() -> None:
    chain = RfChain([Stage("only", 20.0, 3.0)])
    assert chain.noise_figure_db == pytest.approx(3.0, abs=1e-9)


def test_friis_matches_a_hand_worked_cascade() -> None:
    """F = F1 + (F2-1)/G1, worked by hand in linear noise factor."""
    chain = RfChain([Stage("lna", 20.0, 1.0), Stage("rx", 0.0, 8.0)])
    f1, f2, g1 = 10**0.1, 10**0.8, 10**2.0
    expected = 10 * np.log10(f1 + (f2 - 1) / g1)
    assert chain.noise_figure_db == pytest.approx(expected, abs=1e-9)


def test_the_lna_is_what_buys_the_sensitivity() -> None:
    """The reason the external LNA goes first, quantified.

    A HackRF alone sits near 8 dB. The same receiver behind a 20 dB, 1 dB-NF LNA drops to
    1.18 dB, because Friis divides the second stage's contribution by the first stage's
    gain: factor 1.2589 + (6.3096-1)/100 = 1.3120, which is 1.18 dB — not 1.31 dB. The
    noise factor and the noise figure are easy to confuse and this test pins the units.
    """
    bare = RfChain([Stage("HackRF One", 0.0, 8.0)])
    with_lna = RfChain([Stage("external LNA", 20.0, 1.0), Stage("HackRF One", 0.0, 8.0)])

    assert bare.noise_figure_db == pytest.approx(8.0, abs=0.01)
    assert with_lna.noise_figure_db == pytest.approx(1.18, abs=0.02)
    assert bare.noise_figure_db - with_lna.noise_figure_db > 6.0


def test_loss_ahead_of_the_lna_costs_sensitivity_directly() -> None:
    """Why cable length before the LNA matters and after it barely does."""
    before = RfChain([Stage("cable", -3.0, 3.0), Stage("lna", 20.0, 1.0), Stage("rx", 0.0, 8.0)])
    after = RfChain([Stage("lna", 20.0, 1.0), Stage("cable", -3.0, 3.0), Stage("rx", 0.0, 8.0)])
    assert before.noise_figure_db > after.noise_figure_db + 2.5


def test_deployed_chain_is_dominated_by_the_first_stage() -> None:
    chain = RfChain.deployed()
    assert 1.0 < chain.noise_figure_db < 3.0
    assert chain.total_gain_db == pytest.approx(19.3, abs=0.01)


# --------------------------------------------------------------------------------------
# Sensitivity
# --------------------------------------------------------------------------------------


def test_noise_floor_follows_kTB() -> None:
    chain = RfChain([Stage("ideal", 0.0, 0.0)])
    assert chain.noise_floor_dbm(1.0) == pytest.approx(-174.0, abs=1e-9)
    # Ten times the bandwidth is 10 dB more noise.
    assert chain.noise_floor_dbm(10.0) == pytest.approx(-164.0, abs=1e-9)


def test_mds_in_a_channel_bandwidth() -> None:
    chain = RfChain.deployed()
    mds = chain.minimum_detectable_signal_dbm(12_500.0)
    assert -125.0 < mds < -115.0


def test_lna_improves_mds_by_its_noise_figure_advantage() -> None:
    bandwidth = 12_500.0
    bare = RfChain([Stage("HackRF One", 0.0, 8.0)])
    improvement = bare.minimum_detectable_signal_dbm(
        bandwidth
    ) - RfChain.deployed().minimum_detectable_signal_dbm(bandwidth)
    assert improvement > 5.0


# --------------------------------------------------------------------------------------
# Damage limits: the checks that exist so calibration cannot destroy the LNA
# --------------------------------------------------------------------------------------


def test_safe_input_produces_no_violations() -> None:
    assert RfChain.deployed().check_input_safety(-60.0) == []


def test_generator_level_straight_into_the_lna_is_flagged() -> None:
    violations = RfChain.deployed().check_input_safety(+10.0)
    assert any("external LNA" in v for v in violations)


def test_a_level_safe_at_the_antenna_can_still_destroy_the_receiver() -> None:
    """The failure the chain walk exists to catch.

    -10 dBm is harmless at the LNA input. After 20 dB of gain it arrives at the HackRF
    well above its linear limit, so a check that only looked at the first stage would pass
    a configuration that ruins every measurement taken through it.
    """
    violations = RfChain.deployed().check_input_safety(-10.0)
    assert violations != []
    assert any("HackRF" in v for v in violations)


def test_required_attenuation_makes_the_chain_safe() -> None:
    chain = RfChain.deployed()
    for source_dbm in (0.0, 10.0, -20.0):
        attenuation = chain.required_attenuation_db(source_dbm)
        assert attenuation % 10.0 == 0.0, "pads come in 10 dB steps"
        assert chain.check_input_safety(source_dbm - attenuation) == []


def test_conducted_calibration_needs_a_substantial_pad() -> None:
    """Concrete guidance for the Phase 4 bench: a 30-40 dB pad is not optional."""
    assert RfChain.deployed().required_attenuation_db(0.0) >= 30.0


# --------------------------------------------------------------------------------------
# Gain planning
# --------------------------------------------------------------------------------------


def test_gains_snap_to_hardware_steps() -> None:
    gains = quantise_gains(30.0, 21.0)
    assert gains.lna_db % HACKRF_LNA_STEP_DB == 0.0
    assert gains.vga_db % HACKRF_VGA_STEP_DB == 0.0


def test_gains_are_clipped_to_the_hardware_range() -> None:
    assert quantise_gains(999.0, 999.0).lna_db == 40.0
    assert quantise_gains(-5.0, -5.0).vga_db == 0.0


def test_external_gain_is_taken_out_of_the_internal_budget() -> None:
    """The 8-bit ADC makes total gain a budget, not a free parameter."""
    without = plan_gains(external_gain_db=0.0)
    with_lna = plan_gains(external_gain_db=20.0)
    assert with_lna.total_db == pytest.approx(without.total_db - 20.0, abs=HACKRF_VGA_STEP_DB)


def test_planner_fills_front_end_gain_before_baseband() -> None:
    """LNA gain still improves the internal noise figure; baseband VGA gain does not."""
    gains = plan_gains(external_gain_db=0.0, target_total_gain_db=40.0)
    assert gains.lna_db == 40.0
    assert gains.vga_db == 0.0


def test_v0_gain_call_would_not_survive_quantisation() -> None:
    """v0 set GAIN_LNA=0 and passed 8 as the SoapySDR channel index, so no gain applied.

    The planner's answer for the deployed chain keeps real front-end gain, which is the
    point.
    """
    gains = plan_gains(external_gain_db=20.0)
    assert gains.lna_db > 0.0
    assert gains.total_db > 0.0
