"""Cross-check the MATLAB analysis scripts against the Python implementation.

The scripts in `matlab/` derive the same physics independently of `esm446/core/rfchain.py`.
That is only worth anything if the two are actually compared, so this runs both and asserts
they agree. A disagreement means one of them is wrong, which is exactly the value of having
derived the number twice.

Octave is not available everywhere, so these skip rather than fail when it is missing. That
is a deliberate trade: the check is worth having where it can run, and not worth blocking a
build over where it cannot.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

from esm446.core import bands, geolocation
from esm446.core.rfchain import RfChain, Stage

MATLAB_DIR = Path("matlab")

pytestmark = pytest.mark.skipif(
    shutil.which("octave-cli") is None, reason="Octave is not installed"
)


def run_octave(script: str) -> str:
    """Run a snippet in the matlab directory and return its stdout.

    ``--no-init-file`` matters: a personal ``.octaverc`` that changes directory or extends
    the path will otherwise shadow the scripts under test.

    Args:
        script: Octave source to evaluate.

    Returns:
        Captured standard output.
    """
    result = subprocess.run(
        ["octave-cli", "--no-init-file", "--eval", script],
        cwd=MATLAB_DIR,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, f"octave failed:\n{result.stderr}"
    return result.stdout


def extract(output: str, name: str) -> float:
    """Pull a ``name = value`` figure out of Octave's output."""
    match = re.search(rf"{name}\s*=\s*(-?[\d.eE+]+)", output)
    assert match, f"could not find {name!r} in:\n{output}"
    return float(match.group(1))


# --------------------------------------------------------------------------------------
# Link budget
# --------------------------------------------------------------------------------------


def test_noise_figure_agrees_with_python() -> None:
    """Friis, derived twice. The units trap here is easy to fall into in either language."""
    output = run_octave(
        "r = link_budget('verbose', false); printf('nf = %.6f\\n', r.noise_figure_db);"
    )
    assert extract(output, "nf") == pytest.approx(RfChain.deployed().noise_figure_db, abs=0.01)


def test_noise_figure_without_the_lna_agrees() -> None:
    output = run_octave(
        "r = link_budget('lna_gain_db', 0, 'verbose', false); "
        "printf('nf = %.6f\\n', r.noise_figure_db);"
    )
    bare = RfChain(
        [Stage("antenna cable", -0.5, 0.5), Stage("HackRF One", 0.0, 8.0, max_input_dbm=-5.0)]
    )
    assert extract(output, "nf") == pytest.approx(bare.noise_figure_db, abs=0.01)


def test_minimum_detectable_signal_agrees_with_python() -> None:
    output = run_octave("r = link_budget('verbose', false); printf('mds = %.6f\\n', r.mds_dbm);")
    expected = RfChain.deployed().minimum_detectable_signal_dbm(12_500.0)
    assert extract(output, "mds") == pytest.approx(expected, abs=0.05)


# --------------------------------------------------------------------------------------
# Channel plan
# --------------------------------------------------------------------------------------


def test_channel_plan_agrees_on_the_shipping_centre() -> None:
    output = run_octave(
        f"r = channel_plan('centre_hz', {bands.DEFAULT_CENTRE_HZ}, 'verbose', false); "
        "printf('aligned = %d\\n', r.aligned); "
        "printf('unique = %d\\n', r.unique_bins); "
        "printf('images = %d\\n', sum(~isnan(r.image_channels)));"
    )
    assert extract(output, "aligned") == 1.0
    assert extract(output, "unique") == 1.0
    assert extract(output, "images") == 0.0, "an IQ image lands on a channel"


def test_channel_plan_agrees_that_channel_8_is_unusable() -> None:
    """The measured defect, reproduced in the analysis: 15 of 16 channels mirror onto another."""
    output = run_octave(
        "r = channel_plan('centre_hz', 446093750, 'verbose', false); "
        "printf('onchan = %d\\n', r.on_channel); "
        "printf('images = %d\\n', sum(~isnan(r.image_channels)));"
    )
    assert extract(output, "onchan") == 1.0
    assert extract(output, "images") == 15.0

    with pytest.raises(ValueError, match="phantom emitter"):
        bands.assert_centre_is_usable(446_093_750)


def test_channel_bin_mapping_agrees_with_python() -> None:
    output = run_octave(
        f"r = channel_plan('centre_hz', {bands.DEFAULT_CENTRE_HZ}, 'verbose', false); "
        "printf('bin1 = %d\\n', r.bins(1)); printf('bin16 = %d\\n', r.bins(16));"
    )
    for channel, name in ((1, "bin1"), (16, "bin16")):
        expected = bands.channel_bin_index(channel, bands.DEFAULT_CENTRE_HZ, 2_000_000.0, 160)
        assert extract(output, name) == expected


# --------------------------------------------------------------------------------------
# Measurement setup
# --------------------------------------------------------------------------------------


def test_three_metres_aligned_is_reported_as_compressing() -> None:
    """The measurement that decided how the acceptance test has to be arranged."""
    output = run_octave(
        "r = measurement_setup('separation_m', 3.0, 'cross_polarised', false, "
        "'verbose', false); printf('rx = %.4f\\n', r.received_dbm); "
        "printf('linear = %d\\n', r.linear);"
    )
    assert extract(output, "rx") == pytest.approx(-5.0, abs=0.1)
    assert extract(output, "linear") == 0.0


def test_cross_polarisation_makes_three_metres_usable() -> None:
    output = run_octave(
        "r = measurement_setup('separation_m', 3.0, 'cross_polarised', true, "
        "'verbose', false); printf('rx = %.4f\\n', r.received_dbm); "
        "printf('linear = %d\\n', r.linear);"
    )
    assert extract(output, "rx") == pytest.approx(-25.0, abs=0.1)
    assert extract(output, "linear") == 1.0


def test_the_far_field_floor_is_applied() -> None:
    """Free-space loss says nothing useful within a few wavelengths, and the script says so."""
    output = run_octave(
        "r = measurement_setup('separation_m', 0.5, 'verbose', false); "
        "printf('ff = %.4f\\n', r.far_field_m); printf('inff = %d\\n', r.in_far_field); "
        "printf('closest = %.4f\\n', r.min_distance_m);"
    )
    far_field = extract(output, "ff")
    assert far_field == pytest.approx(3 * 299_792_458 / 446.09375e6, abs=0.01)
    assert extract(output, "inff") == 0.0
    assert extract(output, "closest") >= far_field


# --------------------------------------------------------------------------------------
# Monte Carlo geolocation
# --------------------------------------------------------------------------------------


def test_the_closed_form_median_agrees_with_python() -> None:
    """Two random draws cannot be compared, so both are compared against the same algebra.

    With nothing uncertain the median is exact: invert the log-distance model once. If the
    two implementations disagree here, they disagree about the model itself.
    """
    output = run_octave(
        "r = geolocation_monte_carlo('received_dbm', -95, 'with_sensitivity', 0); "
        "printf('median = %.6f\\n', r.d_median_analytic);"
    )
    prior = geolocation.PropagationPrior(
        path_loss_exponent_sigma=0.0,
        shadowing_sigma_db=0.0,
        eirp_sigma_db=0.0,
        calibration_sigma_db=0.0,
    )
    expected = geolocation.estimate_range(
        -95.0, 446.09375e6, prior=prior, draws=64, seed=1
    ).median_m

    assert extract(output, "median") == pytest.approx(expected, rel=1e-6)


def test_the_shadowing_only_percentile_agrees_with_python() -> None:
    """The log-normal closed form, which both Monte Carlo implementations must reproduce.

    d_p = d_median * 10^(z_p * sigma_db / (10 * n)). Octave computes it in closed form; Python
    reaches it by drawing, so agreement to a fraction of a per cent checks the draw as well as
    the algebra.
    """
    output = run_octave(
        "r = geolocation_monte_carlo('received_dbm', -95, 'with_sensitivity', 0); "
        "printf('d95 = %.6f\\n', r.d95_analytic);"
    )
    prior = geolocation.PropagationPrior(
        path_loss_exponent_sigma=0.0,
        eirp_sigma_db=0.0,
        calibration_sigma_db=0.0,
        shadowing_sigma_db=8.0,
    )
    drawn = geolocation.estimate_range(-95.0, 446.09375e6, prior=prior, draws=400_000, seed=5).ring(
        95
    )

    assert extract(output, "d95") == pytest.approx(drawn, rel=0.01)


def test_both_implementations_agree_the_exponent_dominates() -> None:
    """The conclusion, not just the numbers: narrowing the exponent is the only thing worth
    measuring while it is a guess."""
    output = run_octave(
        "r = geolocation_monte_carlo('received_dbm', -95); "
        "printf('exp = %.4f\\n', r.spread.path_loss_exp_sigma); "
        "printf('shadow = %.4f\\n', r.spread.shadowing_sigma_db); "
        "printf('cal = %.4f\\n', r.spread.calibration_sigma_db);"
    )
    assert extract(output, "exp") > 3 * extract(output, "shadow")
    assert extract(output, "exp") > 10 * extract(output, "cal")


def test_every_script_runs_end_to_end() -> None:
    """`esm446_analysis` is what a reader runs, so it has to work as a whole."""
    output = run_octave("esm446_analysis")

    assert "cascaded noise figure" in output
    assert "REJECTED: DC spur on PMR8" in output
    assert "offset-tuned centre          usable" in output
