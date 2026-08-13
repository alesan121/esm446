"""Range from received power, with the uncertainty that actually applies.

One omnidirectional antenna measures how strongly a signal arrives. Inverting a propagation
model turns that into a distance, and the distance is worth very little without an honest
statement of how wrong it can be.

What the v0 estimator got wrong
-------------------------------
`legacy/range_estimator.py` inverted a log-distance model and propagated the uncertainty by
linearisation::

    sigma_m = d_est * (SIGMA_DB / (10 * PATH_LOSS_EXPONENT)) * math.log(10)

Distance depends exponentially on path loss, so a symmetric uncertainty in decibels becomes a
strongly skewed uncertainty in metres. Taking its own numbers -- 10 dB of shadowing, exponent
3.5 -- an estimate of 500 m has a 68 % interval running from about 385 m to 650 m: 115 m short
and 150 m long. A single sigma cannot describe that, and ``d +/- k*sigma`` puts the lower ring
in the wrong place and the upper ring closer than it belongs.

Worse, it treated only shadowing as uncertain. The path loss exponent, the emitter's radiated
power and the calibration offset are all uncertain too, and the exponent sits in the
denominator of the exponent -- a small error there moves the answer more than anything else
in the model.

What this does instead
----------------------
Draw the uncertain parameters, invert the model once per draw, and report empirical
percentiles of the resulting distances. That handles the skew, the multiple sources, and the
correlation between them without any analysis of the distribution's shape.

What a single sensor cannot do
------------------------------
It measures **range, not bearing**. The product is an annulus of position uncertainty centred
on the receiver, and anything drawn as a point would be claiming a measurement that was never
made. Bearing needs a directional antenna or a second coherent receiver; both are outside what
this system has.

The coverage test, and what it does not prove
---------------------------------------------
`tests/test_geolocation.py` checks that the 95 % ring contains the truth about 95 % of the
time over many realisations. That verifies the estimator inverts its own model correctly,
which is a real and easily failed property.

It does not verify the model. The realisations are drawn from the same prior the estimator
assumes, so the coverage would be just as good if the prior were wrong about the environment.
Establishing that needs measured distances against measured power, which needs a calibration,
which is blocked on issue #41.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

#: Speed of light (m/s).
SPEED_OF_LIGHT = 299_792_458.0

#: Reference distance of the log-distance model (m). Path loss is anchored to free space here.
REFERENCE_DISTANCE_M = 1.0

#: Percentiles reported by default. 50 is the estimate; the rest are the rings.
DEFAULT_PERCENTILES = (5, 50, 68, 90, 95)

#: Monte Carlo draws. The 95th percentile of 20 000 draws has a standard error of about 0.6 %
#: of itself, which is far below the width of the interval being reported and costs about a
#: millisecond.
DEFAULT_DRAWS = 20_000


@dataclass(frozen=True)
class PropagationPrior:
    """What is uncertain about the path, and by how much.

    Every figure here is an assumption, and they are gathered in one place so that they can
    be argued with rather than found scattered through the arithmetic.

    Attributes:
        path_loss_exponent: Mean log-distance exponent. 2.0 is free space, 3.5 is the
            urban/mixed value Rappaport tabulates for UHF.
        path_loss_exponent_sigma: Uncertainty in the exponent. This matters more than
            anything else in the model, because the exponent divides the exponent.
        shadowing_sigma_db: Log-normal shadowing about the mean path loss.
        eirp_dbm: Assumed radiated power of the emitter. 27 dBm is the ETSI EN 300 296
            limit for PMR446; a non-type-approved handset can exceed it considerably.
        eirp_sigma_db: Uncertainty in that assumption, which covers both the emitter's
            actual power and its antenna orientation.
        calibration_sigma_db: Uncertainty in the receiver's power calibration.
        receiver_gain_dbi: Receive antenna gain, which the link budget takes as
            conservative rather than as a supplier figure.
    """

    path_loss_exponent: float = 3.5
    path_loss_exponent_sigma: float = 0.5
    shadowing_sigma_db: float = 8.0
    eirp_dbm: float = 27.0
    eirp_sigma_db: float = 3.0
    calibration_sigma_db: float = 2.0
    receiver_gain_dbi: float = 0.0

    def __post_init__(self) -> None:
        if self.path_loss_exponent <= 0:
            raise ValueError(f"path loss exponent must be positive, got {self.path_loss_exponent}")
        if (
            min(
                self.path_loss_exponent_sigma,
                self.shadowing_sigma_db,
                self.eirp_sigma_db,
                self.calibration_sigma_db,
            )
            < 0
        ):
            raise ValueError("uncertainties cannot be negative")


@dataclass
class RangeEstimate:
    """A range and the distribution it came from.

    Attributes:
        percentiles: Distance in metres at each reported percentile. The 95 % entry is the
            radius of a disc that contains the emitter 95 % of the time under the model.
        calibrated: Whether the received power was backed by a measured calibration. Never
            decoration: an uncalibrated range is an arithmetic exercise, not a measurement.
        draws: Monte Carlo draws behind the figures.
        received_dbm: The input power.
        frequency_hz: The emission's frequency.
    """

    percentiles: dict[int, float] = field(default_factory=dict)
    calibrated: bool = False
    draws: int = 0
    received_dbm: float = 0.0
    frequency_hz: float = 0.0

    @property
    def median_m(self) -> float:
        """The central estimate. The median, not the mean: the distribution is skewed."""
        return self.percentiles[50]

    def ring(self, percentile: int) -> float:
        """Radius of the credible ring at a percentile.

        Args:
            percentile: One of the percentiles this estimate was computed for.

        Returns:
            Radius in metres.

        Raises:
            KeyError: If that percentile was not computed.
        """
        return self.percentiles[percentile]

    def as_dict(self) -> dict[str, Any]:
        """Return the estimate as plain data, for JSON and for the CoT message."""
        return {
            "median_m": round(self.median_m, 1),
            "percentiles_m": {str(p): round(d, 1) for p, d in sorted(self.percentiles.items())},
            "calibrated": self.calibrated,
            "draws": self.draws,
            "received_dbm": round(self.received_dbm, 2),
            "frequency_hz": self.frequency_hz,
            "geometry": "range only; a single omnidirectional sensor measures no bearing",
        }

    def describe(self) -> str:
        """Render the estimate as text, limitation included."""
        rings = ", ".join(f"{p}%: {d:,.0f} m" for p, d in sorted(self.percentiles.items()))
        caveat = "" if self.calibrated else "  UNCALIBRATED -- these are not measured distances\n"
        return (
            f"range from {self.received_dbm:.1f} dBm at {self.frequency_hz / 1e6:.4f} MHz\n"
            f"{caveat}"
            f"  {rings}\n"
            f"  annulus about the receiver: range is measured, bearing is not"
        )


def free_space_loss_db(frequency_hz: float, distance_m: float = REFERENCE_DISTANCE_M) -> float:
    """Free-space path loss at a distance.

    Args:
        frequency_hz: Carrier frequency.
        distance_m: Separation. Defaults to the model's reference distance.

    Returns:
        Loss in dB.
    """
    if frequency_hz <= 0 or distance_m <= 0:
        raise ValueError("frequency and distance must be positive")
    return 20.0 * math.log10(4.0 * math.pi * distance_m * frequency_hz / SPEED_OF_LIGHT)


def path_loss_db(
    distance_m: float,
    frequency_hz: float,
    path_loss_exponent: float,
    reference_distance_m: float = REFERENCE_DISTANCE_M,
) -> float:
    """Mean log-distance path loss, the forward model this module inverts.

    Args:
        distance_m: Separation.
        frequency_hz: Carrier frequency.
        path_loss_exponent: Log-distance exponent.
        reference_distance_m: Where the model is anchored to free space.

    Returns:
        Loss in dB, excluding shadowing.
    """
    reference = free_space_loss_db(frequency_hz, reference_distance_m)
    return reference + 10.0 * path_loss_exponent * math.log10(distance_m / reference_distance_m)


def estimate_range(
    received_dbm: float,
    frequency_hz: float,
    prior: PropagationPrior | None = None,
    calibrated: bool = False,
    draws: int = DEFAULT_DRAWS,
    percentiles: tuple[int, ...] = DEFAULT_PERCENTILES,
    seed: int | None = None,
) -> RangeEstimate:
    """Invert the propagation model once per Monte Carlo draw and report percentiles.

    Args:
        received_dbm: Power at the receiver.
        frequency_hz: Frequency of the emission.
        prior: What is uncertain about the path. Defaults to `PropagationPrior`.
        calibrated: Whether ``received_dbm`` is backed by a measured calibration. Carried
            through to the result and to everything downstream of it.
        draws: Monte Carlo draws.
        percentiles: Percentiles to report. 50 is always included.
        seed: Seed for reproducibility. ``None`` draws from the operating system.

    Returns:
        The estimate.

    Raises:
        ValueError: If ``draws`` is not positive or a percentile is out of range.
    """
    if draws <= 0:
        raise ValueError(f"draws must be positive, got {draws}")
    if any(not 0 < p < 100 for p in percentiles):
        raise ValueError(f"percentiles must lie strictly inside 0..100, got {percentiles}")

    prior = prior or PropagationPrior()
    rng = np.random.default_rng(seed)

    # The exponent is truncated at free space. A propagation exponent below 2 describes a
    # waveguide, not an outdoor path, and letting a draw fall there produces distances of
    # kilometres from the same measurement -- a tail the model has no business generating.
    exponent = rng.normal(prior.path_loss_exponent, prior.path_loss_exponent_sigma, draws)
    np.clip(exponent, 2.0, None, out=exponent)

    eirp_dbm = rng.normal(prior.eirp_dbm, prior.eirp_sigma_db, draws)
    shadowing_db = rng.normal(0.0, prior.shadowing_sigma_db, draws)
    calibration_db = rng.normal(0.0, prior.calibration_sigma_db, draws)

    # Path loss the measurement implies, once this draw's calibration error is removed.
    observed_loss_db = eirp_dbm + prior.receiver_gain_dbi - (received_dbm + calibration_db)

    # Strip the free-space anchor and this draw's shadowing to leave the distance-dependent
    # part, then invert 10*n*log10(d/d0) for d.
    reference_db = free_space_loss_db(frequency_hz, REFERENCE_DISTANCE_M)
    distance_term = observed_loss_db - reference_db - shadowing_db
    distances = REFERENCE_DISTANCE_M * np.power(10.0, distance_term / (10.0 * exponent))

    wanted = sorted({50, *percentiles})
    values = np.percentile(distances, wanted)

    estimate = RangeEstimate(
        percentiles={int(p): float(v) for p, v in zip(wanted, values, strict=True)},
        calibrated=calibrated,
        draws=draws,
        received_dbm=received_dbm,
        frequency_hz=frequency_hz,
    )
    if not calibrated:
        logger.warning(
            "geolocation: range computed from an uncalibrated power reading; the figures "
            "describe the model, not the distance to anything"
        )
    return estimate


def estimate_from_report(
    report: Any,
    prior: PropagationPrior | None = None,
    draws: int = DEFAULT_DRAWS,
    seed: int | None = None,
) -> RangeEstimate | None:
    """Estimate the range to an emission, or decline to.

    Returns ``None`` when the report carries no absolute power, which is the state of every
    report this system currently produces: absolute power needs a conducted calibration and
    there is no attenuator to make one with (issue #41). Producing a distance anyway would
    mean inventing the one input the whole calculation rests on.

    Args:
        report: An `esm446.core.node.EmissionReport`.
        prior: What is uncertain about the path.
        draws: Monte Carlo draws.
        seed: Seed for reproducibility.

    Returns:
        The estimate, or ``None`` if the report has no calibrated power.
    """
    if report.estimated_dbm is None:
        logger.info(
            "geolocation: no range for the emission at %.4f MHz -- power is uncalibrated",
            report.frequency_hz / 1e6,
        )
        return None
    return estimate_range(
        received_dbm=report.estimated_dbm,
        frequency_hz=report.frequency_hz,
        prior=prior,
        calibrated=bool(report.calibrated),
        draws=draws,
        seed=seed,
    )
